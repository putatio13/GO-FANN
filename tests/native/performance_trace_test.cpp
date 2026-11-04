#include "profiling/performance_trace.h"
#include <nlohmann/json.hpp>
#include <cassert>
#include <cstdio>
#include <fstream>
#include <limits>
#include <memory>
#include <sys/wait.h>
#include <tuple>

namespace trace = go_fann::performance_trace;
using Json = nlohmann::json;

static trace::Time at(int64_t value) {
    return trace::Time(std::chrono::nanoseconds(value));
}
static void span() {
    trace::span("graph_launch_to_complete", nullptr, at(100), at(200));
}
static std::string contents(const std::string& path) {
    std::ifstream input(path);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}
static std::vector<Json> records(const std::string& path) {
    std::ifstream input(path);
    std::vector<Json> result;
    std::string line;
    while (std::getline(input, line)) result.push_back(Json::parse(line));
    return result;
}
static void expect_error(const std::function<void()>& action) {
    bool caught = false;
    try { action(); } catch (const std::exception&) { caught = true; }
    assert(caught);
}
static void run_isolated(const char* label, const std::function<void(const std::string&)>& action) {
    const pid_t child = fork();
    assert(child >= 0);
    if (child == 0) {
        const std::string path = "/tmp/go-fann-trace-" + std::to_string(getpid()) + ".jsonl";
        std::remove(path.c_str());
        setenv("GO_FANN_PERFORMANCE_TRACE", path.c_str(), 1);
        try { action(path); }
        catch (const std::exception& error) {
            std::fprintf(stderr, "%s: %s\n", label, error.what());
            _exit(1);
        }
        std::remove(path.c_str());
        _exit(0);
    }
    int status = 0;
    assert(waitpid(child, &status, 0) == child);
    assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
}

static void emit_fixture(const char* path) {
    setenv("GO_FANN_PERFORMANCE_TRACE", path, 1);
    trace::Capture capture("pq", "b1.000000_q24_f100", "timed", 24, 0);
    for (const auto& item : std::vector<std::tuple<const char*, int64_t, int64_t>>{
             {"provider_search_window", 100, 200}, {"graph_prepare", 95, 110},
             {"graph_launch_to_complete", 110, 140}, {"graph_launch_to_complete", 120, 135},
             {"highfreq_cpu_rerank_batch", 130, 160}, {"lowfreq_kernel_launch_to_complete", 150, 170},
             {"lowfreq_h2d", 170, 180}, {"merge_routes_to_original", 180, 190}}) {
        trace::span(std::get<0>(item), nullptr, at(std::get<1>(item)), at(std::get<2>(item)));
    }
    trace::value("reported_search_wall_ms", nullptr, 0.0001);
    capture.finish();
}

int main(int argc, char** argv) {
    if (argc == 3 && std::string(argv[1]) == "--emit") {
        emit_fixture(argv[2]);
        return 0;
    }
    assert(argc == 1);
    run_isolated("disabled", [](const std::string& path) {
        for (bool empty : {false, true}) {
            if (empty) setenv("GO_FANN_PERFORMANCE_TRACE", "", 1);
            else unsetenv("GO_FANN_PERFORMANCE_TRACE");
            trace::Capture capture("pq", "b1_q24_f100", "timed", 24, 0);
            { trace::Scope scope("disabled", nullptr); span(); trace::value("rows", nullptr, 1); }
            capture.finish();
            assert(!trace::enabled());
            assert(trace::recorder().events.capacity() == 0);
            assert(trace::recorder().output_fd == -1);
        }
        assert(access(path.c_str(), F_OK) != 0);
    });
    run_isolated("workers and identities", [](const std::string& path) {
        trace::Capture capture("pq", "b1.000000_q24_f100", "timed", 24, 0);
        expect_error([] { trace::begin("pq", "b1_q12_f100", "timed", 12, 1); });
        {
            trace::Scope outer("iteration", nullptr);
            auto worker = [] {
                trace::Scope scope("worker\"\\\n", nullptr);
                trace::value("rows", nullptr, 24789973);
            };
            std::thread first(worker), second(worker);
            first.join(); second.join();
        }
        capture.finish();
        auto result = records(path);
        assert(result.size() == 6);
        assert(result.back()["kind"] == "complete" && result.back()["event_count"] == 5);
        unsigned counters = 0, spans = 0, escaped = 0;
        for (const auto& row : result) {
            assert(row["schema_version"] == 1 && row["backend"] == "pq");
            assert(row["point_id"] == "b1.000000_q24_f100" && row["phase"] == "timed");
            assert(row["queue"] == 24 && row["run"] == 0);
            if (row["kind"] == "complete") continue;
            assert(row["end_ns"].get<int64_t>() >= row["start_ns"].get<int64_t>());
            if (row["kind"] == "counter") { ++counters; assert(row["value"] == 24789973); }
            else { ++spans; if (row["name"] == "worker\"\\\n") ++escaped; }
        }
        assert(counters == 2 && spans == 3 && escaped == 2);
        expect_error([] { trace::begin("pq", "b1.000000_q24_f100", "timed", 24, 0); });
        trace::Capture next("pq", "b1.000000_q24_f200", "timed", 24, 0);
        span(); next.finish();
        result = records(path);
        assert(result.size() == 8 && result.back()["point_id"] == "b1.000000_q24_f200");
    });
    run_isolated("fresh path", [](const std::string& path) {
        { std::ofstream stream(path); stream << "preserve"; }
        expect_error([] { trace::begin("pq", "b1_q24_f100", "timed", 24, 0); });
        assert(contents(path) == "preserve");
    });
    run_isolated("unwritable path", [](const std::string& path) {
        setenv("GO_FANN_PERFORMANCE_TRACE", (path + "/missing/trace").c_str(), 1);
        expect_error([] { trace::begin("pq", "b1_q24_f100", "timed", 24, 0); });
        assert(!trace::enabled());
    });
    run_isolated("fixed destination", [](const std::string& path) {
        trace::Capture capture("pq", "b1_q24_f100", "timed", 24, 0);
        span(); capture.finish();
        setenv("GO_FANN_PERFORMANCE_TRACE", (path + ".other").c_str(), 1);
        expect_error([] { trace::begin("pq", "b1_q24_f200", "timed", 24, 0); });
        assert(access((path + ".other").c_str(), F_OK) != 0);
    });
    run_isolated("invalid identities", [](const std::string& path) {
        for (const auto& point : {"", "b_q24_f100", "b1_q24_f0", "b1..0_q24_f100", "rabitq_b1_q24_f100"}) {
            expect_error([&] { trace::begin("pq", point, "timed", 24, 0); });
        }
        expect_error([] { trace::begin("pq", "b1_q24_f100", "warmup", 24, 0); });
        expect_error([] { trace::begin("pq", "b1_q24_f100", "timed", 24, -1); });
        assert(access(path.c_str(), F_OK) != 0);
    });
    run_isolated("exception cancellation", [](const std::string& path) {
        try {
            trace::Capture capture("pq", "b1_q24_f100", "timed", 24, 0);
            trace::Scope scope("scope", nullptr);
            span();
            throw std::runtime_error("original exception");
        } catch (const std::runtime_error& error) { assert(std::string(error.what()) == "original exception"); }
        assert(!trace::enabled() && contents(path).empty());
        trace::Capture next("pq", "b1_q24_f100", "timed", 24, 0);
        span(); next.finish();
        assert(records(path).size() == 2);
    });
    run_isolated("unfinished and late scopes", [](const std::string& path) {
        trace::begin("pq", "b1_q24_f100", "timed", 24, 0);
        auto late = std::make_unique<trace::Scope>("late", nullptr);
        span();
        expect_error([] { trace::finish(); });
        assert(contents(path).empty());
        trace::Capture next("rabitq", "rabitq_b1.000000_q24_f100", "timed", 24, 0);
        late.reset();  // A canceled generation must not leak into the new iteration.
        span(); next.finish();
        const auto result = records(path);
        assert(result.size() == 2 && result.front()["name"] == "graph_launch_to_complete");
    });
    run_isolated("invalid or empty events", [](const std::string& path) {
        for (int test = 0; test < 3; ++test) {
            trace::Capture capture("pq", "b1_q24_f100", "timed", 24, test);
            if (test == 1) trace::span("bad", nullptr, at(200), at(100));
            if (test == 2) trace::value("bad", nullptr, std::numeric_limits<double>::quiet_NaN());
            expect_error([&] { capture.finish(); });
        }
        assert(contents(path).empty());
    });
    run_isolated("write failure", [](const std::string& path) {
        trace::Capture capture("pq", "b1_q24_f100", "timed", 24, 0);
        span();
        assert(close(trace::recorder().output_fd) == 0);
        trace::recorder().output_fd = -1;
        expect_error([&] { capture.finish(); });
        assert(trace::recorder().poisoned && contents(path).empty());
        expect_error([] { trace::begin("pq", "b1_q24_f200", "timed", 24, 0); });
    });
    return 0;
}
