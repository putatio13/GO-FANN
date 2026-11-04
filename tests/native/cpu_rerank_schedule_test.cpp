#include "app/config_loader.h"
#include "execution/cpu_rerank_schedule.h"
#include "search/pq/pq_posting_streaming.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <string>
#include <unistd.h>

using go_fann::cpu_rerank_schedule::Mode;
namespace schedule = go_fann::cpu_rerank_schedule;
using namespace std::chrono_literals;

static void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

template<class Work>
static void rejects(Work work)
{
    try { work(); }
    catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("invalid scheduling configuration accepted");
}

static void configuration()
{
    bool graph = true, posting = false;
    schedule::apply(Mode::Native, "full_gpu", "search", "coupled", "mailbox",
                    "gpu", "exact", graph, posting);
    require(graph && !posting, "native must preserve independent streaming flags");
    for (const auto mode : {Mode::NoOverlap, Mode::BatchOverlap, Mode::Streaming}) {
        for (const char* backend : {"pq", "rabitq"}) {
            schedule::apply(mode, backend, "benchmark", "coupled", "mailbox",
                            "cpu", "exact", graph, posting);
            require(graph == (mode == Mode::Streaming) && posting == graph,
                    "explicit schedule must set both streaming controls");
        }
    }
    for (const char* invalid : {"", "serial", "typo"})
        rejects([&] { schedule::parse(invalid); });
    for (const char* backend : {"full_gpu", "unknown"})
        rejects([&] { schedule::apply(Mode::NoOverlap, backend, "search", "coupled",
            "mailbox", "cpu", "exact", graph, posting); });
    for (const char* operation : {"build-graph", "latency"})
        rejects([&] { schedule::apply(Mode::NoOverlap, "pq", operation, "coupled",
            "mailbox", "cpu", "exact", graph, posting); });
    rejects([&] { schedule::apply(Mode::NoOverlap, "pq", "search", "finite",
        "mailbox", "cpu", "exact", graph, posting); });
    rejects([&] { schedule::apply(Mode::NoOverlap, "pq", "search", "coupled",
        "device_resident", "cpu", "exact", graph, posting); });
    rejects([&] { schedule::apply(Mode::NoOverlap, "pq", "search", "coupled",
        "mailbox", "gpu", "exact", graph, posting); });
    rejects([&] { schedule::apply(Mode::NoOverlap, "pq", "search", "coupled",
        "mailbox", "cpu", "heuristic", graph, posting); });

    char path[] = "/tmp/go_fann_cpu_schedule_XXXXXX";
    const int fd = mkstemp(path);
    require(fd >= 0, "mkstemp failed");
    close(fd);
    try {
        { std::ofstream file(path); file << "{}"; }
        require(!go_fann::load_config_file(path).cpu_rerank_schedule,
                "omitted schedule must remain absent in loader");
        for (const char* value : {"native", "no_overlap", "batch_overlap", "streaming"}) {
            { std::ofstream file(path); file << "{\"cpu_rerank_schedule\":\"" << value << "\"}"; }
            const auto config = go_fann::load_config_file(path);
            require(config.cpu_rerank_schedule == value, "JSON schedule not loaded");
            (void)schedule::parse(*config.cpu_rerank_schedule);
        }
        for (const char* value : {"null", "false", "42", "{}", "[]"}) {
            { std::ofstream file(path); file << "{\"cpu_rerank_schedule\":" << value << "}"; }
            bool rejected = false;
            try { (void)go_fann::load_config_file(path); }
            catch (const std::runtime_error&) { rejected = true; }
            require(rejected, "malformed explicit JSON schedule was silently ignored");
        }
        std::remove(path);
    } catch (...) { std::remove(path); throw; }
}

static void launch_order()
{
    bool ran = false;
    auto serial = std::async(schedule::host_finalize_launch(Mode::NoOverlap), [&] { ran = true; });
    require(serial.wait_for(0ms) == std::future_status::deferred && !ran,
            "serial CPU finalizer ran before explicit collection");
    serial.get();
    require(ran, "serial CPU finalizer never ran");

    std::promise<void> started, release;
    auto ready = started.get_future();
    auto proceed = release.get_future();
    auto batch = std::async(schedule::host_finalize_launch(Mode::BatchOverlap), [&] {
        started.set_value();
        proceed.get();
    });
    const bool active_before_next_gpu_phase = ready.wait_for(5s) == std::future_status::ready;
    release.set_value();
    batch.get();
    require(active_before_next_gpu_phase, "batch CPU finalizer was not asynchronous");
}

static void two_provider_gate()
{
    schedule::GpuPhaseGate gate;
    std::promise<void> primary_gpu_done;
    auto ready = primary_gpu_done.get_future();
    std::atomic<bool> cpu_started{false};
    auto primary = std::async(std::launch::async, [&] {
        auto cpu = std::async(schedule::host_finalize_launch(Mode::NoOverlap), [&] {
            cpu_started = true;
        });
        primary_gpu_done.set_value();
        gate.wait();
        cpu.get();
    });
    ready.get();
    const bool blocked = !cpu_started && primary.wait_for(0ms) == std::future_status::timeout;
    gate.release(); // Secondary GPU phase completed.
    primary.get();
    require(blocked && cpu_started, "CPU rerank crossed the two-provider GPU barrier");

    schedule::GpuPhaseGate failed;
    auto waiter = std::async(std::launch::async, [&] {
        try { failed.wait(); }
        catch (const std::runtime_error& error) { return std::string(error.what()) == "secondary failure"; }
        return false;
    });
    failed.fail(std::make_exception_ptr(std::runtime_error("secondary failure")));
    require(waiter.get(), "secondary failure did not release waiting primary");
    gate.fail(std::make_exception_ptr(std::runtime_error("late CPU failure")));
    gate.wait(); // A failure after release must not replace successful GPU completion.

    // Keep the completion promise alive while the worker fails before publish,
    // matching the dual-provider caller's lifetime. Failure must wake that caller.
    std::promise<void> primary_complete;
    auto complete = primary_complete.get_future();
    auto failed_primary = std::async(std::launch::async, [&] {
        bool published = false;
        try {
            throw std::runtime_error("primary setup failure");
        } catch (...) {
            if (!published) primary_complete.set_exception(std::current_exception());
            throw;
        }
    });
    const bool notified = complete.wait_for(5s) == std::future_status::ready;
    bool worker_failed = false, caller_failed = false;
    try { failed_primary.get(); } catch (const std::runtime_error&) { worker_failed = true; }
    if (notified) {
        try { complete.get(); } catch (const std::runtime_error&) { caller_failed = true; }
    }
    require(notified && worker_failed && caller_failed,
            "primary failure before GPU publication did not notify the waiting caller");
}

static void streaming_order()
{
    std::array<std::atomic<bool>, 2> ready{false, false};
    std::atomic<bool> cancelled{false};
    std::promise<void> first_consumed;
    auto consumed = first_consumed.get_future();
    std::vector<size_t> order;
    auto consumer = std::async(std::launch::async, [&] {
        go_fann::pq_posting_streaming::consume_ready(2, 1,
            [&](size_t row) { return ready[row].load(); },
            [&](const std::vector<size_t>& rows) {
                order.push_back(rows.front());
                if (rows.front() == 0) first_consumed.set_value();
            }, cancelled);
    });
    ready[0] = true;
    const bool overlapped = consumed.wait_for(5s) == std::future_status::ready;
    ready[1] = true; // Later GPU result was unavailable during the first CPU callback.
    if (!overlapped) cancelled = true;
    consumer.get();
    require(overlapped && order == std::vector<size_t>({0, 1}),
            "streaming did not consume results before the producer finished");
}

int main()
{
    try {
        configuration();
        launch_order();
        two_provider_gate();
        streaming_order();
        std::puts("CPU rerank schedule configuration and execution-order tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "CPU rerank schedule test failed: %s\n", error.what());
        return 1;
    }
}
