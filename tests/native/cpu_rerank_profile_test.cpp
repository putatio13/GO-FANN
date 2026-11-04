#include "profiling/cpu_rerank_profile.h"
#include <cassert>
#include <cstdio>
#include <unistd.h>

using namespace go_fann::cpu_rerank_profile;

int main()
{
    unsetenv("GO_FANN_CPU_RERANK_PROFILE");
    std::vector<uint32_t> ids{7, 2, 7, 1};
    std::vector<std::pair<uint32_t, float>> reference, profiled;
    std::vector<char> order;
    {
        Task off("disabled");
        assert(!off.enabled());
        int calls = 0;
        for (int i = 0; i < 129; ++i)
            assert(off.sample(Detail::Predicate, [&] { return calls++; }) == i);
        assert(calls == 129);
        off.evaluate_batch(ids, [&](uint32_t id) { order.push_back('d'); return id * 1.25f; },
            [&](uint32_t id, float d) { order.push_back('s'); reference.emplace_back(id, d); });
    }
    assert(order == std::vector<char>({'d','s','d','s','d','s','d','s'}));
    const std::string path = "/tmp/flashanns-rerank-profile-" + std::to_string(getpid()) + ".jsonl";
    {
        Reporter reporter(path);
        Iteration iteration(reporter, "timed", 12, 0);
        bool nested_rejected = false;
        try { Iteration nested(reporter, "timed", 16, 0); }
        catch (const std::logic_error&) { nested_rejected = true; }
        assert(nested_rejected);
        {
            Task on("route");
            assert(on.enabled());
            on.query();
            on.candidates(ids.size());
            on.stage(Stage::Candidates);
            int calls = 0;
            for (int i = 0; i < 129; ++i)
                assert(on.sample(Detail::Predicate, [&] { return calls++; }) == i);
            assert(calls == 129);
            order.clear();
            on.evaluate_batch(ids, [&](uint32_t id) { order.push_back('d'); return id * 1.25f; },
                [&](uint32_t id, float d) { order.push_back('s'); profiled.emplace_back(id, d); });
            on.stage(Stage::Writeback);
        }
        assert(profiled == reference);
        assert(order == std::vector<char>({'d','d','d','d','s','s','s','s'}));
        std::vector<std::thread> workers;
        for (int i = 0; i < 4; ++i) workers.emplace_back([&] {
            Task task("parallel");
            task.evaluate_batch(ids, [](uint32_t id) { return float(id); }, [](uint32_t, float) {});
        });
        for (auto& worker : workers) worker.join();
        iteration.finish();
        assert(current_session.load() == nullptr);
        Iteration next(reporter, "warmup", 16, 1);
        next.finish();
    }
    std::ifstream input(path);
    std::string line;
    size_t tasks = 0, iterations = 0;
    while (std::getline(input, line)) {
        const auto r = nlohmann::json::parse(line);
        if (r["record_type"] == "task") {
            ++tasks;
            double sum = 0;
            for (const auto& stage : r["stage_ms"].items()) {
                assert(stage.value().get<double>() >= 0);
                sum += stage.value().get<double>();
            }
            assert(std::abs(sum - r["wall_ms"].get<double>()) < 1e-8);
            assert(r["exact_evaluations"] == ids.size());
            const auto& detail = r["nested_candidate_samples"]["predicate_check"];
            assert(detail["calls"] == (r["path"] == "route" ? 129 : 0));
            assert(detail["samples"] == (r["path"] == "route" ? 3 : 0));
            assert(detail["sample_stride"] == 64);
        } else {
            ++iterations;
            if (r["phase"] == "timed") {
                assert(r["task_count"] == 5);
                assert(r["exact_evaluations"] == 20);
            } else assert(r["task_count"] == 0);
        }
    }
    assert(tasks == 5 && iterations == 2);
    std::remove(path.c_str());
    bool unwritable_rejected = false;
    try { Reporter invalid("/no-such-directory/rerank.jsonl"); }
    catch (const std::runtime_error&) { unwritable_rejected = true; }
    assert(unwritable_rejected);

    std::vector<uint32_t> wide_ids(173);
    for (size_t i = 0; i < wide_ids.size(); ++i) wide_ids[i] = (i * 13) % 179;
    for (bool profiling : {false, true}) {
        if (profiling) setenv("GO_FANN_CPU_RERANK_PROFILE", path.c_str(), 1);
        else unsetenv("GO_FANN_CPU_RERANK_PROFILE");
        Reporter reporter;
        Iteration iteration(reporter, "timed", 12, 0);
        std::vector<uint32_t> selected;
        size_t evaluated = 0;
        {
            Task task("native_batch");
            task.evaluate_native_batch(wide_ids,
                [&](const uint32_t* rows, size_t count, float* distances) {
                    evaluated += count;
                    for (size_t i = 0; i < count; ++i) distances[i] = float(rows[i]);
                }, [&](uint32_t id, float distance) {
                    assert(distance == float(id));
                    selected.push_back(id);
                });
        }
        iteration.finish();
        assert(selected == wide_ids && evaluated == wide_ids.size());
    }
    for (bool profiling : {false, true}) {
        if (profiling) setenv("GO_FANN_CPU_RERANK_PROFILE", path.c_str(), 1);
        else unsetenv("GO_FANN_CPU_RERANK_PROFILE");
        Reporter reporter;
        Iteration iteration(reporter, "timed", 12, 0);
        std::vector<float> scores;
        std::vector<size_t> tiles;
        {
            Task task("incremental_scores");
            task.stage(Stage::QueryLockWait);
            task.stage(Stage::IncrementalAdmit);
            task.stage(Stage::QueryPrepare);
            task.evaluate_into(wide_ids, scores, [&](const uint32_t* rows, size_t n, float* output) {
                tiles.push_back(n);
                for (size_t i=0;i<n;++i) output[i]=float(rows[i]);
            });
            task.stage(Stage::QueryLockWait);
            task.stage(Stage::IncrementalCommit);
        }
        iteration.finish();
        assert(tiles == std::vector<size_t>({64,64,45}));
        for(size_t i=0;i<wide_ids.size();++i) assert(scores[i]==float(wide_ids[i]));
    }
    unsetenv("GO_FANN_CPU_RERANK_PROFILE");
    std::remove(path.c_str());
}
