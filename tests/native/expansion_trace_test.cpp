#include "profiling/expansion_trace.h"
#include "data/dataloader.h"
#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <nlohmann/json.hpp>
#include <unistd.h>

using namespace go_fann;
using namespace go_fann::expansion_trace;

static void rejects(const std::function<void()>& operation) {
    bool rejected = false;
    try { operation(); } catch (const std::exception&) { rejected = true; }
    assert(rejected);
}

int main() {
    unsetenv("GO_FANN_EXPANSION_TRACE_DIR");
    assert(!enabled());
    View{}.append(999, 999);
    View{}.finish(999, 999);
    assert(checked_slots(2, 3) == 6);
    rejects([] { checked_slots(SIZE_MAX, 2); });
    setenv("GO_FANN_EXPANSION_TRACE_CAPACITY", "0", 1);
    rejects([] { capacity_from_environment(); });
    setenv("GO_FANN_EXPANSION_TRACE_CAPACITY", "2", 1);
    assert(capacity_from_environment() == 2);

    GraphRuntime runtime;
    runtime.original_query_count = 1;
    runtime.query_vectors.rows = 2;
    runtime.base_vectors.rows = 10;
    runtime.graph.cols = 32;
    runtime.graph.rows = 3;
    runtime.graph.data.resize(96);
    runtime.route_to_original_query = {0, 0};
    runtime.query_group_ids = {0, 0};
    runtime.start_local_ids = {0, 1};
    runtime.route_enabled = {1, 1};
    runtime.query_is_lowfreq = {0, 0};
    runtime.route_execution_modes.assign(2, QueryRouteExecution::GraphTraversal);
    runtime.route_providers.resize(2);
    runtime.route_candidate_sources.resize(2);
    runtime.metadata.label_size = {3};
    runtime.metadata.label_offset = {0};
    runtime.metadata.row_to_sample = {7, 3, 9};
    const std::vector<uint32_t> active{1, 0};
    auto slots = make_route_slots(runtime, active);
    assert(slots == std::vector<uint32_t>({1, 0}));
    rejects([&] { make_route_slots(runtime, {0}); });
    rejects([&] { make_route_slots(runtime, {0, 0}); });

    std::vector<uint32_t> rows(4);
    std::vector<Counters> counts(2);
    View view{rows.data(), slots.data(), counts.data(), 2};
    view.append(0, 2);
    view.append(0, 2); // repeated expansions remain present
    view.append(1, 0);
    view.finish(0, 2);
    view.finish(1, 1);
    assert(rows[2] == 2 && rows[3] == 2 && counts[1].recorded == 2);

    const auto root = std::filesystem::temp_directory_path() /
        ("flashanns-expansion-test-" + std::to_string(getpid()));
    std::filesystem::create_directories(root);
    const auto graph = root / "graph.bin";
    std::ofstream(graph).put('x');
    setenv("GO_FANN_EXPANSION_TRACE_DIR", root.c_str(), 1);
    export_invocation(runtime, active, rows, counts, 2, "test", graph.string(), "");
    const auto invocation = root / ("invocation_" + std::to_string(getpid()) + "_0");
    nlohmann::json manifest;
    std::ifstream(invocation / "manifest.json") >> manifest;
    assert(manifest.at("complete") == true && manifest.at("event_count") == 3);
    assert(manifest.at("metadata_format") == "trace_row_to_sample_u32");
    assert(std::filesystem::file_size(invocation / "events.bin") == 24);
    assert(std::filesystem::file_size(invocation / "row_to_sample.bin") == 12);
    std::ifstream event_file(invocation / "events.bin", std::ios::binary);
    uint32_t values[6]{};
    event_file.read(reinterpret_cast<char*>(values), sizeof(values));
    assert(values[0] == 2 && values[1] == 9 && values[2] == 2 && values[3] == 9);
    assert(values[4] == 0 && values[5] == 7);
    view.append(0, 1); // overflow never changes search; export rejects evidence
    view.finish(0, 3);
    rejects([&] { export_invocation(runtime, active, rows, counts, 2, "test", graph.string(), ""); });
    counts[1] = Counters{2, 2, 0}; // incomplete route with apparently valid counts
    rejects([&] { export_invocation(runtime, active, rows, counts, 2, "test", graph.string(), ""); });
    std::filesystem::remove_all(root);
}
