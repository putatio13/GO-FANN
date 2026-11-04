#include "profiling/expansion_trace.h"
#include "data/dataloader.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <unordered_set>
#include <unistd.h>

namespace go_fann::expansion_trace {
namespace {
std::string environment(const char* key) {
    const char* value = std::getenv(key);
    return value ? value : "";
}
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(std::string("expansion trace: ") + message);
}
std::string quote(const std::string& text) {
    std::string out = "\"";
    for (char c : text) { if (c == '"') out += '"'; out += c; }
    return out + '"';
}
void write_u32(std::ostream& out, uint32_t value) {
    const char bytes[4] = {static_cast<char>(value), static_cast<char>(value >> 8),
                          static_cast<char>(value >> 16), static_cast<char>(value >> 24)};
    out.write(bytes, 4);
}
bool graph_route(const GraphRuntime& runtime, uint32_t route) {
    return runtime.route_enabled.at(route) && !runtime.query_is_lowfreq.at(route) &&
        runtime.route_execution_modes.at(route) == QueryRouteExecution::GraphTraversal;
}
}

bool enabled() { return !environment("GO_FANN_EXPANSION_TRACE_DIR").empty(); }

uint32_t capacity_from_environment() {
    const std::string value = environment("GO_FANN_EXPANSION_TRACE_CAPACITY");
    if (value.empty()) return 1024;
    require(value.find_first_not_of("0123456789") == std::string::npos, "capacity must be a positive integer");
    const uint64_t parsed = std::stoull(value);
    require(parsed > 0 && parsed <= std::numeric_limits<int32_t>::max(), "capacity out of range");
    return static_cast<uint32_t>(parsed);
}

size_t checked_slots(size_t routes, uint32_t capacity) {
    require(capacity > 0, "zero capacity");
    require(routes <= std::numeric_limits<size_t>::max() / capacity / sizeof(uint32_t),
            "trace allocation size overflow");
    return routes * capacity;
}

std::vector<uint32_t> make_route_slots(const GraphRuntime& runtime,
                                     const std::vector<uint32_t>& active) {
    const size_t count = runtime.query_vectors.rows;
    require(count <= UINT32_MAX && active.size() <= UINT32_MAX, "route domain exceeds uint32");
    require(runtime.route_to_original_query.size() == count && runtime.query_group_ids.size() == count &&
            runtime.start_local_ids.size() == count && runtime.route_enabled.size() == count &&
            runtime.query_is_lowfreq.size() == count && runtime.route_execution_modes.size() == count &&
            runtime.route_providers.size() == count && runtime.route_candidate_sources.size() == count,
            "incomplete route descriptor arrays");
    std::vector<uint32_t> slots(count, UINT32_MAX);
    for (size_t i = 0; i < active.size(); ++i) {
        const uint32_t route = active[i];
        require(route < count, "active route out of bounds");
        require(slots[route] == UINT32_MAX, "duplicate active route");
        require(graph_route(runtime, route), "active route is not an enabled graph traversal");
        require(runtime.route_to_original_query[route] < runtime.original_query_count, "original query out of bounds");
        const uint32_t group = runtime.query_group_ids[route];
        require(group < runtime.metadata.label_size.size() && group < runtime.metadata.label_offset.size(),
                "active group out of bounds");
        const uint64_t nodes = runtime.metadata.label_size[group];
        const uint64_t offset = runtime.metadata.label_offset[group];
        require(nodes > 0 && offset + nodes <= runtime.metadata.row_to_sample.size(), "invalid graph support");
        require(runtime.start_local_ids[route] < nodes, "seed out of bounds");
        slots[route] = static_cast<uint32_t>(i);
    }
    // This diagnostic supports complete workloads, not route or query minibatches.
    for (uint32_t route = 0; route < count; ++route)
        require(!graph_route(runtime, route) || slots[route] != UINT32_MAX, "missing enabled graph route");
    return slots;
}

void export_invocation(const GraphRuntime& runtime, const std::vector<uint32_t>& active,
                       const std::vector<uint32_t>& rows, const std::vector<Counters>& counters,
                       uint32_t capacity, const std::string& backend,
                       const std::string& graph_path, const std::string& metadata_path) {
    const auto slots = make_route_slots(runtime, active);
    require(rows.size() == checked_slots(active.size(), capacity) && counters.size() == active.size(),
            "trace buffer shape mismatch");
    const std::string graph = std::filesystem::canonical(graph_path).string();
    const std::string identity = "graph:" + graph;
    static std::atomic<uint64_t> next_invocation{0};
    const std::string invocation = std::to_string(getpid()) + "_" + std::to_string(next_invocation.fetch_add(1));
    const auto root = std::filesystem::path(environment("GO_FANN_EXPANSION_TRACE_DIR"));
    require(!root.empty(), "missing trace directory");
    std::filesystem::create_directories(root);
    const auto directory = root / ("invocation_" + invocation);
    require(std::filesystem::create_directory(directory), "invocation directory already exists");
    std::string metadata;
    std::string metadata_format = "graph_metadata_sidecar";
    if (!metadata_path.empty()) {
        require(std::filesystem::is_regular_file(metadata_path), "matched metadata sidecar is missing");
        metadata = std::filesystem::canonical(metadata_path).string();
    } else {
        // Some maintained workloads build metadata in memory. Export the exact
        // loaded row map rather than reconstructing it from labels offline.
        const auto mapping_path = directory / "row_to_sample.bin";
        std::ofstream mapping(mapping_path, std::ios::binary);
        require(bool(mapping), "cannot create runtime row mapping");
        for (uint32_t sample : runtime.metadata.row_to_sample) write_u32(mapping, sample);
        mapping.close();
        require(bool(mapping), "runtime row mapping write failed");
        metadata = std::filesystem::canonical(mapping_path).string();
        metadata_format = "trace_row_to_sample_u32";
    }
    std::ofstream route_file(directory / "routes.csv");
    std::ofstream event_file(directory / "events.bin", std::ios::binary);
    require(bool(route_file) && bool(event_file), "cannot create trace output");
    route_file << "route_id,original_query_id,index_identity,group_id,graph_node_count,label_offset,start_local_id,"
                  "enabled,provider,execution_mode,candidate_source,is_lowfreq,is_graph_route,selected_for_trace,executed,"
                  "complete,overflow_count,expansion_count,event_offset,event_count,owned_clause_ids\n";
    uint64_t total_events = 0, total_overflow = 0;
    bool complete = true;
    for (uint32_t route = 0; route < slots.size(); ++route) {
        const bool executed = slots[route] != UINT32_MAX;
        const uint32_t group = runtime.query_group_ids[route];
        const bool valid_group = group < runtime.metadata.label_size.size() && group < runtime.metadata.label_offset.size();
        const uint64_t nodes = valid_group ? runtime.metadata.label_size[group] : 0;
        const uint64_t offset = valid_group ? runtime.metadata.label_offset[group] : 0;
        const Counters count = executed ? counters[slots[route]] : Counters{};
        const uint64_t overflow = count.recorded > capacity ? count.recorded - capacity : 0;
        const uint32_t retained = std::min(count.recorded, capacity);
        const bool route_complete = !executed || (count.completed == 1 && count.expanded == count.recorded && !overflow);
        complete = complete && route_complete;
        total_overflow += overflow;
        route_file << route << ',' << runtime.route_to_original_query[route] << ',' << quote(identity) << ','
                   << group << ',' << nodes << ',' << offset << ',' << runtime.start_local_ids[route] << ','
                   << unsigned(runtime.route_enabled[route]) << ',' << unsigned(runtime.route_providers[route]) << ','
                   << unsigned(runtime.route_execution_modes[route]) << ',' << unsigned(runtime.route_candidate_sources[route]) << ','
                   << unsigned(runtime.query_is_lowfreq[route]) << ',' << graph_route(runtime, route) << ','
                   << executed << ',' << executed << ',' << route_complete << ',' << overflow << ','
                   << count.expanded << ',' << total_events << ',' << retained << ',';
        if (runtime.route_clause_indptr.size() == slots.size() + 1) {
            const uint64_t begin = runtime.route_clause_indptr[route], end = runtime.route_clause_indptr[route + 1];
            require(begin <= end && end <= runtime.route_clause_ids.size(), "invalid owned clauses");
            for (uint64_t j = begin; j < end; ++j) {
                if (j != begin) route_file << ';';
                route_file << runtime.route_clause_ids[j];
            }
        } else if (runtime.route_to_clause.size() == slots.size()) route_file << runtime.route_to_clause[route];
        route_file << '\n';
        for (uint32_t i = 0; i < retained; ++i) {
            const uint32_t local = rows[static_cast<size_t>(slots[route]) * capacity + i];
            require(local < nodes && offset + local < runtime.metadata.row_to_sample.size(), "expanded local row out of bounds");
            const uint32_t sample = runtime.metadata.row_to_sample[offset + local];
            require(sample < runtime.base_vectors.rows, "expanded sample ID out of bounds");
            write_u32(event_file, local);
            write_u32(event_file, sample);
        }
        total_events += retained;
    }
    route_file.close(); event_file.close();
    require(bool(route_file) && bool(event_file), "trace output write failed");
    nlohmann::json manifest = {
        {"schema_version", 1}, {"event_kind", "expanded_adjacency"},
        {"event_format", "little-endian uint32 local_row,uint32 sample_id; route-major"},
        {"invocation_id", invocation}, {"phase", "diagnostic"}, {"repetition", 0},
        {"backend", backend}, {"index_identity", identity}, {"graph_path", graph},
        {"metadata_path", metadata}, {"metadata_format", metadata_format},
        {"metadata_total_rows", runtime.metadata.row_to_sample.size()},
        {"graph_file_size", std::filesystem::file_size(graph)},
        {"metadata_file_size", std::filesystem::file_size(metadata)},
        {"original_query_count", runtime.original_query_count}, {"route_count", slots.size()},
        {"expected_executed_graph_routes", active.size()}, {"event_count", total_events},
        {"degree", runtime.graph.cols}, {"row_bytes", runtime.graph.cols * sizeof(uint32_t)},
        {"graph_data_offset_bytes", reinterpret_cast<uintptr_t>(runtime.graph.data.data()) % 4096},
        {"page_bytes", 4096}, {"layout", "original contiguous host adjacency layout; static-prefetch events use original row address model"},
        {"capacity_per_route", capacity}, {"complete", complete}, {"overflow_count", total_overflow},
        {"base_point_count", runtime.base_vectors.rows}, {"routes_file", "routes.csv"}, {"events_file", "events.bin"},
        {"execution_mode_graph", static_cast<unsigned>(QueryRouteExecution::GraphTraversal)},
        {"run_identity", environment("GO_FANN_EXPANSION_TRACE_RUN_ID")},
        {"binary_sha256", environment("GO_FANN_EXPANSION_TRACE_BINARY_SHA256")},
        {"config_sha256", environment("GO_FANN_EXPANSION_TRACE_CONFIG_SHA256")}
    };
    std::ofstream manifest_file(directory / "manifest.json");
    manifest_file << manifest.dump(2) << '\n'; manifest_file.close();
    require(bool(manifest_file), "manifest write failed");
    require(complete, "incomplete expansion trace or capacity overflow; increase capacity and recollect");
}
} // namespace go_fann::expansion_trace
