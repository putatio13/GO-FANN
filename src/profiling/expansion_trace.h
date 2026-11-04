#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <stdexcept>
#include <vector>

struct GraphRuntime;
namespace go_fann::expansion_trace {

// One query leader owns a slot for its entire coupled traversal. No atomics or
// global inter-route ordering are needed. Repeats are retained, never deduped.
struct Counters {
    uint32_t recorded = 0;
    uint32_t expanded = 0;
    uint32_t completed = 0;
};
struct View {
    uint32_t* rows = nullptr;
    const uint32_t* route_slots = nullptr;
    Counters* counters = nullptr;
    uint32_t capacity = 0;
#ifdef __CUDACC__
    __host__ __device__
#endif
    void append(uint32_t route, uint32_t local_row) const {
        if (!rows) return;
        const uint32_t slot = route_slots[route];
        const uint32_t ordinal = counters[slot].recorded++;
        if (ordinal < capacity) rows[static_cast<size_t>(slot) * capacity + ordinal] = local_row;
    }
#ifdef __CUDACC__
    __host__ __device__
#endif
    void finish(uint32_t route, uint32_t expanded) const {
        if (!rows) return;
        Counters& value = counters[route_slots[route]];
        value.expanded = expanded;
        value.completed = 1;
    }
};

bool enabled();
uint32_t capacity_from_environment();
size_t checked_slots(size_t routes, uint32_t capacity);
std::vector<uint32_t> make_route_slots(const GraphRuntime& runtime,
                                     const std::vector<uint32_t>& active);
void export_invocation(const GraphRuntime& runtime, const std::vector<uint32_t>& active,
                       const std::vector<uint32_t>& rows, const std::vector<Counters>& counters,
                       uint32_t capacity, const std::string& backend,
                       const std::string& graph_path, const std::string& metadata_path);

template<class Options>
void validate_options(const Options& options) {
    if (!enabled()) return;
    if (options.warmup_runs != 0 || options.timed_runs != 1 ||
        options.query_worker_mode != "coupled" || options.graph_transport != "mailbox" ||
        options.graph_storage_mode != "memory" || options.latency_persistent_queue) {
        throw std::runtime_error("expansion trace requires a single diagnostic coupled memory "
                                 "mailbox search (warmup_runs=0,timed_runs=1)");
    }
}
} // namespace go_fann::expansion_trace
