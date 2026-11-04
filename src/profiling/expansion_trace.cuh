#pragma once
#include "profiling/expansion_trace.h"
#include "data/dataloader.h"
#include <cuda_runtime.h>
#include <stdexcept>

namespace go_fann::expansion_trace {
// Scoped to one complete graph invocation; allocate/reset before its timer and
// download/export after its timer. A disabled collector makes no CUDA calls.
class Invocation {
    View view_{};
    size_t active_count_ = 0;
    static void check(cudaError_t result) {
        if (result != cudaSuccess)
            throw std::runtime_error(std::string("expansion trace CUDA: ") + cudaGetErrorString(result));
    }
    void release() noexcept {
        if (view_.rows) cudaFree(view_.rows);
        if (view_.route_slots) cudaFree(const_cast<uint32_t*>(view_.route_slots));
        if (view_.counters) cudaFree(view_.counters);
        view_ = {};
    }
public:
    Invocation(const GraphRuntime& runtime, const std::vector<uint32_t>& active) {
        if (!enabled()) return;
        const auto slots = make_route_slots(runtime, active);
        view_.capacity = capacity_from_environment();
        active_count_ = active.size();
        if (active.empty()) return;
        try {
            check(cudaMalloc(&view_.rows, checked_slots(active.size(), view_.capacity) * sizeof(uint32_t)));
            uint32_t* device_slots = nullptr;
            check(cudaMalloc(&device_slots, slots.size() * sizeof(uint32_t)));
            view_.route_slots = device_slots;
            check(cudaMemcpy(device_slots, slots.data(), slots.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
            check(cudaMalloc(&view_.counters, active.size() * sizeof(Counters)));
            check(cudaMemset(view_.counters, 0, active.size() * sizeof(Counters)));
        } catch (...) { release(); throw; }
    }
    ~Invocation() { release(); }
    Invocation(const Invocation&) = delete;
    Invocation& operator=(const Invocation&) = delete;
    View view() const { return view_; }
    void write(const GraphRuntime& runtime, const std::vector<uint32_t>& active,
               const std::string& backend, const std::string& graph, const std::string& metadata) {
        if (!enabled()) return;
        std::vector<Counters> counts(active_count_);
        std::vector<uint32_t> rows(checked_slots(active_count_, view_.capacity));
        if (active_count_) {
            check(cudaMemcpy(counts.data(), view_.counters, counts.size() * sizeof(Counters), cudaMemcpyDeviceToHost));
            check(cudaMemcpy(rows.data(), view_.rows, rows.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        }
        export_invocation(runtime, active, rows, counts, view_.capacity, backend, graph, metadata);
    }
};
} // namespace go_fann::expansion_trace
