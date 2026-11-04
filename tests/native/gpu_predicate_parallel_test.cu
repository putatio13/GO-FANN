#include "predicates/gpu_predicate_parallel.cuh"
#include <algorithm>
#include <cstdio>
#include "smmh2.h"
#include <cstdlib>
#include <vector>

using namespace go_fann::gpu_predicate;

#define CUDA(call) do { const auto e = (call); if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e)); std::exit(1); \
} } while (0)

constexpr unsigned kRoutes = 7, kRows = 67, kCandidates = 97, kOffset = 3;
constexpr unsigned kCounts[] = {0, 1, 17, 31, 32, 33, 63, 64, 65, 97};

template<class T> T* upload(const std::vector<T>& values) {
    T* p = nullptr;
    CUDA(cudaMallocManaged(&p, values.size() * sizeof(T)));
    std::copy(values.begin(), values.end(), p);
    return p;
}

__device__ bool same_counter(const Counters& a, const Counters& b) {
    return a.label_probes == b.label_probes && a.range_compares == b.range_compares &&
           a.entailed_fastpaths == b.entailed_fastpaths && a.expanded == b.expanded &&
           a.proposed == b.proposed && a.passed == b.passed && a.lookup_ns == b.lookup_ns &&
           a.saturated == b.saturated && a.scores == b.scores && a.trace_count == b.trace_count;
}

__global__ void parity_fixture(View v, const uint32_t* mapping,
                               const uint8_t* expected, Counters* observed,
                               const Counters* expected_counters, unsigned* errors) {
    const unsigned lane = threadIdx.x % 32, route = threadIdx.x / 32;
    __shared__ uint32_t candidates[kRoutes][kCandidates];
    __shared__ uint32_t masks[kRoutes][(kCandidates + 31) / 32];
    const unsigned counts[] = {0, 1, 17, 31, 32, 33, 63, 64, 65, 97};
    for (unsigned i = lane; i < kCandidates; i += 32)
        candidates[route][i] = (i * 37) % kCandidates;
    __syncwarp();
    for (unsigned count : counts) {
        if (lane < 4) masks[route][lane] = 0xdeadbeefu;
        __syncwarp();
        const bool parallel = prepare_candidate_pass_masks(
            v, route, candidates[route], count, kOffset, mapping, masks[route], lane);
        const bool should_parallel = GO_FANN_GPU_PREDICATE_PARALLEL && enabled(v, route);
        if (parallel != should_parallel) atomicAdd(errors, 1);
        if (lane == 0) {
            BiasedPair scalar_heap[18], parallel_heap[18];
            int scalar_size = 1, parallel_size = 1;
            for (unsigned i = 0; i < count; ++i) {
                const auto sample = mapping[kOffset + candidates[route][i]];
                const bool oracle = expected[route * (kRows + 1) + (sample < kRows ? sample : kRows)];
                const bool scalar = !enabled(v, route) || matches(v, route, sample);
                const bool actual = parallel ? ((masks[route][i / 32] >> (i % 32)) & 1u) : scalar;
                if (actual != (!enabled(v, route) || oracle) || scalar != actual) atomicAdd(errors, 1);
                // Match the production ordered insert/evict branch, including
                // ties and failed-but-navigable candidates. Verify whole entries.
                const float base = float(int(i % 17) - 7);
                bool saturated = false;
                BiasedPair a{priority(base, 8, scalar, saturated), i, base};
                BiasedPair b{priority(base, 8, actual, saturated), i, base};
                if (!(scalar_size >= 9 && scalar_heap[2].first < a.first)) {
                    smmh2::insert(scalar_heap, scalar_size, a);
                    if (scalar_size >= 10) smmh2::pop_max(scalar_heap, scalar_size);
                }
                if (!(parallel_size >= 9 && parallel_heap[2].first < b.first)) {
                    smmh2::insert(parallel_heap, parallel_size, b);
                    if (parallel_size >= 10) smmh2::pop_max(parallel_heap, parallel_size);
                }
                if (scalar_size != parallel_size) atomicAdd(errors, 1);
                for (int j = 1; j < scalar_size; ++j)
                    if (scalar_heap[j].first != parallel_heap[j].first ||
                        scalar_heap[j].second != parallel_heap[j].second ||
                        scalar_heap[j].base != parallel_heap[j].base) atomicAdd(errors, 1);
            }
            // Every inactive ballot lane is false; unwritten chunks stay intact.
            if (parallel && count % 32 && (masks[route][count / 32] >> (count % 32))) atomicAdd(errors, 1);
            for (unsigned chunk = (parallel ? (count + 31) / 32 : 0); chunk < 4; ++chunk)
                if (masks[route][chunk] != 0xdeadbeefu) atomicAdd(errors, 1);
        }
        __syncwarp();
        View diagnostic = v;
        diagnostic.counters = observed;
        // An observer must never enter the parallel evaluator, even at count 0.
        if (prepare_candidate_pass_masks(diagnostic, route, candidates[route],
                                         count, kOffset, mapping, masks[route], lane))
            atomicAdd(errors, 1);
        if (lane == 0)
            for (unsigned i = 0; i < count; ++i)
                if (enabled(v, route))
                    matches(diagnostic, route, mapping[kOffset + candidates[route][i]]);
        __syncwarp();
    }
    if (lane == 0 && !same_counter(observed[route], expected_counters[route])) atomicAdd(errors, 1);
}

int main() {
    std::vector<uint32_t> csr_offsets{0}, csr_labels, directory{0, 1};
    std::vector<uint16_t> payload;
    for (unsigned sample = 0; sample < kRows; ++sample) {
        if (sample % 2 == 0) csr_labels.push_back(0);
        if (sample % 3 == 0) csr_labels.push_back(1);
        if (sample % 5 == 0) csr_labels.push_back(65536);
        csr_offsets.push_back(csr_labels.size());
    }
    std::vector<Descriptor> descriptors;
    for (unsigned label = 0; label < 2; ++label) {
        const auto begin = payload.size();
        for (unsigned sample = 0; sample < kRows; ++sample)
            if (sample % (label + 2) == 0) payload.push_back(sample);
        descriptors.push_back({begin, uint32_t(payload.size() - begin), 0});
    }
    std::vector<float> values;
    for (unsigned sample = 0; sample < kRows; ++sample) {
        values.push_back(float(sample % 9));
        values.push_back(float(sample % 7));
    }
    std::vector<uint32_t> mapping(kOffset + kCandidates, UINT32_MAX);
    for (unsigned i = 0; i < kCandidates; ++i)
        mapping[kOffset + i] = i % 13 == 0 ? UINT32_MAX : (i * 19) % (kRows + 1);
    View v;
    v.rows = kRows; v.chunks = 1; v.range_stride = 2;
    v.directory = upload(directory); v.descriptors = upload(descriptors); v.payload = upload(payload);
    const auto* csr_row = upload(csr_offsets); const auto* csr_data = upload(csr_labels);
    v.route_offsets = upload<uint32_t>({0, 1, 3, 4, 4, 5, 6, 7});
    v.route_handles = upload<uint32_t>({0, 1});
    v.route_enabled = upload<uint8_t>({1, 1, 1, 1, 0, 1, 1});
    v.clauses = upload<Clause>({{0,0,0,0}, {0,1,0,1}, {1,2,0,0},
                              {0,0,1,2}, {0,1,0,0}, {0,2,0,0}, {0,0,0,4}});
    v.ranges = upload<RangeAtom>({{0,1,3,0}, {1,3,2,4}, {0,2,0,7}, {0,0,0,0}});
    v.base_ranges = upload(values);
    const auto* device_mapping = upload(mapping);
    unsigned* errors = upload<unsigned>({0});
    for (unsigned storage = 0; storage < 3; ++storage) {
        View current = v;
        if (storage) current.directory = nullptr;
        if (storage == 1) { current.csr_offsets = csr_row; current.csr_labels = csr_data; }
        std::vector<uint8_t> expected(kRoutes * (kRows + 1));
        std::vector<Counters> reference(kRoutes);
        for (unsigned route = 0; route < kRoutes; ++route) {
            for (unsigned sample = 0; sample <= kRows; ++sample)
                expected[route * (kRows + 1) + sample] = matches(current, route, sample);
            View diagnostic = current; diagnostic.counters = reference.data();
            for (unsigned count : kCounts)
                for (unsigned i = 0; i < count; ++i)
                    if (enabled(current, route))
                        matches(diagnostic, route, mapping[kOffset + (i * 37) % kCandidates]);
        }
        auto* device_expected = upload(expected);
        auto* expected_counters = upload(reference);
        auto* observed = upload(std::vector<Counters>(kRoutes));
        parity_fixture<<<1, kRoutes * 32>>>(current, device_mapping, device_expected,
                                           observed, expected_counters, errors);
        CUDA(cudaGetLastError()); CUDA(cudaDeviceSynchronize());
        CUDA(cudaFree(device_expected)); CUDA(cudaFree(expected_counters)); CUDA(cudaFree(observed));
    }
    if (*errors) { std::fprintf(stderr, "%u predicate parity errors\n", *errors); return 1; }
    std::printf("GPU candidate predicate parity passed (parallel=%d): CSR/Roaring/no-index, "
                "DNF/ranges/entailed/disabled/invalid, partial chunks, ordered heaps, counters\n",
                GO_FANN_GPU_PREDICATE_PARALLEL);
    for (const void* p : std::vector<const void*>{v.directory, v.descriptors, v.payload,
                         v.route_offsets, v.route_handles, v.route_enabled, v.clauses,
                         v.ranges, v.base_ranges, csr_row, csr_data, device_mapping, errors})
        CUDA(cudaFree(const_cast<void*>(p)));
}
