#include "predicates/gpu_predicate_parallel.cuh"
#include "gpu_predicate_csr_lookup_cases.h"
#include <cstdio>
#include <cstdlib>

using namespace go_fann::gpu_predicate;
#define CUDA(call) do { const auto e = (call); if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e)); std::exit(1); \
} } while (0)

template<class T> T* upload(const std::vector<T>& values) {
    T* p = nullptr;
    CUDA(cudaMallocManaged(&p, values.size() * sizeof(T)));
    std::copy(values.begin(), values.end(), p);
    return p;
}

template<uint32_t Threshold>
__device__ bool check_probe(const View& v, const csr_lookup_test::Probe& probe) {
    if (probe.row >= v.rows) return true;
    return csr_contains_sorted<Threshold>(v.csr_labels, v.csr_offsets[probe.row],
        v.csr_offsets[probe.row + 1], probe.label) == probe.expected;
}

__global__ void lookup_fixture(View v, const csr_lookup_test::Probe* probes,
                               uint32_t count, unsigned* errors) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const auto probe = probes[i];
    if (!check_probe<0>(v, probe) || !check_probe<1>(v, probe) ||
        !check_probe<8>(v, probe) || !check_probe<16>(v, probe) ||
        !check_probe<32>(v, probe) || !check_probe<64>(v, probe) ||
        contains(v, probe.label, probe.row) != probe.expected)
        atomicAdd(errors, 1);
    if (csr_contains_sorted<8>(nullptr, UINT32_MAX, UINT32_MAX, UINT32_MAX))
        atomicAdd(errors, 1);
}

__global__ void candidate_fixture(View v, const uint32_t* mapping,
                                  const uint8_t* expected, unsigned* errors) {
    constexpr unsigned max_candidates = 97;
    const unsigned lane = threadIdx.x % 32, route = threadIdx.x / 32;
    __shared__ uint32_t candidates[3][max_candidates], masks[3][4];
    for (unsigned i = lane; i < max_candidates; i += 32) candidates[route][i] = i;
    __syncwarp();
    const unsigned counts[] = {1, 17, 31, 32, 33, 65, 97};
    for (unsigned count : counts) {
        if (lane < 4) masks[route][lane] = 0xdeadbeefu;
        __syncwarp();
        if (!prepare_candidate_pass_masks(v, route, candidates[route], count,
                                          0, mapping, masks[route], lane))
            atomicAdd(errors, 1);
        if (lane == 0) {
            for (unsigned i = 0; i < count; ++i)
                if (((masks[route][i / 32] >> (i % 32)) & 1u) != expected[route * max_candidates + i])
                    atomicAdd(errors, 1);
            if (count % 32 && (masks[route][count / 32] >> (count % 32))) atomicAdd(errors, 1);
            for (unsigned chunk = (count + 31) / 32; chunk < 4; ++chunk)
                if (masks[route][chunk] != 0xdeadbeefu) atomicAdd(errors, 1);
        }
        __syncwarp();
    }
}

int main() {
    const csr_lookup_test::Cases data;
    View v;
    v.rows = data.offsets.size() - 1;
    v.csr_offsets = upload(data.offsets); v.csr_labels = upload(data.labels);
    v.route_offsets = upload<uint32_t>({0, 1, 2, 3});
    v.route_handles = upload<uint32_t>({0, 65536, UINT32_MAX});
    std::vector<uint32_t> mapping(97);
    std::vector<uint8_t> expected(3 * 97);
    for (uint32_t i = 0; i < 97; ++i) {
        mapping[i] = i % 19 == 0 ? UINT32_MAX : i % v.rows;
        for (unsigned route = 0; route < 3; ++route)
            expected[route * 97 + i] = data.expected(mapping[i], v.route_handles[route]);
    }
    auto* probes = upload(data.probes);
    auto* device_mapping = upload(mapping);
    auto* device_expected = upload(expected);
    auto* errors = upload<unsigned>({0});
    lookup_fixture<<<(data.probes.size() + 127) / 128, 128>>>(v, probes, data.probes.size(), errors);
    CUDA(cudaGetLastError()); CUDA(cudaDeviceSynchronize());
    candidate_fixture<<<1, 96>>>(v, device_mapping, device_expected, errors);
    CUDA(cudaGetLastError()); CUDA(cudaDeviceSynchronize());
    if (*errors) { std::fprintf(stderr, "%u CSR lookup parity errors\n", *errors); return 1; }
    std::printf("GPU CSR lookup parity passed: thresholds 0/1/8/16/32/64, long/empty rows, "
                "uint32 IDs, candidate divergence/tails (active threshold=%d)\n",
                GO_FANN_GPU_PREDICATE_CSR_THRESHOLD);
    for (const void* p : std::vector<const void*>{v.csr_offsets, v.csr_labels,
                          v.route_offsets, v.route_handles, probes, device_mapping,
                          device_expected, errors})
        CUDA(cudaFree(const_cast<void*>(p)));
}
