#include <cuda_runtime.h>
#include "pq_resident_posting_fixture.h"
#include <iostream>

__global__ void probe(go_fann::pq_resident_posting::View v, uint32_t* out) {
    pq_resident_probe(v, out);
}

int main() {
    PqResidentFixture host;
    PqResidentFixture* device = nullptr;
    uint32_t* result = nullptr;
    assert(cudaMalloc(&device, sizeof(host)) == cudaSuccess);
    assert(cudaMemcpy(device, &host, sizeof(host), cudaMemcpyHostToDevice) == cudaSuccess);
    assert(cudaMalloc(&result, 9 * sizeof(uint32_t)) == cudaSuccess);
    // Form device pointers without dereferencing device memory on the host.
    auto v = host.view();
    v.group_offsets = device->offsets; v.group_sizes = device->sizes;
    v.sample_ids = device->ids; v.row_masks = device->masks;
    v.query_groups = device->groups; v.query_masks = device->query_masks;
    v.range_modes = device->modes; v.range_lower = device->lower;
    v.range_upper = device->upper; v.base_ranges = device->values;
    probe<<<1, 1>>>(v, result);
    assert(cudaGetLastError() == cudaSuccess);
    uint32_t out[9];
    assert(cudaMemcpy(out, result, sizeof(out), cudaMemcpyDeviceToHost) == cudaSuccess);
    pq_resident_expect(out);
    assert(cudaFree(result) == cudaSuccess);
    assert(cudaFree(device) == cudaSuccess);
    std::cout << "PQ resident CUDA producer mapping parity passed\n";
}
