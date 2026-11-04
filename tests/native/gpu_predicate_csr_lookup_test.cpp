#include "predicates/gpu_predicate_index.cuh"
#include "gpu_predicate_csr_lookup_cases.h"
#include <cassert>
#include <cstdio>
#ifdef __linux__
#include <sys/mman.h>
#endif

using namespace go_fann::gpu_predicate;

template<uint32_t Threshold>
void check(const csr_lookup_test::Cases& data) {
    // Empty rows must not read the pointer, including the largest valid offset.
    assert(!csr_contains_sorted<Threshold>(nullptr, 0, 0, 0));
    assert(!csr_contains_sorted<Threshold>(nullptr, UINT32_MAX, UINT32_MAX, UINT32_MAX));
    for (const auto& probe : data.probes) {
        if (probe.row >= data.offsets.size() - 1) continue;
        assert(csr_contains_sorted<Threshold>(data.labels.data(), data.offsets[probe.row],
                   data.offsets[probe.row + 1], probe.label) == probe.expected);
    }
}

template<uint32_t Threshold>
void check_last_offset(const uint32_t* labels, uint32_t begin) {
    for (uint32_t i = begin; i < UINT32_MAX; ++i) {
        assert(csr_contains_sorted<Threshold>(labels, begin, UINT32_MAX, labels[i]));
        assert(!csr_contains_sorted<Threshold>(labels, begin, UINT32_MAX, labels[i] - 1));
    }
    assert(!csr_contains_sorted<Threshold>(labels, begin, UINT32_MAX, UINT32_MAX));
}

int main() {
    const csr_lookup_test::Cases data;
    check<0>(data); check<1>(data); check<8>(data); check<16>(data); check<32>(data); check<64>(data);
    View v;
    v.rows = data.offsets.size() - 1;
    v.csr_offsets = data.offsets.data(); v.csr_labels = data.labels.data();
    for (const auto& probe : data.probes)
        assert(contains(v, probe.label, probe.row) == probe.expected);
#ifdef __linux__
    // Reserve virtual address space only. Touch the final page to test real
    // uint32 offsets near UINT32_MAX without allocating a 16 GiB physical CSR.
    const size_t bytes = uint64_t(UINT32_MAX) * sizeof(uint32_t);
    void* mapping = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    assert(mapping != MAP_FAILED);
    auto* labels = static_cast<uint32_t*>(mapping);
    const uint32_t begin = UINT32_MAX - 65;
    for (uint32_t i = begin; i < UINT32_MAX; ++i) labels[i] = 2 + (i - begin) * 2;
    check_last_offset<0>(labels, begin); check_last_offset<1>(labels, begin);
    check_last_offset<8>(labels, begin); check_last_offset<16>(labels, begin);
    check_last_offset<32>(labels, begin); check_last_offset<64>(labels, begin);
    assert(munmap(mapping, bytes) == 0);
#endif
    std::printf("CSR lookup host parity passed: thresholds 0/1/8/16/32/64, %zu probes, "
                "empty/long rows, full uint32 IDs/offsets (active threshold=%d)\n",
                data.probes.size(), GO_FANN_GPU_PREDICATE_CSR_THRESHOLD);
}
