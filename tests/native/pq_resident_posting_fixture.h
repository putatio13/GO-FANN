#pragma once
#include "search/pq/pq_resident_posting.cuh"
#include <cassert>
#include <limits>

struct PqResidentFixture {
    uint32_t offsets[2] = {0, 3}, sizes[2] = {3, 3};
    uint32_t ids[6] = {4, 2, 0, 5, 1, 3};
    uint64_t masks[6] = {1, 2, 1, 1, 2, 3};
    uint32_t groups[3] = {1, 0, 1};
    uint64_t query_masks[3] = {2, 1, 1};
    uint8_t modes[3] = {1, 2, 3};
    float lower[3] = {2, 0, 2}, upper[3] = {0, 5, 4};
    float values[6] = {2, 1, 0, 4, std::numeric_limits<float>::quiet_NaN(), 3};
    go_fann::pq_resident_posting::View view() const {
        return {offsets,sizes,ids,masks,groups,query_masks,modes,lower,upper,values,
                2,6,6,3,1,true};
    }
};

// Query ID 2 is processed first: compact position zero must not select route 0.
__host__ __device__ inline void pq_resident_probe(
    go_fann::pq_resident_posting::View view, uint32_t* out)
{
    using namespace go_fann::pq_resident_posting;
    const uint32_t routes[] = {2, 0, 1};
    for (uint32_t q = 0; q < 3; ++q) {
        uint32_t begin = 0, count = 0;
        bounds(view, routes[q], begin, count);
        for (uint32_t i = 0; i < 3; ++i)
            out[q * 3 + i] = i < count ? candidate(view, routes[q], begin + i) : UINT32_MAX;
    }
}

inline void pq_resident_expect(const uint32_t* out) {
    const uint32_t expected[] = {5,UINT32_MAX,UINT32_MAX, UINT32_MAX,UINT32_MAX,3,
                                UINT32_MAX,UINT32_MAX,0};
    for (size_t i = 0; i < 9; ++i) assert(out[i] == expected[i]);
}
