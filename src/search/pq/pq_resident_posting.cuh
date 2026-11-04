#pragma once

#include "predicates/range_filter.cuh"

namespace go_fann::pq_resident_posting {

// All query metadata is indexed by routed query ID, never compact batch ID.
struct View {
    const uint32_t* group_offsets = nullptr;
    const uint32_t* group_sizes = nullptr;
    const uint32_t* sample_ids = nullptr;
    const uint64_t* row_masks = nullptr;
    const uint32_t* query_groups = nullptr;
    const uint64_t* query_masks = nullptr;
    const uint8_t* range_modes = nullptr;
    const float* range_lower = nullptr;
    const float* range_upper = nullptr;
    const float* base_ranges = nullptr;
    uint32_t groups = 0;
    uint32_t rows = 0;
    uint32_t samples = 0;
    uint32_t routes = 0;
    uint32_t range_count = 0;
    bool active = false;
};

__host__ __device__ inline bool bounds(const View& v, uint32_t route,
                                     uint32_t& begin, uint32_t& count)
{
    begin = count = 0;
    if (route >= v.routes) return false;
    const uint32_t group = v.query_groups[route];
    if (group >= v.groups) return false;
    begin = v.group_offsets[group];
    count = v.group_sizes[group];
    if (begin > v.rows || count > v.rows - begin) {
        begin = count = 0;
        return false;
    }
    return true;
}

// Apply producer constraints before scoring.
// Full DNF/CSR admission remains a separate operation.
__host__ __device__ inline uint32_t candidate(const View& v, uint32_t route,
                                             uint32_t row)
{
    if (route >= v.routes || row >= v.rows ||
        (v.row_masks[row] & v.query_masks[route]) == 0) return UINT32_MAX;
    const uint32_t sample = v.sample_ids[row];
    if (sample >= v.samples) return UINT32_MAX;
    if (v.base_ranges != nullptr &&
        range_filter::count_matching_active_ranges(v.range_modes, v.range_lower,
            v.range_upper, v.base_ranges, route, sample, v.range_count) !=
        range_filter::count_active_ranges(v.range_modes, route, v.range_count))
        return UINT32_MAX;
    return sample;
}

} // namespace go_fann::pq_resident_posting
