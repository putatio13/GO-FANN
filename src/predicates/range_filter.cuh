#pragma once

#include <cstddef>
#include <cmath>
#include <cstdint>

#ifndef __CUDACC__
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#ifndef __forceinline__
#define __forceinline__ inline __attribute__((always_inline))
#endif
#endif

#include "data/dataloader.h"

namespace go_fann::range_filter {

__host__ __device__ __forceinline__ bool range_filter_active(uint8_t query_range_mode)
{
    return query_range_mode != 0;
}

__host__ __device__ __forceinline__ bool range_filter_matches_single(uint8_t query_range_mode,
                                                                     float query_range_lower,
                                                                     float query_range_upper,
                                                                     float base_range_value)
{
    switch (query_range_mode) {
    case 0:
        return true;
    case 1:
        return base_range_value > query_range_lower;
    case 2:
        return base_range_value < query_range_upper;
    case 3:
        return base_range_value >= query_range_lower && base_range_value < query_range_upper;
    default:
        return false;
    }
}

__host__ __device__ __forceinline__ uint32_t count_matching_active_ranges(const uint8_t* query_range_modes,
                                                                          const float* query_range_lower_bounds,
                                                                          const float* query_range_upper_bounds,
                                                                          const float* base_range_values,
                                                                          uint32_t query_idx,
                                                                          uint32_t sample_idx,
                                                                          uint32_t range_count)
{
    if (query_range_modes == nullptr || query_range_lower_bounds == nullptr || query_range_upper_bounds == nullptr) {
        return 0;
    }
    uint32_t matched_active_dims = 0;
    for (uint32_t range_idx = 0; range_idx < range_count; ++range_idx) {
        const size_t query_flat = static_cast<size_t>(query_idx) * range_count + range_idx;
        const uint8_t mode = query_range_modes[query_flat];
        if (!range_filter_active(mode)) {
            continue;
        }
        if (base_range_values == nullptr) {
            return 0;
        }
        const size_t base_flat = static_cast<size_t>(sample_idx) * range_count + range_idx;
        if (range_filter_matches_single(mode,
                                        query_range_lower_bounds[query_flat],
                                        query_range_upper_bounds[query_flat],
                                        base_range_values[base_flat])) {
            ++matched_active_dims;
        }
    }
    return matched_active_dims;
}

__host__ __device__ __forceinline__ uint32_t count_active_ranges(const uint8_t* query_range_modes,
                                                                 uint32_t query_idx,
                                                                 uint32_t range_count)
{
    if (query_range_modes == nullptr) {
        return 0;
    }
    uint32_t active = 0;
    for (uint32_t range_idx = 0; range_idx < range_count; ++range_idx) {
        if (range_filter_active(query_range_modes[static_cast<size_t>(query_idx) * range_count + range_idx])) {
            ++active;
        }
    }
    return active;
}

__host__ __device__ __forceinline__ bool query_has_active_range(const uint8_t* query_range_modes,
                                                                uint32_t query_idx,
                                                                uint32_t range_count)
{
    if (query_range_modes == nullptr) {
        return false;
    }
    for (uint32_t range_idx = 0; range_idx < range_count; ++range_idx) {
        if (range_filter_active(query_range_modes[static_cast<size_t>(query_idx) * range_count + range_idx])) {
            return true;
        }
    }
    return false;
}

template <typename DistT>
__host__ __device__ __forceinline__ DistT apply_filter_bonuses(DistT dist,
                                                               bool bit_match,
                                                               uint32_t matched_active_dims,
                                                               float bit_filter_bonus,
                                                               float range_filter_bonus)
{
    if (bit_match) {
        dist *= bit_filter_bonus;
    }
    for (uint32_t i = 0; i < matched_active_dims; ++i) {
        dist *= range_filter_bonus;
    }
    return dist;
}

inline bool sample_matches_all_active_ranges_host(const QueryRangeTable& ranges,
                                                  uint32_t query_idx,
                                                  const std::vector<float>& base_range_values,
                                                  uint32_t base_range_count,
                                                  uint32_t sample_idx)
{
    if (ranges.range_count != base_range_count) {
        return false;
    }
    return count_matching_active_ranges(ranges.modes.data(),
                                        ranges.lower_bounds.data(),
                                        ranges.upper_bounds.data(),
                                        base_range_values.data(),
                                        query_idx,
                                        sample_idx,
                                        base_range_count) ==
           count_active_ranges(ranges.modes.data(), query_idx, base_range_count);
}

}  // namespace go_fann::range_filter
