#pragma once

#include "search/common/posting_shortlist_heap.cuh"

#include <cfloat>
#include <cuda_runtime.h>

namespace go_fann::posting_segments {

// A segment retains K candidates; their union contains the original global K.
// Called collectively by one 32-thread CTA, with K slots of shared scratch.
template <typename Id>
__device__ __forceinline__ void merge_one_query(uint32_t query,
                                 const uint32_t* query_indptr,
                                 const Id* segment_ids,
                                 const float* segment_distances,
                                 const uint32_t* segment_pass_counts,
                                 Id* result,
                                 uint32_t* pass_counts,
                                 int shortlist_k,
                                 float* distances,
                                 Id* ids)
{
    for (int i = threadIdx.x; i < shortlist_k; i += blockDim.x) {
        distances[i] = FLT_MAX;
        ids[i] = UINT32_MAX;
    }
    __syncwarp();
    if (threadIdx.x == 0) {
        uint32_t passes = 0;
        for (uint32_t part = query_indptr[query]; part < query_indptr[query + 1]; ++part) {
            passes += segment_pass_counts[part];
            const size_t offset = static_cast<size_t>(part) * shortlist_k;
            for (int i = 0; i < shortlist_k; ++i) {
                posting_shortlist_heap::insert(segment_distances[offset + i],
                                               segment_ids[offset + i],
                                               distances, ids, shortlist_k);
            }
        }
        pass_counts[query] = passes;
    }
    __syncwarp();
    for (int i = threadIdx.x; i < shortlist_k; i += blockDim.x) {
        result[static_cast<size_t>(query) * shortlist_k + i] = ids[i];
    }
}

template <int Capacity, typename Id>
__global__ void merge_shortlists(const uint32_t* query_indptr,
                                 const Id* segment_ids,
                                 const float* segment_distances,
                                 const uint32_t* segment_pass_counts,
                                 Id* result,
                                 uint32_t* pass_counts,
                                 int shortlist_k)
{
    __shared__ float distances[Capacity];
    __shared__ Id ids[Capacity];
    merge_one_query(blockIdx.x, query_indptr, segment_ids, segment_distances,
                    segment_pass_counts, result, pass_counts, shortlist_k, distances, ids);
}

template <typename Id>
struct Completion {
    const uint32_t* query_positions = nullptr;
    const uint32_t* query_indptr = nullptr;
    uint32_t* counters = nullptr;
    Id* merged_ids = nullptr;
    uint32_t* merged_pass_counts = nullptr;
    uint32_t* ready = nullptr;
};

}  // namespace go_fann::posting_segments
