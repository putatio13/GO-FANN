#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#define GO_FANN_POSTING_HEAP_INLINE __host__ __device__ __forceinline__
#else
#define GO_FANN_POSTING_HEAP_INLINE inline
#endif

namespace go_fann::posting_shortlist_heap {

template <typename Id>
GO_FANN_POSTING_HEAP_INLINE bool pair_is_worse(float lhs_dist,
                                                 Id lhs_id,
                                                 float rhs_dist,
                                                 Id rhs_id)
{
    return lhs_dist > rhs_dist || (lhs_dist == rhs_dist && lhs_id > rhs_id);
}

template <typename Id>
GO_FANN_POSTING_HEAP_INLINE void sift_down(float* distances,
                                             Id* ids,
                                             int heap_size,
                                             int root)
{
    while (true) {
        const int left = root * 2 + 1;
        if (left >= heap_size) break;
        const int right = left + 1;
        int worse_child = left;
        if (right < heap_size &&
            pair_is_worse(distances[right],
                          ids[right],
                          distances[left],
                          ids[left])) {
            worse_child = right;
        }
        if (!pair_is_worse(distances[worse_child],
                           ids[worse_child],
                           distances[root],
                           ids[root])) {
            break;
        }
        const float swap_dist = distances[root];
        const Id swap_id = ids[root];
        distances[root] = distances[worse_child];
        ids[root] = ids[worse_child];
        distances[worse_child] = swap_dist;
        ids[worse_child] = swap_id;
        root = worse_child;
    }
}

// Admission uses the exact same distance/ID tie and sentinel ordering as insert.
// Reading only the worst valid-heap entry permits score-first predicate elision.
template <typename Id>
GO_FANN_POSTING_HEAP_INLINE bool can_enter(float dist,
                                            Id sample_id,
                                            const float* distances,
                                            const Id* ids,
                                            int shortlist_k)
{
    return sample_id != static_cast<Id>(UINT32_MAX) && shortlist_k > 0 &&
        pair_is_worse(distances[0], ids[0], dist, sample_id);
}

template <typename Id>
GO_FANN_POSTING_HEAP_INLINE void insert(float dist,
                                          Id sample_id,
                                          float* distances,
                                          Id* ids,
                                          int shortlist_k)
{
    if (!can_enter(dist, sample_id, distances, ids, shortlist_k)) {
        return;
    }
    distances[0] = dist;
    ids[0] = sample_id;
    sift_down(distances, ids, shortlist_k, 0);
}

template <typename Id>
GO_FANN_POSTING_HEAP_INLINE void sort(float* distances,
                                        Id* ids,
                                        int shortlist_k)
{
    for (int heap_size = shortlist_k; heap_size > 1; --heap_size) {
        const int tail = heap_size - 1;
        const float swap_dist = distances[0];
        const Id swap_id = ids[0];
        distances[0] = distances[tail];
        ids[0] = ids[tail];
        distances[tail] = swap_dist;
        ids[tail] = swap_id;
        sift_down(distances, ids, tail, 0);
    }
}

}  // namespace go_fann::posting_shortlist_heap

#undef GO_FANN_POSTING_HEAP_INLINE
