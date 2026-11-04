#pragma once

#include "predicates/gpu_predicate_parallel.cuh"

namespace go_fann::pq_gpu_predicate {

// PQ disk mailboxes already carry original sample IDs. Preserve this mapping
// instead of allocating an expanded row-to-sample table solely for predicates.
// All lanes of the primary query warp enter uniformly after list publication.
template<class LocalRow>
__device__ inline bool prepare_candidate_pass_masks(
    const gpu_predicate::View& view, uint32_t route, const LocalRow* local_rows,
    const uint32_t* sample_ids, uint32_t count, uint32_t label_offset,
    const uint32_t* row_to_sample, bool sample_payload, uint32_t* output,
    uint32_t lane)
{
#if GO_FANN_GPU_PREDICATE_PARALLEL
    if (view.counters || !gpu_predicate::enabled(view, route)) return false;
    for (uint32_t base = 0; base < count; base += 32) {
        const uint32_t i = base + lane;
        bool pass = false;
        if (i < count) {
            const uint32_t sample = sample_payload ? sample_ids[i]
                : row_to_sample[label_offset + local_rows[i]];
            pass = gpu_predicate::matches(view, route, sample);
        }
        const uint32_t mask = __ballot_sync(0xffffffffu, pass);
        if (lane == 0) output[base / 32] = mask;
    }
    __syncwarp();
    return true;
#else
    return false;
#endif
}

}  // namespace go_fann::pq_gpu_predicate
