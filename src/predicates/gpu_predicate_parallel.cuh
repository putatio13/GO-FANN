#pragma once

#include "predicates/gpu_predicate_index.cuh"
#include <cuda_runtime.h>

#ifndef GO_FANN_GPU_PREDICATE_PARALLEL
#define GO_FANN_GPU_PREDICATE_PARALLEL 1
#endif
static_assert(GO_FANN_GPU_PREDICATE_PARALLEL == 0 ||
              GO_FANN_GPU_PREDICATE_PARALLEL == 1,
              "GO_FANN_GPU_PREDICATE_PARALLEL must be 0 or 1");

namespace go_fann::gpu_predicate {

// All 32 lanes of one query warp call with identical arguments except lane.
// Each lane checks one candidate, rather than cooperating on one short row.
// Output needs ceil(count / 32) words in warp-private shared memory. Its bits
// retain candidate order, so the caller can keep serial heap/visited mutations.
// The caller must have published local_rows before entering this function.
// A false return leaves output untouched and requires the existing scalar path.
__device__ inline bool prepare_candidate_pass_masks(
    const View& view, uint32_t route, const uint32_t* local_rows,
    uint32_t count, uint32_t label_offset, const uint32_t* row_to_sample,
    uint32_t* output, uint32_t lane)
{
#if GO_FANN_GPU_PREDICATE_PARALLEL
    // matches() has non-atomic route counters. Preserve the original ordered
    // scalar observer, including its trace order and measured lookup duration.
    if (view.counters || !enabled(view, route)) return false;
    for (uint32_t base = 0; base < count; base += 32) {
        const uint32_t i = base + lane;
        const bool pass = i < count && matches(
            view, route, row_to_sample[label_offset + local_rows[i]]);
        const uint32_t mask = __ballot_sync(0xffffffffu, pass);
        if (lane == 0) output[base / 32] = mask;
    }
    __syncwarp();
    return true;
#else
    return false;
#endif
}

}  // namespace go_fann::gpu_predicate
