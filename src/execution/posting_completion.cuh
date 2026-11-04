#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace go_fann::posting_completion {

// All lanes of one 32-thread CTA must call this after writing their row.
__device__ __forceinline__ void publish(uint32_t* ready, uint32_t row)
{
    if (ready == nullptr) return;
    // Every writer fences its payload before lane zero publishes completion.
    __threadfence_system();
    __syncwarp();
    if (threadIdx.x == 0) reinterpret_cast<volatile uint32_t*>(ready)[row] = 1;
}

}  // namespace go_fann::posting_completion
