#pragma once

#include <cuda_runtime.h>

#include "execution/async_io.h"

namespace go_fann::request_ring_device {

__device__ __forceinline__ uint64_t load_consumed_sequence(
    const volatile uint64_t* address)
{
    uint64_t value;
    asm volatile("ld.global.cg.u64 %0, [%1];"
                 : "=l"(value)
                 : "l"(address)
                 : "memory");
    return value;
}

__device__ __forceinline__ void store_published_sequence(
    uint64_t* address,
    uint64_t value)
{
    asm volatile("st.release.sys.b64 [%0], %1;"
                 :
                 : "l"(address), "l"(value)
                 : "memory");
}

__device__ __forceinline__ void publish_host_request(
    HostRequestRingEntry* entries,
    const volatile uint64_t* consumed_sequences,
    unsigned long long* tails,
    uint32_t ring_idx,
    uint32_t capacity_per_ring,
    uint32_t query_id,
    HostRequestKind kind)
{
    if (entries == nullptr || consumed_sequences == nullptr || tails == nullptr ||
        capacity_per_ring == 0) {
        return;
    }
    const uint64_t ticket = atomicAdd(tails + ring_idx, 1ULL);
    const size_t slot = static_cast<size_t>(ring_idx) * capacity_per_ring +
        static_cast<size_t>(ticket % capacity_per_ring);
    const uint64_t required_consumed = ticket < capacity_per_ring
        ? 0ULL
        : ticket - capacity_per_ring + 1ULL;
    while (load_consumed_sequence(consumed_sequences + slot) != required_consumed) {
        __nanosleep(64);
    }

    volatile HostRequestRingEntry* entry = entries + slot;
    entry->query_id = query_id;
    entry->kind = static_cast<uint32_t>(kind);
    store_published_sequence(&entries[slot].published_sequence, ticket + 1ULL);
}

}  // namespace go_fann::request_ring_device
