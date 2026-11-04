#pragma once

#include <cstdint>

namespace go_fann::native_visited_hash {

constexpr uint64_t kHashMultiplier = 0x4bcb391f924ed183ULL;
constexpr uint64_t kHashOffset = 0xa0ab69ccd854fc0aULL;

template <int Capacity>
__device__ __forceinline__ uint32_t code(uint32_t value)
{
    const uint64_t mixed = static_cast<uint64_t>(value) ^
        (static_cast<uint64_t>(value >> 16) * kHashMultiplier) ^ kHashOffset;
    return static_cast<uint32_t>(mixed % Capacity);
}

template <int Capacity>
__device__ __forceinline__ void clear(uint32_t* data, int lane)
{
    for (int i = lane; i < Capacity; i += 32) {
        data[i] = UINT32_MAX;
    }
}

template <int Capacity>
__device__ __forceinline__ bool test(const uint32_t* data, uint32_t value)
{
    uint32_t slot = code<Capacity>(value);
    for (int probes = 0; probes < Capacity; ++probes) {
        if (data[slot] == UINT32_MAX) {
            return false;
        }
        if (data[slot] == value) {
            return true;
        }
        slot = (slot + 1) % Capacity;
    }
    return false;
}

template <int Capacity>
__device__ __forceinline__ void add(uint32_t* data, uint32_t value)
{
    uint32_t slot = code<Capacity>(value);
    for (int probes = 0; probes < Capacity; ++probes) {
        if (data[slot] == value) {
            return;
        }
        bool available = data[slot] == UINT32_MAX;
#ifdef __ENABLE_VISITED_DEL
        available = available || data[slot] == static_cast<uint32_t>(-2);
#endif
        if (available) {
            data[slot] = value;
            return;
        }
        slot = (slot + 1) % Capacity;
    }
}

}  // namespace go_fann::native_visited_hash
