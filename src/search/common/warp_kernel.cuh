#pragma once

#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>

#include "type_def.h"

struct Node;

namespace go_fann::warp_kernel {

constexpr uint32_t kFullMask = 0xffffffffu;
constexpr int kWarpThreads = 32;

template <class A, class B>
struct KernelPair {
    A first;
    B second;

    __device__ KernelPair() {}

    __device__ bool operator<(KernelPair& other) const { return first < other.first; }
    __device__ bool operator>(KernelPair& other) const { return first > other.first; }
};

__device__ inline void sleep_cuda_busy(float t, float mhz = 1410.0f)
{
    clock_t t0 = clock64();
    clock_t t1 = t0;
    while ((t1 - t0) / (mhz * 1000.0f) < t) {
        t1 = clock64();
    }
}

__device__ inline void sleep_cuda(float t)
{
    unsigned int ns = static_cast<unsigned int>(t * 1000000.0f);
    __nanosleep(ns);
}

__device__ inline void sleep_cuda_us(float t)
{
    unsigned int ns = static_cast<unsigned int>(t * 1000.0f);
    __nanosleep(ns);
}

__device__ __forceinline__ uint32_t linear_scan_prefetch(const uint32_t* d_prefetch_index,
                                                         uint32_t target,
                                                         uint32_t prefetch_len)
{
    for (uint32_t i = 0; i < prefetch_len; i++) {
        if (d_prefetch_index[i] == target) {
            return i;
        }
    }
    return UINT32_MAX;
}

__device__ __forceinline__ uint32_t binary_scan_prefetch(const uint32_t* d_prefetch_index,
                                                         uint32_t target,
                                                         uint32_t prefetch_len)
{
    int left = 0;
    int right = static_cast<int>(prefetch_len) - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        uint32_t val = __ldg(&d_prefetch_index[mid]);
        if (val == target) {
            return static_cast<uint32_t>(mid);
        }
        if (val < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return UINT32_MAX;
}

__device__ __forceinline__ uint32_t n_ary_scan_prefetch(const uint32_t* d_prefetch_index,
                                                        uint32_t target,
                                                        uint32_t prefetch_len)
{
    int len = static_cast<int>(prefetch_len);
    int base = 0;
    int range = len;
    int subtid = threadIdx.x % kWarpThreads;

    while (range > kWarpThreads) {
        int step_size = range / kWarpThreads;
        int idx = base + subtid * step_size;
        if (idx >= len) {
            idx = len - 1;
        }

        long long val = __ldg(&d_prefetch_index[idx]);
        unsigned int ballot = __ballot_sync(kFullMask, val <= target);
        if (ballot == 0) {
            range = step_size;
        } else {
            int last_le_lane = kWarpThreads - 1 - __clz(ballot);
            base = base + last_le_lane * step_size;
            range = (last_le_lane == (kWarpThreads - 1)) ? (range - (kWarpThreads - 1) * step_size)
                                                         : (step_size + 1);
        }
    }

    int final_idx = base + subtid;
    bool found = false;
    if (final_idx < len) {
        found = __ldg(&d_prefetch_index[final_idx]) == target;
    }

    unsigned int found_mask = __ballot_sync(kFullMask, found);
    if (found_mask) {
        int winner_lane = __ffs(found_mask) - 1;
        return static_cast<uint32_t>(__shfl_sync(kFullMask, final_idx, winner_lane));
    }
    return UINT32_MAX;
}

__device__ __forceinline__ uint64_t get_global_timer_ns()
{
    uint64_t ret;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(ret));
    return ret;
}

__device__ __forceinline__ uint32_t get_smid()
{
    uint32_t ret;
    asm volatile("mov.u32 %0, %smid;" : "=r"(ret));
    return ret;
}

__device__ __forceinline__ int64_t load_bypass_l1(volatile int64_t* ptr)
{
    int64_t ret;
    asm volatile("ld.global.cg.s64 %0, [%1];" : "=l"(ret) : "l"(ptr) : "memory");
    return ret;
}

__device__ inline void wait_for_data_ready(volatile int64_t* d_gdr_data, int bid)
{
    while (load_bypass_l1(&d_gdr_data[bid]) != 1) {
        __nanosleep(100000);
    }
}

__host__ __device__ inline int next_wait_sleep_us(int current_sleep_us,
                                                  int wait_loops,
                                                  int min_sleep_us = 2,
                                                  int max_sleep_us = 64)
{
    if (wait_loops == 0) {
        const int next = current_sleep_us / 2;
        return next < min_sleep_us ? min_sleep_us : next;
    }
    if (wait_loops == 1) {
        return current_sleep_us;
    }
    const int grown = current_sleep_us * 2;
    return grown > max_sleep_us ? max_sleep_us : grown;
}

template <class BackendDeviceState>
struct SearchKernelInputs {
    BackendDeviceState backend;
    uint32_t* d_neighbor_memory = nullptr;
    volatile int64_t* d_status_memory = nullptr;
    volatile int64_t* d_gdr_data = nullptr;
    volatile int64_t* d_status_flag = nullptr;
    const Node* d_prefetch_node_array = nullptr;
    const uint32_t* d_prefetch_index = nullptr;
    const uint32_t* d_prefetch_begin_by_label = nullptr;
    const uint32_t* d_prefetch_end_by_label = nullptr;
    const uint32_t* d_query_labels = nullptr;
    const uint32_t* d_start_local_ids = nullptr;
    const uint32_t* d_row_to_sample = nullptr;
    const uint32_t* d_label_offset = nullptr;
};

}
