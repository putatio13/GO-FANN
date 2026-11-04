#pragma once

#include <cstdint>

// Pure queue operations are shared by the CUDA kernel and host-side oracle tests.
#ifdef __CUDACC__
#define GO_FANN_SORTED_HD __host__ __device__ __forceinline__
#else
#define GO_FANN_SORTED_HD inline
#endif

namespace go_fann::backend_full_gpu::sorted_frontier {

constexpr uint32_t kUsed = 0x80000000u;
constexpr uint32_t kLocalMask = 0x07ffffffu;
constexpr uint32_t kKeyMask = 0x7fffffffu;
constexpr uint32_t kMaxRoutes = 16;

struct Entry {
    float distance;
    uint32_t payload;
};
static_assert(sizeof(Entry) == 8, "Sorted frontier entries must remain eight bytes");

GO_FANN_SORTED_HD uint32_t pack(uint32_t route, uint32_t local)
{
    return (route << 27u) | local;
}
GO_FANN_SORTED_HD uint32_t route(Entry entry) { return (entry.payload >> 27u) & 15u; }
GO_FANN_SORTED_HD uint32_t local(Entry entry) { return entry.payload & kLocalMask; }
GO_FANN_SORTED_HD uint32_t key(Entry entry) { return entry.payload & kKeyMask; }
GO_FANN_SORTED_HD bool valid(Entry entry) { return entry.distance == entry.distance; }

// NaNs represent temporary invalid batch slots only; no NaN enters the queue.
// Infinities and the entire 31-bit identity domain are valid queue entries.
GO_FANN_SORTED_HD bool less(Entry a, Entry b)
{
    if (!valid(a)) return false;
    if (!valid(b)) return true;
    return a.distance < b.distance ||
        (a.distance == b.distance && key(a) < key(b));
}
GO_FANN_SORTED_HD int lower_bound(const Entry* entries, int count, Entry needle)
{
    int first = 0;
    while (first < count) {
        const int mid = first + (count - first) / 2;
        if (less(entries[mid], needle)) first = mid + 1;
        else count = mid;
    }
    return first;
}
GO_FANN_SORTED_HD int upper_bound(const Entry* entries, int count, Entry needle)
{
    int first = 0;
    while (first < count) {
        const int mid = first + (count - first) / 2;
        if (!less(needle, entries[mid])) first = mid + 1;
        else count = mid;
    }
    return first;
}
GO_FANN_SORTED_HD int sort_serial(Entry* entries, int count)
{
    int size = 0;
    for (int i = 0; i < count; ++i) {
        const Entry entry = entries[i];
        if (!valid(entry)) continue;
        int pos = size;
        while (pos != 0 && less(entry, entries[pos - 1])) {
            entries[pos] = entries[pos - 1];
            --pos;
        }
        entries[pos] = entry;
        ++size;
    }
    return size;
}
// Inputs must be sorted and identity-unique across both sides. Discovery
// visited admission establishes that precondition in the search kernel.
GO_FANN_SORTED_HD int merge_serial(const Entry* old_entries, int old_count,
                                     const Entry* batch, int batch_count,
                                     Entry* destination, int capacity)
{
    int i = 0, j = 0, size = 0;
    while (size < capacity && (i < old_count || j < batch_count)) {
        const bool take_old = i < old_count &&
            (j == batch_count || !less(batch[j], old_entries[i]));
        destination[size++] = take_old ? old_entries[i++] : batch[j++];
    }
    return size;
}

#ifdef __CUDACC__
// Every thread in a 128-thread CTA participates. For a small batch only warp0
// performs the sorting network, while the final barrier publishes its output.
__device__ __forceinline__ void sort_parallel(Entry* batch, uint32_t width)
{
    const uint32_t tid = threadIdx.x;
    if (width == 32) {
        if (tid < 32) {
            Entry value = batch[tid];
            for (uint32_t size = 2; size <= 32; size <<= 1) {
                for (uint32_t stride = size >> 1; stride; stride >>= 1) {
                    Entry other{__shfl_xor_sync(0xffffffffu, value.distance, stride),
                                __shfl_xor_sync(0xffffffffu, value.payload, stride)};
                    const bool take_min = ((tid & size) == 0) == ((tid & stride) == 0);
                    if (take_min ? less(other, value) : less(value, other)) value = other;
                }
            }
            batch[tid] = value;
        }
        __syncthreads();
        return;
    }
    for (uint32_t size = 2; size <= width; size <<= 1) {
        for (uint32_t stride = size >> 1; stride; stride >>= 1) {
            if (tid < width && (tid ^ stride) > tid) {
                const Entry a = batch[tid], b = batch[tid ^ stride];
                if ((tid & size) == 0 ? less(b, a) : less(a, b)) {
                    batch[tid] = b;
                    batch[tid ^ stride] = a;
                }
            }
            __syncthreads();
        }
    }
}

__device__ __forceinline__ void merge_parallel(const Entry* old_entries, int old_count,
                                               const Entry* batch, int batch_count,
                                               Entry* destination, int capacity)
{
    for (int i = static_cast<int>(threadIdx.x); i < old_count; i += blockDim.x) {
        const Entry value = old_entries[i];
        const int rank = i + lower_bound(batch, batch_count, value);
        if (rank < capacity) destination[rank] = value;
    }
    for (int j = static_cast<int>(threadIdx.x); j < batch_count; j += blockDim.x) {
        const Entry value = batch[j];
        const int rank = j + upper_bound(old_entries, old_count, value);
        if (rank < capacity) destination[rank] = value;
    }
    __syncthreads();
}
#endif

}  // namespace go_fann::backend_full_gpu::sorted_frontier

#undef GO_FANN_SORTED_HD
