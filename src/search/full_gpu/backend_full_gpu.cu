#include "search/full_gpu/backend_full_gpu.cuh"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <numeric>
#include <nlohmann/json.hpp>
#include "index/graph_file_format.h"
#include <limits>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <unordered_map>

#include <immintrin.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <gdrapi.h>
#include <gdrconfig.h>

#include "fixhash.h"
#include "smmh2.h"
#include "predicates/range_filter.cuh"
#include "index/range_posting_index.h"
#include "search/common/warp_kernel.cuh"
#include "search/full_gpu/full_gpu_sorted_frontier.cuh"

namespace go_fann::backend_full_gpu {

namespace {

std::vector<uint32_t> materialize_range_posting_ids(const GraphRuntime& runtime);

constexpr uint32_t kMaxTopk = defaults::max_topk;
constexpr uint32_t kMaxSearchQueue = 512;
constexpr uint32_t kMaxGraphDegree = defaults::max_degree;
constexpr uint32_t kHighfreqBlockThreads = 128;
constexpr int kHashTableCapacity = 4096;
#ifndef GO_FANN_FULL_GPU_COOPERATIVE_PICKUP
#define GO_FANN_FULL_GPU_COOPERATIVE_PICKUP 1
#endif
#ifndef GO_FANN_FULL_GPU_UINT8_DP4A
#define GO_FANN_FULL_GPU_UINT8_DP4A 1
#endif
constexpr bool kFullGpuCooperativePickup = GO_FANN_FULL_GPU_COOPERATIVE_PICKUP != 0;
constexpr bool kFullGpuUint8Dp4a = GO_FANN_FULL_GPU_UINT8_DP4A != 0;
static_assert(GO_FANN_FULL_GPU_COOPERATIVE_PICKUP == 0 ||
                  GO_FANN_FULL_GPU_COOPERATIVE_PICKUP == 1,
              "GO_FANN_FULL_GPU_COOPERATIVE_PICKUP must be 0 or 1");
static_assert(GO_FANN_FULL_GPU_UINT8_DP4A == 0 || GO_FANN_FULL_GPU_UINT8_DP4A == 1,
              "GO_FANN_FULL_GPU_UINT8_DP4A must be 0 or 1");
constexpr uint32_t kPoolUsedMask = 0x80000000u;
constexpr uint64_t kVisitedHashMul = 0x4bcb391f924ed183ULL;
constexpr uint64_t kVisitedHashXor = 0xa0ab69ccd854fc0aULL;
constexpr uint32_t kVisitedEmpty = UINT32_MAX;
constexpr uint32_t kPackedVisitedEmpty = 0x00ffffffu;
constexpr uint8_t kQueryFilterModeBitOnlyNoRange = 0;
constexpr uint8_t kQueryFilterModeSingleExtraNoRange = 1;
constexpr uint8_t kQueryFilterModeGenericNoRange = 2;
constexpr uint8_t kQueryFilterModeBitOnlyWithRange = 3;
constexpr uint8_t kQueryFilterModeSingleExtraWithRange = 4;
constexpr uint8_t kQueryFilterModeGenericWithRange = 5;
constexpr size_t kGdrAlignBytes = 65536;

[[noreturn]] void fail_full_gpu(const std::string& message)
{
    throw std::runtime_error("full_gpu backend failed: " + message);
}

enum SearchEarlyStopMode : int {
    kSearchEarlyStopFixed = 0,
    kSearchEarlyStopOff = 1,
    kSearchEarlyStopAdaptive = 2,
};

// Route expansion always evaluates the complete stored adjacency row.
// Future overlap search may add graph contexts, but does not cap degree here.

int parse_search_early_stop_mode(const std::string& mode)
{
    if (mode == defaults::search_early_stop_mode_fixed) {
        return kSearchEarlyStopFixed;
    }
    if (mode == defaults::search_early_stop_mode_off) {
        return kSearchEarlyStopOff;
    }
    if (mode == defaults::search_early_stop_mode_adaptive) {
        return kSearchEarlyStopAdaptive;
    }
    fail_full_gpu("unsupported search_early_stop_mode: " + mode);
    return kSearchEarlyStopFixed;
}

__device__ __forceinline__ int adaptive_early_stop_limit(int queue_limit,
                                                         int expanded_count,
                                                         int pass_count,
                                                         int stop_width)
{
    constexpr int kPriorPass = 8;
    constexpr int kPriorTotal = 16;
    constexpr int kBaseNumerator = 1;
    constexpr int kBaseDenominator = 2;
    constexpr int kMinBaseTopkMultiplier = 2;
    constexpr int kExtraNumerator = 1;
    constexpr int kExtraDenominator = 2;
    constexpr int kMaxExtraPatience = 32;
    const int queue_base = queue_limit > 1 ? queue_limit : 1;
    const int base_numerator = queue_base * kBaseNumerator;
    int base_limit = (base_numerator + kBaseDenominator - 1) / kBaseDenominator;
    const int min_base_limit = stop_width > 0 ? stop_width * kMinBaseTopkMultiplier : 1;
    base_limit = base_limit > min_base_limit ? base_limit : min_base_limit;
    const int denom = (expanded_count + kPriorTotal) > 1 ? (expanded_count + kPriorTotal) : 1;
    const int pass_with_prior = pass_count + kPriorPass;
    const int pass_estimate = pass_with_prior < denom ? pass_with_prior : denom;
    const int reject_estimate = (denom - pass_estimate) > 0 ? (denom - pass_estimate) : 0;
    const int64_t numerator = static_cast<int64_t>(queue_limit) *
                              kExtraNumerator *
                              reject_estimate;
    int extra = static_cast<int>((numerator + (kExtraDenominator * denom - 1)) /
                                 (kExtraDenominator * denom));
    extra = extra > 0 ? extra : 0;
    extra = extra < kMaxExtraPatience ? extra : kMaxExtraPatience;
    return base_limit + extra;
}

__device__ __forceinline__ bool early_stop_reached(int early_stop_mode,
                                                   int bad_count,
                                                   int fixed_finish_limit,
                                                   int queue_limit,
                                                   int expanded_count,
                                                   int pass_count,
                                                   int stop_width)
{
    if (early_stop_mode == kSearchEarlyStopOff) {
        return false;
    }
    if (early_stop_mode == kSearchEarlyStopAdaptive) {
        return bad_count >= adaptive_early_stop_limit(queue_limit,
                                                      expanded_count,
                                                      pass_count,
                                                      stop_width);
    }
    return bad_count >= fixed_finish_limit;
}

__device__ __forceinline__ uint32_t load_bypass_l1_u32(const volatile uint32_t* ptr)
{
    uint32_t ret;
    asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(ret) : "l"(ptr) : "memory");
    return ret;
}

void check_cuda(cudaError_t status, const char* context)
{
    if (status != cudaSuccess) {
        fail_full_gpu(std::string(context) + ": " + cudaGetErrorString(status));
    }
}

void check_gdr(int status, const char* context)
{
    if (status != 0) {
        fail_full_gpu(std::string(context) + " failed with code " + std::to_string(status));
    }
}

size_t round_up_gdr_bytes(size_t value)
{
    return ((value + kGdrAlignBytes - 1) / kGdrAlignBytes) * kGdrAlignBytes;
}

template <typename T>
T* align_gdr_pointer(T* ptr)
{
    const uintptr_t raw = reinterpret_cast<uintptr_t>(ptr);
    const uintptr_t aligned =
        (raw + kGdrAlignBytes - 1) & ~(static_cast<uintptr_t>(kGdrAlignBytes) - 1);
    return reinterpret_cast<T*>(aligned);
}

template <typename LaunchFunc>
void dispatch_full_gpu_kernel_by_queue_size(int queue_size, LaunchFunc&& launch)
{
    if (queue_size <= 16) {
        launch.template operator()<16>();
    } else if (queue_size <= 32) {
        launch.template operator()<32>();
    } else if (queue_size <= 48) {
        launch.template operator()<48>();
    } else if (queue_size <= 64) {
        launch.template operator()<64>();
    } else if (queue_size <= 96) {
        launch.template operator()<96>();
    } else if (queue_size <= 128) {
        launch.template operator()<128>();
    } else if (queue_size <= 192) {
        launch.template operator()<192>();
    } else if (queue_size <= 256) {
        launch.template operator()<256>();
    } else if (queue_size <= 384) {
        launch.template operator()<384>();
    } else if (queue_size <= 512) {
        launch.template operator()<512>();
    } else {
        fail_full_gpu("unsupported queue_size (must be <= 512)");
    }
}

template <typename KernelFunc>
static uint32_t resident_worker_capacity_for_latency_kernel(KernelFunc kernel,
                                                            int thread_in_block,
                                                            size_t shared_bytes,
                                                            const char* occupancy_context,
                                                            const char* device_context)
{
    int blocks_per_sm = 0;
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm,
                                                             kernel,
                                                             thread_in_block,
                                                             shared_bytes),
               occupancy_context);
    cudaDeviceProp prop{};
    int device_id = 0;
    check_cuda(cudaGetDevice(&device_id), device_context);
    check_cuda(cudaGetDeviceProperties(&prop, device_id), device_context);
    return static_cast<uint32_t>(std::max(0, blocks_per_sm)) *
           static_cast<uint32_t>(prop.multiProcessorCount);
}

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    ~DeviceBuffer()
    {
        reset();
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* data()
    {
        return ptr_;
    }

    const T* data() const
    {
        return ptr_;
    }

    size_t size() const
    {
        return size_;
    }

    void reset()
    {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
        }
        ptr_ = nullptr;
        size_ = 0;
    }

    void allocate(size_t count, const char* context)
    {
        reset();
        if (count == 0) {
            return;
        }
        check_cuda(cudaMalloc(&ptr_, count * sizeof(T)), context);
        size_ = count;
    }

    void upload(const std::vector<T>& values, const char* context)
    {
        allocate(values.size(), context);
        if (!values.empty()) {
            check_cuda(cudaMemcpy(ptr_,
                                  values.data(),
                                  values.size() * sizeof(T),
                                  cudaMemcpyHostToDevice),
                       context);
        }
    }

    void copy_from_host(const T* values, size_t count, const char* context)
    {
        copy_from_host_at(values, count, 0, context);
    }

    void copy_from_host_at(const T* values, size_t count, size_t offset, const char* context)
    {
        if (offset > size_ || count > size_ - offset) {
            fail_full_gpu(std::string(context) + " exceeds allocated device buffer capacity");
        }
        if (count != 0) {
            check_cuda(cudaMemcpy(ptr_ + offset, values, count * sizeof(T), cudaMemcpyHostToDevice), context);
        }
    }

    void copy_from_host_async(const T* values, size_t count, cudaStream_t stream, const char* context)
    {
        copy_from_host_async_at(values, count, 0, stream, context);
    }

    void copy_from_host_async_at(const T* values,
                                 size_t count,
                                 size_t offset,
                                 cudaStream_t stream,
                                 const char* context)
    {
        if (offset > size_ || count > size_ - offset) {
            fail_full_gpu(std::string(context) + " exceeds allocated device buffer capacity");
        }
        if (count != 0) {
            check_cuda(cudaMemcpyAsync(ptr_ + offset,
                                       values,
                                       count * sizeof(T),
                                       cudaMemcpyHostToDevice,
                                       stream),
                       context);
        }
    }

    void copy_to_host(T* values, size_t count, const char* context) const
    {
        copy_to_host_at(values, count, 0, context);
    }

    void copy_to_host_at(T* values, size_t count, size_t offset, const char* context) const
    {
        if (offset > size_ || count > size_ - offset) {
            fail_full_gpu(std::string(context) + " exceeds allocated device buffer capacity");
        }
        if (count != 0) {
            check_cuda(cudaMemcpy(values, ptr_ + offset, count * sizeof(T), cudaMemcpyDeviceToHost), context);
        }
    }

    void copy_to_host_async(T* values, size_t count, cudaStream_t stream, const char* context) const
    {
        copy_to_host_async_at(values, count, 0, stream, context);
    }

    void copy_to_host_async_at(T* values,
                               size_t count,
                               size_t offset,
                               cudaStream_t stream,
                               const char* context) const
    {
        if (offset > size_ || count > size_ - offset) {
            fail_full_gpu(std::string(context) + " exceeds allocated device buffer capacity");
        }
        if (count != 0) {
            check_cuda(cudaMemcpyAsync(values,
                                       ptr_ + offset,
                                       count * sizeof(T),
                                       cudaMemcpyDeviceToHost,
                                       stream),
                       context);
        }
    }

    std::vector<T> download(const char* context) const
    {
        std::vector<T> values(size_);
        if (size_ != 0) {
            check_cuda(cudaMemcpy(values.data(),
                                  ptr_,
                                  size_ * sizeof(T),
                                  cudaMemcpyDeviceToHost),
                       context);
        }
        return values;
    }

private:
    T* ptr_ = nullptr;
    size_t size_ = 0;
};

template <typename T>
class PinnedHostBuffer {
public:
    PinnedHostBuffer() = default;
    ~PinnedHostBuffer()
    {
        reset();
    }

    PinnedHostBuffer(const PinnedHostBuffer&) = delete;
    PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;

    T* data()
    {
        return ptr_;
    }

    const T* data() const
    {
        return ptr_;
    }

    size_t size() const
    {
        return size_;
    }

    void reset()
    {
        if (ptr_ != nullptr) {
            cudaFreeHost(ptr_);
        }
        ptr_ = nullptr;
        size_ = 0;
    }

    void allocate(size_t count, const char* context)
    {
        if (count == size_) {
            return;
        }
        reset();
        if (count == 0) {
            return;
        }
        check_cuda(cudaMallocHost(&ptr_, count * sizeof(T)), context);
        size_ = count;
    }

private:
    T* ptr_ = nullptr;
    size_t size_ = 0;
};

template <typename T>
class MappedPinnedHostBuffer {
public:
    MappedPinnedHostBuffer() = default;
    ~MappedPinnedHostBuffer()
    {
        reset();
    }

    MappedPinnedHostBuffer(const MappedPinnedHostBuffer&) = delete;
    MappedPinnedHostBuffer& operator=(const MappedPinnedHostBuffer&) = delete;
    MappedPinnedHostBuffer(MappedPinnedHostBuffer&& other) noexcept
        : host_ptr_(other.host_ptr_), device_ptr_(other.device_ptr_), size_(other.size_)
    {
        other.host_ptr_ = nullptr;
        other.device_ptr_ = nullptr;
        other.size_ = 0;
    }
    MappedPinnedHostBuffer& operator=(MappedPinnedHostBuffer&& other) noexcept
    {
        if (this != &other) {
            reset();
            host_ptr_ = other.host_ptr_;
            device_ptr_ = other.device_ptr_;
            size_ = other.size_;
            other.host_ptr_ = nullptr;
            other.device_ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    T* host_data()
    {
        return host_ptr_;
    }

    const T* host_data() const
    {
        return host_ptr_;
    }

    T* device_data()
    {
        return device_ptr_;
    }

    size_t size() const
    {
        return size_;
    }

    void reset()
    {
        if (host_ptr_ != nullptr) {
            cudaFreeHost(host_ptr_);
        }
        host_ptr_ = nullptr;
        device_ptr_ = nullptr;
        size_ = 0;
    }

    void allocate(size_t count, const char* context)
    {
        reset();
        if (count == 0) {
            return;
        }
        check_cuda(cudaHostAlloc(&host_ptr_, count * sizeof(T), cudaHostAllocMapped), context);
        check_cuda(cudaHostGetDevicePointer(&device_ptr_, host_ptr_, 0),
                   "cudaHostGetDevicePointer full_gpu mapped pinned buffer");
        std::fill(host_ptr_, host_ptr_ + count, T{});
        size_ = count;
    }

private:
    T* host_ptr_ = nullptr;
    T* device_ptr_ = nullptr;
    size_t size_ = 0;
};

struct GdrMappingHandle {
    gdr_t g = nullptr;
    gdr_mh_t mh{};
    void* cpu_ptr = nullptr;
    bool pinned = false;
};

template <typename T>
class GdrDeviceBuffer {
public:
    GdrDeviceBuffer() = default;
    ~GdrDeviceBuffer()
    {
        reset();
    }

    GdrDeviceBuffer(const GdrDeviceBuffer&) = delete;
    GdrDeviceBuffer& operator=(const GdrDeviceBuffer&) = delete;
    GdrDeviceBuffer(GdrDeviceBuffer&& other) noexcept
        : raw_ptr_(other.raw_ptr_),
          device_ptr_(other.device_ptr_),
          mapped_ptr_(other.mapped_ptr_),
          mapped_bytes_(other.mapped_bytes_),
          size_(other.size_),
          mapping_(other.mapping_)
    {
        other.raw_ptr_ = nullptr;
        other.device_ptr_ = nullptr;
        other.mapped_ptr_ = nullptr;
        other.mapped_bytes_ = 0;
        other.size_ = 0;
        other.mapping_ = {};
    }
    GdrDeviceBuffer& operator=(GdrDeviceBuffer&& other) noexcept
    {
        if (this != &other) {
            reset();
            raw_ptr_ = other.raw_ptr_;
            device_ptr_ = other.device_ptr_;
            mapped_ptr_ = other.mapped_ptr_;
            mapped_bytes_ = other.mapped_bytes_;
            size_ = other.size_;
            mapping_ = other.mapping_;
            other.raw_ptr_ = nullptr;
            other.device_ptr_ = nullptr;
            other.mapped_ptr_ = nullptr;
            other.mapped_bytes_ = 0;
            other.size_ = 0;
            other.mapping_ = {};
        }
        return *this;
    }

    T* device_data()
    {
        return device_ptr_;
    }

    const T* device_data() const
    {
        return device_ptr_;
    }

    T* host_data()
    {
        return mapped_ptr_;
    }

    const T* host_data() const
    {
        return mapped_ptr_;
    }

    size_t size() const
    {
        return size_;
    }

    void reset()
    {
        if (mapping_.g != nullptr && mapping_.cpu_ptr != nullptr) {
            gdr_unmap(mapping_.g, mapping_.mh, mapping_.cpu_ptr, mapped_bytes_);
        }
        if (mapping_.g != nullptr && mapping_.pinned) {
            gdr_unpin_buffer(mapping_.g, mapping_.mh);
        }
        if (raw_ptr_ != nullptr) {
            cudaFree(raw_ptr_);
        }
        raw_ptr_ = nullptr;
        device_ptr_ = nullptr;
        mapped_ptr_ = nullptr;
        mapped_bytes_ = 0;
        size_ = 0;
        mapping_ = {};
    }

    void allocate(gdr_t g, size_t count, const char* context)
    {
        reset();
        if (count == 0) {
            return;
        }
        if (g == nullptr) {
            fail_full_gpu(std::string(context) + " requires an open GDR handle");
        }
        const size_t bytes = count * sizeof(T);
        mapped_bytes_ = round_up_gdr_bytes(bytes);
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&raw_ptr_), mapped_bytes_ + kGdrAlignBytes),
                   context);
        device_ptr_ = align_gdr_pointer(raw_ptr_);
        mapping_.g = g;
        check_gdr(gdr_pin_buffer(mapping_.g,
                                 reinterpret_cast<unsigned long>(device_ptr_),
                                 mapped_bytes_,
                                 0,
                                 0,
                                 &mapping_.mh),
                  context);
        mapping_.pinned = true;
        check_gdr(gdr_map(mapping_.g, mapping_.mh, &mapping_.cpu_ptr, mapped_bytes_), context);
        mapped_ptr_ = reinterpret_cast<T*>(mapping_.cpu_ptr);
        std::fill_n(mapped_ptr_, count, T{});
        _mm_sfence();
        size_ = count;
    }

    void copy_from_host_at(const T* values, size_t count, size_t offset, const char* context)
    {
        if (offset > size_ || count > size_ - offset) {
            fail_full_gpu(std::string(context) + " exceeds allocated GDR buffer capacity");
        }
        if (count == 0) {
            return;
        }
        std::memcpy(mapped_ptr_ + offset, values, count * sizeof(T));
    }

    void store_at(size_t offset, const T& value, const char* context)
    {
        if (offset >= size_) {
            fail_full_gpu(std::string(context) + " exceeds allocated GDR buffer capacity");
        }
        mapped_ptr_[offset] = value;
    }

private:
    T* raw_ptr_ = nullptr;
    T* device_ptr_ = nullptr;
    T* mapped_ptr_ = nullptr;
    size_t mapped_bytes_ = 0;
    size_t size_ = 0;
    GdrMappingHandle mapping_{};
};

struct UInt32Csr {
    std::vector<uint32_t> indptr;
    std::vector<uint32_t> indices;
};

struct QueryFilterFastPath {
    std::vector<uint8_t> modes;
    std::vector<uint32_t> single_extra_labels;
    std::vector<uint32_t> active_range_counts;
};

struct FullGpuQueryDeviceView {
    const uint32_t* query_vector_ids = nullptr;
    const uint32_t* query_group_ids = nullptr;
    const uint32_t* start_local_ids = nullptr;
    const uint64_t* query_filter_bits = nullptr;
    const uint8_t* query_filter_modes = nullptr;
    const uint32_t* query_single_extra_labels = nullptr;
    const uint32_t* query_active_range_counts = nullptr;
    const uint32_t* query_label_indptr = nullptr;
    const uint32_t* query_label_indices = nullptr;
    const uint8_t* query_range_modes = nullptr;
    const float* query_range_lower_bounds = nullptr;
    const float* query_range_upper_bounds = nullptr;
};

struct alignas(64) PersistentLatencyRequestControl {
    uint32_t ready_seq = 0;
    uint32_t stop = 0;
    uint32_t route_count = 0;
    uint32_t high_route_count = 0;
    uint32_t low_route_count = 0;
    uint32_t reserved0 = 0;
    uint32_t reserved1 = 0;
    uint32_t reserved2 = 0;
};

struct alignas(64) PersistentLatencyCompletionControl {
    uint32_t done_seq = 0;
    uint32_t reserved0 = 0;
    uint32_t reserved1 = 0;
    uint32_t reserved2 = 0;
    uint64_t gpu_start_clock = 0;
    uint64_t gpu_stop_clock = 0;
};

UInt32Csr build_sorted_label_csr(const SparseLabelMatrix& matrix)
{
    UInt32Csr csr;
    csr.indptr.resize(static_cast<size_t>(matrix.rows) + 1, 0);
    if (matrix.indptr.size() == static_cast<size_t>(matrix.rows) + 1) {
        if (matrix.indices.size() > std::numeric_limits<uint32_t>::max()) {
            fail_full_gpu("label CSR nnz exceeds uint32 range");
        }
        csr.indices = matrix.indices;
        for (uint32_t row = 0; row < matrix.rows; ++row) {
            const SparseLabelRowSpan span = sparse_label_row_span(matrix, row);
            if (!span.valid) {
                fail_full_gpu("sparse label CSR is not aligned with row count");
            }
            const uint64_t begin64 = matrix.indptr[row];
            const uint64_t end64 = matrix.indptr[static_cast<size_t>(row) + 1];
            if (begin64 > std::numeric_limits<uint32_t>::max() ||
                end64 > std::numeric_limits<uint32_t>::max()) {
                fail_full_gpu("label CSR row pointer exceeds uint32 range");
            }
            const uint32_t begin = static_cast<uint32_t>(begin64);
            const uint32_t end = static_cast<uint32_t>(end64);
            csr.indptr[row] = begin;
            if (begin < end) {
                auto row_begin = csr.indices.begin() + begin;
                auto row_end = csr.indices.begin() + end;
                if (!std::is_sorted(row_begin, row_end)) {
                    std::sort(row_begin, row_end);
                }
            }
        }
        csr.indptr[matrix.rows] = static_cast<uint32_t>(matrix.indptr[matrix.rows]);
        return csr;
    }

    for (uint32_t row = 0; row < matrix.rows; ++row) {
        if (row >= matrix.row_labels.size()) {
            fail_full_gpu("sparse label row_labels are not aligned with row count");
        }
        std::vector<uint32_t> labels = matrix.row_labels[row];
        std::sort(labels.begin(), labels.end());
        labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
        if (csr.indices.size() + labels.size() > std::numeric_limits<uint32_t>::max()) {
            fail_full_gpu("label CSR nnz exceeds uint32 range");
        }
        csr.indices.insert(csr.indices.end(), labels.begin(), labels.end());
        csr.indptr[static_cast<size_t>(row) + 1] = static_cast<uint32_t>(csr.indices.size());
    }
    return csr;
}

QueryFilterFastPath build_query_filter_fast_path(const UInt32Csr& csr,
                                                 const QueryRangeTable& query_ranges)
{
    if (csr.indptr.empty()) {
        return {};
    }

    const size_t row_count = csr.indptr.size() - 1;
    if (query_ranges.rows != row_count) {
        fail_full_gpu("query range rows are not aligned with routed query labels");
    }
    const size_t range_elem_count = row_count * query_ranges.range_count;
    if (query_ranges.modes.size() != range_elem_count ||
        query_ranges.lower_bounds.size() != range_elem_count ||
        query_ranges.upper_bounds.size() != range_elem_count) {
        fail_full_gpu("query range buffers are not aligned with routed query labels");
    }

    QueryFilterFastPath fast_path;
    fast_path.modes.resize(row_count, kQueryFilterModeBitOnlyNoRange);
    fast_path.single_extra_labels.resize(row_count, UINT32_MAX);
    fast_path.active_range_counts.resize(row_count, 0);
    for (size_t row = 0; row < row_count; ++row) {
        const uint32_t begin = csr.indptr[row];
        const uint32_t end = csr.indptr[row + 1];
        const uint32_t label_count = end - begin;
        const uint32_t active_range_count =
            range_filter::count_active_ranges(query_ranges.modes.data(),
                                              static_cast<uint32_t>(row),
                                              query_ranges.range_count);
        fast_path.active_range_counts[row] = active_range_count;
        const bool has_range = active_range_count != 0;

        if (label_count == 0) {
            fast_path.modes[row] = has_range ? kQueryFilterModeBitOnlyWithRange
                                             : kQueryFilterModeBitOnlyNoRange;
        } else if (label_count == 1) {
            fast_path.modes[row] = has_range ? kQueryFilterModeSingleExtraWithRange
                                             : kQueryFilterModeSingleExtraNoRange;
            fast_path.single_extra_labels[row] = csr.indices[begin];
        } else {
            fast_path.modes[row] = has_range ? kQueryFilterModeGenericWithRange
                                             : kQueryFilterModeGenericNoRange;
        }
    }
    return fast_path;
}

UInt32Csr build_sorted_query_label_csr_without_primary(const SparseLabelMatrix& matrix,
                                                       const std::vector<uint32_t>& primary_labels)
{
    if (primary_labels.size() < matrix.rows) {
        fail_full_gpu("query_filter_labels are not aligned with query label rows");
    }

    UInt32Csr csr;
    csr.indptr.resize(static_cast<size_t>(matrix.rows) + 1, 0);
    for (uint32_t row = 0; row < matrix.rows; ++row) {
        if (row >= matrix.row_labels.size()) {
            fail_full_gpu("sparse query label row_labels are not aligned with row count");
        }
        std::vector<uint32_t> labels = matrix.row_labels[row];
        std::sort(labels.begin(), labels.end());
        labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
        labels.erase(std::remove(labels.begin(), labels.end(), primary_labels[row]), labels.end());
        if (csr.indices.size() + labels.size() > std::numeric_limits<uint32_t>::max()) {
            fail_full_gpu("query label CSR nnz exceeds uint32 range");
        }
        csr.indices.insert(csr.indices.end(), labels.begin(), labels.end());
        csr.indptr[static_cast<size_t>(row) + 1] = static_cast<uint32_t>(csr.indices.size());
    }
    return csr;
}

#include "search/full_gpu/full_gpu_overlap_types.cuh"

struct FullGpuDeviceRuntime {
    std::unique_ptr<FullGpuOverlapDevice> overlap;
    DeviceBuffer<uint8_t> base_u8;
    DeviceBuffer<float> base_f32;
    DeviceBuffer<__half> base_f16;
    DeviceBuffer<uint8_t> query_u8;
    DeviceBuffer<float> query_f32;
    DeviceBuffer<__half> query_f16;
    DeviceBuffer<uint32_t> graph;
    DeviceBuffer<uint32_t> label_size;
    DeviceBuffer<uint32_t> label_offset;
    DeviceBuffer<uint32_t> row_to_sample;
    DeviceBuffer<uint64_t> row_filter_mask;
    DeviceBuffer<uint32_t> query_group_ids;
    DeviceBuffer<uint32_t> start_local_ids;
    DeviceBuffer<uint64_t> query_filter_bits;
    DeviceBuffer<uint8_t> query_filter_modes;
    DeviceBuffer<uint32_t> query_single_extra_labels;
    DeviceBuffer<uint32_t> query_active_range_counts;
    DeviceBuffer<uint32_t> query_label_indptr;
    DeviceBuffer<uint32_t> query_label_indices;
    DeviceBuffer<uint32_t> sample_label_indptr;
    DeviceBuffer<uint32_t> sample_label_indices;
    DeviceBuffer<uint8_t> query_range_modes;
    DeviceBuffer<float> query_range_lower_bounds;
    DeviceBuffer<float> query_range_upper_bounds;
    DeviceBuffer<float> base_range_values;
    // Range-posting payloads are resident on the GPU for full-GPU execution.
    DeviceBuffer<uint32_t> range_posting_sample_ids;
    DeviceBuffer<uint64_t> route_range_posting_begins;
    DeviceBuffer<uint64_t> route_range_posting_sizes;
    DeviceBuffer<uint32_t> range_posting_route_ids;
    DeviceBuffer<uint32_t> high_query_indices;
    DeviceBuffer<uint32_t> low_query_indices;
    DeviceBuffer<uint32_t> low_label_size;
    DeviceBuffer<uint32_t> low_label_offset;
    DeviceBuffer<uint32_t> low_sample_ids;
    DeviceBuffer<uint64_t> low_row_filter_mask;
    DeviceBuffer<idx_t> route_results;
    DeviceBuffer<float> route_distances;
    DeviceBuffer<int> route_steps;
    DeviceBuffer<uint32_t> latency_route_indices;
    DeviceBuffer<uint32_t> latency_high_query_indices;
    DeviceBuffer<uint32_t> latency_low_query_indices;
    DeviceBuffer<uint32_t> latency_query_vector_ids;
    DeviceBuffer<uint32_t> latency_query_group_ids;
    DeviceBuffer<uint32_t> latency_start_local_ids;
    DeviceBuffer<uint64_t> latency_query_filter_bits;
    DeviceBuffer<uint8_t> latency_query_filter_modes;
    DeviceBuffer<uint32_t> latency_query_single_extra_labels;
    DeviceBuffer<uint32_t> latency_query_active_range_counts;
    DeviceBuffer<uint32_t> latency_query_label_indptr;
    DeviceBuffer<uint32_t> latency_query_label_indices;
    DeviceBuffer<uint8_t> latency_query_range_modes;
    DeviceBuffer<float> latency_query_range_lower_bounds;
    DeviceBuffer<float> latency_query_range_upper_bounds;
    DeviceBuffer<idx_t> latency_results;
    DeviceBuffer<float> latency_distances;
};

struct PersistentLatencyDeviceRuntime {
    GdrDeviceBuffer<PersistentLatencyRequestControl> requests;
    MappedPinnedHostBuffer<PersistentLatencyCompletionControl> completions;
    GdrDeviceBuffer<uint32_t> route_indices;
    GdrDeviceBuffer<uint32_t> high_query_indices;
    GdrDeviceBuffer<uint32_t> low_query_indices;
    GdrDeviceBuffer<uint32_t> query_vector_ids;
    GdrDeviceBuffer<uint32_t> query_group_ids;
    GdrDeviceBuffer<uint32_t> start_local_ids;
    GdrDeviceBuffer<uint64_t> query_filter_bits;
    GdrDeviceBuffer<uint8_t> query_filter_modes;
    GdrDeviceBuffer<uint32_t> query_single_extra_labels;
    GdrDeviceBuffer<uint32_t> query_active_range_counts;
    GdrDeviceBuffer<uint32_t> query_label_indptr;
    GdrDeviceBuffer<uint32_t> query_label_indices;
    GdrDeviceBuffer<uint8_t> query_range_modes;
    GdrDeviceBuffer<float> query_range_lower_bounds;
    GdrDeviceBuffer<float> query_range_upper_bounds;
    MappedPinnedHostBuffer<idx_t> results;
    MappedPinnedHostBuffer<float> distances;
    uint32_t route_capacity = 0;
    uint32_t slot_route_capacity = 0;
    uint32_t label_nnz_capacity_per_route = 0;
    uint32_t range_count = 0;
    uint32_t topk = 0;
};

__device__ __forceinline__ bool sorted_labels_contain(const uint32_t* labels,
                                                      uint32_t begin,
                                                      uint32_t end,
                                                      uint32_t needle)
{
    while (begin < end) {
        const uint32_t mid = begin + ((end - begin) >> 1);
        const uint32_t value = labels[mid];
        if (value == needle) {
            return true;
        }
        if (value < needle) {
            begin = mid + 1;
        } else {
            end = mid;
        }
    }
    return false;
}

__device__ __forceinline__ bool labels_match_device(const uint32_t* query_label_indptr,
                                                    const uint32_t* query_label_indices,
                                                    const uint32_t* sample_label_indptr,
                                                    const uint32_t* sample_label_indices,
                                                    uint32_t query_idx,
                                                    uint32_t sample_idx)
{
    const uint32_t q_begin = query_label_indptr[query_idx];
    const uint32_t q_end = query_label_indptr[query_idx + 1];
    const uint32_t s_begin = sample_label_indptr[sample_idx];
    const uint32_t s_end = sample_label_indptr[sample_idx + 1];
    for (uint32_t pos = q_begin; pos < q_end; ++pos) {
        if (!sorted_labels_contain(sample_label_indices, s_begin, s_end, query_label_indices[pos])) {
            return false;
        }
    }
    return true;
}

__device__ __forceinline__ bool sample_label_contains_device(const uint32_t* sample_label_indptr,
                                                             const uint32_t* sample_label_indices,
                                                             uint32_t sample_idx,
                                                             uint32_t label)
{
    const uint32_t s_begin = sample_label_indptr[sample_idx];
    const uint32_t s_end = sample_label_indptr[sample_idx + 1];
    return sorted_labels_contain(sample_label_indices, s_begin, s_end, label);
}

__device__ __forceinline__ bool ranges_match_device(const uint8_t* query_range_modes,
                                                    const float* query_range_lower_bounds,
                                                    const float* query_range_upper_bounds,
                                                    const float* base_range_values,
                                                    uint32_t range_count,
                                                    uint32_t active_range_count,
                                                    uint32_t query_idx,
                                                    uint32_t sample_idx)
{
    if (active_range_count == 0) {
        return true;
    }
    if (base_range_values == nullptr) {
        return false;
    }
    const uint32_t matched = range_filter::count_matching_active_ranges(query_range_modes,
                                                                        query_range_lower_bounds,
                                                                        query_range_upper_bounds,
                                                                        base_range_values,
                                                                        query_idx,
                                                                        sample_idx,
                                                                        range_count);
    return matched == active_range_count;
}

__device__ __forceinline__ bool candidate_passes_device(uint64_t row_mask,
                                                        uint64_t query_filter_bit,
                                                        uint8_t query_filter_mode,
                                                        uint32_t query_active_range_count,
                                                        uint32_t query_single_extra_label,
                                                        const uint8_t* query_range_modes,
                                                        const float* query_range_lower_bounds,
                                                        const float* query_range_upper_bounds,
                                                        const float* base_range_values,
                                                        uint32_t range_count,
                                                        const uint32_t* query_label_indptr,
                                                        const uint32_t* query_label_indices,
                                                        const uint32_t* sample_label_indptr,
                                                        const uint32_t* sample_label_indices,
                                                        uint32_t query_idx,
                                                        uint32_t sample_idx)
{
    if (query_filter_bit == 0 || (row_mask & query_filter_bit) == 0) {
        return false;
    }

    switch (query_filter_mode) {
    case kQueryFilterModeBitOnlyNoRange:
        return true;
    case kQueryFilterModeSingleExtraNoRange:
        return sample_label_contains_device(sample_label_indptr,
                                            sample_label_indices,
                                            sample_idx,
                                            query_single_extra_label);
    case kQueryFilterModeGenericNoRange:
        return labels_match_device(query_label_indptr,
                                   query_label_indices,
                                   sample_label_indptr,
                                   sample_label_indices,
                                   query_idx,
                                   sample_idx);
    case kQueryFilterModeBitOnlyWithRange:
        return ranges_match_device(query_range_modes,
                                   query_range_lower_bounds,
                                   query_range_upper_bounds,
                                   base_range_values,
                                   range_count,
                                   query_active_range_count,
                                   query_idx,
                                   sample_idx);
    case kQueryFilterModeSingleExtraWithRange:
        return ranges_match_device(query_range_modes,
                                   query_range_lower_bounds,
                                   query_range_upper_bounds,
                                   base_range_values,
                                   range_count,
                                   query_active_range_count,
                                   query_idx,
                                   sample_idx) &&
               sample_label_contains_device(sample_label_indptr,
                                            sample_label_indices,
                                            sample_idx,
                                            query_single_extra_label);
    default:
        if (query_active_range_count != 0 &&
            !ranges_match_device(query_range_modes,
                                 query_range_lower_bounds,
                                 query_range_upper_bounds,
                                 base_range_values,
                                 range_count,
                                 query_active_range_count,
                                 query_idx,
                                 sample_idx)) {
            return false;
        }
        return labels_match_device(query_label_indptr,
                                   query_label_indices,
                                   sample_label_indptr,
                                   sample_label_indices,
                                   query_idx,
                                   sample_idx);
    }
}

template <typename T>
__device__ __forceinline__ void load_query_row_to_shared(const T* query_vectors,
                                                         uint32_t query_idx,
                                                         uint32_t dim,
                                                         T* shared_query)
{
    const size_t query_offset = static_cast<size_t>(query_idx) * dim;
    for (uint32_t d = threadIdx.x; d < dim; d += blockDim.x) {
        shared_query[d] = query_vectors[query_offset + d];
    }
}

template <typename T>
__device__ __forceinline__ float l2_distance_device(const T* base_vectors,
                                                    const T* query_row,
                                                    uint32_t dim,
                                                    uint32_t sample_idx,
                                                    uint32_t query_squared_norm)
{
    (void)query_squared_norm;
    float partial = 0.0f;
    const size_t sample_offset = static_cast<size_t>(sample_idx) * dim;
    const uint32_t lane = threadIdx.x & (warp_kernel::kWarpThreads - 1);
    for (uint32_t d = lane; d < dim; d += warp_kernel::kWarpThreads) {
        const float diff = static_cast<float>(base_vectors[sample_offset + d]) -
                           static_cast<float>(query_row[d]);
        partial += diff * diff;
    }
    for (int offset = warp_kernel::kWarpThreads / 2; offset > 0; offset >>= 1) {
        partial += __shfl_down_sync(warp_kernel::kFullMask, partial, offset);
    }
    return __shfl_sync(warp_kernel::kFullMask, partial, 0);
}

// FP16 payload path.  Values are converted to float before subtraction and
// accumulation, preserving the FP32 distance accumulator used by the FP32
// path while reducing global-memory traffic by 2x.
__device__ __forceinline__ float l2_distance_device(const __half* base_vectors,
                                                    const __half* query_row,
                                                    uint32_t dim,
                                                    uint32_t sample_idx,
                                                    uint32_t query_squared_norm)
{
    (void)query_squared_norm;
    float partial = 0.0f;
    const size_t sample_offset = static_cast<size_t>(sample_idx) * dim;
    const uint32_t lane = threadIdx.x & (warp_kernel::kWarpThreads - 1);
    if ((dim & 1u) == 0) {
        const auto* base2 = reinterpret_cast<const __half2*>(base_vectors + sample_offset);
        const auto* query2 = reinterpret_cast<const __half2*>(query_row);
        const uint32_t dim2 = dim / 2;
        for (uint32_t d = lane; d < dim2; d += warp_kernel::kWarpThreads) {
            const float2 b = __half22float2(base2[d]);
            const float2 q = __half22float2(query2[d]);
            const float dx = b.x - q.x;
            const float dy = b.y - q.y;
            partial += dx * dx + dy * dy;
        }
    } else {
        for (uint32_t d = lane; d < dim; d += warp_kernel::kWarpThreads) {
            const float diff = __half2float(base_vectors[sample_offset + d]) -
                               __half2float(query_row[d]);
            partial += diff * diff;
        }
    }
    for (int offset = warp_kernel::kWarpThreads / 2; offset > 0; offset >>= 1) {
        partial += __shfl_down_sync(warp_kernel::kFullMask, partial, offset);
    }
    return __shfl_sync(warp_kernel::kFullMask, partial, 0);
}

__device__ __forceinline__ uint32_t query_squared_norm_device(const uint8_t* query_row,
                                                              uint32_t dim)
{
    const uint32_t lane = threadIdx.x & (warp_kernel::kWarpThreads - 1);
    uint32_t partial = 0;
    if constexpr (kFullGpuUint8Dp4a) {
        if ((dim & 3u) == 0) {
            const auto* packed_query = reinterpret_cast<const uint32_t*>(query_row);
            const uint32_t packed_dim = dim / 4;
            for (uint32_t d = lane; d < packed_dim; d += warp_kernel::kWarpThreads) {
                const uint32_t query_value = packed_query[d];
                partial = __dp4a(query_value, query_value, partial);
            }
        } else {
            for (uint32_t d = lane; d < dim; d += warp_kernel::kWarpThreads) {
                const uint32_t value = query_row[d];
                partial += value * value;
            }
        }
    }
    for (int offset = warp_kernel::kWarpThreads / 2; offset > 0; offset >>= 1) {
        partial += __shfl_down_sync(warp_kernel::kFullMask, partial, offset);
    }
    return __shfl_sync(warp_kernel::kFullMask, partial, 0);
}

template <typename T>
__device__ __forceinline__ uint32_t query_squared_norm_device(const T* query_row, uint32_t dim)
{
    (void)query_row;
    (void)dim;
    return 0;
}

__device__ __forceinline__ float l2_distance_device(const uint8_t* base_vectors,
                                                    const uint8_t* query_row,
                                                    uint32_t dim,
                                                    uint32_t sample_idx,
                                                    uint32_t query_squared_norm)
{
    const uint32_t lane = threadIdx.x & (warp_kernel::kWarpThreads - 1);
    const size_t sample_offset = static_cast<size_t>(sample_idx) * dim;
    if constexpr (kFullGpuUint8Dp4a) {
        if ((dim & 3u) == 0) {
            const auto* packed_base =
                reinterpret_cast<const uint32_t*>(base_vectors + sample_offset);
            const auto* packed_query = reinterpret_cast<const uint32_t*>(query_row);
            const uint32_t packed_dim = dim / 4;
            uint32_t base_squared_norm = 0;
            uint32_t base_query_dot = 0;
            for (uint32_t d = lane; d < packed_dim; d += warp_kernel::kWarpThreads) {
                const uint32_t base_value = packed_base[d];
                const uint32_t query_value = packed_query[d];
                base_squared_norm = __dp4a(base_value, base_value, base_squared_norm);
                base_query_dot = __dp4a(base_value, query_value, base_query_dot);
            }
            for (int offset = warp_kernel::kWarpThreads / 2; offset > 0; offset >>= 1) {
                base_squared_norm +=
                    __shfl_down_sync(warp_kernel::kFullMask, base_squared_norm, offset);
                base_query_dot +=
                    __shfl_down_sync(warp_kernel::kFullMask, base_query_dot, offset);
            }
            const uint32_t squared_distance =
                base_squared_norm + query_squared_norm - 2u * base_query_dot;
            return static_cast<float>(
                __shfl_sync(warp_kernel::kFullMask, squared_distance, 0));
        }
    }

    uint32_t partial = 0;
    for (uint32_t d = lane; d < dim; d += warp_kernel::kWarpThreads) {
        const int diff = static_cast<int>(base_vectors[sample_offset + d]) -
                         static_cast<int>(query_row[d]);
        partial += static_cast<uint32_t>(diff * diff);
    }
    for (int offset = warp_kernel::kWarpThreads / 2; offset > 0; offset >>= 1) {
        partial += __shfl_down_sync(warp_kernel::kFullMask, partial, offset);
    }
    return static_cast<float>(__shfl_sync(warp_kernel::kFullMask, partial, 0));
}

__device__ __forceinline__ uint32_t visited_hash_code(uint32_t value)
{
    const uint64_t mixed = static_cast<uint64_t>(value) ^
                           ((static_cast<uint64_t>(value) >> 16) * kVisitedHashMul) ^
                           kVisitedHashXor;
    return static_cast<uint32_t>(mixed % static_cast<uint64_t>(kHashTableCapacity));
}

struct PackedVisitedEntry {
    uint8_t bytes[3];
};

static_assert(sizeof(PackedVisitedEntry) == 3, "PackedVisitedEntry must remain byte-packed");

__device__ __forceinline__ uint32_t visited_hash_load(const uint32_t* entry)
{
    return *entry;
}

__device__ __forceinline__ uint32_t visited_hash_load(const PackedVisitedEntry* entry)
{
    return static_cast<uint32_t>(entry->bytes[0]) |
           (static_cast<uint32_t>(entry->bytes[1]) << 8) |
           (static_cast<uint32_t>(entry->bytes[2]) << 16);
}

__device__ __forceinline__ void visited_hash_store(uint32_t* entry, uint32_t value)
{
    *entry = value;
}

__device__ __forceinline__ void visited_hash_store(PackedVisitedEntry* entry, uint32_t value)
{
    entry->bytes[0] = static_cast<uint8_t>(value);
    entry->bytes[1] = static_cast<uint8_t>(value >> 8);
    entry->bytes[2] = static_cast<uint8_t>(value >> 16);
}

__device__ __forceinline__ uint32_t visited_hash_empty(const uint32_t*)
{
    return kVisitedEmpty;
}

__device__ __forceinline__ uint32_t visited_hash_empty(const PackedVisitedEntry*)
{
    return kPackedVisitedEmpty;
}

template <typename Entry>
__device__ __forceinline__ bool visited_hash_test(const Entry* data, uint32_t value)
{
    uint32_t code = visited_hash_code(value);
    for (int probes = 0; probes < kHashTableCapacity; ++probes) {
        const uint32_t current = visited_hash_load(data + code);
        if (current == visited_hash_empty(data)) {
            return false;
        }
        if (current == value) {
            return true;
        }
        code = static_cast<uint32_t>((code + 1u) % kHashTableCapacity);
    }
    return false;
}

template <typename Entry>
__device__ __forceinline__ void visited_hash_add(Entry* data, uint32_t value)
{
    uint32_t code = visited_hash_code(value);
    for (int probes = 0; probes < kHashTableCapacity; ++probes) {
        const uint32_t current = visited_hash_load(data + code);
        if (current == value) {
            return;
        }
        if (current == visited_hash_empty(data)) {
            visited_hash_store(data + code, value);
            return;
        }
        code = static_cast<uint32_t>((code + 1u) % kHashTableCapacity);
    }
}

__device__ bool better_result(float lhs_dist, uint32_t lhs_id, float rhs_dist, uint32_t rhs_id)
{
    return lhs_dist < rhs_dist || (lhs_dist == rhs_dist && lhs_id < rhs_id);
}

__device__ bool insert_best_result(float dist,
                                   uint32_t sample_id,
                                   uint32_t topk,
                                   float* best_dist,
                                   uint32_t* best_ids,
                                   uint32_t& best_size)
{
    bool changed = false;
    for (uint32_t i = 0; i < best_size; ++i) {
        if (best_ids[i] == sample_id) {
            if (dist >= best_dist[i]) {
                return false;
            }
            for (uint32_t j = i; j + 1 < best_size; ++j) {
                best_dist[j] = best_dist[j + 1];
                best_ids[j] = best_ids[j + 1];
            }
            --best_size;
            changed = true;
            break;
        }
    }

    if (best_size == topk &&
        !better_result(dist, sample_id, best_dist[topk - 1], best_ids[topk - 1])) {
        return changed;
    }

    uint32_t pos = best_size;
    if (best_size < topk) {
        ++best_size;
    } else {
        pos = topk - 1;
    }
    while (pos > 0 && better_result(dist, sample_id, best_dist[pos - 1], best_ids[pos - 1])) {
        best_dist[pos] = best_dist[pos - 1];
        best_ids[pos] = best_ids[pos - 1];
        --pos;
    }
    best_dist[pos] = dist;
    best_ids[pos] = sample_id;
    return true;
}

// Resident range-posting scan used by the full-GPU backend.  Each posting
// route owns one block; lane zero performs the exact scan and keeps a sorted
// top-k.  The posting list itself is resident on device, so no host/SSD path
// is involved.  This deliberately shares the same range/label predicates as
// graph traversal to preserve result legality.
template <typename T>
__global__ void full_gpu_range_posting_kernel(const T* base_vectors,
                                              const T* query_vectors,
                                              uint32_t base_rows,
                                              uint32_t dim,
                                              const uint32_t* route_ids,
                                              const uint32_t* query_vector_ids,
                                              const uint64_t* posting_begins,
                                              const uint64_t* posting_sizes,
                                              const uint32_t* posting_sample_ids,
                                              const uint32_t* query_label_indptr,
                                              const uint32_t* query_label_indices,
                                              const uint32_t* sample_label_indptr,
                                              const uint32_t* sample_label_indices,
                                              const uint8_t* query_range_modes,
                                              const float* query_range_lower_bounds,
                                              const float* query_range_upper_bounds,
                                              const float* base_range_values,
                                              uint32_t range_count,
                                              uint32_t topk,
                                              uint32_t route_count,
                                              idx_t* route_results,
                                              float* route_distances,
                                              int* route_steps)
{
    const uint32_t posting_idx = blockIdx.x;
    if (posting_idx >= route_count) return;
    const uint32_t route_idx = route_ids[posting_idx];
    const uint32_t vector_query_idx = query_vector_ids == nullptr
        ? route_idx : query_vector_ids[route_idx];
    float best_dist[kMaxTopk];
    uint32_t best_ids[kMaxTopk];
    __shared__ float lane_dist[warp_kernel::kWarpThreads][kMaxTopk];
    __shared__ uint32_t lane_ids[warp_kernel::kWarpThreads][kMaxTopk];
    uint32_t best_size = 0;
    for (uint32_t i = 0; i < kMaxTopk; ++i) {
        best_dist[i] = FLT_MAX;
        best_ids[i] = UINT32_MAX;
    }
    const uint64_t begin = posting_begins[route_idx];
    const uint64_t count = posting_sizes[route_idx];
    for (uint64_t pos = threadIdx.x; pos < count; pos += blockDim.x) {
        const uint32_t sample_id = posting_sample_ids[begin + pos];
        if (sample_id >= base_rows) continue;
        const bool range_ok = ranges_match_device(query_range_modes,
                                                  query_range_lower_bounds,
                                                  query_range_upper_bounds,
                                                  base_range_values,
                                                  range_count,
                                                  range_count,
                                                  route_idx,
                                                  sample_id);
        if (!range_ok) continue;
        const bool labels_ok = labels_match_device(query_label_indptr,
                                                   query_label_indices,
                                                   sample_label_indptr,
                                                   sample_label_indices,
                                                   route_idx,
                                                   sample_id);
        if (!labels_ok) continue;
        const T* query_row = query_vectors + static_cast<size_t>(vector_query_idx) * dim;
        float dist = 0.0f;
        const size_t sample_offset = static_cast<size_t>(sample_id) * dim;
        if constexpr (std::is_same_v<T, __half>) {
            if ((dim & 1u) == 0) {
                const auto* base2 = reinterpret_cast<const __half2*>(base_vectors + sample_offset);
                const auto* query2 = reinterpret_cast<const __half2*>(query_row);
                for (uint32_t d = 0; d < dim / 2; ++d) {
                    const float2 b = __half22float2(base2[d]);
                    const float2 q = __half22float2(query2[d]);
                    const float dx = b.x - q.x;
                    const float dy = b.y - q.y;
                    dist += dx * dx + dy * dy;
                }
            } else {
                for (uint32_t d = 0; d < dim; ++d) {
                    const float diff = __half2float(base_vectors[sample_offset + d]) -
                                       __half2float(query_row[d]);
                    dist += diff * diff;
                }
            }
        } else {
            for (uint32_t d = 0; d < dim; ++d) {
                const float diff = static_cast<float>(base_vectors[sample_offset + d]) -
                                   static_cast<float>(query_row[d]);
                dist += diff * diff;
            }
        }
        insert_best_result(dist, sample_id, topk, best_dist, best_ids, best_size);
    }
    for (uint32_t i = 0; i < kMaxTopk; ++i) {
        lane_dist[threadIdx.x][i] = best_dist[i];
        lane_ids[threadIdx.x][i] = best_ids[i];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        best_size = 0;
        for (uint32_t lane = 0; lane < warp_kernel::kWarpThreads; ++lane) {
            for (uint32_t i = 0; i < kMaxTopk; ++i) {
                if (lane_ids[lane][i] != UINT32_MAX) {
                    insert_best_result(lane_dist[lane][i], lane_ids[lane][i],
                                       topk, best_dist, best_ids, best_size);
                }
            }
        }
        const uint32_t out_base = route_idx * topk;
        for (uint32_t i = 0; i < topk; ++i) {
            route_results[out_base + i] = best_ids[i];
            route_distances[out_base + i] = best_dist[i];
        }
        route_steps[route_idx] = static_cast<int>(count > INT_MAX ? INT_MAX : count);
    }
}

__host__ __device__ __forceinline__ uint32_t pack_pool_payload(uint32_t local_row, bool used)
{
    return static_cast<uint32_t>(local_row | (used ? kPoolUsedMask : 0u));
}

__host__ __device__ __forceinline__ uint32_t unpack_pool_local_row(uint32_t payload)
{
    return static_cast<uint32_t>(payload & ~kPoolUsedMask);
}

__host__ __device__ __forceinline__ bool pool_payload_used(uint32_t payload)
{
    return (payload & kPoolUsedMask) != 0u;
}

__host__ __device__ __forceinline__ uint32_t mark_pool_payload_used(uint32_t payload)
{
    return static_cast<uint32_t>(payload | kPoolUsedMask);
}

template <class T>
__device__ bool pickup_next_unused_from_pool(T* pool, int size, T& out)
{
    int best = -1;
    for (int i = 1; i < size; ++i) {
        if (pool_payload_used(pool[i].second)) {
            continue;
        }
        if (best < 0 || pool[i].first < pool[best].first) {
            best = i;
        }
    }
    if (best < 0) {
        return false;
    }
    out = pool[best];
    pool[best].second = mark_pool_payload_used(pool[best].second);
    return true;
}

template <class T>
__device__ bool pickup_next_unused_from_pool_block(T* pool,
                                                   int size,
                                                   T& out,
                                                   int* warp_best_indices,
                                                   float* warp_best_distances,
                                                   int& selected_index,
                                                   uint32_t group_threads = 0)
{
    const uint32_t thread_count = group_threads == 0 ? blockDim.x : group_threads;
    const uint32_t group_thread = group_threads == 0 ? threadIdx.x : threadIdx.x % group_threads;
    const int lane = static_cast<int>(threadIdx.x & (warp_kernel::kWarpThreads - 1));
    const int warp_id = static_cast<int>(group_thread / warp_kernel::kWarpThreads);
    int best_index = -1;
    float best_distance = FLT_MAX;
    for (int i = static_cast<int>(group_thread) + 1; i < size; i += thread_count) {
        if (pool_payload_used(pool[i].second)) {
            continue;
        }
        const float distance = pool[i].first;
        if (best_index < 0 || distance < best_distance ||
            (distance == best_distance && i < best_index)) {
            best_index = i;
            best_distance = distance;
        }
    }
    for (int offset = warp_kernel::kWarpThreads / 2; offset > 0; offset >>= 1) {
        const int other_index = __shfl_down_sync(warp_kernel::kFullMask, best_index, offset);
        const float other_distance =
            __shfl_down_sync(warp_kernel::kFullMask, best_distance, offset);
        if (other_index >= 0 &&
            (best_index < 0 || other_distance < best_distance ||
             (other_distance == best_distance && other_index < best_index))) {
            best_index = other_index;
            best_distance = other_distance;
        }
    }
    if (lane == 0) {
        warp_best_indices[warp_id] = best_index;
        warp_best_distances[warp_id] = best_distance;
    }
    __syncthreads();

    if (group_thread == 0) {
        selected_index = -1;
        float selected_distance = FLT_MAX;
        const int warp_count = static_cast<int>(thread_count / warp_kernel::kWarpThreads);
        for (int warp = 0; warp < warp_count; ++warp) {
            const int candidate_index = warp_best_indices[warp];
            const float candidate_distance = warp_best_distances[warp];
            if (candidate_index >= 0 &&
                (selected_index < 0 || candidate_distance < selected_distance ||
                 (candidate_distance == selected_distance && candidate_index < selected_index))) {
                selected_index = candidate_index;
                selected_distance = candidate_distance;
            }
        }
        if (selected_index >= 0) {
            out = pool[selected_index];
            pool[selected_index].second = mark_pool_payload_used(pool[selected_index].second);
        }
    }
    __syncthreads();
    return selected_index >= 0;
}

// Route-local sorted queue helpers.  The queue is kept in ascending distance
// order, while the used bit remains part of the payload so pickup preserves
// the baseline's expand-once semantics.  A bounded merge retains the best
// queue_limit entries (including already-used entries).
template <int kQueueSize>
__device__ bool pickup_next_unused_sorted(sorted_frontier::Entry* pool,
                                          int count,
                                          sorted_frontier::Entry& out)
{
    (void)kQueueSize;
    for (int i = 0; i < count; ++i) {
        const auto entry = pool[i];
        if ((entry.payload & sorted_frontier::kUsed) == 0u) {
            out = entry;
            pool[i].payload |= sorted_frontier::kUsed;
            return true;
        }
    }
    return false;
}

template <int kQueueSize>
__device__ int merge_route_local_sorted(const sorted_frontier::Entry* old_entries,
                                        int old_count,
                                        const sorted_frontier::Entry* batch,
                                        int batch_count,
                                        sorted_frontier::Entry* destination)
{
    sorted_frontier::merge_parallel(old_entries, old_count, batch, batch_count,
                                    destination, kQueueSize);
    return min(kQueueSize, old_count + batch_count);
}

// Query-level distance reuse primitive for future overlap graphs.  Distances
// are keyed by sample id, independently of the graph occurrence that
// requested them.  Graph expansion state must remain separate and use a
// (graph_id, local_row) identity.
struct FullGpuSampleDistanceCacheEntry {
    uint32_t sample_id;
    float distance;
};

template <uint32_t kCapacity>
__device__ __forceinline__ bool lookup_sample_distance(
    const FullGpuSampleDistanceCacheEntry (&cache)[kCapacity],
    uint32_t sample_id,
    float& distance)
{
    static_assert((kCapacity & (kCapacity - 1)) == 0,
                  "sample distance cache capacity must be a power of two");
    const uint32_t start = (sample_id * 2654435761u) & (kCapacity - 1);
    for (uint32_t probe = 0; probe < kCapacity; ++probe) {
        const auto& entry = cache[(start + probe) & (kCapacity - 1)];
        if (entry.sample_id == sample_id) {
            distance = entry.distance;
            return true;
        }
        if (entry.sample_id == UINT32_MAX) return false;
    }
    return false;
}

template <uint32_t kCapacity>
__device__ __forceinline__ void publish_sample_distance(
    FullGpuSampleDistanceCacheEntry (&cache)[kCapacity],
    uint32_t sample_id,
    float distance)
{
    const uint32_t start = (sample_id * 2654435761u) & (kCapacity - 1);
    for (uint32_t probe = 0; probe < kCapacity; ++probe) {
        auto& entry = cache[(start + probe) & (kCapacity - 1)];
        if (entry.sample_id == sample_id || entry.sample_id == UINT32_MAX) {
            entry.distance = distance;
            __threadfence_block();
            entry.sample_id = sample_id;
            return;
        }
    }
}

template <int kQueueSize, typename T, bool kUsePackedVisited = false>
__device__ void run_full_gpu_highfreq_route(const T* base_vectors,
                                            const T* query_vectors,
                                            uint32_t base_rows,
                                            uint32_t dim,
                                            const uint32_t* graph,
                                            uint32_t graph_degree,
                                            uint32_t stored_degree,
                                            uint32_t query_idx,
                                            const uint32_t* query_vector_ids,
                                            const uint32_t* query_group_ids,
                                            const uint32_t* start_local_ids,
                                            const uint32_t* label_size,
                                            const uint32_t* label_offset,
                                            const uint32_t* row_to_sample,
                                            const uint64_t* row_filter_mask,
                                            const uint64_t* query_filter_bits,
                                            const uint8_t* query_filter_modes,
                                            const uint32_t* query_single_extra_labels,
                                            const uint32_t* query_active_range_counts,
                                            const uint32_t* query_label_indptr,
                                            const uint32_t* query_label_indices,
                                            const uint32_t* sample_label_indptr,
                                            const uint32_t* sample_label_indices,
                                            const uint8_t* query_range_modes,
                                            const float* query_range_lower_bounds,
                                            const float* query_range_upper_bounds,
                                            const float* base_range_values,
                                            uint32_t range_count,
                                            uint32_t finish_count,
                                            int early_stop_mode,
                                            uint32_t max_steps,
                                            uint32_t topk,
                                            idx_t* route_results,
                                            float* route_distances,
                                            int* route_steps,
                                            bool reuse_shared_query)
{
    const uint32_t lane = threadIdx.x & (warp_kernel::kWarpThreads - 1);
    const uint32_t warp_id = threadIdx.x / warp_kernel::kWarpThreads;
    const uint32_t warp_count = blockDim.x / warp_kernel::kWarpThreads;
    const uint32_t result_base = query_idx * topk;
    const uint32_t vector_query_idx = query_vector_ids == nullptr ? query_idx : query_vector_ids[query_idx];
    const uint32_t group_id = query_group_ids[query_idx];
    if (group_id == UINT32_MAX || blockDim.x < warp_kernel::kWarpThreads ||
        (blockDim.x % warp_kernel::kWarpThreads) != 0) {
        if (threadIdx.x == 0) {
            route_steps[query_idx] = 0;
        }
        return;
    }

    const uint32_t group_size = label_size[group_id];
    const uint32_t group_offset = label_offset[group_id];
    const uint32_t seed_local = start_local_ids[query_idx];
    const uint64_t query_filter_bit = query_filter_bits[query_idx];
    constexpr uint32_t queue_limit = static_cast<uint32_t>(kQueueSize);
    const uint32_t evaluation_degree = stored_degree;

    uint32_t query_active_range_count = 0;
    uint32_t query_single_extra_label = UINT32_MAX;
    uint32_t query_filter_mode = kQueryFilterModeBitOnlyNoRange;
    bool current_result_improved = false;

    constexpr uint32_t visited_storage_words =
        kUsePackedVisited ? (kHashTableCapacity * sizeof(PackedVisitedEntry)) / sizeof(uint32_t)
                          : kHashTableCapacity;
    using VisitedEntry = std::conditional_t<kUsePackedVisited, PackedVisitedEntry, uint32_t>;
    __shared__ uint32_t visited_hash_storage[visited_storage_words];
    VisitedEntry* visited_hash = reinterpret_cast<VisitedEntry*>(visited_hash_storage);
    __shared__ warp_kernel::KernelPair<float, uint32_t> queue_pool[kQueueSize + 2];
    __shared__ uint32_t candidate_local[kMaxGraphDegree];
    __shared__ float candidate_dist[kMaxGraphDegree];
    __shared__ uint32_t current_local_shared;
    extern __shared__ __align__(16) unsigned char shared_query_storage[];
    T* shared_query = reinterpret_cast<T*>(shared_query_storage);
    __shared__ float best_dist[kMaxTopk];
    __shared__ uint32_t best_ids[kMaxTopk];
    __shared__ uint32_t best_size;
    __shared__ int heap_size;
    __shared__ int expanded_count;
    __shared__ int pass_count;
    __shared__ uint32_t steps;
    __shared__ uint32_t finished_hits;
    __shared__ uint32_t first_step;
    __shared__ uint32_t done;
    __shared__ uint32_t query_squared_norm;
    __shared__ int pickup_warp_best_indices[kHighfreqBlockThreads / warp_kernel::kWarpThreads];
    __shared__ float pickup_warp_best_distances[kHighfreqBlockThreads / warp_kernel::kWarpThreads];
    __shared__ int pickup_selected_index;

    if (group_size == 0 || seed_local == UINT32_MAX || seed_local >= group_size ||
        topk == 0 || topk > kMaxTopk || queue_limit == 0) {
        if (threadIdx.x == 0) {
            route_steps[query_idx] = 0;
        }
        return;
    }

    if (!reuse_shared_query) {
        load_query_row_to_shared(query_vectors, vector_query_idx, dim, shared_query);
        __syncthreads();
        if (warp_id == 0) {
            const uint32_t norm = query_squared_norm_device(shared_query, dim);
            if (lane == 0) {
                query_squared_norm = norm;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        query_active_range_count = query_active_range_counts[query_idx];
        query_single_extra_label = query_single_extra_labels[query_idx];
        query_filter_mode = query_filter_modes[query_idx];
    }
    for (uint32_t i = threadIdx.x; i < visited_storage_words; i += blockDim.x) {
        visited_hash_storage[i] = UINT32_MAX;
    }
    if (threadIdx.x == 0) {
        best_size = 0;
        heap_size = 1;
        expanded_count = 0;
        pass_count = 0;
        steps = 0;
        finished_hits = 0;
        first_step = 1;
        done = 0;
        for (uint32_t i = 0; i < kMaxTopk; ++i) {
            best_dist[i] = FLT_MAX;
            best_ids[i] = UINT32_MAX;
        }
    }
    __syncthreads();

    uint32_t seed_sample = UINT32_MAX;
    uint32_t seed_should_eval = 0;
    warp_kernel::KernelPair<float, uint32_t> now;
    warp_kernel::KernelPair<float, uint32_t> next;

    if (warp_id == 0) {
        if (lane == 0) {
            const uint32_t seed_global = group_offset + seed_local;
            seed_sample = row_to_sample[seed_global];
            if (seed_sample < base_rows && !visited_hash_test(visited_hash, seed_local)) {
                seed_should_eval = 1;
            }
        }
        seed_sample = __shfl_sync(warp_kernel::kFullMask, seed_sample, 0);
        seed_should_eval = __shfl_sync(warp_kernel::kFullMask, seed_should_eval, 0);
        if (seed_should_eval) {
            const float seed_dist = l2_distance_device(base_vectors,
                                                       shared_query,
                                                       dim,
                                                       seed_sample,
                                                       query_squared_norm);
            if (lane == 0) {
                warp_kernel::KernelPair<float, uint32_t> kp;
                kp.first = seed_dist;
                kp.second = pack_pool_payload(seed_local, false);

                smmh2::insert(queue_pool, heap_size, kp);
                if (heap_size >= static_cast<int>(queue_limit) + 2) {
                    smmh2::pop_max(queue_pool, heap_size);
                }

                visited_hash_add(visited_hash, seed_local);
            }
        } else if (lane == 0) {
            done = 1;
        }
    }

    __syncthreads();

    if constexpr (kFullGpuCooperativePickup) {
        const bool found = pickup_next_unused_from_pool_block(queue_pool,
                                                               heap_size,
                                                               now,
                                                               pickup_warp_best_indices,
                                                               pickup_warp_best_distances,
                                                               pickup_selected_index);
        if (threadIdx.x == 0 && !found) done = 1;
    } else if (threadIdx.x == 0 && !done &&
               !pickup_next_unused_from_pool(queue_pool, heap_size, now)) {
        done = 1;
    }
    __syncthreads();
    if (done) {
        if (threadIdx.x == 0) {
            route_steps[query_idx] = 0;
        }
        return;
    }

    while (true) {
        __syncthreads();
        if (done) {
            break;
        }

        if (threadIdx.x == 0 && max_steps != 0 && steps >= max_steps) {
            done = 1;
        }
        __syncthreads();
        if (done) {
            break;
        }

        if (threadIdx.x == 0) {
            current_local_shared = unpack_pool_local_row(now.second);
            const uint32_t now_global = group_offset + current_local_shared;
            const uint32_t now_sample = row_to_sample[now_global];
            const bool passes_filter = candidate_passes_device(row_filter_mask[now_global],
                                                               query_filter_bit,
                                                               static_cast<uint8_t>(query_filter_mode),
                                                               query_active_range_count,
                                                               query_single_extra_label,
                                                               query_range_modes,
                                                               query_range_lower_bounds,
                                                               query_range_upper_bounds,
                                                               base_range_values,
                                                               range_count,
                                                               query_label_indptr,
                                                               query_label_indices,
                                                               sample_label_indptr,
                                                               sample_label_indices,
                                                               query_idx,
                                                               now_sample);
            ++expanded_count;
            current_result_improved = false;
            if (passes_filter) {
                ++pass_count;
                current_result_improved = insert_best_result(now.first,
                                                             now_sample,
                                                             topk,
                                                             best_dist,
                                                             best_ids,
                                                             best_size);
            }
            ++steps;
        }
        __syncthreads();

        const uint32_t current_global = group_offset + current_local_shared;
        const size_t graph_base = static_cast<size_t>(current_global) * graph_degree;
        for (uint32_t edge = warp_id; edge < evaluation_degree; edge += warp_count) {
            uint32_t local = UINT32_MAX;
            uint32_t sample = UINT32_MAX;
            uint32_t should_eval = 0;
            if (lane == 0) {
                const uint32_t graph_edge = edge;
                if (graph_edge < stored_degree) {
                    local = graph[graph_base + graph_edge];
                }
                if (local != UINT32_MAX &&
                    local < group_size &&
                    !visited_hash_test(visited_hash, local)) {
                    const uint32_t candidate_global = group_offset + local;
                    sample = row_to_sample[candidate_global];
                    if (sample < base_rows) {
                        should_eval = 1;
                    } else {
                        local = UINT32_MAX;
                    }
                }
            }
            local = __shfl_sync(warp_kernel::kFullMask, local, 0);
            sample = __shfl_sync(warp_kernel::kFullMask, sample, 0);
            should_eval = __shfl_sync(warp_kernel::kFullMask, should_eval, 0);
            float dist = FLT_MAX;
            if (should_eval) {
                dist = l2_distance_device(base_vectors,
                                          shared_query,
                                          dim,
                                          sample,
                                          query_squared_norm);
            }
            if (lane == 0) {
                candidate_local[edge] = should_eval ? local : UINT32_MAX;
                candidate_dist[edge] = dist;
            }
        }
        __syncthreads();


        if (threadIdx.x == 0) {
            for (uint32_t edge = 0; edge < evaluation_degree; ++edge) {
                if (candidate_local[edge] == UINT32_MAX) continue;
                warp_kernel::KernelPair<float, uint32_t> kp;
                kp.first = candidate_dist[edge];
                kp.second = pack_pool_payload(candidate_local[edge], false);
                if (heap_size > 2 &&
                    heap_size >= static_cast<int>(queue_limit) + 1 &&
                    queue_pool[2].first < kp.first) {
                    continue;
                }
                smmh2::insert(queue_pool, heap_size, kp);
                visited_hash_add(visited_hash, candidate_local[edge]);
                if (heap_size >= static_cast<int>(queue_limit) + 2) {
                    smmh2::pop_max(queue_pool, heap_size);
                }
            }
        }
        __syncthreads();

        bool found_next = false;

        if constexpr (kFullGpuCooperativePickup) {
            found_next = pickup_next_unused_from_pool_block(queue_pool,
                                                            heap_size,
                                                            next,
                                                            pickup_warp_best_indices,
                                                            pickup_warp_best_distances,
                                                            pickup_selected_index);
        } else if (threadIdx.x == 0) {
            found_next = pickup_next_unused_from_pool(queue_pool, heap_size, next);
        }

        if (threadIdx.x == 0) {
            if (!found_next) {
                done = 1;
            } else {
                if (!first_step) {
                    const uint32_t next_local = unpack_pool_local_row(next.second);
                    const uint32_t next_sample = row_to_sample[group_offset + next_local];
                    if (early_stop_mode == kSearchEarlyStopAdaptive && current_result_improved) {
                        finished_hits = 0;
                    } else if (best_size == topk &&
                        !better_result(next.first, next_sample, best_dist[topk - 1], best_ids[topk - 1])) {
                        ++finished_hits;
                        if (early_stop_reached(early_stop_mode,
                                               static_cast<int>(finished_hits),
                                               static_cast<int>(finish_count),
                                               static_cast<int>(queue_limit),
                                               expanded_count,
                                               pass_count,
                                               static_cast<int>(topk))) {
                            done = 1;
                        }
                    } else {
                        finished_hits = 0;
                    }
                }
                if (!done) {
                    now = next;
                    first_step = 0;
                }
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        for (uint32_t i = 0; i < topk; ++i) {
            route_results[result_base + i] = i < best_size ? static_cast<idx_t>(best_ids[i])
                                                           : static_cast<idx_t>(UINT32_MAX);
            route_distances[result_base + i] = i < best_size ? best_dist[i] : FLT_MAX;
        }
        route_steps[query_idx] = static_cast<int>(steps);
    }
}

template <int kQueueSize, typename T, bool kUsePackedVisited = false>
__global__ void full_gpu_highfreq_kernel(const T* base_vectors,
                                         const T* query_vectors,
                                         uint32_t base_rows,
                                         uint32_t dim,
                                         const uint32_t* graph,
                                         uint32_t graph_degree,
                                         uint32_t stored_degree,
                                         const uint32_t* active_query_ids,
                                         uint32_t active_query_count,
                                         const uint32_t* query_vector_ids,
                                         const uint32_t* query_group_ids,
                                         const uint32_t* start_local_ids,
                                         const uint32_t* label_size,
                                         const uint32_t* label_offset,
                                         const uint32_t* row_to_sample,
                                         const uint64_t* row_filter_mask,
                                         const uint64_t* query_filter_bits,
                                         const uint8_t* query_filter_modes,
                                         const uint32_t* query_single_extra_labels,
                                         const uint32_t* query_active_range_counts,
                                         const uint32_t* query_label_indptr,
                                         const uint32_t* query_label_indices,
                                         const uint32_t* sample_label_indptr,
                                         const uint32_t* sample_label_indices,
                                         const uint8_t* query_range_modes,
                                         const float* query_range_lower_bounds,
                                         const float* query_range_upper_bounds,
                                         const float* base_range_values,
                                         uint32_t range_count,
                                         uint32_t finish_count,
                                         int early_stop_mode,
                                         uint32_t max_steps,
                                         uint32_t topk,
                                         idx_t* route_results,
                                         float* route_distances,
                                         int* route_steps)
{
    const uint32_t active_idx = blockIdx.x;
    if (active_idx >= active_query_count) return;
    const uint32_t query_idx = active_query_ids[active_idx];
    if (query_idx == UINT32_MAX) return;
    run_full_gpu_highfreq_route<kQueueSize, T, kUsePackedVisited>(base_vectors,
                                            query_vectors,
                                            base_rows,
                                            dim,
                                            graph,
                                            graph_degree,
                                            stored_degree,
                                            query_idx,
                                            query_vector_ids,
                                            query_group_ids,
                                            start_local_ids,
                                            label_size,
                                            label_offset,
                                            row_to_sample,
                                            row_filter_mask,
                                            query_filter_bits,
                                            query_filter_modes,
                                            query_single_extra_labels,
                                            query_active_range_counts,
                                            query_label_indptr,
                                            query_label_indices,
                                            sample_label_indptr,
                                            sample_label_indices,
                                            query_range_modes,
                                            query_range_lower_bounds,
                                            query_range_upper_bounds,
                                            base_range_values,
                                            range_count,
                                            finish_count,
                                            early_stop_mode,
                                            max_steps,
                                            topk,
                                            route_results,
                                            route_distances,
                                            route_steps,
                                            false);
}

template <typename T>
__global__ void full_gpu_lowfreq_kernel(const T* base_vectors,
                                        const T* query_vectors,
                                        uint32_t base_rows,
                                        uint32_t dim,
                                        const uint32_t* active_query_ids,
                                        uint32_t active_query_count,
                                        const uint32_t* query_vector_ids,
                                        const uint32_t* query_group_ids,
                                        const uint64_t* query_filter_bits,
                                        const uint32_t* low_label_size,
                                        const uint32_t* low_label_offset,
                                        const uint32_t* low_sample_ids,
                                        const uint64_t* low_row_filter_mask,
                                        const uint8_t* query_filter_modes,
                                        const uint32_t* query_single_extra_labels,
                                        const uint32_t* query_active_range_counts,
                                        const uint32_t* query_label_indptr,
                                        const uint32_t* query_label_indices,
                                        const uint32_t* sample_label_indptr,
                                        const uint32_t* sample_label_indices,
                                        const uint8_t* query_range_modes,
                                        const float* query_range_lower_bounds,
                                        const float* query_range_upper_bounds,
                                        const float* base_range_values,
                                        uint32_t range_count,
                                        uint32_t topk,
                                        idx_t* route_results,
                                        float* route_distances,
                                        int* route_steps)
{
    const uint32_t active_idx = blockIdx.x;
    const uint32_t lane = threadIdx.x & (warp_kernel::kWarpThreads - 1);
    if (active_idx >= active_query_count) {
        return;
    }

    const uint32_t query_idx = active_query_ids[active_idx];
    const uint32_t vector_query_idx = query_vector_ids == nullptr ? query_idx : query_vector_ids[query_idx];
    const uint32_t result_base = query_idx * topk;
    const uint32_t group_id = query_group_ids[query_idx];
    if (group_id == UINT32_MAX) {
        if (lane == 0) {
            route_steps[query_idx] = 0;
        }
        return;
    }
    const uint32_t begin = low_label_offset[group_id];
    const uint32_t count = low_label_size[group_id];
    const uint64_t query_filter_bit = query_filter_bits[query_idx];
    uint32_t query_active_range_count = 0;
    uint32_t query_single_extra_label = UINT32_MAX;
    uint32_t query_filter_mode = kQueryFilterModeBitOnlyNoRange;
    extern __shared__ unsigned char shared_query_storage[];
    T* shared_query = reinterpret_cast<T*>(shared_query_storage);
    __shared__ float best_dist[kMaxTopk];
    __shared__ uint32_t best_ids[kMaxTopk];
    __shared__ uint32_t best_size;
    __shared__ uint32_t evaluated;
    __shared__ uint32_t query_squared_norm;

    load_query_row_to_shared(query_vectors, vector_query_idx, dim, shared_query);
    __syncwarp(warp_kernel::kFullMask);
    const uint32_t norm = query_squared_norm_device(shared_query, dim);
    if (lane == 0) {
        query_squared_norm = norm;
    }
    if (lane == 0) {
        query_active_range_count = query_active_range_counts[query_idx];
        query_single_extra_label = query_single_extra_labels[query_idx];
        query_filter_mode = query_filter_modes[query_idx];
    }
    query_active_range_count = __shfl_sync(warp_kernel::kFullMask, query_active_range_count, 0);
    query_single_extra_label = __shfl_sync(warp_kernel::kFullMask, query_single_extra_label, 0);
    query_filter_mode = __shfl_sync(warp_kernel::kFullMask, query_filter_mode, 0);
    __syncwarp(warp_kernel::kFullMask);

    if (lane == 0) {
        best_size = 0;
        evaluated = 0;
        for (uint32_t i = 0; i < kMaxTopk; ++i) {
            best_dist[i] = FLT_MAX;
            best_ids[i] = UINT32_MAX;
        }
    }
    __syncwarp(warp_kernel::kFullMask);

    for (uint32_t j = 0; j < count; ++j) {
        uint32_t sample_id = UINT32_MAX;
        uint32_t should_eval = 0;
        if (lane == 0) {
            const uint32_t row = begin + j;
            sample_id = low_sample_ids[row];
            if (sample_id < base_rows &&
                candidate_passes_device(low_row_filter_mask[row],
                                        query_filter_bit,
                                        static_cast<uint8_t>(query_filter_mode),
                                        query_active_range_count,
                                        query_single_extra_label,
                                        query_range_modes,
                                        query_range_lower_bounds,
                                        query_range_upper_bounds,
                                        base_range_values,
                                        range_count,
                                        query_label_indptr,
                                        query_label_indices,
                                        sample_label_indptr,
                                        sample_label_indices,
                                        query_idx,
                                        sample_id)) {
                should_eval = 1;
            }
        }
        sample_id = __shfl_sync(warp_kernel::kFullMask, sample_id, 0);
        should_eval = __shfl_sync(warp_kernel::kFullMask, should_eval, 0);
        if (should_eval) {
            const float dist = l2_distance_device(base_vectors,
                                                  shared_query,
                                                  dim,
                                                  sample_id,
                                                  query_squared_norm);
            if (lane == 0) {
                ++evaluated;
                insert_best_result(dist, sample_id, topk, best_dist, best_ids, best_size);
            }
        }
    }

    if (lane == 0) {
        for (uint32_t i = 0; i < topk; ++i) {
            route_results[result_base + i] = i < best_size ? static_cast<idx_t>(best_ids[i])
                                                           : static_cast<idx_t>(UINT32_MAX);
            route_distances[result_base + i] = i < best_size ? best_dist[i] : FLT_MAX;
        }
        route_steps[query_idx] = static_cast<int>(evaluated);
    }
}

__global__ void init_route_outputs_kernel(idx_t* route_results,
                                          float* route_distances,
                                          int* route_steps,
                                          uint32_t route_count,
                                          uint32_t topk)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = route_count * topk;
    if (idx < total) {
        route_results[idx] = static_cast<idx_t>(UINT32_MAX);
        route_distances[idx] = FLT_MAX;
    }
    if (idx < route_count) {
        route_steps[idx] = 0;
    }
}

template <typename T>
__device__ void persistent_latency_scan_low_route(const T* base_vectors,
                                                  const T* query_vectors,
                                                  uint32_t base_rows,
                                                  uint32_t dim,
                                                  uint32_t route_idx,
                                                  const uint32_t* query_vector_ids,
                                                  const uint32_t* query_group_ids,
                                                  const uint64_t* query_filter_bits,
                                                  const uint32_t* low_label_size,
                                                  const uint32_t* low_label_offset,
                                                  const uint32_t* low_sample_ids,
                                                  const uint64_t* low_row_filter_mask,
                                                  const uint8_t* query_filter_modes,
                                                  const uint32_t* query_single_extra_labels,
                                                  const uint32_t* query_active_range_counts,
                                                  const uint32_t* query_label_indptr,
                                                  const uint32_t* query_label_indices,
                                                  const uint32_t* sample_label_indptr,
                                                  const uint32_t* sample_label_indices,
                                                  const uint8_t* query_range_modes,
                                                  const float* query_range_lower_bounds,
                                                  const float* query_range_upper_bounds,
                                                  const float* base_range_values,
                                                  uint32_t range_count,
                                                  uint32_t topk,
                                                  idx_t* compact_results,
                                                  float* compact_distances,
                                                  int* route_steps)
{
    extern __shared__ unsigned char shared_query_storage[];
    T* shared_query = reinterpret_cast<T*>(shared_query_storage);
    __shared__ float best_dist[kMaxTopk];
    __shared__ uint32_t best_ids[kMaxTopk];
    __shared__ uint32_t best_size;
    __shared__ uint32_t evaluated;
    __shared__ uint32_t query_squared_norm;

    const uint32_t lane = threadIdx.x & (warp_kernel::kWarpThreads - 1);
    const uint32_t vector_query_idx = query_vector_ids[route_idx];
    load_query_row_to_shared(query_vectors, vector_query_idx, dim, shared_query);
    __syncthreads();
    if (threadIdx.x < warp_kernel::kWarpThreads) {
        const uint32_t norm = query_squared_norm_device(shared_query, dim);
        if (lane == 0) {
            query_squared_norm = norm;
        }
    }
    if (threadIdx.x == 0) {
        best_size = 0;
        evaluated = 0;
        for (uint32_t i = 0; i < kMaxTopk; ++i) {
            best_dist[i] = FLT_MAX;
            best_ids[i] = UINT32_MAX;
        }
    }
    __syncthreads();

    const uint32_t group_id = query_group_ids[route_idx];
    if (group_id != UINT32_MAX && topk != 0 && topk <= kMaxTopk) {
        const uint32_t begin = low_label_offset[group_id];
        const uint32_t count = low_label_size[group_id];
        const uint64_t query_filter_bit = query_filter_bits[route_idx];
        const uint8_t query_filter_mode = query_filter_modes[route_idx];
        const uint32_t query_single_extra_label = query_single_extra_labels[route_idx];
        const uint32_t query_active_range_count = query_active_range_counts[route_idx];

        for (uint32_t j = 0; j < count; ++j) {
            uint32_t sample_id = UINT32_MAX;
            uint32_t should_eval = 0;
            if (threadIdx.x == 0) {
                const uint32_t row = begin + j;
                sample_id = low_sample_ids[row];
                if (sample_id < base_rows &&
                    candidate_passes_device(low_row_filter_mask[row],
                                            query_filter_bit,
                                            query_filter_mode,
                                            query_active_range_count,
                                            query_single_extra_label,
                                            query_range_modes,
                                            query_range_lower_bounds,
                                            query_range_upper_bounds,
                                            base_range_values,
                                            range_count,
                                            query_label_indptr,
                                            query_label_indices,
                                            sample_label_indptr,
                                            sample_label_indices,
                                            route_idx,
                                            sample_id)) {
                    should_eval = 1;
                }
            }
            sample_id = __shfl_sync(warp_kernel::kFullMask, sample_id, 0);
            should_eval = __shfl_sync(warp_kernel::kFullMask, should_eval, 0);
            if (threadIdx.x < warp_kernel::kWarpThreads && should_eval) {
                const float dist = l2_distance_device(base_vectors,
                                                      shared_query,
                                                      dim,
                                                      sample_id,
                                                      query_squared_norm);
                if (lane == 0) {
                    ++evaluated;
                    insert_best_result(dist, sample_id, topk, best_dist, best_ids, best_size);
                }
            }
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        const size_t result_base = static_cast<size_t>(route_idx) * topk;
        for (uint32_t i = 0; i < topk; ++i) {
            compact_results[result_base + i] = i < best_size ? static_cast<idx_t>(best_ids[i])
                                                             : static_cast<idx_t>(UINT32_MAX);
            compact_distances[result_base + i] = i < best_size ? best_dist[i] : FLT_MAX;
        }
        route_steps[route_idx] = static_cast<int>(evaluated);
    }
    __syncthreads();
}

template <int kQueueSize, typename T>
__global__ void persistent_latency_queue_kernel(const T* base_vectors,
                                                const T* query_vectors,
                                                uint32_t base_rows,
                                                uint32_t dim,
                                                const uint32_t* graph,
                                                uint32_t graph_degree,
                                                const PersistentLatencyRequestControl* requests,
                                                PersistentLatencyCompletionControl* completions,
                                                uint32_t slot_route_capacity,
                                                const uint32_t* route_indices,
                                                const uint32_t* high_query_indices,
                                                const uint32_t* low_query_indices,
                                                const uint32_t* query_vector_ids,
                                                const uint32_t* query_group_ids,
                                                const uint32_t* start_local_ids,
                                                const uint64_t* query_filter_bits,
                                                const uint8_t* query_filter_modes,
                                                const uint32_t* query_single_extra_labels,
                                                const uint32_t* query_active_range_counts,
                                                const uint32_t* query_label_indptr,
                                                const uint32_t* query_label_indices,
                                                const uint8_t* query_range_modes,
                                                const float* query_range_lower_bounds,
                                                const float* query_range_upper_bounds,
                                                const uint32_t* label_size,
                                                const uint32_t* label_offset,
                                                const uint32_t* row_to_sample,
                                                const uint64_t* row_filter_mask,
                                                const uint32_t* low_label_size,
                                                const uint32_t* low_label_offset,
                                                const uint32_t* low_sample_ids,
                                                const uint64_t* low_row_filter_mask,
                                                const uint32_t* sample_label_indptr,
                                                const uint32_t* sample_label_indices,
                                                const float* base_range_values,
                                                uint32_t range_count,
                                                uint32_t finish_count,
                                                int early_stop_mode,
                                                uint32_t max_steps,
                                                uint32_t topk,
                                                idx_t* compact_results,
                                                float* compact_distances,
                                                int* route_steps)
{
    const uint32_t slot_idx = blockIdx.x;
    const uint32_t slot_stride = slot_route_capacity + 1;
    const uint32_t slot_offset = slot_idx * slot_stride;
    const volatile PersistentLatencyRequestControl* request = requests + slot_idx;
    volatile PersistentLatencyCompletionControl* completion = completions + slot_idx;
    __shared__ uint32_t current_seq;
    __shared__ uint32_t route_count;
    __shared__ uint32_t high_route_count;
    __shared__ uint32_t low_route_count;
    __shared__ uint32_t should_stop;
    uint32_t last_seq = 0;

    while (true) {
        if (threadIdx.x == 0) {
            should_stop = 0;
            uint32_t seq = last_seq;
            while (seq == last_seq) {
                const uint32_t stop_value = load_bypass_l1_u32(&request->stop);
                if (stop_value != 0) {
                    should_stop = 1;
                    break;
                }
                seq = load_bypass_l1_u32(&request->ready_seq);
                if (seq == last_seq) {
                    __nanosleep(256);
                }
            }
            current_seq = seq;
            route_count = load_bypass_l1_u32(&request->route_count);
            high_route_count = load_bypass_l1_u32(&request->high_route_count);
            low_route_count = load_bypass_l1_u32(&request->low_route_count);
            if (should_stop == 0) {
                completion->gpu_start_clock = clock64();
            }
        }
        __syncthreads();

        if (should_stop != 0) {
            return;
        }
        last_seq = current_seq;

        for (uint32_t idx = threadIdx.x; idx < route_count * topk; idx += blockDim.x) {
            const uint32_t local_route = idx / topk;
            const uint32_t local_topk = idx % topk;
            const uint32_t route_idx = route_indices[slot_offset + local_route];
            const size_t flat = static_cast<size_t>(route_idx) * topk + local_topk;
            compact_results[flat] = static_cast<idx_t>(UINT32_MAX);
            compact_distances[flat] = FLT_MAX;
        }
        for (uint32_t idx = threadIdx.x; idx < route_count; idx += blockDim.x) {
            route_steps[route_indices[slot_offset + idx]] = 0;
        }
        __syncthreads();

        for (uint32_t i = 0; i < high_route_count; ++i) {
            const uint32_t route_idx = high_query_indices[slot_offset + i];
            run_full_gpu_highfreq_route<kQueueSize>(base_vectors,
                                                    query_vectors,
                                                    base_rows,
                                                    dim,
                                                    graph,
                                                    graph_degree,
                                                    graph_degree,
                                                    route_idx,
                                                    query_vector_ids,
                                                    query_group_ids,
                                                    start_local_ids,
                                                    label_size,
                                                    label_offset,
                                                    row_to_sample,
                                                    row_filter_mask,
                                                    query_filter_bits,
                                                    query_filter_modes,
                                                    query_single_extra_labels,
                                                    query_active_range_counts,
                                                    query_label_indptr,
                                                    query_label_indices,
                                                    sample_label_indptr,
                                                    sample_label_indices,
                                                    query_range_modes,
                                                    query_range_lower_bounds,
                                                    query_range_upper_bounds,
                                                    base_range_values,
                                                    range_count,
                                                    finish_count,
                                                    early_stop_mode,
                                                    max_steps,
                                                    topk,
                                                    compact_results,
                                                    compact_distances,
                                                    route_steps,
                                                    false);
        }
        for (uint32_t i = 0; i < low_route_count; ++i) {
            const uint32_t route_idx = low_query_indices[slot_offset + i];
            persistent_latency_scan_low_route(base_vectors,
                                              query_vectors,
                                              base_rows,
                                              dim,
                                              route_idx,
                                              query_vector_ids,
                                              query_group_ids,
                                              query_filter_bits,
                                              low_label_size,
                                              low_label_offset,
                                              low_sample_ids,
                                              low_row_filter_mask,
                                              query_filter_modes,
                                              query_single_extra_labels,
                                              query_active_range_counts,
                                              query_label_indptr,
                                              query_label_indices,
                                              sample_label_indptr,
                                              sample_label_indices,
                                              query_range_modes,
                                              query_range_lower_bounds,
                                              query_range_upper_bounds,
                                              base_range_values,
                                              range_count,
                                              topk,
                                              compact_results,
                                              compact_distances,
                                              route_steps);
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            completion->gpu_stop_clock = clock64();
            __threadfence_system();
            completion->done_seq = current_seq;
        }
        __syncthreads();
    }
}

__global__ void init_active_route_outputs_kernel(const uint32_t* active_query_ids,
                                                 uint32_t active_count,
                                                 idx_t* route_results,
                                                 float* route_distances,
                                                 int* route_steps,
                                                 uint32_t topk)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = active_count * topk;
    if (idx < total) {
        const uint32_t route_idx = active_query_ids[idx / topk];
        const uint32_t local_topk = idx % topk;
        const uint32_t flat = route_idx * topk + local_topk;
        route_results[flat] = static_cast<idx_t>(UINT32_MAX);
        route_distances[flat] = FLT_MAX;
    }
    if (idx < active_count) {
        route_steps[active_query_ids[idx]] = 0;
    }
}

__global__ void gather_active_route_results_kernel(const uint32_t* active_query_ids,
                                                   uint32_t active_count,
                                                   const idx_t* route_results,
                                                   const float* route_distances,
                                                   uint32_t topk,
                                                   idx_t* compact_results,
                                                   float* compact_distances)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = active_count * topk;
    if (idx >= total) {
        return;
    }
    const uint32_t route_idx = active_query_ids[idx / topk];
    const uint32_t local_topk = idx % topk;
    const uint32_t source = route_idx * topk + local_topk;
    compact_results[idx] = route_results[source];
    compact_distances[idx] = route_distances[source];
}

void save_results_to_csv(const std::vector<std::vector<idx_t>>& results,
                         const std::string& filename)
{
    std::ofstream file(filename);
    if (!file.is_open()) {
        fail_full_gpu("could not open output file " + filename);
    }

    for (const auto& row : results) {
        for (size_t i = 0; i < row.size(); ++i) {
            file << row[i];
            if (i + 1 < row.size()) {
                file << ',';
            }
        }
        file << '\n';
    }
}

void validate_runtime_for_full_gpu(const GraphRuntime& runtime,
                                   const FullGpuSearchOptions& options)
{
    if (runtime.graph.data.empty()) {
        fail_full_gpu("full_gpu requires a materialized graph payload after loading");
    }
    if (runtime.vector_scalar_type == DenseVectorScalarType::Uint8) {
        if (runtime.base_vectors.data.empty() || runtime.query_vectors.data.empty()) {
            fail_full_gpu("full_gpu requires materialized uint8 base/query vector payloads after loading");
        }
    } else if (dense_vector_scalar_is_float(runtime.vector_scalar_type)) {
        if (runtime.base_float_vectors.data.empty() || runtime.query_float_vectors.data.empty()) {
            fail_full_gpu("full_gpu requires materialized float base/query vector payloads after loading");
        }
    } else {
        fail_full_gpu("unsupported dense vector scalar type");
    }
    if (options.use_fp16 && runtime.vector_scalar_type != DenseVectorScalarType::Float32) {
        fail_full_gpu("use_fp16 requires FP32 .fbin/.fvecs vectors with matching query vectors");
    }
    if (!runtime.low_query_indices.empty() &&
        (runtime.lowfreq_index.sample_ids.empty() || runtime.lowfreq_index.row_filter_mask.empty())) {
        fail_full_gpu("low-frequency routes require a lowfreq posting index");
    }
    if (runtime.graph.cols > static_cast<uint64_t>(options.degree)) {
        fail_full_gpu("graph degree exceeds configured search degree");
    }
    if (runtime.query_ranges.range_count != runtime.base_range_count && !runtime.base_range_values.empty()) {
        fail_full_gpu("query/base range dimension mismatch");
    }
}

GraphRuntime load_full_gpu_runtime(const FullGpuSearchOptions& options)
{
    GraphLoadConfig load_config;
    load_config.graph_path = options.graph_path;
    load_config.graph_storage_mode = "memory";
    load_config.lowfreq_index_path = options.lowfreq_index_path;
    load_config.data_label_spmat_path = options.data_label_spmat_path;
    load_config.query_label_spmat_path = options.query_label_spmat_path;
    load_config.query_manifest_jsonl_path = options.query_manifest_jsonl_path;
    load_config.query_manifest_families = options.query_manifest_families;
    load_config.query_manifest_subtypes = options.query_manifest_subtypes;
    load_config.query_manifest_max_or_terms = options.query_manifest_max_or_terms;
    load_config.query_manifest_max_fuzzy_terms = options.query_manifest_max_fuzzy_terms;
    load_config.query_manifest_min_ngram_score = options.query_manifest_min_ngram_score;
    load_config.query_manifest_ngram_size = options.query_manifest_ngram_size;
    load_config.base_vector_path = options.base_vector_path;
    load_config.base_range_fbin_path = options.base_range_fbin_path;
    load_config.query_vector_path = options.query_vector_path;
    load_config.query_range_path = options.query_range_path;
    load_config.range_label_manifest_path = options.range_label_manifest_path;
    load_config.seed_sidecar_path = options.seed_sidecar_path;
    load_config.original_label_seed_sidecar_path = options.original_label_seed_sidecar_path;
    load_config.metadata_sidecar_path = options.metadata_sidecar_path;
    load_config.seed_selection_mode = options.seed_selection_mode;
    load_config.group_path = options.group_path;
    load_config.routing_mode = options.routing_mode;
    load_config.and_dual_route_posting_ratio_threshold =
        options.and_dual_route_posting_ratio_threshold;
    load_config.dataset_cutoff = options.dataset_cutoff;
    load_config.specificity_threshold = options.specificity_threshold;
    if (range_posting::parse_mode(options.range_posting_mode) != range_posting::Mode::Off) {
        load_config.supported_query_route_providers |= kQueryRouteProviderRangePosting;
        load_config.range_posting_index_path = options.range_posting_index_path;
        load_config.range_access_planner.mode = options.range_posting_mode;
        load_config.range_access_planner.scan_threshold = options.range_posting_scan_threshold;
        load_config.range_access_planner.and_filter_mode = options.range_posting_and_filter_mode;
        load_config.range_access_planner.range_label_and_policy = options.range_label_and_policy;
        load_config.range_access_planner.drop_wide_hybrid_or = options.range_posting_drop_wide_hybrid_or;
        load_config.range_access_planner.range_graph_hybrid_or_enabled = options.range_graph_hybrid_or_enabled;
        load_config.range_access_planner.range_graph_hybrid_and_enabled = options.range_graph_hybrid_and_enabled;
    }
    load_config.prefetch_hops = 0;
    load_config.dim = options.dim;
    load_config.degree = options.degree;
    load_config.allow_float_vectors = true;


    load_config.highfreq_prefetch_with_vector_payload = false;
    load_config.load_data_labels_for_filtering = true;
    load_config.vector_consumer = "full_gpu search";
    return load_graph_runtime(load_config);
}

void upload_runtime(const GraphRuntime& runtime,
                    const UInt32Csr& query_label_csr,
                    const QueryFilterFastPath& query_filter_fast_path,
                    const UInt32Csr& sample_label_csr,
                    bool use_original_query_vectors,
                    bool use_fp16,
                    FullGpuDeviceRuntime& device)
{
    if (runtime.vector_scalar_type == DenseVectorScalarType::Uint8) {
        device.base_u8.upload(runtime.base_vectors.data, "cudaMemcpy full_gpu base_u8");
        const std::vector<uint8_t>& query_data = use_original_query_vectors
            ? runtime.original_query_vectors.data
            : runtime.query_vectors.data;
        device.query_u8.upload(query_data, "cudaMemcpy full_gpu query_u8");
    } else if (use_fp16) {
        if (runtime.vector_scalar_type == DenseVectorScalarType::Float16) {
            fail_full_gpu("full_gpu FP16 device payload requires FP32 host vectors");
        }
        std::vector<__half> base_half(runtime.base_float_vectors.data.size());
        std::transform(runtime.base_float_vectors.data.begin(),
                       runtime.base_float_vectors.data.end(),
                       base_half.begin(),
                       [] (float value) { return __float2half(value); });
        const std::vector<float>& query_data = use_original_query_vectors
            ? runtime.original_query_float_vectors.data
            : runtime.query_float_vectors.data;
        std::vector<__half> query_half(query_data.size());
        std::transform(query_data.begin(), query_data.end(), query_half.begin(),
                       [] (float value) { return __float2half(value); });
        device.base_f16.upload(base_half, "cudaMemcpy full_gpu base_f16");
        device.query_f16.upload(query_half, "cudaMemcpy full_gpu query_f16");
    } else {
        device.base_f32.upload(runtime.base_float_vectors.data, "cudaMemcpy full_gpu base_f32");
        const std::vector<float>& query_data = use_original_query_vectors
            ? runtime.original_query_float_vectors.data
            : runtime.query_float_vectors.data;
        device.query_f32.upload(query_data, "cudaMemcpy full_gpu query_f32");
    }

    device.graph.upload(runtime.graph.data, "cudaMemcpy full_gpu graph");
    device.label_size.upload(runtime.metadata.label_size, "cudaMemcpy full_gpu label_size");
    device.label_offset.upload(runtime.metadata.label_offset, "cudaMemcpy full_gpu label_offset");
    device.row_to_sample.upload(runtime.metadata.row_to_sample, "cudaMemcpy full_gpu row_to_sample");
    device.row_filter_mask.upload(runtime.metadata.row_filter_mask, "cudaMemcpy full_gpu row_filter_mask");
    device.query_group_ids.upload(runtime.query_group_ids, "cudaMemcpy full_gpu query_group_ids");
    device.start_local_ids.upload(runtime.start_local_ids, "cudaMemcpy full_gpu start_local_ids");
    device.query_filter_bits.upload(runtime.query_filter_bits, "cudaMemcpy full_gpu query_filter_bits");
    device.query_filter_modes.upload(query_filter_fast_path.modes,
                                     "cudaMemcpy full_gpu query_filter_modes");
    device.query_single_extra_labels.upload(query_filter_fast_path.single_extra_labels,
                                            "cudaMemcpy full_gpu query_single_extra_labels");
    device.query_active_range_counts.upload(query_filter_fast_path.active_range_counts,
                                            "cudaMemcpy full_gpu query_active_range_counts");
    device.query_label_indptr.upload(query_label_csr.indptr, "cudaMemcpy full_gpu query_label_indptr");
    device.query_label_indices.upload(query_label_csr.indices, "cudaMemcpy full_gpu query_label_indices");
    device.sample_label_indptr.upload(sample_label_csr.indptr, "cudaMemcpy full_gpu sample_label_indptr");
    device.sample_label_indices.upload(sample_label_csr.indices, "cudaMemcpy full_gpu sample_label_indices");
    device.query_range_modes.upload(runtime.query_ranges.modes, "cudaMemcpy full_gpu query_range_modes");
    device.query_range_lower_bounds.upload(runtime.query_ranges.lower_bounds,
                                           "cudaMemcpy full_gpu query_range_lower_bounds");
    device.query_range_upper_bounds.upload(runtime.query_ranges.upper_bounds,
                                           "cudaMemcpy full_gpu query_range_upper_bounds");
    device.base_range_values.upload(runtime.base_range_values, "cudaMemcpy full_gpu base_range_values");
    const std::vector<uint32_t> posting_ids = materialize_range_posting_ids(runtime);
    device.range_posting_sample_ids.upload(posting_ids,
                                            "cudaMemcpy full_gpu range posting sample ids");
    std::vector<uint64_t> posting_begins = runtime.route_range_posting_begins;
    if (runtime.range_posting_index != nullptr) {
        const uint64_t rows = runtime.range_posting_index->row_count();
        for (size_t r = 0; r < posting_begins.size(); ++r) {
            const uint32_t dim = r < runtime.route_range_posting_indices.size()
                ? runtime.route_range_posting_indices[r] : UINT32_MAX;
            if (dim != UINT32_MAX) posting_begins[r] += static_cast<uint64_t>(dim) * rows;
        }
    }
    device.route_range_posting_begins.upload(posting_begins,
                                              "cudaMemcpy full_gpu route posting begins");
    device.route_range_posting_sizes.upload(runtime.route_range_posting_sizes,
                                             "cudaMemcpy full_gpu route posting sizes");
    device.high_query_indices.upload(runtime.high_query_indices, "cudaMemcpy full_gpu high_query_indices");
    device.low_query_indices.upload(runtime.low_query_indices, "cudaMemcpy full_gpu low_query_indices");
    device.low_label_size.upload(runtime.lowfreq_index.label_size, "cudaMemcpy full_gpu low_label_size");
    device.low_label_offset.upload(runtime.lowfreq_index.label_offset, "cudaMemcpy full_gpu low_label_offset");
    device.low_sample_ids.upload(runtime.lowfreq_index.sample_ids, "cudaMemcpy full_gpu low_sample_ids");
    device.low_row_filter_mask.upload(runtime.lowfreq_index.row_filter_mask,
                                      "cudaMemcpy full_gpu low_row_filter_mask");
}

uint32_t runtime_route_count(const GraphRuntime& runtime)
{
    if (runtime.vector_scalar_type == DenseVectorScalarType::Uint8) {
        return runtime.query_vectors.rows;
    }
    if (dense_vector_scalar_is_float(runtime.vector_scalar_type)) {
        return runtime.query_float_vectors.rows;
    }
    return 0;
}

uint32_t runtime_base_row_count(const GraphRuntime& runtime)
{
    if (runtime.vector_scalar_type == DenseVectorScalarType::Uint8) {
        return runtime.base_vectors.rows;
    }
    if (dense_vector_scalar_is_float(runtime.vector_scalar_type)) {
        return runtime.base_float_vectors.rows;
    }
    return 0;
}

std::vector<uint32_t> materialize_range_posting_ids(const GraphRuntime& runtime)
{
    if (runtime.range_posting_index == nullptr) return {};
    const auto& index = *runtime.range_posting_index;
    const size_t per_dim = static_cast<size_t>(index.row_count());
    std::vector<uint32_t> ids;
    ids.reserve(per_dim * index.range_count());
    for (const auto& dim : index.dimensions()) {
        if (dim.row_count != per_dim || dim.sample_ids == nullptr) {
            fail_full_gpu("range posting dimension is not row-aligned");
        }
        ids.insert(ids.end(), dim.sample_ids, dim.sample_ids + per_dim);
    }
    return ids;
}

uint32_t max_query_label_nnz_per_route(const SparseLabelMatrix& matrix)
{
    uint64_t max_nnz = 0;
    for (uint32_t row = 0; row < matrix.rows; ++row) {
        if (static_cast<size_t>(row) + 1 >= matrix.indptr.size()) {
            fail_full_gpu("query label indptr is smaller than query label row count");
        }
        const uint64_t row_nnz = matrix.indptr[static_cast<size_t>(row) + 1] -
                                 matrix.indptr[row];
        max_nnz = std::max(max_nnz, row_nnz);
    }
    if (max_nnz > std::numeric_limits<uint32_t>::max()) {
        fail_full_gpu("query label row nnz exceeds uint32 range");
    }
    return static_cast<uint32_t>(max_nnz);
}

void init_all_route_outputs(FullGpuDeviceRuntime& device,
                            uint32_t route_count,
                            uint32_t topk,
                            cudaStream_t stream = 0)
{
    if (route_count == 0) {
        return;
    }
    const uint32_t init_threads = 256;
    const uint32_t init_blocks = (route_count * topk + init_threads - 1) / init_threads;
    init_route_outputs_kernel<<<std::max(1u, init_blocks), init_threads, 0, stream>>>(
        device.route_results.data(),
        device.route_distances.data(),
        device.route_steps.data(),
        route_count,
        topk);
    check_cuda(cudaGetLastError(), "launch full_gpu init_route_outputs_kernel");
}

void init_active_route_outputs(FullGpuDeviceRuntime& device,
                               const uint32_t* active_query_indices,
                               uint32_t active_count,
                               uint32_t topk,
                               cudaStream_t stream = 0)
{
    if (active_count == 0) {
        return;
    }
    const uint32_t init_threads = 256;
    const uint32_t total = active_count * topk;
    const uint32_t init_blocks = (total + init_threads - 1) / init_threads;
    init_active_route_outputs_kernel<<<std::max(1u, init_blocks), init_threads, 0, stream>>>(
        active_query_indices,
        active_count,
        device.route_results.data(),
        device.route_distances.data(),
        device.route_steps.data(),
        topk);
    check_cuda(cudaGetLastError(), "launch full_gpu init_active_route_outputs_kernel");
}

void gather_active_route_results(FullGpuDeviceRuntime& device,
                                 const uint32_t* active_query_indices,
                                 uint32_t active_count,
                                 uint32_t topk,
                                 idx_t* compact_results,
                                 float* compact_distances,
                                 cudaStream_t stream = 0)
{
    if (active_count == 0) {
        return;
    }
    const uint32_t gather_threads = 256;
    const uint32_t total = active_count * topk;
    const uint32_t gather_blocks = (total + gather_threads - 1) / gather_threads;
    gather_active_route_results_kernel<<<std::max(1u, gather_blocks), gather_threads, 0, stream>>>(
        active_query_indices,
        active_count,
        device.route_results.data(),
        device.route_distances.data(),
        topk,
        compact_results,
        compact_distances);
    check_cuda(cudaGetLastError(), "launch full_gpu gather_active_route_results_kernel");
}

FullGpuQueryDeviceView default_query_device_view(const FullGpuDeviceRuntime& device)
{
    FullGpuQueryDeviceView view;
    view.query_group_ids = device.query_group_ids.data();
    view.start_local_ids = device.start_local_ids.data();
    view.query_filter_bits = device.query_filter_bits.data();
    view.query_filter_modes = device.query_filter_modes.data();
    view.query_single_extra_labels = device.query_single_extra_labels.data();
    view.query_active_range_counts = device.query_active_range_counts.data();
    view.query_label_indptr = device.query_label_indptr.data();
    view.query_label_indices = device.query_label_indices.data();
    view.query_range_modes = device.query_range_modes.data();
    view.query_range_lower_bounds = device.query_range_lower_bounds.data();
    view.query_range_upper_bounds = device.query_range_upper_bounds.data();
    return view;
}

#include "search/full_gpu/full_gpu_overlap_search.cuh"

void launch_search_kernels_for_routes(const GraphRuntime& runtime,
                                      const FullGpuSearchOptions& options,
                                      FullGpuDeviceRuntime& device,
                                      const FullGpuQueryDeviceView& query_view,
                                      const uint32_t* high_query_indices,
                                      uint32_t high_query_count,
                                      const uint32_t* low_query_indices,
                                      uint32_t low_query_count,
                                      cudaStream_t stream = 0)
{
    const uint32_t topk = static_cast<uint32_t>(options.topk);
    const uint32_t base_rows = runtime_base_row_count(runtime);
    const uint32_t dim = static_cast<uint32_t>(options.dim);
    const uint32_t graph_degree = static_cast<uint32_t>(runtime.graph.cols);
    // The maintained route-local path always evaluates the complete stored row.
    const uint32_t traversal_degree = graph_degree;
    const uint32_t range_count = runtime.query_ranges.range_count;
    const uint32_t finish_count = static_cast<uint32_t>(std::max(1, options.search_finish_count));
    const uint32_t max_steps = static_cast<uint32_t>(std::max(0, options.search_max_steps));
    const int early_stop_mode = parse_search_early_stop_mode(options.search_early_stop_mode);
    const float* base_ranges = device.base_range_values.size() == 0 ? nullptr : device.base_range_values.data();
    const bool use_packed_visited =
        std::all_of(runtime.metadata.label_size.begin(),
                    runtime.metadata.label_size.end(),
                    [](uint32_t size) { return size <= kPackedVisitedEmpty; });
    dispatch_full_gpu_kernel_by_queue_size(options.search_queue_size, [&]<int kQueueSize>() {
        if (high_query_count != 0) {
            const uint32_t active_count = high_query_count;
            const uint32_t* active_ids = high_query_indices;
            auto launch_highfreq = [&]<typename T, bool kUsePackedVisited>(const T* base_vectors,
                                                                           const T* query_vectors) {
                const size_t shared_bytes = static_cast<size_t>(dim) * sizeof(T);

                    full_gpu_highfreq_kernel<kQueueSize, T, kUsePackedVisited>
                        <<<active_count, kHighfreqBlockThreads, shared_bytes, stream>>>(
                        base_vectors,
                        query_vectors,
                        base_rows,
                        dim,
                        device.graph.data(),
                        graph_degree,
                        traversal_degree,
                        active_ids,
                        active_count,
                        query_view.query_vector_ids,
                        query_view.query_group_ids,
                        query_view.start_local_ids,
                        device.label_size.data(),
                        device.label_offset.data(),
                        device.row_to_sample.data(),
                        device.row_filter_mask.data(),
                        query_view.query_filter_bits,
                        query_view.query_filter_modes,
                        query_view.query_single_extra_labels,
                        query_view.query_active_range_counts,
                        query_view.query_label_indptr,
                        query_view.query_label_indices,
                        device.sample_label_indptr.data(),
                        device.sample_label_indices.data(),
                        query_view.query_range_modes,
                        query_view.query_range_lower_bounds,
                        query_view.query_range_upper_bounds,
                        base_ranges,
                        range_count,
                        finish_count,
                        early_stop_mode,
                        max_steps,
                        topk,
                        device.route_results.data(),
                        device.route_distances.data(),
                        device.route_steps.data());

            };
            if (runtime.vector_scalar_type == DenseVectorScalarType::Uint8) {
                if (use_packed_visited) {
                    launch_highfreq.template operator()<uint8_t, true>(device.base_u8.data(),
                                                                        device.query_u8.data());
                } else {
                    launch_highfreq.template operator()<uint8_t, false>(device.base_u8.data(),
                                                                         device.query_u8.data());
                }
            } else if (options.use_fp16 && use_packed_visited) {
                launch_highfreq.template operator()<__half, true>(device.base_f16.data(),
                                                                    device.query_f16.data());
            } else if (options.use_fp16) {
                launch_highfreq.template operator()<__half, false>(device.base_f16.data(),
                                                                     device.query_f16.data());
            } else if (use_packed_visited) {
                launch_highfreq.template operator()<float, true>(device.base_f32.data(),
                                                                  device.query_f32.data());
            } else {
                launch_highfreq.template operator()<float, false>(device.base_f32.data(),
                                                                   device.query_f32.data());
            }
            check_cuda(cudaGetLastError(), "launch full_gpu_highfreq_kernel");
        }
    });

    if (low_query_count != 0) {
        const uint32_t active_count = low_query_count;
        if (runtime.vector_scalar_type == DenseVectorScalarType::Uint8) {
            const size_t shared_bytes = static_cast<size_t>(dim) * sizeof(uint8_t);
            full_gpu_lowfreq_kernel<<<active_count, warp_kernel::kWarpThreads, shared_bytes, stream>>>(
                device.base_u8.data(),
                device.query_u8.data(),
                base_rows,
                dim,
                low_query_indices,
                active_count,
                query_view.query_vector_ids,
                query_view.query_group_ids,
                query_view.query_filter_bits,
                device.low_label_size.data(),
                device.low_label_offset.data(),
                device.low_sample_ids.data(),
                device.low_row_filter_mask.data(),
                query_view.query_filter_modes,
                query_view.query_single_extra_labels,
                query_view.query_active_range_counts,
                query_view.query_label_indptr,
                query_view.query_label_indices,
                device.sample_label_indptr.data(),
                device.sample_label_indices.data(),
                query_view.query_range_modes,
                query_view.query_range_lower_bounds,
                query_view.query_range_upper_bounds,
                base_ranges,
                range_count,
                topk,
                device.route_results.data(),
                device.route_distances.data(),
                device.route_steps.data());
        } else if (options.use_fp16) {
            const size_t shared_bytes = static_cast<size_t>(dim) * sizeof(__half);
            full_gpu_lowfreq_kernel<<<active_count, warp_kernel::kWarpThreads, shared_bytes, stream>>>(
                device.base_f16.data(),
                device.query_f16.data(),
                base_rows,
                dim,
                low_query_indices,
                active_count,
                query_view.query_vector_ids,
                query_view.query_group_ids,
                query_view.query_filter_bits,
                device.low_label_size.data(),
                device.low_label_offset.data(),
                device.low_sample_ids.data(),
                device.low_row_filter_mask.data(),
                query_view.query_filter_modes,
                query_view.query_single_extra_labels,
                query_view.query_active_range_counts,
                query_view.query_label_indptr,
                query_view.query_label_indices,
                device.sample_label_indptr.data(),
                device.sample_label_indices.data(),
                query_view.query_range_modes,
                query_view.query_range_lower_bounds,
                query_view.query_range_upper_bounds,
                base_ranges,
                range_count,
                topk,
                device.route_results.data(),
                device.route_distances.data(),
                device.route_steps.data());
        } else {
            const size_t shared_bytes = static_cast<size_t>(dim) * sizeof(float);
            full_gpu_lowfreq_kernel<<<active_count, warp_kernel::kWarpThreads, shared_bytes, stream>>>(
                device.base_f32.data(),
                device.query_f32.data(),
                base_rows,
                dim,
                low_query_indices,
                active_count,
                query_view.query_vector_ids,
                query_view.query_group_ids,
                query_view.query_filter_bits,
                device.low_label_size.data(),
                device.low_label_offset.data(),
                device.low_sample_ids.data(),
                device.low_row_filter_mask.data(),
                query_view.query_filter_modes,
                query_view.query_single_extra_labels,
                query_view.query_active_range_counts,
                query_view.query_label_indptr,
                query_view.query_label_indices,
                device.sample_label_indptr.data(),
                device.sample_label_indices.data(),
                query_view.query_range_modes,
                query_view.query_range_lower_bounds,
                query_view.query_range_upper_bounds,
                base_ranges,
                range_count,
                topk,
                device.route_results.data(),
                device.route_distances.data(),
                device.route_steps.data());
        }
        check_cuda(cudaGetLastError(), "launch full_gpu_lowfreq_kernel");
    }
    if (device.overlap) {
        launch_full_gpu_overlap(runtime, options, device);
    }
}

struct FullGpuIterationTimingEvents {
    cudaEvent_t graph_start = nullptr;
    cudaEvent_t graph_stop = nullptr;
    cudaEvent_t posting_start = nullptr;
    cudaEvent_t posting_stop = nullptr;
};

void launch_search_iteration(const GraphRuntime& runtime,
                             const FullGpuSearchOptions& options,
                             FullGpuDeviceRuntime& device,
                             const FullGpuIterationTimingEvents* timing = nullptr)
{
    const uint32_t route_count = runtime_route_count(runtime);
    const uint32_t topk = static_cast<uint32_t>(options.topk);
    init_all_route_outputs(device, route_count, topk);
    const FullGpuQueryDeviceView query_view = default_query_device_view(device);
    if (timing != nullptr) {
        check_cuda(cudaEventRecord(timing->graph_start), "cudaEventRecord full_gpu graph start");
    }
    launch_search_kernels_for_routes(runtime,
                                     options,
                                     device,
                                     query_view,
                                     device.high_query_indices.data(),
                                     static_cast<uint32_t>(runtime.high_query_indices.size()),
                                     device.low_query_indices.data(),
                                     static_cast<uint32_t>(runtime.low_query_indices.size()));
    if (timing != nullptr) {
        check_cuda(cudaEventRecord(timing->graph_stop), "cudaEventRecord full_gpu graph stop");
        check_cuda(cudaEventRecord(timing->posting_start), "cudaEventRecord full_gpu posting start");
    }
    if (runtime.range_posting_index != nullptr &&
        device.range_posting_route_ids.size() != 0) {
        const uint32_t posting_routes = static_cast<uint32_t>(device.range_posting_route_ids.size());
        if (posting_routes != 0) {
            const uint32_t blocks = posting_routes;
            if (runtime.vector_scalar_type == DenseVectorScalarType::Uint8) {
                full_gpu_range_posting_kernel<<<blocks, warp_kernel::kWarpThreads>>>(
                    device.base_u8.data(), device.query_u8.data(),
                    runtime_base_row_count(runtime), static_cast<uint32_t>(options.dim),
                    device.range_posting_route_ids.data(), nullptr,
                    device.route_range_posting_begins.data(), device.route_range_posting_sizes.data(),
                    device.range_posting_sample_ids.data(), device.query_label_indptr.data(),
                    device.query_label_indices.data(), device.sample_label_indptr.data(),
                    device.sample_label_indices.data(), device.query_range_modes.data(),
                    device.query_range_lower_bounds.data(), device.query_range_upper_bounds.data(),
                    device.base_range_values.data(), runtime.query_ranges.range_count,
                    static_cast<uint32_t>(options.topk), posting_routes,
                    device.route_results.data(), device.route_distances.data(), device.route_steps.data());
            } else if (options.use_fp16) {
                full_gpu_range_posting_kernel<<<blocks, warp_kernel::kWarpThreads>>>(
                    device.base_f16.data(), device.query_f16.data(),
                    runtime_base_row_count(runtime), static_cast<uint32_t>(options.dim),
                    device.range_posting_route_ids.data(), nullptr,
                    device.route_range_posting_begins.data(), device.route_range_posting_sizes.data(),
                    device.range_posting_sample_ids.data(), device.query_label_indptr.data(),
                    device.query_label_indices.data(), device.sample_label_indptr.data(),
                    device.sample_label_indices.data(), device.query_range_modes.data(),
                    device.query_range_lower_bounds.data(), device.query_range_upper_bounds.data(),
                    device.base_range_values.data(), runtime.query_ranges.range_count,
                    static_cast<uint32_t>(options.topk), posting_routes,
                    device.route_results.data(), device.route_distances.data(), device.route_steps.data());
            } else {
                full_gpu_range_posting_kernel<<<blocks, warp_kernel::kWarpThreads>>>(
                    device.base_f32.data(), device.query_f32.data(),
                    runtime_base_row_count(runtime), static_cast<uint32_t>(options.dim),
                    device.range_posting_route_ids.data(), nullptr,
                    device.route_range_posting_begins.data(), device.route_range_posting_sizes.data(),
                    device.range_posting_sample_ids.data(), device.query_label_indptr.data(),
                    device.query_label_indices.data(), device.sample_label_indptr.data(),
                    device.sample_label_indices.data(), device.query_range_modes.data(),
                    device.query_range_lower_bounds.data(), device.query_range_upper_bounds.data(),
                    device.base_range_values.data(), runtime.query_ranges.range_count,
                    static_cast<uint32_t>(options.topk), posting_routes,
                    device.route_results.data(), device.route_distances.data(), device.route_steps.data());
            }
            check_cuda(cudaGetLastError(), "launch full_gpu_range_posting_kernel");
        }
    }
    if (timing != nullptr) {
        check_cuda(cudaEventRecord(timing->posting_stop), "cudaEventRecord full_gpu posting stop");
    }
}

void launch_latency_iteration(const GraphRuntime& runtime,
                              const FullGpuSearchOptions& options,
                              FullGpuDeviceRuntime& device,
                              const FullGpuQueryDeviceView& query_view,
                              uint32_t active_route_count,
                              uint32_t high_query_count,
                              uint32_t low_query_count,
                              cudaStream_t stream = 0,
                              const uint32_t* active_route_indices = nullptr,
                              const uint32_t* high_query_indices = nullptr,
                              const uint32_t* low_query_indices = nullptr,
                              idx_t* compact_results = nullptr,
                              float* compact_distances = nullptr)
{
    const uint32_t topk = static_cast<uint32_t>(options.topk);
    if (active_route_indices == nullptr) {
        active_route_indices = device.latency_route_indices.data();
    }
    if (high_query_indices == nullptr) {
        high_query_indices = device.latency_high_query_indices.data();
    }
    if (low_query_indices == nullptr) {
        low_query_indices = device.latency_low_query_indices.data();
    }
    if (compact_results == nullptr) {
        compact_results = device.latency_results.data();
    }
    if (compact_distances == nullptr) {
        compact_distances = device.latency_distances.data();
    }
    init_active_route_outputs(device,
                              active_route_indices,
                              active_route_count,
                              topk,
                              stream);
    launch_search_kernels_for_routes(runtime,
                                     options,
                                     device,
                                     query_view,
                                     high_query_indices,
                                     high_query_count,
                                     low_query_indices,
                                     low_query_count,
                                     stream);
    gather_active_route_results(device,
                                active_route_indices,
                                active_route_count,
                                topk,
                                compact_results,
                                compact_distances,
                                stream);
}

std::vector<std::vector<idx_t>> merge_route_results_to_original(const GraphRuntime& runtime,
                                                                const std::vector<idx_t>& flat_ids,
                                                                const std::vector<float>& flat_distances,
                                                                uint32_t topk)
{
    std::vector<std::vector<idx_t>> merged(runtime.original_query_count,
                                           std::vector<idx_t>(topk, static_cast<idx_t>(UINT32_MAX)));
    std::vector<std::unordered_map<idx_t, float>> candidates(runtime.original_query_count);

    const size_t route_count = runtime_route_count(runtime);
    for (size_t route_idx = 0; route_idx < route_count; ++route_idx) {
        if (route_idx >= runtime.route_to_original_query.size()) {
            continue;
        }
        const uint32_t original_query = runtime.route_to_original_query[route_idx];
        if (original_query >= runtime.original_query_count) {
            continue;
        }
        auto& candidate_map = candidates[original_query];
        for (uint32_t j = 0; j < topk; ++j) {
            const size_t flat = route_idx * topk + j;
            const idx_t sample_id = flat_ids[flat];
            if (sample_id == static_cast<idx_t>(UINT32_MAX)) {
                continue;
            }
            const float dist = flat_distances[flat];
            auto it = candidate_map.find(sample_id);
            if (it == candidate_map.end() || dist < it->second) {
                candidate_map[sample_id] = dist;
            }
        }
    }

    for (uint32_t query_idx = 0; query_idx < runtime.original_query_count; ++query_idx) {
        std::vector<std::pair<float, idx_t>> scored;
        scored.reserve(candidates[query_idx].size());
        for (const auto& entry : candidates[query_idx]) {
            scored.emplace_back(entry.second, entry.first);
        }
        std::sort(scored.begin(), scored.end(), [](const auto& lhs, const auto& rhs) {
            if (lhs.first != rhs.first) {
                return lhs.first < rhs.first;
            }
            return lhs.second < rhs.second;
        });
        for (uint32_t i = 0; i < topk && i < scored.size(); ++i) {
            merged[query_idx][i] = scored[i].second;
        }
    }
    return merged;
}

struct LatencyBatchRoutes {
    uint32_t original_count = 0;
    uint32_t unroutable_original_count = 0;
    std::vector<uint32_t> original_query_ids;
    std::vector<uint32_t> route_ids;
    std::vector<uint32_t> high_route_ids;
    std::vector<uint32_t> low_route_ids;
};

struct LatencyOnlineMetadata {
    std::vector<uint32_t> local_route_ids;
    std::vector<uint32_t> query_vector_ids;
    std::vector<uint32_t> query_group_ids;
    std::vector<uint32_t> start_local_ids;
    std::vector<uint64_t> query_filter_bits;
    std::vector<uint8_t> query_filter_modes;
    std::vector<uint32_t> query_single_extra_labels;
    std::vector<uint32_t> query_active_range_counts;
    std::vector<uint32_t> query_label_indptr;
    std::vector<uint32_t> query_label_indices;
    std::vector<uint8_t> query_range_modes;
    std::vector<float> query_range_lower_bounds;
    std::vector<float> query_range_upper_bounds;
    std::vector<uint32_t> local_high_route_ids;
    std::vector<uint32_t> local_low_route_ids;
};

struct LatencyBatchMeasurement {
    double planner_ms = 0.0;
    double metadata_ms = 0.0;
    double e2e_ms = 0.0;
    double total_with_planner_ms = 0.0;
    double gpu_ms = 0.0;
    uint32_t first_original_query = 0;
    uint32_t original_count = 0;
    uint32_t route_count = 0;
    uint32_t high_route_count = 0;
    uint32_t low_route_count = 0;
    uint32_t unroutable_original_count = 0;
};

struct AsyncLatencySlot {
    cudaStream_t stream = nullptr;
    cudaEvent_t gpu_start = nullptr;
    cudaEvent_t gpu_stop = nullptr;
    cudaEvent_t done = nullptr;
    size_t offset = 0;
    bool busy = false;
    bool planned = false;
    std::chrono::steady_clock::time_point outer_start;
    std::chrono::steady_clock::time_point planner_stop;
    std::chrono::steady_clock::time_point e2e_start;
    LatencyBatchRoutes batch;
    LatencyOnlineMetadata online_metadata;
    std::vector<uint32_t> slot_query_label_indptr;
    std::vector<uint32_t> slot_local_route_ids;
    std::vector<uint32_t> slot_local_high_route_ids;
    std::vector<uint32_t> slot_local_low_route_ids;
    PinnedHostBuffer<idx_t> compact_ids;
    PinnedHostBuffer<float> compact_distances;
    std::vector<std::vector<idx_t>> merged_results;
    LatencyBatchMeasurement measurement;
};

struct PersistentLatencySlot {
    size_t slot_index = 0;
    size_t offset = 0;
    uint32_t ready_seq = 0;
    bool busy = false;
    bool queued_to_gpu = false;
    std::chrono::steady_clock::time_point outer_start;
    std::chrono::steady_clock::time_point planner_stop;
    std::chrono::steady_clock::time_point e2e_start;
    LatencyBatchRoutes batch;
    LatencyOnlineMetadata online_metadata;
    std::vector<uint32_t> slot_query_label_indptr;
    std::vector<uint32_t> slot_local_route_ids;
    std::vector<uint32_t> slot_local_high_route_ids;
    std::vector<uint32_t> slot_local_low_route_ids;
    PinnedHostBuffer<idx_t> compact_ids;
    PinnedHostBuffer<float> compact_distances;
    std::vector<std::vector<idx_t>> merged_results;
    LatencyBatchMeasurement measurement;
};

struct PersistentLatencyQueue {
    PersistentLatencyDeviceRuntime device;
    cudaStream_t worker_stream = nullptr;
    gdr_t g = nullptr;
    uint32_t slot_count = 0;
    double clock_cycles_per_ms = 0.0;
    bool launched = false;
};

void reset_persistent_latency_device(PersistentLatencyDeviceRuntime& device)
{
    device.requests.reset();
    device.completions.reset();
    device.route_indices.reset();
    device.high_query_indices.reset();
    device.low_query_indices.reset();
    device.query_vector_ids.reset();
    device.query_group_ids.reset();
    device.start_local_ids.reset();
    device.query_filter_bits.reset();
    device.query_filter_modes.reset();
    device.query_single_extra_labels.reset();
    device.query_active_range_counts.reset();
    device.query_label_indptr.reset();
    device.query_label_indices.reset();
    device.query_range_modes.reset();
    device.query_range_lower_bounds.reset();
    device.query_range_upper_bounds.reset();
    device.results.reset();
    device.distances.reset();
    device.route_capacity = 0;
    device.slot_route_capacity = 0;
    device.label_nnz_capacity_per_route = 0;
    device.range_count = 0;
    device.topk = 0;
}

struct LatencyPlannerRoute {
    uint32_t original_query = UINT32_MAX;
    uint32_t route_idx = UINT32_MAX;
    bool is_lowfreq = false;
    bool is_highfreq = false;
};

struct LatencySummary {
    double avg = 0.0;
    double min = 0.0;
    double p50 = 0.0;
    double p90 = 0.0;
    double p95 = 0.0;
    double p99 = 0.0;
    double max = 0.0;
};

std::vector<std::vector<uint32_t>> build_routes_by_original_query(const GraphRuntime& runtime)
{
    std::vector<std::vector<uint32_t>> routes(runtime.original_query_count);
    const uint32_t route_count = runtime_route_count(runtime);
    for (uint32_t route_idx = 0; route_idx < route_count; ++route_idx) {
        if (route_idx >= runtime.route_to_original_query.size()) {
            continue;
        }
        const uint32_t original_query = runtime.route_to_original_query[route_idx];
        if (original_query < runtime.original_query_count) {
            routes[original_query].push_back(route_idx);
        }
    }
    return routes;
}

bool route_is_lowfreq(const GraphRuntime& runtime, uint32_t route_idx)
{
    return route_idx < runtime.query_is_lowfreq.size() && runtime.query_is_lowfreq[route_idx] != 0;
}

bool route_is_highfreq(const GraphRuntime& runtime, uint32_t route_idx)
{
    return route_idx < runtime.query_group_ids.size() && runtime.query_group_ids[route_idx] != UINT32_MAX;
}

bool planner_should_replace_route_choice(const std::string& routing_mode,
                                         uint32_t current_posting,
                                         uint32_t best_posting)
{
    if (routing_mode == "greedy_min_posting") {
        return current_posting < best_posting;
    }
    if (routing_mode == "greedy_max_posting") {
        return current_posting > best_posting;
    }
    fail_full_gpu("unsupported routing_mode for latency planner timing: " + routing_mode);
}

bool query_has_any_active_range_for_latency(const QueryRangeTable& ranges, uint32_t query_idx)
{
    if (query_idx >= ranges.rows) {
        return false;
    }
    for (uint32_t range_idx = 0; range_idx < ranges.range_count; ++range_idx) {
        if (ranges.modes[ranges.flat_index(query_idx, range_idx)] != 0) {
            return true;
        }
    }
    return false;
}

LatencyPlannerRoute find_materialized_route(const GraphRuntime& runtime,
                                            const std::vector<uint32_t>& route_ids,
                                            uint32_t original_query,
                                            uint32_t chosen_label,
                                            uint32_t chosen_group)
{
    LatencyPlannerRoute planned;
    planned.original_query = original_query;
    for (uint32_t route_idx : route_ids) {
        if (route_idx >= runtime.route_to_original_query.size() ||
            route_idx >= runtime.query_filter_labels.size() ||
            route_idx >= runtime.query_group_ids.size()) {
            continue;
        }
        if (runtime.route_to_original_query[route_idx] == original_query &&
            runtime.query_filter_labels[route_idx] == chosen_label &&
            runtime.query_group_ids[route_idx] == chosen_group) {
            planned.route_idx = route_idx;
            planned.is_lowfreq = route_is_lowfreq(runtime, route_idx);
            planned.is_highfreq = !planned.is_lowfreq && route_is_highfreq(runtime, route_idx);
            return planned;
        }
    }
    return planned;
}

std::vector<LatencyPlannerRoute> plan_latency_routes_for_original_query(
    const GraphRuntime& runtime,
    const std::vector<std::vector<uint32_t>>& routes_by_original_query,
    uint32_t original_query,
    const std::string& routing_mode)
{
    if (original_query >= runtime.original_query_labels_matrix.row_labels.size()) {
        fail_full_gpu("latency planner original query label row is out of range");
    }
    if (original_query >= routes_by_original_query.size()) {
        return {};
    }

    const auto& labels = runtime.original_query_labels_matrix.row_labels[original_query];
    const bool has_active_range = query_has_any_active_range_for_latency(runtime.original_query_ranges,
                                                                        original_query);
    if (labels.empty()) {
        if (has_active_range) {
            return {};
        }
        fail_full_gpu("latency planner encountered empty-label query without active range");
    }

    const auto& materialized_routes = routes_by_original_query[original_query];
    std::vector<LatencyPlannerRoute> planned_routes;

    const QueryLabelOp label_op = original_query < runtime.original_query_label_ops.size()
        ? runtime.original_query_label_ops[original_query]
        : QueryLabelOp::And;
    if (label_op == QueryLabelOp::Or) {
        for (uint32_t original_label : labels) {
            if (original_label >= runtime.group_layout.label_to_group.size()) {
                fail_full_gpu("latency planner query label is outside group mapping");
            }
            const uint32_t group = runtime.group_layout.label_to_group[original_label];
            if (group == UINT32_MAX || group >= runtime.metadata.label_count) {
                fail_full_gpu("latency planner query label has unmapped group");
            }
            if (original_label >= runtime.metadata.label_filter_bit.size() ||
                runtime.metadata.label_filter_bit[original_label] == 0) {
                fail_full_gpu("latency planner query label has no valid filter bit");
            }
            const uint32_t posting_size = runtime.metadata.group_posting_size[group];
            const bool is_highfreq = runtime.metadata.label_size[group] > 0;
            const bool is_lowfreq = !is_highfreq &&
                                    group < runtime.lowfreq_index.label_size.size() &&
                                    runtime.lowfreq_index.label_size[group] > 0;
            if (posting_size == 0 && !is_lowfreq) {
                continue;
            }
            planned_routes.push_back(find_materialized_route(runtime,
                                                             materialized_routes,
                                                             original_query,
                                                             original_label,
                                                             group));
        }
        return planned_routes;
    }

    uint32_t chosen_lowfreq_label = UINT32_MAX;
    uint32_t chosen_lowfreq_group = UINT32_MAX;
    uint32_t chosen_lowfreq_posting = UINT32_MAX;
    for (uint32_t original_label : labels) {
        if (original_label >= runtime.group_layout.label_to_group.size()) {
            fail_full_gpu("latency planner query label is outside group mapping");
        }
        const uint32_t group = runtime.group_layout.label_to_group[original_label];
        if (group == UINT32_MAX || group >= runtime.metadata.label_count) {
            fail_full_gpu("latency planner query label has unmapped group");
        }
        if (original_label >= runtime.metadata.label_filter_bit.size() ||
            runtime.metadata.label_filter_bit[original_label] == 0) {
            fail_full_gpu("latency planner query label has no valid filter bit");
        }
        const uint32_t posting_size = runtime.metadata.group_posting_size[group];
        const bool is_highfreq = runtime.metadata.label_size[group] > 0;
        const bool is_lowfreq = !is_highfreq &&
                                group < runtime.lowfreq_index.label_size.size() &&
                                runtime.lowfreq_index.label_size[group] > 0;
        if (is_lowfreq && (chosen_lowfreq_label == UINT32_MAX ||
                           posting_size < chosen_lowfreq_posting)) {
            chosen_lowfreq_label = original_label;
            chosen_lowfreq_group = group;
            chosen_lowfreq_posting = posting_size;
        }
    }
    if (chosen_lowfreq_label != UINT32_MAX) {
        planned_routes.push_back(find_materialized_route(runtime,
                                                         materialized_routes,
                                                         original_query,
                                                         chosen_lowfreq_label,
                                                         chosen_lowfreq_group));
        return planned_routes;
    }

    if (routing_mode == "per_label_multi_route") {
        for (uint32_t original_label : labels) {
            if (original_label >= runtime.group_layout.label_to_group.size()) {
                fail_full_gpu("latency planner query label is outside group mapping");
            }
            const uint32_t group = runtime.group_layout.label_to_group[original_label];
            if (group == UINT32_MAX || group >= runtime.metadata.label_count) {
                fail_full_gpu("latency planner query label has unmapped group");
            }
            const uint32_t posting_size = runtime.metadata.group_posting_size[group];
            if (posting_size == 0 || runtime.metadata.label_size[group] == 0) {
                continue;
            }
            planned_routes.push_back(find_materialized_route(runtime,
                                                             materialized_routes,
                                                             original_query,
                                                             original_label,
                                                             group));
        }
        return planned_routes;
    }

    uint32_t chosen_original_label = UINT32_MAX;
    uint32_t chosen_group = UINT32_MAX;
    uint32_t chosen_group_size = UINT32_MAX;
    for (uint32_t original_label : labels) {
        if (original_label >= runtime.group_layout.label_to_group.size()) {
            fail_full_gpu("latency planner query label is outside group mapping");
        }
        const uint32_t group = runtime.group_layout.label_to_group[original_label];
        if (group == UINT32_MAX || group >= runtime.metadata.label_count) {
            fail_full_gpu("latency planner query label has unmapped group");
        }
        if (original_label >= runtime.metadata.label_filter_bit.size() ||
            runtime.metadata.label_filter_bit[original_label] == 0) {
            fail_full_gpu("latency planner query label has no valid filter bit");
        }
        const uint32_t posting_size = runtime.metadata.group_posting_size[group];
        if (posting_size == 0) {
            continue;
        }
        if (chosen_original_label == UINT32_MAX ||
            planner_should_replace_route_choice(routing_mode, posting_size, chosen_group_size)) {
            chosen_original_label = original_label;
            chosen_group = group;
            chosen_group_size = posting_size;
        }
    }
    if (chosen_original_label != UINT32_MAX) {
        planned_routes.push_back(find_materialized_route(runtime,
                                                         materialized_routes,
                                                         original_query,
                                                         chosen_original_label,
                                                         chosen_group));
    }
    return planned_routes;
}

void append_planned_route_to_latency_batch(const GraphRuntime& runtime,
                                           uint32_t route_idx,
                                           LatencyBatchRoutes& batch)
{
    if (route_idx >= runtime_route_count(runtime)) {
        return;
    }
    if (route_is_lowfreq(runtime, route_idx)) {
        batch.route_ids.push_back(route_idx);
        batch.low_route_ids.push_back(route_idx);
    } else if (route_is_highfreq(runtime, route_idx)) {
        batch.route_ids.push_back(route_idx);
        batch.high_route_ids.push_back(route_idx);
    }
}

LatencyOnlineMetadata build_latency_online_metadata(const GraphRuntime& runtime,
                                                    const LatencyBatchRoutes& batch)
{
    LatencyOnlineMetadata metadata;
    const uint32_t route_count = static_cast<uint32_t>(batch.route_ids.size());
    metadata.local_route_ids.reserve(route_count);
    metadata.query_vector_ids.reserve(route_count);
    metadata.query_group_ids.reserve(route_count);
    metadata.start_local_ids.reserve(route_count);
    metadata.query_filter_bits.reserve(route_count);
    metadata.query_filter_modes.reserve(route_count);
    metadata.query_single_extra_labels.reserve(route_count);
    metadata.query_active_range_counts.reserve(route_count);
    metadata.query_label_indptr.reserve(static_cast<size_t>(route_count) + 1);
    metadata.query_label_indptr.push_back(0);
    metadata.query_range_modes.reserve(static_cast<size_t>(route_count) * runtime.query_ranges.range_count);
    metadata.query_range_lower_bounds.reserve(static_cast<size_t>(route_count) * runtime.query_ranges.range_count);
    metadata.query_range_upper_bounds.reserve(static_cast<size_t>(route_count) * runtime.query_ranges.range_count);
    metadata.local_high_route_ids.reserve(batch.high_route_ids.size());
    metadata.local_low_route_ids.reserve(batch.low_route_ids.size());

    for (uint32_t local_idx = 0; local_idx < route_count; ++local_idx) {
        const uint32_t route_idx = batch.route_ids[local_idx];
        if (route_idx >= runtime.route_to_original_query.size() ||
            route_idx >= runtime.query_group_ids.size() ||
            route_idx >= runtime.start_local_ids.size() ||
            route_idx >= runtime.query_filter_bits.size() ||
            route_idx + 1 >= runtime.query_labels_matrix.indptr.size()) {
            fail_full_gpu("latency online metadata route is outside materialized metadata");
        }
        const uint32_t original_query = runtime.route_to_original_query[route_idx];
        if (original_query >= runtime.original_query_count) {
            fail_full_gpu("latency online metadata route references invalid original query");
        }

        metadata.local_route_ids.push_back(local_idx);
        if (route_is_lowfreq(runtime, route_idx)) {
            metadata.local_low_route_ids.push_back(local_idx);
        } else if (route_is_highfreq(runtime, route_idx)) {
            metadata.local_high_route_ids.push_back(local_idx);
        }
        metadata.query_vector_ids.push_back(original_query);
        metadata.query_group_ids.push_back(runtime.query_group_ids[route_idx]);
        metadata.start_local_ids.push_back(runtime.start_local_ids[route_idx]);
        metadata.query_filter_bits.push_back(runtime.query_filter_bits[route_idx]);

        const uint32_t label_begin =
            static_cast<uint32_t>(runtime.query_labels_matrix.indptr[route_idx]);
        const uint32_t label_end =
            static_cast<uint32_t>(runtime.query_labels_matrix.indptr[static_cast<size_t>(route_idx) + 1]);
        const uint32_t active_range_count =
            range_filter::count_active_ranges(runtime.query_ranges.modes.data(),
                                              route_idx,
                                              runtime.query_ranges.range_count);
        metadata.query_active_range_counts.push_back(active_range_count);
        const uint32_t label_count = label_end - label_begin;
        const bool has_range = active_range_count != 0;
        if (label_count == 0) {
            metadata.query_filter_modes.push_back(has_range ? kQueryFilterModeBitOnlyWithRange
                                                            : kQueryFilterModeBitOnlyNoRange);
            metadata.query_single_extra_labels.push_back(UINT32_MAX);
        } else if (label_count == 1) {
            metadata.query_filter_modes.push_back(has_range ? kQueryFilterModeSingleExtraWithRange
                                                            : kQueryFilterModeSingleExtraNoRange);
            metadata.query_single_extra_labels.push_back(runtime.query_labels_matrix.indices[label_begin]);
        } else {
            metadata.query_filter_modes.push_back(has_range ? kQueryFilterModeGenericWithRange
                                                            : kQueryFilterModeGenericNoRange);
            metadata.query_single_extra_labels.push_back(UINT32_MAX);
        }

        metadata.query_label_indices.insert(metadata.query_label_indices.end(),
                                            runtime.query_labels_matrix.indices.begin() + label_begin,
                                            runtime.query_labels_matrix.indices.begin() + label_end);
        metadata.query_label_indptr.push_back(static_cast<uint32_t>(metadata.query_label_indices.size()));

        for (uint32_t range_idx = 0; range_idx < runtime.query_ranges.range_count; ++range_idx) {
            const size_t src = runtime.query_ranges.flat_index(route_idx, range_idx);
            metadata.query_range_modes.push_back(runtime.query_ranges.modes[src]);
            metadata.query_range_lower_bounds.push_back(runtime.query_ranges.lower_bounds[src]);
            metadata.query_range_upper_bounds.push_back(runtime.query_ranges.upper_bounds[src]);
        }
    }

    return metadata;
}

FullGpuQueryDeviceView upload_latency_online_metadata(FullGpuDeviceRuntime& device,
                                                      const LatencyOnlineMetadata& metadata)
{
    const size_t route_count = metadata.query_vector_ids.size();
    device.latency_query_vector_ids.copy_from_host(metadata.query_vector_ids.data(),
                                                   route_count,
                                                   "cudaMemcpy full_gpu latency query vector ids");
    device.latency_query_group_ids.copy_from_host(metadata.query_group_ids.data(),
                                                  route_count,
                                                  "cudaMemcpy full_gpu latency query group ids");
    device.latency_start_local_ids.copy_from_host(metadata.start_local_ids.data(),
                                                  route_count,
                                                  "cudaMemcpy full_gpu latency start local ids");
    device.latency_query_filter_bits.copy_from_host(metadata.query_filter_bits.data(),
                                                    route_count,
                                                    "cudaMemcpy full_gpu latency query filter bits");
    device.latency_query_filter_modes.copy_from_host(metadata.query_filter_modes.data(),
                                                     route_count,
                                                     "cudaMemcpy full_gpu latency query filter modes");
    device.latency_query_single_extra_labels.copy_from_host(
        metadata.query_single_extra_labels.data(),
        route_count,
        "cudaMemcpy full_gpu latency query single extra labels");
    device.latency_query_active_range_counts.copy_from_host(
        metadata.query_active_range_counts.data(),
        route_count,
        "cudaMemcpy full_gpu latency query active range counts");
    device.latency_query_label_indptr.copy_from_host(metadata.query_label_indptr.data(),
                                                     metadata.query_label_indptr.size(),
                                                     "cudaMemcpy full_gpu latency query label indptr");
    device.latency_query_label_indices.copy_from_host(metadata.query_label_indices.data(),
                                                      metadata.query_label_indices.size(),
                                                      "cudaMemcpy full_gpu latency query label indices");
    device.latency_query_range_modes.copy_from_host(metadata.query_range_modes.data(),
                                                    metadata.query_range_modes.size(),
                                                    "cudaMemcpy full_gpu latency query range modes");
    device.latency_query_range_lower_bounds.copy_from_host(
        metadata.query_range_lower_bounds.data(),
        metadata.query_range_lower_bounds.size(),
        "cudaMemcpy full_gpu latency query range lower bounds");
    device.latency_query_range_upper_bounds.copy_from_host(
        metadata.query_range_upper_bounds.data(),
        metadata.query_range_upper_bounds.size(),
        "cudaMemcpy full_gpu latency query range upper bounds");

    if (!metadata.local_high_route_ids.empty()) {
        device.latency_high_query_indices.copy_from_host(
            metadata.local_high_route_ids.data(),
            metadata.local_high_route_ids.size(),
            "cudaMemcpy full_gpu latency local high route ids");
    }
    if (!metadata.local_low_route_ids.empty()) {
        device.latency_low_query_indices.copy_from_host(
            metadata.local_low_route_ids.data(),
            metadata.local_low_route_ids.size(),
            "cudaMemcpy full_gpu latency local low route ids");
    }

    FullGpuQueryDeviceView view;
    view.query_vector_ids = device.latency_query_vector_ids.data();
    view.query_group_ids = device.latency_query_group_ids.data();
    view.start_local_ids = device.latency_start_local_ids.data();
    view.query_filter_bits = device.latency_query_filter_bits.data();
    view.query_filter_modes = device.latency_query_filter_modes.data();
    view.query_single_extra_labels = device.latency_query_single_extra_labels.data();
    view.query_active_range_counts = device.latency_query_active_range_counts.data();
    view.query_label_indptr = device.latency_query_label_indptr.data();
    view.query_label_indices = device.latency_query_label_indices.data();
    view.query_range_modes = device.latency_query_range_modes.data();
    view.query_range_lower_bounds = device.latency_query_range_lower_bounds.data();
    view.query_range_upper_bounds = device.latency_query_range_upper_bounds.data();
    return view;
}

FullGpuQueryDeviceView upload_latency_online_metadata_at(FullGpuDeviceRuntime& device,
                                                         const LatencyOnlineMetadata& metadata,
                                                         const std::vector<uint32_t>& slot_query_label_indptr,
                                                         const std::vector<uint32_t>& slot_local_route_ids,
                                                         const std::vector<uint32_t>& slot_local_high_route_ids,
                                                         const std::vector<uint32_t>& slot_local_low_route_ids,
                                                         size_t slot_offset,
                                                         uint32_t label_nnz_capacity_per_route,
                                                         uint32_t range_count,
                                                         cudaStream_t stream)
{
    const size_t route_count = metadata.query_vector_ids.size();
    const size_t label_index_offset = slot_offset * label_nnz_capacity_per_route;
    const size_t range_offset = slot_offset * range_count;
    device.latency_query_vector_ids.copy_from_host_async_at(
        metadata.query_vector_ids.data(),
        route_count,
        slot_offset,
        stream,
        "cudaMemcpyAsync full_gpu latency slot query vector ids");
    device.latency_query_group_ids.copy_from_host_async_at(
        metadata.query_group_ids.data(),
        route_count,
        slot_offset,
        stream,
        "cudaMemcpyAsync full_gpu latency slot query group ids");
    device.latency_start_local_ids.copy_from_host_async_at(
        metadata.start_local_ids.data(),
        route_count,
        slot_offset,
        stream,
        "cudaMemcpyAsync full_gpu latency slot start local ids");
    device.latency_query_filter_bits.copy_from_host_async_at(
        metadata.query_filter_bits.data(),
        route_count,
        slot_offset,
        stream,
        "cudaMemcpyAsync full_gpu latency slot query filter bits");
    device.latency_query_filter_modes.copy_from_host_async_at(
        metadata.query_filter_modes.data(),
        route_count,
        slot_offset,
        stream,
        "cudaMemcpyAsync full_gpu latency slot query filter modes");
    device.latency_query_single_extra_labels.copy_from_host_async_at(
        metadata.query_single_extra_labels.data(),
        route_count,
        slot_offset,
        stream,
        "cudaMemcpyAsync full_gpu latency slot query single extra labels");
    device.latency_query_active_range_counts.copy_from_host_async_at(
        metadata.query_active_range_counts.data(),
        route_count,
        slot_offset,
        stream,
        "cudaMemcpyAsync full_gpu latency slot query active range counts");

    device.latency_query_label_indptr.copy_from_host_async_at(
        slot_query_label_indptr.data(),
        slot_query_label_indptr.size(),
        slot_offset,
        stream,
        "cudaMemcpyAsync full_gpu latency slot query label indptr");
    device.latency_query_label_indices.copy_from_host_async_at(
        metadata.query_label_indices.data(),
        metadata.query_label_indices.size(),
        label_index_offset,
        stream,
        "cudaMemcpyAsync full_gpu latency slot query label indices");
    device.latency_query_range_modes.copy_from_host_async_at(
        metadata.query_range_modes.data(),
        metadata.query_range_modes.size(),
        range_offset,
        stream,
        "cudaMemcpyAsync full_gpu latency slot query range modes");
    device.latency_query_range_lower_bounds.copy_from_host_async_at(
        metadata.query_range_lower_bounds.data(),
        metadata.query_range_lower_bounds.size(),
        range_offset,
        stream,
        "cudaMemcpyAsync full_gpu latency slot query range lower bounds");
    device.latency_query_range_upper_bounds.copy_from_host_async_at(
        metadata.query_range_upper_bounds.data(),
        metadata.query_range_upper_bounds.size(),
        range_offset,
        stream,
        "cudaMemcpyAsync full_gpu latency slot query range upper bounds");

    device.latency_route_indices.copy_from_host_async_at(
        slot_local_route_ids.data(),
        slot_local_route_ids.size(),
        slot_offset,
        stream,
        "cudaMemcpyAsync full_gpu latency slot local route ids");

    if (!metadata.local_high_route_ids.empty()) {
        device.latency_high_query_indices.copy_from_host_async_at(
            slot_local_high_route_ids.data(),
            slot_local_high_route_ids.size(),
            slot_offset,
            stream,
            "cudaMemcpyAsync full_gpu latency slot local high route ids");
    }
    if (!metadata.local_low_route_ids.empty()) {
        device.latency_low_query_indices.copy_from_host_async_at(
            slot_local_low_route_ids.data(),
            slot_local_low_route_ids.size(),
            slot_offset,
            stream,
            "cudaMemcpyAsync full_gpu latency slot local low route ids");
    }

    FullGpuQueryDeviceView view;
    view.query_vector_ids = device.latency_query_vector_ids.data();
    view.query_group_ids = device.latency_query_group_ids.data();
    view.start_local_ids = device.latency_start_local_ids.data();
    view.query_filter_bits = device.latency_query_filter_bits.data();
    view.query_filter_modes = device.latency_query_filter_modes.data();
    view.query_single_extra_labels = device.latency_query_single_extra_labels.data();
    view.query_active_range_counts = device.latency_query_active_range_counts.data();
    view.query_label_indptr = device.latency_query_label_indptr.data();
    view.query_label_indices = device.latency_query_label_indices.data();
    view.query_range_modes = device.latency_query_range_modes.data();
    view.query_range_lower_bounds = device.latency_query_range_lower_bounds.data();
    view.query_range_upper_bounds = device.latency_query_range_upper_bounds.data();
    return view;
}

void upload_persistent_latency_metadata_gdr(PersistentLatencyQueue& queue,
                                            const LatencyOnlineMetadata& metadata,
                                            const std::vector<uint32_t>& slot_query_label_indptr,
                                            const std::vector<uint32_t>& slot_local_route_ids,
                                            const std::vector<uint32_t>& slot_local_high_route_ids,
                                            const std::vector<uint32_t>& slot_local_low_route_ids,
                                            size_t slot_offset)
{
    PersistentLatencyDeviceRuntime& target = queue.device;
    const size_t route_count = metadata.query_vector_ids.size();
    const size_t label_index_offset =
        slot_offset * target.label_nnz_capacity_per_route;
    const size_t range_offset = slot_offset * target.range_count;
    target.query_vector_ids.copy_from_host_at(
        metadata.query_vector_ids.data(),
        route_count,
        slot_offset,
        "gdr copy full_gpu persistent query vector ids");
    target.query_group_ids.copy_from_host_at(
        metadata.query_group_ids.data(),
        route_count,
        slot_offset,
        "gdr copy full_gpu persistent query group ids");
    target.start_local_ids.copy_from_host_at(
        metadata.start_local_ids.data(),
        route_count,
        slot_offset,
        "gdr copy full_gpu persistent start local ids");
    target.query_filter_bits.copy_from_host_at(
        metadata.query_filter_bits.data(),
        route_count,
        slot_offset,
        "gdr copy full_gpu persistent query filter bits");
    target.query_filter_modes.copy_from_host_at(
        metadata.query_filter_modes.data(),
        route_count,
        slot_offset,
        "gdr copy full_gpu persistent query filter modes");
    target.query_single_extra_labels.copy_from_host_at(
        metadata.query_single_extra_labels.data(),
        route_count,
        slot_offset,
        "gdr copy full_gpu persistent query single extra labels");
    target.query_active_range_counts.copy_from_host_at(
        metadata.query_active_range_counts.data(),
        route_count,
        slot_offset,
        "gdr copy full_gpu persistent query active range counts");
    target.query_label_indptr.copy_from_host_at(
        slot_query_label_indptr.data(),
        slot_query_label_indptr.size(),
        slot_offset,
        "gdr copy full_gpu persistent query label indptr");
    target.query_label_indices.copy_from_host_at(
        metadata.query_label_indices.data(),
        metadata.query_label_indices.size(),
        label_index_offset,
        "gdr copy full_gpu persistent query label indices");
    target.query_range_modes.copy_from_host_at(
        metadata.query_range_modes.data(),
        metadata.query_range_modes.size(),
        range_offset,
        "gdr copy full_gpu persistent query range modes");
    target.query_range_lower_bounds.copy_from_host_at(
        metadata.query_range_lower_bounds.data(),
        metadata.query_range_lower_bounds.size(),
        range_offset,
        "gdr copy full_gpu persistent query range lower bounds");
    target.query_range_upper_bounds.copy_from_host_at(
        metadata.query_range_upper_bounds.data(),
        metadata.query_range_upper_bounds.size(),
        range_offset,
        "gdr copy full_gpu persistent query range upper bounds");
    target.route_indices.copy_from_host_at(
        slot_local_route_ids.data(),
        slot_local_route_ids.size(),
        slot_offset,
        "gdr copy full_gpu persistent route ids");
    target.high_query_indices.copy_from_host_at(
        slot_local_high_route_ids.data(),
        slot_local_high_route_ids.size(),
        slot_offset,
        "gdr copy full_gpu persistent high route ids");
    target.low_query_indices.copy_from_host_at(
        slot_local_low_route_ids.data(),
        slot_local_low_route_ids.size(),
        slot_offset,
        "gdr copy full_gpu persistent low route ids");
}

LatencyBatchRoutes prepare_latency_batch_routes(
    const GraphRuntime& runtime,
    const std::vector<std::vector<uint32_t>>& routes_by_original_query,
    uint32_t query_window_start,
    uint32_t query_window_count,
    size_t cursor,
    uint32_t requested_batch_size,
    bool wrap_queries,
    bool time_planner,
    const std::string& routing_mode)
{
    LatencyBatchRoutes batch;
    if (query_window_count == 0 || requested_batch_size == 0) {
        return batch;
    }
    if (!wrap_queries && cursor >= query_window_count) {
        return batch;
    }
    const uint32_t remaining_count = wrap_queries
        ? query_window_count
        : query_window_count - static_cast<uint32_t>(cursor);
    batch.original_count = std::min(requested_batch_size, remaining_count);
    batch.original_query_ids.reserve(batch.original_count);
    batch.route_ids.reserve(batch.original_count);
    batch.high_route_ids.reserve(batch.original_count);
    batch.low_route_ids.reserve(batch.original_count);

    for (uint32_t i = 0; i < batch.original_count; ++i) {
        const size_t query_offset = wrap_queries ? ((cursor + i) % query_window_count) : (cursor + i);
        const uint32_t original_query =
            query_window_start + static_cast<uint32_t>(query_offset);
        batch.original_query_ids.push_back(original_query);
        if (original_query >= routes_by_original_query.size()) {
            ++batch.unroutable_original_count;
            continue;
        }
        bool routed = false;
        if (time_planner) {
            const std::vector<LatencyPlannerRoute> planned_routes =
                plan_latency_routes_for_original_query(runtime,
                                                       routes_by_original_query,
                                                       original_query,
                                                       routing_mode);
            for (const auto& planned_route : planned_routes) {
                if (planned_route.route_idx == UINT32_MAX) {
                    continue;
                }
                append_planned_route_to_latency_batch(runtime, planned_route.route_idx, batch);
                routed = true;
            }
        } else {
            for (uint32_t route_idx : routes_by_original_query[original_query]) {
                if (route_idx >= runtime_route_count(runtime)) {
                    continue;
                }
                if (route_is_lowfreq(runtime, route_idx)) {
                    batch.route_ids.push_back(route_idx);
                    batch.low_route_ids.push_back(route_idx);
                    routed = true;
                } else if (route_is_highfreq(runtime, route_idx)) {
                    batch.route_ids.push_back(route_idx);
                    batch.high_route_ids.push_back(route_idx);
                    routed = true;
                }
            }
        }
        if (!routed) {
            ++batch.unroutable_original_count;
        }
    }
    return batch;
}

void merge_latency_batch_results(const GraphRuntime& runtime,
                                 const LatencyBatchRoutes& batch,
                                 const std::vector<idx_t>& compact_ids,
                                 const std::vector<float>& compact_distances,
                                 uint32_t topk,
                                 std::vector<std::vector<idx_t>>& merged_results)
{
    std::unordered_map<uint32_t, size_t> original_to_local;
    original_to_local.reserve(batch.original_query_ids.size());
    for (size_t i = 0; i < batch.original_query_ids.size(); ++i) {
        original_to_local[batch.original_query_ids[i]] = i;
    }

    std::vector<std::unordered_map<idx_t, float>> candidates(batch.original_query_ids.size());
    for (size_t local_route = 0; local_route < batch.route_ids.size(); ++local_route) {
        const uint32_t route_idx = batch.route_ids[local_route];
        if (route_idx >= runtime.route_to_original_query.size()) {
            continue;
        }
        const uint32_t original_query = runtime.route_to_original_query[route_idx];
        auto original_it = original_to_local.find(original_query);
        if (original_it == original_to_local.end()) {
            continue;
        }
        auto& candidate_map = candidates[original_it->second];
        for (uint32_t j = 0; j < topk; ++j) {
            const size_t flat = local_route * topk + j;
            const idx_t sample_id = compact_ids[flat];
            if (sample_id == static_cast<idx_t>(UINT32_MAX)) {
                continue;
            }
            const float dist = compact_distances[flat];
            auto candidate_it = candidate_map.find(sample_id);
            if (candidate_it == candidate_map.end() || dist < candidate_it->second) {
                candidate_map[sample_id] = dist;
            }
        }
    }

    merged_results.assign(batch.original_query_ids.size(),
                          std::vector<idx_t>(topk, static_cast<idx_t>(UINT32_MAX)));
    for (size_t query_idx = 0; query_idx < candidates.size(); ++query_idx) {
        std::vector<std::pair<float, idx_t>> scored;
        scored.reserve(candidates[query_idx].size());
        for (const auto& entry : candidates[query_idx]) {
            scored.emplace_back(entry.second, entry.first);
        }
        std::sort(scored.begin(), scored.end(), [](const auto& lhs, const auto& rhs) {
            if (lhs.first != rhs.first) {
                return lhs.first < rhs.first;
            }
            return lhs.second < rhs.second;
        });
        for (uint32_t i = 0; i < topk && i < scored.size(); ++i) {
            merged_results[query_idx][i] = scored[i].second;
        }
    }
}

double average_value(const std::vector<double>& values)
{
    if (values.empty()) {
        return 0.0;
    }
    double total = 0.0;
    for (double value : values) {
        total += value;
    }
    return total / static_cast<double>(values.size());
}

double percentile_value(const std::vector<double>& sorted_values, double percentile)
{
    if (sorted_values.empty()) {
        return 0.0;
    }
    const double rank = percentile * static_cast<double>(sorted_values.size());
    size_t index = static_cast<size_t>(std::ceil(rank));
    if (index == 0) {
        index = 1;
    }
    index = std::min(index, sorted_values.size()) - 1;
    return sorted_values[index];
}

LatencySummary summarize_latency_values(const std::vector<double>& values)
{
    if (values.empty()) {
        return {};
    }
    std::vector<double> sorted_values = values;
    std::sort(sorted_values.begin(), sorted_values.end());
    LatencySummary summary;
    summary.avg = average_value(values);
    summary.min = sorted_values.front();
    summary.p50 = percentile_value(sorted_values, 0.50);
    summary.p90 = percentile_value(sorted_values, 0.90);
    summary.p95 = percentile_value(sorted_values, 0.95);
    summary.p99 = percentile_value(sorted_values, 0.99);
    summary.max = sorted_values.back();
    return summary;
}

uint32_t max_routes_per_original_query(const std::vector<std::vector<uint32_t>>& routes_by_original_query,
                                       uint32_t query_window_start,
                                       uint32_t query_window_count)
{
    uint32_t max_routes = 1;
    for (uint32_t query_offset = 0; query_offset < query_window_count; ++query_offset) {
        const uint32_t original_query = query_window_start + query_offset;
        if (original_query < routes_by_original_query.size()) {
            max_routes = std::max(max_routes,
                                  static_cast<uint32_t>(routes_by_original_query[original_query].size()));
        }
    }
    return max_routes;
}

void destroy_async_latency_slots(std::vector<AsyncLatencySlot>& slots)
{
    for (auto& slot : slots) {
        if (slot.gpu_start != nullptr) {
            check_cuda(cudaEventDestroy(slot.gpu_start), "cudaEventDestroy full_gpu staggered slot gpu start");
            slot.gpu_start = nullptr;
        }
        if (slot.gpu_stop != nullptr) {
            check_cuda(cudaEventDestroy(slot.gpu_stop), "cudaEventDestroy full_gpu staggered slot gpu stop");
            slot.gpu_stop = nullptr;
        }
        if (slot.done != nullptr) {
            check_cuda(cudaEventDestroy(slot.done), "cudaEventDestroy full_gpu staggered slot done");
            slot.done = nullptr;
        }
        if (slot.stream != nullptr) {
            check_cuda(cudaStreamDestroy(slot.stream), "cudaStreamDestroy full_gpu staggered slot stream");
            slot.stream = nullptr;
        }
    }
}

std::vector<AsyncLatencySlot> create_async_latency_slots(uint32_t concurrency_level,
                                                         uint32_t slot_route_capacity,
                                                         uint32_t topk)
{
    std::vector<AsyncLatencySlot> slots(concurrency_level);
    try {
        for (uint32_t slot_idx = 0; slot_idx < concurrency_level; ++slot_idx) {
            AsyncLatencySlot& slot = slots[slot_idx];
            slot.offset = static_cast<size_t>(slot_idx) * slot_route_capacity;
            check_cuda(cudaStreamCreateWithFlags(&slot.stream, cudaStreamNonBlocking),
                       "cudaStreamCreateWithFlags full_gpu staggered latency slot");
            check_cuda(cudaEventCreate(&slot.gpu_start), "cudaEventCreate full_gpu staggered slot gpu start");
            check_cuda(cudaEventCreate(&slot.gpu_stop), "cudaEventCreate full_gpu staggered slot gpu stop");
            check_cuda(cudaEventCreateWithFlags(&slot.done, cudaEventDisableTiming),
                       "cudaEventCreateWithFlags full_gpu staggered slot done");
            const size_t compact_capacity = static_cast<size_t>(slot_route_capacity) * topk;
            slot.compact_ids.allocate(compact_capacity, "cudaMallocHost full_gpu staggered compact ids");
            slot.compact_distances.allocate(compact_capacity,
                                            "cudaMallocHost full_gpu staggered compact distances");
        }
    } catch (...) {
        destroy_async_latency_slots(slots);
        throw;
    }
    return slots;
}

std::vector<PersistentLatencySlot> create_persistent_latency_slots(uint32_t concurrency_level,
                                                                   uint32_t slot_route_capacity,
                                                                   uint32_t topk)
{
    std::vector<PersistentLatencySlot> slots(concurrency_level);
    const size_t slot_stride = static_cast<size_t>(slot_route_capacity) + 1;
    for (uint32_t slot_idx = 0; slot_idx < concurrency_level; ++slot_idx) {
        PersistentLatencySlot& slot = slots[slot_idx];
        slot.slot_index = slot_idx;
        slot.offset = static_cast<size_t>(slot_idx) * slot_stride;
        const size_t compact_capacity = slot_stride * topk;
        slot.compact_ids.allocate(compact_capacity, "cudaMallocHost full_gpu persistent compact ids");
        slot.compact_distances.allocate(compact_capacity,
                                        "cudaMallocHost full_gpu persistent compact distances");
    }
    return slots;
}

void destroy_persistent_latency_queue(PersistentLatencyQueue& queue)
{
    if (queue.worker_stream != nullptr) {
        check_cuda(cudaStreamDestroy(queue.worker_stream),
                   "cudaStreamDestroy full_gpu persistent worker stream");
        queue.worker_stream = nullptr;
    }
    reset_persistent_latency_device(queue.device);
    if (queue.g != nullptr) {
        gdr_close(queue.g);
        queue.g = nullptr;
    }
    queue.slot_count = 0;
    queue.launched = false;
}

PersistentLatencyQueue create_persistent_latency_queue(uint32_t concurrency_level,
                                                       uint32_t slot_route_capacity,
                                                       uint32_t label_nnz_capacity_per_route,
                                                       uint32_t range_count,
                                                       uint32_t topk)
{
    PersistentLatencyQueue queue;
    try {
        queue.slot_count = concurrency_level;
        queue.g = gdr_open();
        if (queue.g == nullptr) {
            fail_full_gpu("gdr_open returned nullptr for full_gpu persistent queue");
        }
        const uint32_t route_stride = slot_route_capacity + 1;
        const uint32_t route_capacity = std::max<uint32_t>(1, concurrency_level * route_stride);
        queue.device.route_capacity = route_capacity;
        queue.device.slot_route_capacity = slot_route_capacity;
        queue.device.label_nnz_capacity_per_route = label_nnz_capacity_per_route;
        queue.device.range_count = range_count;
        queue.device.topk = topk;
        queue.device.requests.allocate(queue.g,
                                      concurrency_level,
                                      "gdr allocate full_gpu persistent request controls");
        queue.device.completions.allocate(concurrency_level,
                                          "cudaHostAllocMapped full_gpu persistent completions");
        queue.device.route_indices.allocate(queue.g,
                                           route_capacity,
                                           "gdr allocate full_gpu persistent route ids");
        queue.device.high_query_indices.allocate(queue.g,
                                                route_capacity,
                                                "gdr allocate full_gpu persistent high ids");
        queue.device.low_query_indices.allocate(queue.g,
                                               route_capacity,
                                               "gdr allocate full_gpu persistent low ids");
        queue.device.query_vector_ids.allocate(queue.g,
                                              route_capacity,
                                              "gdr allocate full_gpu persistent query vector ids");
        queue.device.query_group_ids.allocate(queue.g,
                                             route_capacity,
                                             "gdr allocate full_gpu persistent query group ids");
        queue.device.start_local_ids.allocate(queue.g,
                                             route_capacity,
                                             "gdr allocate full_gpu persistent start local ids");
        queue.device.query_filter_bits.allocate(queue.g,
                                               route_capacity,
                                               "gdr allocate full_gpu persistent query filter bits");
        queue.device.query_filter_modes.allocate(queue.g,
                                                route_capacity,
                                                "gdr allocate full_gpu persistent query filter modes");
        queue.device.query_single_extra_labels.allocate(
            queue.g,
            route_capacity,
            "gdr allocate full_gpu persistent query single extra labels");
        queue.device.query_active_range_counts.allocate(
            queue.g,
            route_capacity,
            "gdr allocate full_gpu persistent query active range counts");
        queue.device.query_label_indptr.allocate(
            queue.g,
            route_capacity,
            "gdr allocate full_gpu persistent query label indptr");
        queue.device.query_label_indices.allocate(
            queue.g,
            static_cast<size_t>(route_capacity) * label_nnz_capacity_per_route,
            "gdr allocate full_gpu persistent query label indices");
        queue.device.query_range_modes.allocate(
            queue.g,
            static_cast<size_t>(route_capacity) * range_count,
            "gdr allocate full_gpu persistent query range modes");
        queue.device.query_range_lower_bounds.allocate(
            queue.g,
            static_cast<size_t>(route_capacity) * range_count,
            "gdr allocate full_gpu persistent query range lower bounds");
        queue.device.query_range_upper_bounds.allocate(
            queue.g,
            static_cast<size_t>(route_capacity) * range_count,
            "gdr allocate full_gpu persistent query range upper bounds");
        queue.device.results.allocate(static_cast<size_t>(route_capacity) * topk,
                                      "cudaHostAllocMapped full_gpu persistent results");
        queue.device.distances.allocate(static_cast<size_t>(route_capacity) * topk,
                                        "cudaHostAllocMapped full_gpu persistent distances");
        check_cuda(cudaStreamCreateWithFlags(&queue.worker_stream, cudaStreamNonBlocking),
                   "cudaStreamCreateWithFlags full_gpu persistent worker stream");
        cudaDeviceProp prop{};
        int device_id = 0;
        check_cuda(cudaGetDevice(&device_id), "cudaGetDevice full_gpu persistent queue");
        check_cuda(cudaGetDeviceProperties(&prop, device_id),
                   "cudaGetDeviceProperties full_gpu persistent queue");
        queue.clock_cycles_per_ms = static_cast<double>(prop.clockRate);
    } catch (...) {
        destroy_persistent_latency_queue(queue);
        throw;
    }
    return queue;
}

void launch_persistent_latency_worker(const GraphRuntime& runtime,
                                      const FullGpuSearchOptions& options,
                                      FullGpuDeviceRuntime& device,
                                      PersistentLatencyQueue& queue,
                                      uint32_t slot_route_capacity)
{
    if (queue.launched) {
        return;
    }
    if (options.use_fp16) {
        fail_full_gpu("FP16 full_gpu payload is currently supported by search mode only; latency persistent queue uses FP32");
    }
    const uint32_t topk = static_cast<uint32_t>(options.topk);
    const uint32_t base_rows = runtime_base_row_count(runtime);
    const uint32_t dim = static_cast<uint32_t>(options.dim);
    const uint32_t graph_degree = static_cast<uint32_t>(runtime.graph.cols);
    const uint32_t range_count = runtime.query_ranges.range_count;
    const uint32_t finish_count = static_cast<uint32_t>(std::max(1, options.search_finish_count));
    const uint32_t max_steps = static_cast<uint32_t>(std::max(0, options.search_max_steps));
    const int early_stop_mode = parse_search_early_stop_mode(options.search_early_stop_mode);
    const float* base_ranges = device.base_range_values.size() == 0 ? nullptr : device.base_range_values.data();
    const size_t shared_bytes = runtime.vector_scalar_type == DenseVectorScalarType::Uint8
        ? static_cast<size_t>(dim) * sizeof(uint8_t)
        : static_cast<size_t>(dim) * sizeof(float);
    cudaDeviceProp prop{};
    int device_id = 0;
    check_cuda(cudaGetDevice(&device_id), "cudaGetDevice full_gpu persistent occupancy");
    check_cuda(cudaGetDeviceProperties(&prop, device_id),
               "cudaGetDeviceProperties full_gpu persistent occupancy");

    dispatch_full_gpu_kernel_by_queue_size(options.search_queue_size, [&]<int kQueueSize>() {
        if (runtime.vector_scalar_type == DenseVectorScalarType::Uint8) {
            int blocks_per_sm = 0;
            check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                           &blocks_per_sm,
                           persistent_latency_queue_kernel<kQueueSize, uint8_t>,
                           kHighfreqBlockThreads,
                           shared_bytes),
                       "cudaOccupancyMaxActiveBlocksPerMultiprocessor full_gpu persistent u8");
            const uint32_t resident_blocks =
                static_cast<uint32_t>(std::max(0, blocks_per_sm)) *
                static_cast<uint32_t>(prop.multiProcessorCount);
            if (queue.slot_count > resident_blocks) {
                fail_full_gpu("latency_persistent_queue concurrency exceeds resident worker capacity");
            }
            persistent_latency_queue_kernel<kQueueSize, uint8_t><<<queue.slot_count,
                                                                  kHighfreqBlockThreads,
                                                                  shared_bytes,
                                                                  queue.worker_stream>>>(
                device.base_u8.data(),
                device.query_u8.data(),
                base_rows,
                dim,
                device.graph.data(),
                graph_degree,
                queue.device.requests.device_data(),
                queue.device.completions.device_data(),
                slot_route_capacity,
                queue.device.route_indices.device_data(),
                queue.device.high_query_indices.device_data(),
                queue.device.low_query_indices.device_data(),
                queue.device.query_vector_ids.device_data(),
                queue.device.query_group_ids.device_data(),
                queue.device.start_local_ids.device_data(),
                queue.device.query_filter_bits.device_data(),
                queue.device.query_filter_modes.device_data(),
                queue.device.query_single_extra_labels.device_data(),
                queue.device.query_active_range_counts.device_data(),
                queue.device.query_label_indptr.device_data(),
                queue.device.query_label_indices.device_data(),
                queue.device.query_range_modes.device_data(),
                queue.device.query_range_lower_bounds.device_data(),
                queue.device.query_range_upper_bounds.device_data(),
                device.label_size.data(),
                device.label_offset.data(),
                device.row_to_sample.data(),
                device.row_filter_mask.data(),
                device.low_label_size.data(),
                device.low_label_offset.data(),
                device.low_sample_ids.data(),
                device.low_row_filter_mask.data(),
                device.sample_label_indptr.data(),
                device.sample_label_indices.data(),
                base_ranges,
                range_count,
                finish_count,
                early_stop_mode,
                max_steps,
                topk,
                queue.device.results.device_data(),
                queue.device.distances.device_data(),
                device.route_steps.data());
        } else {
            int blocks_per_sm = 0;
            check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                           &blocks_per_sm,
                           persistent_latency_queue_kernel<kQueueSize, float>,
                           kHighfreqBlockThreads,
                           shared_bytes),
                       "cudaOccupancyMaxActiveBlocksPerMultiprocessor full_gpu persistent f32");
            const uint32_t resident_blocks =
                static_cast<uint32_t>(std::max(0, blocks_per_sm)) *
                static_cast<uint32_t>(prop.multiProcessorCount);
            if (queue.slot_count > resident_blocks) {
                fail_full_gpu("latency_persistent_queue concurrency exceeds resident worker capacity");
            }
            persistent_latency_queue_kernel<kQueueSize, float><<<queue.slot_count,
                                                                kHighfreqBlockThreads,
                                                                shared_bytes,
                                                                queue.worker_stream>>>(
                device.base_f32.data(),
                device.query_f32.data(),
                base_rows,
                dim,
                device.graph.data(),
                graph_degree,
                queue.device.requests.device_data(),
                queue.device.completions.device_data(),
                slot_route_capacity,
                queue.device.route_indices.device_data(),
                queue.device.high_query_indices.device_data(),
                queue.device.low_query_indices.device_data(),
                queue.device.query_vector_ids.device_data(),
                queue.device.query_group_ids.device_data(),
                queue.device.start_local_ids.device_data(),
                queue.device.query_filter_bits.device_data(),
                queue.device.query_filter_modes.device_data(),
                queue.device.query_single_extra_labels.device_data(),
                queue.device.query_active_range_counts.device_data(),
                queue.device.query_label_indptr.device_data(),
                queue.device.query_label_indices.device_data(),
                queue.device.query_range_modes.device_data(),
                queue.device.query_range_lower_bounds.device_data(),
                queue.device.query_range_upper_bounds.device_data(),
                device.label_size.data(),
                device.label_offset.data(),
                device.row_to_sample.data(),
                device.row_filter_mask.data(),
                device.low_label_size.data(),
                device.low_label_offset.data(),
                device.low_sample_ids.data(),
                device.low_row_filter_mask.data(),
                device.sample_label_indptr.data(),
                device.sample_label_indices.data(),
                base_ranges,
                range_count,
                finish_count,
                early_stop_mode,
                max_steps,
                topk,
                queue.device.results.device_data(),
                queue.device.distances.device_data(),
                device.route_steps.data());
        }
    });
    check_cuda(cudaGetLastError(), "launch full_gpu persistent latency queue kernel");
    queue.launched = true;
}

void stop_persistent_latency_worker(PersistentLatencyQueue& queue)
{
    if (!queue.launched) {
        return;
    }
    for (uint32_t slot_idx = 0; slot_idx < queue.slot_count; ++slot_idx) {
        PersistentLatencyRequestControl* request =
            queue.device.requests.host_data() + slot_idx;
        request->stop = 1;
        _mm_sfence();
        request->ready_seq = std::numeric_limits<uint32_t>::max();
    }
    _mm_sfence();
    std::atomic_thread_fence(std::memory_order_seq_cst);
    check_cuda(cudaStreamSynchronize(queue.worker_stream),
               "cudaStreamSynchronize full_gpu persistent worker stop");
    queue.launched = false;
}

void submit_async_latency_slot(const GraphRuntime& runtime,
                               const FullGpuSearchOptions& options,
                               FullGpuDeviceRuntime& device,
                               AsyncLatencySlot& slot,
                               const LatencyBatchRoutes& batch,
                               uint32_t slot_route_capacity,
                               uint32_t label_nnz_capacity_per_route,
                               uint32_t topk)
{
    slot.batch = batch;
    slot.measurement = {};
    slot.measurement.first_original_query = batch.original_query_ids.empty()
        ? 0
        : batch.original_query_ids.front();
    slot.measurement.original_count = batch.original_count;
    slot.measurement.route_count = static_cast<uint32_t>(batch.route_ids.size());
    slot.measurement.high_route_count = static_cast<uint32_t>(batch.high_route_ids.size());
    slot.measurement.low_route_count = static_cast<uint32_t>(batch.low_route_ids.size());
    slot.measurement.unroutable_original_count = batch.unroutable_original_count;
    slot.e2e_start = options.latency_time_planner ? slot.planner_stop : slot.outer_start;

    if (batch.route_ids.empty()) {
        slot.merged_results.assign(batch.original_query_ids.size(),
                                   std::vector<idx_t>(topk, static_cast<idx_t>(UINT32_MAX)));
        slot.busy = true;
        slot.planned = true;
        return;
    }

    const auto metadata_start = std::chrono::steady_clock::now();
    slot.online_metadata = build_latency_online_metadata(runtime, batch);
    const size_t label_index_offset = slot.offset * label_nnz_capacity_per_route;
    slot.slot_query_label_indptr.resize(slot.online_metadata.query_label_indptr.size());
    for (size_t i = 0; i < slot.online_metadata.query_label_indptr.size(); ++i) {
        slot.slot_query_label_indptr[i] =
            static_cast<uint32_t>(label_index_offset + slot.online_metadata.query_label_indptr[i]);
    }
    slot.slot_local_route_ids.clear();
    slot.slot_local_route_ids.reserve(slot.online_metadata.local_route_ids.size());
    for (uint32_t local_route_id : slot.online_metadata.local_route_ids) {
        slot.slot_local_route_ids.push_back(static_cast<uint32_t>(slot.offset + local_route_id));
    }
    slot.slot_local_high_route_ids.clear();
    slot.slot_local_high_route_ids.reserve(slot.online_metadata.local_high_route_ids.size());
    for (uint32_t local_route_id : slot.online_metadata.local_high_route_ids) {
        slot.slot_local_high_route_ids.push_back(static_cast<uint32_t>(slot.offset + local_route_id));
    }
    slot.slot_local_low_route_ids.clear();
    slot.slot_local_low_route_ids.reserve(slot.online_metadata.local_low_route_ids.size());
    for (uint32_t local_route_id : slot.online_metadata.local_low_route_ids) {
        slot.slot_local_low_route_ids.push_back(static_cast<uint32_t>(slot.offset + local_route_id));
    }
    const FullGpuQueryDeviceView query_view =
        upload_latency_online_metadata_at(device,
                                          slot.online_metadata,
                                          slot.slot_query_label_indptr,
                                          slot.slot_local_route_ids,
                                          slot.slot_local_high_route_ids,
                                          slot.slot_local_low_route_ids,
                                          slot.offset,
                                          label_nnz_capacity_per_route,
                                          runtime.query_ranges.range_count,
                                          slot.stream);
    const auto metadata_stop = std::chrono::steady_clock::now();
    slot.measurement.metadata_ms =
        std::chrono::duration<double, std::milli>(metadata_stop - metadata_start).count();

    check_cuda(cudaEventRecord(slot.gpu_start, slot.stream),
               "cudaEventRecord full_gpu staggered slot gpu start");
    launch_latency_iteration(runtime,
                             options,
                             device,
                             query_view,
                             static_cast<uint32_t>(batch.route_ids.size()),
                             static_cast<uint32_t>(batch.high_route_ids.size()),
                             static_cast<uint32_t>(batch.low_route_ids.size()),
                             slot.stream,
                             device.latency_route_indices.data() + slot.offset,
                             device.latency_high_query_indices.data() + slot.offset,
                             device.latency_low_query_indices.data() + slot.offset,
                             device.latency_results.data() + slot.offset * topk,
                             device.latency_distances.data() + slot.offset * topk);
    check_cuda(cudaEventRecord(slot.gpu_stop, slot.stream),
               "cudaEventRecord full_gpu staggered slot gpu stop");
    const size_t compact_count = batch.route_ids.size() * topk;
    device.latency_results.copy_to_host_async_at(
        slot.compact_ids.data(),
        compact_count,
        slot.offset * topk,
        slot.stream,
        "cudaMemcpyAsync full_gpu staggered compact ids host");
    device.latency_distances.copy_to_host_async_at(
        slot.compact_distances.data(),
        compact_count,
        slot.offset * topk,
        slot.stream,
        "cudaMemcpyAsync full_gpu staggered compact distances host");
    check_cuda(cudaEventRecord(slot.done, slot.stream), "cudaEventRecord full_gpu staggered slot done");
    slot.busy = true;
    slot.planned = false;
}

void submit_persistent_latency_slot(const GraphRuntime& runtime,
                                    const FullGpuSearchOptions& options,
                                    FullGpuDeviceRuntime& device,
                                    PersistentLatencyQueue& queue,
                                    PersistentLatencySlot& slot,
                                    const LatencyBatchRoutes& batch,
                                    uint32_t slot_route_capacity,
                                    uint32_t label_nnz_capacity_per_route,
                                    uint32_t topk)
{
    slot.batch = batch;
    slot.measurement = {};
    slot.measurement.first_original_query = batch.original_query_ids.empty()
        ? 0
        : batch.original_query_ids.front();
    slot.measurement.original_count = batch.original_count;
    slot.measurement.route_count = static_cast<uint32_t>(batch.route_ids.size());
    slot.measurement.high_route_count = static_cast<uint32_t>(batch.high_route_ids.size());
    slot.measurement.low_route_count = static_cast<uint32_t>(batch.low_route_ids.size());
    slot.measurement.unroutable_original_count = batch.unroutable_original_count;
    slot.e2e_start = options.latency_time_planner ? slot.planner_stop : slot.outer_start;

    if (batch.route_ids.empty()) {
        slot.merged_results.assign(batch.original_query_ids.size(),
                                   std::vector<idx_t>(topk, static_cast<idx_t>(UINT32_MAX)));
        slot.busy = true;
        slot.queued_to_gpu = false;
        return;
    }

    const auto metadata_start = std::chrono::steady_clock::now();
    slot.online_metadata = build_latency_online_metadata(runtime, batch);
    const size_t label_index_offset = slot.offset * label_nnz_capacity_per_route;
    slot.slot_query_label_indptr.resize(slot.online_metadata.query_label_indptr.size());
    for (size_t i = 0; i < slot.online_metadata.query_label_indptr.size(); ++i) {
        slot.slot_query_label_indptr[i] =
            static_cast<uint32_t>(label_index_offset + slot.online_metadata.query_label_indptr[i]);
    }
    slot.slot_local_route_ids.clear();
    slot.slot_local_route_ids.reserve(slot.online_metadata.local_route_ids.size());
    for (uint32_t local_route_id : slot.online_metadata.local_route_ids) {
        slot.slot_local_route_ids.push_back(static_cast<uint32_t>(slot.offset + local_route_id));
    }
    slot.slot_local_high_route_ids.clear();
    slot.slot_local_high_route_ids.reserve(slot.online_metadata.local_high_route_ids.size());
    for (uint32_t local_route_id : slot.online_metadata.local_high_route_ids) {
        slot.slot_local_high_route_ids.push_back(static_cast<uint32_t>(slot.offset + local_route_id));
    }
    slot.slot_local_low_route_ids.clear();
    slot.slot_local_low_route_ids.reserve(slot.online_metadata.local_low_route_ids.size());
    for (uint32_t local_route_id : slot.online_metadata.local_low_route_ids) {
        slot.slot_local_low_route_ids.push_back(static_cast<uint32_t>(slot.offset + local_route_id));
    }

    (void)device;
    (void)slot_route_capacity;
    (void)label_nnz_capacity_per_route;
    upload_persistent_latency_metadata_gdr(queue,
                                           slot.online_metadata,
                                           slot.slot_query_label_indptr,
                                           slot.slot_local_route_ids,
                                           slot.slot_local_high_route_ids,
                                           slot.slot_local_low_route_ids,
                                           slot.offset);
    _mm_sfence();
    std::atomic_thread_fence(std::memory_order_release);
    const auto metadata_stop = std::chrono::steady_clock::now();
    slot.measurement.metadata_ms =
        std::chrono::duration<double, std::milli>(metadata_stop - metadata_start).count();

    PersistentLatencyCompletionControl& completion =
        queue.device.completions.host_data()[slot.slot_index];
    completion.done_seq = 0;
    completion.gpu_start_clock = 0;
    completion.gpu_stop_clock = 0;
    slot.ready_seq += 1;
    PersistentLatencyRequestControl* request =
        queue.device.requests.host_data() + slot.slot_index;
    request->route_count = static_cast<uint32_t>(batch.route_ids.size());
    request->high_route_count = static_cast<uint32_t>(batch.high_route_ids.size());
    request->low_route_count = static_cast<uint32_t>(batch.low_route_ids.size());
    request->stop = 0;
    _mm_sfence();
    std::atomic_thread_fence(std::memory_order_release);
    request->ready_seq = slot.ready_seq;
    _mm_sfence();
    std::atomic_thread_fence(std::memory_order_seq_cst);
    slot.busy = true;
    slot.queued_to_gpu = true;
}

LatencyBatchMeasurement finish_async_latency_slot(const GraphRuntime& runtime,
                                                  const FullGpuSearchOptions& options,
                                                  AsyncLatencySlot& slot)
{
    const uint32_t topk = static_cast<uint32_t>(options.topk);
    if (!slot.batch.route_ids.empty()) {
        const size_t compact_count = slot.batch.route_ids.size() * topk;
        std::vector<idx_t> compact_ids(slot.compact_ids.data(),
                                       slot.compact_ids.data() + compact_count);
        std::vector<float> compact_distances(slot.compact_distances.data(),
                                             slot.compact_distances.data() + compact_count);
        merge_latency_batch_results(runtime,
                                    slot.batch,
                                    compact_ids,
                                    compact_distances,
                                    topk,
                                    slot.merged_results);
        float gpu_ms = 0.0f;
        check_cuda(cudaEventElapsedTime(&gpu_ms, slot.gpu_start, slot.gpu_stop),
                   "cudaEventElapsedTime full_gpu staggered slot");
        slot.measurement.gpu_ms = static_cast<double>(gpu_ms);
    }

    const auto e2e_stop = std::chrono::steady_clock::now();
    slot.measurement.e2e_ms =
        std::chrono::duration<double, std::milli>(e2e_stop - slot.e2e_start).count();
    slot.measurement.total_with_planner_ms = slot.measurement.e2e_ms;
    if (options.latency_time_planner) {
        slot.measurement.planner_ms =
            std::chrono::duration<double, std::milli>(slot.planner_stop - slot.outer_start).count();
        slot.measurement.total_with_planner_ms =
            std::chrono::duration<double, std::milli>(e2e_stop - slot.outer_start).count();
    }
    slot.busy = false;
    slot.planned = false;
    return slot.measurement;
}

LatencyBatchMeasurement finish_persistent_latency_slot(const GraphRuntime& runtime,
                                                       const FullGpuSearchOptions& options,
                                                       FullGpuDeviceRuntime& device,
                                                       PersistentLatencyQueue& queue,
                                                       PersistentLatencySlot& slot)
{
    const uint32_t topk = static_cast<uint32_t>(options.topk);
    if (!slot.batch.route_ids.empty()) {
        const size_t compact_count = slot.batch.route_ids.size() * topk;
        (void)device;
        std::atomic_thread_fence(std::memory_order_acquire);
        const idx_t* compact_ids_begin =
            queue.device.results.host_data() + slot.offset * topk;
        const float* compact_distances_begin =
            queue.device.distances.host_data() + slot.offset * topk;
        std::vector<idx_t> compact_ids(compact_ids_begin,
                                       compact_ids_begin + compact_count);
        std::vector<float> compact_distances(compact_distances_begin,
                                             compact_distances_begin + compact_count);
        merge_latency_batch_results(runtime,
                                    slot.batch,
                                    compact_ids,
                                    compact_distances,
                                    topk,
                                    slot.merged_results);
        const PersistentLatencyCompletionControl& completion =
            queue.device.completions.host_data()[slot.slot_index];
        if (queue.clock_cycles_per_ms > 0.0 &&
            completion.gpu_stop_clock >= completion.gpu_start_clock) {
            slot.measurement.gpu_ms =
                static_cast<double>(completion.gpu_stop_clock - completion.gpu_start_clock) /
                queue.clock_cycles_per_ms;
        }
    }

    const auto e2e_stop = std::chrono::steady_clock::now();
    slot.measurement.e2e_ms =
        std::chrono::duration<double, std::milli>(e2e_stop - slot.e2e_start).count();
    slot.measurement.total_with_planner_ms = slot.measurement.e2e_ms;
    if (options.latency_time_planner) {
        slot.measurement.planner_ms =
            std::chrono::duration<double, std::milli>(slot.planner_stop - slot.outer_start).count();
        slot.measurement.total_with_planner_ms =
            std::chrono::duration<double, std::milli>(e2e_stop - slot.outer_start).count();
    }
    slot.busy = false;
    slot.queued_to_gpu = false;
    return slot.measurement;
}

void validate_full_gpu_options(const FullGpuSearchOptions& options)
{
    if (options.topk <= 0 || options.topk > static_cast<int>(kMaxTopk)) {
        fail_full_gpu("topk must be in [1, 10]");
    }
    if (options.search_queue_size <= 0 || options.search_queue_size > static_cast<int>(kMaxSearchQueue)) {
        fail_full_gpu("search_queue_size must be in [1, 512]");
    }
    if (options.search_finish_count <= 0) {
        fail_full_gpu("search_finish_count must be >= 1");
    }
    if (!defaults::is_valid_search_early_stop_mode(options.search_early_stop_mode)) {
        fail_full_gpu("search_early_stop_mode must be one of: fixed, off, adaptive");
    }
    if (options.timed_runs <= 0) {
        fail_full_gpu("timed_runs must be >= 1");
    }
}

LatencyBatchMeasurement run_latency_batch(const GraphRuntime& runtime,
                                          const FullGpuSearchOptions& options,
                                          FullGpuDeviceRuntime& device,
                                          const LatencyBatchRoutes& batch,
                                          std::chrono::steady_clock::time_point e2e_start,
                                          cudaEvent_t gpu_start,
                                          cudaEvent_t gpu_stop,
                                          std::vector<idx_t>& compact_ids,
                                          std::vector<float>& compact_distances,
                                          std::vector<std::vector<idx_t>>& merged_results)
{
    const uint32_t topk = static_cast<uint32_t>(options.topk);

    double metadata_ms = 0.0;
    FullGpuQueryDeviceView query_view = default_query_device_view(device);
    if (!batch.route_ids.empty()) {
        const auto metadata_start = std::chrono::steady_clock::now();
        if (options.latency_online_metadata) {
            const LatencyOnlineMetadata online_metadata =
                build_latency_online_metadata(runtime, batch);
            device.latency_route_indices.copy_from_host(
                online_metadata.local_route_ids.data(),
                online_metadata.local_route_ids.size(),
                "cudaMemcpy full_gpu latency local route ids");
            query_view = upload_latency_online_metadata(device, online_metadata);
        } else {
            device.latency_route_indices.copy_from_host(batch.route_ids.data(),
                                                        batch.route_ids.size(),
                                                        "cudaMemcpy full_gpu latency route ids");
            if (!batch.high_route_ids.empty()) {
                device.latency_high_query_indices.copy_from_host(
                    batch.high_route_ids.data(),
                    batch.high_route_ids.size(),
                    "cudaMemcpy full_gpu latency high route ids");
            }
            if (!batch.low_route_ids.empty()) {
                device.latency_low_query_indices.copy_from_host(
                    batch.low_route_ids.data(),
                    batch.low_route_ids.size(),
                    "cudaMemcpy full_gpu latency low route ids");
            }
        }
        const auto metadata_stop = std::chrono::steady_clock::now();
        metadata_ms = std::chrono::duration<double, std::milli>(metadata_stop - metadata_start).count();
    }

    float gpu_ms = 0.0f;
    bool has_gpu_work = false;
    if (!batch.route_ids.empty()) {
        has_gpu_work = true;
        check_cuda(cudaEventRecord(gpu_start), "cudaEventRecord full_gpu latency start");
        launch_latency_iteration(runtime,
                                 options,
                                 device,
                                 query_view,
                                 static_cast<uint32_t>(batch.route_ids.size()),
                                 static_cast<uint32_t>(batch.high_route_ids.size()),
                                 static_cast<uint32_t>(batch.low_route_ids.size()));
        check_cuda(cudaEventRecord(gpu_stop), "cudaEventRecord full_gpu latency stop");

        const size_t compact_count = batch.route_ids.size() * topk;
        compact_ids.resize(compact_count);
        compact_distances.resize(compact_count);
        device.latency_results.copy_to_host(compact_ids.data(),
                                            compact_count,
                                            "cudaMemcpy full_gpu latency compact ids host");
        device.latency_distances.copy_to_host(compact_distances.data(),
                                              compact_count,
                                              "cudaMemcpy full_gpu latency compact distances host");
    } else {
        compact_ids.clear();
        compact_distances.clear();
        merged_results.assign(batch.original_query_ids.size(),
                              std::vector<idx_t>(topk, static_cast<idx_t>(UINT32_MAX)));
    }

    if (!batch.route_ids.empty()) {
        merge_latency_batch_results(runtime,
                                    batch,
                                    compact_ids,
                                    compact_distances,
                                    topk,
                                    merged_results);
    }

    const auto e2e_stop = std::chrono::steady_clock::now();
    if (has_gpu_work) {
        check_cuda(cudaEventSynchronize(gpu_stop), "cudaEventSynchronize full_gpu latency stop");
        check_cuda(cudaEventElapsedTime(&gpu_ms, gpu_start, gpu_stop),
                   "cudaEventElapsedTime full_gpu latency");
    }
    LatencyBatchMeasurement measurement;
    measurement.metadata_ms = metadata_ms;
    measurement.e2e_ms = std::chrono::duration<double, std::milli>(e2e_stop - e2e_start).count();
    measurement.total_with_planner_ms = measurement.e2e_ms;
    measurement.gpu_ms = static_cast<double>(gpu_ms);
    measurement.first_original_query = batch.original_query_ids.empty() ? 0 : batch.original_query_ids.front();
    measurement.original_count = batch.original_count;
    measurement.route_count = static_cast<uint32_t>(batch.route_ids.size());
    measurement.high_route_count = static_cast<uint32_t>(batch.high_route_ids.size());
    measurement.low_route_count = static_cast<uint32_t>(batch.low_route_ids.size());
    measurement.unroutable_original_count = batch.unroutable_original_count;
    return measurement;
}

void write_latency_summary_row(std::ofstream& csv,
                               int batch_size,
                               int warmup_runs,
                               int timed_runs,
                               bool run_all_queries,
                               bool time_planner,
                               bool online_metadata,
                               uint32_t query_window_count,
                               const std::vector<LatencyBatchMeasurement>& measurements)
{
    std::vector<double> planner_values;
    std::vector<double> metadata_values;
    std::vector<double> e2e_values;
    std::vector<double> total_values;
    std::vector<double> gpu_values;
    planner_values.reserve(measurements.size());
    metadata_values.reserve(measurements.size());
    e2e_values.reserve(measurements.size());
    total_values.reserve(measurements.size());
    gpu_values.reserve(measurements.size());

    uint64_t submitted_original_queries = 0;
    uint64_t submitted_routes = 0;
    uint64_t submitted_high_routes = 0;
    uint64_t submitted_low_routes = 0;
    uint64_t unroutable_original_queries = 0;
    for (const auto& measurement : measurements) {
        planner_values.push_back(measurement.planner_ms);
        metadata_values.push_back(measurement.metadata_ms);
        e2e_values.push_back(measurement.e2e_ms);
        total_values.push_back(measurement.total_with_planner_ms);
        gpu_values.push_back(measurement.gpu_ms);
        submitted_original_queries += measurement.original_count;
        submitted_routes += measurement.route_count;
        submitted_high_routes += measurement.high_route_count;
        submitted_low_routes += measurement.low_route_count;
        unroutable_original_queries += measurement.unroutable_original_count;
    }

    const LatencySummary planner = summarize_latency_values(planner_values);
    const LatencySummary metadata = summarize_latency_values(metadata_values);
    const LatencySummary e2e = summarize_latency_values(e2e_values);
    const LatencySummary total = summarize_latency_values(total_values);
    const LatencySummary gpu = summarize_latency_values(gpu_values);
    const double batch_count = measurements.empty() ? 1.0 : static_cast<double>(measurements.size());
    const double avg_routes_per_batch = static_cast<double>(submitted_routes) / batch_count;
    const double avg_high_routes_per_batch = static_cast<double>(submitted_high_routes) / batch_count;
    const double avg_low_routes_per_batch = static_cast<double>(submitted_low_routes) / batch_count;
    const double avg_unroutable_originals_per_batch =
        static_cast<double>(unroutable_original_queries) / batch_count;

    csv << batch_size << ','
        << warmup_runs << ','
        << timed_runs << ','
        << (run_all_queries ? 1 : 0) << ','
        << (time_planner ? 1 : 0) << ','
        << (online_metadata ? 1 : 0) << ','
        << query_window_count << ','
        << measurements.size() << ','
        << submitted_original_queries << ','
        << submitted_routes << ','
        << avg_routes_per_batch << ','
        << avg_high_routes_per_batch << ','
        << avg_low_routes_per_batch << ','
        << avg_unroutable_originals_per_batch << ','
        << planner.avg << ','
        << planner.min << ','
        << planner.p50 << ','
        << planner.p90 << ','
        << planner.p95 << ','
        << planner.p99 << ','
        << planner.max << ','
        << metadata.avg << ','
        << metadata.min << ','
        << metadata.p50 << ','
        << metadata.p90 << ','
        << metadata.p95 << ','
        << metadata.p99 << ','
        << metadata.max << ','
        << e2e.avg << ','
        << e2e.min << ','
        << e2e.p50 << ','
        << e2e.p90 << ','
        << e2e.p95 << ','
        << e2e.p99 << ','
        << e2e.max << ','
        << total.avg << ','
        << total.min << ','
        << total.p50 << ','
        << total.p90 << ','
        << total.p95 << ','
        << total.p99 << ','
        << total.max << ','
        << gpu.avg << ','
        << gpu.min << ','
        << gpu.p50 << ','
        << gpu.p90 << ','
        << gpu.p95 << ','
        << gpu.p99 << ','
        << gpu.max << '\n';

    std::printf("[full_gpu latency] batch_size=%d runs=%zu planner_avg=%.4fms metadata_avg=%.4fms e2e_avg=%.4fms total_avg=%.4fms e2e_p50=%.4fms e2e_p95=%.4fms gpu_avg=%.4fms routes/batch=%.2f high/batch=%.2f low/batch=%.2f\n",
                batch_size,
                measurements.size(),
                planner.avg,
                metadata.avg,
                e2e.avg,
                total.avg,
                e2e.p50,
                e2e.p95,
                gpu.avg,
                avg_routes_per_batch,
                avg_high_routes_per_batch,
                avg_low_routes_per_batch);
}

void write_latency_trace_header(std::ofstream& csv)
{
    csv << "batch_size,batch_index,first_original_query,original_count,route_count,"
        << "high_route_count,low_route_count,unroutable_original_count,"
        << "planner_ms,metadata_ms,e2e_ms,total_with_planner_ms,gpu_ms\n";
}

void write_latency_trace_row(std::ofstream& csv,
                             int batch_size,
                             size_t batch_index,
                             const LatencyBatchMeasurement& measurement)
{
    csv << batch_size << ','
        << batch_index << ','
        << measurement.first_original_query << ','
        << measurement.original_count << ','
        << measurement.route_count << ','
        << measurement.high_route_count << ','
        << measurement.low_route_count << ','
        << measurement.unroutable_original_count << ','
        << measurement.planner_ms << ','
        << measurement.metadata_ms << ','
        << measurement.e2e_ms << ','
        << measurement.total_with_planner_ms << ','
        << measurement.gpu_ms << '\n';
}

void write_staggered_latency_header(std::ofstream& csv)
{
    csv << "concurrency_level,warmup_runs,timed_runs,run_all_queries,time_planner,online_metadata,"
        << "query_window_count,measured_requests,submitted_original_queries,submitted_routes,"
        << "avg_routes_per_request,avg_high_routes_per_request,avg_low_routes_per_request,"
        << "avg_unroutable_originals_per_request,scheduler,device_overlap,"
        << "planner_avg_ms,planner_p50_ms,planner_p95_ms,"
        << "metadata_avg_ms,metadata_p50_ms,metadata_p95_ms,"
        << "e2e_avg_ms,e2e_p50_ms,e2e_p95_ms,"
        << "total_with_planner_avg_ms,total_with_planner_p50_ms,total_with_planner_p95_ms,"
        << "gpu_avg_ms,gpu_p50_ms,gpu_p95_ms,"
        << "closed_loop_elapsed_ms,closed_loop_qps,effective_inflight\n";
}

void write_staggered_latency_row(std::ofstream& csv,
                                 int concurrency_level,
                                 int warmup_runs,
                                 int timed_runs,
                                 bool run_all_queries,
                                 bool time_planner,
                                 bool online_metadata,
                                 const char* scheduler,
                                 bool device_overlap,
                                 uint32_t query_window_count,
                                 double closed_loop_elapsed_ms,
                                 const std::vector<LatencyBatchMeasurement>& measurements)
{
    std::vector<double> planner_values;
    std::vector<double> metadata_values;
    std::vector<double> e2e_values;
    std::vector<double> total_values;
    std::vector<double> gpu_values;
    planner_values.reserve(measurements.size());
    metadata_values.reserve(measurements.size());
    e2e_values.reserve(measurements.size());
    total_values.reserve(measurements.size());
    gpu_values.reserve(measurements.size());

    uint64_t submitted_original_queries = 0;
    uint64_t submitted_routes = 0;
    uint64_t submitted_high_routes = 0;
    uint64_t submitted_low_routes = 0;
    uint64_t unroutable_original_queries = 0;
    for (const auto& measurement : measurements) {
        planner_values.push_back(measurement.planner_ms);
        metadata_values.push_back(measurement.metadata_ms);
        e2e_values.push_back(measurement.e2e_ms);
        total_values.push_back(measurement.total_with_planner_ms);
        gpu_values.push_back(measurement.gpu_ms);
        submitted_original_queries += measurement.original_count;
        submitted_routes += measurement.route_count;
        submitted_high_routes += measurement.high_route_count;
        submitted_low_routes += measurement.low_route_count;
        unroutable_original_queries += measurement.unroutable_original_count;
    }

    const LatencySummary planner = summarize_latency_values(planner_values);
    const LatencySummary metadata = summarize_latency_values(metadata_values);
    const LatencySummary e2e = summarize_latency_values(e2e_values);
    const LatencySummary total = summarize_latency_values(total_values);
    const LatencySummary gpu = summarize_latency_values(gpu_values);
    const double request_count = measurements.empty() ? 1.0 : static_cast<double>(measurements.size());
    const double closed_loop_qps = closed_loop_elapsed_ms > 0.0
        ? (static_cast<double>(submitted_original_queries) * 1000.0 / closed_loop_elapsed_ms)
        : 0.0;
    const double effective_inflight = closed_loop_elapsed_ms > 0.0
        ? (total.avg * closed_loop_qps / 1000.0)
        : 0.0;

    csv << concurrency_level << ','
        << warmup_runs << ','
        << timed_runs << ','
        << (run_all_queries ? 1 : 0) << ','
        << (time_planner ? 1 : 0) << ','
        << (online_metadata ? 1 : 0) << ','
        << query_window_count << ','
        << measurements.size() << ','
        << submitted_original_queries << ','
        << submitted_routes << ','
        << (static_cast<double>(submitted_routes) / request_count) << ','
        << (static_cast<double>(submitted_high_routes) / request_count) << ','
        << (static_cast<double>(submitted_low_routes) / request_count) << ','
        << (static_cast<double>(unroutable_original_queries) / request_count) << ','
        << scheduler << ','
        << (device_overlap ? 1 : 0) << ','
        << planner.avg << ','
        << planner.p50 << ','
        << planner.p95 << ','
        << metadata.avg << ','
        << metadata.p50 << ','
        << metadata.p95 << ','
        << e2e.avg << ','
        << e2e.p50 << ','
        << e2e.p95 << ','
        << total.avg << ','
        << total.p50 << ','
        << total.p95 << ','
        << gpu.avg << ','
        << gpu.p50 << ','
        << gpu.p95 << ','
        << closed_loop_elapsed_ms << ','
        << closed_loop_qps << ','
        << effective_inflight << '\n';

    std::printf("[full_gpu latency staggered] concurrency=%d requests=%zu scheduler=%s device_overlap=%d total_avg=%.4fms total_p50=%.4fms total_p95=%.4fms gpu_avg=%.4fms closed_loop=%.4fms qps=%.2f effective_inflight=%.2f\n",
                concurrency_level,
                measurements.size(),
                scheduler,
                device_overlap ? 1 : 0,
                total.avg,
                total.p50,
                total.p95,
                gpu.avg,
                closed_loop_elapsed_ms,
                closed_loop_qps,
                effective_inflight);
}

std::string default_latency_trace_output_path(const std::string& summary_path)
{
    return summary_path + ".trace.csv";
}

double run_timed_iterations(const GraphRuntime& runtime,
                            const FullGpuSearchOptions& options,
                            FullGpuDeviceRuntime& device,
                            double* graph_ms_out,
                            double* posting_ms_out)
{
    for (int i = 0; i < options.warmup_runs; ++i) {
        launch_search_iteration(runtime, options, device);
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize full_gpu warmup");
    }

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    check_cuda(cudaEventCreate(&start), "cudaEventCreate full_gpu start");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate full_gpu stop");
    FullGpuIterationTimingEvents timing;
    check_cuda(cudaEventCreate(&timing.graph_start), "cudaEventCreate full_gpu graph start");
    check_cuda(cudaEventCreate(&timing.graph_stop), "cudaEventCreate full_gpu graph stop");
    check_cuda(cudaEventCreate(&timing.posting_start), "cudaEventCreate full_gpu posting start");
    check_cuda(cudaEventCreate(&timing.posting_stop), "cudaEventCreate full_gpu posting stop");

    float elapsed_ms = 0.0f;
    double graph_ms = 0.0;
    double posting_ms = 0.0;
    for (int i = 0; i < options.timed_runs; ++i) {
        check_cuda(cudaEventRecord(start), "cudaEventRecord full_gpu start");
        launch_search_iteration(runtime, options, device, &timing);
        check_cuda(cudaEventRecord(stop), "cudaEventRecord full_gpu stop");
        check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize full_gpu stop");
        float iteration_ms = 0.0f;
        check_cuda(cudaEventElapsedTime(&iteration_ms, start, stop), "cudaEventElapsedTime full_gpu");
        float graph_iteration_ms = 0.0f;
        float posting_iteration_ms = 0.0f;
        check_cuda(cudaEventElapsedTime(&graph_iteration_ms, timing.graph_start, timing.graph_stop),
                   "cudaEventElapsedTime full_gpu graph");
        check_cuda(cudaEventElapsedTime(&posting_iteration_ms, timing.posting_start, timing.posting_stop),
                   "cudaEventElapsedTime full_gpu posting");
        elapsed_ms += iteration_ms;
        graph_ms += graph_iteration_ms;
        posting_ms += posting_iteration_ms;
    }

    check_cuda(cudaEventDestroy(start), "cudaEventDestroy full_gpu start");
    check_cuda(cudaEventDestroy(stop), "cudaEventDestroy full_gpu stop");
    check_cuda(cudaEventDestroy(timing.graph_start), "cudaEventDestroy full_gpu graph start");
    check_cuda(cudaEventDestroy(timing.graph_stop), "cudaEventDestroy full_gpu graph stop");
    check_cuda(cudaEventDestroy(timing.posting_start), "cudaEventDestroy full_gpu posting start");
    check_cuda(cudaEventDestroy(timing.posting_stop), "cudaEventDestroy full_gpu posting stop");
    if (graph_ms_out != nullptr) *graph_ms_out = graph_ms;
    if (posting_ms_out != nullptr) *posting_ms_out = posting_ms;
    return static_cast<double>(elapsed_ms);
}

}  // namespace

int run_full_gpu_search(const FullGpuSearchOptions& options)
{
    validate_full_gpu_options(options);

    GraphRuntime runtime = load_full_gpu_runtime(options);
    validate_runtime_for_full_gpu(runtime, options);

    const UInt32Csr query_label_csr =
        build_sorted_query_label_csr_without_primary(runtime.query_labels_matrix,
                                                     runtime.query_filter_labels);
    const QueryFilterFastPath query_filter_fast_path =
        build_query_filter_fast_path(query_label_csr, runtime.query_ranges);
    const UInt32Csr sample_label_csr = build_sorted_label_csr(runtime.data_labels);

    FullGpuDeviceRuntime device;
    upload_runtime(runtime, query_label_csr, query_filter_fast_path, sample_label_csr, false,
                   options.use_fp16, device);

    const uint32_t route_count = runtime_route_count(runtime);
    const uint32_t topk = static_cast<uint32_t>(options.topk);
    device.route_results.allocate(static_cast<size_t>(route_count) * topk,
                                  "cudaMalloc full_gpu route_results");
    device.route_distances.allocate(static_cast<size_t>(route_count) * topk,
                                    "cudaMalloc full_gpu route_distances");
    device.route_steps.allocate(route_count, "cudaMalloc full_gpu route_steps");
    std::vector<uint32_t> posting_route_ids;
    if (runtime.range_posting_index != nullptr) {
        for (uint32_t route = 0; route < route_count; ++route) {
            if (route < runtime.route_enabled.size() && runtime.route_enabled[route] != 0 &&
                route < runtime.route_providers.size() &&
                runtime.route_providers[route] == QueryRouteProvider::RangePosting &&
                route < runtime.route_execution_modes.size() &&
                (runtime.route_execution_modes[route] == QueryRouteExecution::GpuResidentScan ||
                 runtime.route_execution_modes[route] == QueryRouteExecution::GpuResidentScanPostFilter)) {
                posting_route_ids.push_back(route);
            }
        }
    }
    device.range_posting_route_ids.upload(posting_route_ids,
                                          "cudaMalloc full_gpu range posting route ids");
    prepare_full_gpu_overlap(runtime, options, device);
    const auto wall_start = std::chrono::steady_clock::now();
    double graph_ms = 0.0;
    double posting_ms = 0.0;
    const double kernel_ms = route_count == 0
        ? 0.0
        : run_timed_iterations(runtime, options, device, &graph_ms, &posting_ms);
    const auto wall_end = std::chrono::steady_clock::now();
    const double wall_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

    std::vector<std::vector<idx_t>> merged_results(
        runtime.original_query_count,
        std::vector<idx_t>(topk, static_cast<idx_t>(UINT32_MAX)));
    if (route_count != 0) {
        const std::vector<idx_t> flat_ids = device.route_results.download("cudaMemcpy full_gpu route_results host");
        const std::vector<float> flat_distances =
            device.route_distances.download("cudaMemcpy full_gpu route_distances host");
        merged_results = merge_route_results_to_original(runtime, flat_ids, flat_distances, topk);
    }
    save_results_to_csv(merged_results, options.output_csv_path);

    if (options.print_summary) {
        const size_t posting_route_count = device.range_posting_route_ids.size();
        const size_t processed_route_count = runtime.high_query_indices.size() +
            runtime.low_query_indices.size() + posting_route_count;
        const double avg_kernel_ms = kernel_ms / static_cast<double>(options.timed_runs);
        const double avg_graph_ms = graph_ms / static_cast<double>(options.timed_runs);
        const double avg_posting_ms = posting_ms / static_cast<double>(options.timed_runs);
        const double qps = avg_kernel_ms > 0.0
            ? static_cast<double>(processed_route_count) / (avg_kernel_ms / 1000.0)
            : 0.0;
        std::printf("FullGPU mixed query split per run: total(original)=%u highfreq(route)=%zu lowfreq(route)=%zu range_posting(route)=%zu unroutable(route)=%zu\n",
                    runtime.original_query_count,
                    runtime.high_query_indices.size(),
                    runtime.low_query_indices.size(),
                    posting_route_count,
                    runtime.query_group_ids.size() - processed_route_count);
        std::printf("FullGPU route query evaluations: %zu\n", processed_route_count);
        std::printf("FullGPU kernel time total: %f ms\n", kernel_ms);
        std::printf("FullGPU kernel time avg: %f ms\n", avg_kernel_ms);
        std::printf("FullGPU graph kernel time avg: %f ms\n", avg_graph_ms);
        std::printf("FullGPU posting kernel time avg: %f ms\n", avg_posting_ms);
        std::printf("FullGPU effective route QPS: %f\n", qps);
        std::printf("FullGPU wall time: %f ms\n", wall_ms);
        std::printf("FullGPU results written to %s\n", options.output_csv_path.c_str());
    }

    return 0;
}

int run_full_gpu_latency_benchmark(const FullGpuSearchOptions& options)
{
    validate_full_gpu_options(options);
    if (options.latency_batch_sizes.empty()) {
        fail_full_gpu("latency_batch_sizes must contain at least one batch size");
    }
    for (int batch_size : options.latency_batch_sizes) {
        if (batch_size <= 0) {
            fail_full_gpu("latency batch sizes must be >= 1");
        }
    }
    if (options.latency_query_start < 0) {
        fail_full_gpu("latency_query_start must be >= 0");
    }
    if (options.latency_query_limit < 0) {
        fail_full_gpu("latency_query_limit must be >= 0");
    }
    for (int concurrency_level : options.latency_concurrency_levels) {
        if (concurrency_level < 0) {
            fail_full_gpu("latency concurrency levels must be >= 0");
        }
    }
    if (options.latency_persistent_queue &&
        (!options.latency_staggered || !options.latency_online_metadata)) {
        fail_full_gpu("latency_persistent_queue requires latency_staggered and latency_online_metadata");
    }

    GraphRuntime runtime = load_full_gpu_runtime(options);
    validate_runtime_for_full_gpu(runtime, options);

    const uint32_t route_count = runtime_route_count(runtime);
    const uint32_t topk = static_cast<uint32_t>(options.topk);
    const uint32_t label_nnz_capacity_per_route =
        max_query_label_nnz_per_route(runtime.query_labels_matrix);
    if (route_count > runtime.query_group_ids.size() ||
        route_count > runtime.route_to_original_query.size()) {
        fail_full_gpu("route metadata is smaller than full_gpu route count");
    }
    if (options.latency_query_start >= static_cast<int>(runtime.original_query_count)) {
        fail_full_gpu("latency_query_start is beyond original query count");
    }

    const uint32_t query_window_start = static_cast<uint32_t>(options.latency_query_start);
    const uint32_t remaining_queries = runtime.original_query_count - query_window_start;
    const uint32_t query_window_count = options.latency_query_limit == 0
        ? remaining_queries
        : std::min(static_cast<uint32_t>(options.latency_query_limit), remaining_queries);
    if (query_window_count == 0) {
        fail_full_gpu("latency query window is empty");
    }
    const std::vector<std::vector<uint32_t>> routes_by_original_query =
        build_routes_by_original_query(runtime);
    const uint32_t max_routes_per_request =
        max_routes_per_original_query(routes_by_original_query, query_window_start, query_window_count);
    const size_t latency_shared_bytes = runtime.vector_scalar_type == DenseVectorScalarType::Uint8
        ? static_cast<size_t>(options.dim) * sizeof(uint8_t)
        : static_cast<size_t>(options.dim) * sizeof(float);
    uint32_t auto_latency_concurrency = 0;
    dispatch_full_gpu_kernel_by_queue_size(options.search_queue_size, [&]<int kQueueSize>() {
        if (runtime.vector_scalar_type == DenseVectorScalarType::Uint8) {
            auto_latency_concurrency = resident_worker_capacity_for_latency_kernel(
                persistent_latency_queue_kernel<kQueueSize, uint8_t>,
                kHighfreqBlockThreads,
                latency_shared_bytes,
                "cudaOccupancyMaxActiveBlocksPerMultiprocessor full_gpu latency u8",
                "cudaGetDeviceProperties full_gpu latency occupancy");
        } else {
            auto_latency_concurrency = resident_worker_capacity_for_latency_kernel(
                persistent_latency_queue_kernel<kQueueSize, float>,
                kHighfreqBlockThreads,
                latency_shared_bytes,
                "cudaOccupancyMaxActiveBlocksPerMultiprocessor full_gpu latency f32",
                "cudaGetDeviceProperties full_gpu latency occupancy");
        }
    });
    uint32_t max_requested_route_slots = 1;
    for (int batch_size : options.latency_batch_sizes) {
        max_requested_route_slots = std::max(max_requested_route_slots,
                                             static_cast<uint32_t>(batch_size) * max_routes_per_request);
    }
    for (int concurrency_level : options.latency_concurrency_levels) {
        const uint32_t resolved_concurrency = concurrency_level == 0
            ? auto_latency_concurrency
            : static_cast<uint32_t>(concurrency_level);
        max_requested_route_slots = std::max(
            max_requested_route_slots,
            resolved_concurrency * max_routes_per_request);
    }
    const uint32_t latency_route_capacity = std::max(route_count, max_requested_route_slots);

    const UInt32Csr query_label_csr =
        build_sorted_query_label_csr_without_primary(runtime.query_labels_matrix,
                                                     runtime.query_filter_labels);
    const QueryFilterFastPath query_filter_fast_path =
        build_query_filter_fast_path(query_label_csr, runtime.query_ranges);
    const UInt32Csr sample_label_csr = build_sorted_label_csr(runtime.data_labels);

    FullGpuDeviceRuntime device;
    upload_runtime(runtime,
                   query_label_csr,
                   query_filter_fast_path,
                   sample_label_csr,
                   options.latency_online_metadata,
                   options.use_fp16,
                   device);
    device.route_results.allocate(static_cast<size_t>(latency_route_capacity) * topk,
                                  "cudaMalloc full_gpu latency route_results");
    device.route_distances.allocate(static_cast<size_t>(latency_route_capacity) * topk,
                                    "cudaMalloc full_gpu latency route_distances");
    device.route_steps.allocate(latency_route_capacity, "cudaMalloc full_gpu latency route_steps");
    device.latency_route_indices.allocate(latency_route_capacity, "cudaMalloc full_gpu latency route ids");
    device.latency_high_query_indices.allocate(latency_route_capacity, "cudaMalloc full_gpu latency high ids");
    device.latency_low_query_indices.allocate(latency_route_capacity, "cudaMalloc full_gpu latency low ids");
    device.latency_query_vector_ids.allocate(latency_route_capacity,
                                             "cudaMalloc full_gpu latency query vector ids");
    device.latency_query_group_ids.allocate(latency_route_capacity,
                                            "cudaMalloc full_gpu latency query group ids");
    device.latency_start_local_ids.allocate(latency_route_capacity,
                                            "cudaMalloc full_gpu latency start local ids");
    device.latency_query_filter_bits.allocate(latency_route_capacity,
                                              "cudaMalloc full_gpu latency query filter bits");
    device.latency_query_filter_modes.allocate(latency_route_capacity,
                                               "cudaMalloc full_gpu latency query filter modes");
    device.latency_query_single_extra_labels.allocate(
        latency_route_capacity,
        "cudaMalloc full_gpu latency query single extra labels");
    device.latency_query_active_range_counts.allocate(
        latency_route_capacity,
        "cudaMalloc full_gpu latency query active range counts");
    device.latency_query_label_indptr.allocate(static_cast<size_t>(latency_route_capacity) + 1,
                                               "cudaMalloc full_gpu latency query label indptr");
    device.latency_query_label_indices.allocate(
        static_cast<size_t>(latency_route_capacity) * label_nnz_capacity_per_route,
        "cudaMalloc full_gpu latency query label indices");
    device.latency_query_range_modes.allocate(
        static_cast<size_t>(latency_route_capacity) * runtime.query_ranges.range_count,
        "cudaMalloc full_gpu latency query range modes");
    device.latency_query_range_lower_bounds.allocate(
        static_cast<size_t>(latency_route_capacity) * runtime.query_ranges.range_count,
        "cudaMalloc full_gpu latency query range lower bounds");
    device.latency_query_range_upper_bounds.allocate(
        static_cast<size_t>(latency_route_capacity) * runtime.query_ranges.range_count,
        "cudaMalloc full_gpu latency query range upper bounds");
    device.latency_results.allocate(static_cast<size_t>(latency_route_capacity) * topk,
                                    "cudaMalloc full_gpu latency compact results");
    device.latency_distances.allocate(static_cast<size_t>(latency_route_capacity) * topk,
                                      "cudaMalloc full_gpu latency compact distances");

    std::vector<idx_t> compact_ids;
    std::vector<float> compact_distances;
    std::vector<std::vector<idx_t>> merged_results;

    cudaEvent_t gpu_start = nullptr;
    cudaEvent_t gpu_stop = nullptr;
    check_cuda(cudaEventCreate(&gpu_start), "cudaEventCreate full_gpu latency start");
    check_cuda(cudaEventCreate(&gpu_stop), "cudaEventCreate full_gpu latency stop");

    std::ofstream csv(options.latency_output_path);
    if (!csv.is_open()) {
        fail_full_gpu("could not open latency output file " + options.latency_output_path);
    }
    if (options.latency_staggered) {
        const std::vector<int>& concurrency_levels = options.latency_concurrency_levels.empty()
            ? options.latency_batch_sizes
            : options.latency_concurrency_levels;
        write_staggered_latency_header(csv);

        try {
            for (int concurrency_level : concurrency_levels) {
                size_t cursor = 0;
                const uint32_t requested_concurrency = concurrency_level == 0
                    ? auto_latency_concurrency
                    : static_cast<uint32_t>(concurrency_level);
                if (requested_concurrency == 0) {
                    fail_full_gpu("latency concurrency level resolved to zero");
                }
                if (options.latency_persistent_queue) {
                    const uint32_t slot_route_capacity = max_routes_per_request;
                    PersistentLatencyQueue queue =
                        create_persistent_latency_queue(requested_concurrency,
                                                        slot_route_capacity,
                                                        label_nnz_capacity_per_route,
                                                        runtime.query_ranges.range_count,
                                                        topk);
                    std::vector<PersistentLatencySlot> slots =
                        create_persistent_latency_slots(requested_concurrency,
                                                        slot_route_capacity,
                                                        topk);

                    auto prepare_next_batch = [&](PersistentLatencySlot& slot) {
                        slot.outer_start = std::chrono::steady_clock::now();
                        LatencyBatchRoutes batch =
                            prepare_latency_batch_routes(runtime,
                                                         routes_by_original_query,
                                                         query_window_start,
                                                         query_window_count,
                                                         cursor,
                                                         1,
                                                         !options.latency_run_all_queries,
                                                         options.latency_time_planner,
                                                         options.routing_mode);
                        slot.planner_stop = std::chrono::steady_clock::now();
                        if (batch.original_count == 0) {
                            return false;
                        }
                        cursor += 1;
                        if (batch.route_ids.size() > slot_route_capacity) {
                            fail_full_gpu("persistent latency route count exceeds slot route capacity");
                        }
                        submit_persistent_latency_slot(runtime,
                                                       options,
                                                       device,
                                                       queue,
                                                       slot,
                                                       batch,
                                                       slot_route_capacity,
                                                       label_nnz_capacity_per_route,
                                                       topk);
                        return true;
                    };

                    auto run_persistent_closed_loop = [&](size_t request_count,
                                                         std::vector<LatencyBatchMeasurement>* out) {
                        size_t submitted = 0;
                        size_t completed = 0;
                        for (PersistentLatencySlot& slot : slots) {
                            if (submitted >= request_count || !prepare_next_batch(slot)) {
                                break;
                            }
                            ++submitted;
                        }

                        while (completed < submitted) {
                            bool made_progress = false;
                            for (PersistentLatencySlot& slot : slots) {
                                if (!slot.busy) {
                                    continue;
                                }
                                bool slot_ready = !slot.queued_to_gpu;
                                if (!slot_ready) {
                                    const volatile PersistentLatencyCompletionControl& completion =
                                        queue.device.completions.host_data()[slot.slot_index];
                                    slot_ready = completion.done_seq == slot.ready_seq;
                                    if (slot_ready) {
                                        std::atomic_thread_fence(std::memory_order_acquire);
                                    }
                                }
                                if (!slot_ready) {
                                    continue;
                                }
                                LatencyBatchMeasurement measurement =
                                    finish_persistent_latency_slot(runtime,
                                                                   options,
                                                                   device,
                                                                   queue,
                                                                   slot);
                                if (out != nullptr) {
                                    out->push_back(measurement);
                                }
                                ++completed;
                                made_progress = true;
                                if (submitted < request_count && prepare_next_batch(slot)) {
                                    ++submitted;
                                }
                            }
                            if (!made_progress) {
                                std::this_thread::yield();
                            }
                        }
                    };

                    try {
                        launch_persistent_latency_worker(runtime,
                                                         options,
                                                         device,
                                                         queue,
                                                         slot_route_capacity);
                        if (options.warmup_runs > 0) {
                            run_persistent_closed_loop(static_cast<size_t>(options.warmup_runs),
                                                       nullptr);
                        }
                        if (options.latency_run_all_queries) {
                            cursor = 0;
                        }

                        const size_t measured_request_count = options.latency_run_all_queries
                            ? static_cast<size_t>(query_window_count)
                            : static_cast<size_t>(options.timed_runs) * requested_concurrency;
                        std::vector<LatencyBatchMeasurement> measurements;
                        measurements.reserve(measured_request_count);
                        const auto closed_loop_start = std::chrono::steady_clock::now();
                        run_persistent_closed_loop(measured_request_count, &measurements);
                        const auto closed_loop_stop = std::chrono::steady_clock::now();
                        const double closed_loop_elapsed_ms =
                            std::chrono::duration<double, std::milli>(closed_loop_stop -
                                                                       closed_loop_start).count();
                        stop_persistent_latency_worker(queue);
                        destroy_persistent_latency_queue(queue);

                        write_staggered_latency_row(csv,
                                                    requested_concurrency,
                                                    options.warmup_runs,
                                                    options.timed_runs,
                                                    options.latency_run_all_queries,
                                                    options.latency_time_planner,
                                                    options.latency_online_metadata,
                                                    "persistent_queue",
                                                    true,
                                                    query_window_count,
                                                    closed_loop_elapsed_ms,
                                                    measurements);
                    } catch (...) {
                        stop_persistent_latency_worker(queue);
                        destroy_persistent_latency_queue(queue);
                        throw;
                    }
                    continue;
                }
                for (int warmup = 0; warmup < options.warmup_runs; ++warmup) {
                    const LatencyBatchRoutes batch =
                        prepare_latency_batch_routes(runtime,
                                                     routes_by_original_query,
                                                     query_window_start,
                                                     query_window_count,
                                                     cursor,
                                                     1,
                                                     true,
                                                     options.latency_time_planner,
                                                     options.routing_mode);
                    const auto e2e_start = std::chrono::steady_clock::now();
                    (void)run_latency_batch(runtime,
                                            options,
                                            device,
                                            batch,
                                            e2e_start,
                                            gpu_start,
                                            gpu_stop,
                                            compact_ids,
                                            compact_distances,
                                            merged_results);
                    cursor += 1;
                }
                if (options.latency_run_all_queries) {
                    cursor = 0;
                }

                const size_t measured_request_count = options.latency_run_all_queries
                    ? static_cast<size_t>(query_window_count)
                    : static_cast<size_t>(options.timed_runs) * requested_concurrency;
                std::vector<LatencyBatchMeasurement> measurements;
                measurements.reserve(measured_request_count);

                const auto closed_loop_start = std::chrono::steady_clock::now();
                if (options.latency_online_metadata) {
                    const uint32_t slot_route_capacity = max_routes_per_request;
                    std::vector<AsyncLatencySlot> slots =
                        create_async_latency_slots(requested_concurrency,
                                                   slot_route_capacity,
                                                   topk);
                    size_t submitted = 0;
                    size_t completed = 0;
                    auto prepare_next_batch = [&](AsyncLatencySlot& slot) {
                        slot.outer_start = std::chrono::steady_clock::now();
                        LatencyBatchRoutes batch =
                            prepare_latency_batch_routes(runtime,
                                                         routes_by_original_query,
                                                         query_window_start,
                                                         query_window_count,
                                                         cursor,
                                                         1,
                                                         !options.latency_run_all_queries,
                                                         options.latency_time_planner,
                                                         options.routing_mode);
                        slot.planner_stop = std::chrono::steady_clock::now();
                        if (batch.original_count == 0) {
                            return false;
                        }
                        cursor += 1;
                        if (batch.route_ids.size() > slot_route_capacity) {
                            fail_full_gpu("staggered latency route count exceeds slot route capacity");
                        }
                        submit_async_latency_slot(runtime,
                                                  options,
                                                  device,
                                                  slot,
                                                  batch,
                                                  slot_route_capacity,
                                                  label_nnz_capacity_per_route,
                                                  topk);
                        ++submitted;
                        return true;
                    };

                    try {
                        for (AsyncLatencySlot& slot : slots) {
                            if (submitted >= measured_request_count || !prepare_next_batch(slot)) {
                                break;
                            }
                        }

                        while (completed < submitted) {
                            bool made_progress = false;
                            for (AsyncLatencySlot& slot : slots) {
                                if (!slot.busy) {
                                    continue;
                                }
                                bool slot_ready = slot.batch.route_ids.empty();
                                if (!slot_ready) {
                                    const cudaError_t status = cudaEventQuery(slot.done);
                                    if (status == cudaSuccess) {
                                        slot_ready = true;
                                    } else if (status == cudaErrorNotReady) {
                                        slot_ready = false;
                                    } else {
                                        check_cuda(status, "cudaEventQuery full_gpu staggered slot done");
                                    }
                                }
                                if (!slot_ready) {
                                    continue;
                                }
                                measurements.push_back(finish_async_latency_slot(runtime, options, slot));
                                ++completed;
                                made_progress = true;
                                if (submitted < measured_request_count && prepare_next_batch(slot)) {
                                    continue;
                                }
                            }
                            if (!made_progress) {
                                std::this_thread::yield();
                            }
                        }
                    } catch (...) {
                        for (AsyncLatencySlot& slot : slots) {
                            if (slot.stream != nullptr) {
                                check_cuda(cudaStreamSynchronize(slot.stream),
                                           "cudaStreamSynchronize full_gpu staggered slot cleanup");
                            }
                        }
                        destroy_async_latency_slots(slots);
                        throw;
                    }
                    destroy_async_latency_slots(slots);
                } else {
                    for (size_t timed = 0; timed < measured_request_count; ++timed) {
                        const auto outer_start = std::chrono::steady_clock::now();
                        const LatencyBatchRoutes batch =
                            prepare_latency_batch_routes(runtime,
                                                         routes_by_original_query,
                                                         query_window_start,
                                                         query_window_count,
                                                         cursor,
                                                         1,
                                                         !options.latency_run_all_queries,
                                                         options.latency_time_planner,
                                                         options.routing_mode);
                        const auto planner_stop = std::chrono::steady_clock::now();
                        if (batch.original_count == 0) {
                            break;
                        }
                        const auto e2e_start = options.latency_time_planner ? planner_stop : outer_start;
                        measurements.push_back(run_latency_batch(runtime,
                                                                 options,
                                                                 device,
                                                                 batch,
                                                                 e2e_start,
                                                                 gpu_start,
                                                                 gpu_stop,
                                                                 compact_ids,
                                                                 compact_distances,
                                                                 merged_results));
                        const auto outer_stop = std::chrono::steady_clock::now();
                        if (options.latency_time_planner) {
                            measurements.back().planner_ms =
                                std::chrono::duration<double, std::milli>(planner_stop - outer_start).count();
                            measurements.back().total_with_planner_ms =
                                std::chrono::duration<double, std::milli>(outer_stop - outer_start).count();
                        }
                        cursor += 1;
                    }
                }
                const auto closed_loop_stop = std::chrono::steady_clock::now();
                const double closed_loop_elapsed_ms =
                    std::chrono::duration<double, std::milli>(closed_loop_stop - closed_loop_start).count();

                write_staggered_latency_row(csv,
                                            requested_concurrency,
                                            options.warmup_runs,
                                            options.timed_runs,
                                            options.latency_run_all_queries,
                                            options.latency_time_planner,
                                            options.latency_online_metadata,
                                            options.latency_online_metadata
                                                ? "async_slot_closed_loop"
                                                : "host_serial_closed_loop",
                                            options.latency_online_metadata,
                                            query_window_count,
                                            closed_loop_elapsed_ms,
                                            measurements);
            }
        } catch (...) {
            check_cuda(cudaEventDestroy(gpu_start), "cudaEventDestroy full_gpu latency start");
            check_cuda(cudaEventDestroy(gpu_stop), "cudaEventDestroy full_gpu latency stop");
            throw;
        }

        check_cuda(cudaEventDestroy(gpu_start), "cudaEventDestroy full_gpu latency start");
        check_cuda(cudaEventDestroy(gpu_stop), "cudaEventDestroy full_gpu latency stop");
        std::printf("FullGPU staggered latency CSV written to %s\n", options.latency_output_path.c_str());
        return 0;
    }
    csv << "batch_size,warmup_runs,timed_runs,run_all_queries,time_planner,online_metadata,query_window_count,"
        << "measured_batches,submitted_original_queries,"
        << "submitted_routes,avg_routes_per_batch,avg_high_routes_per_batch,"
        << "avg_low_routes_per_batch,avg_unroutable_originals_per_batch,"
        << "planner_avg_ms,planner_min_ms,planner_p50_ms,planner_p90_ms,planner_p95_ms,planner_p99_ms,planner_max_ms,"
        << "metadata_avg_ms,metadata_min_ms,metadata_p50_ms,metadata_p90_ms,metadata_p95_ms,metadata_p99_ms,metadata_max_ms,"
        << "e2e_avg_ms,e2e_min_ms,e2e_p50_ms,e2e_p90_ms,e2e_p95_ms,e2e_p99_ms,e2e_max_ms,"
        << "total_with_planner_avg_ms,total_with_planner_min_ms,total_with_planner_p50_ms,total_with_planner_p90_ms,total_with_planner_p95_ms,total_with_planner_p99_ms,total_with_planner_max_ms,"
        << "gpu_avg_ms,gpu_min_ms,gpu_p50_ms,gpu_p90_ms,gpu_p95_ms,gpu_p99_ms,gpu_max_ms\n";

    const std::string trace_output_path = options.latency_trace_output_path.empty()
        ? (options.latency_run_all_queries ? default_latency_trace_output_path(options.latency_output_path)
                                           : std::string())
        : options.latency_trace_output_path;
    std::ofstream trace_csv;
    if (!trace_output_path.empty()) {
        trace_csv.open(trace_output_path);
        if (!trace_csv.is_open()) {
            fail_full_gpu("could not open latency trace output file " + trace_output_path);
        }
        write_latency_trace_header(trace_csv);
    }

    try {
        for (int batch_size : options.latency_batch_sizes) {
            size_t cursor = 0;
            const uint32_t requested_batch_size = static_cast<uint32_t>(batch_size);
            for (int warmup = 0; warmup < options.warmup_runs; ++warmup) {
                const LatencyBatchRoutes batch =
                    prepare_latency_batch_routes(runtime,
                                                 routes_by_original_query,
                                                 query_window_start,
                                                 query_window_count,
                                                 cursor,
                                                 requested_batch_size,
                                                 true,
                                                 options.latency_time_planner,
                                                 options.routing_mode);
                const auto e2e_start = std::chrono::steady_clock::now();
                (void)run_latency_batch(runtime,
                                        options,
                                        device,
                                        batch,
                                        e2e_start,
                                        gpu_start,
                                        gpu_stop,
                                        compact_ids,
                                        compact_distances,
                                        merged_results);
                cursor += std::max(1u, batch.original_count);
            }
            if (options.latency_run_all_queries) {
                cursor = 0;
            }

            std::vector<LatencyBatchMeasurement> measurements;
            const size_t measured_batch_count = options.latency_run_all_queries
                ? (static_cast<size_t>(query_window_count) + requested_batch_size - 1) /
                      requested_batch_size
                : static_cast<size_t>(options.timed_runs);
            measurements.reserve(measured_batch_count);
            for (size_t timed = 0; timed < measured_batch_count; ++timed) {
                const auto outer_start = std::chrono::steady_clock::now();
                const LatencyBatchRoutes batch =
                    prepare_latency_batch_routes(runtime,
                                                 routes_by_original_query,
                                                 query_window_start,
                                                 query_window_count,
                                                 cursor,
                                                 requested_batch_size,
                                                 !options.latency_run_all_queries,
                                                 options.latency_time_planner,
                                                 options.routing_mode);
                const auto planner_stop = std::chrono::steady_clock::now();
                if (batch.original_count == 0) {
                    break;
                }
                const auto e2e_start = options.latency_time_planner ? planner_stop : outer_start;
                measurements.push_back(run_latency_batch(runtime,
                                                         options,
                                                         device,
                                                         batch,
                                                         e2e_start,
                                                         gpu_start,
                                                         gpu_stop,
                                                         compact_ids,
                                                         compact_distances,
                                                         merged_results));
                const auto outer_stop = std::chrono::steady_clock::now();
                if (options.latency_time_planner) {
                    measurements.back().planner_ms =
                        std::chrono::duration<double, std::milli>(planner_stop - outer_start).count();
                    measurements.back().total_with_planner_ms =
                        std::chrono::duration<double, std::milli>(outer_stop - outer_start).count();
                }
                if (trace_csv.is_open()) {
                    write_latency_trace_row(trace_csv,
                                            batch_size,
                                            timed,
                                            measurements.back());
                }
                cursor += std::max(1u, batch.original_count);
            }
            write_latency_summary_row(csv,
                                      batch_size,
                                      options.warmup_runs,
                                      options.timed_runs,
                                      options.latency_run_all_queries,
                                      options.latency_time_planner,
                                      options.latency_online_metadata,
                                      query_window_count,
                                      measurements);
        }
    } catch (...) {
        cudaEventDestroy(gpu_start);
        cudaEventDestroy(gpu_stop);
        throw;
    }

    check_cuda(cudaEventDestroy(gpu_start), "cudaEventDestroy full_gpu latency start");
    check_cuda(cudaEventDestroy(gpu_stop), "cudaEventDestroy full_gpu latency stop");
    std::printf("FullGPU latency CSV written to %s\n", options.latency_output_path.c_str());
    if (!trace_output_path.empty()) {
        std::printf("FullGPU latency trace CSV written to %s\n", trace_output_path.c_str());
    }
    return 0;
}

}  // namespace go_fann::backend_full_gpu
