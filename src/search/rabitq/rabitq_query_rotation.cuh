#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace go_fann {

__global__ inline void rabitq_row_norms_kernel(const float* matrix, float* norms,
                                              uint32_t rows, uint32_t dim)
{
    const uint32_t row = blockIdx.x;
    if (row >= rows) return;
    float sum = 0.0f;
    for (uint32_t d = threadIdx.x; d < dim; d += blockDim.x) {
        const float value = matrix[static_cast<size_t>(row) * dim + d];
        sum += value * value;
    }
    __shared__ float partials[256];
    partials[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = 128; stride; stride >>= 1) {
        if (threadIdx.x < stride) partials[threadIdx.x] += partials[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) norms[row] = partials[0];
}

__global__ inline void rabitq_finish_centroid_distances_kernel(float* distances,
    const float* query_norms, const float* anchor_norms, uint32_t queries, uint32_t anchors)
{
    const size_t pair = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pair >= static_cast<size_t>(queries) * anchors) return;
    distances[pair] = fmaxf(0.0f, distances[pair] +
        query_norms[pair / anchors] + anchor_norms[pair % anchors]);
}

inline void check_query_rotation_blas(cublasStatus_t status, const char* operation)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed: cuBLAS status " +
                                 std::to_string(static_cast<int>(status)));
    }
}

// One handle per preparation, never constructed inside the search loop.
// Do not silently down-convert FP32 operands or select Tensor Core math.
class RabitqQueryRotation {
public:
    explicit RabitqQueryRotation(bool enabled) {
        if (!enabled) return;
        check_query_rotation_blas(cublasCreate(&handle_), "cublasCreate query rotation");
        try {
            check_query_rotation_blas(cublasSetMathMode(handle_, CUBLAS_PEDANTIC_MATH),
                                      "cublasSetMathMode query rotation");
            check_query_rotation_blas(cublasSetPointerMode(handle_, CUBLAS_POINTER_MODE_HOST),
                                      "cublasSetPointerMode query rotation");
        } catch (...) {
            cublasDestroy(handle_);
            handle_ = nullptr;
            throw;
        }
    }
    ~RabitqQueryRotation() { if (handle_) cublasDestroy(handle_); }
    RabitqQueryRotation(const RabitqQueryRotation&) = delete;
    RabitqQueryRotation& operator=(const RabitqQueryRotation&) = delete;

    void apply(const float* query, const float* rotation, float* output,
               uint32_t count, uint32_t dim, uint32_t padded_dim,
               cudaStream_t stream = nullptr) {
        constexpr auto max_dim = static_cast<uint32_t>(std::numeric_limits<int>::max());
        if (!handle_ || !query || !rotation || !output || !count || !dim ||
            padded_dim < dim || count > max_dim || dim > max_dim || padded_dim > max_dim) {
            throw std::invalid_argument("Invalid FP32 query rotation shape or state");
        }
        check_query_rotation_blas(cublasSetStream(handle_, stream), "cublasSetStream query rotation");
        const float alpha = 1.0f, beta = 0.0f;
        // Row-major Q[N,D] R[D,P] = Y[N,P]. Interpret the same storage as
        // column-major Y^T[P,N] = R^T[P,D] Q^T[D,N]; no transpose or copies.
        check_query_rotation_blas(cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            static_cast<int>(padded_dim), static_cast<int>(count), static_cast<int>(dim),
            &alpha, rotation, static_cast<int>(padded_dim), query, static_cast<int>(dim),
            &beta, output, static_cast<int>(padded_dim)), "cublasSgemm query rotation");
    }

    void centroid_distances(const float* query, const float* anchors, float* output,
                            float* query_norms, float* anchor_norms,
                            uint32_t count, uint32_t anchor_count, uint32_t dim,
                            cudaStream_t stream = nullptr) {
        constexpr auto max_dim = static_cast<uint32_t>(std::numeric_limits<int>::max());
        if (!handle_ || !query || !anchors || !output || !query_norms || !anchor_norms ||
            !count || !anchor_count || !dim || count > max_dim || anchor_count > max_dim || dim > max_dim)
            throw std::invalid_argument("Invalid FP32 centroid distance shape or state");
        check_query_rotation_blas(cublasSetStream(handle_, stream), "cublasSetStream centroid distances");
        rabitq_row_norms_kernel<<<count,256,0,stream>>>(query,query_norms,count,dim);
        rabitq_row_norms_kernel<<<anchor_count,256,0,stream>>>(anchors,anchor_norms,anchor_count,dim);
        // Column-major output[A,N] = -2 * anchors^T[A,D] * queries[D,N].
        const float alpha = -2.0f, beta = 0.0f;
        check_query_rotation_blas(cublasSgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            static_cast<int>(anchor_count), static_cast<int>(count), static_cast<int>(dim),
            &alpha, anchors, static_cast<int>(dim), query, static_cast<int>(dim),
            &beta, output, static_cast<int>(anchor_count)), "cublasSgemm centroid distances");
        const size_t pairs = static_cast<size_t>(count) * anchor_count;
        if ((pairs + 255) / 256 > static_cast<size_t>(std::numeric_limits<int>::max()))
            throw std::invalid_argument("Centroid distance grid too large");
        rabitq_finish_centroid_distances_kernel<<<(pairs+255)/256,256,0,stream>>>(
            output,query_norms,anchor_norms,count,anchor_count);
        const auto status = cudaGetLastError();
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    }
private:
    cublasHandle_t handle_ = nullptr;
};

// Record on the same stream; resolve after the existing final synchronization.
// Adjacent spans are device intervals, nested inside preprocessing wall time.
class RabitqPreprocessEvents {
public:
    explicit RabitqPreprocessEvents(bool enabled) : enabled_(enabled) {
        if (!enabled_) return;
        for (auto& event : events_) {
            const auto status = cudaEventCreate(&event);
            if (status != cudaSuccess) {
                release();
                throw std::runtime_error(cudaGetErrorString(status));
            }
        }
    }
    ~RabitqPreprocessEvents() { release(); }
    RabitqPreprocessEvents(const RabitqPreprocessEvents&) = delete;
    RabitqPreprocessEvents& operator=(const RabitqPreprocessEvents&) = delete;
    void record(size_t stage) {
        if (enabled_) {
            const auto status = cudaEventRecord(events_[stage], nullptr);
            if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
        }
    }
    double elapsed(size_t stage) const {
        float value = 0;
        if (enabled_) {
            const auto status = cudaEventElapsedTime(&value, events_[stage], events_[stage + 1]);
            if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
        }
        return value;
    }
private:
    void release() { for (auto& event : events_) if (event) { cudaEventDestroy(event); event = nullptr; } }
    bool enabled_;
    cudaEvent_t events_[6]{};
};

}  // namespace go_fann
