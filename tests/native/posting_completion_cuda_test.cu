#include "execution/posting_completion.cuh"
#include "execution/posting_completion.h"

#include <cassert>
#include <cstdio>
#include <future>

namespace {
constexpr uint32_t rows = 2048;
constexpr uint32_t width = 100;

void check(cudaError_t status)
{
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

__global__ void produce(uint32_t* ids, float* distances, uint32_t* ready, uint32_t generation)
{
    const uint32_t row = blockIdx.x;
    if (threadIdx.x == 0) {
        const auto start = clock64();
        while (clock64() - start < (row % 19) * 1000) {}
    }
    __syncwarp();
    for (uint32_t i = threadIdx.x; i < width; i += blockDim.x) {
        const uint32_t value = (generation * rows + row) * width + i;
        ids[row * width + i] = value;
        distances[row * width + i] = static_cast<float>(value);
    }
    go_fann::posting_completion::publish(ready, row);
}
}  // namespace

int main()
{
    uint32_t *host_ids = nullptr, *device_ids = nullptr, *host_ready = nullptr, *device_ready = nullptr;
    float *host_distances = nullptr, *device_distances = nullptr;
    try {
        check(cudaHostAlloc(&host_ids, rows * width * sizeof(uint32_t), cudaHostAllocMapped));
        check(cudaHostAlloc(&host_distances, rows * width * sizeof(float), cudaHostAllocMapped));
        check(cudaHostAlloc(&host_ready, rows * sizeof(uint32_t), cudaHostAllocMapped));
        check(cudaHostGetDevicePointer(&device_ids, host_ids, 0));
        check(cudaHostGetDevicePointer(&device_distances, host_distances, 0));
        check(cudaHostGetDevicePointer(&device_ready, host_ready, 0));
        std::vector<uint32_t> originals(rows);
        for (uint32_t row = 0; row < rows; ++row) originals[row] = row / 17;
        const auto groups = go_fann::posting_completion::Groups::make(originals, rows);
        for (uint32_t generation = 0; generation < 8; ++generation) {
            std::fill_n(host_ready, rows, 0U);
            std::fill_n(host_ids, rows * width, UINT32_MAX);
            std::atomic_thread_fence(std::memory_order_seq_cst);
            std::atomic<bool> cancelled{false};
            size_t seen = 0;
            auto consumer = std::async(std::launch::async, [&] {
                go_fann::posting_completion::consume(groups, 8, cancelled,
                    [&](size_t row) { return __atomic_load_n(host_ready + row, __ATOMIC_ACQUIRE) != 0; },
                    [&](const std::vector<size_t>& positions) {
                        for (size_t row : positions) {
                            for (uint32_t i = 0; i < width; ++i) {
                                const uint32_t expected = (generation * rows + row) * width + i;
                                if (host_ids[row * width + i] != expected ||
                                    host_distances[row * width + i] != static_cast<float>(expected))
                                    throw std::runtime_error("ready preceded complete payload");
                            }
                            ++seen;
                        }
                    });
            });
            try {
                produce<<<rows, 32>>>(device_ids, device_distances, device_ready, generation);
                check(cudaGetLastError());
                check(cudaDeviceSynchronize());
                for (uint32_t row = 0; row < rows; ++row)
                    if (!__atomic_load_n(host_ready + row, __ATOMIC_ACQUIRE))
                        throw std::runtime_error("producer missed a ready row");
                consumer.get();
            } catch (...) {
                cancelled.store(true, std::memory_order_release);
                if (consumer.valid()) consumer.wait();
                throw;
            }
            assert(seen == rows);
        }
        check(cudaFreeHost(host_ids)); host_ids = nullptr;
        check(cudaFreeHost(host_distances)); host_distances = nullptr;
        check(cudaFreeHost(host_ready)); host_ready = nullptr;
        std::puts("Posting completion CUDA publication tests passed (8 changing generations)");
        return 0;
    } catch (const std::exception& error) {
        if (host_ids) cudaFreeHost(host_ids);
        if (host_distances) cudaFreeHost(host_distances);
        if (host_ready) cudaFreeHost(host_ready);
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
