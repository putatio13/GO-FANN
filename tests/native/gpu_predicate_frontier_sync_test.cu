#include <cuda_runtime.h>
#include <cfloat>
#include <cstdio>
#include <cstdlib>

namespace {
struct Pair { float first; unsigned second; };
constexpr unsigned kBlocks = 1000;
constexpr unsigned kRepeats = 10;

void check(cudaError_t error) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA failure: %s\n", cudaGetErrorString(error));
        std::exit(2);
    }
}

// Native cooperative pickup takes size by value. A barrier inside this function
// cannot repair a size value captured before the producer finished writing it.
template<bool kBarrierInside>
__device__ int pickup(Pair* frontier, int size, int lane) {
    if constexpr (kBarrierInside) __syncwarp();
    int best = -1;
    float score = FLT_MAX;
    for (int i = lane + 1; i < size; i += 32) {
        if (best < 0 || frontier[i].first < score) {
            best = i;
            score = frontier[i].first;
        }
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        const int other = __shfl_down_sync(0xffffffffu, best, offset);
        const float other_score = __shfl_down_sync(0xffffffffu, score, offset);
        if (other >= 0 && (best < 0 || other_score < score ||
            (!(score < other_score) && !(other_score < score) && other < best))) {
            best = other;
            score = other_score;
        }
    }
    return __shfl_sync(0xffffffffu, best, 0);
}

template<bool kBarrierBefore, bool kBarrierInside = false>
__global__ void publication_fixture(unsigned* failures) {
    __shared__ Pair frontier[3];
    __shared__ int size;
    const int lane = threadIdx.x;
    if (lane == 0) {
        size = 1;
        frontier[1] = {100, 1};
        frontier[2] = {100, 2};
    }
    __syncwarp();
    // Widen the valid producer/consumer scheduling window without changing the
    // dependency: lane0 inserts; every lane then reads heap size and its stripe.
    if (lane == 0) {
        __nanosleep(100000);
        frontier[1] = {10, 1};
        frontier[2] = {0, 2};
        size = 3;
    }
    if constexpr (kBarrierBefore) __syncwarp();
    const int selected = pickup<kBarrierInside>(frontier, size, lane);
    if (lane == 0 && selected != 2) atomicAdd(failures, 1);
}
}

int main() {
    int device = 0;
    cudaDeviceProp properties{};
    check(cudaGetDevice(&device));
    check(cudaGetDeviceProperties(&properties, device));
    if (properties.major < 7) {
        std::fprintf(stderr, "This independent-thread-scheduling regression requires SM70+.\n");
        return 77;
    }
    unsigned* failures = nullptr;
    check(cudaMallocManaged(&failures, 3 * sizeof(unsigned)));
    failures[0] = failures[1] = failures[2] = 0;
    for (unsigned repeat = 0; repeat < kRepeats; ++repeat) {
        publication_fixture<false><<<kBlocks, 32>>>(failures);
        check(cudaGetLastError());
        publication_fixture<false, true><<<kBlocks, 32>>>(failures + 1);
        check(cudaGetLastError());
        publication_fixture<true><<<kBlocks, 32>>>(failures + 2);
        check(cudaGetLastError());
    }
    check(cudaDeviceSynchronize());
    std::printf("unsynchronized_failures=%u callee_barrier_failures=%u "
                "producer_barrier_failures=%u trials=%u\n",
                failures[0], failures[1], failures[2], kBlocks * kRepeats);
    // Require the negative controls to exercise the race, not merely a clean
    // positive control that could also pass without exposing the dependency.
    const bool passed = failures[0] > 0 && failures[1] > 0 && failures[2] == 0;
    check(cudaFree(failures));
    return passed ? 0 : 1;
}
