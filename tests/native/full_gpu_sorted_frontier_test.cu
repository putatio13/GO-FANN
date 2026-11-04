#include <cuda_runtime.h>

#include "search/full_gpu/full_gpu_sorted_frontier.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <random>
#include <set>
#include <vector>

namespace sf = go_fann::backend_full_gpu::sorted_frontier;

namespace {

void check_cuda(cudaError_t status, const char* what)
{
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(status));
        std::exit(2);
    }
}

bool host_less(const sf::Entry& a, const sf::Entry& b)
{
    const bool av = a.distance == a.distance;
    const bool bv = b.distance == b.distance;
    if (!av) return false;
    if (!bv) return true;
    const uint32_t ak = a.payload & sf::kKeyMask;
    const uint32_t bk = b.payload & sf::kKeyMask;
    return a.distance < b.distance || (a.distance == b.distance && ak < bk);
}

bool same_entry(const sf::Entry& a, const sf::Entry& b)
{
    return (std::isnan(a.distance) && std::isnan(b.distance)) ||
        (a.distance == b.distance && a.payload == b.payload);
}

sf::Entry make_entry(uint32_t route, uint32_t local, float distance, bool used)
{
    return sf::Entry{distance, sf::pack(route, local) | (used ? sf::kUsed : 0u)};
}

std::vector<sf::Entry> sorted_valid(std::vector<sf::Entry> values)
{
    values.erase(std::remove_if(values.begin(), values.end(), [](const sf::Entry& e) {
        return !(e.distance == e.distance);
    }), values.end());
    std::sort(values.begin(), values.end(), host_less);
    return values;
}

std::vector<sf::Entry> host_merge(const std::vector<sf::Entry>& old,
                                  const std::vector<sf::Entry>& batch,
                                  int capacity)
{
    std::vector<sf::Entry> output;
    output.reserve(std::min<int>(capacity, old.size() + batch.size()));
    std::merge(old.begin(), old.end(), batch.begin(), batch.end(),
               std::back_inserter(output), host_less);
    if (static_cast<int>(output.size()) > capacity) output.resize(capacity);
    return output;
}

void assert_vector(const std::vector<sf::Entry>& actual,
                   const std::vector<sf::Entry>& expected,
                   const char* label);

void assert_serial_helper(const std::vector<sf::Entry>& old,
                          const std::vector<sf::Entry>& batch,
                          int capacity,
                          const std::vector<sf::Entry>& expected)
{
    std::vector<sf::Entry> sorted_batch = batch;
    const int sorted_count = sf::sort_serial(sorted_batch.data(), static_cast<int>(sorted_batch.size()));
    std::vector<sf::Entry> merged(static_cast<size_t>(capacity));
    const int merged_count = sf::merge_serial(old.data(), static_cast<int>(old.size()),
                                              sorted_batch.data(), sorted_count,
                                              merged.data(), capacity);
    merged.resize(static_cast<size_t>(merged_count));
    assert_vector(merged, expected, "serial-helper");
}

__global__ void sort_and_merge_kernel(sf::Entry* batch,
                                      const sf::Entry* old,
                                      sf::Entry* destination,
                                      int old_count,
                                      int batch_width,
                                      int batch_count,
                                      int capacity)
{
    sf::sort_parallel(batch, static_cast<uint32_t>(batch_width));
    // Invalid NaN padding is sorted to the tail.  Only the valid prefix enters
    // the merge, matching the production admission contract.
    sf::merge_parallel(old, old_count, batch, batch_count, destination, capacity);
}

void assert_vector(const std::vector<sf::Entry>& actual,
                   const std::vector<sf::Entry>& expected,
                   const char* label)
{
    if (actual.size() != expected.size()) {
        std::fprintf(stderr, "%s: size %zu != %zu\n", label, actual.size(), expected.size());
        std::exit(3);
    }
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!same_entry(actual[i], expected[i])) {
            std::fprintf(stderr,
                         "%s[%zu]: got (%.8g, 0x%08x), expected (%.8g, 0x%08x)\n",
                         label, i, actual[i].distance, actual[i].payload,
                         expected[i].distance, expected[i].payload);
            std::exit(3);
        }
    }
}

std::vector<sf::Entry> gpu_merge(const std::vector<sf::Entry>& old,
                                 const std::vector<sf::Entry>& batch,
                                 int batch_width,
                                 int batch_count,
                                 int capacity)
{
    const size_t old_bytes = old.size() * sizeof(sf::Entry);
    const size_t batch_bytes = static_cast<size_t>(batch_width) * sizeof(sf::Entry);
    sf::Entry* d_old = nullptr;
    sf::Entry* d_batch = nullptr;
    sf::Entry* d_destination = nullptr;
    check_cuda(cudaMalloc(&d_old, old_bytes), "cudaMalloc old");
    check_cuda(cudaMalloc(&d_batch, batch_bytes), "cudaMalloc batch");
    check_cuda(cudaMalloc(&d_destination, static_cast<size_t>(capacity) * sizeof(sf::Entry)),
               "cudaMalloc destination");
    check_cuda(cudaMemcpy(d_old, old.data(), old_bytes, cudaMemcpyHostToDevice),
               "copy old");
    check_cuda(cudaMemcpy(d_batch, batch.data(), batch_bytes, cudaMemcpyHostToDevice),
               "copy batch");
    std::vector<sf::Entry> empty(static_cast<size_t>(capacity),
                                 sf::Entry{std::numeric_limits<float>::quiet_NaN(), 0});
    check_cuda(cudaMemcpy(d_destination, empty.data(), empty.size() * sizeof(sf::Entry),
                          cudaMemcpyHostToDevice), "clear destination");

    const int threads = std::max(32, std::max(batch_width, capacity));
    sort_and_merge_kernel<<<1, threads>>>(d_batch, d_old, d_destination,
                                          static_cast<int>(old.size()), batch_width,
                                          batch_count, capacity);
    check_cuda(cudaGetLastError(), "sort_and_merge launch");
    check_cuda(cudaDeviceSynchronize(), "sort_and_merge synchronize");

    std::vector<sf::Entry> result(static_cast<size_t>(std::min<int>(capacity,
        static_cast<int>(old.size()) + batch_count)));
    if (!result.empty()) {
        check_cuda(cudaMemcpy(result.data(), d_destination,
                              result.size() * sizeof(sf::Entry), cudaMemcpyDeviceToHost),
                   "copy destination");
    }
    cudaFree(d_old);
    cudaFree(d_batch);
    cudaFree(d_destination);
    return result;
}

void run_single_round_matrix()
{
    std::mt19937 rng(20260915u);
    int cases = 0;
    for (int capacity : {16, 128, 256, 512}) {
        for (int width : {32, 64, 128}) {
            for (int round = 0; round < 6; ++round) {
                std::vector<sf::Entry> old;
                old.reserve(capacity);
                for (int i = 0; i < capacity / 2; ++i) {
                    // Route is part of the identity, so equal local rows in
                    // different routes remain independent candidates.
                    const uint32_t route = static_cast<uint32_t>(i % 16);
                    const uint32_t local = static_cast<uint32_t>(0x07ffffffu - i);
                    const float score = (i % 7 == 0) ? std::numeric_limits<float>::infinity()
                                                     : static_cast<float>(i % 5);
                    old.push_back(make_entry(route, local, score, (i & 1) != 0));
                }
                old = sorted_valid(std::move(old));

                std::vector<sf::Entry> batch(static_cast<size_t>(width),
                    sf::Entry{std::numeric_limits<float>::quiet_NaN(), 0});
                const int valid_count = std::min(width, 11 + round);
                std::set<uint32_t> occupied;
                for (const sf::Entry& entry : old) occupied.insert(entry.payload & sf::kKeyMask);
                for (int i = 0; i < valid_count; ++i) {
                    const uint32_t route = static_cast<uint32_t>((i + 3 * round) % 16);
                    const uint32_t local = static_cast<uint32_t>(0x06000000u + round * 1024 + i);
                    const uint32_t key = sf::pack(route, local) & sf::kKeyMask;
                    if (!occupied.insert(key).second) std::abort();
                    float score = (i % 4 == 0) ? 1.0f : static_cast<float>((i + round) % 6);
                    if (i == valid_count - 1 && round == 5) score = -std::numeric_limits<float>::infinity();
                    batch[static_cast<size_t>(i)] = make_entry(route, local, score, (i & 1) == 0);
                }

                std::vector<sf::Entry> expected_batch = batch;
                std::sort(expected_batch.begin(), expected_batch.begin() + valid_count, host_less);
                const int output_count = std::min(capacity, static_cast<int>(old.size()) + valid_count);
                const auto expected = host_merge(old,
                    std::vector<sf::Entry>(expected_batch.begin(), expected_batch.begin() + valid_count),
                    capacity);
                assert_serial_helper(old, batch, capacity, expected);
                const auto actual = gpu_merge(old, batch, width, valid_count, capacity);
                assert_vector(actual, expected, "single-round");
                ++cases;
                (void)output_count;
            }
        }
    }
    std::printf("sorted frontier CUDA matrix: %d cases passed (seed=20260915)\n", cases);
}

void run_trace()
{
    constexpr int capacity = 128;
    constexpr int width = 64;
    std::mt19937 rng(7u);
    std::vector<sf::Entry> queue;
    for (int i = 0; i < 64; ++i) {
        queue.push_back(make_entry(static_cast<uint32_t>(i % 4),
                                   static_cast<uint32_t>(1000 + i),
                                   static_cast<float>(i % 13), false));
    }
    queue = sorted_valid(std::move(queue));
    std::set<uint32_t> seen;
    for (const sf::Entry& e : queue) seen.insert(e.payload & sf::kKeyMask);
    for (int round = 0; round < 12; ++round) {
        int parent = -1;
        for (int i = 0; i < static_cast<int>(queue.size()); ++i) {
            if ((queue[static_cast<size_t>(i)].payload & sf::kUsed) == 0) {
                parent = i;
                break;
            }
        }
        if (parent >= 0) queue[static_cast<size_t>(parent)].payload |= sf::kUsed;

        std::vector<sf::Entry> batch(static_cast<size_t>(width),
            sf::Entry{std::numeric_limits<float>::quiet_NaN(), 0});
        const int valid_count = 20 + (round % 13);
        for (int i = 0; i < valid_count; ++i) {
            uint32_t route = static_cast<uint32_t>((round + i) % 16);
            uint32_t local = static_cast<uint32_t>(0x01000000u + round * 1000 + i);
            uint32_t key = sf::pack(route, local) & sf::kKeyMask;
            while (!seen.insert(key).second) {
                ++local;
                key = sf::pack(route, local) & sf::kKeyMask;
            }
            batch[static_cast<size_t>(i)] = make_entry(route, local,
                std::uniform_real_distribution<float>(-2.0f, 15.0f)(rng), (i & 3) == 0);
        }
        const auto sorted_batch = [&] {
            auto copy = batch;
            std::sort(copy.begin(), copy.begin() + valid_count, host_less);
            return std::vector<sf::Entry>(copy.begin(), copy.begin() + valid_count);
        }();
        const auto expected = host_merge(queue, sorted_batch, capacity);
        const auto actual = gpu_merge(queue, batch, width, valid_count, capacity);
        assert_vector(actual, expected, "multi-round trace");
        queue = actual;
        std::printf("trace round=%d parent=%d queue=%zu\n", round, parent, queue.size());
    }
    std::printf("sorted frontier multi-round parent trace passed (12 rounds)\n");
}

void run_edge_cases()
{
    constexpr int capacity = 128;
    constexpr int width = 32;
    std::vector<sf::Entry> old;
    for (uint32_t route = 0; route < 16; ++route) {
        // Identical local rows across routes exercise the packed route/local
        // identity.  Every queue entry is already consumed.
        old.push_back(make_entry(route, 0x07ffffffu,
                                 route == 15 ? std::numeric_limits<float>::infinity()
                                             : static_cast<float>(route), true));
    }
    old = sorted_valid(std::move(old));
    std::vector<sf::Entry> batch(static_cast<size_t>(width),
        sf::Entry{std::numeric_limits<float>::quiet_NaN(), 0});
    batch[0] = make_entry(0, 0x1234u, -std::numeric_limits<float>::infinity(), false);
    batch[1] = make_entry(7, 0x1234u, 1.0f, true);
    batch[2] = make_entry(15, 0x1234u, 1.0f, false);
    const auto sorted_batch = [&] {
        auto copy = batch;
        std::sort(copy.begin(), copy.begin() + 3, host_less);
        return std::vector<sf::Entry>(copy.begin(), copy.begin() + 3);
    }();
    const auto expected = host_merge(old, sorted_batch, capacity);
    assert_serial_helper(old, batch, capacity, expected);
    assert_vector(gpu_merge(old, batch, width, 3, capacity), expected, "edge-case");

    // An entirely invalid batch must leave an all-used queue unchanged.
    std::vector<sf::Entry> invalid(static_cast<size_t>(width),
        sf::Entry{std::numeric_limits<float>::quiet_NaN(), 0});
    const auto unchanged = host_merge(old, {}, capacity);
    assert_vector(gpu_merge(old, invalid, width, 0, capacity), unchanged,
                  "all-invalid-batch");
    std::printf("sorted frontier edge cases passed (all-used, all-invalid, max key)\n");
}

}  // namespace

int main()
{
    run_single_round_matrix();
    run_trace();
    run_edge_cases();
    return 0;
}
