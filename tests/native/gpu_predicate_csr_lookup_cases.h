#pragma once
#include <algorithm>
#include <cstdint>
#include <vector>

namespace csr_lookup_test {
struct Probe { uint32_t row, label; bool expected; };
struct Cases {
    std::vector<uint32_t> offsets{0}, labels;
    std::vector<Probe> probes;

    Cases() {
        // Every experimental threshold has adjacent lengths; the long rows
        // exercise binary search and heavily divergent candidate warps.
        for (uint32_t length : {0u, 1u, 2u, 7u, 8u, 9u, 15u, 16u, 17u,
                                31u, 32u, 33u, 63u, 64u, 65u, 1025u, 65537u}) {
            const uint32_t row = offsets.size() - 1;
            const uint32_t first = row % 3 == 0 ? 0 : row % 3 == 1 ? 65536 :
                UINT32_MAX - (length ? (length - 1) * 7 : 0);
            for (uint32_t i = 0; i < length; ++i) labels.push_back(first + 7 * i);
            offsets.push_back(labels.size());
            for (uint32_t key : {0u, 1u, 65535u, 65536u, 65537u, UINT32_MAX - 1, UINT32_MAX})
                add(row, key);
            const uint32_t samples = std::min(length, 65u);
            for (uint32_t i = 0; i < samples; ++i) {
                const uint32_t position = samples == 1 ? 0 :
                    uint64_t(i) * (length - 1) / (samples - 1);
                const uint32_t key = labels[offsets[row] + position];
                add(row, key);
                if (key) add(row, key - 1);
                if (key < UINT32_MAX) add(row, key + 1);
            }
        }
        add(offsets.size() - 1, 65536);  // first invalid sample
        add(UINT32_MAX, UINT32_MAX);
        while (probes.size() % 32 != 13) add(0, 7);  // partial CUDA warp
    }

    bool expected(uint32_t row, uint32_t key) const {
        return row < offsets.size() - 1 && std::binary_search(
            labels.begin() + offsets[row], labels.begin() + offsets[row + 1], key);
    }
    void add(uint32_t row, uint32_t key) { probes.push_back({row, key, expected(row, key)}); }
};
}  // namespace csr_lookup_test
