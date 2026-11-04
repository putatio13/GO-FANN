#include "predicates/gpu_predicate_csr.h"
#include "predicates/gpu_predicate_index.cuh"
#include <algorithm>
#include <cassert>
#include <functional>
#include <random>

using namespace go_fann::gpu_predicate;
int main()
{
    const std::vector<uint64_t> offsets{0, 3, 3, 5};
    const std::vector<uint32_t> labels{0, 65536, 7330000, 7, 65535};
    const auto compact = validate_csr_offsets(3, 7330001, offsets, labels, 3);
    View v;
    v.rows = 3; v.csr_offsets = compact.data(); v.csr_labels = labels.data();
    assert(enabled(v, 0));
    for (uint32_t row = 0; row <= 3; ++row) {
        for (uint32_t label : {0u, 7u, 65535u, 65536u, 7330000u, UINT32_MAX}) {
            const bool expected = row < 3 && std::binary_search(
                labels.begin() + offsets[row], labels.begin() + offsets[row + 1], label);
            assert(contains(v, label, row) == expected);
        }
    }
    const auto rejects = [](const std::function<void()>& run) {
        bool rejected = false;
        try { run(); } catch (const std::runtime_error&) { rejected = true; }
        assert(rejected);
    };
    rejects([&] { validate_csr_offsets(3, 7330001, offsets, labels, 4); });
    rejects([&] { validate_csr_offsets(4, 7330001, offsets, labels, 4); });
    rejects([&] { validate_csr_offsets(3, 7330000, offsets, labels, 3); });
    rejects([&] { validate_csr_offsets(3, 7330001, {1, 3, 3, 5}, labels, 3); });
    rejects([&] { validate_csr_offsets(3, 7330001, {0, 4, 3, 5}, labels, 3); });
    rejects([&] { validate_csr_offsets(3, 7330001, {0, 3, 6, 5}, labels, 3); });
    rejects([&] { validate_csr_offsets(3, 7330001, {0, 3, 3, uint64_t(UINT32_MAX) + 1}, labels, 3); });
    rejects([&] { validate_csr_offsets(1, 8, {0, 2}, {7, 0}, 1); });
    rejects([&] { validate_csr_offsets(1, 8, {0, 2}, {7, 7}, 1); });
    assert(validate_csr_offsets(0, 0, {0}, {}, 0) == std::vector<uint32_t>{0});
    rejects([&] { validate_csr_offsets(0, 0, {}, {}, 0); });

    // Include long rows, empty rows and original IDs outside uint16.
    std::mt19937 random(729);
    std::vector<uint64_t> random_offsets{0};
    std::vector<uint32_t> random_labels;
    for (uint32_t row = 0; row < 128; ++row) {
        std::vector<uint32_t> values;
        for (uint32_t n = random() % 100; n; --n) values.push_back(random() % 8000000);
        std::sort(values.begin(), values.end());
        values.erase(std::unique(values.begin(), values.end()), values.end());
        random_labels.insert(random_labels.end(), values.begin(), values.end());
        random_offsets.push_back(random_labels.size());
    }
    const auto random_compact = validate_csr_offsets(128, 8000000, random_offsets, random_labels, 128);
    v.rows = 128; v.csr_offsets = random_compact.data(); v.csr_labels = random_labels.data();
    for (uint32_t row = 0; row < 128; ++row) {
        for (uint64_t i = random_offsets[row]; i < random_offsets[row + 1]; ++i)
            assert(contains(v, random_labels[i], row));
        for (unsigned n = 0; n < 100; ++n) {
            const uint32_t label = random() % 8000000;
            assert(contains(v, label, row) == std::binary_search(
                random_labels.begin() + random_offsets[row], random_labels.begin() + random_offsets[row + 1], label));
        }
    }
}
