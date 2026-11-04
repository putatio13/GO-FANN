#pragma once
#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace go_fann::gpu_predicate {
// Validate the complete source before narrowing. Labels remain borrowed from
// the runtime through synchronous upload; no pruning, remapping or label copy.
inline std::vector<uint32_t> validate_csr_offsets(
    uint32_t rows, uint32_t cols, const std::vector<uint64_t>& offsets,
    const std::vector<uint32_t>& labels, uint32_t expected_rows)
{
    const auto check = [](bool valid, const char* why) {
        if (!valid) throw std::runtime_error(std::string("GPU predicate CSR: ") + why);
    };
    check(rows == expected_rows, "sample domain differs from base vectors");
    check(offsets.size() == uint64_t(rows) + 1, "offset count differs from rows + 1");
    check(labels.size() <= UINT32_MAX, "label relation count exceeds uint32 offsets");
    check(offsets.front() == 0 && offsets.back() == labels.size(), "invalid first/terminal offset");
    std::vector<uint32_t> narrowed(offsets.size());
    for (std::size_t row = 0; row < offsets.size(); ++row) {
        check(offsets[row] <= labels.size() && offsets[row] <= UINT32_MAX, "offset out of bounds");
        if (row != 0) check(offsets[row - 1] <= offsets[row], "decreasing offsets");
        narrowed[row] = static_cast<uint32_t>(offsets[row]);
    }
    for (uint64_t row = 0; row < rows; ++row) {
        for (uint64_t i = offsets[row]; i < offsets[row + 1]; ++i) {
            check(labels[i] < cols, "label outside original token domain");
            if (i != offsets[row]) check(labels[i - 1] < labels[i], "row is not sorted unique");
        }
    }
    return narrowed;
}
}
