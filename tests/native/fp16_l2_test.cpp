#include "data/fp16_l2.h"

#include <cassert>
#include <cmath>
#include <vector>

namespace fp16 = go_fann::fp16_l2;

int main()
{
    assert(fp16::select("scalar") == fp16::Kernel::Scalar);
    bool rejected = false;
    try { (void)fp16::select("not-a-kernel"); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
    for (uint32_t cols : {1U, 7U, 8U, 15U, 16U, 37U, 512U, 4096U}) {
        constexpr size_t count = 19;
        std::vector<uint16_t> values(count * (cols + 1));
        std::vector<const uint16_t*> rows;
        std::vector<float> query(cols), output(count), single(count);
        for (uint32_t col = 0; col < cols; ++col) query[col] = float(int(col % 23) - 11) / 7;
        for (size_t row = 0; row < count; ++row) {
            auto* data = values.data() + row * (cols + 1) + 1;
            rows.push_back(data); // Deliberately not vector-aligned.
            for (uint32_t col = 0; col < cols; ++col) {
                const uint32_t hash = uint32_t(row * 997 + col * 251 + 17);
                data[col] = uint16_t((hash % 0x7c00) | ((hash & 1) << 15));
            }
        }
        for (auto kernel : {fp16::Kernel::Scalar, fp16::Kernel::Avx2, fp16::Kernel::Avx512}) {
            if (!fp16::supported(kernel)) continue;
            fp16::batch(kernel, rows.data(), count, query.data(), cols, single.data(), 1);
            for (size_t width : {1U, 4U, 8U}) {
                fp16::batch(kernel, rows.data(), count, query.data(), cols, output.data(), width);
                for (size_t row = 0; row < count; ++row) {
                    double reference = 0;
                    for (uint32_t col = 0; col < cols; ++col) {
                        const double delta = double(query[col]) - fp16::decode(rows[row][col]);
                        reference += delta * delta;
                    }
                    assert(std::isfinite(output[row]));
                    assert(std::abs(output[row] - reference) <= 2e-5 * std::max(1.0, reference));
                    assert(output[row] == single[row]); // Batch width must not change reduction order.
                }
            }
            std::vector<uint16_t> special(cols, 0);
            const uint16_t* p = special.data();
            for (uint16_t value : {uint16_t(0x0001), uint16_t(0x8000), uint16_t(0x7c00), uint16_t(0x7e00)}) {
                special[0] = value;
                float result;
                fp16::batch(kernel, &p, 1, query.data(), cols, &result, 1);
                if (value == 0x7c00) assert(std::isinf(result));
                else if (value == 0x7e00) assert(std::isnan(result));
                else assert(std::isfinite(result));
            }
        }
    }
}
