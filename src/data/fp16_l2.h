#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define GO_FANN_FP16_X86 1
#else
#define GO_FANN_FP16_X86 0
#endif

namespace go_fann::fp16_l2 {

enum class Kernel { Scalar, Avx2, Avx512 };

inline float decode(uint16_t value) noexcept
{
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
    uint32_t exponent = (value >> 10) & 0x1fu;
    uint32_t mantissa = value & 0x03ffu;
    uint32_t bits;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            exponent = 1;
            while ((mantissa & 0x0400u) == 0) { mantissa <<= 1; --exponent; }
            bits = sign | ((exponent + 112u) << 23) | ((mantissa & 0x03ffu) << 13);
        }
    } else if (exponent == 0x1fu) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    }
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

inline bool supported(Kernel kernel)
{
    if (kernel == Kernel::Scalar) return true;
#if GO_FANN_FP16_X86
    __builtin_cpu_init();
    return __builtin_cpu_supports("f16c") && __builtin_cpu_supports("avx2") &&
        (kernel != Kernel::Avx512 || __builtin_cpu_supports("avx512f"));
#else
    return false;
#endif
}

inline Kernel select(const std::string& requested)
{
    if (requested.empty() || requested == "auto") {
        if (supported(Kernel::Avx512)) return Kernel::Avx512;
        if (supported(Kernel::Avx2)) return Kernel::Avx2;
        return Kernel::Scalar;
    }
    Kernel kernel;
    if (requested == "scalar") kernel = Kernel::Scalar;
    else if (requested == "avx2") kernel = Kernel::Avx2;
    else if (requested == "avx512") kernel = Kernel::Avx512;
    else throw std::invalid_argument("FP16 rerank kernel must be auto, scalar, avx2 or avx512");
    if (!supported(kernel)) throw std::invalid_argument("requested FP16 rerank ISA is unavailable");
    return kernel;
}

inline const char* name(Kernel kernel)
{
    switch (kernel) {
        case Kernel::Scalar: return "fp16_scalar_l2";
        case Kernel::Avx2: return "fp16_avx2_l2";
        case Kernel::Avx512: return "fp16_avx512_l2";
    }
    return "unknown";
}

inline float scalar(const uint16_t* base, const float* query, uint32_t cols)
{
    float sum = 0.0f;
    for (uint32_t col = 0; col < cols; ++col) {
        const float diff = query[col] - decode(base[col]);
        sum += diff * diff;
    }
    return sum;
}

#if GO_FANN_FP16_X86
template <size_t Width>
__attribute__((target("avx2,f16c")))
inline void avx2(const uint16_t* const* rows, const float* query, uint32_t cols, float* out)
{
    __m256 sums[Width];
    for (size_t row = 0; row < Width; ++row) sums[row] = _mm256_setzero_ps();
    uint32_t col = 0;
    for (; col + 8 <= cols; col += 8) {
        const __m256 q = _mm256_loadu_ps(query + col);
        for (size_t row = 0; row < Width; ++row) {
            const __m256 b = _mm256_cvtph_ps(_mm_loadu_si128(
                reinterpret_cast<const __m128i*>(rows[row] + col)));
            const __m256 diff = _mm256_sub_ps(q, b);
            sums[row] = _mm256_add_ps(sums[row], _mm256_mul_ps(diff, diff));
        }
    }
    for (size_t row = 0; row < Width; ++row) {
        alignas(32) float lanes[8];
        _mm256_store_ps(lanes, sums[row]);
        float sum = 0.0f;
        for (float lane : lanes) sum += lane;
        for (uint32_t tail = col; tail < cols; ++tail) {
            const float diff = query[tail] - decode(rows[row][tail]);
            sum += diff * diff;
        }
        out[row] = sum;
    }
}

template <size_t Width>
__attribute__((target("avx512f,avx2,f16c")))
inline void avx512(const uint16_t* const* rows, const float* query, uint32_t cols, float* out)
{
    __m512 sums[Width];
    for (size_t row = 0; row < Width; ++row) sums[row] = _mm512_setzero_ps();
    uint32_t col = 0;
    for (; col + 16 <= cols; col += 16) {
        const __m512 q = _mm512_loadu_ps(query + col);
        for (size_t row = 0; row < Width; ++row) {
            const __m512 b = _mm512_cvtph_ps(_mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(rows[row] + col)));
            const __m512 diff = _mm512_sub_ps(q, b);
            sums[row] = _mm512_add_ps(sums[row], _mm512_mul_ps(diff, diff));
        }
    }
    for (size_t row = 0; row < Width; ++row) {
        alignas(64) float lanes[16];
        _mm512_store_ps(lanes, sums[row]);
        float sum = 0.0f;
        for (float lane : lanes) sum += lane;
        for (uint32_t tail = col; tail < cols; ++tail) {
            const float diff = query[tail] - decode(rows[row][tail]);
            sum += diff * diff;
        }
        out[row] = sum;
    }
}
#endif

template <size_t Width>
inline void score(Kernel kernel, const uint16_t* const* rows,
                  const float* query, uint32_t cols, float* out)
{
#if GO_FANN_FP16_X86
    if (kernel == Kernel::Avx512) { avx512<Width>(rows, query, cols, out); return; }
    if (kernel == Kernel::Avx2) { avx2<Width>(rows, query, cols, out); return; }
#else
    (void)kernel;
#endif
    for (size_t row = 0; row < Width; ++row) out[row] = scalar(rows[row], query, cols);
}

// rows must already be validated. No allocation or reordering; dimensions can
// have arbitrary tails and hvecs payload addresses need not be SIMD-aligned.
inline void batch(Kernel kernel, const uint16_t* const* rows, size_t count,
                  const float* query, uint32_t cols, float* out, size_t width)
{
    size_t i = 0;
    if (width == 8) {
        for (; count - i >= 8; i += 8) score<8>(kernel, rows + i, query, cols, out + i);
    }
    if (width >= 4) {
        for (; count - i >= 4; i += 4) score<4>(kernel, rows + i, query, cols, out + i);
    }
    for (; i < count; ++i) score<1>(kernel, rows + i, query, cols, out + i);
}

} // namespace go_fann::fp16_l2
