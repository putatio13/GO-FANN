#pragma once
#include <cstdint>
#include <cfloat>
#include <cmath>

#ifndef GO_FANN_GPU_PREDICATE_CSR_THRESHOLD
#define GO_FANN_GPU_PREDICATE_CSR_THRESHOLD 0
#endif
static_assert(GO_FANN_GPU_PREDICATE_CSR_THRESHOLD >= 0 &&
              GO_FANN_GPU_PREDICATE_CSR_THRESHOLD <= UINT32_MAX,
              "CSR lookup threshold must fit uint32");

namespace go_fann::gpu_predicate {
struct Descriptor { uint64_t offset; uint32_t count; uint32_t reserved; };
static_assert(sizeof(Descriptor) == 16);
struct Trace { uint32_t sample; uint32_t pass; };
constexpr uint32_t trace_capacity = 128;
struct Counters {
    uint64_t expanded, proposed, passed, lookup_ns, saturated, scores;
    uint64_t label_probes, range_compares, entailed_fastpaths;
    uint32_t trace_count;
    Trace trace[trace_capacity];
};
struct RangeAtom { uint32_t dimension, mode; float lower, upper; };
struct Clause { uint32_t label_begin, label_end, range_begin, range_end; };
struct View {
    const uint32_t* directory = nullptr;
    const Descriptor* descriptors = nullptr;
    const uint16_t* payload = nullptr;
    const uint32_t* route_offsets = nullptr;
    const uint32_t* route_handles = nullptr;
    Counters* counters = nullptr;
    uint32_t rows = 0, chunks = 0;
    float alpha = 0;
    // When clauses is present, route_offsets indexes DNF clauses rather than
    // the legacy conjunction's handles. A zero enabled byte retains CPU checks.
    const Clause* clauses = nullptr;
    const RangeAtom* ranges = nullptr;
    const uint8_t* route_enabled = nullptr;
    const float* base_ranges = nullptr;
    uint32_t range_stride = 0;
    bool pending_base_stop = false;
    const uint8_t* route_prefetch_disabled = nullptr;
    const uint8_t* route_compiled = nullptr;
    // Full sample-major CSR. Handles are original token IDs in this storage.
    const uint32_t* csr_offsets = nullptr;
    const uint32_t* csr_labels = nullptr;
};
#ifdef __CUDACC__
#define PRED_HD __host__ __device__
#else
#define PRED_HD
#endif
// The uploaded CSR guarantees begin <= end and strictly sorted original IDs.
// Threshold 0 preserves the existing linear scan. Positive thresholds limit
// serial work on unusually long rows without adding dependent binary loads to
// short rows. This helper has no warp collectives: lanes may take either path.
template<uint32_t Threshold>
PRED_HD inline bool csr_contains_sorted(const uint32_t* labels, uint32_t begin,
                                       uint32_t end, uint32_t handle) {
    if constexpr (Threshold > 0) {
        if (end - begin > Threshold) {
            while (begin < end) {
                const uint32_t middle = begin + (end - begin) / 2;
                const uint32_t label = labels[middle];
                if (label == handle) return true;
                if (label < handle) begin = middle + 1;
                else end = middle;
            }
            return false;
        }
    }
    for (uint32_t i = begin; i < end; ++i) {
        const uint32_t label = labels[i];
        if (label >= handle) return label == handle;
    }
    return false;
}
PRED_HD inline bool contains(const View& v, uint32_t handle, uint32_t sample) {
    if (sample >= v.rows) return false;
    if (v.csr_offsets) {
        return csr_contains_sorted<GO_FANN_GPU_PREDICATE_CSR_THRESHOLD>(
            v.csr_labels, v.csr_offsets[sample], v.csr_offsets[sample + 1], handle);
    }
    if (v.directory == nullptr) return false;
    const uint32_t id = v.directory[static_cast<uint64_t>(handle) * v.chunks + (sample >> 16)];
    if (id == UINT32_MAX) return false;
    const Descriptor d = v.descriptors[id];
    const uint16_t key = static_cast<uint16_t>(sample);
    uint32_t lo = 0, hi = d.count;
    while (lo < hi) {
        const uint32_t mid = lo + (hi - lo) / 2;
        if (v.payload[d.offset + mid] < key) lo = mid + 1; else hi = mid;
    }
    return lo < d.count && v.payload[d.offset + lo] == key;
}
PRED_HD inline bool enabled(const View& v, uint32_t route) {
    return v.route_enabled ? v.route_enabled[route] != 0 : (v.directory != nullptr || v.csr_offsets != nullptr);
}
PRED_HD inline bool range_matches(const RangeAtom& a, float value) {
    switch (a.mode) {
        case 0: return true;
        case 1: return value > a.lower;
        case 2: return value < a.upper;
        case 3: return value >= a.lower && value < a.upper;
        default: return false;
    }
}
PRED_HD inline bool matches(const View& v, uint32_t route, uint32_t sample) {
    if (sample >= v.rows) return false;
    if (v.clauses) {
        for (uint32_t i = v.route_offsets[route]; i < v.route_offsets[route + 1]; ++i) {
            const Clause c = v.clauses[i];
            bool pass = true;
            if (v.counters && c.label_begin == c.label_end && c.range_begin == c.range_end)
                ++v.counters[route].entailed_fastpaths;
            for (uint32_t j = c.label_begin; pass && j < c.label_end; ++j) {
                if (v.counters) ++v.counters[route].label_probes;
                pass = contains(v, v.route_handles[j], sample);
            }
            for (uint32_t j = c.range_begin; pass && j < c.range_end; ++j) {
                if (v.counters) ++v.counters[route].range_compares;
                const auto a = v.ranges[j];
                pass = v.base_ranges && a.dimension < v.range_stride &&
                    range_matches(a, v.base_ranges[uint64_t(sample) * v.range_stride + a.dimension]);
            }
            if (pass) return true;
        }
        return false;
    }
    for (uint32_t i = v.route_offsets[route]; i < v.route_offsets[route + 1]; ++i) {
        if (v.counters) ++v.counters[route].label_probes;
        if (!contains(v, v.route_handles[i], sample)) return false;
    }
    return true;
}
struct BiasedPair {
    float first;
    uint32_t second;
    float base;
    PRED_HD bool operator<(const BiasedPair& o) const { return first < o.first; }
    PRED_HD bool operator>(const BiasedPair& o) const { return first > o.first; }
};
static_assert(sizeof(BiasedPair) == 12);
template<class Pair> PRED_HD inline float base_score(const Pair& p) { return p.first; }
PRED_HD inline float base_score(const BiasedPair& p) { return p.base; }
// Caller supplies validity/used semantics; reserved entries must be included
// even though cooperative pickup already marked their pool copies as used.
template<class Pair, class Pending>
PRED_HD inline float minimum_pending_base(const Pair* pool, int size,
                                         const Pair& current, const Pair& next,
                                         bool current_valid, bool next_valid,
                                         Pending pending) {
    float minimum = FLT_MAX;
    if (current_valid) minimum = fminf(minimum, base_score(current));
    if (next_valid) minimum = fminf(minimum, base_score(next));
    for (int i = 1; i < size; ++i)
        if (pending(pool[i].second)) minimum = fminf(minimum, base_score(pool[i]));
    return minimum;
}
PRED_HD inline float priority(float base, float penalty, bool pass, bool& saturated) {
    // Preserve sentinels and nonfinite legacy values; never promote them.
    if (pass || penalty == 0 || !std::isfinite(base) || base == FLT_MAX) return base;
    if (base > FLT_MAX - penalty) { saturated = true; return FLT_MAX; }
    return base + penalty;
}
#undef PRED_HD
}
