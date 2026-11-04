#include "predicates/gpu_predicate_index.cuh"
#include "search/common/posting_shortlist_heap.cuh"
#include "route_predicate_fixture.h"
#include <cuda_runtime.h>
#include <cfloat>
#include <cstdio>
#include <cstdlib>

namespace gp = go_fann::gpu_predicate;
namespace heap = go_fann::posting_shortlist_heap;

#define CUDA(call) do { const auto e = (call); if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e)); std::exit(1); } } while (0)

// Use the production exact predicate and posting heap, with supplied distances
// so loss caused by shortlist truncation is independent of quantization noise.
__device__ unsigned scan(gp::View view, uint32_t route, const uint32_t* samples,
                        int begin, int end, bool filter_first, uint32_t* ids,
                        float* distances)
{
    unsigned scores = 0;
    for (int i = 0; i < 2; ++i) { ids[i] = UINT32_MAX; distances[i] = FLT_MAX; }
    for (int i = begin; i < end; ++i) {
        const uint32_t sample = samples[i]; // Never evaluate posting-local i.
        if (sample >= view.rows) continue;
        if (filter_first && gp::enabled(view, route) && !gp::matches(view, route, sample)) continue;
        ++scores;
        heap::insert(float(sample), sample, distances, ids, 2);
    }
    heap::sort(distances, ids, 2);
    return scores;
}

struct ScanWork { unsigned scores = 0; unsigned predicates = 0; };

__device__ ScanWork scan_order(gp::View view, uint32_t route,
                              const uint32_t* samples, const float* scores,
                              int begin, int end, bool score_first,
                              uint32_t* ids, float* distances)
{
    ScanWork work;
    for (int i = 0; i < 2; ++i) { ids[i] = UINT32_MAX; distances[i] = FLT_MAX; }
    for (int i = begin; i < end; ++i) {
        const uint32_t sample = samples[i];
        if (sample >= view.rows) continue;
        if (score_first) {
            ++work.scores;
            if (!heap::can_enter(scores[i], sample, distances, ids, 2)) continue;
        }
        if (gp::enabled(view, route)) {
            ++work.predicates;
            if (!gp::matches(view, route, sample)) continue;
        }
        if (!score_first) ++work.scores;
        heap::insert(scores[i], sample, distances, ids, 2);
    }
    heap::sort(distances, ids, 2);
    return work;
}

__device__ void verify_order(gp::View view, uint32_t route,
                             const uint32_t* samples, const float* scores,
                             int begin, int end, bool expect_saved_predicates,
                             unsigned* errors)
{
    uint32_t filter_ids[2], score_ids[2]; float filter_dist[2], score_dist[2];
    const auto filter = scan_order(view, route, samples, scores, begin, end, false,
                                   filter_ids, filter_dist);
    const auto score = scan_order(view, route, samples, scores, begin, end, true,
                                  score_ids, score_dist);
    unsigned valid_domain = 0;
    for (int i = begin; i < end; ++i) valid_domain += samples[i] < view.rows;
    if (score.scores != valid_domain || score.predicates > filter.predicates ||
        filter.scores > score.scores) ++*errors;
    if (expect_saved_predicates && score.predicates >= filter.predicates) ++*errors;
    for (int i = 0; i < 2; ++i)
        if (filter_ids[i] != score_ids[i] || filter_dist[i] != score_dist[i]) ++*errors;
}

__global__ void posting_admission_fixture(unsigned* errors)
{
    if (threadIdx.x != 0) return;
    // Two invalid nearest samples beat every valid sample under unfiltered
    // top-2. Token 7 also equals a hypothetical virtual range/group ID.
    const uint32_t csr_offsets[] = {0, 1, 2, 3, 4, 5};
    const uint32_t csr_labels[] = {2, 2, 7, 9, 7};
    const float numeric[] = {15, 15, 15, 100, 20};
    const uint32_t handles[] = {7, 9, 99};
    const gp::RangeAtom ranges[] = {{0, 3, 10, 20}};
    const gp::Clause clauses[] = {
        {0, 1, 0, 1}, // token 7 AND [10,20)
        {0, 1, 0, 1}, {1, 2, 0, 0}, // (7 AND range) OR 9
        {2, 3, 0, 0}, // no matching samples
        {0, 0, 0, 0}  // genuinely unconditional; still reject invalid sample IDs
    };
    const uint32_t route_offsets[] = {0, 1, 3, 4, 4, 5, 5};
    const uint8_t enabled[] = {1, 1, 1, 1, 1, 0};
    gp::View view;
    view.rows = 5; view.csr_offsets = csr_offsets; view.csr_labels = csr_labels;
    view.route_handles = handles; view.clauses = clauses; view.ranges = ranges;
    view.route_offsets = route_offsets; view.route_enabled = enabled;
    view.base_ranges = numeric; view.range_stride = 1;
    const uint32_t samples[] = {4, 0, 3, 1, 2, 5, UINT32_MAX};
    uint32_t old_ids[2], new_ids[2]; float old_dist[2], new_dist[2];
    const unsigned old_scores = scan(view, 1, samples, 0, 7, false, old_ids, old_dist);
    const unsigned new_scores = scan(view, 1, samples, 0, 7, true, new_ids, new_dist);
    if (old_ids[0] != 0 || old_ids[1] != 1 || old_scores != 5) ++*errors;
    if (gp::matches(view, 1, old_ids[0]) || gp::matches(view, 1, old_ids[1])) ++*errors;
    if (new_ids[0] != 2 || new_ids[1] != 3 || new_scores != 2) ++*errors;
    if (new_dist[0] != 2 || new_dist[1] != 3) ++*errors; // Posting has no bias.
    scan(view, 0, samples, 0, 7, true, new_ids, new_dist);
    if (new_ids[0] != 2 || new_ids[1] != UINT32_MAX) ++*errors;
    for (uint32_t route = 2; route <= 3; ++route) {
        if (scan(view, route, samples, 0, 7, true, new_ids, new_dist) != 0 ||
            new_ids[0] != UINT32_MAX || new_ids[1] != UINT32_MAX) ++*errors;
    }
    // Disabled view is a fallback, not an empty-match predicate. It retains
    // old producer behavior; the caller must keep its original CPU checks.
    if (gp::enabled(view, 5) ||
        scan(view, 5, samples, 0, 7, true, new_ids, new_dist) != 5 ||
        new_ids[0] != 0 || new_ids[1] != 1) ++*errors;
    if (gp::matches(view, 4, 5) || gp::matches(view, 4, UINT32_MAX) ||
        !gp::matches(view, 4, 0)) ++*errors;

    // Same raw scores and original IDs: score-first must retain exactly the
    // filtered shortlist for DNF, numeric boundaries, empty/entailed/disabled
    // plans, both full and partial heaps, and per-segment scans.
    const float arbitrary_scores[] = {0.0f, -1.0f, 1.0f, 1.0f, 1.0f, -2.0f, -3.0f};
    for (uint32_t route = 0; route < 6; ++route) {
        verify_order(view, route, samples, arbitrary_scores, 0, 7, false, errors);
        verify_order(view, route, samples, arbitrary_scores, 0, 3, false, errors);
        verify_order(view, route, samples, arbitrary_scores, 3, 7, false, errors);
    }
    // Two accepted early rows establish a valid cutoff. Later rows still get
    // scored but their predicates can be skipped. A tied smaller ID replaces
    // the larger ID; tied larger IDs cannot enter. This is not raw Top-L then
    // postfilter, and it never stops scanning source rows.
    const uint32_t ordered[] = {3, 2, 4, 1, 0};
    const float tied_scores[] = {1, 1, 1, 1, 2};
    verify_order(view, 1, ordered, tied_scores, 0, 5, true, errors);
    verify_order(view, 4, ordered, tied_scores, 0, 5, true, errors);

    // Each segment must filter before its own bounded shortlist; filtering
    // only the global merged shortlist would already have lost valid IDs.
    uint32_t segment_ids[4]; float segment_dist[4];
    scan(view, 1, samples, 0, 3, true, segment_ids, segment_dist);
    scan(view, 1, samples, 3, 7, true, segment_ids + 2, segment_dist + 2);
    for (int i = 0; i < 2; ++i) { new_ids[i] = UINT32_MAX; new_dist[i] = FLT_MAX; }
    for (int i = 0; i < 4; ++i) heap::insert(segment_dist[i], segment_ids[i], new_dist, new_ids, 2);
    heap::sort(new_dist, new_ids, 2);
    if (new_ids[0] != 2 || new_ids[1] != 3) ++*errors;
    // The score-first segments must also merge to exactly the same global
    // valid shortlist, including the sample-ID tiebreak across segments.
    uint32_t expected_ids[2]; float expected_dist[2];
    scan_order(view, 1, samples, arbitrary_scores, 0, 7, false,
               expected_ids, expected_dist);
    scan_order(view, 1, samples, arbitrary_scores, 0, 3, true,
               segment_ids, segment_dist);
    scan_order(view, 1, samples, arbitrary_scores, 3, 7, true,
               segment_ids + 2, segment_dist + 2);
    for (int i = 0; i < 2; ++i) { new_ids[i] = UINT32_MAX; new_dist[i] = FLT_MAX; }
    for (int i = 0; i < 4; ++i) heap::insert(segment_dist[i], segment_ids[i], new_dist, new_ids, 2);
    heap::sort(new_dist, new_ids, 2);
    for (int i = 0; i < 2; ++i)
        if (new_ids[i] != expected_ids[i] || new_dist[i] != expected_dist[i]) ++*errors;
}

// Exercise the actual host planner -> uploaded clauses -> GPU evaluator path.
// These are rows of posting(1), not arbitrary CSR rows: sample 3 lacks anchor 1
// and must never be passed to a plan specialized for this producer.
struct AnchorFixture {
    uint32_t offsets[7]{};
    uint32_t handles[12]{};
    gp::Clause clauses[6]{};
    gp::RangeAtom ranges[6]{};
    uint8_t enabled[6]{};
    uint32_t csr_offsets[5]{0, 2, 4, 5, 7};
    uint32_t csr_labels[7]{1, 2, 1, 3, 1, 2, 3};
    float values[8]{10, 5, 20, 15, 30, 25, 10, 15};
    gp::Counters counters[6]{};
};

__global__ void posting_anchor_fixture(AnchorFixture* f, unsigned* errors)
{
    if (threadIdx.x != 0) return;
    gp::View v;
    v.rows = 4; v.csr_offsets = f->csr_offsets; v.csr_labels = f->csr_labels;
    v.route_offsets = f->offsets; v.route_handles = f->handles;
    v.clauses = f->clauses; v.ranges = f->ranges; v.route_enabled = f->enabled;
    v.base_ranges = f->values; v.range_stride = 2; v.counters = f->counters;
    const uint32_t samples[]{2, 0, 1};
    for (uint32_t query = 0; query < 3; ++query) {
        unsigned matches = 0;
        for (uint32_t sample : samples) {
            const bool old_pass = gp::matches(v, query * 2, sample);
            const bool new_pass = gp::matches(v, query * 2 + 1, sample);
            if (old_pass != new_pass) ++*errors;
            matches += new_pass;
        }
        if (matches != (query == 0 ? 3U : 1U)) ++*errors;
        if (gp::matches(v, query * 2 + 1, UINT32_MAX)) ++*errors;
    }
    if (f->counters[0].label_probes != 3 || f->counters[1].label_probes != 0 ||
        f->counters[1].entailed_fastpaths != 3 ||
        f->counters[2].label_probes != 6 || f->counters[3].label_probes != 3 ||
        f->counters[4].label_probes != 3 || f->counters[5].label_probes != 0 ||
        f->counters[4].range_compares != 3 || f->counters[5].range_compares != 3)
        ++*errors;
    v.counters = nullptr;
    const float scores[]{1, 0, 1};
    for (uint32_t query = 0; query < 3; ++query) {
        for (int score_first = 0; score_first < 2; ++score_first) {
            uint32_t before[2], after[2]; float before_dist[2], after_dist[2];
            scan_order(v, query * 2, samples, scores, 0, 3, score_first, before, before_dist);
            scan_order(v, query * 2 + 1, samples, scores, 0, 3, score_first, after, after_dist);
            for (int i = 0; i < 2; ++i)
                if (before[i] != after[i] || before_dist[i] != after_dist[i]) ++*errors;
        }
    }
}

static void check_compiled_anchor(unsigned* errors)
{
    namespace rp = go_fann::route_predicate;
    AnchorFixture* f = nullptr;
    CUDA(cudaMallocManaged(&f, sizeof(*f)));
    *f = AnchorFixture{};
    uint32_t label_end = 0, range_end = 0;
    for (uint32_t query = 0; query < 3; ++query) {
        auto r = predicate_fixture();
        r.query_is_lowfreq = {1}; r.high_query_indices.clear(); r.low_query_indices = {0};
        r.route_providers = {QueryRouteProvider::Label};
        r.route_candidate_sources = {QueryCandidateSource::LabelPosting};
        r.route_execution_modes = {QueryRouteExecution::GpuResidentScan};
        if (query == 1) {
            r.query_labels_matrix.indptr = {0, 1}; r.query_labels_matrix.indices = {2};
        }
        if (query == 2) r.query_ranges = {1, 2, {3, 0}, {10, 0}, {20, 0}};
        for (uint32_t optimized = 0; optimized < 2; ++optimized) {
            if (optimized) {
                r.lowfreq_index.group_count = 1; r.lowfreq_index.total_rows = 3;
                r.lowfreq_index.label_offset = {0}; r.lowfreq_index.label_size = {3};
                r.lowfreq_index.sample_ids = {2, 0, 1}; r.lowfreq_index.row_filter_mask = {1, 1, 1};
            }
            const auto plan = rp::compile(r, rp::Producer::LabelPosting);
            const auto& route = plan.routes[0];
            predicate_expect(route.available && route.clauses.size() == 1,
                             "posting GPU fixture requires one available clause");
            predicate_expect(route.entailed == (optimized != 0 && query == 0),
                             "posting planner did not produce expected entailment");
            const uint32_t slot = query * 2 + optimized;
            const auto& c = route.clauses[0];
            auto& dst = f->clauses[slot];
            dst.label_begin = label_end; dst.range_begin = range_end;
            for (uint32_t label : c.labels) f->handles[label_end++] = label;
            for (const auto& a : c.ranges)
                f->ranges[range_end++] = {a.dimension, a.mode, a.lower, a.upper};
            dst.label_end = label_end; dst.range_end = range_end;
            f->offsets[slot + 1] = slot + 1; f->enabled[slot] = 1;
        }
    }
    posting_anchor_fixture<<<1, 32>>>(f, errors);
    CUDA(cudaGetLastError()); CUDA(cudaDeviceSynchronize());
    std::printf("Posting anchor CSR probes (3 candidates): single %llu->%llu, AND %llu->%llu, range %llu->%llu\n",
        static_cast<unsigned long long>(f->counters[0].label_probes),
        static_cast<unsigned long long>(f->counters[1].label_probes),
        static_cast<unsigned long long>(f->counters[2].label_probes),
        static_cast<unsigned long long>(f->counters[3].label_probes),
        static_cast<unsigned long long>(f->counters[4].label_probes),
        static_cast<unsigned long long>(f->counters[5].label_probes));
    CUDA(cudaFree(f));
}

int main()
{
    unsigned* errors = nullptr;
    CUDA(cudaMallocManaged(&errors, sizeof(*errors)));
    *errors = 0;
    posting_admission_fixture<<<1, 32>>>(errors);
    CUDA(cudaGetLastError());
    CUDA(cudaDeviceSynchronize());
    check_compiled_anchor(errors);
    const unsigned count = *errors;
    CUDA(cudaFree(errors));
    if (count) {
        std::fprintf(stderr, "posting admission fixture failed: %u errors\n", count);
        return 1;
    }
    std::puts("Posting predicate CUDA recovery, score-order parity/work counts, DNF, segments and fallback fixture passed");
}
