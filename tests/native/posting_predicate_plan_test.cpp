#include "route_predicate_fixture.h"
#include "planning/route_predicate_plan.h"
#include <limits>

namespace rp = go_fann::route_predicate;

static GraphRuntime posting_fixture()
{
    auto r = predicate_fixture();
    r.query_is_lowfreq = {1};
    r.high_query_indices.clear();
    r.low_query_indices = {0};
    r.route_providers = {QueryRouteProvider::Label};
    r.route_candidate_sources = {QueryCandidateSource::LabelPosting};
    r.route_execution_modes = {QueryRouteExecution::GpuResidentScan};
    r.original_query_labels_matrix.rows = 1;
    r.original_query_labels_matrix.cols = 4;
    r.original_query_labels_matrix.indptr = {0, 1};
    r.original_query_labels_matrix.indices = {1};
    r.lowfreq_index.group_count = 1;
    r.lowfreq_index.total_rows = 3;
    r.lowfreq_index.label_offset = {0};
    r.lowfreq_index.label_size = {3};
    r.lowfreq_index.sample_ids = {0, 1, 2}; // Exactly the samples carrying token 1.
    r.lowfreq_index.row_filter_mask = {1, 1, 1};
    return r;
}

template<class F> static void rejects(F fn, const char* message)
{
    bool rejected = false;
    try { fn(); } catch (const std::runtime_error&) { rejected = true; }
    predicate_expect(rejected, message);
}

static rp::Plan range_plan(GraphRuntime& r)
{
    return rp::compile(r, rp::Producer::RangePosting, nullptr, {0}, {0});
}

// Oracle reads the original clause table, not the compiled residual plan.
// Numeric comparisons intentionally do not reuse the GPU range helper.
static bool original_owned_dnf(const GraphRuntime& r, uint32_t sample)
{
    for (uint32_t id : r.route_clause_ids) {
        if (!sample_matches_clause_labels(r, id, sample)) continue;
        bool pass = true;
        if (r.original_query_clauses.uses_original_range[id]) {
            for (uint32_t dim = 0; dim < r.base_range_count; ++dim) {
                const auto mode = r.original_query_ranges.modes[dim];
                const float value = r.base_range_values[sample * r.base_range_count + dim];
                const float lower = r.original_query_ranges.lower_bounds[dim];
                const float upper = r.original_query_ranges.upper_bounds[dim];
                if ((mode == 1 && !(value > lower)) ||
                    (mode == 2 && !(value < upper)) ||
                    (mode == 3 && !(value >= lower && value < upper))) pass = false;
            }
        }
        if (pass) return true;
    }
    return false;
}

static void compare_original_dnf()
{
    auto r = posting_fixture();
    r.base_vectors.rows = r.data_labels.rows = 16;
    r.data_labels.indptr = {0};
    r.data_labels.indices.clear();
    r.base_range_values.clear();
    const float values[] = {9, 10, 19, 20, std::numeric_limits<float>::quiet_NaN()};
    for (uint32_t sample = 0; sample < 16; ++sample) {
        for (uint32_t label = 0; label < 4; ++label)
            if (sample & (1U << label)) r.data_labels.indices.push_back(label);
        r.data_labels.indptr.push_back(r.data_labels.indices.size());
        r.base_range_values.push_back(values[sample % 5]);
        r.base_range_values.push_back(values[(sample + 2) % 5]);
    }
    r.lowfreq_index.sample_ids.clear();
    for (uint32_t sample = 0; sample < 16; ++sample)
        if (sample & (1U << 1)) r.lowfreq_index.sample_ids.push_back(sample);
    r.lowfreq_index.total_rows = r.lowfreq_index.sample_ids.size();
    r.lowfreq_index.label_size = {r.lowfreq_index.total_rows};
    r.lowfreq_index.row_filter_mask.assign(r.lowfreq_index.total_rows, 1);
    predicate_clauses(r, {{1, 2}, {3}, {}}, {1, 0, 1});
    for (uint8_t mode = 0; mode <= 3; ++mode) {
        r.original_query_ranges = {1, 2, {mode, 3}, {10, 9}, {20, 21}};
        for (uint32_t owned = 1; owned < 8; ++owned) {
            r.route_clause_ids.clear();
            for (uint32_t id = 0; id < 3; ++id)
                if (owned & (1U << id)) r.route_clause_ids.push_back(id);
            r.route_clause_indptr = {0, r.route_clause_ids.size()};
            r.route_providers[0] = QueryRouteProvider::Label;
            r.route_candidate_sources[0] = QueryCandidateSource::LabelPosting;
            const auto label_plan = rp::compile(r, rp::Producer::LabelPosting);
            r.route_providers[0] = QueryRouteProvider::RangePosting;
            r.route_candidate_sources[0] = QueryCandidateSource::RangePosting;
            const auto numeric_plan = range_plan(r);
            for (uint32_t sample = 0; sample < 16; ++sample) {
                const bool expected = original_owned_dnf(r, sample);
                // A residual is valid on its producer's candidates only.
                if (std::find(r.lowfreq_index.sample_ids.begin(), r.lowfreq_index.sample_ids.end(),
                              sample) != r.lowfreq_index.sample_ids.end())
                    predicate_expect(rp::evaluate(label_plan, r, 0, sample) == expected,
                                     "label posting differs from original owned DNF oracle");
                predicate_expect(rp::evaluate(numeric_plan, r, 0, sample) == expected,
                                 "range posting differs from original owned DNF oracle");
            }
        }
    }
}

int main()
{
    compare_original_dnf();
    {
        auto r = posting_fixture();
        const auto p = rp::compile(r, rp::Producer::LabelPosting);
        predicate_expect(p.routes[0].available && p.routes[0].entailed &&
                         p.routes[0].clauses[0].labels.empty(),
                         "single-label posting retained a guaranteed anchor");
        for (const auto sample : r.lowfreq_index.sample_ids)
            predicate_expect(rp::evaluate(p, r, 0, sample), "entailed posting rejected its own row");
        r.group_layout.groups[0] = {1, 2, 3};
        r.group_layout.label_to_group = {UINT32_MAX, 0, 0, 0};
        r.lowfreq_index.total_rows = 4;
        r.lowfreq_index.label_size = {4};
        r.lowfreq_index.sample_ids = {0, 1, 2, 3};
        r.lowfreq_index.row_filter_mask = {3, 5, 1, 6};
        const auto group = rp::compile(r, rp::Producer::LabelPosting);
        predicate_expect(!rp::evaluate(group, r, 0, 3), "group union became singleton proof");
        predicate_expect(!rp::evaluate(group, r, 0, 4), "sample outside base domain accepted");
        auto different = posting_fixture();
        predicate_expect(!rp::evaluate(group, different, 0, 0), "wrong producer identity accepted");
    }
    {
        // The selected label need not equal its storage group or row offset.
        auto r = posting_fixture();
        r.group_layout.groups = {{2}, {1}};
        r.group_layout.label_to_group = {UINT32_MAX, 1, 0, UINT32_MAX};
        r.query_group_ids = {1};
        r.lowfreq_index.group_count = 2;
        r.lowfreq_index.total_rows = 5;
        r.lowfreq_index.label_offset = {0, 2};
        r.lowfreq_index.label_size = {2, 3};
        r.lowfreq_index.sample_ids = {0, 3, 0, 1, 2};
        r.lowfreq_index.row_filter_mask.assign(5, 1);
        const auto p = rp::compile(r, rp::Producer::LabelPosting);
        predicate_expect(p.routes[0].entailed, "nonzero posting offset lost singleton proof");
        for (uint32_t row = 2; row < 5; ++row)
            predicate_expect(rp::evaluate(p, r, 0, r.lowfreq_index.sample_ids[row]),
                             "nonidentity posting mapping rejected an eligible candidate");
    }
    {
        auto r = posting_fixture();
        r.query_labels_matrix.indptr = {0, 1};
        r.query_labels_matrix.indices = {2};
        const auto conjunction = rp::compile(r, rp::Producer::LabelPosting);
        predicate_expect(!conjunction.routes[0].entailed &&
                         conjunction.routes[0].clauses[0].labels == std::vector<uint32_t>{2},
                         "ordinary conjunction retained guaranteed anchor or lost residual");
        for (const auto sample : r.lowfreq_index.sample_ids) {
            const auto labels = sparse_label_row_span(r.data_labels, sample);
            const bool expected = std::find(labels.begin, labels.end, 1) != labels.end &&
                                  std::find(labels.begin, labels.end, 2) != labels.end;
            predicate_expect(rp::evaluate(conjunction, r, 0, sample) == expected,
                             "ordinary posting residual differs from full conjunction");
        }
        r.query_labels_matrix.indptr = {0, 0};
        r.query_labels_matrix.indices.clear();
        r.query_ranges = {1, 2, {3, 0}, {10, 0}, {20, 0}};
        const auto numeric = rp::compile(r, rp::Producer::LabelPosting);
        predicate_expect(!numeric.routes[0].entailed &&
                         numeric.routes[0].clauses[0].labels.empty() &&
                         numeric.routes[0].clauses[0].ranges.size() == 1,
                         "anchor elimination removed numeric residual");
        for (const auto sample : r.lowfreq_index.sample_ids)
            predicate_expect(rp::evaluate(numeric, r, 0, sample) == (sample == 0),
                             "anchor elimination changed numeric residual");
    }
    {
        // Graph metadata alone, malformed intervals/mappings and absent lists
        // must never justify removing an original token check.
        const auto conservative = [](const GraphRuntime& r) {
            const auto p = rp::compile(r, rp::Producer::LabelPosting);
            predicate_expect(p.routes[0].available && !p.routes[0].entailed &&
                             p.routes[0].clauses[0].labels == std::vector<uint32_t>{1} &&
                             !rp::evaluate(p, r, 0, 3), "invalid posting proof removed anchor");
        };
        auto r = posting_fixture(); r.lowfreq_index = {}; conservative(r);
        r = posting_fixture(); r.query_group_ids.clear(); conservative(r);
        r = posting_fixture(); r.query_group_ids[0] = 1; conservative(r);
        r = posting_fixture(); r.group_layout.label_to_group[1] = 1; conservative(r);
        r = posting_fixture(); r.group_layout.groups[0] = {2}; conservative(r);
        r = posting_fixture(); r.lowfreq_index.group_count = 2; conservative(r);
        r = posting_fixture(); r.lowfreq_index.label_offset.clear(); conservative(r);
        r = posting_fixture(); r.lowfreq_index.label_size.clear(); conservative(r);
        r = posting_fixture(); r.lowfreq_index.label_offset[0] = UINT32_MAX; conservative(r);
        r = posting_fixture(); r.lowfreq_index.label_size[0] = UINT32_MAX; conservative(r);
        r = posting_fixture(); r.lowfreq_index.label_size[0] = 0; conservative(r);
        r = posting_fixture(); r.lowfreq_index.sample_ids.pop_back(); conservative(r);
        r = posting_fixture(); r.lowfreq_index.row_filter_mask.pop_back(); conservative(r);
        r = posting_fixture(); r.data_labels.cols = 1;
        rejects([&] { rp::compile(r, rp::Producer::LabelPosting); },
                "out-of-domain anchor removed before original-token validation");
    }
    {
        auto r = posting_fixture();
        predicate_clauses(r, {{1}}, {0});
        const auto entailed = rp::compile(r, rp::Producer::LabelPosting);
        predicate_expect(entailed.routes[0].entailed, "single-label DNF posting not entailed");
        r.label_intervals.has_interval = {0, 1};
        const auto virtual_plan = rp::compile(r, rp::Producer::LabelPosting);
        predicate_expect(!virtual_plan.routes[0].entailed &&
                         virtual_plan.routes[0].clauses[0].labels == std::vector<uint32_t>{1} &&
                         !rp::evaluate(virtual_plan, r, 0, 3), "virtual ID collision proved original token");
        r.label_intervals.has_interval.clear();
        auto primary = posting_fixture();
        predicate_clauses(primary, {{1}}, {0});
        const auto secondary = rp::compile(r, rp::Producer::LabelPosting, &primary);
        predicate_expect(!secondary.routes[0].entailed &&
                         secondary.routes[0].clauses[0].labels == std::vector<uint32_t>{1},
                         "secondary posting ID collision proved primary token");
    }
    {
        auto r = posting_fixture();
        predicate_clauses(r, {{1, 2}, {3}}, {1, 0});
        r.original_query_ranges = {1, 2, {2, 0}, {0, 0}, {15, 0}};
        const auto p = rp::compile(r, rp::Producer::LabelPosting);
        predicate_expect(p.routes[0].clauses.size() == 2, "DNF alternatives flattened");
        predicate_expect(p.routes[0].clauses[0].labels == std::vector<uint32_t>{2} &&
                         p.routes[0].clauses[1].labels == std::vector<uint32_t>{3},
                         "posting did not remove only its guaranteed DNF anchor");
        for (const auto sample : r.lowfreq_index.sample_ids)
            predicate_expect(rp::evaluate(p, r, 0, sample) == (sample != 2),
                             "DNF range scope or anchor membership incorrect");
        // The route owns only the first original clause, not every OR branch.
        r.route_clause_indptr = {0, 1};
        r.route_clause_ids = {0};
        const auto subset = rp::compile(r, rp::Producer::LabelPosting);
        predicate_expect(rp::evaluate(subset, r, 0, 0) &&
                         !rp::evaluate(subset, r, 0, 1), "unowned clause admitted");
        r.route_clause_ids = {42};
        rejects([&] { rp::compile(r, rp::Producer::LabelPosting); }, "invalid owned clause accepted");
        r.route_clause_ids.clear(); r.route_clause_indptr = {0, 0};
        predicate_expect(!rp::compile(r, rp::Producer::LabelPosting).routes[0].available, "missing DNF ownership inferred");
        r.route_clause_indptr.clear();
        predicate_expect(!rp::compile(r, rp::Producer::LabelPosting).routes[0].available,
                         "absent label route clause table inferred DNF ownership");
        r.route_providers[0] = QueryRouteProvider::RangePosting;
        r.route_candidate_sources[0] = QueryCandidateSource::RangePosting;
        predicate_expect(!range_plan(r).routes[0].available,
                         "absent range route clause table inferred DNF ownership");
    }
    {
        auto r = posting_fixture();
        r.route_providers[0] = QueryRouteProvider::RangePosting;
        r.route_candidate_sources[0] = QueryCandidateSource::RangePosting;
        r.route_execution_modes[0] = QueryRouteExecution::GpuResidentScanPostFilter;
        r.label_intervals.has_interval = {0, 1}; // Virtual bin equals token 1.
        predicate_clauses(r, {{1, 2}, {3}}, {1, 0});
        r.original_query_ranges = {1, 2, {3, 0}, {10, 0}, {20, 0}};
        const auto p = range_plan(r);
        predicate_expect(p.routes[0].available &&
                         p.routes[0].clauses[0].labels == std::vector<uint32_t>({1, 2}),
                         "virtual range bin discharged a real token");
        predicate_expect(rp::evaluate(p, r, 0, 0) && rp::evaluate(p, r, 0, 1) &&
                         !rp::evaluate(p, r, 0, 2), "range DNF differs from original predicate");
        r.route_clause_indptr = {0, 1}; r.route_clause_ids = {0};
        const auto scoped = range_plan(r);
        predicate_expect(!rp::evaluate(scoped, r, 0, 1) &&
                         !rp::evaluate(scoped, r, 0, 3), "range route widened clause ownership");
        r.route_execution_modes[0] = QueryRouteExecution::HostCandidateStream;
        predicate_expect(!range_plan(r).routes[0].available,
                         "host-prefiltered candidate stream changed policy");
        r.route_execution_modes[0] = QueryRouteExecution::GpuResidentScan;
        r.route_enabled[0] = 0;
        predicate_expect(!range_plan(r).routes[0].available, "disabled range route compiled");
    }
    {
        auto r = posting_fixture();
        r.original_query_labels_matrix.indptr = {0, 0};
        r.original_query_labels_matrix.indices.clear();
        r.query_filter_labels[0] = 2; // A representative's virtual/graph ID is irrelevant.
        r.original_query_ranges = {1, 2, {3, 0}, {10, 0}, {20, 0}};
        const auto p = range_plan(r);
        predicate_expect(p.routes[0].available && p.routes[0].clauses[0].labels.empty() &&
                         rp::evaluate(p, r, 0, 0) && !rp::evaluate(p, r, 0, 1),
                         "pure range representative inherited label predicate");
        rejects([&] { rp::compile(r, rp::Producer::RangePosting, nullptr, {0, 0}, {0, 1}); },
                "conflicting representative identity accepted");
        rejects([&] { rp::compile(r, rp::Producer::RangePosting, nullptr, {0}, {}); }, "misaligned range mapping accepted");
    }
    {
        auto r = posting_fixture();
        for (const auto mode : {QueryRouteExecution::HostCandidateStream,
                               QueryRouteExecution::GraphTraversal, QueryRouteExecution::None}) {
            r.route_execution_modes[0] = mode;
            predicate_expect(!rp::compile(r, rp::Producer::LabelPosting).routes[0].available, "unsupported label producer compiled");
        }
        r.route_execution_modes[0] = QueryRouteExecution::GpuResidentScan;
        r.route_candidate_sources[0] = QueryCandidateSource::RangePosting;
        predicate_expect(!rp::compile(r, rp::Producer::LabelPosting).routes[0].available, "mismatched label source compiled");
        r.route_candidate_sources[0] = QueryCandidateSource::LabelPosting;
        r.route_enabled[0] = 0;
        predicate_expect(!rp::compile(r, rp::Producer::LabelPosting).routes[0].available, "disabled label route compiled");
        r.route_enabled[0] = 1;
        r.query_filter_labels[0] = 99;
        rejects([&] { rp::compile(r, rp::Producer::LabelPosting); }, "token outside original CSR domain accepted");
    }
    {
        auto r = posting_fixture();
        predicate_clauses(r, {{2}}, {0});
        auto original = posting_fixture();
        predicate_clauses(original, {{2}}, {0});
        r.data_labels.indices.assign(7, 0); // Virtual data cannot supply token membership.
        const auto p = rp::compile(r, rp::Producer::LabelPosting, &original);
        predicate_expect(rp::evaluate(p, r, 0, 0) && !rp::evaluate(p, r, 0, 1),
                         "original-label source not respected");
        original.base_vectors.rows = 5;
        rejects([&] { rp::compile(r, rp::Producer::LabelPosting, &original); }, "mismatched original source rows accepted");
    }
    {
        auto r = posting_fixture();
        predicate_clauses(r, {{0}}, {0}); // In-domain absent token: no matches.
        const auto p = rp::compile(r, rp::Producer::LabelPosting);
        for (uint32_t sample = 0; sample < r.base_vectors.rows; ++sample)
            predicate_expect(!rp::evaluate(p, r, 0, sample), "empty result predicate admitted candidate");
        // Labels and ranges are scoped to query 0, never query 1's clause.
        r.original_query_count = 2;
        r.original_query_clauses.rows = 2;
        r.original_query_clauses.query_indptr = {0, 1, 2};
        r.original_query_clauses.label_indptr = {0, 1, 2};
        r.original_query_clauses.labels = {0, 1};
        r.original_query_clauses.uses_original_range = {0, 0};
        r.route_clause_ids = {1};
        rejects([&] { rp::compile(r, rp::Producer::LabelPosting); }, "cross-query clause ownership accepted");
    }
}
