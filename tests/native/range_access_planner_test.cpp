#include "data/dataloader.h"
#include "index/range_posting_index.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace rp = go_fann::range_posting;

namespace {

void require(bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}

uint64_t align_up(uint64_t value)
{
    constexpr uint64_t alignment = 64;
    return ((value + alignment - 1) / alignment) * alignment;
}

void write_zeroes(std::ofstream& output, uint64_t target)
{
    const uint64_t current = static_cast<uint64_t>(output.tellp());
    require(current <= target, "range posting fixture offsets overlap");
    const std::vector<char> zeroes(static_cast<size_t>(target - current), 0);
    output.write(zeroes.data(), static_cast<std::streamsize>(zeroes.size()));
}

void write_index(const std::filesystem::path& path)
{
    const std::vector<float> values{1.0f, 2.0f, 4.0f};
    const std::vector<uint64_t> offsets{0, 2, 4, 6};
    const std::vector<uint32_t> ids{0, 1, 2, 3, 4, 5};
    rp::FileHeader header{};
    std::memcpy(header.magic, rp::kMagic, sizeof(rp::kMagic));
    header.version = rp::kVersion;
    header.header_bytes = sizeof(rp::FileHeader) + sizeof(rp::DimensionDescriptor);
    header.row_count = ids.size();
    header.range_count = 1;

    rp::DimensionDescriptor descriptor{};
    descriptor.range_index = 0;
    descriptor.unique_count = values.size();
    descriptor.row_count = ids.size();
    uint64_t cursor = align_up(header.header_bytes);
    descriptor.values_offset = cursor;
    descriptor.values_bytes = values.size() * sizeof(float);
    cursor = align_up(cursor + descriptor.values_bytes);
    descriptor.offsets_offset = cursor;
    descriptor.offsets_bytes = offsets.size() * sizeof(uint64_t);
    cursor = align_up(cursor + descriptor.offsets_bytes);
    descriptor.sample_ids_offset = cursor;
    descriptor.sample_ids_bytes = ids.size() * sizeof(uint32_t);
    header.file_bytes = align_up(cursor + descriptor.sample_ids_bytes);

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(&header), sizeof(header));
    output.write(reinterpret_cast<const char*>(&descriptor), sizeof(descriptor));
    write_zeroes(output, descriptor.values_offset);
    output.write(reinterpret_cast<const char*>(values.data()), descriptor.values_bytes);
    write_zeroes(output, descriptor.offsets_offset);
    output.write(reinterpret_cast<const char*>(offsets.data()), descriptor.offsets_bytes);
    write_zeroes(output, descriptor.sample_ids_offset);
    output.write(reinterpret_cast<const char*>(ids.data()), descriptor.sample_ids_bytes);
    write_zeroes(output, header.file_bytes);
    require(output.good(), "unable to write range posting fixture");
}

GraphRuntime make_runtime(bool lowfreq)
{
    GraphRuntime runtime;
    runtime.base_vectors.rows = 6;
    runtime.base_range_count = 1;
    runtime.original_query_count = 1;
    runtime.original_query_label_ops = {QueryLabelOp::And};
    runtime.original_query_clauses.rows = 1;
    runtime.original_query_clauses.query_indptr = {0, 1};
    runtime.original_query_clauses.label_indptr = {0, 1};
    runtime.original_query_clauses.labels = {0};
    runtime.original_query_clauses.top_branch_ids = {0};
    runtime.original_query_clauses.uses_original_range = {1};
    runtime.query_labels_matrix.rows = 2;
    runtime.query_labels_matrix.cols = 1;
    runtime.query_labels_matrix.indptr = {0, 0, 1};
    runtime.query_labels_matrix.indices = {0};
    runtime.query_labels_matrix.row_labels = {{}, {0}};
    runtime.query_ranges.rows = 2;
    runtime.query_ranges.range_count = 1;
    runtime.query_ranges.modes = {3, 3};
    runtime.query_ranges.lower_bounds = {2.0f, 2.0f};
    runtime.query_ranges.upper_bounds = {4.0f, 4.0f};
    runtime.route_to_original_query = {0, 0};
    runtime.route_to_clause = {0, 0};
    runtime.route_top_branch_ids = {0, 0};
    runtime.route_enabled = {1, 1};
    runtime.route_providers = {QueryRouteProvider::Label, QueryRouteProvider::RangePosting};
    runtime.route_execution_modes = {
        lowfreq ? QueryRouteExecution::GpuResidentScan
                : QueryRouteExecution::GraphTraversal,
        QueryRouteExecution::HostCandidateStream};
    runtime.route_candidate_sources = {
        lowfreq ? QueryCandidateSource::LabelPosting : QueryCandidateSource::None,
        QueryCandidateSource::RangePosting};
    runtime.query_group_ids = {0, UINT32_MAX};
    runtime.query_filter_labels = {0, UINT32_MAX};
    runtime.query_filter_bits = {1, 0};
    runtime.start_local_ids = {lowfreq ? UINT32_MAX : 0, UINT32_MAX};
    runtime.query_is_lowfreq = {lowfreq ? uint8_t{1} : uint8_t{0}, 0};
    runtime.metadata.group_posting_size = {5};
    runtime.lowfreq_index.label_size = {lowfreq ? uint32_t{2} : uint32_t{0}};
    runtime.high_query_indices = lowfreq ? std::vector<uint32_t>{}
                                         : std::vector<uint32_t>{0};
    runtime.low_query_indices = lowfreq ? std::vector<uint32_t>{0}
                                        : std::vector<uint32_t>{};
    return runtime;
}

GraphRuntime make_two_clause_runtime(bool first_lowfreq = true)
{
    GraphRuntime runtime;
    runtime.base_vectors.rows = 6;
    runtime.base_range_count = 1;
    runtime.original_query_count = 1;
    runtime.original_query_label_ops = {QueryLabelOp::And};
    runtime.original_query_clauses.rows = 1;
    runtime.original_query_clauses.query_indptr = {0, 2};
    runtime.original_query_clauses.label_indptr = {0, 1, 2};
    runtime.original_query_clauses.labels = {0, 1};
    runtime.original_query_clauses.top_branch_ids = {0, 1};
    runtime.original_query_clauses.uses_original_range = {1, 1};
    runtime.query_labels_matrix.rows = 4;
    runtime.query_labels_matrix.cols = 2;
    runtime.query_labels_matrix.indptr = {0, 1, 2, 3, 4};
    runtime.query_labels_matrix.indices = {0, 0, 1, 1};
    runtime.query_labels_matrix.row_labels = {{0}, {0}, {1}, {1}};
    runtime.query_ranges.rows = 4;
    runtime.query_ranges.range_count = 1;
    runtime.query_ranges.modes = {3, 3, 3, 3};
    runtime.query_ranges.lower_bounds = {2.0f, 2.0f, 2.0f, 2.0f};
    runtime.query_ranges.upper_bounds = {4.0f, 4.0f, 4.0f, 4.0f};
    runtime.route_to_original_query = {0, 0, 0, 0};
    runtime.route_to_clause = {0, 0, 1, 1};
    runtime.route_top_branch_ids = {0, 0, 1, 1};
    runtime.route_enabled = {1, 1, 1, 1};
    runtime.route_providers = {
        QueryRouteProvider::Label, QueryRouteProvider::RangePosting,
        QueryRouteProvider::Label, QueryRouteProvider::RangePosting};
    runtime.route_execution_modes = {
        first_lowfreq ? QueryRouteExecution::GpuResidentScan
                      : QueryRouteExecution::GraphTraversal,
        QueryRouteExecution::HostCandidateStream,
        QueryRouteExecution::GraphTraversal, QueryRouteExecution::HostCandidateStream};
    runtime.route_candidate_sources = {
        first_lowfreq ? QueryCandidateSource::LabelPosting
                      : QueryCandidateSource::None,
        QueryCandidateSource::RangePosting,
        QueryCandidateSource::None, QueryCandidateSource::RangePosting};
    runtime.query_group_ids = {0, UINT32_MAX, 1, UINT32_MAX};
    runtime.query_filter_labels = {0, UINT32_MAX, 1, UINT32_MAX};
    runtime.query_filter_bits = {1, 0, 2, 0};
    runtime.start_local_ids = {
        first_lowfreq ? UINT32_MAX : 0, UINT32_MAX, 0, UINT32_MAX};
    runtime.query_is_lowfreq = {first_lowfreq ? uint8_t{1} : uint8_t{0}, 0, 0, 0};
    runtime.metadata.group_posting_size = {
        first_lowfreq ? uint32_t{2} : uint32_t{5}, 5};
    runtime.lowfreq_index.label_size = {first_lowfreq ? uint32_t{2} : uint32_t{0}, 0};
    runtime.high_query_indices = first_lowfreq ? std::vector<uint32_t>{2}
                                               : std::vector<uint32_t>{0, 2};
    runtime.low_query_indices = first_lowfreq ? std::vector<uint32_t>{0}
                                              : std::vector<uint32_t>{};
    return runtime;
}

GraphRuntime make_range_only_or_runtime()
{
    GraphRuntime runtime;
    runtime.base_vectors.rows = 6;
    runtime.base_range_count = 1;
    runtime.original_query_count = 1;
    runtime.original_query_label_ops = {QueryLabelOp::Or};
    runtime.original_query_clauses.rows = 1;
    runtime.original_query_clauses.query_indptr = {0, 1};
    runtime.original_query_clauses.label_indptr = {0, 0};
    runtime.original_query_clauses.top_branch_ids = {0};
    runtime.original_query_clauses.uses_original_range = {1};
    runtime.query_labels_matrix.rows = 1;
    runtime.query_labels_matrix.indptr = {0, 0};
    runtime.query_labels_matrix.row_labels = {{}};
    runtime.query_ranges.rows = 1;
    runtime.query_ranges.range_count = 1;
    runtime.query_ranges.modes = {3};
    runtime.query_ranges.lower_bounds = {2.0f};
    runtime.query_ranges.upper_bounds = {4.0f};
    runtime.route_to_original_query = {0};
    runtime.route_to_clause = {0};
    runtime.route_top_branch_ids = {0};
    runtime.route_enabled = {1};
    runtime.route_providers = {QueryRouteProvider::RangePosting};
    runtime.route_execution_modes = {QueryRouteExecution::GpuResidentScan};
    runtime.route_candidate_sources = {QueryCandidateSource::RangePosting};
    runtime.query_group_ids = {UINT32_MAX};
    runtime.query_filter_labels = {UINT32_MAX};
    runtime.query_filter_bits = {0};
    runtime.start_local_ids = {UINT32_MAX};
    runtime.query_is_lowfreq = {0};
    return runtime;
}

GraphRuntime make_pure_range_runtime()
{
    GraphRuntime runtime = make_range_only_or_runtime();
    runtime.original_query_label_ops = {QueryLabelOp::And};
    runtime.original_query_clauses.query_indptr = {0, 0};
    runtime.original_query_clauses.label_indptr = {0};
    runtime.original_query_clauses.top_branch_ids.clear();
    runtime.original_query_clauses.uses_original_range.clear();
    runtime.query_labels_matrix.rows = 3;
    runtime.query_labels_matrix.indptr = {0, 0, 0, 0};
    runtime.query_labels_matrix.row_labels = {{}, {}, {}};
    runtime.query_ranges.rows = 3;
    runtime.query_ranges.modes = {3, 3, 3};
    runtime.query_ranges.lower_bounds = {2.0f, 2.0f, 2.0f};
    runtime.query_ranges.upper_bounds = {4.0f, 4.0f, 4.0f};
    runtime.route_to_original_query = {0, 0, 0};
    runtime.route_to_clause = {UINT32_MAX, UINT32_MAX, UINT32_MAX};
    runtime.route_top_branch_ids = {UINT32_MAX, UINT32_MAX, UINT32_MAX};
    runtime.route_enabled = {1, 1, 1};
    runtime.route_providers = {
        QueryRouteProvider::RangePosting,
        QueryRouteProvider::Label,
        QueryRouteProvider::Label};
    runtime.route_execution_modes = {
        QueryRouteExecution::GpuResidentScan,
        QueryRouteExecution::GraphTraversal,
        QueryRouteExecution::GraphTraversal};
    runtime.route_candidate_sources = {
        QueryCandidateSource::RangePosting,
        QueryCandidateSource::None,
        QueryCandidateSource::None};
    runtime.query_group_ids = {UINT32_MAX, 0, 1};
    runtime.query_filter_labels = {UINT32_MAX, 0, 1};
    runtime.query_filter_bits = {0, 1, 2};
    runtime.start_local_ids = {UINT32_MAX, 0, 0};
    runtime.query_is_lowfreq = {0, 0, 0};
    runtime.high_query_indices = {1, 2};
    return runtime;
}

RangeAccessPlannerConfig config(
    const std::string& filter_mode,
    const std::string& policy = "auto")
{
    RangeAccessPlannerConfig value;
    value.mode = "select";
    value.and_filter_mode = filter_mode;
    value.range_label_and_policy = policy;
    value.scan_threshold = 4;
    return value;
}

}  // namespace

int main()
{
    const auto path = std::filesystem::temp_directory_path() /
        ("flashanns-range-access-planner-" + std::to_string(getpid()) + ".bin");
    write_index(path);
    rp::Index index = rp::Index::load(path.string());
    try {
        GraphRuntime automatic = make_runtime(false);
        plan_graph_range_access(automatic, index, config("auto"));
        require(automatic.route_enabled == std::vector<uint8_t>({0, 1}),
                "auto mode did not replace the high-frequency label route");
        require(automatic.route_execution_modes[1] ==
                    QueryRouteExecution::GpuResidentScanPostFilter,
                "auto mode did not choose shortlist post-filtering");

        RangeAccessPlannerConfig narrow_auto = config("auto");
        GraphRuntime wide_automatic = make_runtime(false);
        plan_graph_range_access(wide_automatic, index, narrow_auto);
        require(wide_automatic.route_enabled == std::vector<uint8_t>({0, 1}) &&
                    wide_automatic.route_execution_modes[1] ==
                        QueryRouteExecution::GpuResidentScanPostFilter,
                "auto mode incorrectly used shortlist as its posting threshold");

        RangeAccessPlannerConfig over_threshold = config("auto");
        over_threshold.scan_threshold = 1;
        GraphRuntime threshold_limited = make_runtime(false);
        plan_graph_range_access(threshold_limited, index, over_threshold);
        require(threshold_limited.route_enabled == std::vector<uint8_t>({1, 0}),
                "auto mode selected range posting above its scan threshold");

        RangeAccessPlannerConfig explicit_post = config("post");
        GraphRuntime wide_post = make_runtime(false);
        plan_graph_range_access(wide_post, index, explicit_post);
        require(wide_post.route_enabled == std::vector<uint8_t>({0, 1}) &&
                    wide_post.route_execution_modes[1] ==
                        QueryRouteExecution::GpuResidentScanPostFilter,
                "explicit post mode did not expose the wider approximate path");

        GraphRuntime lowfreq = make_runtime(true);
        plan_graph_range_access(lowfreq, index, config("auto"));
        require(lowfreq.route_enabled == std::vector<uint8_t>({1, 0}),
                "low-frequency label posting did not take hard priority");

        GraphRuntime forced_label = make_runtime(false);
        plan_graph_range_access(
            forced_label, index, config("post", "label_prefer"));
        require(forced_label.route_enabled == std::vector<uint8_t>({1, 0}) &&
                    forced_label.range_access_planner_profile.label_selected_clauses == 1,
                "label-prefer policy did not retain only the label provider");

        GraphRuntime forced_range_lowfreq = make_runtime(true);
        plan_graph_range_access(
            forced_range_lowfreq, index, config("post", "range_prefer"));
        require(forced_range_lowfreq.route_enabled == std::vector<uint8_t>({0, 1}) &&
                    forced_range_lowfreq.route_execution_modes[1] ==
                        QueryRouteExecution::GpuResidentScanPostFilter &&
                    forced_range_lowfreq.range_access_planner_profile
                            .lowfreq_priority_clauses == 0,
                "range-prefer policy did not override low-frequency label priority");

        GraphRuntime forced_dual = make_runtime(true);
        plan_graph_range_access(
            forced_dual, index, config("post", "dual"));
        require(forced_dual.route_enabled == std::vector<uint8_t>({1, 1}) &&
                    forced_dual.low_query_indices == std::vector<uint32_t>({0}) &&
                    forced_dual.range_access_planner_profile.dual_selected_clauses == 1,
                "dual policy did not retain label and range-posting providers");

        bool rejected_unsafe_forced_range = false;
        try {
            RangeAccessPlannerConfig unsafe = config("post", "range_prefer");
            unsafe.scan_threshold = 1;
            GraphRuntime no_range_provider = make_runtime(false);
            plan_graph_range_access(no_range_provider, index, unsafe);
        } catch (const std::runtime_error& error) {
            rejected_unsafe_forced_range =
                std::string(error.what()).find("no threshold-safe range provider") !=
                std::string::npos;
        }
        require(rejected_unsafe_forced_range,
                "range-prefer policy silently crossed the posting threshold");

        GraphRuntime pre = make_runtime(false);
        plan_graph_range_access(pre, index, config("pre"));
        require(pre.route_enabled == std::vector<uint8_t>({0, 1}) &&
                    pre.route_execution_modes[1] ==
                        QueryRouteExecution::HostCandidateStream,
                "pre mode did not retain the CPU candidate path");

        GraphRuntime label = make_runtime(false);
        plan_graph_range_access(label, index, config("label"));
        require(label.route_enabled == std::vector<uint8_t>({1, 0}),
                "label mode did not disable range-posting AND");

        GraphRuntime clauses = make_two_clause_runtime();
        plan_graph_range_access(clauses, index, config("auto"));
        require(clauses.route_enabled == std::vector<uint8_t>({1, 0, 0, 1}),
                "range access was selected per query instead of per DNF clause");
        require(clauses.low_query_indices == std::vector<uint32_t>({0}) &&
                    clauses.high_query_indices.empty(),
                "clause-local provider selection left stale label execution queues");

        GraphRuntime coalesced = make_two_clause_runtime(false);
        plan_graph_range_access(coalesced, index, config("auto"));
        require(coalesced.route_enabled ==
                    std::vector<uint8_t>({0, 1, 0, 0}) &&
                    coalesced.route_clause_indptr ==
                        std::vector<uint64_t>({0, 1, 3, 4, 5}) &&
                    coalesced.route_clause_ids ==
                        std::vector<uint32_t>({0, 0, 1, 1, 1}),
                "equivalent range clauses were not coalesced onto one route");
        require(coalesced.range_access_planner_profile.range_posting_routes == 1 &&
                    coalesced.range_access_planner_profile.postfilter_routes == 1,
                "coalesced range route telemetry counted logical duplicates");

        GraphRuntime range_graph = make_two_clause_runtime(false);
        RangeAccessPlannerConfig graph_config = config("auto");
        graph_config.scan_threshold = 1;
        graph_config.range_graph_hybrid_and_enabled = true;
        plan_graph_range_access(range_graph, index, graph_config);
        require(range_graph.route_enabled ==
                    std::vector<uint8_t>({0, 0, 0, 0}) &&
                    range_graph.range_graph_and_original_queries ==
                        std::vector<uint8_t>({1}) &&
                    range_graph.range_graph_clause_indptr ==
                        std::vector<uint64_t>({0, 2}) &&
                    range_graph.range_graph_clause_ids ==
                        std::vector<uint32_t>({0, 1}),
                "range graph did not retain its assigned DNF clause set");

        GraphRuntime forced_range_graph = make_runtime(true);
        RangeAccessPlannerConfig forced_range_graph_config =
            config("post", "range_prefer");
        forced_range_graph_config.scan_threshold = 1;
        forced_range_graph_config.range_graph_hybrid_and_enabled = true;
        plan_graph_range_access(
            forced_range_graph, index, forced_range_graph_config);
        require(forced_range_graph.route_enabled == std::vector<uint8_t>({0, 0}) &&
                    forced_range_graph.range_graph_and_original_queries ==
                        std::vector<uint8_t>({1}),
                "range-prefer policy did not hand a wide low-frequency clause to the range graph");

        GraphRuntime dual_range_graph = make_runtime(true);
        RangeAccessPlannerConfig dual_range_graph_config = config("post", "dual");
        dual_range_graph_config.scan_threshold = 1;
        dual_range_graph_config.range_graph_hybrid_and_enabled = true;
        plan_graph_range_access(dual_range_graph, index, dual_range_graph_config);
        require(dual_range_graph.route_enabled == std::vector<uint8_t>({1, 0}) &&
                    dual_range_graph.range_graph_and_original_queries ==
                        std::vector<uint8_t>({1}) &&
                    dual_range_graph.range_access_planner_profile.dual_selected_clauses == 1,
                "dual policy did not retain the label provider alongside the range graph");

        GraphRuntime range_or = make_range_only_or_runtime();
        RangeAccessPlannerConfig range_or_config = config("auto");
        range_or_config.scan_threshold = 1;
        range_or_config.range_graph_hybrid_or_enabled = true;
        plan_graph_range_access(range_or, index, range_or_config);
        require(range_or.route_enabled == std::vector<uint8_t>({0}) &&
                    range_or.range_graph_or_original_queries ==
                        std::vector<uint8_t>({1}) &&
                    range_or.range_graph_clause_indptr ==
                        std::vector<uint64_t>({0, 1}) &&
                    range_or.range_graph_clause_ids ==
                        std::vector<uint32_t>({0}),
                "range OR did not honor the posting threshold and graph fallback");

        GraphRuntime narrow_pure_range = make_pure_range_runtime();
        plan_graph_range_access(narrow_pure_range, index, config("auto"));
        require(narrow_pure_range.route_enabled ==
                    std::vector<uint8_t>({1, 0, 0}) &&
                    narrow_pure_range.high_query_indices.empty() &&
                    narrow_pure_range.range_graph_or_original_queries ==
                        std::vector<uint8_t>({0}),
                "narrow pure range did not select its posting route");

        GraphRuntime wide_pure_range = make_pure_range_runtime();
        RangeAccessPlannerConfig pure_range_graph_config = config("auto");
        pure_range_graph_config.scan_threshold = 1;
        plan_graph_range_access(wide_pure_range, index, pure_range_graph_config);
        require(wide_pure_range.route_enabled ==
                    std::vector<uint8_t>({0, 1, 1}) &&
                    wide_pure_range.high_query_indices ==
                        std::vector<uint32_t>({1, 2}) &&
                    wide_pure_range.range_graph_or_original_queries ==
                        std::vector<uint8_t>({0}),
                "wide pure range did not retain its primary range-graph routes");
    } catch (...) {
        std::filesystem::remove(path);
        throw;
    }
    std::filesystem::remove(path);
    return 0;
}
