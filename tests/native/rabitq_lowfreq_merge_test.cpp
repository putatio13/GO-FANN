#include "app/config_loader.h"
#include "search/rabitq/rabitq_lowfreq_merge.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using go_fann::backend_rabitq::lowfreq_merge::CandidateId;
using go_fann::backend_rabitq::lowfreq_merge::Diagnostics;
using go_fann::backend_rabitq::lowfreq_merge::FinalMergeStrategy;
using go_fann::backend_rabitq::lowfreq_merge::RouteGroup;

void require(bool condition, const char* message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_grouping_dedup_filter_cap_and_empty_routes()
{
    const std::vector<uint32_t> route_to_original = {0, 1, 0, 1};
    const std::vector<uint32_t> route_ids = {2, 1, 0, 3};
    const auto groups = go_fann::backend_rabitq::lowfreq_merge::build_route_groups(
        route_ids, route_to_original, 2);
    require(groups.size() == 2, "non-contiguous routes did not form two original-query groups");
    require(groups[0].original_query == 0 &&
                groups[0].route_positions == std::vector<size_t>({0, 2}),
            "original-query grouping assumed contiguous route order");

    const CandidateId invalid = UINT32_MAX;
    const std::vector<CandidateId> ids = {
        10, 20, invalid, 40,
        invalid, invalid, invalid, invalid,
        10, 30, 20, 50,
        invalid, invalid, invalid, invalid,
    };
    const std::vector<float> distances = {
        0.5f, 0.2f, 0.0f, 0.4f,
        0.0f, 0.0f, 0.0f, 0.0f,
        0.1f, 0.2f, 0.2f, 0.3f,
        0.0f, 0.0f, 0.0f, 0.0f,
    };
    Diagnostics diagnostics;
    const auto retained =
        go_fann::backend_rabitq::lowfreq_merge::select_group_candidates(
            groups[0],
            route_ids,
            ids,
            distances,
            4,
            100,
            3,
            [](uint32_t route_id, uint32_t sample_id) {
                return !(route_id == 0 && sample_id == 30);
            },
            diagnostics);

    require(retained.size() == 3, "merged shortlist cap was not enforced");
    require(retained[0].sample_id == 10 && retained[0].approximate_distance == 0.1f,
            "duplicate sample did not retain its best approximate distance");
    require(retained[1].sample_id == 20 &&
                retained[1].source_route_id == 2 && retained[1].source_rank == 1,
            "duplicate approximate-distance tie was not resolved by deterministic provenance");
    require(retained[2].sample_id == 50,
            "route-local filter rejection or cap ordering produced the wrong retained sample");
    require(diagnostics.route_candidate_slots == 8 &&
                diagnostics.route_valid_candidates == 6 &&
                diagnostics.unique_candidates == 4 &&
                diagnostics.retained_candidates == 3,
            "merged candidate diagnostics are inconsistent");

    using go_fann::backend_rabitq::lowfreq_merge::CandidateStage;
    std::vector<CandidateStage> stages;
    size_t hash_calls = 0;
    Diagnostics observed_diagnostics;
    const auto observed =
        go_fann::backend_rabitq::lowfreq_merge::select_group_candidates(
            groups[0], route_ids, ids, distances, 4, 100, 3,
            [](uint32_t route_id, uint32_t sample_id) {
                return !(route_id == 0 && sample_id == 30);
            }, observed_diagnostics,
            [&](CandidateStage stage) { stages.push_back(stage); },
            [&](auto&& operation) { ++hash_calls; return operation(); });
    require(stages == std::vector<CandidateStage>({CandidateStage::Collect, CandidateStage::Rank}),
            "candidate observer stages are out of order");
    require(hash_calls == diagnostics.route_valid_candidates && observed.size() == retained.size(),
            "candidate observer changed hash calls or retained count");
    for (size_t i = 0; i < retained.size(); ++i) {
        require(observed[i].sample_id == retained[i].sample_id &&
                    observed[i].approximate_distance == retained[i].approximate_distance &&
                    observed[i].source_route_id == retained[i].source_route_id &&
                    observed[i].source_rank == retained[i].source_rank,
                "candidate observer changed selection or provenance");
    }
    require(observed_diagnostics.route_candidate_slots == diagnostics.route_candidate_slots &&
                observed_diagnostics.route_valid_candidates == diagnostics.route_valid_candidates &&
                observed_diagnostics.unique_candidates == diagnostics.unique_candidates &&
                observed_diagnostics.retained_candidates == diagnostics.retained_candidates,
            "candidate observer changed diagnostic work counts");

    Diagnostics empty_diagnostics;
    const auto empty =
        go_fann::backend_rabitq::lowfreq_merge::select_group_candidates(
            groups[1],
            route_ids,
            ids,
            distances,
            4,
            100,
            3,
            [](uint32_t, uint32_t) { return true; },
            empty_diagnostics);
    require(empty.empty(), "empty routes produced merged candidates");
    require(empty_diagnostics.route_candidate_slots == 8 &&
                empty_diagnostics.route_valid_candidates == 0,
            "empty-route diagnostics are inconsistent");
}

void test_equal_approximate_distance_uses_stable_sample_order()
{
    const std::vector<uint32_t> route_to_original = {0, 0};
    const std::vector<uint32_t> route_ids = {1, 0};
    const auto groups = go_fann::backend_rabitq::lowfreq_merge::build_route_groups(
        route_ids, route_to_original, 1);
    const std::vector<CandidateId> ids = {
        60, UINT32_MAX,
        UINT32_MAX, 50,
    };
    const std::vector<float> distances = {
        0.25f, 0.0f,
        0.0f, 0.25f,
    };
    Diagnostics diagnostics;
    const auto retained =
        go_fann::backend_rabitq::lowfreq_merge::select_group_candidates(
            groups[0],
            route_ids,
            ids,
            distances,
            2,
            100,
            1,
            [](uint32_t, uint32_t) { return true; },
            diagnostics);
    require(retained.size() == 1 && retained[0].sample_id == 50,
            "source rank changed the (approximate distance, sample id) cap order");
}

void test_invalid_samples_and_original_query_mappings_are_rejected()
{
    const std::vector<uint32_t> route_ids = {0};
    const auto groups = go_fann::backend_rabitq::lowfreq_merge::build_route_groups(
        route_ids, std::vector<uint32_t>{0}, 1);
    Diagnostics diagnostics;
    const auto retained =
        go_fann::backend_rabitq::lowfreq_merge::select_group_candidates(
            groups[0],
            route_ids,
            std::vector<CandidateId>{100, 99},
            std::vector<float>{0.1f, 0.2f},
            2,
            100,
            2,
            [](uint32_t, uint32_t) { return true; },
            diagnostics);
    require(retained.size() == 1 && retained[0].sample_id == 99,
            "out-of-range sample id entered the merged shortlist");
    require(diagnostics.route_candidate_slots == 2 &&
                diagnostics.route_valid_candidates == 1 &&
                diagnostics.unique_candidates == 1 &&
                diagnostics.retained_candidates == 1,
            "invalid-sample diagnostics are inconsistent");

    bool threw = false;
    try {
        (void)go_fann::backend_rabitq::lowfreq_merge::build_route_groups(
            route_ids, std::vector<uint32_t>{1}, 1);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    require(threw, "out-of-range original query mapping was accepted");
}

void test_group_result_uses_one_representative_and_all_member_steps()
{
    RouteGroup group;
    group.original_query = 4;
    group.route_positions = {0, 2};
    std::vector<std::vector<uint64_t>> route_results(3);
    std::vector<int> route_steps(3, -1);
    std::vector<std::vector<float>> route_distances(3);
    const std::vector<uint64_t> result = {7, 8};
    const std::vector<float> distances = {1.5f, 2.5f};
    Diagnostics diagnostics;
    go_fann::backend_rabitq::lowfreq_merge::write_group_result_to_representative_route(
        group, result, distances, 2, route_results, route_steps, route_distances, diagnostics);
    require(route_results[0] == result && route_distances[0] == distances,
            "merged result was not written to the deterministic representative route");
    require(route_results[2].empty() && route_distances[2].empty(),
            "non-representative route materialized a duplicate result row");
    require(route_steps[0] == 2 && route_steps[2] == 2,
            "merged rerank step was not written to every member route");
    require(route_results[1].empty() && route_steps[1] == -1,
            "compact group output modified an unrelated route");
    require(diagnostics.representative_result_rows == 1 &&
                diagnostics.compact_empty_result_rows == 1 &&
                diagnostics.route_step_rows == 2,
            "compact group output diagnostics are inconsistent");
}

void test_empty_group_result_keeps_all_rows_empty_and_writes_steps()
{
    RouteGroup group;
    group.original_query = 4;
    group.route_positions = {0, 2};
    std::vector<std::vector<uint64_t>> route_results(3);
    std::vector<int> route_steps(3, -1);
    std::vector<std::vector<float>> route_distances(3);
    Diagnostics diagnostics;
    go_fann::backend_rabitq::lowfreq_merge::write_group_result_to_representative_route(
        group, {}, {}, 0, route_results, route_steps, route_distances, diagnostics);

    require(route_results[0].empty() && route_results[2].empty() &&
                route_distances[0].empty() && route_distances[2].empty(),
            "empty merged group materialized a representative result row");
    require(route_steps[0] == 0 && route_steps[2] == 0,
            "empty merged group did not propagate its zero rerank step");
    require(diagnostics.representative_result_rows == 0 &&
                diagnostics.compact_empty_result_rows == 2 &&
                diagnostics.route_step_rows == 2,
            "empty merged group diagnostics are inconsistent");
}

void test_host_staging_buffers_grow_only_when_required()
{
    std::vector<uint64_t> buffer;
    const bool initial_reuse =
        go_fann::backend_rabitq::lowfreq_merge::ensure_reusable_host_staging_slots(
            buffer, 3);
    require(!initial_reuse && buffer.size() == 3,
            "empty host staging buffer was incorrectly reported as reused");
    buffer = {11, 12, 13};
    const uint64_t* initial_data = buffer.data();

    const bool smaller_reuse =
        go_fann::backend_rabitq::lowfreq_merge::ensure_reusable_host_staging_slots(
            buffer, 2);
    require(smaller_reuse && buffer.size() == 3 && buffer.data() == initial_data &&
                buffer == std::vector<uint64_t>({11, 12, 13}),
            "smaller host staging request resized or initialized reused storage");

    const bool larger_reuse =
        go_fann::backend_rabitq::lowfreq_merge::ensure_reusable_host_staging_slots(
            buffer, 5);
    require(!larger_reuse && buffer.size() == 5 && buffer[0] == 11 && buffer[1] == 12 &&
                buffer[2] == 13,
            "host staging growth discarded the existing prefix");
}

void test_empty_route_rows_preserve_broadcast_final_output()
{
    const std::vector<uint32_t> route_to_original = {0, 0, 0, 1};
    const std::vector<int> route_steps = {5, 2, 2, 3};
    const std::vector<std::vector<uint64_t>> broadcast_results = {
        {9, 7, 4}, {7, 8}, {7, 8}, {},
    };
    const std::vector<std::vector<float>> broadcast_distances = {
        {0.5f, 2.0f, 1.0f}, {0.4f, 0.5f}, {0.4f, 0.5f}, {},
    };
    const std::vector<std::vector<uint64_t>> compact_results = {
        {9, 7, 4}, {7, 8}, {}, {},
    };
    const std::vector<std::vector<float>> compact_distances = {
        {0.5f, 2.0f, 1.0f}, {0.4f, 0.5f}, {}, {},
    };

    std::vector<std::vector<uint64_t>> broadcast_merged;
    std::vector<std::vector<uint64_t>> compact_merged;
    std::vector<int> broadcast_steps;
    std::vector<int> compact_steps;
    go_fann::backend_rabitq::lowfreq_merge::merge_route_results_to_original(
        route_to_original,
        2,
        broadcast_results,
        broadcast_distances,
        route_steps,
        3,
        broadcast_merged,
        broadcast_steps);
    go_fann::backend_rabitq::lowfreq_merge::merge_route_results_to_original(
        route_to_original,
        2,
        compact_results,
        compact_distances,
        route_steps,
        3,
        compact_merged,
        compact_steps);

    require(compact_merged == broadcast_merged,
            "empty compact route rows changed final original-query results");
    require(compact_steps == broadcast_steps && compact_steps == std::vector<int>({9, 3}),
            "compact route rows changed all-route step aggregation");
    require(compact_merged[0] == std::vector<uint64_t>({7, 8, 9}),
            "compact final merge changed mixed-route deduplication or distance tie ordering");
    require(compact_merged[1] ==
                std::vector<uint64_t>({UINT32_MAX, UINT32_MAX, UINT32_MAX}),
            "all-empty original query did not preserve empty final output");
}

void test_single_route_fast_path_preserves_results_distances_and_steps()
{
    const std::vector<uint32_t> route_to_original = {0, 1};
    const std::vector<std::vector<uint64_t>> route_results = {
        {5, 7, UINT32_MAX}, {},
    };
    const std::vector<std::vector<float>> route_distances = {
        {0.1f, 0.2f, std::numeric_limits<float>::max()}, {},
    };
    const std::vector<int> route_steps = {11, 0};
    std::vector<std::vector<uint64_t>> merged_results;
    std::vector<std::vector<float>> merged_distances;
    std::vector<int> merged_steps;

    const auto strategy =
        go_fann::backend_rabitq::lowfreq_merge::merge_route_results_to_original(
            route_to_original,
            2,
            route_results,
            route_distances,
            route_steps,
            2,
            merged_results,
            merged_steps,
            &merged_distances);

    require(strategy == FinalMergeStrategy::SingleRouteCopy,
            "single-route input did not use the direct-copy merge");
    require(merged_results ==
                std::vector<std::vector<uint64_t>>({{5, 7}, {UINT32_MAX, UINT32_MAX}}),
            "single-route fast path changed result ordering or empty rows");
    require(merged_distances ==
                std::vector<std::vector<float>>({{0.1f, 0.2f},
                                                  {std::numeric_limits<float>::max(),
                                                   std::numeric_limits<float>::max()}}),
            "single-route fast path did not preserve distances");
    require(merged_steps == std::vector<int>({11, 0}),
            "single-route fast path did not preserve route steps");
}

void test_sorted_multi_route_heap_merge_supports_large_dynamic_route_caps()
{
    constexpr size_t route_count_per_query = 32;
    constexpr size_t query_count = 2;
    constexpr size_t route_count = route_count_per_query * query_count;
    std::vector<uint32_t> route_to_original(route_count);
    std::vector<std::vector<uint64_t>> route_results(route_count);
    std::vector<std::vector<float>> route_distances(route_count);
    std::vector<int> route_steps(route_count, 1);
    for (size_t route_idx = 0; route_idx < route_count; ++route_idx) {
        const uint32_t original_query = static_cast<uint32_t>(route_idx % query_count);
        const size_t query_route_idx = route_idx / query_count;
        route_to_original[route_idx] = original_query;
        route_results[route_idx] = {
            42 + original_query,
            100 + static_cast<uint64_t>(original_query) * 1000 + query_route_idx,
        };
        route_distances[route_idx] = {
            0.01f + static_cast<float>(query_route_idx) * 0.001f,
            1.0f + static_cast<float>(query_route_idx),
        };
    }

    std::vector<std::vector<uint64_t>> merged_results;
    std::vector<std::vector<float>> merged_distances;
    std::vector<int> merged_steps;
    const auto strategy =
        go_fann::backend_rabitq::lowfreq_merge::merge_route_results_to_original(
            route_to_original,
            query_count,
            route_results,
            route_distances,
            route_steps,
            10,
            merged_results,
            merged_steps,
            &merged_distances);

    require(strategy == FinalMergeStrategy::SortedRouteHeap,
            "sorted multi-route input did not use the heap merge");
    require(merged_results.size() == query_count && merged_results[0].size() == 10 &&
                merged_results[1].size() == 10,
            "heap merge returned the wrong output shape");
    for (size_t original_query = 0; original_query < query_count; ++original_query) {
        require(merged_results[original_query][0] == 42 + original_query,
                "heap merge did not deduplicate the shared best candidate");
        for (size_t rank = 1; rank < 10; ++rank) {
            const uint64_t expected =
                99 + rank + static_cast<uint64_t>(original_query) * 1000;
            require(merged_results[original_query][rank] == expected,
                    "heap merge did not preserve global distance order");
        }
    }
    require(merged_steps ==
                std::vector<int>({static_cast<int>(route_count_per_query),
                                  static_cast<int>(route_count_per_query)}),
            "heap merge did not aggregate every route step");

    auto unsorted_results = route_results;
    auto unsorted_distances = route_distances;
    for (size_t route_idx = 0; route_idx < route_count; ++route_idx) {
        std::reverse(unsorted_results[route_idx].begin(), unsorted_results[route_idx].end());
        std::reverse(unsorted_distances[route_idx].begin(),
                     unsorted_distances[route_idx].end());
    }
    std::vector<std::vector<uint64_t>> fallback_results;
    std::vector<std::vector<float>> fallback_distances;
    std::vector<int> fallback_steps;
    go_fann::backend_rabitq::lowfreq_merge::merge_route_results_to_original(
        route_to_original,
        query_count,
        unsorted_results,
        unsorted_distances,
        route_steps,
        10,
        fallback_results,
        fallback_steps,
        &fallback_distances);
    require(merged_results == fallback_results && merged_distances == fallback_distances &&
                merged_steps == fallback_steps,
            "heap merge disagrees with the compatible hash fallback");
}

void test_unsorted_multi_route_input_uses_compatible_fallback()
{
    const std::vector<uint32_t> route_to_original = {0, 0};
    const std::vector<std::vector<uint64_t>> route_results = {{9, 7}, {8}};
    const std::vector<std::vector<float>> route_distances = {{0.9f, 0.7f}, {0.8f}};
    const std::vector<int> route_steps = {2, 3};
    std::vector<std::vector<uint64_t>> merged_results;
    std::vector<std::vector<float>> merged_distances;
    std::vector<int> merged_steps;

    const bool used_fast_path =
        go_fann::backend_rabitq::lowfreq_merge::try_merge_sorted_routes_fast_path(
            route_to_original,
            1,
            route_results,
            route_distances,
            route_steps,
            3,
            merged_results,
            merged_steps,
            &merged_distances);
    require(!used_fast_path, "unsorted route input was accepted by the heap merge");

    const auto strategy =
        go_fann::backend_rabitq::lowfreq_merge::merge_route_results_to_original(
            route_to_original,
            1,
            route_results,
            route_distances,
            route_steps,
            3,
            merged_results,
            merged_steps,
            &merged_distances);
    require(strategy == FinalMergeStrategy::HashDedupSort,
            "unsorted route input did not use the compatible hash fallback");
    require(merged_results == std::vector<std::vector<uint64_t>>({{7, 8, 9}}),
            "unsorted route fallback changed final result order");
    require(merged_steps == std::vector<int>({5}),
            "unsorted route fallback changed step aggregation");
}

void test_route_step_overflow_is_rejected()
{
    bool threw = false;
    try {
        std::vector<std::vector<uint64_t>> merged_results;
        std::vector<int> merged_steps;
        go_fann::backend_rabitq::lowfreq_merge::merge_route_results_to_original(
            std::vector<uint32_t>{0, 0},
            1,
            std::vector<std::vector<uint64_t>>(2),
            std::vector<std::vector<float>>(2),
            std::vector<int>{std::numeric_limits<int>::max(), 1},
            1,
            merged_results,
            merged_steps);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    require(threw, "overflowing route-step aggregation was accepted");
}

void test_invalid_route_mapping_is_rejected()
{
    bool threw = false;
    try {
        (void)go_fann::backend_rabitq::lowfreq_merge::build_route_groups(
            std::vector<uint32_t>{2}, std::vector<uint32_t>{0, 0}, 1);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    require(threw, "out-of-range route mapping was accepted");
}

void test_config_loader_accepts_root_and_runtime_fields()
{
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path root_path =
        std::filesystem::temp_directory_path() /
        ("flashanns-lowfreq-merge-root-" + std::to_string(nonce) + ".json");
    const std::filesystem::path runtime_path =
        std::filesystem::temp_directory_path() /
        ("flashanns-lowfreq-merge-runtime-" + std::to_string(nonce) + ".json");
    const std::filesystem::path rabitq_path =
        std::filesystem::temp_directory_path() /
        ("flashanns-lowfreq-merge-rabitq-" + std::to_string(nonce) + ".json");
    {
        std::ofstream output(root_path);
        output << "{\"lowfreq_rerank_merge_routes\":true}\n";
    }
    {
        std::ofstream output(runtime_path);
        output << "{\"runtime\":{\"lowfreq_rerank_merge_routes\":true}}\n";
    }
    {
        std::ofstream output(rabitq_path);
        output << "{\"rabitq\":{\"lowfreq_rerank_merge_routes\":true}}\n";
    }
    try {
        const auto root = go_fann::load_config_file(root_path.string());
        const auto runtime = go_fann::load_config_file(runtime_path.string());
        const auto rabitq = go_fann::load_config_file(rabitq_path.string());
        require(root.lowfreq_rerank_merge_routes.value_or(false),
                "root lowfreq_rerank_merge_routes was not loaded");
        require(runtime.lowfreq_rerank_merge_routes.value_or(false),
                "runtime lowfreq_rerank_merge_routes was not loaded");
        require(rabitq.lowfreq_rerank_merge_routes.value_or(false),
                "rabitq lowfreq_rerank_merge_routes was not loaded");
    } catch (...) {
        std::filesystem::remove(root_path);
        std::filesystem::remove(runtime_path);
        std::filesystem::remove(rabitq_path);
        throw;
    }
    std::filesystem::remove(root_path);
    std::filesystem::remove(runtime_path);
    std::filesystem::remove(rabitq_path);
}

}  // namespace

int main()
{
    test_grouping_dedup_filter_cap_and_empty_routes();
    test_equal_approximate_distance_uses_stable_sample_order();
    test_invalid_samples_and_original_query_mappings_are_rejected();
    test_group_result_uses_one_representative_and_all_member_steps();
    test_empty_group_result_keeps_all_rows_empty_and_writes_steps();
    test_host_staging_buffers_grow_only_when_required();
    test_empty_route_rows_preserve_broadcast_final_output();
    test_single_route_fast_path_preserves_results_distances_and_steps();
    test_sorted_multi_route_heap_merge_supports_large_dynamic_route_caps();
    test_unsorted_multi_route_input_uses_compatible_fallback();
    test_route_step_overflow_is_rejected();
    test_invalid_route_mapping_is_rejected();
    test_config_loader_accepts_root_and_runtime_fields();
    return 0;
}
