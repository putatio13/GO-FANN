#pragma once

#include "data/dataloader.h"

#include <fstream>
#include <chrono>
#include <stdexcept>
#include <nlohmann/json.hpp>

namespace go_fann {

inline double setup_elapsed_ms(std::chrono::steady_clock::time_point start,
                               std::chrono::steady_clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// Durations are host steady-clock wall time. Nested detail objects are never
// added to their parent's total; parallel fuzzy worker times are not summed.
inline nlohmann::json query_preparation_timing_json(
    const GraphQueryPreparationTiming& t)
{
    nlohmann::json query = {
        {"vector_inspect", t.query_vector_inspect_ms},
        {"vector_read_and_copy", t.query_vector_read_ms},
        {"label_spmat_read_and_copy", t.query_label_spmat_read_ms},
        {"label_materialize", t.query_label_materialize_ms},
        {"manifest_prepare", t.query_manifest_prepare_ms},
        {"range_read", t.query_range_read_ms},
        {"route_generation", t.route_generation_ms},
        {"range_access_plan", t.range_access_plan_ms}};
    nlohmann::json manifest = {
        {"file_read", t.query_manifest_file_read_detail_ms},
        {"json_parse", t.query_manifest_json_parse_detail_ms},
        {"record_select", t.query_manifest_record_select_detail_ms},
        {"exact_extract", t.query_manifest_exact_extract_detail_ms},
        {"fuzzy_expand_wall", t.query_manifest_fuzzy_expand_wall_detail_ms},
        {"row_materialize", t.query_manifest_row_materialize_detail_ms}};
    double manifest_parts = 0.0;
    for (const auto& item : manifest.items()) manifest_parts += item.value().get<double>();
    manifest["other_decode_dispatch"] = t.query_manifest_prepare_ms - manifest_parts;
    nlohmann::json reusable = t.static_stages_ms;
    reusable["manifest_discovery"] = t.static_manifest_discovery_ms;
    reusable["manifest_ngram_load"] = t.static_manifest_ngram_load_ms;
    reusable["manifest_operator_load"] = t.static_manifest_operator_load_ms;
    reusable["manifest_fallback_index_build"] = t.static_manifest_fallback_index_build_ms;
    reusable["label_cardinality_init"] = t.static_label_cardinality_init_ms;
    reusable["range_posting_index_load"] = t.static_range_posting_index_load_ms;
    reusable["host_candidate_label_load"] = t.host_candidate_label_load_ms;
    reusable["prefetch_build"] = t.prefetch_build_ms;
    double reusable_total = 0.0;
    for (const auto& item : reusable.items()) reusable_total += item.value().get<double>();
    return {{"runtime_load_total_ms", t.runtime_load_total_ms},
            {"query_prepare_total_ms", t.query_stage_sum_ms()},
            {"query_stages_ms", std::move(query)},
            {"manifest_detail_ms", std::move(manifest)},
            {"route_detail_ms", {{"select_and_emit", t.route_select_emit_ms},
                                 {"union", t.route_union_ms},
                                 {"payload_materialize", t.route_payload_materialize_ms}}},
            {"reusable_static_total_ms", reusable_total},
            {"reusable_static_stages_ms", std::move(reusable)},
            {"other_runtime_load_ms",
             t.runtime_load_total_ms - t.query_stage_sum_ms() - reusable_total}};
}

class QuerySetupReporter {
public:
    QuerySetupReporter(const std::string& path, const std::string& backend)
        : path_(path), backend_(backend), stream_(path)
    {
        if (!stream_) throw std::runtime_error("unable to open query setup report " + path);
    }

    void append(nlohmann::json record)
    {
        record["schema_version"] = 1;
        record["backend"] = backend_;
        record["time_unit"] = "ms";
        stream_ << record.dump() << '\n';
        stream_.flush();
        if (!stream_) throw std::runtime_error("unable to write query setup report " + path_);
    }

private:
    std::string path_;
    std::string backend_;
    std::ofstream stream_;
};

} // namespace go_fann
