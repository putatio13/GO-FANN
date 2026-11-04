#include "profiling/query_setup_timing.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <unistd.h>

namespace {
void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

void test_nonoverlapping_accounting()
{
    GraphQueryPreparationTiming timing;
    timing.runtime_load_total_ms = 100;
    timing.query_vector_read_ms = 3;
    timing.query_manifest_prepare_ms = 10;
    timing.query_manifest_file_read_detail_ms = 1;
    timing.query_manifest_json_parse_detail_ms = 2;
    timing.query_manifest_fuzzy_expand_wall_detail_ms = 4;
    timing.query_manifest_row_materialize_detail_ms = 1;
    timing.route_generation_ms = 5;
    timing.static_label_cardinality_init_ms = 20;
    timing.static_manifest_operator_load_ms = 30;
    timing.static_stages_ms["load_graph"] = 25;
    const auto record = go_fann::query_preparation_timing_json(timing);
    require(record.at("query_prepare_total_ms") == 18,
            "nested manifest details or reusable cardinality counted as query work");
    require(record.at("reusable_static_total_ms") == 75,
            "reusable initialization was not accounted separately");
    require(record.at("other_runtime_load_ms") == 7,
            "runtime remainder failed to reconcile with elapsed wall time");
    require(record.at("manifest_detail_ms").at("other_decode_dispatch") == 2,
            "manifest remainder failed to reconcile");
    const auto empty = go_fann::query_preparation_timing_json({});
    require(empty.at("query_prepare_total_ms") == 0 &&
                empty.at("other_runtime_load_ms") == 0,
            "an unused provider should report zero timing");
}

void test_jsonl_persistence_and_escaping(const std::string& path)
{
    go_fann::QuerySetupReporter reporter(path, "pq");
    reporter.append({{"record_type", "workload_setup"}, {"path", "a\"b\nc"}});
    // Readers can consume completed rows before reporter destruction.
    std::ifstream input(path);
    std::string line;
    require(static_cast<bool>(std::getline(input, line)), "setup row was not flushed");
    const auto record = nlohmann::json::parse(line);
    require(record.at("schema_version") == 1 && record.at("backend") == "pq",
            "report metadata was not preserved");
    require(record.at("path") == "a\"b\nc", "path was not JSON escaped");
    require(!std::getline(input, line), "embedded newline broke the JSONL record");
    bool failed = false;
    try {
        go_fann::QuerySetupReporter invalid(path + "/no/such/file", "rabitq");
    } catch (const std::runtime_error&) {
        failed = true;
    }
    require(failed, "invalid report path did not fail explicitly");
}
} // namespace

int main()
{
    char path[] = "/tmp/query-setup-test-XXXXXX";
    const int fd = mkstemp(path);
    if (fd < 0) return 1;
    close(fd);
    try {
        test_nonoverlapping_accounting();
        test_jsonl_persistence_and_escaping(path);
        std::filesystem::remove(path);
        std::puts("query setup timing tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::filesystem::remove(path);
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
