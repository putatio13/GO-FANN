#include "app/config_loader.h"

#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

go_fann::AppConfigValues load_config(const std::string& contents)
{
    char path[] = "/tmp/flashanns_range_posting_config_XXXXXX";
    const int fd = mkstemp(path);
    require(fd >= 0, "mkstemp failed");
    close(fd);
    std::ofstream output(path);
    require(output.is_open(), "failed to open temporary config");
    output << contents;
    output.close();
    try {
        auto config = go_fann::load_config_file(path);
        std::remove(path);
        return config;
    } catch (...) {
        std::remove(path);
        throw;
    }
}

void test_range_posting_config()
{
    const auto absent = load_config(R"({})");
    require(!absent.range_posting_index_path.has_value(), "missing path must stay unset");
    require(!absent.range_posting_mode.has_value(), "missing mode must stay unset");
    require(!absent.range_posting_scan_threshold.has_value(), "missing threshold must stay unset");
    require(!absent.range_posting_shortlist.has_value(), "missing shortlist must stay unset");
    require(!absent.range_posting_postfilter_shortlist_multiplier.has_value(),
            "missing postfilter multiplier must stay unset");
    require(!absent.range_posting_and_filter_mode.has_value(),
            "missing AND filter mode must stay unset");
    require(!absent.range_label_and_policy.has_value(),
            "missing range-label AND policy must stay unset");
    require(!absent.range_posting_drop_wide_hybrid_or.has_value(),
            "missing wide hybrid OR policy must stay unset");
    require(!absent.range_graph_hybrid_or_enabled.has_value(),
            "missing range graph policy must stay unset");
    require(!absent.range_graph_hybrid_and_enabled.has_value(),
            "missing range graph AND policy must stay unset");

    const auto root = load_config(
        R"({"range_posting_index_path":"root.bin","range_posting_mode":"select","range_posting_scan_threshold":1234,"range_posting_shortlist":200,"range_posting_postfilter_shortlist_multiplier":7,"range_posting_and_filter_mode":"post","range_label_and_policy":"range_prefer","range_posting_drop_wide_hybrid_or":true,"range_graph_hybrid_or_enabled":true,"range_graph_hybrid_and_enabled":true,"range_graph_path":"range.bin","range_graph_storage_mode":"memory","range_graph_group_path":"range.groups"})");
    require(root.range_posting_index_path == "root.bin", "root path was not loaded");
    require(root.range_posting_mode == "select", "root mode was not loaded");
    require(root.range_posting_scan_threshold == 1234, "root threshold was not loaded");
    require(root.range_posting_shortlist == 200, "root shortlist was not loaded");
    require(root.range_posting_postfilter_shortlist_multiplier == 7,
            "root postfilter multiplier was not loaded");
    require(root.range_posting_and_filter_mode == "post",
            "root AND filter mode was not loaded");
    require(root.range_label_and_policy == "range_prefer",
            "root range-label AND policy was not loaded");
    require(root.range_posting_drop_wide_hybrid_or.value_or(false),
            "root wide hybrid OR policy was not loaded");
    require(root.range_graph_hybrid_or_enabled.value_or(false),
            "root range graph policy was not loaded");
    require(root.range_graph_hybrid_and_enabled.value_or(false),
            "root range graph AND policy was not loaded");
    require(root.range_graph_path == "range.bin", "root range graph path was not loaded");
    require(root.range_graph_storage_mode == "memory",
            "root range graph storage mode was not loaded");
    require(root.range_graph_group_path == "range.groups",
            "root range graph group path was not loaded");

    const auto nested = load_config(
        R"({"range_posting_index_path":"root.bin","range_posting_mode":"select","range_posting_scan_threshold":1234,"range_posting_shortlist":200,"range_posting_postfilter_shortlist_multiplier":5,"range_posting_and_filter_mode":"label","range_label_and_policy":"label_prefer","range_posting_drop_wide_hybrid_or":false,"range_graph_hybrid_or_enabled":false,"range_graph_hybrid_and_enabled":false,"range_graph_path":"root-range.bin","runtime":{"range_posting_index_path":"nested.bin","range_posting_mode":"hybrid","range_posting_scan_threshold":5678,"range_posting_shortlist":400,"range_posting_postfilter_shortlist_multiplier":10,"range_posting_and_filter_mode":"pre","range_label_and_policy":"dual","range_posting_drop_wide_hybrid_or":true,"range_graph_hybrid_or_enabled":true,"range_graph_hybrid_and_enabled":true,"range_graph_path":"nested-range.bin"}})");
    require(nested.range_posting_index_path == "nested.bin", "runtime path must override root");
    require(nested.range_posting_mode == "hybrid", "runtime mode must override root");
    require(nested.range_posting_scan_threshold == 5678,
            "runtime threshold must override root");
    require(nested.range_posting_shortlist == 400, "runtime shortlist must override root");
    require(nested.range_posting_postfilter_shortlist_multiplier == 10,
            "runtime postfilter multiplier must override root");
    require(nested.range_posting_and_filter_mode == "pre",
            "runtime AND filter mode must override root");
    require(nested.range_label_and_policy == "dual",
            "runtime range-label AND policy must override root");
    require(nested.range_posting_drop_wide_hybrid_or.value_or(false),
            "runtime wide hybrid OR policy must override root");
    require(nested.range_graph_hybrid_or_enabled.value_or(false),
            "runtime range graph policy must override root");
    require(nested.range_graph_hybrid_and_enabled.value_or(false),
            "runtime range graph AND policy must override root");
    require(nested.range_graph_path == "nested-range.bin",
            "runtime range graph path must override root");
}

}  // namespace

int main()
{
    try {
        test_range_posting_config();
        std::puts("Range posting config tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Range posting config test failed: %s\n", error.what());
        return 1;
    }
}
