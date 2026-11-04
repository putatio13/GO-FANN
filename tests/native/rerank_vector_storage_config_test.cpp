#include "app/config_loader.h"
#include "app/runtime_defaults.h"

#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::string write_config(const std::string& contents)
{
    char path[] = "/tmp/flashanns_rerank_storage_config_XXXXXX";
    const int fd = mkstemp(path);
    require(fd >= 0, "mkstemp failed");
    close(fd);
    std::ofstream output(path);
    require(output.is_open(), "failed to open temporary config");
    output << contents;
    output.close();
    return path;
}

void test_mode_helpers()
{
    using namespace go_fann::defaults;
    require(is_valid_rerank_vector_storage_mode("default"),
            "default rerank storage mode was rejected");
    require(is_valid_rerank_vector_storage_mode("mmap_fp16"),
            "mmap_fp16 rerank storage mode was rejected");
    require(is_valid_rerank_vector_storage_mode("memory_int8"),
            "memory_int8 rerank storage mode was rejected");
    require(!is_valid_rerank_vector_storage_mode("page_cache"),
            "unknown rerank storage mode was accepted");
    require(rerank_vector_storage_is_default("default") &&
                rerank_vector_storage_is_mmap_fp16("mmap_fp16") &&
                rerank_vector_storage_is_memory_int8("memory_int8"),
            "rerank storage mode classification is inconsistent");
}

void test_nested_config_precedence()
{
    const std::string path = write_config(R"({
        "rerank_vector_storage_mode": "default",
        "rerank_vector_artifact_path": "/root.bin",
        "runtime": {
            "rerank_vector_storage_mode": "memory_int8",
            "rerank_vector_artifact_path": "/runtime.bin"
        }
    })");
    const go_fann::AppConfigValues config = go_fann::load_config_file(path);
    require(config.rerank_vector_storage_mode.has_value() &&
                *config.rerank_vector_storage_mode == "memory_int8",
            "runtime rerank storage mode did not win nested precedence");
    require(config.rerank_vector_artifact_path.has_value() &&
                *config.rerank_vector_artifact_path == "/runtime.bin",
            "runtime rerank artifact path did not win nested precedence");
    std::remove(path.c_str());
}

void expect_rejected(const std::string& contents, const std::string& diagnostic)
{
    const std::string path = write_config(contents);
    bool rejected = false;
    try {
        (void)go_fann::load_config_file(path);
    } catch (const std::runtime_error& error) {
        rejected = std::string(error.what()).find(diagnostic) != std::string::npos;
    }
    std::remove(path.c_str());
    require(rejected, "retired storage configuration was ignored or gave the wrong diagnostic");
}

void test_retired_storage_rejection()
{
    // These fail during config parsing, without loading any graph/vector input.
    expect_rejected(R"({"base_vector_path":"/missing/base.fbin","base_vectors_on_disk":true})", "base_vectors_on_disk");
    expect_rejected(R"({"runtime":{"enable_vector_cache":false}})", "runtime.enable_vector_cache");
    expect_rejected(R"({"runtime":{"vector_cache_mode":null}})", "runtime.vector_cache_mode");
    expect_rejected(R"({"disk_io":{"io_mode":"async"}})", "disk_io");
    expect_rejected(R"({"runtime":{"block_cache_enabled":false}})", "block_cache_enabled");
    expect_rejected(R"({"benchmark":{"benchmark_cache_warmup":false}})", "benchmark_cache_warmup");
    expect_rejected(R"({"runtime":{"graph_payload_mode":"disk"}})", "only memory");
    expect_rejected(R"({"graph_storage_mode":"disk","runtime":{"graph_storage_mode":"memory"}})", "only memory");
    expect_rejected(R"({"range_graph_storage_mode":"disk"})", "only memory");
    expect_rejected(R"({"graph_storage_mode":null})", "only memory");
    expect_rejected(R"({"vecflow":{"graph_storage_mode":"memory"}})", "vecflow");
    for (const std::string arg : {"--base-vectors-on-disk=1", "--base_vectors_on_disk=0",
                                  "--enable-row-cache", "--no-vector-use-direct-io", "--disable-vector-cache",
                                  "--avoid-direct-io-for-block-cache",
                                  "--graph-storage-mode=disk", "--range_graph_storage_mode=disk",
                                  "--benchmark-cache-warmup=0", "--graph-disk-path=/missing"}) {
        bool rejected = false;
        try { go_fann::reject_obsolete_cli_option(arg); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "retired CLI storage option was accepted");
    }
    for (const std::string arg : {"--graph-storage-mode=memory", "--warmup-runs=2",
                                  "--rerank-vector-storage-mode=mmap_fp16", "--graph-file-path=graph.bin"})
        go_fann::reject_obsolete_cli_option(arg);
    const std::string path = write_config(R"({"graph_storage_mode":"memory","range_graph_storage_mode":"memory",
        "graph_file_path":"graph.serialized.bin","rerank_vector_storage_mode":"mmap_fp16","warmup_runs":2,"timed_runs":3})");
    const auto config = go_fann::load_config_file(path);
    std::remove(path.c_str());
    require(config.graph_storage_mode == "memory" && config.range_graph_storage_mode == "memory" &&
            config.graph_file_path == "graph.serialized.bin" && config.rerank_vector_storage_mode == "mmap_fp16" &&
            config.warmup_runs == 2 && config.timed_runs == 3, "supported resident/mmap configuration changed");
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        test_mode_helpers();
        test_nested_config_precedence();
        test_retired_storage_rejection();
        for (int i = 1; i < argc; ++i) {
            const auto config = go_fann::load_config_file(argv[i]);
            require(config.graph_storage_mode == "memory", "paper template must explicitly select memory graph storage");
        }
        std::puts("rerank vector storage config tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "rerank vector storage config test failed: %s\n", error.what());
        return 1;
    }
}
