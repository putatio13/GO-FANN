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

go_fann::AppConfigValues load_config(const std::string& contents)
{
    char path[] = "/tmp/flashanns_pq_query_worker_mode_XXXXXX";
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

void test_defaults_and_validation()
{
    using namespace go_fann::defaults;
    require(default_pq_query_worker_mode == "coupled",
            "PQ query worker mode must preserve the coupled default");
    require(is_valid_pq_query_worker_mode("coupled"), "coupled mode must be valid");
    require(is_valid_pq_query_worker_mode("finite"), "finite mode must be valid");
    require(!is_valid_pq_query_worker_mode("persistent"),
            "unknown query worker mode must be rejected");
    require(!pq_query_worker_mode_is_finite("coupled"),
            "coupled mode must not enable finite state");
    require(pq_query_worker_mode_is_finite("finite"),
            "finite mode must enable finite state");
    require(default_pq_finite_token_transport == "device_copy",
            "finite token transport must preserve device-copy behavior by default");
    require(is_valid_pq_finite_token_transport("device_copy"),
            "device_copy finite token transport must be valid");
    require(is_valid_pq_finite_token_transport("mapped"),
            "mapped finite token transport must be valid");
    require(!is_valid_pq_finite_token_transport("managed"),
            "unknown finite token transport must be rejected");
    require(default_pq_finite_dispatch_mode == "batch",
            "finite dispatch must preserve batch behavior by default");
    require(is_valid_pq_finite_dispatch_mode("batch"),
            "batch finite dispatch mode must be valid");
    require(is_valid_pq_finite_dispatch_mode("persistent"),
            "persistent finite dispatch mode must be valid");
    require(is_valid_pq_finite_dispatch_mode("ready_ring"),
            "ready-ring finite dispatch mode must be valid");
    require(!is_valid_pq_finite_dispatch_mode("ring"),
            "unknown finite dispatch mode must be rejected");
    require(!pq_finite_dispatch_mode_is_persistent("batch"),
            "batch dispatch must not launch persistent workers");
    require(pq_finite_dispatch_mode_is_persistent("persistent"),
            "persistent dispatch must launch persistent workers");
    require(pq_finite_dispatch_mode_is_ready_ring("ready_ring"),
            "ready-ring dispatch must launch direct ring workers");
    require(!pq_finite_dispatch_mode_is_ready_ring("persistent"),
            "fixed-slot persistent dispatch must not select the direct ready ring");
    require(!pq_finite_dispatch_mode_requires_completion_path("batch"),
            "batch dispatch must retain mailbox-scan fallback support");
    require(pq_finite_dispatch_mode_requires_completion_path("persistent"),
            "persistent dispatch must require completion publication");
    require(pq_finite_dispatch_mode_requires_completion_path("ready_ring"),
            "ready-ring dispatch must require completion publication");
    require(default_pq_finite_partial_batch_timeout_us == 50,
            "finite partial batch timeout must preserve the 50us default");
    require(is_valid_pq_finite_partial_batch_timeout_us(1),
            "minimum finite partial batch timeout must be valid");
    require(is_valid_pq_finite_partial_batch_timeout_us(1000000),
            "maximum finite partial batch timeout must be valid");
    require(!is_valid_pq_finite_partial_batch_timeout_us(0),
            "zero finite partial batch timeout must be rejected");
    require(!is_valid_pq_finite_partial_batch_timeout_us(1000001),
            "finite partial batch timeout above the maximum must be rejected");
}

void test_config_precedence()
{
    const auto absent = load_config(R"({})");
    require(!absent.pq_graph_transport.has_value(),
            "missing graph transport must preserve the runtime default");
    require(!absent.pq_query_worker_mode.has_value(),
            "missing query worker mode must preserve the runtime default");
    require(!absent.pq_request_mode.has_value(),
            "missing request mode must preserve the runtime default");
    require(!absent.pq_request_batch_queries.has_value(),
            "missing request batch size must preserve the runtime default");
    require(!absent.pq_request_consumed_memory.has_value(),
            "missing request consumed memory must preserve the runtime default");
    require(!absent.pq_finite_token_transport.has_value(),
            "missing finite token transport must preserve the runtime default");
    require(!absent.pq_finite_dispatch_mode.has_value(),
            "missing finite dispatch mode must preserve the runtime default");
    require(!absent.pq_finite_partial_batch_timeout_us.has_value(),
            "missing finite partial batch timeout must preserve the runtime default");
    require(!absent.host_reply_wait_policy.has_value(),
            "missing host reply wait policy must preserve the runtime default");

    const auto root_wait_policy =
        load_config(R"({"host_reply_wait_policy":"adaptive"})");
    require(root_wait_policy.host_reply_wait_policy == "adaptive",
            "root host reply wait policy was not loaded");

    const auto root = load_config(R"({"pq_query_worker_mode":"finite"})");
    require(root.pq_query_worker_mode == "finite",
            "root compatibility query worker mode was not loaded");

    const auto root_graph_transport =
        load_config(R"({"pq_graph_transport":"device_resident"})");
    require(root_graph_transport.pq_graph_transport == "device_resident",
            "root compatibility graph transport was not loaded");

    const auto root_request = load_config(
        R"({"pq_request_mode":"batch_ring","pq_request_batch_queries":64,"pq_request_consumed_memory":"host"})");
    require(root_request.pq_request_mode == "batch_ring",
            "root compatibility request mode was not loaded");
    require(root_request.pq_request_batch_queries == 64,
            "root compatibility request batch size was not loaded");
    require(root_request.pq_request_consumed_memory == "host",
            "root compatibility request consumed memory was not loaded");

    const auto root_transport =
        load_config(R"({"pq_finite_token_transport":"mapped"})");
    require(root_transport.pq_finite_token_transport == "mapped",
            "root compatibility finite token transport was not loaded");

    const auto root_dispatch =
        load_config(R"({"pq_finite_dispatch_mode":"persistent"})");
    require(root_dispatch.pq_finite_dispatch_mode == "persistent",
            "root compatibility finite dispatch mode was not loaded");

    const auto root_timeout =
        load_config(R"({"pq_finite_partial_batch_timeout_us":75})");
    require(root_timeout.pq_finite_partial_batch_timeout_us == 75,
            "root compatibility finite partial batch timeout was not loaded");

    const auto nested = load_config(
        R"({"pq_query_worker_mode":"coupled","pq":{"query_worker_mode":"finite"}})");
    require(nested.pq_query_worker_mode == "finite",
            "pq.query_worker_mode must override the root compatibility value");

    const auto nested_graph_transport = load_config(
        R"({"pq_graph_transport":"mailbox","pq":{"graph_transport":"device_resident"}})");
    require(nested_graph_transport.pq_graph_transport == "device_resident",
            "pq.graph_transport must override the root compatibility value");

    const auto nested_request = load_config(
        R"({"pq_request_mode":"scan","pq_request_batch_queries":64,"pq_request_consumed_memory":"device","pq":{"request_mode":"batch_ring","request_batch_queries":256,"request_consumed_memory":"host"}})");
    require(nested_request.pq_request_mode == "batch_ring",
            "pq.request_mode must override the root compatibility value");
    require(nested_request.pq_request_batch_queries == 256,
            "pq.request_batch_queries must override the root compatibility value");
    require(nested_request.pq_request_consumed_memory == "host",
            "pq.request_consumed_memory must override the root compatibility value");

    const auto nested_transport = load_config(
        R"({"pq_finite_token_transport":"device_copy","pq":{"finite_token_transport":"mapped"}})");
    require(nested_transport.pq_finite_token_transport == "mapped",
            "pq.finite_token_transport must override the root compatibility value");

    const auto nested_dispatch = load_config(
        R"({"pq_finite_dispatch_mode":"batch","pq":{"finite_dispatch_mode":"persistent"}})");
    require(nested_dispatch.pq_finite_dispatch_mode == "persistent",
            "pq.finite_dispatch_mode must override the root compatibility value");

    const auto nested_timeout = load_config(
        R"({"pq_finite_partial_batch_timeout_us":75,"pq":{"finite_partial_batch_timeout_us":125}})");
    require(nested_timeout.pq_finite_partial_batch_timeout_us == 125,
            "pq.finite_partial_batch_timeout_us must override the root compatibility value");

    const auto nested_wait_policy = load_config(
        R"({"host_reply_wait_policy":"fixed","pq":{"host_reply_wait_policy":"adaptive"}})");
    require(nested_wait_policy.host_reply_wait_policy == "adaptive",
            "pq.host_reply_wait_policy must override the root value");
}

void test_posting_streaming_config()
{
    require(!load_config("{}").pq_posting_streaming, "streaming absent must retain API default");
    auto config = load_config(R"({"pq_posting_streaming":false,"pq_posting_stream_batch_size":4,"pq":{"posting_streaming":true,"posting_stream_batch_size":32}})");
    require(config.pq_posting_streaming == true, "nested streaming must override root");
    require(config.pq_posting_stream_batch_size == 32, "nested batch size must override root");
    for (const char* value : {"0", "-1", "1048577", "4294967297", "1.5", "true", "null", "\"32\""}) {
        for (bool nested : {false, true}) {
            const std::string text = nested ? std::string("{\"pq\":{\"posting_stream_batch_size\":") + value + "}}"
                                            : std::string("{\"pq_posting_stream_batch_size\":") + value + "}";
            bool rejected = false;
            try { load_config(text); } catch (const std::runtime_error&) { rejected = true; }
            require(rejected, "invalid stream batch size must not silently narrow");
        }
    }
}

}  // namespace

int main()
{
    try {
        require(!load_config("{}").pq_cpu_persistent_workers.has_value(),
            "omitted persistent worker flag must preserve backend default");
    require(load_config(R"({"pq_cpu_persistent_workers":true})").pq_cpu_persistent_workers == true,
            "root persistent worker flag must load");
    require(load_config(R"({"pq_cpu_persistent_workers":true,"pq":{"cpu_persistent_workers":false}})")
                .pq_cpu_persistent_workers == false,
            "nested persistent worker flag must override root");
    require(load_config(R"({"pq":{"cpu_persistent_workers":true}})").pq_cpu_persistent_workers == true,
            "nested persistent worker flag must load");
    require(!load_config("{}").rerank_distance_reuse.has_value(), "omitted reuse should retain disabled default");
    require(load_config(R"({"rerank_distance_reuse":false})").rerank_distance_reuse == false, "explicit cache opt-out must load");
    require(load_config(R"({"rerank_distance_reuse":true})").rerank_distance_reuse == true, "explicit cache enable must load");
    test_defaults_and_validation();
        test_config_precedence();
        test_posting_streaming_config();
        std::puts("PQ query worker mode config tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "PQ query worker mode config test failed: %s\n", error.what());
        return 1;
    }
}
