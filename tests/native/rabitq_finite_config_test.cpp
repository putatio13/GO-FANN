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
    char path[] = "/tmp/flashanns_rabitq_finite_config_XXXXXX";
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

void test_config_precedence()
{
    const auto absent = load_config(R"({})");
    require(!absent.rabitq_query_worker_mode.has_value(),
            "missing RabitQ worker mode must preserve the coupled default");
    require(!absent.rabitq_finite_dispatch_mode.has_value(),
            "missing RabitQ finite dispatch must preserve the batch default");
    require(!absent.rabitq_finite_partial_batch_timeout_us.has_value(),
            "missing RabitQ finite timeout must preserve the runtime default");
    require(!absent.rabitq_graph_transport.has_value(),
            "missing RabitQ graph transport must preserve the mailbox default");
    require(!absent.rabitq_request_mode.has_value(),
            "missing RabitQ request mode must preserve the scan default");
    require(!absent.rabitq_request_batch_queries.has_value(),
            "missing RabitQ request batch size must preserve the runtime default");
    require(!absent.rabitq_request_consumed_memory.has_value(),
            "missing RabitQ request consumed memory must preserve the runtime default");
    require(!absent.rabitq_gpu_graph_row_cache_mib.has_value(),
            "missing RabitQ GPU graph row cache size must preserve the runtime default");
    require(!absent.rabitq_use_int8_score.has_value(),
            "missing RabitQ int8 score flag must preserve the FP32 default");
    require(!absent.host_demand_batch_size.has_value(),
            "missing host demand batch size must preserve the runtime default");
    require(!absent.host_demand_batch_timeout_us.has_value(),
            "missing host demand batch timeout must preserve the runtime default");

    const auto root = load_config(
        R"({"rabitq_query_worker_mode":"finite","rabitq_finite_dispatch_mode":"batch","rabitq_finite_partial_batch_timeout_us":75,"rabitq_graph_transport":"device_resident","rabitq_request_mode":"batch_ring","rabitq_request_batch_queries":128,"rabitq_request_consumed_memory":"device","rabitq_gpu_graph_row_cache_mib":256,"rabitq_use_int8_score":true,"host_demand_batch_size":16,"host_demand_batch_timeout_us":5})");
    require(root.rabitq_query_worker_mode == "finite", "root RabitQ worker mode was not loaded");
    require(root.rabitq_finite_dispatch_mode == "batch", "root RabitQ dispatch mode was not loaded");
    require(root.rabitq_finite_partial_batch_timeout_us == 75,
            "root RabitQ partial batch timeout was not loaded");
    require(root.rabitq_graph_transport == "device_resident",
            "root RabitQ graph transport was not loaded");
    require(root.rabitq_request_mode == "batch_ring",
            "root RabitQ request mode was not loaded");
    require(root.rabitq_request_batch_queries == 128,
            "root RabitQ request batch size was not loaded");
    require(root.rabitq_request_consumed_memory == "device",
            "root RabitQ request consumed memory was not loaded");
    require(root.rabitq_gpu_graph_row_cache_mib == 256,
            "root RabitQ GPU graph row cache size was not loaded");
    require(root.rabitq_use_int8_score == true,
            "root RabitQ int8 score flag was not loaded");
    require(root.host_demand_batch_size == 16,
            "root host demand batch size was not loaded");
    require(root.host_demand_batch_timeout_us == 5,
            "root host demand batch timeout was not loaded");

    const auto nested = load_config(
        R"({"rabitq_query_worker_mode":"coupled","rabitq_finite_dispatch_mode":"persistent","rabitq_finite_partial_batch_timeout_us":50,"rabitq_graph_transport":"mailbox","rabitq_request_mode":"scan","rabitq_request_batch_queries":512,"rabitq_request_consumed_memory":"host","rabitq_gpu_graph_row_cache_mib":64,"rabitq_use_int8_score":false,"rabitq":{"query_worker_mode":"finite","finite_dispatch_mode":"batch","finite_partial_batch_timeout_us":125,"graph_transport":"device_resident","request_mode":"batch_ring","request_batch_queries":64,"request_consumed_memory":"device","gpu_graph_row_cache_mib":1024,"use_int8_score":true}})");
    require(nested.rabitq_query_worker_mode == "finite",
            "rabitq.query_worker_mode must override root compatibility value");
    require(nested.rabitq_finite_dispatch_mode == "batch",
            "rabitq.finite_dispatch_mode must override root compatibility value");
    require(nested.rabitq_finite_partial_batch_timeout_us == 125,
            "rabitq.finite_partial_batch_timeout_us must override root compatibility value");
    require(nested.rabitq_graph_transport == "device_resident",
            "rabitq.graph_transport must override root compatibility value");
    require(nested.rabitq_request_mode == "batch_ring",
            "rabitq.request_mode must override root compatibility value");
    require(nested.rabitq_request_batch_queries == 64,
            "rabitq.request_batch_queries must override root compatibility value");
    require(nested.rabitq_request_consumed_memory == "device",
            "rabitq.request_consumed_memory must override root compatibility value");
    require(nested.rabitq_gpu_graph_row_cache_mib == 1024,
            "rabitq.gpu_graph_row_cache_mib must override root compatibility value");
    require(nested.rabitq_use_int8_score == true,
            "rabitq.use_int8_score must override root compatibility value");
}

}  // namespace

int main()
{
    try {
        test_config_precedence();
        std::puts("RabitQ finite config tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "RabitQ finite config test failed: %s\n", error.what());
        return 1;
    }
}
