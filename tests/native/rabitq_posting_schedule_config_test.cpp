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

go_fann::AppConfigValues load(const std::string& contents)
{
    char path[] = "/tmp/flashanns_posting_schedule_XXXXXX";
    const int fd = mkstemp(path);
    require(fd >= 0, "mkstemp failed");
    close(fd);
    try {
        std::ofstream output(path);
        output.exceptions(std::ios::badbit | std::ios::failbit);
        output << contents;
        output.close();
        auto config = go_fann::load_config_file(path);
        std::remove(path);
        return config;
    } catch (...) {
        std::remove(path);
        throw;
    }
}

}  // namespace

int main()
{
    try {
        require(!load("{}").rabitq_posting_schedule.has_value(),
                "absent policy must preserve runtime default");
        for (const std::string policy : {"legacy", "graph_first", "posting_first"}) {
            require(load("{\"rabitq_posting_schedule\":\"" + policy + "\"}")
                        .rabitq_posting_schedule == policy,
                    "root schedule was not loaded");
            require(load("{\"rabitq_posting_schedule\":\"legacy\",\"rabitq\":{"
                         "\"posting_schedule\":\"" + policy + "\"}}")
                        .rabitq_posting_schedule == policy,
                    "nested schedule must override root");
        }
        require(!load("{}").rabitq_range_posting_segment_rows.has_value(),
                "absent segmentation must preserve default");
        require(load(R"({"rabitq_range_posting_segment_rows":1024})")
                    .rabitq_range_posting_segment_rows == 1024,
                "root segment size was not loaded");
        require(load(R"({"rabitq_range_posting_segment_rows":1024,"rabitq":{"range_posting_segment_rows":4096}})")
                    .rabitq_range_posting_segment_rows == 4096,
                "nested segment size must override root");
        require(!load("{}").rabitq_posting_streaming.has_value(), "absent stream flag must preserve default");
        require(load(R"({"rabitq_posting_streaming":true,"rabitq_posting_stream_batch_size":37})")
                    .rabitq_posting_streaming == true, "root streaming flag missing");
        auto stream = load(R"({"rabitq_posting_streaming":false,"rabitq_posting_stream_batch_size":37,"rabitq":{"posting_streaming":true,"posting_stream_batch_size":2048}})");
        require(stream.rabitq_posting_streaming == true && stream.rabitq_posting_stream_batch_size == 2048,
                "nested stream controls must override root");
        require(!load("{}").rabitq_posting_persistent_workers.has_value(),
                "absent persistent workers must preserve default");
        require(load(R"({"rabitq_posting_persistent_workers":true})")
                    .rabitq_posting_persistent_workers == true, "root persistent flag missing");
        require(load(R"({"rabitq_posting_persistent_workers":true,"rabitq":{"posting_persistent_workers":false}})")
                    .rabitq_posting_persistent_workers == false, "nested persistent flag precedence");
        require(!load("{}").rabitq_graph_persistent_workers.has_value(),
                "absent graph persistent workers must preserve default");
        require(load(R"({"rabitq_graph_persistent_workers":true})")
                    .rabitq_graph_persistent_workers == true, "root graph persistent flag missing");
        auto pools = load(R"({"rabitq_graph_persistent_workers":true,"rabitq_posting_persistent_workers":false,"rabitq":{"graph_persistent_workers":false,"posting_persistent_workers":true}})");
        require(pools.rabitq_graph_persistent_workers == false &&
                    pools.rabitq_posting_persistent_workers == true,
                "graph/posting controls must be independent with nested precedence");
        std::puts("RabitQ posting schedule config tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "RabitQ posting schedule config test failed: %s\n", error.what());
        return 1;
    }
}
