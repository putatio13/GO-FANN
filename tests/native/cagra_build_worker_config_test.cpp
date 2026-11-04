#include "app/config_loader.h"
#include "index/graph_builder.h"

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
    char path[] = "/tmp/flashanns_cagra_worker_config_XXXXXX";
    const int fd = mkstemp(path);
    require(fd >= 0, "mkstemp failed");
    close(fd);
    std::ofstream(path) << contents;
    try {
        auto config = go_fann::load_config_file(path);
        std::remove(path);
        return config;
    } catch (...) {
        std::remove(path);
        throw;
    }
}
}

int main()
{
    require(go_fann::graph_builder::GraphBuildOptions{}.cagra_build_worker_count == 4,
            "absent native worker setting must retain four workers");
    require(!load("{}").cagra_build_worker_count, "absent config must stay unset");
    require(load(R"({"cagra_build_worker_count":8})").cagra_build_worker_count == 8,
            "root worker count not parsed");
    require(load(R"({"graph_builder":{"builder":"cagra","cagra_build_worker_count":4}})")
                .cagra_build_worker_count == 4,
            "nested worker count not parsed");
    require(load(R"({"graph_builder":{"cagra_build_worker_count":2},"cagra_build_worker_count":8})")
                .cagra_build_worker_count == 8,
            "root override must take precedence");
    std::puts("CAGRA build worker config tests passed");
}
