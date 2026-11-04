#pragma once

#include <cstdint>
#include <string>

namespace go_fann::graph_builder {

struct GraphBuildOptions {
    std::string graph_path;
    std::string graph_file_path;
    std::string lowfreq_index_path;
    std::string seed_sidecar_path;
    std::string original_label_seed_sidecar_path;
    std::string metadata_sidecar_path;
    std::string base_vector_path;
    std::string data_label_spmat_path;
    std::string group_path;
    std::string builder = "cagra";
    bool force_rebuild = false;
    bool run_analyze = false;
    uint32_t row_limit = 5000000;
    uint32_t cagra_build_worker_count = 4;
    uint32_t stream_worker_count = 0;
    bool stream_auto_workers = false;
    uint64_t stream_memory_budget_bytes = 0;
    uint64_t stream_per_worker_memory_cap_bytes = 0;
    bool stream_allow_concurrent_cagra = false;
    std::string cagra_ace_build_dir;
    bool cagra_ace_force_disk = false;
    int specificity_threshold = 2000;
    int dim = 128;
    int degree = 64;
};

int run_graph_builder(const GraphBuildOptions& options);

}
