#include "data/node_layout.h"

void collect_local_neighbors_from_graph(const uint32_t* graph_rows,
                                        uint64_t graph_degree,
                                        uint32_t start_local_index,
                                        std::vector<uint32_t>& local_index_array,
                                        std::unordered_set<uint32_t>& visited,
                                        int hops,
                                        uint32_t label_size) {
    if (hops == 0 || start_local_index >= label_size || visited.count(start_local_index)) {
        return;
    }

    visited.insert(start_local_index);
    local_index_array.push_back(start_local_index);

    const uint32_t* row_ptr = graph_rows + static_cast<size_t>(start_local_index) * graph_degree;
    for (uint64_t i = 0; i < graph_degree; ++i) {
        const uint32_t next = row_ptr[i];
        if (next == UINT32_MAX || next >= label_size) {
            continue;
        }
        collect_local_neighbors_from_graph(
            graph_rows, graph_degree, next, local_index_array, visited, hops - 1, label_size);
    }
}
