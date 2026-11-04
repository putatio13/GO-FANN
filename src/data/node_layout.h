#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_set>
#include <vector>

constexpr size_t kNodeDataWords = 48;
constexpr size_t kNodeNeighborCapacity = 64;

struct Node {
    uint32_t index;
    uint32_t data[kNodeDataWords];
    uint32_t sample_id;
    uint32_t neighbor[kNodeNeighborCapacity];
    uint32_t neighbor_sample[kNodeNeighborCapacity];
    uint32_t neighbor_count;
};

void collect_local_neighbors_from_graph(const uint32_t* graph_rows,
                                        uint64_t graph_degree,
                                        uint32_t start_local_index,
                                        std::vector<uint32_t>& local_index_array,
                                        std::unordered_set<uint32_t>& visited,
                                        int hops,
                                        uint32_t label_size);
