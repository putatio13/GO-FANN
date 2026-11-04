#pragma once

// Included inside the backend's private namespace after DeviceBuffer.
// A context is one graph group. Its unique row-key interval identifies graph
// occurrences; sample IDs identify distances and output deduplication only.
struct FullGpuOverlapContext {
    const uint32_t* graph = nullptr;
    const uint32_t* samples = nullptr;
    const uint32_t* inverse_samples = nullptr;
    const uint32_t* inverse_locals = nullptr;
    uint32_t size = 0;
    uint32_t degree = 0;
    uint32_t key_begin = 0;
    uint32_t contiguous_sample_begin = UINT32_MAX;
    uint32_t bridge = 0;
};

struct FullGpuOverlapQuery {
    uint32_t context_begin = 0;
    uint32_t context_end = 0;
    uint32_t route_begin = 0;
    uint32_t route_end = 0;
    uint32_t output_route = 0;
    uint32_t original_query = 0;
    uint8_t range_mode = 0;
    float lower = 0;
    float upper = 0;
};

struct FullGpuOverlapCounters {
    uint64_t expansions = 0;
    uint64_t bridge_expansions = 0;
    uint64_t transitions = 0;
    uint64_t distance_evaluations = 0;
    uint64_t distance_reuses = 0;
    uint64_t retries = 0;
};

struct FullGpuOverlapDevice {
    DeviceBuffer<uint32_t> graph;
    DeviceBuffer<uint32_t> samples;
    DeviceBuffer<uint32_t> inverse_samples;
    DeviceBuffer<uint32_t> inverse_locals;
    DeviceBuffer<FullGpuOverlapContext> contexts;
    DeviceBuffer<uint32_t> query_contexts;
    DeviceBuffer<uint32_t> seeds;
    DeviceBuffer<FullGpuOverlapQuery> queries;
    DeviceBuffer<uint32_t> route_ids;
    DeviceBuffer<uint32_t> route_positions;
    DeviceBuffer<uint32_t> initial_fallback_ids;
    DeviceBuffer<uint32_t> fallback_ids;
    DeviceBuffer<FullGpuOverlapCounters> counters;
    uint64_t selected_bridges = 0;
};
