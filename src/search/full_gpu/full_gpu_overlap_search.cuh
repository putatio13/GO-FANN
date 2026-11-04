#pragma once

// Included after the backend's distance, filter, queue, and runtime helpers.
__device__ __forceinline__ uint32_t overlap_local_for_sample(
    const FullGpuOverlapContext& context, uint32_t sample)
{
    if (context.contiguous_sample_begin != UINT32_MAX) {
        const uint32_t local = sample - context.contiguous_sample_begin;
        return sample >= context.contiguous_sample_begin && local < context.size
            ? local : UINT32_MAX;
    }
    uint32_t lo = 0, hi = context.size;
    while (lo < hi) {
        const uint32_t mid = lo + (hi - lo) / 2;
        if (context.inverse_samples[mid] < sample) lo = mid + 1;
        else hi = mid;
    }
    return lo < context.size && context.inverse_samples[lo] == sample
        ? context.inverse_locals[lo] : UINT32_MAX;
}

// Exact graph occurrence keys. Saturation requests a route-local retry; it
// never silently drops expansion state or substitutes a sample fingerprint.
__device__ int overlap_visit(uint32_t* visited, uint32_t key)
{
    uint32_t slot = visited_hash_code(key);
    for (int probe = 0; probe < kHashTableCapacity; ++probe) {
        if (visited[slot] == key) return 0;
        if (visited[slot] == UINT32_MAX) {
            visited[slot] = key;
            return 1;
        }
        slot = (slot + 1) % kHashTableCapacity;
    }
    return -1;
}

__device__ void overlap_insert(sorted_frontier::Entry* queue, int& count,
                               int limit, sorted_frontier::Entry entry)
{
    const int pos = sorted_frontier::lower_bound(queue, count, entry);
    if (pos >= limit) return;
    const int end = min(count, limit - 1);
    for (int i = end; i > pos; --i) queue[i] = queue[i - 1];
    queue[pos] = entry;
    count = min(count + 1, limit);
}

template <int kQueueSize, typename T>
__global__ void full_gpu_overlap_kernel(
    const T* base, const T* queries, uint32_t dim,
    const float* ranks, const FullGpuOverlapContext* contexts,
    const uint32_t* query_contexts, const uint32_t* seeds,
    const FullGpuOverlapQuery* query_records,
    const uint32_t* route_ids, const uint32_t* route_positions,
    uint32_t* fallback_ids, uint32_t topk, uint32_t finish_count,
    int early_stop_mode, uint32_t max_steps,
    idx_t* results, float* distances, int* route_steps,
    FullGpuOverlapCounters* counters)
{
    using Entry = sorted_frontier::Entry;
    const auto query = query_records[blockIdx.x];
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x / 32u;
    extern __shared__ __align__(16) unsigned char query_storage[];
    T* query_row = reinterpret_cast<T*>(query_storage);
    __shared__ uint32_t norm;
    __shared__ uint32_t visited[kHashTableCapacity];
    __shared__ FullGpuSampleDistanceCacheEntry cache[512];
    __shared__ Entry queue[kQueueSize];
    __shared__ Entry merged[kQueueSize];
    __shared__ Entry batch[kMaxGraphDegree];
    __shared__ uint32_t batch_samples[kMaxGraphDegree];
    __shared__ uint32_t batch_cached[kMaxGraphDegree];
    __shared__ float best_dist[kMaxTopk];
    __shared__ uint32_t best_ids[kMaxTopk];
    __shared__ uint32_t best_size;
    __shared__ int queue_size, batch_size;
    __shared__ uint32_t failed, finished;
    __shared__ Entry current;
    __shared__ uint32_t current_context, current_local;
    __shared__ FullGpuOverlapCounters stats;
    uint32_t bad_count = 0, pass_count = 0;
    load_query_row_to_shared(queries, query.original_query, dim, query_row);
    for (uint32_t i = threadIdx.x; i < kHashTableCapacity; i += blockDim.x) visited[i] = UINT32_MAX;
    for (uint32_t i = threadIdx.x; i < 512; i += blockDim.x) cache[i].sample_id = UINT32_MAX;
    if (threadIdx.x == 0) {
        queue_size = 0; best_size = 0; failed = 0; finished = 0;
        stats = FullGpuOverlapCounters{};
    }
    __syncthreads();
    if (warp == 0) {
        const uint32_t value = query_squared_norm_device(query_row, dim);
        if (lane == 0) norm = value;
    }
    __syncthreads();

    // Every selected child retains its ordinary seed, and every selected
    // bridge has a deterministic seed. There is no fixed route-count pack.
    for (uint32_t pos = query.context_begin; pos < query.context_end; ++pos) {
        const auto context = contexts[query_contexts[pos]];
        const uint32_t local = seeds[pos];
        if (warp == 0) {
            const uint32_t sample = context.samples[local];
            const float distance = l2_distance_device(base, query_row, dim, sample, norm);
            if (lane == 0) {
                ++stats.distance_evaluations;
                const uint32_t key = context.key_begin + local;
                const int admission = overlap_visit(visited, key);
                if (admission < 0) failed = 1;
                if (admission > 0) overlap_insert(queue, queue_size, kQueueSize, Entry{distance, key});
                cache[(sample * 2654435761u) & 511u] = {sample, distance};
            }
        }
        __syncthreads();
    }

    while (true) {
        if (threadIdx.x == 0) {
            if (failed || (max_steps && stats.expansions >= max_steps) ||
                !pickup_next_unused_sorted<kQueueSize>(queue, queue_size, current)) {
                finished = 1;
            } else {
                const uint32_t key = sorted_frontier::key(current);
                current_context = UINT32_MAX;
                for (uint32_t p = query.context_begin; p < query.context_end; ++p) {
                    const uint32_t id = query_contexts[p];
                    const auto c = contexts[id];
                    if (key >= c.key_begin && key - c.key_begin < c.size) {
                        current_context = id;
                        current_local = key - c.key_begin;
                        break;
                    }
                }
                if (current_context == UINT32_MAX) { failed = 1; finished = 1; }
            }
        }
        __syncthreads();
        if (finished) break;
        const auto context = contexts[current_context];
        if (threadIdx.x == 0) {
            const uint32_t sample = context.samples[current_local];
            const bool passes = range_filter::range_filter_matches_single(
                query.range_mode, query.lower, query.upper, ranks[sample]);
            ++stats.expansions;
            stats.bridge_expansions += context.bridge;
            const bool improved = passes && insert_best_result(
                current.distance, sample, topk, best_dist, best_ids, best_size);
            pass_count += passes;
            if (early_stop_mode == kSearchEarlyStopAdaptive && improved) bad_count = 0;
            else if (best_size == topk && !better_result(current.distance, sample,
                     best_dist[topk - 1], best_ids[topk - 1])) ++bad_count;
            else bad_count = 0;

            // A discovered sample exposes its other graph occurrences. The
            // new occurrence competes for a later expansion; its adjacency
            // is not expanded eagerly with the current graph's row.
            for (uint32_t pos = query.context_begin; pos < query.context_end; ++pos) {
                const uint32_t id = query_contexts[pos];
                if (id == current_context) continue;
                const auto target = contexts[id];
                const uint32_t local = overlap_local_for_sample(target, sample);
                if (local == UINT32_MAX) continue;
                const Entry entry{current.distance, target.key_begin + local};
                if (queue_size == kQueueSize && sorted_frontier::less(queue[queue_size - 1], entry)) continue;
                const int admission = overlap_visit(visited, entry.payload);
                if (admission < 0) { failed = 1; break; }
                if (admission > 0) {
                    overlap_insert(queue, queue_size, kQueueSize, entry);
                    ++stats.transitions;
                    ++stats.distance_reuses;
                }
            }
        }
        __syncthreads();

        for (uint32_t edge = warp; edge < context.degree; edge += 4) {
            uint32_t local = UINT32_MAX, sample = UINT32_MAX, cached = 0;
            float distance = FLT_MAX;
            if (lane == 0) {
                local = context.graph[static_cast<size_t>(current_local) * context.degree + edge];
                if (local < context.size && !visited_hash_test(visited, context.key_begin + local)) {
                    sample = context.samples[local];
                    const auto entry = cache[(sample * 2654435761u) & 511u];
                    if (entry.sample_id == sample) { distance = entry.distance; cached = 1; }
                } else local = UINT32_MAX;
            }
            local = __shfl_sync(0xffffffffu, local, 0);
            sample = __shfl_sync(0xffffffffu, sample, 0);
            cached = __shfl_sync(0xffffffffu, cached, 0);
            distance = __shfl_sync(0xffffffffu, distance, 0);
            if (local != UINT32_MAX && !cached) distance = l2_distance_device(base, query_row, dim, sample, norm);
            if (lane == 0) {
                batch[edge] = {distance, local == UINT32_MAX ? UINT32_MAX : context.key_begin + local};
                batch_samples[edge] = sample;
                batch_cached[edge] = cached;
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            batch_size = 0;
            for (uint32_t edge = 0; edge < context.degree; ++edge) {
                const Entry entry = batch[edge];
                if (entry.payload == UINT32_MAX) continue;
                const uint32_t sample = batch_samples[edge];
                stats.distance_evaluations += !batch_cached[edge];
                stats.distance_reuses += batch_cached[edge];
                // All cache writes happen after the readers' block barrier.
                cache[(sample * 2654435761u) & 511u] = {sample, entry.distance};
                if (queue_size == kQueueSize && sorted_frontier::less(queue[queue_size - 1], entry)) continue;
                const int admission = overlap_visit(visited, entry.payload);
                if (admission < 0) { failed = 1; break; }
                if (admission > 0) batch[batch_size++] = entry;
            }
            for (uint32_t i = batch_size; i < kMaxGraphDegree; ++i) batch[i] = {nanf(""), 0};
        }
        __syncthreads();
        // Existing rank-scatter merge primitives preserve one-writer queue
        // publication; only discovery admission is serialized.
        sorted_frontier::sort_parallel(batch, kMaxGraphDegree);
        sorted_frontier::merge_parallel(queue, queue_size, batch, batch_size, merged, kQueueSize);
        const int new_count = min(kQueueSize, queue_size + batch_size);
        for (int i = threadIdx.x; i < new_count; i += blockDim.x) queue[i] = merged[i];
        __syncthreads();
        if (threadIdx.x == 0) {
            queue_size = new_count;
            if (early_stop_reached(early_stop_mode, bad_count, finish_count,
                                  kQueueSize, static_cast<int>(stats.expansions), pass_count, topk)) finished = 1;
        }
        __syncthreads();
        if (finished || failed) break;
    }
    if (threadIdx.x == 0) {
        if (failed) {
            stats.retries = 1;
            for (uint32_t p = query.route_begin; p < query.route_end; ++p)
                fallback_ids[route_positions[p]] = route_ids[p];
        } else {
            // Preserve each route-local result and union the shared frontier
            // candidates into it.  Replacing route slots would discard the
            // diversity that the route-local merger already found when the
            // child graphs are disconnected.
            for (uint32_t p = query.route_begin; p < query.route_end; ++p) {
                const size_t output = static_cast<size_t>(route_ids[p]) * topk;
                float merged_dist[kMaxTopk];
                uint32_t merged_ids[kMaxTopk];
                uint32_t merged_size = 0;
                for (uint32_t k = 0; k < topk; ++k) {
                    const uint32_t id = results[output + k];
                    if (id == UINT32_MAX) continue;
                    insert_best_result(distances[output + k], id, topk,
                                       merged_dist, merged_ids, merged_size);
                }
                for (uint32_t k = 0; k < best_size; ++k) {
                    insert_best_result(best_dist[k], best_ids[k], topk,
                                       merged_dist, merged_ids, merged_size);
                }
                for (uint32_t k = 0; k < topk; ++k) {
                    results[output + k] = k < merged_size ? merged_ids[k] : UINT32_MAX;
                    distances[output + k] = k < merged_size ? merged_dist[k] : FLT_MAX;
                }
                route_steps[route_ids[p]] = static_cast<int>(stats.expansions);
            }
        }
        counters[blockIdx.x] = stats;
    }
}

// This reader consumes the native metadata sidecar, without reloading base
// vectors or constructing a second query planner. Bounds and all memberships
// are validated before any device pointer is created.
GraphMetadata load_overlap_metadata(const std::string& path, uint32_t base_rows,
                                     uint64_t graph_rows)
{
    std::ifstream input(path, std::ios::binary);
    GraphMetadataSidecarHeader header;
    input.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!input || header.version != kGraphMetadataSidecarVersion ||
        header.num_points != base_rows || header.graph_rows != graph_rows ||
        header.total_rows != graph_rows || header.label_count != header.group_count ||
        header.label_count == 0 || header.label_count > UINT32_MAX || graph_rows > UINT32_MAX ||
        header.original_label_count != header.label_count)
        fail_full_gpu("overlap metadata header mismatch: " + path);
    GraphMetadata result;
    result.label_count = header.label_count;
    result.original_label_count = header.original_label_count;
    result.total_rows = graph_rows;
    result.specificity_threshold = header.specificity_threshold;
    auto read = [&](auto& values, size_t count) {
        using Value = typename std::decay_t<decltype(values)>::value_type;
        values.resize(count);
        input.read(reinterpret_cast<char*>(values.data()), count * sizeof(Value));
        if (!input) fail_full_gpu("truncated overlap metadata: " + path);
    };
    read(result.cat_freq, result.label_count);
    read(result.group_posting_size, result.label_count);
    read(result.label_size, result.label_count);
    read(result.label_offset, result.label_count);
    read(result.row_to_sample, result.total_rows);
    input.seekg(static_cast<std::streamoff>(result.total_rows) * sizeof(uint64_t), std::ios::cur);
    read(result.label_filter_bit, result.original_label_count);
    if (input.peek() != std::ifstream::traits_type::eof()) fail_full_gpu("overlap metadata trailing bytes");
    uint64_t end = 0;
    for (uint32_t group = 0; group < result.label_count; ++group) {
        if (result.label_offset[group] != end || result.label_size[group] == 0)
            fail_full_gpu("overlap groups must be nonempty contiguous row spans");
        end += result.label_size[group];
    }
    if (end != graph_rows) fail_full_gpu("overlap metadata group spans do not cover graph");
    for (uint32_t sample : result.row_to_sample)
        if (sample >= base_rows) fail_full_gpu("overlap sample ID out of range");
    return result;
}

void prepare_full_gpu_overlap(const GraphRuntime& runtime,
                              const FullGpuSearchOptions& options,
                              FullGpuDeviceRuntime& device)
{
    if (options.overlap_graph_path.empty() && options.overlap_manifest_path.empty()) return;
    if (options.overlap_graph_path.empty() || options.overlap_manifest_path.empty())
        fail_full_gpu("overlap_graph_path and overlap_manifest_path must be provided together");
    const uint32_t base_rows = runtime_base_row_count(runtime);
    if (runtime.base_range_count != 1 || runtime.original_query_ranges.range_count != 1 ||
        runtime.base_range_values.size() != base_rows)
        fail_full_gpu("overlap currently requires one materialized numeric range column");
    std::ifstream manifest_input(options.overlap_manifest_path);
    nlohmann::json manifest;
    manifest_input >> manifest;
    if (manifest.at("schema") != "flashanns.finest_boundary_bridge" ||
        manifest.at("version") != 1 || manifest.at("dataset_size") != base_rows)
        fail_full_gpu("overlap manifest schema/dataset mismatch");
    auto artifact_path = [&](const char* field) {
        std::filesystem::path path(manifest.at("paths").at(field).get<std::string>());
        if (path.is_relative()) path = std::filesystem::path(options.overlap_manifest_path).parent_path() / path;
        return path.string();
    };
    auto bridge_graph = graph_file::load_graph_file_as_contiguous_matrix(options.overlap_graph_path);
    if (bridge_graph.cols == 0 || bridge_graph.cols > kMaxGraphDegree ||
        runtime.graph.rows + bridge_graph.rows >= sorted_frontier::kUsed)
        fail_full_gpu("overlap graph degree or exact graph-row key domain unsupported");
    const auto bridge_metadata = load_overlap_metadata(
        artifact_path("metadata_sidecar_path"), base_rows, bridge_graph.rows);
    std::vector<uint32_t> bridge_seeds;
    if (!load_seed_sidecar(artifact_path("seed_sidecar_path"), bridge_metadata, base_rows,
                          bridge_graph.cols, bridge_metadata.specificity_threshold, bridge_seeds))
        fail_full_gpu("overlap seed sidecar mismatch");
    if (manifest.at("graph_degree").get<uint32_t>() != bridge_graph.cols ||
        manifest.at("boundaries").size() != bridge_metadata.label_count)
        fail_full_gpu("overlap manifest graph/group count mismatch");
    std::vector<float> bridge_boundary(bridge_metadata.label_count);
    std::vector<uint8_t> assigned(bridge_metadata.label_count, 0);
    for (const auto& entry : manifest.at("boundaries")) {
        const uint32_t group = entry.at("bridge_group_id");
        if (group >= assigned.size() || assigned[group]++) fail_full_gpu("duplicate/invalid bridge group");
        const float boundary = entry.at("rank"), lo = entry.at("strip_lower"), hi = entry.at("strip_upper");
        if (!(lo < boundary && boundary < hi)) fail_full_gpu("invalid overlap strip bounds");
        bridge_boundary[group] = boundary;
        const uint32_t offset = bridge_metadata.label_offset[group];
        const uint32_t size = bridge_metadata.label_size[group];
        if (bridge_seeds[group] >= size) fail_full_gpu("overlap seed local ID out of bounds");
        for (uint32_t local = 0; local < size; ++local) {
            const uint32_t sample = bridge_metadata.row_to_sample[offset + local];
            const float rank = runtime.base_range_values[sample];
            if (!(rank >= lo && rank < hi)) fail_full_gpu("overlap row outside manifest strip");
            for (uint32_t edge = 0; edge < bridge_graph.cols; ++edge) {
                const uint32_t neighbor = bridge_graph.data[(static_cast<size_t>(offset) + local) * bridge_graph.cols + edge];
                if (neighbor != UINT32_MAX && neighbor >= size) fail_full_gpu("overlap neighbor local ID out of bounds");
            }
        }
    }
    auto overlap = std::make_unique<FullGpuOverlapDevice>();
    overlap->graph.upload(bridge_graph.data, "upload overlap graph");
    overlap->samples.upload(bridge_metadata.row_to_sample, "upload overlap row_to_sample");
    std::vector<FullGpuOverlapContext> contexts;
    std::vector<uint32_t> inverse_samples, inverse_locals, inverse_offsets;
    std::vector<float> group_min, group_max;
    auto add_contexts = [&](const GraphMetadata& metadata, const uint32_t* graph,
                            const uint32_t* samples, uint32_t degree, uint32_t key_offset, bool bridge) {
        for (uint32_t group = 0; group < metadata.label_count; ++group) {
            const uint32_t offset = metadata.label_offset[group], size = metadata.label_size[group];
            if (static_cast<uint64_t>(offset) + size > metadata.row_to_sample.size())
                fail_full_gpu("overlap primary group row span out of bounds");
            FullGpuOverlapContext context;
            context.graph = graph + static_cast<size_t>(offset) * degree;
            context.samples = samples + offset;
            context.degree = degree; context.size = size;
            context.key_begin = key_offset + offset; context.bridge = bridge;
            bool contiguous = size != 0;
            float lo = FLT_MAX, hi = -FLT_MAX;
            std::vector<std::pair<uint32_t, uint32_t>> inverse;
            for (uint32_t local = 0; local < size; ++local) {
                const uint32_t sample = metadata.row_to_sample[offset + local];
                if (sample >= base_rows) fail_full_gpu("primary overlap sample ID out of bounds");
                contiguous &= static_cast<uint64_t>(metadata.row_to_sample[offset]) + local == sample;
                lo = std::min(lo, runtime.base_range_values[sample]);
                hi = std::max(hi, runtime.base_range_values[sample]);
            }
            group_min.push_back(lo); group_max.push_back(hi);
            if (contiguous) {
                context.contiguous_sample_begin = metadata.row_to_sample[offset];
                inverse_offsets.push_back(UINT32_MAX);
            } else {
                inverse_offsets.push_back(inverse_samples.size());
                inverse.reserve(size);
                for (uint32_t local = 0; local < size; ++local)
                    inverse.emplace_back(metadata.row_to_sample[offset + local], local);
                std::sort(inverse.begin(), inverse.end());
                for (size_t i = 0; i < inverse.size(); ++i) {
                    if (i && inverse[i - 1].first == inverse[i].first)
                        fail_full_gpu("duplicate sample within overlap graph context");
                    inverse_samples.push_back(inverse[i].first);
                    inverse_locals.push_back(inverse[i].second);
                }
            }
            contexts.push_back(context);
        }
    };
    add_contexts(runtime.metadata, device.graph.data(), device.row_to_sample.data(), runtime.graph.cols, 0, false);
    const uint32_t bridge_context_begin = contexts.size();
    add_contexts(bridge_metadata, overlap->graph.data(), overlap->samples.data(), bridge_graph.cols, runtime.graph.rows, true);
    overlap->inverse_samples.upload(inverse_samples, "upload overlap inverse samples");
    overlap->inverse_locals.upload(inverse_locals, "upload overlap inverse locals");
    for (size_t i = 0; i < contexts.size(); ++i) {
        if (inverse_offsets[i] != UINT32_MAX) {
            contexts[i].inverse_samples = overlap->inverse_samples.data() + inverse_offsets[i];
            contexts[i].inverse_locals = overlap->inverse_locals.data() + inverse_offsets[i];
        }
    }
    overlap->contexts.upload(contexts, "upload overlap graph contexts");
    std::vector<std::vector<uint32_t>> routes(runtime.original_query_count);
    for (uint32_t pos = 0; pos < runtime.high_query_indices.size(); ++pos) {
        const uint32_t route = runtime.high_query_indices[pos];
        routes.at(runtime.route_to_original_query.at(route)).push_back(pos);
    }
    std::vector<FullGpuOverlapQuery> queries;
    std::vector<uint32_t> query_contexts, seeds, route_ids, route_positions;
    auto fallback = runtime.high_query_indices;
    for (uint32_t original = 0; original < routes.size(); ++original) {
        auto positions = routes[original];
        if (positions.size() < 2 || runtime.original_query_ranges.modes[original] != 3 ||
            original >= runtime.original_query_labels_matrix.row_labels.size() ||
            !runtime.original_query_labels_matrix.row_labels[original].empty()) continue;
        std::sort(positions.begin(), positions.end(), [&](uint32_t a, uint32_t b) {
            return group_min[runtime.query_group_ids[runtime.high_query_indices[a]]] <
                   group_min[runtime.query_group_ids[runtime.high_query_indices[b]]];
        });
        FullGpuOverlapQuery query;
        query.context_begin = query_contexts.size(); query.route_begin = route_ids.size();
        query.original_query = original; query.output_route = runtime.high_query_indices[positions.front()];
        query.range_mode = 3;
        query.lower = runtime.original_query_ranges.lower_bounds[original];
        query.upper = runtime.original_query_ranges.upper_bounds[original];
        std::vector<uint32_t> bridge_groups;
        for (size_t i = 1; i < positions.size() && bridge_groups.size() < options.overlap_max_bridges; ++i) {
            const uint32_t left = runtime.query_group_ids[runtime.high_query_indices[positions[i - 1]]];
            const uint32_t right = runtime.query_group_ids[runtime.high_query_indices[positions[i]]];
            for (uint32_t group = 0; group < bridge_metadata.label_count; ++group) {
                const float boundary = bridge_boundary[group];
                // Only a junction of selected canonical graph routes needs a
                // bridge. Internal finest boundaries of one parent do not.
                if (group_max[left] + 1.0f == boundary && group_min[right] == boundary &&
                    query.lower < boundary && boundary < query.upper) {
                    bridge_groups.push_back(group);
                    break;
                }
            }
        }
        if (options.overlap_max_bridges && bridge_groups.empty()) continue;
        for (uint32_t pos : positions) {
            const uint32_t route = runtime.high_query_indices[pos];
            const uint32_t group = runtime.query_group_ids[route];
            if (runtime.start_local_ids[route] >= contexts[group].size) fail_full_gpu("invalid overlap primary seed");
            query_contexts.push_back(group); seeds.push_back(runtime.start_local_ids[route]);
            route_ids.push_back(route); route_positions.push_back(pos); fallback[pos] = UINT32_MAX;
        }
        for (uint32_t group : bridge_groups) {
            query_contexts.push_back(bridge_context_begin + group); seeds.push_back(bridge_seeds[group]);
        }
        overlap->selected_bridges += bridge_groups.size();
        query.context_end = query_contexts.size(); query.route_end = route_ids.size();
        queries.push_back(query);
    }
    overlap->query_contexts.upload(query_contexts, "upload overlap query contexts");
    overlap->seeds.upload(seeds, "upload overlap seeds");
    overlap->queries.upload(queries, "upload overlap queries");
    overlap->route_ids.upload(route_ids, "upload overlap routes");
    overlap->route_positions.upload(route_positions, "upload overlap route positions");
    overlap->initial_fallback_ids.upload(fallback, "upload overlap initial fallback");
    overlap->fallback_ids.upload(fallback, "upload overlap fallback");
    overlap->counters.allocate(queries.size(), "allocate overlap counters");
    std::printf("FullGPU overlap dispatch: queries=%zu primary_contexts=%zu bridge_contexts=%llu total_contexts=%zu max_bridges=%u graph_rows=%llu degree=%llu\n",
                queries.size(), route_ids.size(), static_cast<unsigned long long>(overlap->selected_bridges),
                query_contexts.size(), options.overlap_max_bridges,
                static_cast<unsigned long long>(bridge_graph.rows), static_cast<unsigned long long>(bridge_graph.cols));
    device.overlap = std::move(overlap);
}

void launch_full_gpu_overlap(const GraphRuntime& runtime, const FullGpuSearchOptions& options,
                             FullGpuDeviceRuntime& device)
{
    auto& overlap = *device.overlap;
    check_cuda(cudaMemcpyAsync(overlap.fallback_ids.data(), overlap.initial_fallback_ids.data(),
                               overlap.fallback_ids.size() * sizeof(uint32_t), cudaMemcpyDeviceToDevice),
               "reset overlap fallback IDs");
    if (!overlap.queries.size()) return;
    dispatch_full_gpu_kernel_by_queue_size(options.search_queue_size, [&]<int kQueueSize>() {
        auto launch = [&]<typename T>(const T* base, const T* query) {
            full_gpu_overlap_kernel<kQueueSize, T><<<overlap.queries.size(), 128, options.dim * sizeof(T)>>>(
                base, query, options.dim, device.base_range_values.data(), overlap.contexts.data(),
                overlap.query_contexts.data(), overlap.seeds.data(), overlap.queries.data(),
                overlap.route_ids.data(), overlap.route_positions.data(), overlap.fallback_ids.data(),
                options.topk, std::max(1, options.search_finish_count),
                parse_search_early_stop_mode(options.search_early_stop_mode), std::max(0, options.search_max_steps),
                device.route_results.data(), device.route_distances.data(), device.route_steps.data(), overlap.counters.data());
        };
        if (runtime.vector_scalar_type == DenseVectorScalarType::Uint8) launch(device.base_u8.data(), device.query_u8.data());
        else if (options.use_fp16) launch(device.base_f16.data(), device.query_f16.data());
        else launch(device.base_f32.data(), device.query_f32.data());
    });
    check_cuda(cudaGetLastError(), "launch full_gpu_overlap_kernel");
}
