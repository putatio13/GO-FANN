#include "index/graph_builder.h"
#include "index/shared_resources.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <omp.h>
#include <optional>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/vamana.hpp>
#include <cuvs/util/file_io.hpp>
#include <raft/core/copy.hpp>
#include <raft/core/detail/mdspan_numpy_serializer.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <rmm/device_uvector.hpp>

#include <sys/stat.h>
#include <unistd.h>

#include "data/dataloader.h"
#include "data/vector_file_reader.h"
#include "index/graph_file_format.h"
#include "app/runtime_defaults.h"

namespace go_fann::graph_builder {

namespace {

[[noreturn]] void fail_graph_builder(const std::string& message)
{
    throw std::runtime_error("graph_builder failed: " + message);
}

std::string default_group_path_from_graph_path(const std::string& graph_path)
{
    std::filesystem::path path;
    path = graph_path;
    if (path.extension() == ".bin") {
        path.replace_extension(".identity.groups");
        return path.string();
    }
    return graph_path + ".identity.groups";
}

std::string default_lowfreq_index_path_from_graph_path(const std::string& graph_path)
{
    std::filesystem::path path;
    path = graph_path;
    if (path.extension() == ".bin") {
        path.replace_extension(".lowfreq.bin");
        return path.string();
    }
    return graph_path + ".lowfreq.bin";
}

std::string default_sidecar_path_from_graph_path(const std::string& graph_path,
                                                 const std::string& extension)
{
    std::filesystem::path path;
    path = graph_path;
    if (path.extension() == ".bin") {
        path.replace_extension(extension);
        return path.string();
    }
    return graph_path + extension;
}

std::string default_graph_file_path_from_graph_path(const std::string& graph_path)
{
    std::filesystem::path path;
    path = graph_path;
    if (path.extension() == ".bin" && std::filesystem::path(path.stem()).extension() == ".disk") {
        return graph_path;
    }
    if (path.extension() == ".bin") {
        path.replace_extension(".disk.bin");
        return path.string();
    }
    return graph_path + ".disk.bin";
}

std::string resolved_seed_sidecar_path(const GraphBuildOptions& options)
{
    if (options.seed_sidecar_path.empty() ||
        options.seed_sidecar_path == defaults::seed_sidecar_path) {
        return default_sidecar_path_from_graph_path(options.graph_path, ".seed");
    }
    return options.seed_sidecar_path;
}

std::string resolved_original_label_seed_sidecar_path(const GraphBuildOptions& options)
{
    if (options.original_label_seed_sidecar_path.empty() ||
        options.original_label_seed_sidecar_path == defaults::original_label_seed_sidecar_path) {
        return default_sidecar_path_from_graph_path(options.graph_path, ".orig_label.seed");
    }
    return options.original_label_seed_sidecar_path;
}

std::string resolved_metadata_sidecar_path(const GraphBuildOptions& options)
{
    return options.metadata_sidecar_path.empty()
        ? default_sidecar_path_from_graph_path(options.graph_path, ".metadata")
        : options.metadata_sidecar_path;
}

std::string graph_chunk_dir_from_graph_path(const std::string& graph_path)
{
    std::filesystem::path path;
    path = graph_path;
    if (path.extension() == ".bin") {
        path.replace_extension(".chunks");
        return path.string();
    }
    return graph_path + ".chunks";
}

std::string graph_tmp_path(const std::string& graph_path)
{
    return graph_path + ".tmp";
}

std::string lowfreq_tmp_path(const std::string& lowfreq_path)
{
    return lowfreq_path + ".tmp";
}

std::string graph_file_tmp_path(const std::string& graph_file_path)
{
    return graph_file_path + ".convert.tmp";
}

std::string chunk_file_path(const std::string& chunk_dir, uint32_t label)
{
    std::ostringstream oss;
    oss << chunk_dir << "/label_" << label << ".bin";
    return oss.str();
}

size_t expected_chunk_bytes(uint32_t label_size, int degree)
{
    return static_cast<size_t>(label_size) * static_cast<size_t>(degree) * sizeof(uint32_t);
}

bool cagra_label_uses_ace(uint32_t label_size)
{
    return label_size >= 2'000'000;
}

struct NumpyUint32FileInfo {
    std::string path;
    std::vector<uint64_t> shape;
    uint64_t payload_offset = 0;
    uint64_t payload_bytes = 0;
};

size_t checked_graph_word_count(uint64_t rows,
                                uint64_t cols,
                                const std::string& component)
{
    if (cols != 0 && rows > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) / cols) {
        fail_graph_builder("CAGRA " + component + " word count exceeds size_t");
    }
    return static_cast<size_t>(rows * cols);
}

uint64_t checked_graph_byte_offset(uint64_t rows,
                                   uint64_t cols,
                                   const std::string& component)
{
    if (cols != 0 && rows > UINT64_MAX / cols) {
        fail_graph_builder("CAGRA " + component + " word offset overflows uint64");
    }
    const uint64_t words = rows * cols;
    if (words > UINT64_MAX / sizeof(uint32_t)) {
        fail_graph_builder("CAGRA " + component + " byte offset overflows uint64");
    }
    return words * sizeof(uint32_t);
}

NumpyUint32FileInfo inspect_numpy_uint32_file(const cuvs::util::file_descriptor& fd,
                                              const std::string& component)
{
    if (!fd.is_valid()) {
        fail_graph_builder("CAGRA " + component + " file descriptor is invalid");
    }
    if (::lseek(fd.get(), 0, SEEK_SET) == -1) {
        fail_graph_builder("failed to seek CAGRA " + component + " file");
    }

    auto stream = fd.make_istream();
    const auto header = raft::detail::numpy_serializer::read_header(stream);
    if (header.fortran_order || header.dtype.kind != 'u' ||
        header.dtype.itemsize != sizeof(uint32_t)) {
        fail_graph_builder("CAGRA " + component +
                           " must be a row-major uint32 NumPy array");
    }

    NumpyUint32FileInfo info;
    info.path = fd.get_path();
    info.shape.assign(header.shape.begin(), header.shape.end());
    uint64_t element_count = 1;
    for (const uint64_t extent : info.shape) {
        if (extent != 0 && element_count > UINT64_MAX / extent) {
            fail_graph_builder("CAGRA " + component + " shape overflows uint64");
        }
        element_count *= extent;
    }
    if (element_count > UINT64_MAX / sizeof(uint32_t)) {
        fail_graph_builder("CAGRA " + component + " payload size overflows uint64");
    }
    info.payload_bytes = element_count * sizeof(uint32_t);

    unsigned char preamble[10]{};
    const ssize_t preamble_bytes = ::pread(fd.get(), preamble, sizeof(preamble), 0);
    if (preamble_bytes != static_cast<ssize_t>(sizeof(preamble)) ||
        std::memcmp(preamble, "\x93NUMPY", 6) != 0 ||
        preamble[6] != 1 || preamble[7] != 0) {
        fail_graph_builder("CAGRA " + component + " has an invalid NumPy v1 header");
    }
    info.payload_offset = 10 + static_cast<uint64_t>(preamble[8]) +
        (static_cast<uint64_t>(preamble[9]) << 8);

    struct stat file_stat {};
    if (::fstat(fd.get(), &file_stat) != 0 || file_stat.st_size < 0) {
        fail_graph_builder("failed to stat CAGRA " + component + " file");
    }
    const uint64_t file_bytes = static_cast<uint64_t>(file_stat.st_size);
    if (info.payload_offset > file_bytes ||
        info.payload_bytes != file_bytes - info.payload_offset) {
        fail_graph_builder("CAGRA " + component + " file size does not match its NumPy shape");
    }
    return info;
}

std::vector<uint32_t> read_numpy_uint32_vector(const cuvs::util::file_descriptor& fd,
                                               uint64_t expected_rows,
                                               const std::string& component)
{
    const auto info = inspect_numpy_uint32_file(fd, component);
    if (info.shape.size() != 1 || info.shape[0] != expected_rows) {
        fail_graph_builder("CAGRA " + component + " shape mismatch: expected [" +
                           std::to_string(expected_rows) + "]");
    }
    std::vector<uint32_t> values(static_cast<size_t>(expected_rows));
    cuvs::util::read_large_file(fd,
                                values.data(),
                                static_cast<size_t>(info.payload_bytes),
                                info.payload_offset);
    return values;
}

void validate_cagra_mapping(const std::vector<uint32_t>& mapping)
{
    constexpr size_t kRowsPerWord = sizeof(uint64_t) * 8;
    const size_t state_words = mapping.size() / kRowsPerWord +
        (mapping.size() % kRowsPerWord == 0 ? 0 : 1);
    std::vector<uint64_t> seen(state_words, 0);
    for (size_t internal_row = 0; internal_row < mapping.size(); ++internal_row) {
        const uint32_t original_row = mapping[internal_row];
        if (original_row >= mapping.size()) {
            fail_graph_builder("CAGRA source mapping contains out-of-range row " +
                               std::to_string(original_row) + " at internal row " +
                               std::to_string(internal_row));
        }
        const size_t state_word = static_cast<size_t>(original_row) / kRowsPerWord;
        const uint64_t state_bit = uint64_t{1} << (original_row % kRowsPerWord);
        if ((seen[state_word] & state_bit) != 0) {
            fail_graph_builder("CAGRA source mapping is not a permutation; duplicate original row " +
                               std::to_string(original_row));
        }
        seen[state_word] |= state_bit;
    }
}

template <typename IndexT>
std::vector<uint32_t> export_cagra_graph(raft::resources& res,
                                         const IndexT& index,
                                         uint32_t expected_rows,
                                         int requested_degree)
{
    if (expected_rows == 0 || requested_degree <= 0) {
        fail_graph_builder("invalid expected CAGRA graph shape");
    }
    if (static_cast<uint64_t>(index.size()) != expected_rows) {
        fail_graph_builder("CAGRA index row count mismatch: expected " +
                           std::to_string(expected_rows) + ", got " +
                           std::to_string(index.size()));
    }

    const auto graph_view = index.graph();
    const bool has_device_graph = graph_view.extent(0) != 0 || graph_view.extent(1) != 0 ||
        graph_view.data_handle() != nullptr;
    const bool has_graph_file = index.graph_fd().has_value();
    if (has_device_graph == has_graph_file) {
        fail_graph_builder(has_device_graph
                               ? "CAGRA index exposes ambiguous device and graph file storage"
                               : "CAGRA index exposes neither device nor graph file storage");
    }

    uint64_t built_rows = 0;
    uint64_t built_cols = 0;
    NumpyUint32FileInfo graph_file_info;
    if (has_device_graph) {
        built_rows = static_cast<uint64_t>(graph_view.extent(0));
        built_cols = static_cast<uint64_t>(graph_view.extent(1));
        if (graph_view.data_handle() == nullptr) {
            fail_graph_builder("CAGRA device graph has a null data handle");
        }
    } else {
        const auto& graph_fd = *index.graph_fd();
        graph_file_info = inspect_numpy_uint32_file(graph_fd, "graph");
        if (graph_file_info.shape.size() != 2) {
            fail_graph_builder("CAGRA graph file must be a 2D NumPy array");
        }
        built_rows = graph_file_info.shape[0];
        built_cols = graph_file_info.shape[1];
    }
    if (built_rows != expected_rows || built_cols == 0 ||
        built_cols != static_cast<uint64_t>(index.graph_degree())) {
        fail_graph_builder("CAGRA graph shape mismatch: expected rows=" +
                           std::to_string(expected_rows) + ", got " +
                           std::to_string(built_rows) + "x" + std::to_string(built_cols) +
                           ", index degree=" + std::to_string(index.graph_degree()));
    }

    std::optional<std::vector<uint32_t>> source_mapping;
    const auto device_mapping = index.source_indices();
    const bool has_device_mapping = device_mapping.has_value();
    const bool has_file_mapping = index.mapping_fd().has_value();
    if (has_device_mapping && has_file_mapping) {
        fail_graph_builder("CAGRA index exposes ambiguous device and disk source mappings");
    }
    if (has_graph_file != has_file_mapping) {
        fail_graph_builder(has_graph_file
                               ? "CAGRA graph file is missing its disk source mapping"
                               : "CAGRA device graph unexpectedly exposes a disk source mapping");
    }
    if (has_graph_file && has_device_mapping) {
        fail_graph_builder("CAGRA graph file unexpectedly exposes a device source mapping");
    }
    if (has_device_mapping) {
        if (device_mapping->extent(0) != static_cast<int64_t>(expected_rows) ||
            device_mapping->data_handle() == nullptr) {
            fail_graph_builder("CAGRA device source mapping has an invalid shape or data handle");
        }
        source_mapping.emplace(static_cast<size_t>(expected_rows));
        raft::copy(source_mapping->data(),
                   device_mapping->data_handle(),
                   source_mapping->size(),
                   raft::resource::get_cuda_stream(res));
        raft::resource::sync_stream(res);
    } else if (has_file_mapping) {
        source_mapping.emplace(read_numpy_uint32_vector(*index.mapping_fd(),
                                                        expected_rows,
                                                        "source mapping"));
    }

    if (source_mapping) {
        validate_cagra_mapping(*source_mapping);
    }

    const size_t output_words = checked_graph_word_count(
        expected_rows, static_cast<uint64_t>(requested_degree), "output graph");
    std::vector<uint32_t> output(output_words, UINT32_MAX);
    const uint64_t copied_cols = std::min<uint64_t>(built_cols,
                                                    static_cast<uint64_t>(requested_degree));
    constexpr size_t kGraphReadChunkBytes = 64ULL * 1024ULL * 1024ULL;
    const uint64_t built_row_bytes = checked_graph_byte_offset(1, built_cols, "row size");
    const uint64_t chunk_rows = std::max<uint64_t>(
        1, static_cast<uint64_t>(kGraphReadChunkBytes) / built_row_bytes);
    const size_t chunk_words = checked_graph_word_count(
        std::min<uint64_t>(expected_rows, chunk_rows), built_cols, "read chunk");
    std::vector<uint32_t> chunk(chunk_words);

    for (uint64_t row_begin = 0; row_begin < expected_rows; row_begin += chunk_rows) {
        const uint64_t rows = std::min<uint64_t>(chunk_rows, expected_rows - row_begin);
        const size_t words = checked_graph_word_count(rows, built_cols, "read chunk");
        const uint64_t graph_byte_offset =
            checked_graph_byte_offset(row_begin, built_cols, "graph row offset");
        if (has_device_graph) {
            raft::copy(chunk.data(),
                       graph_view.data_handle() + graph_byte_offset / sizeof(uint32_t),
                       words,
                       raft::resource::get_cuda_stream(res));
            raft::resource::sync_stream(res);
        } else {
            if (graph_byte_offset > UINT64_MAX - graph_file_info.payload_offset) {
                fail_graph_builder("CAGRA graph payload offset overflows uint64");
            }
            const uint64_t byte_offset = graph_file_info.payload_offset + graph_byte_offset;
            cuvs::util::read_large_file(*index.graph_fd(),
                                        chunk.data(),
                                        words * sizeof(uint32_t),
                                        byte_offset);
        }

        for (uint64_t row_delta = 0; row_delta < rows; ++row_delta) {
            const uint64_t internal_row = row_begin + row_delta;
            const uint32_t output_row = source_mapping
                ? (*source_mapping)[static_cast<size_t>(internal_row)]
                : static_cast<uint32_t>(internal_row);
            uint32_t* output_row_ptr = output.data() +
                static_cast<size_t>(output_row) * static_cast<size_t>(requested_degree);
            const uint32_t* input_row_ptr = chunk.data() +
                static_cast<size_t>(row_delta) * static_cast<size_t>(built_cols);
            bool has_valid_neighbor = false;
            for (uint64_t col = 0; col < copied_cols; ++col) {
                const uint32_t internal_neighbor = input_row_ptr[col];
                if (internal_neighbor == UINT32_MAX) {
                    continue;
                }
                if (internal_neighbor >= expected_rows) {
                    fail_graph_builder("CAGRA graph contains out-of-range neighbor " +
                                       std::to_string(internal_neighbor) + " at internal row " +
                                       std::to_string(internal_row));
                }
                output_row_ptr[col] = source_mapping
                    ? (*source_mapping)[internal_neighbor]
                    : internal_neighbor;
                has_valid_neighbor = true;
            }
            if (!has_valid_neighbor) {
                fail_graph_builder("CAGRA graph row " + std::to_string(output_row) +
                                   " has no valid neighbor");
            }
        }
    }
    std::printf("[graph-build] validated CAGRA graph storage=%s rows=%u built_degree=%lu "
                "output_degree=%d source_mapping=%s\n",
                has_graph_file ? "disk" : "device",
                expected_rows,
                static_cast<unsigned long>(built_cols),
                requested_degree,
                source_mapping ? "internal_to_original" : "identity");
    std::fflush(stdout);
    return output;
}

bool file_size_matches(const std::string& path, uintmax_t expected_size)
{
    std::error_code ec;
    const uintmax_t actual_size = std::filesystem::file_size(std::filesystem::path{path.c_str()}, ec);
    return !ec && actual_size == expected_size;
}

int seed_sidecar_thread_count()
{
    const int max_threads = std::max(1, omp_get_max_threads());
    const char* value = std::getenv("GO_FANN_SEED_THREADS");
    if (value != nullptr && *value != '\0') {
        const unsigned long parsed = std::strtoul(value, nullptr, 10);
        if (parsed > 0) {
            return std::max(1, std::min<int>(static_cast<int>(parsed), max_threads));
        }
    }
    return std::max(1, std::min(16, max_threads));
}

struct BuildSeedSidecars {
    std::vector<uint32_t> seed_local_id_by_label;
    std::vector<uint32_t> seed_by_original_label;
};

struct LabelSeedUpdate {
    uint32_t label = UINT32_MAX;
    uint32_t seed = UINT32_MAX;
    std::vector<std::pair<uint32_t, uint32_t>> original_label_seeds;
};

BuildSeedSidecars make_empty_seed_sidecars(const GraphMetadata& metadata)
{
    BuildSeedSidecars sidecars;
    sidecars.seed_local_id_by_label.assign(metadata.label_count, UINT32_MAX);
    sidecars.seed_by_original_label.assign(metadata.original_label_count, UINT32_MAX);
    return sidecars;
}

template <typename ScalarT, typename IncludeRow>
uint32_t centroid_seed_for_label_rows(const std::vector<ScalarT>& label_data,
                                      uint32_t label_size,
                                      int dim,
                                      IncludeRow include_row)
{
    if (dim <= 0) {
        fail_graph_builder("dimension must be positive while building seed sidecars");
    }
    const size_t expected_values = static_cast<size_t>(label_size) * static_cast<size_t>(dim);
    if (label_data.size() != expected_values) {
        fail_graph_builder("label data size mismatch while building seed sidecars");
    }

    const int thread_count = label_size < 8192
        ? 1
        : std::min<int>(seed_sidecar_thread_count(), static_cast<int>(label_size));
    std::vector<double> centroid(static_cast<size_t>(dim), 0.0);
    uint64_t included_rows = 0;
    if (thread_count == 1) {
        for (uint32_t local_row = 0; local_row < label_size; ++local_row) {
            if (!include_row(local_row)) {
                continue;
            }
            const ScalarT* vec = label_data.data() + static_cast<size_t>(local_row) * dim;
            for (int d = 0; d < dim; ++d) {
                centroid[static_cast<size_t>(d)] += static_cast<double>(vec[d]);
            }
            ++included_rows;
        }
    } else {
        std::vector<double> partial_centroids(static_cast<size_t>(thread_count) *
                                                 static_cast<size_t>(dim),
                                             0.0);
        std::vector<uint64_t> partial_counts(static_cast<size_t>(thread_count), 0);
#pragma omp parallel num_threads(thread_count)
        {
            const int tid = omp_get_thread_num();
            double* local_centroid =
                partial_centroids.data() + static_cast<size_t>(tid) * static_cast<size_t>(dim);
            uint64_t local_count = 0;
#pragma omp for schedule(static)
            for (int64_t row64 = 0; row64 < static_cast<int64_t>(label_size); ++row64) {
                const uint32_t local_row = static_cast<uint32_t>(row64);
                if (!include_row(local_row)) {
                    continue;
                }
                const ScalarT* vec = label_data.data() + static_cast<size_t>(local_row) * dim;
                for (int d = 0; d < dim; ++d) {
                    local_centroid[static_cast<size_t>(d)] += static_cast<double>(vec[d]);
                }
                ++local_count;
            }
            partial_counts[static_cast<size_t>(tid)] = local_count;
        }
        for (int tid = 0; tid < thread_count; ++tid) {
            included_rows += partial_counts[static_cast<size_t>(tid)];
            const double* local_centroid =
                partial_centroids.data() + static_cast<size_t>(tid) * static_cast<size_t>(dim);
            for (int d = 0; d < dim; ++d) {
                centroid[static_cast<size_t>(d)] += local_centroid[static_cast<size_t>(d)];
            }
        }
    }
    if (included_rows == 0) {
        return UINT32_MAX;
    }
    for (int d = 0; d < dim; ++d) {
        centroid[static_cast<size_t>(d)] /= static_cast<double>(included_rows);
    }

    double best_distance = std::numeric_limits<double>::max();
    uint32_t best_local_row = UINT32_MAX;
    if (thread_count == 1) {
        for (uint32_t local_row = 0; local_row < label_size; ++local_row) {
            if (!include_row(local_row)) {
                continue;
            }
            const ScalarT* vec = label_data.data() + static_cast<size_t>(local_row) * dim;
            double distance = 0.0;
            for (int d = 0; d < dim; ++d) {
                const double diff = static_cast<double>(vec[d]) - centroid[static_cast<size_t>(d)];
                distance += diff * diff;
            }
            if (distance < best_distance) {
                best_distance = distance;
                best_local_row = local_row;
            }
        }
    } else {
        std::vector<double> partial_best_distances(static_cast<size_t>(thread_count),
                                                   std::numeric_limits<double>::max());
        std::vector<uint32_t> partial_best_rows(static_cast<size_t>(thread_count), UINT32_MAX);
#pragma omp parallel num_threads(thread_count)
        {
            const int tid = omp_get_thread_num();
            double local_best_distance = std::numeric_limits<double>::max();
            uint32_t local_best_row = UINT32_MAX;
#pragma omp for schedule(static)
            for (int64_t row64 = 0; row64 < static_cast<int64_t>(label_size); ++row64) {
                const uint32_t local_row = static_cast<uint32_t>(row64);
                if (!include_row(local_row)) {
                    continue;
                }
                const ScalarT* vec = label_data.data() + static_cast<size_t>(local_row) * dim;
                double distance = 0.0;
                for (int d = 0; d < dim; ++d) {
                    const double diff =
                        static_cast<double>(vec[d]) - centroid[static_cast<size_t>(d)];
                    distance += diff * diff;
                }
                if (distance < local_best_distance ||
                    (distance == local_best_distance && local_row < local_best_row)) {
                    local_best_distance = distance;
                    local_best_row = local_row;
                }
            }
            partial_best_distances[static_cast<size_t>(tid)] = local_best_distance;
            partial_best_rows[static_cast<size_t>(tid)] = local_best_row;
        }
        for (int tid = 0; tid < thread_count; ++tid) {
            const double distance = partial_best_distances[static_cast<size_t>(tid)];
            const uint32_t row = partial_best_rows[static_cast<size_t>(tid)];
            if (distance < best_distance ||
                (distance == best_distance && row < best_local_row)) {
                best_distance = distance;
                best_local_row = row;
            }
        }
    }
    return best_local_row;
}

template <typename ScalarT>
LabelSeedUpdate compute_label_seed_update(const GraphMetadata& metadata,
                                          const GroupLayout& group_layout,
                                          uint32_t label,
                                          const std::vector<ScalarT>& label_data,
                                          int dim)
{
    if (label >= metadata.label_count ||
        label >= metadata.label_size.size() ||
        label >= metadata.label_offset.size() ||
        label >= group_layout.groups.size()) {
        fail_graph_builder("metadata/group-layout mismatch while building seed sidecars");
    }

    LabelSeedUpdate update;
    update.label = label;
    const uint32_t label_size = metadata.label_size[label];
    if (label_size == 0) {
        return update;
    }
    const uint32_t row_begin = metadata.label_offset[label];
    update.seed = centroid_seed_for_label_rows(label_data, label_size, dim, [](uint32_t) {
        return true;
    });

    update.original_label_seeds.reserve(group_layout.groups[label].size());
    for (uint32_t original_label : group_layout.groups[label]) {
        if (original_label >= metadata.label_filter_bit.size()) {
            fail_graph_builder("group layout original label exceeds metadata while building seed sidecars");
        }
        const uint64_t filter_bit = metadata.label_filter_bit[original_label];
        if (filter_bit == 0) {
            continue;
        }
        if (group_layout.groups[label].size() == 1) {
            if (update.seed != UINT32_MAX) {
                update.original_label_seeds.emplace_back(original_label, update.seed);
            }
            continue;
        }
        const uint32_t original_seed =
            centroid_seed_for_label_rows(label_data,
                                         label_size,
                                         dim,
                                         [&](uint32_t local_row) {
                                             const uint32_t global_row = row_begin + local_row;
                                             return global_row < metadata.row_filter_mask.size() &&
                                                    (metadata.row_filter_mask[global_row] & filter_bit) != 0;
                                         });
        if (original_seed != UINT32_MAX) {
            update.original_label_seeds.emplace_back(original_label, original_seed);
        }
    }
    return update;
}

void apply_label_seed_update(BuildSeedSidecars& sidecars, const LabelSeedUpdate& update)
{
    if (update.label == UINT32_MAX) {
        return;
    }
    if (update.label >= sidecars.seed_local_id_by_label.size()) {
        fail_graph_builder("label seed update exceeds sidecar label count");
    }
    sidecars.seed_local_id_by_label[update.label] = update.seed;
    for (const auto& [original_label, seed] : update.original_label_seeds) {
        if (original_label >= sidecars.seed_by_original_label.size()) {
            fail_graph_builder("original-label seed update exceeds sidecar label count");
        }
        sidecars.seed_by_original_label[original_label] = seed;
    }
}

template <typename MatrixT>
BuildSeedSidecars build_seed_sidecars_from_loaded_base(const MatrixT& base_vectors,
                                                       const GraphMetadata& metadata,
                                                       const GroupLayout& group_layout,
                                                       int dim)
{
    BuildSeedSidecars sidecars;
    sidecars.seed_local_id_by_label = build_seed_local_ids(base_vectors, metadata, dim);
    sidecars.seed_by_original_label =
        build_seeds_by_original_label(base_vectors, metadata, group_layout, dim);
    return sidecars;
}

uint64_t metadata_sidecar_row_limit_key(const GraphBuildOptions& options, uint64_t num_points)
{
    return options.row_limit > 0 ? options.row_limit : num_points;
}

void remove_if_exists(const std::string& path)
{
    std::error_code ec;
    std::filesystem::remove(std::filesystem::path{path.c_str()}, ec);
}

class ScopedDirectoryCleanup {
public:
    explicit ScopedDirectoryCleanup(std::filesystem::path path) : path_(std::move(path)) {}

    ~ScopedDirectoryCleanup()
    {
        if (enabled_) {
            std::error_code ec;
            std::filesystem::remove_all(path_, ec);
        }
    }

    ScopedDirectoryCleanup(const ScopedDirectoryCleanup&) = delete;
    ScopedDirectoryCleanup& operator=(const ScopedDirectoryCleanup&) = delete;

private:
    std::filesystem::path path_;
    bool enabled_ = true;
};

void rename_replace(const std::string& from, const std::string& to)
{
    std::error_code ec;
    std::filesystem::remove(std::filesystem::path{to.c_str()}, ec);
    ec.clear();
    std::filesystem::rename(std::filesystem::path{from.c_str()}, std::filesystem::path{to.c_str()}, ec);
    if (ec) {
        fail_graph_builder("failed to rename " + from + " to " + to + ": " + ec.message());
    }
}

void write_chunk_file(const std::string& path, const std::vector<uint32_t>& chunk)
{
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        fail_graph_builder("unable to open chunk output " + path);
    }
    output.write(reinterpret_cast<const char*>(chunk.data()),
                 static_cast<std::streamsize>(chunk.size() * sizeof(uint32_t)));
    if (!output) {
        fail_graph_builder("failed to write chunk payload to " + path);
    }
}

struct PendingChunkWrite {
    std::string path;
    std::vector<uint32_t> chunk;
};

class ChunkWriter {
public:
    explicit ChunkWriter(size_t max_queue_depth)
        : max_queue_depth_(std::max<size_t>(1, max_queue_depth)), worker_([this] { run(); })
    {
    }

    ~ChunkWriter()
    {
        try {
            finish();
        } catch (...) {
        }
    }

    void enqueue(std::string path, std::vector<uint32_t>&& chunk)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        producer_cv_.wait(lock, [this] { return failed_ || queue_.size() < max_queue_depth_; });
        if (failed_) {
            rethrow_locked();
        }
        queue_.push(PendingChunkWrite{std::move(path), std::move(chunk)});
        consumer_cv_.notify_one();
    }

    void finish()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        done_ = true;
        consumer_cv_.notify_one();
        lock.unlock();
        if (worker_.joinable()) {
            worker_.join();
        }
        lock.lock();
        if (failed_) {
            rethrow_locked();
        }
    }

private:
    [[noreturn]] void rethrow_locked()
    {
        if (error_) {
            std::rethrow_exception(error_);
        }
        fail_graph_builder("chunk writer failed without exception payload");
    }

    void run()
    {
        try {
            while (true) {
                PendingChunkWrite item;
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    consumer_cv_.wait(lock, [this] { return done_ || !queue_.empty(); });
                    if (queue_.empty()) {
                        break;
                    }
                    item.path = std::move(queue_.front().path);
                    item.chunk = std::move(queue_.front().chunk);
                    queue_.pop();
                    producer_cv_.notify_one();
                }
                write_chunk_file(item.path, item.chunk);
            }
        } catch (...) {
            std::unique_lock<std::mutex> lock(mutex_);
            failed_ = true;
            error_ = std::current_exception();
            producer_cv_.notify_all();
            consumer_cv_.notify_all();
        }
    }

    size_t max_queue_depth_ = 1;
    std::mutex mutex_;
    std::condition_variable producer_cv_;
    std::condition_variable consumer_cv_;
    std::queue<PendingChunkWrite> queue_;
    bool done_ = false;
    bool failed_ = false;
    std::exception_ptr error_;
    std::thread worker_;
};

std::optional<rmm::device_uvector<uint8_t>> upload_base_vectors_if_fit(raft::resources& res,
                                                                        const DenseUint8Matrix& base_vectors,
                                                                        int dim)
{
    const size_t bytes = static_cast<size_t>(base_vectors.rows) * static_cast<size_t>(dim);
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    raft::resource::sync_stream(res);
    const cudaError_t status = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (status != cudaSuccess) {
        return std::nullopt;
    }
    constexpr size_t reserve_bytes = static_cast<size_t>(512) << 20;
    if (bytes + reserve_bytes >= free_bytes) {
        return std::nullopt;
    }
    rmm::device_uvector<uint8_t> device_base(bytes, raft::resource::get_cuda_stream(res));
    raft::copy(device_base.data(),
               base_vectors.data.data(),
               bytes,
               raft::resource::get_cuda_stream(res));
    raft::resource::sync_stream(res);
    return std::optional<rmm::device_uvector<uint8_t>>(std::move(device_base));
}

std::vector<uint8_t> build_label_dataset_from_device(raft::resources& res,
                                                     const rmm::device_uvector<uint8_t>& device_base,
                                                     const GraphMetadata& metadata,
                                                     uint32_t label,
                                                     int dim)
{
    const uint32_t label_size = metadata.label_size[label];
    const uint32_t row_begin = metadata.label_offset[label];
    if (label_size == 0) {
        return {};
    }
    rmm::device_uvector<uint8_t> filtered_dataset(static_cast<size_t>(label_size) * dim,
                                                  raft::resource::get_cuda_stream(res));
    const uint8_t* device_base_ptr = device_base.data();
    uint8_t* filtered_ptr = filtered_dataset.data();
    for (uint32_t local_row = 0; local_row < label_size; ++local_row) {
        const uint32_t sample = metadata.row_to_sample[row_begin + local_row];
        raft::copy(filtered_ptr + static_cast<size_t>(local_row) * dim,
                   device_base_ptr + static_cast<size_t>(sample) * dim,
                   static_cast<size_t>(dim),
                   raft::resource::get_cuda_stream(res));
    }
    std::vector<uint8_t> host_dataset(static_cast<size_t>(label_size) * dim);
    raft::copy(host_dataset.data(),
               filtered_dataset.data(),
               host_dataset.size(),
               raft::resource::get_cuda_stream(res));
    raft::resource::sync_stream(res);
    return host_dataset;
}

void append_chunk_file_to_graph_writer(go_fann::graph_file::GraphFileSequentialWriter& writer,
                                      const std::string& input_path,
                                      uint32_t label_size,
                                      int degree)
{
    std::ifstream input(input_path, std::ios::binary);
    if (!input) {
        fail_graph_builder("unable to open chunk input " + input_path);
    }

    constexpr uint64_t target_buffer_bytes = static_cast<uint64_t>(64) << 20;
    if (degree <= 0) {
        fail_graph_builder("degree must be > 0 while appending chunk " + input_path);
    }
    const uint64_t row_words = static_cast<uint64_t>(degree);
    const uint64_t target_rows = std::max<uint64_t>(
        1, target_buffer_bytes / (row_words * sizeof(uint32_t)));
    const uint64_t rows_per_batch = std::min<uint64_t>(target_rows, label_size);
    std::vector<uint32_t> buffer(static_cast<size_t>(rows_per_batch) * static_cast<size_t>(degree));

    uint32_t rows_done = 0;
    while (rows_done < label_size) {
        const uint32_t rows_now = static_cast<uint32_t>(
            std::min<uint64_t>(rows_per_batch, static_cast<uint64_t>(label_size - rows_done)));
        const size_t words_now = static_cast<size_t>(rows_now) * static_cast<size_t>(degree);
        input.read(reinterpret_cast<char*>(buffer.data()),
                   static_cast<std::streamsize>(words_now * sizeof(uint32_t)));
        if (!input) {
            fail_graph_builder("failed to read chunk payload from " + input_path);
        }
        writer.append_rows(buffer.data(), rows_now);
        rows_done += rows_now;
    }

    char extra = 0;
    if (input.read(&extra, 1)) {
        fail_graph_builder("chunk payload has trailing bytes in " + input_path);
    }
    if (!input.eof()) {
        fail_graph_builder("failed while checking chunk EOF for " + input_path);
    }
}

void print_build_progress(uint32_t completed_labels,
                          uint32_t total_labels,
                          uint32_t completed_rows,
                          uint32_t total_rows,
                          uint32_t current_label,
                          uint32_t current_label_size)
{
    const double progress = total_rows == 0
        ? 100.0
        : 100.0 * static_cast<double>(completed_rows) / static_cast<double>(total_rows);
    const int bar_width = 40;
    const int filled = static_cast<int>(progress / 100.0 * bar_width);
    std::printf("\r[graph-build] [");
    for (int i = 0; i < bar_width; ++i) {
        if (i < filled) {
            std::printf("=");
        } else if (i == filled && filled < bar_width) {
            std::printf(">");
        } else {
            std::printf(" ");
        }
    }
    std::printf("] %6.2f%% labels %u/%u rows %u/%u current_label=%u size=%u",
                progress,
                completed_labels,
                total_labels,
                completed_rows,
                total_rows,
                current_label,
                current_label_size);
    std::fflush(stdout);
}

void write_identity_groups(const std::string& path, uint32_t label_count)
{
    std::ofstream output(path);
    if (!output) {
        fail_graph_builder("unable to write group file " + path);
    }
    for (uint32_t label = 0; label < label_count; ++label) {
        output << label << '\n';
    }
}

size_t scalar_size_for(DenseVectorScalarType scalar_type)
{
    if (scalar_type == DenseVectorScalarType::Uint8) {
        return sizeof(uint8_t);
    }
    if (scalar_type == DenseVectorScalarType::Float16) {
        return sizeof(uint16_t);
    }
    return sizeof(float);
}

uint64_t checked_u64_mul(uint64_t lhs, uint64_t rhs, const char* description)
{
    if (rhs != 0 && lhs > std::numeric_limits<uint64_t>::max() / rhs) {
        fail_graph_builder(std::string(description) + " overflows uint64_t");
    }
    return lhs * rhs;
}

uint64_t estimate_stream_label_memory_bytes(uint32_t label_size,
                                            int dim,
                                            int degree,
                                            size_t scalar_size)
{
    const uint64_t rows = label_size;
    const uint64_t dims = static_cast<uint64_t>(std::max(dim, 1));
    const uint64_t deg = static_cast<uint64_t>(std::max(degree, 1));
    const uint64_t input_bytes =
        checked_u64_mul(checked_u64_mul(rows, dims, "stream label input estimate"),
                        static_cast<uint64_t>(scalar_size),
                        "stream label input estimate");
    const uint64_t float_workspace_bytes =
        checked_u64_mul(checked_u64_mul(rows, dims, "stream label workspace estimate"),
                        static_cast<uint64_t>(sizeof(float)),
                        "stream label workspace estimate");
    const uint64_t graph_bytes =
        checked_u64_mul(checked_u64_mul(rows, deg, "stream label graph estimate"),
                        static_cast<uint64_t>(sizeof(uint32_t)),
                        "stream label graph estimate");
    const uint64_t cagra_factor = 4;
    uint64_t total = input_bytes + graph_bytes;
    if (total < input_bytes || total < graph_bytes) {
        return std::numeric_limits<uint64_t>::max();
    }
    const uint64_t cagra_workspace = float_workspace_bytes * cagra_factor;
    if (float_workspace_bytes != 0 && cagra_workspace / float_workspace_bytes != cagra_factor) {
        return std::numeric_limits<uint64_t>::max();
    }
    if (total > std::numeric_limits<uint64_t>::max() - cagra_workspace) {
        return std::numeric_limits<uint64_t>::max();
    }
    return total + cagra_workspace;
}

uint64_t current_gpu_memory_budget_bytes()
{
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    const cudaError_t status = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (status != cudaSuccess || free_bytes == 0) {
        return static_cast<uint64_t>(4) << 30;
    }
    constexpr uint64_t reserve_bytes = static_cast<uint64_t>(1) << 30;
    const uint64_t free_u64 = static_cast<uint64_t>(free_bytes);
    if (free_u64 <= reserve_bytes) {
        return free_u64 / 2;
    }
    return free_u64 - reserve_bytes;
}

struct StreamBuildSchedule {
    uint32_t workers = 1;
    uint64_t memory_budget_bytes = 0;
    uint64_t per_worker_memory_cap_bytes = 0;
    bool allow_concurrent_cagra = false;
};

StreamBuildSchedule make_stream_schedule(const GraphBuildOptions& options,
                                         const GraphMetadata& metadata,
                                         size_t scalar_size)
{
    StreamBuildSchedule schedule;
    schedule.memory_budget_bytes = options.stream_memory_budget_bytes == 0
        ? current_gpu_memory_budget_bytes()
        : options.stream_memory_budget_bytes;
    schedule.per_worker_memory_cap_bytes = options.stream_per_worker_memory_cap_bytes;
    schedule.allow_concurrent_cagra = options.stream_allow_concurrent_cagra;

    uint32_t nonzero_labels = 0;
    uint64_t max_estimate = 1;
    for (uint32_t label = 0; label < metadata.label_count; ++label) {
        const uint32_t label_size = metadata.label_size[label];
        if (label_size == 0) {
            continue;
        }
        ++nonzero_labels;
        max_estimate = std::max(max_estimate,
                                estimate_stream_label_memory_bytes(label_size,
                                                                   options.dim,
                                                                   options.degree,
                                                                   scalar_size));
    }

    const uint32_t hardware_threads = std::max(1u, std::thread::hardware_concurrency());
    uint32_t requested_workers = options.stream_worker_count == 0 ? 1 : options.stream_worker_count;
    if (options.stream_auto_workers) {
        const uint64_t per_worker_floor =
            std::max<uint64_t>(max_estimate, static_cast<uint64_t>(512) << 20);
        const uint64_t budget_worker_count =
            std::max<uint64_t>(1, schedule.memory_budget_bytes / per_worker_floor);
        const uint32_t budget_workers =
            budget_worker_count > std::numeric_limits<uint32_t>::max()
                ? std::numeric_limits<uint32_t>::max()
                : static_cast<uint32_t>(budget_worker_count);
        requested_workers = std::max<uint32_t>(requested_workers, budget_workers);
    }
    schedule.workers = std::max<uint32_t>(1,
                                          std::min({requested_workers,
                                                    std::max<uint32_t>(1, nonzero_labels),
                                                    hardware_threads}));
    if (!schedule.allow_concurrent_cagra) {
        schedule.workers = std::max<uint32_t>(1, schedule.workers);
    }
    if (schedule.per_worker_memory_cap_bytes == 0 && schedule.workers > 0) {
        schedule.per_worker_memory_cap_bytes =
            std::max<uint64_t>(1, schedule.memory_budget_bytes / schedule.workers);
    }
    return schedule;
}

class MemoryPermitPool {
public:
    explicit MemoryPermitPool(uint64_t budget) : budget_(std::max<uint64_t>(1, budget)) {}

    uint64_t acquire(uint64_t requested)
    {
        const uint64_t charge = std::max<uint64_t>(1, std::min(requested, budget_));
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&] { return in_use_ <= budget_ && charge <= budget_ - in_use_; });
        in_use_ += charge;
        return charge;
    }

    void release(uint64_t charge)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        in_use_ = charge > in_use_ ? 0 : in_use_ - charge;
        cv_.notify_all();
    }

private:
    uint64_t budget_ = 1;
    uint64_t in_use_ = 0;
    std::mutex mutex_;
    std::condition_variable cv_;
};

class ScopedMemoryPermit {
public:
    ScopedMemoryPermit(MemoryPermitPool& pool, uint64_t requested) : pool_(&pool)
    {
        charge_ = pool_->acquire(requested);
    }

    ~ScopedMemoryPermit()
    {
        if (pool_ != nullptr) {
            pool_->release(charge_);
        }
    }

    ScopedMemoryPermit(const ScopedMemoryPermit&) = delete;
    ScopedMemoryPermit& operator=(const ScopedMemoryPermit&) = delete;

private:
    MemoryPermitPool* pool_ = nullptr;
    uint64_t charge_ = 0;
};

struct GraphBuildRuntimeContext {
    GroupLayout group_layout;
    GraphMetadata metadata;
    LowfreqPostingIndex lowfreq_index;
    std::string lowfreq_index_path;
    std::string graph_tmp;
};

GraphBuildRuntimeContext prepare_graph_build_context(const GraphBuildOptions& options,
                                                     const SparseLabelMatrix& data_labels)
{
    const std::string group_path = options.group_path.empty()
        ? default_group_path_from_graph_path(options.graph_path)
        : options.group_path;
    GraphBuildRuntimeContext context;
    context.group_layout = options.group_path.empty()
        ? build_identity_group_layout(data_labels.cols)
        : load_group_layout(group_path, data_labels.cols);
    GraphBuildIndexes indexes =
        build_graph_build_indexes(data_labels,
                                          options.specificity_threshold,
                                          context.group_layout);
    context.metadata = std::move(indexes.metadata);
    context.lowfreq_index = std::move(indexes.lowfreq_index);
    const std::filesystem::path graph_parent =
        std::filesystem::path{options.graph_path.c_str()}.parent_path();
    if (!graph_parent.empty()) {
        std::filesystem::create_directories(graph_parent);
    }

    if (options.group_path.empty()) {
        write_identity_groups(group_path, data_labels.cols);
    }
    context.lowfreq_index_path = options.lowfreq_index_path.empty()
        ? default_lowfreq_index_path_from_graph_path(options.graph_path)
        : options.lowfreq_index_path;
    context.graph_tmp = graph_tmp_path(options.graph_path);
    return context;
}

template <typename ScalarT, typename MatrixT>
std::vector<ScalarT> build_label_dataset(const MatrixT& base_vectors,
                                         const GraphMetadata& metadata,
                                         uint32_t label,
                                         int dim)
{
    const uint32_t label_size = metadata.label_size[label];
    const uint32_t row_begin = metadata.label_offset[label];
    std::vector<ScalarT> label_data(static_cast<size_t>(label_size) * dim);
    for (uint32_t local_row = 0; local_row < label_size; ++local_row) {
        const uint32_t sample = metadata.row_to_sample[row_begin + local_row];
        const ScalarT* src = base_vectors.row_ptr(sample);
        std::copy(src, src + dim, label_data.begin() + static_cast<size_t>(local_row) * dim);
    }
    return label_data;
}

std::vector<float> build_label_dataset_from_file(VectorFileReader& reader,
                                                 const GraphMetadata& metadata,
                                                 uint32_t label,
                                                 int dim)
{
    const uint32_t label_size = metadata.label_size[label];
    const uint32_t row_begin = metadata.label_offset[label];
    std::vector<float> label_data(static_cast<size_t>(label_size) * dim);
    for (uint32_t local_row = 0; local_row < label_size; ++local_row) {
        const uint32_t sample = metadata.row_to_sample[row_begin + local_row];
        reader.read_float_row(sample,
                             label_data.data() + static_cast<size_t>(local_row) * dim,
                             static_cast<uint32_t>(dim));
    }
    return label_data;
}

std::vector<uint8_t> build_uint8_label_dataset_from_file(VectorFileReader& reader,
                                                         const GraphMetadata& metadata,
                                                         uint32_t label,
                                                         int dim)
{
    const uint32_t label_size = metadata.label_size[label];
    const uint32_t row_begin = metadata.label_offset[label];
    std::vector<uint8_t> label_data(static_cast<size_t>(label_size) * dim);
    for (uint32_t local_row = 0; local_row < label_size; ++local_row) {
        const uint32_t sample = metadata.row_to_sample[row_begin + local_row];
        reader.read_uint8_row(sample,
                             label_data.data() + static_cast<size_t>(local_row) * dim,
                             static_cast<uint32_t>(dim));
    }
    return label_data;
}

template <typename ScalarT>
std::vector<uint32_t> build_cagra_graph_chunk(raft::resources& res,
                                              const std::vector<ScalarT>& label_data,
                                              uint32_t label_size,
                                              int dim,
                                              int degree,
                                              const std::string& ace_build_dir = {},
                                              bool ace_force_disk = false,
                                              bool guarantee_connectivity = true)
{
    auto dataset_view = raft::make_host_matrix_view<const ScalarT, int64_t>(
        label_data.data(), static_cast<int64_t>(label_size), static_cast<int64_t>(dim));
    cuvs::neighbors::cagra::index_params params;
    params.graph_degree = static_cast<size_t>(degree);
    params.intermediate_graph_degree = static_cast<size_t>(std::max(degree * 2, degree));
    params.attach_dataset_on_build = false;
    params.guarantee_connectivity = guarantee_connectivity;
    if (cagra_label_uses_ace(label_size)) {
        auto ace = cuvs::neighbors::cagra::graph_build_params::ace_params{};
        ace.npartitions = std::max<size_t>(2, (static_cast<size_t>(label_size) + 1'999'999) / 2'000'000);
        if (!ace_build_dir.empty()) {
            std::filesystem::create_directories(ace_build_dir);
            ace.build_dir = ace_build_dir;
        }
        ace.use_disk = ace_force_disk;
        std::printf("[graph-build] ACE rows=%u partitions=%lu force_disk=%d build_dir=%s\n",
                    label_size,
                    static_cast<unsigned long>(ace.npartitions),
                    ace.use_disk ? 1 : 0,
                    ace.build_dir.c_str());
        std::fflush(stdout);
        params.graph_build_params = ace;
    }

    auto index = cuvs::neighbors::cagra::build(res, params, dataset_view);
    return export_cagra_graph(res, index, label_size, degree);
}

uint64_t stream_label_charge_bytes(const StreamBuildSchedule& schedule, uint64_t estimate)
{
    if (schedule.per_worker_memory_cap_bytes > 0 &&
        estimate > schedule.per_worker_memory_cap_bytes) {
        return schedule.memory_budget_bytes;
    }
    return estimate;
}

uint32_t count_nonzero_labels(const GraphMetadata& metadata)
{
    uint32_t total_nonzero_labels = 0;
    for (uint32_t label = 0; label < metadata.label_count; ++label) {
        if (metadata.label_size[label] > 0) {
            ++total_nonzero_labels;
        }
    }
    return total_nonzero_labels;
}

void publish_lowfreq_index(const std::string& lowfreq_index_path,
                           const LowfreqPostingIndex& lowfreq_index)
{
    const std::filesystem::path lowfreq_parent =
        std::filesystem::path{lowfreq_index_path.c_str()}.parent_path();
    if (!lowfreq_parent.empty()) {
        std::filesystem::create_directories(lowfreq_parent);
    }
    const std::string lowfreq_tmp = lowfreq_tmp_path(lowfreq_index_path);
    save_lowfreq_posting_index(lowfreq_tmp, lowfreq_index);
    rename_replace(lowfreq_tmp, lowfreq_index_path);
}

std::string resolved_graph_file_path(const GraphBuildOptions& options)
{
    return options.graph_file_path.empty()
        ? default_graph_file_path_from_graph_path(options.graph_path)
        : options.graph_file_path;
}

void ensure_parent_directory(const std::string& path)
{
    const std::filesystem::path parent = std::filesystem::path{path.c_str()}.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
}

void publish_graph_build_sidecars(const GraphBuildOptions& options,
                                  const GraphMetadata& metadata,
                                  const GroupLayout& group_layout,
                                  uint64_t num_points,
                                  uint64_t graph_degree,
                                  const BuildSeedSidecars& sidecars)
{
    const std::string metadata_sidecar_path = resolved_metadata_sidecar_path(options);
    ensure_parent_directory(metadata_sidecar_path);
    save_metadata_sidecar(metadata_sidecar_path,
                          metadata,
                          num_points,
                          metadata_sidecar_row_limit_key(options, num_points),
                          metadata.total_rows,
                          graph_degree,
                          options.specificity_threshold,
                          group_layout);
    std::printf("[graph-build] wrote metadata sidecar %s\n", metadata_sidecar_path.c_str());

    const std::string seed_sidecar_path = resolved_seed_sidecar_path(options);
    if (!seed_sidecar_path.empty()) {
        if (sidecars.seed_local_id_by_label.size() != metadata.label_count) {
            fail_graph_builder("group seed sidecar vector size mismatch");
        }
        ensure_parent_directory(seed_sidecar_path);
        save_seed_sidecar(seed_sidecar_path,
                          metadata,
                          num_points,
                          graph_degree,
                          options.specificity_threshold,
                          sidecars.seed_local_id_by_label);
        std::printf("[graph-build] wrote seed sidecar %s\n", seed_sidecar_path.c_str());
    }

    const std::string original_label_seed_sidecar_path =
        resolved_original_label_seed_sidecar_path(options);
    if (!original_label_seed_sidecar_path.empty()) {
        if (sidecars.seed_by_original_label.size() != metadata.original_label_count) {
            fail_graph_builder("original-label seed sidecar vector size mismatch");
        }
        ensure_parent_directory(original_label_seed_sidecar_path);
        save_original_label_seed_sidecar(original_label_seed_sidecar_path,
                                         metadata,
                                         group_layout,
                                         num_points,
                                         graph_degree,
                                         options.specificity_threshold,
                                         sidecars.seed_by_original_label);
        std::printf("[graph-build] wrote original-label seed sidecar %s\n",
                    original_label_seed_sidecar_path.c_str());
    }
    std::fflush(stdout);
}

void finalize_streamed_graph_outputs(const GraphBuildOptions& options,
                                     const std::string& graph_tmp,
                                     const std::string& lowfreq_index_path,
                                     const LowfreqPostingIndex& lowfreq_index,
                                     const GraphMetadata& metadata,
                                     const GroupLayout& group_layout,
                                     uint64_t num_points,
                                     const BuildSeedSidecars& sidecars,
                                     uint32_t total_rows,
                                     int degree)
{
    const uintmax_t expected_size = 16 + static_cast<uintmax_t>(total_rows) *
                                      static_cast<uintmax_t>(degree) * sizeof(uint32_t);
    if (!file_size_matches(graph_tmp, expected_size)) {
        fail_graph_builder("graph file size mismatch");
    }

    const std::string graph_file_path = resolved_graph_file_path(options);
    ensure_parent_directory(graph_file_path);
    const std::string graph_tmp_file = graph_file_tmp_path(graph_file_path);
    go_fann::graph_file::convert_legacy_graph_file(graph_tmp, graph_tmp_file);
    rename_replace(graph_tmp_file, graph_file_path);
    remove_if_exists(graph_tmp);
    std::printf("Graph file written to %s\n", graph_file_path.c_str());

    publish_lowfreq_index(lowfreq_index_path, lowfreq_index);

    publish_graph_build_sidecars(options,
                                 metadata,
                                 group_layout,
                                 num_points,
                                 static_cast<uint64_t>(degree),
                                 sidecars);

    std::printf("Graph build complete.\n");
}

void finalize_direct_graph_file_outputs(const GraphBuildOptions& options,
                                        const std::string& graph_tmp_file,
                                        const std::string& lowfreq_index_path,
                                        const LowfreqPostingIndex& lowfreq_index,
                                        const GraphMetadata& metadata,
                                        const GroupLayout& group_layout,
                                        uint64_t num_points,
                                        const BuildSeedSidecars& sidecars,
                                        uint64_t total_rows,
                                        int degree)
{
    const auto header = go_fann::graph_file::read_graph_file_header(graph_tmp_file);
    if (header.rows != total_rows || header.degree != static_cast<uint32_t>(degree)) {
        fail_graph_builder("direct graph file header mismatch");
    }
    const std::string graph_file_path = resolved_graph_file_path(options);
    rename_replace(graph_tmp_file, graph_file_path);
    std::printf("Graph file written directly to %s\n", graph_file_path.c_str());

    publish_lowfreq_index(lowfreq_index_path, lowfreq_index);

    publish_graph_build_sidecars(options,
                                 metadata,
                                 group_layout,
                                 num_points,
                                 static_cast<uint64_t>(degree),
                                 sidecars);

    std::printf("Graph build complete.\n");
}

enum class StreamChunkBuilder {
    Cagra,
    Vamana,
};

const char* stream_chunk_builder_name(StreamChunkBuilder builder)
{
    return builder == StreamChunkBuilder::Cagra ? "CAGRA" : "Vamana";
}

template <typename ScalarT>
std::vector<uint32_t> build_vamana_graph_chunk(raft::resources& res,
                                               const std::vector<ScalarT>& label_data,
                                               uint32_t label_size,
                                               int dim,
                                               int degree);

template <typename ScalarT>
std::vector<uint32_t> build_stream_graph_chunk(StreamChunkBuilder builder,
                                               raft::resources& res,
                                               const std::vector<ScalarT>& label_data,
                                               uint32_t label_size,
                                               int dim,
                                               int degree,
                                               const std::string& ace_build_dir,
                                               bool ace_force_disk)
{
    if (builder == StreamChunkBuilder::Cagra) {
        return build_cagra_graph_chunk<ScalarT>(res,
                                                label_data,
                                                label_size,
                                                dim,
                                                degree,
                                                ace_build_dir,
                                                ace_force_disk);
    }
    return build_vamana_graph_chunk<ScalarT>(res, label_data, label_size, dim, degree);
}

template <typename ScalarT>
std::vector<uint32_t> build_parallel_stream_graph_chunk(StreamChunkBuilder builder,
                                                        raft::resources& res,
                                                        const std::vector<ScalarT>& label_data,
                                                        uint32_t label_size,
                                                        int dim,
                                                        int degree,
                                                        bool guarantee_connectivity,
                                                        const std::string& ace_build_dir,
                                                        bool ace_force_disk)
{
    if (builder == StreamChunkBuilder::Cagra) {
        return build_cagra_graph_chunk<ScalarT>(res,
                                                label_data,
                                                label_size,
                                                dim,
                                                degree,
                                                ace_build_dir,
                                                ace_force_disk,
                                                guarantee_connectivity);
    }
    return build_vamana_graph_chunk<ScalarT>(res, label_data, label_size, dim, degree);
}

bool stream_chunk_requires_serial_build(StreamChunkBuilder builder, uint32_t label_size)
{
    return builder == StreamChunkBuilder::Cagra && cagra_label_uses_ace(label_size);
}

uint64_t stream_label_charge_bytes(const StreamBuildSchedule& schedule,
                                   uint64_t estimate,
                                   StreamChunkBuilder builder,
                                   uint32_t label_size)
{
    if (stream_chunk_requires_serial_build(builder, label_size)) {
        return schedule.memory_budget_bytes;
    }
    return stream_label_charge_bytes(schedule, estimate);
}

template <typename ScalarT, typename LabelLoaderFactory>
int run_graph_builder_streamed_parallel(const GraphBuildOptions& options,
                                        const GraphMetadata& metadata,
                                        const GroupLayout& group_layout,
                                        uint64_t num_points,
                                        const std::string& graph_tmp,
                                        const std::string& lowfreq_index_path,
                                        const LowfreqPostingIndex& lowfreq_index,
                                        DenseVectorScalarType scalar_type,
                                        const char* base_description,
                                        StreamChunkBuilder chunk_builder,
                                        LabelLoaderFactory&& make_label_loader)
{
    const int degree = options.degree;
    const int dim = options.dim;
    const uint32_t total_nonzero_labels = count_nonzero_labels(metadata);
    const StreamBuildSchedule schedule = make_stream_schedule(options,
                                                              metadata,
                                                              scalar_size_for(scalar_type));

    std::filesystem::path chunk_dir = graph_chunk_dir_from_graph_path(options.graph_path);
    chunk_dir += ".tmp";
    std::error_code ec;
    std::filesystem::remove_all(chunk_dir, ec);
    ec.clear();
    std::filesystem::create_directories(chunk_dir, ec);
    if (ec) {
        fail_graph_builder("unable to create graph chunk directory " + chunk_dir.string() +
                           ": " + ec.message());
    }
    ScopedDirectoryCleanup chunk_cleanup(chunk_dir);

    (void)graph_tmp;

    std::printf("Using label-streamed %s build mode (%s, workers=%u, auto_workers=%d, "
                "memory_budget=%llu bytes, per_worker_cap=%llu bytes, concurrent_cagra=%d)\n",
                stream_chunk_builder_name(chunk_builder),
                base_description,
                schedule.workers,
                options.stream_auto_workers ? 1 : 0,
                static_cast<unsigned long long>(schedule.memory_budget_bytes),
                static_cast<unsigned long long>(schedule.per_worker_memory_cap_bytes),
                schedule.allow_concurrent_cagra ? 1 : 0);
    if (chunk_builder == StreamChunkBuilder::Cagra &&
        !schedule.allow_concurrent_cagra && schedule.workers > 1) {
        std::printf("[graph-build] CAGRA labels use the isolated serial phase; "
                    "worker concurrency requires stream_graph_allow_concurrent_cagra=1.\n");
    } else if (!schedule.allow_concurrent_cagra && schedule.workers > 1) {
        std::printf("[graph-build] %s calls are serialized; workers overlap label materialization and chunk writes only. "
                    "Set stream_graph_allow_concurrent_cagra=1 to opt into concurrent cuVS builds.\n",
                    stream_chunk_builder_name(chunk_builder));
    }
    std::fflush(stdout);

    std::atomic<uint32_t> completed_labels{0};
    std::atomic<uint32_t> completed_rows{0};
    std::atomic<uint32_t> failed_labels{0};
    std::atomic<bool> abort_build{false};
    std::mutex progress_mutex;
    std::mutex build_mutex;
    std::mutex error_mutex;
    std::mutex seed_mutex;
    std::exception_ptr first_error;
    BuildSeedSidecars sidecars = make_empty_seed_sidecars(metadata);
    MemoryPermitPool memory_pool(schedule.memory_budget_bytes);
    ChunkWriter chunk_writer(std::max<size_t>(2, schedule.workers * 2));
    auto build_start_time = std::chrono::high_resolution_clock::now();

    std::vector<uint32_t> isolated_labels;
    std::vector<uint32_t> worker_labels;
    isolated_labels.reserve(metadata.label_count);
    worker_labels.reserve(metadata.label_count);
    for (uint32_t label = 0; label < metadata.label_count; ++label) {
        const uint32_t label_size = metadata.label_size[label];
        if (label_size == 0) {
            continue;
        }
        const bool isolate_cagra =
            chunk_builder == StreamChunkBuilder::Cagra &&
            (!schedule.allow_concurrent_cagra || cagra_label_uses_ace(label_size));
        (isolate_cagra ? isolated_labels : worker_labels).push_back(label);
    }

    auto process_label = [&](uint32_t label,
                             uint32_t worker_id,
                             const char* phase,
                             auto& load_label_data,
                             raft::resources& res,
                             bool guarantee_connectivity) {
        const uint32_t label_size = metadata.label_size[label];
        const uint64_t estimate =
            estimate_stream_label_memory_bytes(label_size,
                                               dim,
                                               degree,
                                               scalar_size_for(scalar_type));
        const uint64_t charge =
            stream_label_charge_bytes(schedule, estimate, chunk_builder, label_size);
        try {
            ScopedMemoryPermit permit(memory_pool, charge);
            auto label_data = load_label_data(label);
            auto seed_update =
                compute_label_seed_update(metadata, group_layout, label, label_data, dim);
            {
                std::lock_guard<std::mutex> lock(seed_mutex);
                apply_label_seed_update(sidecars, seed_update);
            }
            auto chunk = build_parallel_stream_graph_chunk<ScalarT>(
                chunk_builder,
                res,
                label_data,
                label_size,
                dim,
                degree,
                guarantee_connectivity,
                options.cagra_ace_build_dir,
                options.cagra_ace_force_disk);

            const std::string chunk_path = chunk_file_path(chunk_dir.string(), label);
            chunk_writer.enqueue(chunk_path, std::move(chunk));

            const uint32_t done_labels = ++completed_labels;
            const uint32_t done_rows = completed_rows.fetch_add(label_size) + label_size;
            {
                std::lock_guard<std::mutex> lock(progress_mutex);
                print_build_progress(done_labels,
                                     total_nonzero_labels,
                                     done_rows,
                                     metadata.total_rows,
                                     label,
                                     label_size);
            }
            return true;
        } catch (const std::exception& e) {
            ++failed_labels;
            abort_build.store(true, std::memory_order_release);
            {
                std::lock_guard<std::mutex> lock(error_mutex);
                if (!first_error) {
                    first_error = std::current_exception();
                }
            }
            std::lock_guard<std::mutex> lock(progress_mutex);
            std::fprintf(stderr,
                         "\n[ERROR] stream graph label %u failed in %s worker %u: %s\n",
                         label,
                         phase,
                         worker_id,
                         e.what());
            return false;
        } catch (...) {
            ++failed_labels;
            abort_build.store(true, std::memory_order_release);
            {
                std::lock_guard<std::mutex> lock(error_mutex);
                if (!first_error) {
                    first_error = std::current_exception();
                }
            }
            std::lock_guard<std::mutex> lock(progress_mutex);
            std::fprintf(stderr,
                         "\n[ERROR] stream graph label %u failed in %s worker %u: unknown exception\n",
                         label,
                         phase,
                         worker_id);
            return false;
        }
    };

    if (!isolated_labels.empty()) {
        std::printf("[graph-build] isolated serial phase labels=%zu rows_threshold=%u\n",
                    isolated_labels.size(),
                    2'000'000u);
        std::fflush(stdout);
        raft::resources serial_res;
        auto load_label_data = make_label_loader();
        for (uint32_t label : isolated_labels) {
            if (abort_build.load(std::memory_order_acquire) ||
                !process_label(label,
                               0,
                               "isolated",
                               load_label_data,
                               serial_res,
                               true)) {
                break;
            }
        }
    }

    if (!worker_labels.empty() && !abort_build.load(std::memory_order_acquire)) {
        std::printf("[graph-build] worker phase labels=%zu workers=%u\n",
                    worker_labels.size(),
                    schedule.workers);
        std::fflush(stdout);
        shared_resources::n_threads = static_cast<int>(schedule.workers);
        for (uint32_t worker_id = 0; worker_id < schedule.workers; ++worker_id) {
            shared_resources::thread_id = static_cast<int>(worker_id);
            shared_resources::get_stream_from_global_pool();
        }
        shared_resources::configured_raft_resources main_res;
        std::atomic<uint32_t> next_worker_label{0};
        std::vector<std::thread> workers;
        workers.reserve(schedule.workers);
        for (uint32_t worker_id = 0; worker_id < schedule.workers; ++worker_id) {
            workers.emplace_back([&, worker_id] {
                shared_resources::thread_id = static_cast<int>(worker_id);
                shared_resources::n_threads = static_cast<int>(schedule.workers);
                auto worker_res = main_res;
                raft::resources& res = worker_res;
                auto load_label_data = make_label_loader();
                while (!abort_build.load(std::memory_order_acquire)) {
                    const uint32_t worker_label_index = next_worker_label.fetch_add(1);
                    if (worker_label_index >= worker_labels.size()) {
                        break;
                    }
                    const uint32_t label = worker_labels[worker_label_index];
                    if (schedule.allow_concurrent_cagra) {
                        process_label(label,
                                      worker_id,
                                      "concurrent",
                                      load_label_data,
                                      res,
                                      false);
                    } else {
                        std::lock_guard<std::mutex> lock(build_mutex);
                        raft::resources& serial_res = main_res;
                        process_label(label,
                                      worker_id,
                                      "serialized",
                                      load_label_data,
                                      serial_res,
                                      true);
                    }
                }
            });
        }
        for (std::thread& worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }
    chunk_writer.finish();
    if (failed_labels.load() > 0) {
        if (first_error) {
            std::rethrow_exception(first_error);
        }
        fail_graph_builder("one or more streamed graph labels failed");
    }

    auto build_end_time = std::chrono::high_resolution_clock::now();
    auto build_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        build_end_time - build_start_time);
    std::printf("\nGraph building time: %ld ms\n", build_duration.count());
    std::printf("Writing graph chunks directly in disk format...\n");

    const std::string graph_file_path = resolved_graph_file_path(options);
    ensure_parent_directory(graph_file_path);
    const std::string graph_tmp_file = graph_file_tmp_path(graph_file_path);
    remove_if_exists(graph_tmp_file);
    go_fann::graph_file::GraphFileSequentialWriter graph_writer(
        graph_tmp_file, metadata.total_rows, static_cast<uint32_t>(degree));
    for (uint32_t label = 0; label < metadata.label_count; ++label) {
        const uint32_t label_size = metadata.label_size[label];
        if (label_size == 0) {
            continue;
        }
        const std::string chunk_path = chunk_file_path(chunk_dir.string(), label);
        const uintmax_t expected_bytes = expected_chunk_bytes(label_size, degree);
        if (!file_size_matches(chunk_path, expected_bytes)) {
            fail_graph_builder("graph chunk size mismatch for label " + std::to_string(label));
        }
        append_chunk_file_to_graph_writer(graph_writer, chunk_path, label_size, degree);
    }
    graph_writer.close();

    finalize_direct_graph_file_outputs(options,
                                       graph_tmp_file,
                                       lowfreq_index_path,
                                       lowfreq_index,
                                       metadata,
                                       group_layout,
                                       num_points,
                                       sidecars,
                                       metadata.total_rows,
                                       degree);
    return 0;
}

template <typename ScalarT>
std::vector<uint32_t> build_vamana_graph_chunk(raft::resources& res,
                                               const std::vector<ScalarT>& label_data,
                                               uint32_t label_size,
                                               int dim,
                                               int degree)
{
    auto dataset = raft::make_device_matrix<ScalarT, int64_t>(
        res, static_cast<int64_t>(label_size), static_cast<int64_t>(dim));
    raft::copy(dataset.data_handle(),
               label_data.data(),
               label_data.size(),
               raft::resource::get_cuda_stream(res));
    raft::resource::sync_stream(res);
    auto dataset_view = raft::make_const_mdspan(dataset.view());
    cuvs::neighbors::vamana::index_params params;
    params.graph_degree = static_cast<uint32_t>(degree);
    params.visited_size = static_cast<uint32_t>(std::max(degree * 2, degree));
    params.queue_size = 127;

    auto index = cuvs::neighbors::vamana::build(res, params, dataset_view);
    auto graph_view = index.graph();
    const int64_t built_cols = graph_view.extent(1);
    std::vector<uint32_t> built_graph(static_cast<size_t>(label_size) * static_cast<size_t>(built_cols));
    raft::copy(built_graph.data(),
               graph_view.data_handle(),
               graph_view.size(),
               raft::resource::get_cuda_stream(res));
    raft::resource::sync_stream(res);

    std::vector<uint32_t> padded_graph(static_cast<size_t>(label_size) * degree, UINT32_MAX);
    for (uint32_t row = 0; row < label_size; ++row) {
        std::copy_n(built_graph.data() + static_cast<size_t>(row) * built_cols,
                    std::min<int64_t>(built_cols, degree),
                    padded_graph.data() + static_cast<size_t>(row) * degree);
    }
    return padded_graph;
}

template <typename ScalarT, typename MatrixT>
int run_graph_builder_streamed_cagra(const GraphBuildOptions& options,
                                     const MatrixT& base_vectors,
                                     const GraphMetadata& metadata,
                                     const GroupLayout& group_layout,
                                     uint64_t num_points,
                                     const std::string& graph_tmp,
                                     const std::string& lowfreq_index_path,
                                     const LowfreqPostingIndex& lowfreq_index)
{
    if (options.stream_worker_count > 1 || options.stream_auto_workers) {
        return run_graph_builder_streamed_parallel<ScalarT>(
            options,
            metadata,
            group_layout,
            num_points,
            graph_tmp,
            lowfreq_index_path,
            lowfreq_index,
            DenseVectorScalarType::Uint8,
            "host base matrix",
            StreamChunkBuilder::Cagra,
            [&]() {
                return [&base_vectors, &metadata, dim = options.dim](uint32_t label) {
                    return build_label_dataset<ScalarT>(base_vectors, metadata, label, dim);
                };
            });
    }

    const int degree = options.degree;
    const int dim = options.dim;

    const uint32_t total_nonzero_labels = count_nonzero_labels(metadata);

    (void)graph_tmp;
    const std::string graph_file_path = resolved_graph_file_path(options);
    ensure_parent_directory(graph_file_path);
    const std::string graph_tmp_file = graph_file_tmp_path(graph_file_path);
    remove_if_exists(graph_tmp_file);
    go_fann::graph_file::GraphFileSequentialWriter graph_writer(
        graph_tmp_file, metadata.total_rows, static_cast<uint32_t>(degree));

    raft::resources res;
    uint32_t completed_labels = 0;
    uint32_t completed_rows = 0;
    BuildSeedSidecars sidecars = make_empty_seed_sidecars(metadata);
    auto build_start_time = std::chrono::high_resolution_clock::now();

    std::printf("Using label-streamed CAGRA build mode (sequential labels, host base matrix)\n");
    for (uint32_t label = 0; label < metadata.label_count; ++label) {
        const uint32_t label_size = metadata.label_size[label];
        if (label_size == 0) {
            continue;
        }

        auto label_data = build_label_dataset<ScalarT>(base_vectors, metadata, label, dim);
        apply_label_seed_update(
            sidecars,
            compute_label_seed_update(metadata, group_layout, label, label_data, dim));
        auto chunk = build_cagra_graph_chunk<ScalarT>(res,
                                                      label_data,
                                                      label_size,
                                                      dim,
                                                      degree,
                                                      options.cagra_ace_build_dir,
                                                      options.cagra_ace_force_disk);
        graph_writer.append_rows(chunk.data(), label_size);

        ++completed_labels;
        completed_rows += label_size;
        print_build_progress(completed_labels,
                             total_nonzero_labels,
                             completed_rows,
                             metadata.total_rows,
                             label,
                             label_size);
    }
    graph_writer.close();

    auto build_end_time = std::chrono::high_resolution_clock::now();
    auto build_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        build_end_time - build_start_time);
    std::printf("\nGraph building time: %ld ms\n", build_duration.count());

    finalize_direct_graph_file_outputs(options,
                                       graph_tmp_file,
                                       lowfreq_index_path,
                                       lowfreq_index,
                                       metadata,
                                       group_layout,
                                       num_points,
                                       sidecars,
                                       metadata.total_rows,
                                       degree);
    return 0;
}

int run_graph_builder_streamed_cagra_file_float(const GraphBuildOptions& options,
                                                const VectorFileInfo& vector_file,
                                                const GraphMetadata& metadata,
                                                const GroupLayout& group_layout,
                                                uint64_t num_points,
                                                const std::string& graph_tmp,
                                                const std::string& lowfreq_index_path,
                                                const LowfreqPostingIndex& lowfreq_index)
{
    if (options.stream_worker_count > 1 || options.stream_auto_workers) {
        return run_graph_builder_streamed_parallel<float>(
            options,
            metadata,
            group_layout,
            num_points,
            graph_tmp,
            lowfreq_index_path,
            lowfreq_index,
            vector_file.scalar_type,
            "file-backed float base",
            options.builder == "vamana" ? StreamChunkBuilder::Vamana : StreamChunkBuilder::Cagra,
            [&]() {
                auto reader = std::make_shared<VectorFileReader>(vector_file);
                return [reader, &metadata, dim = options.dim](uint32_t label) {
                    return build_label_dataset_from_file(*reader, metadata, label, dim);
                };
            });
    }

    const int degree = options.degree;
    const int dim = options.dim;

    const uint32_t total_nonzero_labels = count_nonzero_labels(metadata);

    (void)graph_tmp;
    const std::string graph_file_path = resolved_graph_file_path(options);
    ensure_parent_directory(graph_file_path);
    const std::string graph_tmp_file = graph_file_tmp_path(graph_file_path);
    remove_if_exists(graph_tmp_file);
    go_fann::graph_file::GraphFileSequentialWriter graph_writer(
        graph_tmp_file, metadata.total_rows, static_cast<uint32_t>(degree));

    VectorFileReader reader(vector_file);
    raft::resources res;
    uint32_t completed_labels = 0;
    uint32_t completed_rows = 0;
    BuildSeedSidecars sidecars = make_empty_seed_sidecars(metadata);
    auto build_start_time = std::chrono::high_resolution_clock::now();

    const StreamChunkBuilder chunk_builder =
        options.builder == "vamana" ? StreamChunkBuilder::Vamana : StreamChunkBuilder::Cagra;

    std::printf("Using label-streamed %s build mode (sequential labels, file-backed float base)\n",
                stream_chunk_builder_name(chunk_builder));
    std::fflush(stdout);
    for (uint32_t label = 0; label < metadata.label_count; ++label) {
        const uint32_t label_size = metadata.label_size[label];
        if (label_size == 0) {
            continue;
        }

        auto label_data = build_label_dataset_from_file(reader, metadata, label, dim);
        apply_label_seed_update(
            sidecars,
            compute_label_seed_update(metadata, group_layout, label, label_data, dim));
        auto chunk = build_stream_graph_chunk<float>(chunk_builder,
                                                     res,
                                                     label_data,
                                                     label_size,
                                                     dim,
                                                     degree,
                                                     options.cagra_ace_build_dir,
                                                     options.cagra_ace_force_disk);
        graph_writer.append_rows(chunk.data(), label_size);

        ++completed_labels;
        completed_rows += label_size;
        print_build_progress(completed_labels,
                             total_nonzero_labels,
                             completed_rows,
                             metadata.total_rows,
                             label,
                             label_size);
    }
    graph_writer.close();

    auto build_end_time = std::chrono::high_resolution_clock::now();
    auto build_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        build_end_time - build_start_time);
    std::printf("\nGraph building time: %ld ms\n", build_duration.count());

    finalize_direct_graph_file_outputs(options,
                                       graph_tmp_file,
                                       lowfreq_index_path,
                                       lowfreq_index,
                                       metadata,
                                       group_layout,
                                       num_points,
                                       sidecars,
                                       metadata.total_rows,
                                       degree);
    return 0;
}

int run_graph_builder_streamed_cagra_file_uint8(const GraphBuildOptions& options,
                                                const VectorFileInfo& vector_file,
                                                const GraphMetadata& metadata,
                                                const GroupLayout& group_layout,
                                                uint64_t num_points,
                                                const std::string& graph_tmp,
                                                const std::string& lowfreq_index_path,
                                                const LowfreqPostingIndex& lowfreq_index)
{
    if (options.stream_worker_count > 1 || options.stream_auto_workers) {
        return run_graph_builder_streamed_parallel<uint8_t>(
            options,
            metadata,
            group_layout,
            num_points,
            graph_tmp,
            lowfreq_index_path,
            lowfreq_index,
            vector_file.scalar_type,
            "file-backed uint8 base",
            StreamChunkBuilder::Cagra,
            [&]() {
                auto reader = std::make_shared<VectorFileReader>(vector_file);
                return [reader, &metadata, dim = options.dim](uint32_t label) {
                    return build_uint8_label_dataset_from_file(*reader, metadata, label, dim);
                };
            });
    }

    const int degree = options.degree;
    const int dim = options.dim;

    const uint32_t total_nonzero_labels = count_nonzero_labels(metadata);

    (void)graph_tmp;
    const std::string graph_file_path = resolved_graph_file_path(options);
    ensure_parent_directory(graph_file_path);
    const std::string graph_tmp_file = graph_file_tmp_path(graph_file_path);
    remove_if_exists(graph_tmp_file);
    go_fann::graph_file::GraphFileSequentialWriter graph_writer(
        graph_tmp_file, metadata.total_rows, static_cast<uint32_t>(degree));

    VectorFileReader reader(vector_file);
    raft::resources res;
    uint32_t completed_labels = 0;
    uint32_t completed_rows = 0;
    BuildSeedSidecars sidecars = make_empty_seed_sidecars(metadata);
    auto build_start_time = std::chrono::high_resolution_clock::now();

    std::printf("Using label-streamed CAGRA build mode (sequential labels, file-backed uint8 base)\n");
    std::fflush(stdout);
    for (uint32_t label = 0; label < metadata.label_count; ++label) {
        const uint32_t label_size = metadata.label_size[label];
        if (label_size == 0) {
            continue;
        }

        const auto label_start_time = std::chrono::high_resolution_clock::now();
        std::printf("[graph-build] label=%u size=%u stage=materialize_uint8_from_disk begin\n",
                    label,
                    label_size);
        std::fflush(stdout);
        auto label_data = build_uint8_label_dataset_from_file(reader, metadata, label, dim);
        apply_label_seed_update(
            sidecars,
            compute_label_seed_update(metadata, group_layout, label, label_data, dim));
        const auto materialized_time = std::chrono::high_resolution_clock::now();
        std::printf("[graph-build] label=%u size=%u stage=materialize_uint8_from_disk ms=%ld\n",
                    label,
                    label_size,
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        materialized_time - label_start_time).count());
        std::printf("[graph-build] label=%u size=%u stage=cagra begin\n", label, label_size);
        std::fflush(stdout);
        auto chunk = build_cagra_graph_chunk<uint8_t>(res,
                                                      label_data,
                                                      label_size,
                                                      dim,
                                                      degree,
                                                      options.cagra_ace_build_dir,
                                                      options.cagra_ace_force_disk);
        const auto cagra_done_time = std::chrono::high_resolution_clock::now();
        std::printf("[graph-build] label=%u size=%u stage=cagra ms=%ld\n",
                    label,
                    label_size,
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        cagra_done_time - materialized_time).count());
        std::printf("[graph-build] label=%u size=%u stage=graph_append begin\n", label, label_size);
        std::fflush(stdout);
        graph_writer.append_rows(chunk.data(), label_size);
        const auto append_done_time = std::chrono::high_resolution_clock::now();
        std::printf("[graph-build] label=%u size=%u stage=graph_append ms=%ld total_label_ms=%ld\n",
                    label,
                    label_size,
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        append_done_time - cagra_done_time).count(),
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        append_done_time - label_start_time).count());
        std::fflush(stdout);

        ++completed_labels;
        completed_rows += label_size;
        print_build_progress(completed_labels,
                             total_nonzero_labels,
                             completed_rows,
                             metadata.total_rows,
                             label,
                             label_size);
    }
    graph_writer.close();

    auto build_end_time = std::chrono::high_resolution_clock::now();
    auto build_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        build_end_time - build_start_time);
    std::printf("\nGraph building time: %ld ms\n", build_duration.count());

    finalize_direct_graph_file_outputs(options,
                                       graph_tmp_file,
                                       lowfreq_index_path,
                                       lowfreq_index,
                                       metadata,
                                       group_layout,
                                       num_points,
                                       sidecars,
                                       metadata.total_rows,
                                       degree);
    return 0;
}

template <typename ScalarT, typename LabelLoaderFactory>
int run_graph_builder_seed_sidecars_only(const GraphBuildOptions& options,
                                         const GraphMetadata& metadata,
                                         const GroupLayout& group_layout,
                                         uint64_t num_points,
                                         DenseVectorScalarType scalar_type,
                                         const char* base_description,
                                         LabelLoaderFactory&& make_label_loader)
{
    const int dim = options.dim;
    const uint32_t total_nonzero_labels = count_nonzero_labels(metadata);
    const StreamBuildSchedule schedule = make_stream_schedule(options,
                                                              metadata,
                                                              scalar_size_for(scalar_type));
    std::printf("Using seed-sidecar-only graph builder mode (%s, workers=%u, "
                "auto_workers=%d, memory_budget=%llu bytes)\n",
                base_description,
                schedule.workers,
                options.stream_auto_workers ? 1 : 0,
                static_cast<unsigned long long>(schedule.memory_budget_bytes));
    std::fflush(stdout);

    std::atomic<uint32_t> next_label{0};
    std::atomic<uint32_t> completed_labels{0};
    std::atomic<uint32_t> completed_rows{0};
    std::atomic<uint32_t> failed_labels{0};
    std::atomic<bool> abort_build{false};
    std::mutex progress_mutex;
    std::mutex error_mutex;
    std::mutex seed_mutex;
    std::exception_ptr first_error;
    BuildSeedSidecars sidecars = make_empty_seed_sidecars(metadata);
    MemoryPermitPool memory_pool(schedule.memory_budget_bytes);

    auto build_start_time = std::chrono::high_resolution_clock::now();
    std::vector<std::thread> workers;
    workers.reserve(schedule.workers);
    for (uint32_t worker_id = 0; worker_id < schedule.workers; ++worker_id) {
        workers.emplace_back([&, worker_id] {
            auto load_label_data = make_label_loader();
            while (true) {
                if (abort_build.load(std::memory_order_acquire)) {
                    break;
                }
                const uint32_t label = next_label.fetch_add(1);
                if (label >= metadata.label_count) {
                    break;
                }
                if (abort_build.load(std::memory_order_acquire)) {
                    break;
                }
                const uint32_t label_size = metadata.label_size[label];
                if (label_size == 0) {
                    continue;
                }

                const uint64_t estimate =
                    estimate_stream_label_memory_bytes(label_size,
                                                       dim,
                                                       options.degree,
                                                       scalar_size_for(scalar_type));
                try {
                    const auto label_start_time = std::chrono::high_resolution_clock::now();
                    {
                        std::lock_guard<std::mutex> lock(progress_mutex);
                        std::printf("[graph-build] label=%u size=%u stage=seed_materialize begin\n",
                                    label,
                                    label_size);
                        std::fflush(stdout);
                    }
                    ScopedMemoryPermit permit(memory_pool, stream_label_charge_bytes(schedule, estimate));
                    auto label_data = load_label_data(label);
                    const auto materialized_time = std::chrono::high_resolution_clock::now();
                    {
                        std::lock_guard<std::mutex> lock(progress_mutex);
                        std::printf("[graph-build] label=%u size=%u stage=seed_materialize ms=%ld\n",
                                    label,
                                    label_size,
                                    std::chrono::duration_cast<std::chrono::milliseconds>(
                                        materialized_time - label_start_time).count());
                        std::fflush(stdout);
                    }
                    auto seed_update =
                        compute_label_seed_update(metadata, group_layout, label, label_data, dim);
                    const auto seed_done_time = std::chrono::high_resolution_clock::now();
                    {
                        std::lock_guard<std::mutex> lock(progress_mutex);
                        std::printf("[graph-build] label=%u size=%u stage=seed_compute ms=%ld\n",
                                    label,
                                    label_size,
                                    std::chrono::duration_cast<std::chrono::milliseconds>(
                                        seed_done_time - materialized_time).count());
                        std::fflush(stdout);
                    }
                    {
                        std::lock_guard<std::mutex> lock(seed_mutex);
                        apply_label_seed_update(sidecars, seed_update);
                    }

                    const uint32_t done_labels = ++completed_labels;
                    const uint32_t done_rows = completed_rows.fetch_add(label_size) + label_size;
                    {
                        std::lock_guard<std::mutex> lock(progress_mutex);
                        print_build_progress(done_labels,
                                             total_nonzero_labels,
                                             done_rows,
                                             metadata.total_rows,
                                             label,
                                             label_size);
                    }
                } catch (const std::exception& e) {
                    ++failed_labels;
                    abort_build.store(true, std::memory_order_release);
                    {
                        std::lock_guard<std::mutex> lock(error_mutex);
                        if (!first_error) {
                            first_error = std::current_exception();
                        }
                    }
                    std::lock_guard<std::mutex> lock(progress_mutex);
                    std::fprintf(stderr,
                                 "\n[ERROR] seed sidecar label %u failed on worker %u: %s\n",
                                 label,
                                 worker_id,
                                 e.what());
                } catch (...) {
                    ++failed_labels;
                    abort_build.store(true, std::memory_order_release);
                    {
                        std::lock_guard<std::mutex> lock(error_mutex);
                        if (!first_error) {
                            first_error = std::current_exception();
                        }
                    }
                    std::lock_guard<std::mutex> lock(progress_mutex);
                    std::fprintf(stderr,
                                 "\n[ERROR] seed sidecar label %u failed on worker %u: unknown exception\n",
                                 label,
                                 worker_id);
                }
            }
        });
    }
    for (std::thread& worker : workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    if (failed_labels.load() > 0) {
        if (first_error) {
            std::rethrow_exception(first_error);
        }
        fail_graph_builder("one or more seed sidecar labels failed");
    }

    auto build_end_time = std::chrono::high_resolution_clock::now();
    auto build_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        build_end_time - build_start_time);
    std::printf("\nSeed sidecar generation time: %ld ms\n", build_duration.count());

    publish_graph_build_sidecars(options,
                                 metadata,
                                 group_layout,
                                 num_points,
                                 static_cast<uint64_t>(options.degree),
                                 sidecars);
    std::printf("Seed sidecar generation complete.\n");
    return 0;
}

int run_graph_builder_file_float_seed_sidecars(const GraphBuildOptions& options,
                                               const VectorFileInfo& vector_file,
                                               const SparseLabelMatrix& data_labels)
{
    if (vector_file.rows != data_labels.rows) {
        fail_graph_builder("base vectors and data labels row count mismatch: vectors=" +
                           std::to_string(vector_file.rows) + ", labels=" +
                           std::to_string(data_labels.rows));
    }
    if (vector_file.cols != static_cast<uint32_t>(options.dim)) {
        fail_graph_builder("base vector dimension mismatch: file dim=" +
                           std::to_string(vector_file.cols) +
                           ", configured dim=" + std::to_string(options.dim));
    }
    if (!dense_vector_scalar_is_float(vector_file.scalar_type)) {
        fail_graph_builder("file-backed seed sidecar generation requires float vector payloads");
    }

    const auto context = prepare_graph_build_context(options, data_labels);
    return run_graph_builder_seed_sidecars_only<float>(
        options,
        context.metadata,
        context.group_layout,
        vector_file.rows,
        vector_file.scalar_type,
        "file-backed float base",
        [&]() {
            auto reader = std::make_shared<VectorFileReader>(vector_file);
            return [reader, &metadata = context.metadata, dim = options.dim](uint32_t label) {
                return build_label_dataset_from_file(*reader, metadata, label, dim);
            };
        });
}

int run_graph_builder_file_uint8_seed_sidecars(const GraphBuildOptions& options,
                                               const VectorFileInfo& vector_file,
                                               const SparseLabelMatrix& data_labels)
{
    if (vector_file.rows != data_labels.rows) {
        fail_graph_builder("base vectors and data labels row count mismatch: vectors=" +
                           std::to_string(vector_file.rows) + ", labels=" +
                           std::to_string(data_labels.rows));
    }
    if (vector_file.cols != static_cast<uint32_t>(options.dim)) {
        fail_graph_builder("base vector dimension mismatch: file dim=" +
                           std::to_string(vector_file.cols) +
                           ", configured dim=" + std::to_string(options.dim));
    }
    if (vector_file.scalar_type != DenseVectorScalarType::Uint8) {
        fail_graph_builder("file-backed uint8 seed sidecar generation requires uint8 vector payloads");
    }

    const auto context = prepare_graph_build_context(options, data_labels);
    return run_graph_builder_seed_sidecars_only<uint8_t>(
        options,
        context.metadata,
        context.group_layout,
        vector_file.rows,
        vector_file.scalar_type,
        "file-backed uint8 base",
        [&]() {
            auto reader = std::make_shared<VectorFileReader>(vector_file);
            return [reader, &metadata = context.metadata, dim = options.dim](uint32_t label) {
                return build_uint8_label_dataset_from_file(*reader, metadata, label, dim);
            };
        });
}

// Batch-write mode: build graph chunk from DEVICE data (thread-safe for parallel)
// This uses device matrices like graph runtime does, avoiding host->device transfers during build
template <typename ScalarT>
void build_cagra_graph_chunk_from_device(raft::resources& res,
                                          const rmm::device_uvector<ScalarT>& device_label_data,
                                          uint32_t label_size,
                                          int dim,
                                          int degree,
                                          uint32_t* host_graph_ptr,
                                          uint64_t offset_rows)
{
    auto dataset_view = raft::make_device_matrix_view<const ScalarT, int64_t>(
        device_label_data.data(), static_cast<int64_t>(label_size), static_cast<int64_t>(dim));
    cuvs::neighbors::cagra::index_params params;
    params.graph_degree = static_cast<size_t>(degree);
    params.intermediate_graph_degree = static_cast<size_t>(std::max(degree * 2, degree));
    params.attach_dataset_on_build = false;
    params.guarantee_connectivity = true;
    if (label_size >= 2'000'000) {
        auto ace = cuvs::neighbors::cagra::graph_build_params::ace_params{};
        ace.npartitions = std::max<size_t>(2, (static_cast<size_t>(label_size) + 1'999'999) / 2'000'000);
        params.graph_build_params = ace;
    }

    auto index = cuvs::neighbors::cagra::build(res, params, dataset_view);
    auto graph = export_cagra_graph(res, index, label_size, degree);
    std::copy(graph.begin(),
              graph.end(),
              host_graph_ptr + offset_rows * static_cast<uint64_t>(degree));
}

// Batch-write mode: build graph chunk from DEVICE MATRIX (exactly like graph runtime)
// Uses raft::device_matrix directly, matching graph runtime's pattern
template <typename ScalarT>
void build_cagra_graph_chunk_from_device_matrix(raft::resources& res,
                                                raft::device_matrix<ScalarT, int64_t>& filtered_dataset,
                                                int dim,
                                                int degree,
                                                uint32_t* host_graph_ptr,
                                                uint64_t offset_rows)
{
    auto dataset_view = raft::make_const_mdspan(filtered_dataset.view());
    const int64_t label_size = dataset_view.extent(0);
    cuvs::neighbors::cagra::index_params params;
    params.graph_degree = static_cast<size_t>(degree);
    params.intermediate_graph_degree = static_cast<size_t>(std::max(degree * 2, degree));
    params.attach_dataset_on_build = false;
    if (label_size >= 2'000'000) {
        auto ace = cuvs::neighbors::cagra::graph_build_params::ace_params{};
        ace.npartitions = std::max<size_t>(2, (static_cast<size_t>(label_size) + 1'999'999) / 2'000'000);
        params.graph_build_params = ace;
    }

    auto index = cuvs::neighbors::cagra::build(res, params, dataset_view);
    auto graph = export_cagra_graph(res,
                                    index,
                                    static_cast<uint32_t>(label_size),
                                    degree);
    std::copy(graph.begin(),
              graph.end(),
              host_graph_ptr + offset_rows * static_cast<uint64_t>(degree));
}

template <typename ScalarT>
void build_vamana_graph_chunk_direct(raft::resources& res,
                                     const std::vector<ScalarT>& label_data,
                                     uint32_t label_size,
                                     int dim,
                                     int degree,
                                     uint32_t* host_graph_ptr,
                                     uint64_t offset_rows)
{
    auto dataset_view = raft::make_host_matrix_view<const ScalarT, int64_t>(
        label_data.data(), static_cast<int64_t>(label_size), static_cast<int64_t>(dim));
    cuvs::neighbors::vamana::index_params params;
    params.graph_degree = static_cast<uint32_t>(degree);
    params.visited_size = static_cast<uint32_t>(std::max(degree * 2, degree));
    params.queue_size = 127;

    auto index = cuvs::neighbors::vamana::build(res, params, dataset_view);
    auto graph_view = index.graph();
    const int64_t built_cols = graph_view.extent(1);

    // Initialize target region with UINT32_MAX
    std::fill_n(host_graph_ptr + offset_rows * degree,
                static_cast<size_t>(label_size) * degree,
                UINT32_MAX);

    // Copy built graph directly to host matrix at correct offset
    std::vector<uint32_t> built_graph(static_cast<size_t>(label_size) * static_cast<size_t>(built_cols));
    raft::copy(built_graph.data(),
               graph_view.data_handle(),
               graph_view.size(),
               raft::resource::get_cuda_stream(res));
    raft::resource::sync_stream(res);

    // Copy with padding
    for (uint32_t row = 0; row < label_size; ++row) {
        std::copy_n(built_graph.data() + static_cast<size_t>(row) * built_cols,
                    std::min<int64_t>(built_cols, degree),
                    host_graph_ptr + (offset_rows + row) * degree);
    }
}

void save_host_graph_to_file(const std::string& path,
                               const std::vector<uint32_t>& host_graph,
                               uint64_t total_rows,
                               int degree)
{
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        fail_graph_builder("unable to open graph output " + path);
    }
    // Header: total_rows (8 bytes) + degree (8 bytes)
    uint64_t degree_u64 = static_cast<uint64_t>(degree);
    output.write(reinterpret_cast<const char*>(&total_rows), sizeof(uint64_t));
    output.write(reinterpret_cast<const char*>(&degree_u64), sizeof(uint64_t));

    // Graph data
    output.write(reinterpret_cast<const char*>(host_graph.data()),
                 static_cast<std::streamsize>(host_graph.size() * sizeof(uint32_t)));
    if (!output) {
        fail_graph_builder("failed to write graph payload to " + path);
    }
}

void save_host_graph_to_file(const std::string& path,
                             const uint32_t* host_graph,
                             uint64_t total_rows,
                             int degree)
{
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        fail_graph_builder("unable to open graph output " + path);
    }
    uint64_t degree_u64 = static_cast<uint64_t>(degree);
    output.write(reinterpret_cast<const char*>(&total_rows), sizeof(uint64_t));
    output.write(reinterpret_cast<const char*>(&degree_u64), sizeof(uint64_t));
    output.write(reinterpret_cast<const char*>(host_graph),
                 static_cast<std::streamsize>(total_rows * static_cast<uint64_t>(degree) * sizeof(uint32_t)));
    if (!output) {
        fail_graph_builder("failed to write graph payload to " + path);
    }
}

}

// Parallel batch-write mode: OpenMP parallel build with pre-allocated host graph matrix
// This is the fastest mode, similar to graph runtime implementation
// Key difference from sequential: base vectors are copied to device ONCE, then each label
// builds from device data using device matrices (avoiding per-label host->device transfers)
template <typename ScalarT, typename MatrixT>
int run_graph_builder_parallel(const GraphBuildOptions& options,
                               const MatrixT& base_vectors,
                               const SparseLabelMatrix& data_labels,
                               const GraphMetadata& metadata,
                               const GroupLayout& group_layout,
                               uint64_t num_points,
                               const std::string& graph_tmp,
                               const std::string& lowfreq_index_path,
                               const LowfreqPostingIndex& lowfreq_index)
{
    // Use 16 threads for large datasets (32 threads caused CUDA errors on yfcc10M)
    const int num_threads = 16;
    omp_set_num_threads(num_threads);

    // Pre-allocate host graph matrix
    const uint64_t total_rows = metadata.total_rows;
    const int degree = options.degree;
    const int dim = options.dim;
    std::vector<uint32_t> host_final_graph(static_cast<size_t>(total_rows) * degree, UINT32_MAX);

    // Count labels to build
    uint32_t total_nonzero_labels = 0;
    for (uint32_t label = 0; label < metadata.label_count; ++label) {
        if (metadata.label_size[label] > 0) {
            ++total_nonzero_labels;
        }
    }

    std::atomic<uint32_t> completed_labels{0};
    std::atomic<uint32_t> completed_rows{0};
    std::atomic<uint32_t> failed_labels{0};

    // ===== CRITICAL: Copy base vectors to device ONCE before parallel region =====
    // This matches graph runtime's approach - device data allows thread-safe cagra::build
    // IMPORTANT: Use configured_raft_resources FIRST to set up the shared memory pool
    std::printf("Copying base vectors to device (%u rows x %d dim)...\n",
                base_vectors.rows, dim);

    // Set up global stream pool for OpenMP threads (BEFORE parallel region)
    shared_resources::n_threads = num_threads;
    for (int i = 0; i < num_threads; ++i) {
        shared_resources::thread_id = i;
        shared_resources::get_stream_from_global_pool();  // pre-create streams
    }

    // Create the main configured_raft_resources FIRST - this sets up shared memory pool
    shared_resources::configured_raft_resources main_res;

    // Create device matrix (like graph runtime's raft::make_device_matrix)
    auto device_dataset = raft::make_device_matrix<ScalarT, int64_t>(
        main_res, static_cast<int64_t>(base_vectors.rows), static_cast<int64_t>(dim));

    // Copy row by row because vector loaders expose row pointers, not a matrix view.
    for (uint32_t row = 0; row < base_vectors.rows; ++row) {
        raft::copy(device_dataset.data_handle() + static_cast<size_t>(row) * dim,
                   base_vectors.row_ptr(row),
                   dim,
                   raft::resource::get_cuda_stream(main_res));
    }
    raft::resource::sync_stream(main_res);
    std::printf("Base vectors copied to device. Starting parallel build...\n");

    // Get raw pointer for use in parallel region
    const ScalarT* device_base_ptr = device_dataset.data_handle();

    auto build_start_time = std::chrono::high_resolution_clock::now();

    // Schedule one label at a time across OpenMP workers.
    #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 1)
    for (uint32_t label = 0; label < metadata.label_count; ++label) {
        const uint32_t label_size = metadata.label_size[label];
        if (label_size == 0) {
            continue;
        }

        // Bind this iteration to its worker-local resources.
        int thread_id = omp_get_thread_num();
        shared_resources::thread_id = thread_id;
        shared_resources::n_threads = num_threads;
        auto thread_res = main_res;
        cudaStream_t thread_stream = thread_res.get_sync_stream();

        try {
            // Build the filtered dataset on the device.
            const uint32_t row_begin = metadata.label_offset[label];

            // Allocate the device matrix for this label.
            auto filtered_dataset = raft::make_device_matrix<ScalarT, int64_t>(
                thread_res, static_cast<int64_t>(label_size), static_cast<int64_t>(dim));

            // Copy vectors for this label from device_base to filtered_dataset
            for (uint32_t local_row = 0; local_row < label_size; ++local_row) {
                const uint32_t sample = metadata.row_to_sample[row_begin + local_row];
                raft::copy_async(filtered_dataset.data_handle() + static_cast<size_t>(local_row) * dim,
                                 device_base_ptr + static_cast<size_t>(sample) * dim,
                                 dim,
                                 thread_stream);
            }
            raft::resource::sync_stream(thread_res);

            const uint64_t offset_rows = metadata.label_offset[label];
            if (options.builder == "cagra") {
                build_cagra_graph_chunk_from_device_matrix<ScalarT>(thread_res, filtered_dataset,
                                                                    dim, degree,
                                                                    host_final_graph.data(), offset_rows);
            } else {
                // Vamana still uses host data path for now.
                auto host_label_data = build_label_dataset<ScalarT>(base_vectors, metadata, label, dim);
                build_vamana_graph_chunk_direct<ScalarT>(thread_res, host_label_data, label_size,
                                                         dim, degree,
                                                         host_final_graph.data(), offset_rows);
            }

            ++completed_labels;
            completed_rows += label_size;

            #pragma omp critical
            {
                print_build_progress(completed_labels.load(), total_nonzero_labels,
                                     completed_rows.load(), total_rows,
                                     label, label_size);
            }
        } catch (const std::exception& e) {
            ++failed_labels;
            #pragma omp critical
            {
                std::fprintf(stderr, "\n[ERROR] Label %u failed: %s\n", label, e.what());
            }
        }
    }

    auto build_end_time = std::chrono::high_resolution_clock::now();
    auto build_duration = std::chrono::duration_cast<std::chrono::milliseconds>(build_end_time - build_start_time);
    std::printf("\nGraph building time: %ld ms\n", build_duration.count());

    if (failed_labels.load() > 0) {
        fail_graph_builder(std::to_string(failed_labels.load()) +
                           " labels failed in parallel graph build; refusing to publish graph");
    }

    BuildSeedSidecars sidecars =
        build_seed_sidecars_from_loaded_base(base_vectors, metadata, group_layout, dim);

    // Save graph to file (single write)
    std::printf("Saving graph to %s...\n", graph_tmp.c_str());
    save_host_graph_to_file(graph_tmp, host_final_graph, total_rows, degree);

    // Verify file size
    const uintmax_t expected_size = 16 + static_cast<uintmax_t>(total_rows) *
                                          static_cast<uintmax_t>(degree) * sizeof(uint32_t);
    if (!file_size_matches(graph_tmp, expected_size)) {
        fail_graph_builder("graph file size mismatch");
    }

    finalize_streamed_graph_outputs(options,
                                    graph_tmp,
                                    lowfreq_index_path,
                                    lowfreq_index,
                                    metadata,
                                    group_layout,
                                    num_points,
                                    sidecars,
                                    total_rows,
                                    degree);

    return 0;
}

template <typename ScalarT, typename MatrixT>
int run_graph_builder_parallel_exact(const GraphBuildOptions& options,
                                             const MatrixT& base_vectors,
                                             const GraphMetadata& metadata,
                                             const GroupLayout& group_layout,
                                             uint64_t num_points,
                                             const std::string& graph_tmp,
                                             const std::string& lowfreq_index_path,
                                             const LowfreqPostingIndex& lowfreq_index)
{
    if (options.cagra_build_worker_count == 0 ||
        options.cagra_build_worker_count > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
        fail_graph_builder("cagra_build_worker_count must be between1 and INT_MAX");
    }
    std::vector<uint32_t> build_labels;
    for (uint32_t label = 0; label < metadata.label_count; ++label) {
        if (metadata.label_size[label] > 0) {
            build_labels.push_back(label);
        }
    }
    // Keep output offsets in metadata order; only task execution order changes.
    std::sort(build_labels.begin(), build_labels.end(), [&](uint32_t lhs, uint32_t rhs) {
        const auto lhs_size = metadata.label_size[lhs];
        const auto rhs_size = metadata.label_size[rhs];
        return lhs_size != rhs_size ? lhs_size > rhs_size : lhs < rhs;
    });
    const uint32_t total_nonzero_labels = static_cast<uint32_t>(build_labels.size());
    const int num_threads = static_cast<int>(std::min({options.cagra_build_worker_count,
        std::max(1u, total_nonzero_labels), std::max(1u, std::thread::hardware_concurrency())}));
    struct WorkerStats {
        uint32_t labels = 0;
        uint64_t rows = 0;
        double wait_ms = 0;
        double gather_ms = 0;
        double build_ms = 0;
        double export_ms = 0;
    };
    std::vector<WorkerStats> worker_stats(num_threads);

    std::atomic<uint32_t> completed_labels{0};
    std::atomic<uint64_t> completed_rows{0};
    std::atomic<uint32_t> failed_labels{0};

    std::printf("Using graph runtime-faithful parallel batch-write mode (OpenMP, %d threads, schedule=dynamic,1)\n", num_threads);
    std::printf("Copying base vectors to device (%u rows x %d dim)...\n",
                base_vectors.rows,
                options.dim);

    shared_resources::n_threads = num_threads;
    for (int i = 0; i < num_threads; ++i) {
        shared_resources::thread_id = i;
        shared_resources::get_stream_from_global_pool();
    }

    shared_resources::configured_raft_resources res;
    auto dataset = raft::make_device_matrix<ScalarT, int64_t>(
        res, static_cast<int64_t>(base_vectors.rows), static_cast<int64_t>(options.dim));
    for (uint32_t row = 0; row < base_vectors.rows; ++row) {
        raft::copy(dataset.data_handle() + static_cast<uint64_t>(row) * static_cast<uint64_t>(options.dim),
                   base_vectors.row_ptr(row),
                   static_cast<size_t>(options.dim),
                   raft::resource::get_cuda_stream(res));
    }
    raft::resource::sync_stream(res);
    std::printf("Base vectors copied to device. Starting parallel build...\n");

    auto host_final_graph = raft::make_host_matrix<uint32_t, int64_t>(
        static_cast<int64_t>(metadata.total_rows), static_cast<int64_t>(options.degree));
    std::fill_n(host_final_graph.data_handle(),
                static_cast<size_t>(metadata.total_rows) * static_cast<size_t>(options.degree),
                UINT32_MAX);

    // Account for the resident dataset first. Reuse the streamed builder's
    // conservative workspace estimate to bound concurrent CAGRA allocations.
    const uint64_t memory_budget = current_gpu_memory_budget_bytes();
    MemoryPermitPool memory_pool(memory_budget);
    std::printf("[graph-build-schedule] requested_workers=%u workers=%d labels=%u "
                "memory_budget_bytes=%llu order=size_descending\n",
                options.cagra_build_worker_count, num_threads, total_nonzero_labels,
                static_cast<unsigned long long>(memory_budget));
    using Clock = std::chrono::steady_clock;
    auto elapsed_ms = [](Clock::time_point from, Clock::time_point to) {
        return std::chrono::duration<double, std::milli>(to - from).count();
    };
    auto cagra_start_time = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 1)
    for (uint32_t task = 0; task < total_nonzero_labels; ++task) {
        const uint32_t label = build_labels[task];
        const uint32_t label_size = metadata.label_size[label];
        const int thread_id = omp_get_thread_num();
        auto& stats = worker_stats[thread_id];
        shared_resources::thread_id = thread_id;
        shared_resources::n_threads = num_threads;
        try {
            const auto wait_start = Clock::now();
            // Very large labels retain exclusive access to the workspace budget.
            const uint64_t estimate = label_size >= 2000000 ? memory_budget :
                estimate_stream_label_memory_bytes(label_size, options.dim, options.degree, sizeof(ScalarT));
            ScopedMemoryPermit permit(memory_pool, estimate);
            const auto gather_start = Clock::now();
            stats.wait_ms += elapsed_ms(wait_start, gather_start);
            auto thread_resources = res;
            cudaStream_t thread_stream = thread_resources.get_sync_stream();
            auto filtered_dataset = raft::make_device_matrix<ScalarT, int64_t>(
                thread_resources, static_cast<int64_t>(label_size), static_cast<int64_t>(options.dim));
            raft::resource::sync_stream(thread_resources);

            const uint32_t row_begin = metadata.label_offset[label];
            for (uint64_t local_row = 0; local_row < static_cast<uint64_t>(label_size); ++local_row) {
                const uint32_t sample = metadata.row_to_sample[row_begin + static_cast<uint32_t>(local_row)];
                raft::copy_async(filtered_dataset.data_handle() + local_row * static_cast<uint64_t>(options.dim),
                                 dataset.data_handle() + static_cast<uint64_t>(sample) * static_cast<uint64_t>(options.dim),
                                 static_cast<size_t>(options.dim),
                                 thread_stream);
            }
            raft::resource::sync_stream(thread_resources);

            const auto build_start = Clock::now();
            stats.gather_ms += elapsed_ms(gather_start, build_start);
            cuvs::neighbors::cagra::index_params index_params;
            index_params.intermediate_graph_degree = static_cast<size_t>(options.degree * 2);
            index_params.graph_degree = static_cast<size_t>(options.degree);
            index_params.attach_dataset_on_build = false;

            auto index = cuvs::neighbors::cagra::build(
                thread_resources, index_params, raft::make_const_mdspan(filtered_dataset.view()));
            raft::resource::sync_stream(thread_resources);
            const auto export_start = Clock::now();
            stats.build_ms += elapsed_ms(build_start, export_start);
            auto graph = export_cagra_graph(thread_resources,
                                            index,
                                            label_size,
                                            options.degree);
            std::copy(graph.begin(),
                      graph.end(),
                      host_final_graph.data_handle() +
                          static_cast<uint64_t>(metadata.label_offset[label]) *
                              static_cast<uint64_t>(options.degree));
            stats.export_ms += elapsed_ms(export_start, Clock::now());
            ++stats.labels;
            stats.rows += label_size;
            #pragma omp critical
            {
                completed_rows += label_size;
                ++completed_labels;
                print_build_progress(completed_labels.load(), total_nonzero_labels,
                                     static_cast<uint32_t>(std::min<uint64_t>(completed_rows.load(), metadata.total_rows)),
                                     metadata.total_rows, label, label_size);
            }
        } catch (const std::exception& e) {
            ++failed_labels;
            #pragma omp critical
            {
                std::fprintf(stderr, "\n[ERROR] Label %u failed: %s\n", label, e.what());
            }
        }
    }
    auto cagra_end_time = std::chrono::high_resolution_clock::now();
    auto cagra_duration = std::chrono::duration_cast<std::chrono::milliseconds>(cagra_end_time - cagra_start_time);
    std::printf("\nGraph building time: %ld ms\n", cagra_duration.count());

    for (int worker = 0; worker < num_threads; ++worker) {
        const auto& stats = worker_stats[worker];
        std::printf("[graph-build-worker] worker=%d labels=%u rows=%llu "
                    "wait_ms=%.3f gather_ms=%.3f build_ms=%.3f export_ms=%.3f\n",
                    worker, stats.labels, static_cast<unsigned long long>(stats.rows),
                    stats.wait_ms, stats.gather_ms, stats.build_ms, stats.export_ms);
    }

    if (failed_labels.load() > 0) {
        fail_graph_builder(std::to_string(failed_labels.load()) +
                           " labels failed in graph runtime parallel graph build; refusing to publish graph");
    }

    BuildSeedSidecars sidecars =
        build_seed_sidecars_from_loaded_base(base_vectors, metadata, group_layout, options.dim);

    std::printf("Saving graph to %s...\n", graph_tmp.c_str());
    save_host_graph_to_file(graph_tmp, host_final_graph.data_handle(), metadata.total_rows, options.degree);

    const uintmax_t expected_size = 16 + static_cast<uintmax_t>(metadata.total_rows) *
                                          static_cast<uintmax_t>(options.degree) * sizeof(uint32_t);
    if (!file_size_matches(graph_tmp, expected_size)) {
        fail_graph_builder("graph file size mismatch");
    }

    finalize_streamed_graph_outputs(options,
                                    graph_tmp,
                                    lowfreq_index_path,
                                    lowfreq_index,
                                    metadata,
                                    group_layout,
                                    num_points,
                                    sidecars,
                                    metadata.total_rows,
                                    options.degree);

    return 0;
}

template <typename MatrixT>
void validate_loaded_base_vectors(const MatrixT& base_vectors,
                                  const DenseVectorMetadata& metadata)
{
    if (base_vectors.rows != metadata.rows || base_vectors.cols != metadata.cols) {
        fail_graph_builder("loaded base vector shape does not match inspected header for " +
                           metadata.path + ": inspected " + std::to_string(metadata.rows) + "x" +
                           std::to_string(metadata.cols) + ", loaded " +
                           std::to_string(base_vectors.rows) + "x" +
                           std::to_string(base_vectors.cols));
    }
}

template <typename ScalarT, typename MatrixT>
int run_graph_builder_typed(const GraphBuildOptions& options,
                            const MatrixT& base_vectors,
                            const SparseLabelMatrix& data_labels)
{
    if (base_vectors.rows != data_labels.rows) {
        fail_graph_builder("base vectors and data labels row count mismatch: vectors=" +
                           std::to_string(base_vectors.rows) + ", labels=" +
                           std::to_string(data_labels.rows));
    }
    if (base_vectors.cols != static_cast<uint32_t>(options.dim)) {
        fail_graph_builder("base vector dimension mismatch: file dim=" +
                           std::to_string(base_vectors.cols) +
                           ", configured dim=" + std::to_string(options.dim));
    }

    const auto context = prepare_graph_build_context(options, data_labels);

    if (options.builder == "cagra") {
        return run_graph_builder_parallel_exact<ScalarT>(options,
                                                                 base_vectors,
                                                                 context.metadata,
                                                                 context.group_layout,
                                                                 base_vectors.rows,
                                                                 context.graph_tmp,
                                                                 context.lowfreq_index_path,
                                                                 context.lowfreq_index);
    }
    if (options.builder == "cagra_stream") {
        return run_graph_builder_streamed_cagra<ScalarT>(options,
                                                         base_vectors,
                                                         context.metadata,
                                                         context.group_layout,
                                                         base_vectors.rows,
                                                         context.graph_tmp,
                                                         context.lowfreq_index_path,
                                                         context.lowfreq_index);
    }

    std::printf("Using parallel batch-write mode (OpenMP)\n");
    return run_graph_builder_parallel<ScalarT>(options,
                                               base_vectors,
                                               data_labels,
                                               context.metadata,
                                               context.group_layout,
                                               base_vectors.rows,
                                               context.graph_tmp,
                                               context.lowfreq_index_path,
                                               context.lowfreq_index);
}

int run_graph_builder_file_float_streamed(const GraphBuildOptions& options,
                                          const VectorFileInfo& vector_file,
                                          const SparseLabelMatrix& data_labels)
{
    if (vector_file.rows != data_labels.rows) {
        fail_graph_builder("base vectors and data labels row count mismatch: vectors=" +
                           std::to_string(vector_file.rows) + ", labels=" +
                           std::to_string(data_labels.rows));
    }
    if (vector_file.cols != static_cast<uint32_t>(options.dim)) {
        fail_graph_builder("base vector dimension mismatch: file dim=" +
                           std::to_string(vector_file.cols) +
                           ", configured dim=" + std::to_string(options.dim));
    }
    if (!dense_vector_scalar_is_float(vector_file.scalar_type)) {
        fail_graph_builder("file-backed streamed graph build requires float vector payloads");
    }

    const auto context = prepare_graph_build_context(options, data_labels);
    return run_graph_builder_streamed_cagra_file_float(options,
                                                       vector_file,
                                                       context.metadata,
                                                       context.group_layout,
                                                       vector_file.rows,
                                                       context.graph_tmp,
                                                       context.lowfreq_index_path,
                                                       context.lowfreq_index);
}

int run_graph_builder_file_uint8_streamed(const GraphBuildOptions& options,
                                          const VectorFileInfo& vector_file,
                                          const SparseLabelMatrix& data_labels)
{
    if (vector_file.rows != data_labels.rows) {
        fail_graph_builder("base vectors and data labels row count mismatch: vectors=" +
                           std::to_string(vector_file.rows) + ", labels=" +
                           std::to_string(data_labels.rows));
    }
    if (vector_file.cols != static_cast<uint32_t>(options.dim)) {
        fail_graph_builder("base vector dimension mismatch: file dim=" +
                           std::to_string(vector_file.cols) +
                           ", configured dim=" + std::to_string(options.dim));
    }
    if (vector_file.scalar_type != DenseVectorScalarType::Uint8) {
        fail_graph_builder("file-backed uint8 streamed graph build requires uint8 vector payloads");
    }

    const auto context = prepare_graph_build_context(options, data_labels);
    return run_graph_builder_streamed_cagra_file_uint8(options,
                                                       vector_file,
                                                       context.metadata,
                                                       context.group_layout,
                                                       vector_file.rows,
                                                       context.graph_tmp,
                                                       context.lowfreq_index_path,
                                                       context.lowfreq_index);
}

int run_graph_builder(const GraphBuildOptions& options)
{
    if (options.builder != "cagra" && options.builder != "vamana" &&
        options.builder != "cagra_stream" && options.builder != "seed_sidecars") {
        fail_graph_builder("unsupported builder " + options.builder);
    }
    if (options.base_vector_path.empty() || options.data_label_spmat_path.empty() ||
        options.graph_path.empty()) {
        fail_graph_builder("missing required input path");
    }
    const auto base_vector_metadata = inspect_dense_vector_file(options.base_vector_path, options.row_limit);
    const auto data_labels = load_spmat_labels(options.data_label_spmat_path, options.row_limit);
    if (base_vector_metadata.rows != data_labels.rows) {
        fail_graph_builder("base vectors and data labels row count mismatch: vectors=" +
                           std::to_string(base_vector_metadata.rows) + ", labels=" +
                           std::to_string(data_labels.rows));
    }
    if (base_vector_metadata.cols != static_cast<uint32_t>(options.dim)) {
        fail_graph_builder("base vector dimension mismatch: file dim=" +
                           std::to_string(base_vector_metadata.cols) +
                           ", configured dim=" + std::to_string(options.dim));
    }

    if (base_vector_metadata.scalar_type == DenseVectorScalarType::Uint8) {
        if (options.builder == "seed_sidecars") {
            const auto base_vector_file =
                inspect_vector_file(options.base_vector_path, options.row_limit);
            return run_graph_builder_file_uint8_seed_sidecars(options, base_vector_file, data_labels);
        }
        if (options.builder == "cagra_stream") {
            const auto base_vector_file =
                inspect_vector_file(options.base_vector_path, options.row_limit);
            std::printf("Using file-backed uint8 base loader for streamed graph build: %s\n",
                        options.base_vector_path.c_str());
            std::fflush(stdout);
            return run_graph_builder_file_uint8_streamed(options, base_vector_file, data_labels);
        }
        const auto base_vectors = load_uint8_matrix_auto(options.base_vector_path, options.row_limit);
        validate_loaded_base_vectors(base_vectors, base_vector_metadata);
        return run_graph_builder_typed<uint8_t>(options, base_vectors, data_labels);
    }
    if (dense_vector_scalar_is_float(base_vector_metadata.scalar_type)) {
        if (options.builder == "seed_sidecars") {
            const auto base_vector_file =
                inspect_vector_file(options.base_vector_path, options.row_limit);
            return run_graph_builder_file_float_seed_sidecars(options, base_vector_file, data_labels);
        }
        if (options.builder == "cagra_stream" || options.builder == "vamana") {
            const auto base_vector_file =
                inspect_vector_file(options.base_vector_path, options.row_limit);
            return run_graph_builder_file_float_streamed(options, base_vector_file, data_labels);
        }
        const auto base_vectors = load_float_matrix_auto(options.base_vector_path, options.row_limit);
        validate_loaded_base_vectors(base_vectors, base_vector_metadata);
        return run_graph_builder_typed<float>(options, base_vectors, data_labels);
    }

    fail_graph_builder("unsupported dense vector scalar type for " + base_vector_metadata.path);
}

}
