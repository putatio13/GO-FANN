#include <cuvs/cluster/kmeans.hpp>
#include <cuvs/distance/distance.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/matrix/sample_rows.cuh>
#include <raft/random/rng_state.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include "ivf_rabitq/gpu_index/ivf_gpu.cuh"

#include <cuda_runtime.h>

#include "vector_reader.h"
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace go_fann_quantization;
enum class BuildMode : uint8_t {
  Batch,
  Streaming,
};

struct Options {
  std::string base_vector_path;
  std::string output_path;
  std::vector<std::string> output_paths;
  uint32_t row_limit = 0;
  uint32_t expected_dim = 0;
  uint64_t seed = 137;
  uint32_t n_lists = 1024;
  uint32_t bits_per_dim = 1;
  std::vector<uint32_t> bits_per_dim_list;
  uint32_t kmeans_iters = 20;
  uint32_t max_train_points_per_cluster = 256;
  bool fast_quantize = true;
  int device = -1;
  BuildMode build_mode = BuildMode::Batch;
  bool build_mode_explicit = false;
  uint32_t chunk_rows = 0;
  bool chunk_rows_explicit = false;
};

constexpr uint32_t kDefaultStreamingChunkRows = 8192;

const char* build_mode_name(BuildMode mode)
{
  switch (mode) {
    case BuildMode::Batch: return "batch";
    case BuildMode::Streaming: return "streaming";
  }
  return "unknown";
}

BuildMode parse_build_mode(const std::string& value)
{
  if (value == "batch") { return BuildMode::Batch; }
  if (value == "streaming") { return BuildMode::Streaming; }
  fail("--build-mode must be one of: batch, streaming");
}

void print_usage(const char* argv0)
{
  std::cerr
    << "Usage: " << argv0 << " --base-vector-path=PATH --output=PATH [options]\n"
    << "\n"
    << "Options:\n"
    << "  Supported inputs: .bvecs/.u8bin/.ibin/.bin uint8, .fbin/.fvecs float32, or .hvecs float16\n"
    << "  --row-limit=N                         Build from the first N vectors (default: all)\n"
    << "  --dim=N                               Validate input dimension\n"
    << "  --seed=N                              Training-row sampling seed (default: 137)\n"
    << "  --n-lists=N                           Number of RabitQ anchors/clusters (default: 1024)\n"
    << "  --bits-per-dim=N                      Total RabitQ bits per dimension, 1..9 (default: 1)\n"
    << "  --bits-per-dim-list=A,B,...           Build multiple bit-width artifacts from one kmeans fit\n"
    << "  --output-list=A,B,...                 Output paths for --bits-per-dim-list (same count)\n"
    << "                                        --output may also be comma-separated in multi-bit mode\n"
    << "  --kmeans-iters=N                      Balanced kmeans iterations (default: 20)\n"
    << "  --max-train-points-per-cluster=N      Training sample cap per cluster (default: 256)\n"
    << "  --build-mode=batch|streaming          Build mode (default: batch)\n"
    << "  --streaming                           Alias for --build-mode=streaming\n"
    << "  --chunk-rows=N                        Streaming device row cap (default: 8192)\n"
    << "  --max-device-rows=N                   Alias for --chunk-rows=N\n"
    << "  --fast-quantize=0|1                   Use fast quantization path (default: 1)\n"
    << "  --device=N                            CUDA device id\n";
}

Options parse_args(int argc, char** argv)
{
  Options opts;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    auto value_after = [&](const char* prefix) -> std::string {
      return arg.substr(std::strlen(prefix));
    };

    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    } else if (arg.rfind("--base-vector-path=", 0) == 0) {
      opts.base_vector_path = value_after("--base-vector-path=");
    } else if (arg.rfind("--output=", 0) == 0) {
      opts.output_path = value_after("--output=");
    } else if (arg.rfind("--output-list=", 0) == 0) {
      opts.output_paths = split_csv(value_after("--output-list="), "--output-list");
    } else if (arg.rfind("--row-limit=", 0) == 0) {
      opts.row_limit = parse_u32(value_after("--row-limit="), "--row-limit");
    } else if (arg.rfind("--dim=", 0) == 0) {
      opts.expected_dim = parse_u32(value_after("--dim="), "--dim");
    } else if (arg.rfind("--seed=", 0) == 0) {
      opts.seed = parse_u64(value_after("--seed="), "--seed");
    } else if (arg.rfind("--n-lists=", 0) == 0) {
      opts.n_lists = parse_u32(value_after("--n-lists="), "--n-lists");
    } else if (arg.rfind("--bits-per-dim=", 0) == 0) {
      opts.bits_per_dim = parse_u32(value_after("--bits-per-dim="), "--bits-per-dim");
    } else if (arg.rfind("--bits-per-dim-list=", 0) == 0) {
      opts.bits_per_dim_list =
        parse_u32_list(value_after("--bits-per-dim-list="), "--bits-per-dim-list");
    } else if (arg.rfind("--kmeans-iters=", 0) == 0) {
      opts.kmeans_iters = parse_u32(value_after("--kmeans-iters="), "--kmeans-iters");
    } else if (arg.rfind("--max-train-points-per-cluster=", 0) == 0) {
      opts.max_train_points_per_cluster =
        parse_u32(value_after("--max-train-points-per-cluster="),
                  "--max-train-points-per-cluster");
    } else if (arg.rfind("--build-mode=", 0) == 0) {
      opts.build_mode = parse_build_mode(value_after("--build-mode="));
      opts.build_mode_explicit = true;
    } else if (arg == "--streaming") {
      opts.build_mode = BuildMode::Streaming;
      opts.build_mode_explicit = true;
    } else if (arg.rfind("--chunk-rows=", 0) == 0) {
      opts.chunk_rows = parse_u32(value_after("--chunk-rows="), "--chunk-rows");
      opts.chunk_rows_explicit = true;
    } else if (arg.rfind("--max-device-rows=", 0) == 0) {
      opts.chunk_rows = parse_u32(value_after("--max-device-rows="), "--max-device-rows");
      opts.chunk_rows_explicit = true;
    } else if (arg.rfind("--fast-quantize=", 0) == 0) {
      opts.fast_quantize = parse_bool(value_after("--fast-quantize="), "--fast-quantize");
    } else if (arg.rfind("--device=", 0) == 0) {
      opts.device = static_cast<int>(parse_u32(value_after("--device="), "--device"));
    } else {
      fail("unknown argument: " + arg);
    }
  }

  if (opts.base_vector_path.empty()) { fail("--base-vector-path is required"); }
  if (opts.output_path.empty() && opts.output_paths.empty()) {
    fail("--output or --output-list is required");
  }
  if (opts.bits_per_dim_list.empty()) { opts.bits_per_dim_list.push_back(opts.bits_per_dim); }
  if (opts.output_paths.empty()) {
    opts.output_paths = split_csv(opts.output_path, "--output");
  }
  if (opts.bits_per_dim_list.size() != opts.output_paths.size()) {
    fail("--bits-per-dim-list and output path list must have the same item count");
  }
  if (opts.n_lists == 0) { fail("--n-lists must be positive"); }
  for (uint32_t bits : opts.bits_per_dim_list) {
    if (bits < 1 || bits > 9) {
      fail("--bits-per-dim/--bits-per-dim-list values must be in [1, 9]");
    }
  }
  opts.bits_per_dim = opts.bits_per_dim_list.front();
  opts.output_path = opts.output_paths.front();
  if (opts.kmeans_iters == 0) { fail("--kmeans-iters must be positive"); }
  if (opts.max_train_points_per_cluster == 0) {
    fail("--max-train-points-per-cluster must be positive");
  }
  if (opts.chunk_rows_explicit && opts.chunk_rows == 0) {
    fail("--chunk-rows/--max-device-rows must be positive");
  }
  if (opts.chunk_rows_explicit) {
    if (opts.build_mode_explicit && opts.build_mode == BuildMode::Batch) {
      fail("--chunk-rows/--max-device-rows require --build-mode=streaming");
    }
    opts.build_mode = BuildMode::Streaming;
  }
  if (opts.build_mode == BuildMode::Streaming && opts.chunk_rows == 0) {
    opts.chunk_rows = kDefaultStreamingChunkRows;
  }
  return opts;
}

__global__ void u8_to_float_kernel(const uint8_t* in, float* out, size_t count)
{
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < count) { out[idx] = static_cast<float>(in[idx]); }
}

__global__ void pad_rows_kernel(const float* in,
                                float* out,
                                size_t rows,
                                size_t dim,
                                size_t padded_dim)
{
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t total = rows * padded_dim;
  if (idx >= total) { return; }
  const size_t col = idx % padded_dim;
  const size_t row = idx / padded_dim;
  out[idx] = col < dim ? in[row * dim + col] : 0.0f;
}

void upload_uint8_as_float(raft::resources const& handle,
                           const uint8_t* input,
                           size_t count,
                           raft::device_matrix<float, int64_t>& output)
{
  auto stream = raft::resource::get_cuda_stream(handle);
  rmm::device_uvector<uint8_t> d_u8(count, stream);
  raft::copy(d_u8.data(), input, count, stream);

  constexpr int block = 256;
  const int grid = static_cast<int>((count + block - 1) / block);
  u8_to_float_kernel<<<grid, block, 0, stream.value()>>>(d_u8.data(), output.data_handle(), count);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  raft::resource::sync_stream(handle);
}

void upload_as_float(raft::resources const& handle,
                     const HostMatrix& input,
                     raft::device_matrix<float, int64_t>& output)
{
  const size_t count = checked_element_count(input.rows, input.cols, "device upload");
  auto stream = raft::resource::get_cuda_stream(handle);
  if (input.scalar_type == InputScalarType::Float32 ||
      input.scalar_type == InputScalarType::Float16) {
    raft::copy(output.data_handle(), input.float_data.data(), count, stream);
    raft::resource::sync_stream(handle);
    return;
  }

  upload_uint8_as_float(handle, input.uint8_data.data(), count, output);
}

size_t padded_dim_for(size_t dim)
{
  return ((dim + 63) / 64) * 64;
}

void write_bytes(std::ofstream& output,
                 const void* data,
                 size_t bytes,
                 const std::string& description)
{
  if (bytes == 0) { return; }
  output.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(bytes));
  if (!output) { fail("failed to write " + description); }
}

void write_cuvs_batch_artifact(
  const std::string& output_path,
  uint32_t bits_per_dim,
  size_t rows,
  size_t dim,
  size_t num_centroids,
  const cuvs::neighbors::ivf_rabitq::detail::DataQuantizerGPU& quantizer,
  const cuvs::neighbors::ivf_rabitq::detail::RotatorGPU& rotator,
  const std::vector<size_t>& cluster_sizes,
  const std::vector<float>& rotated_centroids,
  const std::vector<uint8_t>& short_codes,
  const std::vector<float>& short_factors,
  const std::vector<uint8_t>& long_codes,
  const std::vector<float>& ex_factors,
  const std::vector<cuvs::neighbors::ivf_rabitq::detail::PID>& ids)
{
  std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
  if (!output) { fail("unable to open output artifact " + output_path); }

  const size_t ex_bits = static_cast<size_t>(bits_per_dim - 1);
  const bool batch_flag = true;
  write_bytes(output, &rows, sizeof(rows), "num_vectors");
  write_bytes(output, &dim, sizeof(dim), "num_dimensions");
  write_bytes(output, &num_centroids, sizeof(num_centroids), "num_centroids");
  write_bytes(output, &ex_bits, sizeof(ex_bits), "ex_bits");
  write_bytes(output, &batch_flag, sizeof(batch_flag), "batch flag");
  write_bytes(output,
              quantizer.get_query_scaling_factor(),
              sizeof(cuvs::neighbors::ivf_rabitq::detail::DataQuantizerGPU::FastQuantizeFactors),
              "query scaling factors");
  write_bytes(output,
              cluster_sizes.data(),
              cluster_sizes.size() * sizeof(cluster_sizes[0]),
              "cluster sizes");
  rotator.save(output);
  write_bytes(output,
              rotated_centroids.data(),
              rotated_centroids.size() * sizeof(rotated_centroids[0]),
              "rotated centroids");
  write_bytes(output, short_codes.data(), short_codes.size(), "short codes");
  write_bytes(output,
              short_factors.data(),
              short_factors.size() * sizeof(short_factors[0]),
              "short factors");
  write_bytes(output, long_codes.data(), long_codes.size(), "long codes");
  write_bytes(output,
              ex_factors.data(),
              ex_factors.size() * sizeof(ex_factors[0]),
              "extra factors");
  write_bytes(output, ids.data(), ids.size() * sizeof(ids[0]), "vector ids");
}

void configure_quantizer(cuvs::neighbors::ivf_rabitq::detail::DataQuantizerGPU& quantizer,
                         size_t padded_dim,
                         uint32_t bits_per_dim,
                         bool fast_quantize)
{
  quantizer.fast_quantize_flag = fast_quantize;
  quantizer.compute_query_scaling_factors(padded_dim);
  if (quantizer.fast_quantize_flag) { quantizer.compute_quantize_scaling_factors(); }

  if (quantizer.fast_quantize_flag) {
    const size_t ex_bits = static_cast<size_t>(bits_per_dim - 1);
    if (ex_bits == 3) {
      quantizer.set_quantize_scaling_factors(
        quantizer.get_query_scaling_factor()->const_scaling_factor_4bit);
    } else if (ex_bits == 7) {
      quantizer.set_quantize_scaling_factors(
        quantizer.get_query_scaling_factor()->const_scaling_factor_8bit);
    } else {
      quantizer.compute_quantize_scaling_factors();
    }
  }
}

void write_batch_artifact_for_bits(const Options& opts,
                                   raft::resources& handle,
                                   raft::device_matrix<float, int64_t>& d_dataset,
                                   raft::device_matrix<float, int64_t>& centers,
                                   raft::device_vector<uint32_t, int64_t>& labels,
                                   uint32_t bits_per_dim,
                                   const std::string& output_path)
{
  const int64_t rows = d_dataset.extent(0);
  const int64_t dim = d_dataset.extent(1);
  std::cout << "[rabitq-builder] constructing RabitQ codes bits_per_dim=" << bits_per_dim
            << "\n";
  cuvs::neighbors::ivf_rabitq::detail::IVFGPU index(
    handle,
    static_cast<size_t>(rows),
    static_cast<size_t>(dim),
    opts.n_lists,
    bits_per_dim,
    true);
  index.construct_on_gpu(
    d_dataset.data_handle(), centers.data_handle(), labels.data_handle(), opts.fast_quantize);

  std::cout << "[rabitq-builder] writing cuVS batch artifact to " << output_path << "\n";
  index.save(output_path.c_str(), true);
  raft::resource::sync_stream(handle);
}

void write_streaming_artifact_for_bits(const Options& opts,
                                       const VectorFileMetadata& metadata,
                                       raft::resources& handle,
                                       StreamingVectorReader& reader,
                                       raft::device_matrix<float, int64_t>& centers,
                                       const std::vector<uint32_t>& labels,
                                       const std::vector<size_t>& cluster_sizes,
                                       const std::vector<size_t>& cluster_offsets,
                                       size_t chunk_rows,
                                       uint32_t bits_per_dim,
                                       const std::string& output_path)
{
  using cuvs::neighbors::ivf_rabitq::detail::DataQuantizerGPU;
  using cuvs::neighbors::ivf_rabitq::detail::PID;
  using cuvs::neighbors::ivf_rabitq::detail::RotatorGPU;

  const size_t rows = metadata.rows;
  const size_t dim = metadata.cols;
  const size_t padded_dim = padded_dim_for(dim);
  auto stream = raft::resource::get_cuda_stream(handle);

  std::cout << "[rabitq-builder] quantizing streaming artifact bits_per_dim="
            << bits_per_dim << "\n";
  DataQuantizerGPU quantizer(handle, dim, bits_per_dim - 1, true);
  configure_quantizer(quantizer, padded_dim, bits_per_dim, opts.fast_quantize);
  quantizer.alloc_buffers(chunk_rows);
  RotatorGPU rotator(handle, static_cast<uint32_t>(dim));

  auto d_padded_centers = raft::make_device_matrix<float, int64_t>(
    handle, opts.n_lists, static_cast<int64_t>(padded_dim));
  auto d_rotated_centers = raft::make_device_matrix<float, int64_t>(
    handle, opts.n_lists, static_cast<int64_t>(padded_dim));
  constexpr int block = 256;
  const size_t padded_center_values = static_cast<size_t>(opts.n_lists) * padded_dim;
  const int pad_grid = static_cast<int>((padded_center_values + block - 1) / block);
  pad_rows_kernel<<<pad_grid, block, 0, stream.value()>>>(
    centers.data_handle(), d_padded_centers.data_handle(), opts.n_lists, dim, padded_dim);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  rotator.rotate(d_padded_centers.data_handle(), d_rotated_centers.data_handle(), opts.n_lists);

  const size_t short_code_words = quantizer.short_code_length();
  const size_t short_code_bytes = short_code_words * sizeof(uint32_t);
  const size_t long_code_bytes = quantizer.long_code_length();
  constexpr size_t kShortFactorCount = 3;
  constexpr size_t kExFactorCount = 2;

  std::vector<uint8_t> short_codes(checked_size_product(rows, short_code_bytes, "short codes"));
  std::vector<float> short_factors(checked_size_product(rows, kShortFactorCount, "short factors"));
  std::vector<uint8_t> long_codes(checked_size_product(rows, long_code_bytes, "long codes"));
  std::vector<float> ex_factors(checked_size_product(rows, kExFactorCount, "extra factors"));
  std::vector<PID> ids(rows);

  rmm::device_uvector<float> d_cluster_data(
    checked_size_product(chunk_rows, dim, "device cluster data"), stream);
  rmm::device_uvector<PID> d_local_ids(chunk_rows, stream);
  rmm::device_uvector<uint32_t> d_short_codes(
    checked_size_product(chunk_rows, short_code_words, "device short codes"), stream);
  rmm::device_uvector<float> d_short_factors(
    checked_size_product(chunk_rows, kShortFactorCount, "device short factors"), stream);
  rmm::device_uvector<uint8_t> d_long_codes(
    std::max<size_t>(1, checked_size_product(chunk_rows, long_code_bytes, "device long codes")),
    stream);
  rmm::device_uvector<float> d_ex_factors(
    checked_size_product(chunk_rows, kExFactorCount, "device extra factors"), stream);

  std::vector<size_t> cluster_write_offsets = cluster_offsets;
  std::vector<std::vector<uint32_t>> local_rows_by_cluster(opts.n_lists);
  std::vector<float> chunk_host;
  std::vector<float> cluster_host;
  std::vector<PID> local_ids;
  for (size_t begin = 0; begin < rows; begin += chunk_rows) {
    const size_t count = std::min(chunk_rows, rows - begin);
    reader.read_rows_as_float(static_cast<uint32_t>(begin), static_cast<uint32_t>(count), chunk_host);
    for (auto& local_rows : local_rows_by_cluster) { local_rows.clear(); }
    for (uint32_t local = 0; local < count; ++local) {
      local_rows_by_cluster[labels[begin + local]].push_back(local);
    }

    for (uint32_t cluster = 0; cluster < opts.n_lists; ++cluster) {
      const auto& local_rows = local_rows_by_cluster[cluster];
      const size_t cluster_count = local_rows.size();
      if (cluster_count == 0) { continue; }

      cluster_host.resize(checked_size_product(cluster_count, dim, "cluster host chunk"));
      local_ids.resize(cluster_count);
      const size_t out_begin = cluster_write_offsets[cluster];
      for (size_t i = 0; i < cluster_count; ++i) {
        const uint32_t local_row = local_rows[i];
        const float* src = chunk_host.data() + static_cast<size_t>(local_row) * dim;
        float* dst = cluster_host.data() + i * dim;
        std::copy(src, src + dim, dst);
        local_ids[i] = static_cast<PID>(i);
        ids[out_begin + i] = static_cast<PID>(begin + local_row);
      }

      raft::copy(d_cluster_data.data(), cluster_host.data(), cluster_host.size(), stream);
      raft::copy(d_local_ids.data(), local_ids.data(), local_ids.size(), stream);
      quantizer.quantize_batch_opt(d_cluster_data.data(),
                                   centers.data_handle() + static_cast<size_t>(cluster) * dim,
                                   d_local_ids.data(),
                                   cluster_count,
                                   rotator,
                                   d_short_codes.data(),
                                   d_short_factors.data(),
                                   d_long_codes.data(),
                                   d_ex_factors.data(),
                                   d_rotated_centers.data_handle() +
                                     static_cast<size_t>(cluster) * padded_dim);
      RAFT_CUDA_TRY(cudaMemcpyAsync(short_codes.data() + out_begin * short_code_bytes,
                                    reinterpret_cast<const uint8_t*>(d_short_codes.data()),
                                    cluster_count * short_code_bytes,
                                    cudaMemcpyDeviceToHost,
                                    stream.value()));
      RAFT_CUDA_TRY(cudaMemcpyAsync(short_factors.data() + out_begin * kShortFactorCount,
                                    d_short_factors.data(),
                                    cluster_count * kShortFactorCount * sizeof(float),
                                    cudaMemcpyDeviceToHost,
                                    stream.value()));
      if (long_code_bytes > 0) {
        RAFT_CUDA_TRY(cudaMemcpyAsync(long_codes.data() + out_begin * long_code_bytes,
                                      d_long_codes.data(),
                                      cluster_count * long_code_bytes,
                                      cudaMemcpyDeviceToHost,
                                      stream.value()));
      }
      RAFT_CUDA_TRY(cudaMemcpyAsync(ex_factors.data() + out_begin * kExFactorCount,
                                    d_ex_factors.data(),
                                    cluster_count * kExFactorCount * sizeof(float),
                                    cudaMemcpyDeviceToHost,
                                    stream.value()));
      raft::resource::sync_stream(handle);
      cluster_write_offsets[cluster] += cluster_count;
    }
  }

  for (uint32_t cluster = 0; cluster < opts.n_lists; ++cluster) {
    if (cluster_write_offsets[cluster] != cluster_offsets[cluster + 1]) {
      fail("streaming quantization did not fill every vector id for cluster " +
           std::to_string(cluster));
    }
  }

  std::vector<float> rotated_centroids(
    checked_size_product(static_cast<size_t>(opts.n_lists), padded_dim, "rotated centroids"));
  raft::copy(rotated_centroids.data(),
             d_rotated_centers.data_handle(),
             rotated_centroids.size(),
             stream);
  raft::resource::sync_stream(handle);

  std::cout << "[rabitq-builder] writing streaming cuVS batch artifact to "
            << output_path << "\n";
  write_cuvs_batch_artifact(output_path,
                            bits_per_dim,
                            rows,
                            dim,
                            opts.n_lists,
                            quantizer,
                            rotator,
                            cluster_sizes,
                            rotated_centroids,
                            short_codes,
                            short_factors,
                            long_codes,
                            ex_factors,
                            ids);
}

void build_rabitq_artifact_batch(const Options& opts)
{
  std::cout << "[rabitq-builder] loading vectors from " << opts.base_vector_path << "\n";
  HostMatrix host_data = load_host_matrix(opts.base_vector_path, opts.row_limit);
  require_finite(host_data.float_data);
  if (opts.expected_dim != 0 && host_data.cols != opts.expected_dim) {
    fail("dimension mismatch: input dim=" + std::to_string(host_data.cols) +
         " expected=" + std::to_string(opts.expected_dim));
  }
  if (host_data.rows < opts.n_lists) {
    fail("number of vectors must be >= n_lists");
  }

  std::cout << "[rabitq-builder] rows=" << host_data.rows << " dim=" << host_data.cols
            << " scalar=" << scalar_type_name(host_data.scalar_type)
            << " n_lists=" << opts.n_lists << " bits_per_dim_count="
            << opts.bits_per_dim_list.size() << "\n";

  raft::resources handle;
  auto d_dataset =
    raft::make_device_matrix<float, int64_t>(handle, host_data.rows, host_data.cols);
  upload_as_float(handle, host_data, d_dataset);
  host_data.uint8_data.clear();
  host_data.uint8_data.shrink_to_fit();
  host_data.float_data.clear();
  host_data.float_data.shrink_to_fit();

  const int64_t rows = d_dataset.extent(0);
  const int64_t dim = d_dataset.extent(1);
  auto dataset_view =
    raft::make_device_matrix_view<const float, int64_t>(d_dataset.data_handle(), rows, dim);

  const size_t train_rows_size =
    std::min<size_t>(static_cast<size_t>(rows),
                     static_cast<size_t>(opts.max_train_points_per_cluster) * opts.n_lists);
  if (train_rows_size > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
    fail("training row count exceeds int64_t");
  }
  const int64_t train_rows = static_cast<int64_t>(train_rows_size);

  std::cout << "[rabitq-builder] sampling " << train_rows << " rows for balanced kmeans\n";
  auto trainset = raft::make_device_matrix<float, int64_t>(handle, train_rows, dim);
  raft::random::RngState rng_state{opts.seed};
  raft::matrix::sample_rows<float, int64_t>(
    handle, rng_state, dataset_view, trainset.view());

  auto centers =
    raft::make_device_matrix<float, int64_t>(handle, opts.n_lists, dim);
  cuvs::cluster::kmeans::balanced_params kmeans_params;
  kmeans_params.n_iters = opts.kmeans_iters;
  kmeans_params.metric = cuvs::distance::DistanceType::L2Expanded;

  std::cout << "[rabitq-builder] fitting balanced kmeans\n";
  cuvs::cluster::kmeans::fit(
    handle, kmeans_params, raft::make_const_mdspan(trainset.view()), centers.view());

  auto labels = raft::make_device_vector<uint32_t, int64_t>(handle, rows);
  auto centers_const = raft::make_device_matrix_view<const float, int64_t>(
    centers.data_handle(), opts.n_lists, dim);
  std::cout << "[rabitq-builder] assigning all rows to anchors\n";
  cuvs::cluster::kmeans::predict(
    handle, kmeans_params, dataset_view, centers_const, labels.view());
  raft::resource::sync_stream(handle);

  std::cout << "[rabitq-builder] reusing kmeans centers and row assignments for "
            << opts.bits_per_dim_list.size() << " bit-width artifact(s)\n";
  for (size_t i = 0; i < opts.bits_per_dim_list.size(); ++i) {
    write_batch_artifact_for_bits(opts,
                                  handle,
                                  d_dataset,
                                  centers,
                                  labels,
                                  opts.bits_per_dim_list[i],
                                  opts.output_paths[i]);
  }
  std::cout << "[rabitq-builder] done\n";
}

void build_rabitq_artifact_streaming(const Options& opts)
{
  VectorFileMetadata metadata = inspect_vector_file(opts.base_vector_path, opts.row_limit);
  if (opts.expected_dim != 0 && metadata.cols != opts.expected_dim) {
    fail("dimension mismatch: input dim=" + std::to_string(metadata.cols) +
         " expected=" + std::to_string(opts.expected_dim));
  }
  if (metadata.rows < opts.n_lists) {
    fail("number of vectors must be >= n_lists");
  }
  validate_selected_row_dimensions(metadata);

  const size_t rows = metadata.rows;
  const size_t dim = metadata.cols;
  const size_t chunk_rows = std::min<size_t>(opts.chunk_rows, rows);
  const size_t train_rows_size =
    std::min<size_t>(rows,
                     static_cast<size_t>(opts.max_train_points_per_cluster) * opts.n_lists);
  if (train_rows_size > static_cast<size_t>(std::numeric_limits<int64_t>::max()) ||
      dim > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
    fail("streaming build dimensions exceed int64_t limits");
  }

  std::cout << "[rabitq-builder] rows=" << rows << " dim=" << dim
            << " scalar=" << scalar_type_name(metadata.scalar_type)
            << " n_lists=" << opts.n_lists << " bits_per_dim_count="
            << opts.bits_per_dim_list.size()
            << " mode=" << build_mode_name(opts.build_mode)
            << " chunk_rows=" << chunk_rows << "\n";

  StreamingVectorReader reader(metadata);
  raft::resources handle;
  auto stream = raft::resource::get_cuda_stream(handle);

  std::cout << "[rabitq-builder] reading " << train_rows_size
            << " sampled rows for balanced kmeans\n";
  std::vector<uint32_t> train_ids =
    sample_training_rows(metadata.rows, static_cast<uint32_t>(train_rows_size), opts.seed);
  std::vector<float> train_host;
  reader.read_selected_rows_as_float(train_ids, train_host);

  auto centers =
    raft::make_device_matrix<float, int64_t>(handle, opts.n_lists, static_cast<int64_t>(dim));
  cuvs::cluster::kmeans::balanced_params kmeans_params;
  kmeans_params.n_iters = opts.kmeans_iters;
  kmeans_params.metric = cuvs::distance::DistanceType::L2Expanded;

  std::cout << "[rabitq-builder] fitting balanced kmeans\n";
  {
    auto trainset = raft::make_device_matrix<float, int64_t>(
      handle, static_cast<int64_t>(train_rows_size), static_cast<int64_t>(dim));
    raft::copy(trainset.data_handle(), train_host.data(), train_host.size(), stream);
    raft::resource::sync_stream(handle);
    train_host.clear();
    train_host.shrink_to_fit();
    train_ids.clear();
    train_ids.shrink_to_fit();
    cuvs::cluster::kmeans::fit(
      handle, kmeans_params, raft::make_const_mdspan(trainset.view()), centers.view());
  }

  std::vector<uint32_t> labels(rows);
  auto centers_const = raft::make_device_matrix_view<const float, int64_t>(
    centers.data_handle(), opts.n_lists, static_cast<int64_t>(dim));
  std::vector<float> chunk_host;
  std::cout << "[rabitq-builder] assigning rows to anchors in chunks\n";
  for (size_t begin = 0; begin < rows; begin += chunk_rows) {
    const size_t count = std::min(chunk_rows, rows - begin);
    reader.read_rows_as_float(static_cast<uint32_t>(begin), static_cast<uint32_t>(count), chunk_host);
    auto d_chunk = raft::make_device_matrix<float, int64_t>(
      handle, static_cast<int64_t>(count), static_cast<int64_t>(dim));
    raft::copy(d_chunk.data_handle(), chunk_host.data(), chunk_host.size(), stream);
    auto d_labels = raft::make_device_vector<uint32_t, int64_t>(
      handle, static_cast<int64_t>(count));
    auto chunk_view = raft::make_device_matrix_view<const float, int64_t>(
      d_chunk.data_handle(), static_cast<int64_t>(count), static_cast<int64_t>(dim));
    cuvs::cluster::kmeans::predict(
      handle, kmeans_params, chunk_view, centers_const, d_labels.view());
    raft::copy(labels.data() + begin, d_labels.data_handle(), count, stream);
    raft::resource::sync_stream(handle);
  }
  chunk_host.clear();
  chunk_host.shrink_to_fit();

  std::vector<size_t> cluster_sizes(opts.n_lists, 0);
  for (uint32_t label : labels) {
    if (label >= opts.n_lists) { fail("kmeans predict returned out-of-range anchor id"); }
    ++cluster_sizes[label];
  }
  std::vector<size_t> cluster_offsets(opts.n_lists + 1, 0);
  for (uint32_t i = 0; i < opts.n_lists; ++i) {
    cluster_offsets[i + 1] = cluster_offsets[i] + cluster_sizes[i];
  }
  if (cluster_offsets.back() != rows) { fail("cluster sizes do not sum to row count"); }

  std::cout << "[rabitq-builder] reusing kmeans centers and row assignments for "
            << opts.bits_per_dim_list.size() << " streaming bit-width artifact(s)\n";
  for (size_t i = 0; i < opts.bits_per_dim_list.size(); ++i) {
    write_streaming_artifact_for_bits(opts,
                                      metadata,
                                      handle,
                                      reader,
                                      centers,
                                      labels,
                                      cluster_sizes,
                                      cluster_offsets,
                                      chunk_rows,
                                      opts.bits_per_dim_list[i],
                                      opts.output_paths[i]);
  }
  std::cout << "[rabitq-builder] done\n";
}

void build_rabitq_artifact(const Options& opts)
{
  const auto metadata = inspect_vector_file(opts.base_vector_path, opts.row_limit);
  if (opts.expected_dim && metadata.cols != opts.expected_dim) { fail("dimension mismatch"); }
  if (metadata.rows < opts.n_lists) { fail("number of vectors must be >= n_lists"); }
  validate_selected_row_dimensions(metadata);
  for (const auto& path : opts.output_paths) {
    if (std::filesystem::exists(path)) { fail("output already exists: " + path); }
  }
  if (opts.device >= 0) { RAFT_CUDA_TRY(cudaSetDevice(opts.device)); }
  if (opts.build_mode == BuildMode::Streaming) {
    build_rabitq_artifact_streaming(opts);
  } else {
    build_rabitq_artifact_batch(opts);
  }
}

}  // namespace

int main(int argc, char** argv)
{
  try {
    const Options opts = parse_args(argc, argv);
    build_rabitq_artifact(opts);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
