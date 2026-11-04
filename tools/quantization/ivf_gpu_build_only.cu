/*
 * Build-only subset of the pre-merge cuVS IVF-RaBitQ GPU index.
 *
 * This intentionally omits BatchClusterSearch* and the searcher kernels. The builder only needs
 * construction plus the batch artifact writer consumed by FlashANNS.
 */

#include "ivf_rabitq/gpu_index/ivf_gpu.cuh"

#include "ivf_rabitq/utils/tools.hpp"

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>

#include <cub/device/device_histogram.cuh>
#include <cub/device/device_scan.cuh>

#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace cuvs::neighbors::ivf_rabitq::detail {

IVFGPU::IVFGPU(raft::resources const& handle,
               size_t n,
               size_t dim,
               size_t k,
               size_t bits_per_dim,
               bool batch_flag)
  : handle_(handle),
    batch_flag(batch_flag),
    num_vectors(n),
    num_dimensions(dim),
    num_padded_dim(rd_up_to_multiple_of(dim, 64)),
    num_centroids(k),
    max_cluster_length(0),
    ex_bits(bits_per_dim - 1),
    initializer(nullptr),
    DQ(std::make_unique<DataQuantizerGPU>(handle_, dim, bits_per_dim - 1, batch_flag)),
    Rota(std::make_unique<RotatorGPU>(handle_, dim))
{
}

void IVFGPU::AllocateDeviceMemory()
{
  initializer = std::make_unique<FlatInitializerGPU>(handle_, num_padded_dim, num_centroids);

  const size_t short_data_size = GetShortDataBytesSimple();
  const size_t long_code_size  = GetLongCodeBytes();
  const size_t ex_factor_size  = GetExFactorBytes();
  const size_t pids_size       = GetPIDsBytes();

  short_data_ =
    raft::make_device_vector<uint32_t, int64_t>(handle_, short_data_size / sizeof(uint32_t));
  if (batch_flag) {
    short_factors_batch_ = raft::make_device_vector<float, int64_t>(
      handle_, GetShortDataFactorBytesBatch() / sizeof(float));
  }
  long_code_ =
    raft::make_device_vector<uint8_t, int64_t>(handle_, long_code_size / sizeof(uint8_t));
  ex_factor_ =
    raft::make_device_vector<ExFactor, int64_t>(handle_, ex_factor_size / sizeof(ExFactor));
  ids_ = raft::make_device_vector<PID, int64_t>(handle_, pids_size / sizeof(PID));
  cluster_meta_ = raft::make_device_vector<GPUClusterMeta, int64_t>(handle_, num_centroids);
  raft::resource::sync_stream(handle_);
}

void IVFGPU::AllocateHostMemory()
{
  const size_t short_data_size = GetShortDataBytesSimple();
  const size_t long_code_size  = GetLongCodeBytes();
  const size_t ex_factor_size  = GetExFactorBytes();
  const size_t pids_size       = GetPIDsBytes();

  short_data_host_ =
    raft::make_host_vector<uint32_t, int64_t>(short_data_size / sizeof(uint32_t));
  long_code_host_ =
    raft::make_host_vector<uint8_t, int64_t>(long_code_size / sizeof(uint8_t));
  ex_factor_host_ =
    raft::make_host_vector<ExFactor, int64_t>(ex_factor_size / sizeof(ExFactor));
  ids_host_ = raft::make_host_vector<PID, int64_t>(pids_size / sizeof(PID));
}

void IVFGPU::load_transposed(const char*)
{
  throw std::runtime_error("build-only IVF-RaBitQ does not support load_transposed");
}

void IVFGPU::init_clusters(const std::vector<size_t>& cluster_sizes)
{
  cluster_meta_host_ = raft::make_host_vector<GPUClusterMeta, int64_t>(num_centroids);

  size_t added_vectors = 0;
  for (size_t i = 0; i < num_centroids; ++i) {
    const size_t num = cluster_sizes[i];
    cluster_meta_host_(i) = {num, added_vectors};
    added_vectors += num;
  }

  raft::copy(cluster_meta_.data_handle(), cluster_meta_host_.data_handle(), num_centroids, stream_);
}

void IVFGPU::save(const char* filename, bool save_batch_flag) const
{
  if (num_centroids == 0) {
    std::cerr << "IVF not constructed\n";
    return;
  }

  std::ofstream output(filename, std::ios::binary);
  if (!output.is_open()) {
    std::cerr << "Failed to open file for saving\n";
    return;
  }

  output.write(reinterpret_cast<const char*>(&num_vectors), sizeof(size_t));
  output.write(reinterpret_cast<const char*>(&num_dimensions), sizeof(size_t));
  output.write(reinterpret_cast<const char*>(&num_centroids), sizeof(size_t));
  output.write(reinterpret_cast<const char*>(&ex_bits), sizeof(size_t));
  if (save_batch_flag) { output.write(reinterpret_cast<const char*>(&batch_flag), sizeof(bool)); }
  output.write(reinterpret_cast<const char*>(DQ->get_query_scaling_factor()),
               sizeof(DataQuantizerGPU::FastQuantizeFactors));

  std::vector<GPUClusterMeta> h_cluster_meta(num_centroids);
  raft::copy(h_cluster_meta.data(), cluster_meta_.data_handle(), num_centroids, stream_);
  std::vector<size_t> cluster_sizes(num_centroids);
  raft::resource::sync_stream(handle_);
  for (size_t i = 0; i < num_centroids; ++i) {
    cluster_sizes[i] = h_cluster_meta[i].num;
  }
  output.write(reinterpret_cast<const char*>(cluster_sizes.data()), sizeof(size_t) * num_centroids);

  rotator().save(output);
  initializer->SaveCentroids(output, filename);

  const size_t short_data_size = GetShortDataBytesSimple();
  const size_t long_code_size  = GetLongCodeBytes();
  const size_t ex_factor_size  = GetExFactorBytes();
  const size_t ids_size        = GetPIDsBytes();
  const size_t short_factors_size = GetShortDataFactorBytesBatch();

  auto h_short_data_buf = raft::make_host_vector<uint8_t, int64_t>(short_data_size);
  auto h_long_code_buf  = raft::make_host_vector<uint8_t, int64_t>(long_code_size);
  auto h_ex_factor_buf =
    raft::make_host_vector<ExFactor, int64_t>(ex_factor_size / sizeof(ExFactor));
  auto h_ids_buf                 = raft::make_host_vector<PID, int64_t>(ids_size / sizeof(PID));
  auto h_short_factors_batch_buf = raft::make_host_vector<uint8_t, int64_t>(short_factors_size);

  raft::copy(h_short_data_buf.data_handle(),
             reinterpret_cast<const uint8_t*>(short_data_.data_handle()),
             short_data_size,
             stream_);
  if (batch_flag) {
    raft::copy(h_short_factors_batch_buf.data_handle(),
               reinterpret_cast<const uint8_t*>(short_factors_batch_.data_handle()),
               short_factors_size,
               stream_);
  }
  raft::copy(h_long_code_buf.data_handle(), long_code_.data_handle(), long_code_size, stream_);
  raft::copy(h_ex_factor_buf.data_handle(),
             ex_factor_.data_handle(),
             ex_factor_size / sizeof(ExFactor),
             stream_);
  raft::copy(h_ids_buf.data_handle(), ids_.data_handle(), ids_size / sizeof(PID), stream_);
  raft::resource::sync_stream(handle_);

  output.write(reinterpret_cast<const char*>(h_short_data_buf.data_handle()), short_data_size);
  if (batch_flag) {
    output.write(reinterpret_cast<const char*>(h_short_factors_batch_buf.data_handle()),
                 short_factors_size);
  }
  output.write(reinterpret_cast<const char*>(h_long_code_buf.data_handle()), long_code_size);
  output.write(reinterpret_cast<const char*>(h_ex_factor_buf.data_handle()), ex_factor_size);
  output.write(reinterpret_cast<const char*>(h_ids_buf.data_handle()), ids_size);
}

__global__ void build_cluster_meta_kernel(IVFGPU::GPUClusterMeta* d_cluster_meta,
                                          const unsigned long long* d_counts,
                                          const size_t* d_offsets,
                                          size_t num_centroids)
{
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_centroids) {
    d_cluster_meta[idx] = IVFGPU::GPUClusterMeta(d_counts[idx], d_offsets[idx]);
  }
}

__global__ void scatter_pids_kernel(PID* d_flat_pids,
                                    const PID* d_cluster_ids,
                                    size_t* d_atomic_counters,
                                    size_t num_vectors)
{
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < num_vectors) {
    const PID cid = d_cluster_ids[idx];
    const size_t pos = atomicAdd(reinterpret_cast<unsigned long long*>(&d_atomic_counters[cid]),
                                 1ULL);
    d_flat_pids[pos] = static_cast<PID>(idx);
  }
}

void IVFGPU::construct_on_gpu(const float* device_data,
                              const float* device_centroids,
                              const PID* device_cluster_ids,
                              bool fast_quantize)
{
  DQ->fast_quantize_flag = fast_quantize;
  DQ->compute_query_scaling_factors(num_padded_dim);
  if (DQ->fast_quantize_flag) { DQ->compute_quantize_scaling_factors(); }

  if (DQ->fast_quantize_flag) {
    if (ex_bits == 3) {
      DQ->set_quantize_scaling_factors(DQ->get_query_scaling_factor()->const_scaling_factor_4bit);
    } else if (ex_bits == 7) {
      DQ->set_quantize_scaling_factors(DQ->get_query_scaling_factor()->const_scaling_factor_8bit);
    } else {
      DQ->compute_quantize_scaling_factors();
    }
  }

  AllocateDeviceMemory();

  using CounterT = unsigned long long;
  const int num_levels  = static_cast<int>(num_centroids + 1);
  const int lower_level = 0;
  const int upper_level = static_cast<int>(num_centroids);

  CounterT* d_histogram = nullptr;
  RAFT_CUDA_TRY(cudaMallocAsync(&d_histogram, num_centroids * sizeof(CounterT), stream_));

  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceHistogram::HistogramEven(d_temp_storage,
                                      temp_storage_bytes,
                                      device_cluster_ids,
                                      d_histogram,
                                      num_levels,
                                      lower_level,
                                      upper_level,
                                      num_vectors,
                                      stream_);
  RAFT_CUDA_TRY(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream_));
  cub::DeviceHistogram::HistogramEven(d_temp_storage,
                                      temp_storage_bytes,
                                      device_cluster_ids,
                                      d_histogram,
                                      num_levels,
                                      lower_level,
                                      upper_level,
                                      num_vectors,
                                      stream_);
  RAFT_CUDA_TRY(cudaFreeAsync(d_temp_storage, stream_));

  size_t* d_offsets = nullptr;
  RAFT_CUDA_TRY(cudaMallocAsync(&d_offsets, (num_centroids + 1) * sizeof(size_t), stream_));

  d_temp_storage     = nullptr;
  temp_storage_bytes = 0;
  cub::DeviceScan::ExclusiveSum(
    d_temp_storage, temp_storage_bytes, d_histogram, d_offsets, num_centroids, stream_);
  RAFT_CUDA_TRY(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream_));
  cub::DeviceScan::ExclusiveSum(
    d_temp_storage, temp_storage_bytes, d_histogram, d_offsets, num_centroids, stream_);
  RAFT_CUDA_TRY(cudaFreeAsync(d_temp_storage, stream_));
  RAFT_CUDA_TRY(cudaMemcpyAsync(
    d_offsets + num_centroids, &num_vectors, sizeof(size_t), cudaMemcpyHostToDevice, stream_));

  constexpr int block_size = 256;
  int num_blocks = static_cast<int>((num_centroids + block_size - 1) / block_size);
  build_cluster_meta_kernel<<<num_blocks, block_size, 0, stream_>>>(
    cluster_meta_.data_handle(), d_histogram, d_offsets, num_centroids);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  RAFT_CUDA_TRY(cudaFreeAsync(d_histogram, stream_));

  size_t* d_atomic_counters = nullptr;
  RAFT_CUDA_TRY(cudaMallocAsync(&d_atomic_counters, num_centroids * sizeof(size_t), stream_));
  RAFT_CUDA_TRY(cudaMemcpyAsync(d_atomic_counters,
                                d_offsets,
                                num_centroids * sizeof(size_t),
                                cudaMemcpyDeviceToDevice,
                                stream_));

  num_blocks = static_cast<int>((num_vectors + block_size - 1) / block_size);
  scatter_pids_kernel<<<num_blocks, block_size, 0, stream_>>>(
    ids_.data_handle(), device_cluster_ids, d_atomic_counters, num_vectors);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  RAFT_CUDA_TRY(cudaFreeAsync(d_atomic_counters, stream_));

  std::vector<GPUClusterMeta> h_cluster_meta(num_centroids);
  RAFT_CUDA_TRY(cudaMemcpyAsync(h_cluster_meta.data(),
                                cluster_meta_.data_handle(),
                                num_centroids * sizeof(GPUClusterMeta),
                                cudaMemcpyDeviceToHost,
                                stream_));
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream_));
  RAFT_CUDA_TRY(cudaFreeAsync(d_offsets, stream_));

  float* d_rotated_centroids = nullptr;
  RAFT_CUDA_TRY(cudaMallocAsync(
    reinterpret_cast<void**>(&d_rotated_centroids),
    num_centroids * num_padded_dim * sizeof(float),
    stream_));

  max_cluster_length = 0;
  for (const auto& meta : h_cluster_meta) {
    max_cluster_length = std::max(max_cluster_length, meta.num);
  }
  DQ->alloc_buffers(max_cluster_length);

  for (size_t i = 0; i < num_centroids; ++i) {
    const float* cur_centroid = device_centroids + i * num_dimensions;
    float* cur_rotated_c      = d_rotated_centroids + i * num_padded_dim;
    GPUClusterMeta& cp        = h_cluster_meta[i];
    quantize_cluster(cp, device_data, cur_centroid, cur_rotated_c);
  }

  initializer->AddVectors(d_rotated_centroids);
  RAFT_CUDA_TRY(cudaFreeAsync(d_rotated_centroids, stream_));
  raft::resource::sync_stream(handle_);
}

void IVFGPU::construct(const float* host_data,
                       const float* host_centroids,
                       const PID* host_cluster_ids,
                       bool fast_quantize)
{
  DQ->fast_quantize_flag = fast_quantize;
  DQ->compute_query_scaling_factors(num_padded_dim);
  if (DQ->fast_quantize_flag) { DQ->compute_quantize_scaling_factors(); }

  if (DQ->fast_quantize_flag) {
    if (ex_bits == 3) {
      DQ->set_quantize_scaling_factors(DQ->get_query_scaling_factor()->const_scaling_factor_4bit);
    } else if (ex_bits == 7) {
      DQ->set_quantize_scaling_factors(DQ->get_query_scaling_factor()->const_scaling_factor_8bit);
    } else {
      DQ->compute_quantize_scaling_factors();
    }
  }

  std::vector<size_t> counts(num_centroids, 0);
  for (size_t i = 0; i < num_vectors; ++i) {
    const PID cid = host_cluster_ids[i];
    if (cid >= num_centroids) { throw std::runtime_error("bad RabitQ cluster id"); }
    counts[cid]++;
  }

  std::vector<GPUClusterMeta> h_cluster_meta;
  h_cluster_meta.reserve(num_centroids);
  std::vector<size_t> offsets(num_centroids + 1, 0);
  for (size_t i = 0; i < num_centroids; ++i) {
    h_cluster_meta.emplace_back(counts[i], offsets[i]);
    offsets[i + 1] = offsets[i] + counts[i];
  }

  std::vector<PID> flat_pids(num_vectors);
  std::vector<size_t> write_pos(offsets.begin(), offsets.end() - 1);
  for (size_t i = 0; i < num_vectors; ++i) {
    const PID cid = host_cluster_ids[i];
    flat_pids[write_pos[cid]++] = static_cast<PID>(i);
  }

  const size_t data_bytes     = num_vectors * num_dimensions * sizeof(float);
  const size_t centroid_bytes = num_centroids * num_dimensions * sizeof(float);
  float* d_data               = nullptr;
  float* d_centroid           = nullptr;
  RAFT_CUDA_TRY(cudaMallocAsync(reinterpret_cast<void**>(&d_data), data_bytes, stream_));
  RAFT_CUDA_TRY(cudaMallocAsync(reinterpret_cast<void**>(&d_centroid), centroid_bytes, stream_));
  RAFT_CUDA_TRY(cudaMemcpyAsync(d_data, host_data, data_bytes, cudaMemcpyHostToDevice, stream_));
  RAFT_CUDA_TRY(
    cudaMemcpyAsync(d_centroid, host_centroids, centroid_bytes, cudaMemcpyHostToDevice, stream_));

  AllocateDeviceMemory();
  raft::copy(ids_.data_handle(), flat_pids.data(), num_vectors, stream_);
  raft::copy(cluster_meta_.data_handle(), h_cluster_meta.data(), num_centroids, stream_);

  float* d_rotated_centroids = nullptr;
  RAFT_CUDA_TRY(cudaMallocAsync(
    reinterpret_cast<void**>(&d_rotated_centroids),
    num_centroids * num_padded_dim * sizeof(float),
    stream_));

  max_cluster_length = 0;
  for (const auto& meta : h_cluster_meta) {
    max_cluster_length = std::max(max_cluster_length, meta.num);
  }
  DQ->alloc_buffers(max_cluster_length);

  for (size_t i = 0; i < num_centroids; ++i) {
    const float* cur_centroid = d_centroid + i * num_dimensions;
    float* cur_rotated_c      = d_rotated_centroids + i * num_padded_dim;
    GPUClusterMeta& cp        = h_cluster_meta[i];
    quantize_cluster(cp, d_data, cur_centroid, cur_rotated_c);
  }

  initializer->AddVectors(d_rotated_centroids);
  RAFT_CUDA_TRY(cudaFreeAsync(d_data, stream_));
  RAFT_CUDA_TRY(cudaFreeAsync(d_centroid, stream_));
  RAFT_CUDA_TRY(cudaFreeAsync(d_rotated_centroids, stream_));
  raft::resource::sync_stream(handle_);
}

void IVFGPU::quantize_cluster(GPUClusterMeta& cp,
                              const float* d_data,
                              const float* d_centroid,
                              float* d_rotated_c) const
{
  const size_t num = cp.num;
  const PID* idp = cp.ids(*this);

  if (!batch_flag) {
    DQ->quantize(d_data,
                 d_centroid,
                 idp,
                 num,
                 rotator(),
                 cp.first_block(*this),
                 cp.long_code(*this, 0, DQ->long_code_length()),
                 reinterpret_cast<float*>(cp.ex_factor(*this, 0)),
                 d_rotated_c);
  } else {
    DQ->quantize_batch_opt(d_data,
                           d_centroid,
                           idp,
                           num,
                           rotator(),
                           cp.first_block_batch(*this),
                           cp.short_factor_batch(*this, 0),
                           cp.long_code(*this, 0, DQ->long_code_length()),
                           reinterpret_cast<float*>(cp.ex_factor_batch(*this, 0)),
                           d_rotated_c);
  }
}

}  // namespace cuvs::neighbors::ivf_rabitq::detail
