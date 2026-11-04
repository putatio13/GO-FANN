#include "vector_reader.h"
#include <cuvs/preprocessing/quantize/pq.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/cudart_utils.hpp>

namespace {
using namespace go_fann_quantization;
struct Options {
  std::string input, pivots, codes;
  uint32_t dim = 0, rows = 0, chunks = 64, train_rows = 65536, chunk_rows = 8192, iterations = 25;
  uint64_t seed = 137;
  bool verify = false;
};
Options parse_args(int argc, char** argv)
{
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: pq-builder --input=PATH --pivots=PATH --codes=PATH [options]\n"
        "Inputs: .u8bin/.ibin/.bin/.bvecs uint8, .fbin/.fvecs FP32, .hvecs FP16.\n"
        "--pq-dim=64 --training-rows=65536 --chunk-rows=8192 --iterations=25\n"
        "--seed=137 (training sample) --row-limit=0 (all rows) --dim=0 (infer)\n"
        "--verify: compare exported reconstruction with cuVS inverse_transform\n"
        "Existing outputs are rejected. The public scripts.quantize CLI publishes both files together.\n";
      std::exit(0);
    }
    if (arg == "--verify") { o.verify = true; continue; }
    const auto equal = arg.find('=');
    if (equal == std::string::npos) { fail("expected --name=value: " + arg); }
    const auto key = arg.substr(0, equal), value = arg.substr(equal + 1);
    if (key == "--input") { o.input = value; }
    else if (key == "--pivots") { o.pivots = value; }
    else if (key == "--codes") { o.codes = value; }
    else if (key == "--seed") { o.seed = parse_u64(value, key.c_str()); }
    else {
      const auto n = parse_u32(value, key.c_str());
      if (key == "--dim") { o.dim = n; }
      else if (key == "--row-limit") { o.rows = n; }
      else if (key == "--pq-dim") { o.chunks = n; }
      else if (key == "--training-rows") { o.train_rows = n; }
      else if (key == "--chunk-rows") { o.chunk_rows = n; }
      else if (key == "--iterations") { o.iterations = n; }
      else { fail("unknown option: " + key); }
    }
  }
  if (o.input.empty() || o.pivots.empty() || o.codes.empty()) { fail("--input, --pivots and --codes are required"); }
  if (o.chunks != 16 && o.chunks != 64) { fail("GO-FANN graph search requires pq-dim 16 or 64"); }
  if (!o.chunks || !o.chunk_rows || !o.iterations || o.train_rows < 256) {
    fail("positive pq-dim, chunk-rows, iterations and training-rows >=256 are required");
  }
  if (std::filesystem::absolute(o.pivots).lexically_normal() ==
      std::filesystem::absolute(o.codes).lexically_normal()) { fail("output paths must differ"); }
  for (const auto& path : {o.pivots, o.codes}) {
    if (std::filesystem::exists(path)) { fail("output already exists: " + path); }
  }
  return o;
}
void bytes(std::ofstream& out, const void* data, size_t count)
{
  out.write(static_cast<const char*>(data), count);
  if (!out) { fail("output write failed"); }
}
template<class T>
void matrix(std::ofstream& out, uint32_t rows, uint32_t cols, const std::vector<T>& data)
{
  bytes(out, &rows, 4); bytes(out, &cols, 4); bytes(out, data.data(), data.size() * sizeof(T));
}
void build(const Options& o)
{
  auto meta = inspect_vector_file(o.input, o.rows);
  if (o.dim && o.dim != meta.cols) { fail("dimension mismatch"); }
  if (meta.rows < 256 || meta.cols % o.chunks) { fail("PQ requires >=256 rows and dim divisible by pq-dim"); }
  if (meta.rows > INT32_MAX || meta.cols > INT32_MAX) { fail("PQ matrix dimensions exceed int32 format"); }
  validate_selected_row_dimensions(meta);
  StreamingVectorReader reader(meta);
  const auto train_rows = std::min(meta.rows, o.train_rows);
  std::vector<float> training;
  reader.read_selected_rows_as_float(sample_training_rows(meta.rows, train_rows, o.seed), training);
  require_finite(training);
  raft::resources handle;
  auto stream = raft::resource::get_cuda_stream(handle);
  namespace pq = cuvs::preprocessing::quantize::pq;
  pq::params params;
  params.pq_bits = 8; params.pq_dim = o.chunks;
  params.use_subspaces = true; params.use_vq = false;
  params.kmeans_n_iters = o.iterations;
  params.pq_kmeans_type = cuvs::cluster::kmeans::kmeans_type::KMeans;
  params.max_train_points_per_pq_code = (train_rows + 255u) / 256u;
  auto quant = pq::build(handle, params,
    raft::make_host_matrix_view<const float, int64_t>(training.data(), train_rows, meta.cols));
  raft::resource::sync_stream(handle);
  training.clear(); training.shrink_to_fit();
  const uint32_t width = meta.cols / o.chunks;
  const auto& cb = quant.vpq_codebooks.pq_code_book;
  if (cb.extent(0) != o.chunks * 256 || cb.extent(1) != width) { fail("unexpected cuVS codebook layout"); }
  std::vector<float> raw(256ull * meta.cols), table(raw.size());
  raft::copy(raw.data(), cb.data_handle(), raw.size(), stream);
  raft::resource::sync_stream(handle);
  require_finite(raw);
  for (uint32_t m = 0; m < o.chunks; ++m) {
    for (uint32_t k = 0; k < 256; ++k) {
      for (uint32_t d = 0; d < width; ++d) {
        table[size_t(k) * meta.cols + m * width + d] = raw[(size_t(m) * 256 + k) * width + d];
      }
    }
  }
  // DiskANN pivot matrices: offsets, [256,D] centers, [D,1] zero mean, [M+1,1] boundaries.
  static_assert(sizeof(size_t) == 8, "The runtime artifact requires a 64-bit host");
  const size_t first = 8 + 4 * sizeof(size_t);
  std::vector<size_t> offsets{first, first + 8 + table.size() * 4,
    first + 16 + table.size() * 4 + size_t(meta.cols) * 4, 0};
  offsets[3] = offsets[2] + 8 + (size_t(o.chunks) + 1) * 4;
  std::vector<float> mean(meta.cols, 0);
  std::vector<uint32_t> boundaries(o.chunks + 1);
  for (uint32_t i = 0; i <= o.chunks; ++i) { boundaries[i] = i * width; }
  std::ofstream pivots(o.pivots, std::ios::binary), codes(o.codes, std::ios::binary);
  if (!pivots || !codes) { fail("unable to open output files"); }
  matrix(pivots, 4, 1, offsets); matrix(pivots, 256, meta.cols, table);
  matrix(pivots, meta.cols, 1, mean); matrix(pivots, o.chunks + 1, 1, boundaries);
  bytes(codes, &meta.rows, 4); bytes(codes, &o.chunks, 4);
  std::vector<float> chunk;
  for (uint64_t begin = 0; begin < meta.rows; begin += o.chunk_rows) {
    const uint32_t count = std::min<uint64_t>(o.chunk_rows, meta.rows - begin);
    reader.read_rows_as_float(begin, count, chunk); require_finite(chunk);
    auto encoded = raft::make_device_matrix<uint8_t, int64_t>(handle, count, o.chunks);
    pq::transform(handle, quant,
      raft::make_host_matrix_view<const float, int64_t>(chunk.data(), count, meta.cols), encoded.view());
    std::vector<uint8_t> host_codes(size_t(count) * o.chunks);
    raft::copy(host_codes.data(), encoded.data_handle(), host_codes.size(), stream);
    raft::resource::sync_stream(handle);
    if (o.verify) {
      auto restored = raft::make_device_matrix<float, int64_t>(handle, count, meta.cols);
      pq::inverse_transform(handle, quant, raft::make_const_mdspan(encoded.view()), restored.view());
      std::vector<float> reference(size_t(count) * meta.cols);
      raft::copy(reference.data(), restored.data_handle(), reference.size(), stream);
      raft::resource::sync_stream(handle);
      for (uint32_t r = 0; r < count; ++r) {
        for (uint32_t d = 0; d < meta.cols; ++d) {
          const float actual = table[size_t(host_codes[size_t(r) * o.chunks + d / width]) * meta.cols + d];
          const float expected = reference[size_t(r) * meta.cols + d];
          if (!std::isfinite(expected) || std::abs(actual - expected) > 1e-5f * (1 + std::abs(expected))) {
            fail("exported codebook disagrees with cuVS inverse_transform");
          }
        }
      }
    }
    bytes(codes, host_codes.data(), host_codes.size());
  }
  pivots.close(); codes.close();
  if (!pivots || !codes) { fail("output finalization failed"); }
  std::cout << "PQ rows=" << meta.rows << " dim=" << meta.cols << " chunks=" << o.chunks
    << " training_rows=" << train_rows << " sample_seed=" << o.seed
    << " inverse_transform_verified=" << o.verify << "\n";
}
} // namespace
int main(int argc, char** argv)
{
  try { const auto options = parse_args(argc, argv); build(options); return 0; }
  catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
