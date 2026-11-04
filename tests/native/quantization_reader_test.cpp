// Exercise the actual host readers without creating a CUDA context or running kernels.
#include "../../tools/quantization/vector_reader.h"
using namespace go_fann_quantization;

#include <functional>

namespace {

void require(bool condition, const std::string& message)
{
  if (!condition) { throw std::runtime_error(message); }
}

void expect_error(const std::function<void()>& action, const std::string& expected)
{
  try {
    action();
  } catch (const std::exception& error) {
    require(std::string(error.what()).find(expected) != std::string::npos,
            "unexpected error: " + std::string(error.what()));
    return;
  }
  throw std::runtime_error("expected error containing: " + expected);
}

void write_fbin(const std::string& path, uint32_t rows, uint32_t cols,
                const std::vector<float>& values)
{
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
  output.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
  output.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
  require(static_cast<bool>(output), "failed to write test fixture");
}

void test_fbin(const std::filesystem::path& directory)
{
  constexpr uint32_t rows = 7;
  constexpr uint32_t cols = 96;
  std::vector<float> values(rows * cols);
  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = static_cast<float>(static_cast<int>(i) - 400) * 0.125f;
  }
  const std::string path = (directory / "vectors.fbin").string();
  write_fbin(path, rows, cols, values);
  const HostMatrix batch = load_host_matrix(path, 0);
  require(batch.rows == rows && batch.cols == cols, "batch fbin shape mismatch");
  require(batch.scalar_type == InputScalarType::Float32 && batch.uint8_data.empty(),
          "fbin did not preserve float32 scalar type");
  require(batch.float_data == values, "batch fbin payload changed");

  const auto metadata = inspect_vector_file(path, 5);
  require(metadata.total_rows == rows && metadata.rows == 5 && metadata.cols == cols,
          "fbin prefix metadata mismatch");
  require(metadata.data_offset == 8 && metadata.record_bytes == cols * sizeof(float) &&
            !metadata.per_row_dim_header && metadata.scalar_type == InputScalarType::Float32,
          "fbin row offsets or scalar metadata mismatch");
  StreamingVectorReader reader(metadata);
  std::vector<float> actual;
  reader.read_rows_as_float(1, 3, actual);
  require(actual == std::vector<float>(values.begin() + cols, values.begin() + 4 * cols),
          "streaming contiguous fbin rows changed");
  reader.read_selected_rows_as_float({4, 0, 2, 4}, actual);
  std::vector<float> expected;
  for (uint32_t row : {4u, 0u, 2u, 4u}) {
    expected.insert(expected.end(), values.begin() + row * cols, values.begin() + (row + 1) * cols);
  }
  require(actual == expected, "training fbin row order or values changed");
  reader.read_rows_as_float(5, 0, actual);
  require(actual.empty(), "empty fbin chunk was not empty");
  expect_error([&] { reader.read_rows_as_float(4, 2, actual); }, "outside selected vector prefix");
  expect_error([&] { reader.read_selected_rows_as_float({5}, actual); }, "outside selected prefix");
  const HostMatrix prefix = load_host_matrix(path, 5);
  require(prefix.float_data == std::vector<float>(values.begin(), values.begin() + 5 * cols),
          "batch prefix did not preserve original row IDs");
  expect_error([&] { load_host_matrix(path, rows + 1); }, "exceeds row count");
  expect_error([&] { inspect_vector_file(path, rows + 1); }, "exceeds row count");

  // Every malformed header/payload must fail in both batch and streaming inspection,
  // including a truncated suffix even when a smaller valid prefix was requested.
  auto reject_file = [&](const std::string& expected_error) {
    expect_error([&] { load_host_matrix(path, 1); }, expected_error);
    expect_error([&] { inspect_vector_file(path, 1); }, expected_error);
  };
  write_fbin(path, 0, cols, {});
  reject_file("invalid .fbin header");
  write_fbin(path, rows, 0, {});
  reject_file("invalid .fbin header");
  write_fbin(path, rows, cols, std::vector<float>(values.begin(), values.end() - 1));
  reject_file("payload is truncated");
  write_fbin(path, rows - 1, cols, values);
  reject_file("file size does not match header");
  write_fbin(path, std::numeric_limits<uint32_t>::max(),
             std::numeric_limits<uint32_t>::max(), {});
  reject_file("overflows uint64_t");
  write_fbin(path, rows, cols, values);
  std::filesystem::resize_file(path, 4);
  reject_file("failed to read .fbin dimension");

  // Actual row reads must still detect a file truncated after metadata inspection.
  write_fbin(path, rows, cols, values);
  StreamingVectorReader stale_reader(inspect_vector_file(path, 0));
  std::filesystem::resize_file(path, 8 + 2 * cols * sizeof(float));
  expect_error([&] { stale_reader.read_rows_as_float(2, 1, actual); }, "failed to read fbin chunk");
}

void test_existing_formats(const std::filesystem::path& directory)
{
  const std::vector<float> values{-0.5f, 1.25f, 3.5f, -2.0f, 0.0f, 7.0f};
  const uint32_t rows = 2;
  const uint32_t cols = 3;
  const std::string float_path = (directory / "vectors.fvecs").string();
  {
    std::ofstream output(float_path, std::ios::binary);
    for (uint32_t row = 0; row < rows; ++row) {
      output.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
      output.write(reinterpret_cast<const char*>(values.data() + row * cols), cols * sizeof(float));
    }
  }
  require(load_host_matrix(float_path, 0).float_data == values, "fvecs batch regression");
  StreamingVectorReader float_reader(inspect_vector_file(float_path, 0));
  std::vector<float> actual;
  float_reader.read_rows_as_float(0, rows, actual);
  require(actual == values, "fvecs streaming regression");

  const std::string byte_path = (directory / "vectors.u8bin").string();
  const std::vector<uint8_t> bytes{0, 1, 255, 3, 4, 5};
  {
    std::ofstream output(byte_path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    output.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    output.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  }
  require(load_host_matrix(byte_path, 0).uint8_data == bytes, "u8bin batch regression");
  const auto metadata = inspect_vector_file(byte_path, 0);
  require(metadata.record_bytes == cols && metadata.scalar_type == InputScalarType::Uint8,
          "u8bin scalar or stride regression");
  StreamingVectorReader byte_reader(metadata);
  byte_reader.read_rows_as_float(0, rows, actual);
  require(actual == std::vector<float>(bytes.begin(), bytes.end()), "u8bin streaming regression");
}

void test_half_and_sampling(const std::filesystem::path& directory)
{
  const auto path = (directory / "half.hvecs").string();
  const uint32_t dim = 3;
  const std::vector<uint16_t> bits{0xb800, 0x3d00, 0x4300, 0xc000, 0, 0x4700};
  {
    std::ofstream output(path, std::ios::binary);
    for (int row = 0; row < 2; ++row) {
      output.write(reinterpret_cast<const char*>(&dim), 4);
      output.write(reinterpret_cast<const char*>(bits.data() + row * 3), 6);
    }
  }
  const std::vector<float> values{-0.5f, 1.25f, 3.5f, -2.0f, 0.0f, 7.0f};
  require(load_host_matrix(path, 0).float_data == values, "half batch values changed");
  StreamingVectorReader reader(inspect_vector_file(path, 0));
  std::vector<float> actual;
  reader.read_rows_as_float(0, 2, actual);
  require(actual == values, "half streaming values changed");
  {
    std::fstream output(path, std::ios::in | std::ios::out | std::ios::binary);
    output.seekp(10); const uint32_t wrong_dim = 2;
    output.write(reinterpret_cast<const char*>(&wrong_dim), 4);
  }
  expect_error([&] { validate_selected_row_dimensions(inspect_vector_file(path, 0)); }, "inconsistent vector dimension");
  const auto selected = sample_training_rows(1000000, 256, 137);
  require(selected == sample_training_rows(1000000, 256, 137), "sample seed unstable");
  require(selected != sample_training_rows(1000000, 256, 138), "sample seed ignored");
  require(selected.size() == 256 && std::adjacent_find(selected.begin(), selected.end()) == selected.end(), "sample duplicates");
  require(selected.back() < 1000000 && selected.back() > 500000, "sample only uses input prefix");
  expect_error([&] { sample_training_rows(2, 3, 0); }, "invalid training row count");
  expect_error([&] { parse_u64("-1", "seed"); }, "invalid numeric value");
  expect_error([&] { checked_size_product(SIZE_MAX, 2, "test"); }, "overflows");
}
}  // namespace

int main()
{
  char directory_template[] = "/tmp/rabitq-vector-reader-XXXXXX";
  const char* created = mkdtemp(directory_template);
  if (!created) {
    std::cerr << "failed to create test directory\n";
    return 1;
  }
  const std::filesystem::path directory(created);
  try {
    test_fbin(directory);
    test_existing_formats(directory);
    test_half_and_sampling(directory);
    std::filesystem::remove_all(directory);
    std::cout << "CPU vector reader tests passed (no GPU work)\n";
    return 0;
  } catch (const std::exception& error) {
    std::filesystem::remove_all(directory);
    std::cerr << error.what() << "\n";
    return 1;
  }
}
