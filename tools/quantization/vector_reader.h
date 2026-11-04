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


#include <random>
#include <unordered_set>
#include <cmath>
#pragma once

namespace go_fann_quantization {
struct Uint8Matrix {
  uint32_t rows = 0;
  uint32_t cols = 0;
  std::vector<uint8_t> data;
};

struct FloatMatrix {
  uint32_t rows = 0;
  uint32_t cols = 0;
  std::vector<float> data;
};

enum class InputScalarType : uint8_t {
  Uint8,
  Float32,
  Float16,
};

struct HostMatrix {
  uint32_t rows = 0;
  uint32_t cols = 0;
  InputScalarType scalar_type = InputScalarType::Uint8;
  std::vector<uint8_t> uint8_data;
  std::vector<float> float_data;
};

struct VectorFileMetadata {
  std::string path;
  std::string extension;
  uint32_t rows = 0;
  uint32_t cols = 0;
  uint64_t total_rows = 0;
  uint64_t data_offset = 0;
  uint64_t record_bytes = 0;
  bool per_row_dim_header = false;
  InputScalarType scalar_type = InputScalarType::Uint8;
};

[[noreturn]] void fail(const std::string& message)
{
  throw std::runtime_error("quantization: " + message);
}

float half_to_float(uint16_t value) noexcept
{
  const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
  uint32_t exponent = static_cast<uint32_t>((value >> 10) & 0x1fu);
  uint32_t mantissa = static_cast<uint32_t>(value & 0x03ffu);

  uint32_t bits = 0;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      exponent = 1;
      while ((mantissa & 0x0400u) == 0) {
        mantissa <<= 1;
        --exponent;
      }
      mantissa &= 0x03ffu;
      bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    }
  } else if (exponent == 0x1fu) {
    bits = sign | 0x7f800000u | (mantissa << 13);
  } else {
    bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
  }

  float out = 0.0f;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

void convert_half_row_to_float(const uint16_t* src, float* dst, uint32_t cols)
{
  for (uint32_t col = 0; col < cols; ++col) {
    dst[col] = half_to_float(src[col]);
  }
}

std::string lower_extension(const std::string& path)
{
  std::string ext = std::filesystem::path(path).extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return ext;
}

uint64_t parse_u64(const std::string& value, const char* name)
{
  if (value.empty() || value[0] == '-' || value[0] == '+') { fail(std::string("invalid numeric value for ") + name); }
  size_t pos = 0;
  unsigned long long parsed = 0;
  try {
    parsed = std::stoull(value, &pos);
  } catch (const std::exception&) {
    fail(std::string("invalid numeric value for ") + name + ": " + value);
  }
  if (pos != value.size()) { fail(std::string("invalid numeric value for ") + name + ": " + value); }
  return static_cast<uint64_t>(parsed);
}

uint32_t parse_u32(const std::string& value, const char* name)
{
  const uint64_t parsed = parse_u64(value, name);
  if (parsed > std::numeric_limits<uint32_t>::max()) {
    fail(std::string("value too large for ") + name + ": " + value);
  }
  return static_cast<uint32_t>(parsed);
}

std::vector<std::string> split_csv(const std::string& value, const char* name)
{
  std::vector<std::string> items;
  size_t begin = 0;
  while (begin <= value.size()) {
    const size_t comma = value.find(',', begin);
    const size_t end = comma == std::string::npos ? value.size() : comma;
    std::string item = value.substr(begin, end - begin);
    if (item.empty()) {
      fail(std::string(name) + " contains an empty item");
    }
    items.push_back(std::move(item));
    if (comma == std::string::npos) { break; }
    begin = comma + 1;
  }
  if (items.empty()) { fail(std::string(name) + " must contain at least one item"); }
  return items;
}

std::vector<uint32_t> parse_u32_list(const std::string& value, const char* name)
{
  std::vector<std::string> items = split_csv(value, name);
  std::vector<uint32_t> parsed;
  parsed.reserve(items.size());
  for (const std::string& item : items) { parsed.push_back(parse_u32(item, name)); }
  return parsed;
}

bool parse_bool(const std::string& value, const char* name)
{
  if (value == "1" || value == "true" || value == "TRUE" || value == "on") { return true; }
  if (value == "0" || value == "false" || value == "FALSE" || value == "off") { return false; }
  fail(std::string("invalid boolean value for ") + name + ": " + value);
}

void read_exact(std::ifstream& input, void* dst, size_t bytes, const std::string& description)
{
  if (bytes > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
    fail(description + " exceeds streamsize range");
  }
  input.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(bytes));
  if (!input) { fail("failed to read " + description); }
}

size_t checked_element_count(uint32_t rows, uint32_t cols, const std::string& description)
{
  const size_t row_count = static_cast<size_t>(rows);
  const size_t col_count = static_cast<size_t>(cols);
  if (col_count != 0 && row_count > std::numeric_limits<size_t>::max() / col_count) {
    fail(description + " element count overflows size_t");
  }
  return row_count * col_count;
}

size_t checked_size_product(size_t lhs, size_t rhs, const std::string& description)
{
  if (rhs != 0 && lhs > std::numeric_limits<size_t>::max() / rhs) {
    fail(description + " size overflows size_t");
  }
  return lhs * rhs;
}

uint64_t checked_u64_product(uint64_t lhs, uint64_t rhs, const std::string& description)
{
  if (rhs != 0 && lhs > std::numeric_limits<uint64_t>::max() / rhs) {
    fail(description + " size overflows uint64_t");
  }
  return lhs * rhs;
}

uint64_t checked_u64_sum(uint64_t lhs, uint64_t rhs, const std::string& description)
{
  if (lhs > std::numeric_limits<uint64_t>::max() - rhs) {
    fail(description + " size overflows uint64_t");
  }
  return lhs + rhs;
}

uint32_t effective_rows_from_limit(uint64_t total_rows, uint32_t row_limit, const std::string& path)
{
  if (total_rows == 0) { fail("vector file has no rows: " + path); }
  if (row_limit > 0) {
    if (static_cast<uint64_t>(row_limit) > total_rows) {
      fail("--row-limit=" + std::to_string(row_limit) + " exceeds row count " +
           std::to_string(total_rows) + " for " + path);
    }
    return row_limit;
  }
  if (total_rows > std::numeric_limits<uint32_t>::max()) {
    fail("row count exceeds uint32 range for " + path +
         "; use --row-limit to build a prefix subset");
  }
  return static_cast<uint32_t>(total_rows);
}

Uint8Matrix load_u8bin(const std::string& path, uint32_t row_limit)
{
  std::ifstream input(path, std::ios::binary);
  if (!input) { fail("unable to open vector file " + path); }

  uint32_t rows = 0;
  uint32_t cols = 0;
  read_exact(input, &rows, sizeof(rows), "u8bin row count from " + path);
  read_exact(input, &cols, sizeof(cols), "u8bin dimension from " + path);
  if (rows == 0 || cols == 0) { fail("invalid u8bin header in " + path); }

  const uint32_t effective_rows = effective_rows_from_limit(rows, row_limit, path);
  Uint8Matrix out;
  out.rows = effective_rows;
  out.cols = cols;
  out.data.resize(checked_element_count(effective_rows, cols, "u8bin payload"));
  read_exact(input, out.data.data(), out.data.size(), "u8bin payload from " + path);
  return out;
}

Uint8Matrix load_bvecs(const std::string& path, uint32_t row_limit)
{
  std::ifstream input(path, std::ios::binary);
  if (!input) { fail("unable to open bvecs file " + path); }

  uint32_t dim = 0;
  read_exact(input, &dim, sizeof(dim), "first bvecs dimension from " + path);
  if (dim == 0) { fail("invalid bvecs dimension in " + path); }

  input.seekg(0, std::ios::end);
  const std::streamoff file_size = input.tellg();
  const std::streamoff record_size = sizeof(uint32_t) + static_cast<std::streamoff>(dim);
  if (file_size <= 0 || record_size <= 0 || file_size % record_size != 0) {
    fail("bvecs file size is not row-aligned for " + path);
  }
  const uint64_t total_rows = static_cast<uint64_t>(file_size / record_size);
  const uint32_t effective_rows = effective_rows_from_limit(total_rows, row_limit, path);

  Uint8Matrix out;
  out.rows = effective_rows;
  out.cols = dim;
  out.data.resize(checked_element_count(effective_rows, dim, "bvecs payload"));

  input.seekg(0, std::ios::beg);
  for (uint32_t row = 0; row < effective_rows; ++row) {
    uint32_t row_dim = 0;
    read_exact(input, &row_dim, sizeof(row_dim), "bvecs row dimension from " + path);
    if (row_dim != dim) {
      fail("inconsistent bvecs dimension at row " + std::to_string(row) + " in " + path);
    }
    read_exact(input,
               out.data.data() + static_cast<size_t>(row) * dim,
               dim,
               "bvecs row payload from " + path);
  }
  return out;
}

Uint8Matrix load_uint8_matrix(const std::string& path, uint32_t row_limit)
{
  const std::string ext = lower_extension(path);
  if (ext == ".bvecs") { return load_bvecs(path, row_limit); }
  if (ext == ".ibin" || ext == ".u8bin" || ext == ".bin") { return load_u8bin(path, row_limit); }
  fail("unsupported vector format " + ext + " for " + path);
}

FloatMatrix load_fvecs(const std::string& path, uint32_t row_limit)
{
  std::ifstream input(path, std::ios::binary);
  if (!input) { fail("unable to open float vector file " + path); }
  const std::string ext = lower_extension(path);
  if (ext != ".fvecs" && ext != ".hvecs") {
    fail("unsupported float vector format " + ext + " for " + path);
  }

  uint32_t dim = 0;
  read_exact(input, &dim, sizeof(dim), "first " + ext + " dimension from " + path);
  if (dim == 0) { fail("invalid " + ext + " dimension 0 in " + path); }

  input.seekg(0, std::ios::end);
  const std::streamoff file_size = input.tellg();
  if (file_size < 0) { fail("failed to determine " + ext + " file size for " + path); }
  const std::streamoff record_size =
    static_cast<std::streamoff>(sizeof(uint32_t)) +
    static_cast<std::streamoff>(dim) *
      static_cast<std::streamoff>(ext == ".hvecs" ? sizeof(uint16_t) : sizeof(float));
  if (record_size <= 0 || file_size % record_size != 0) {
    fail(ext + " file size is not row-aligned for dim=" + std::to_string(dim) + " in " + path);
  }

  const uint64_t total_rows = static_cast<uint64_t>(file_size / record_size);
  const uint32_t effective_rows = effective_rows_from_limit(total_rows, row_limit, path);

  FloatMatrix out;
  out.rows = effective_rows;
  out.cols = dim;
  out.data.resize(checked_element_count(effective_rows, dim, "fvecs payload"));

  input.seekg(0, std::ios::beg);
  for (uint32_t row = 0; row < effective_rows; ++row) {
    uint32_t row_dim = 0;
    read_exact(input, &row_dim, sizeof(row_dim), ext + " row dimension from " + path);
    if (row_dim != dim) {
      fail("inconsistent " + ext + " dimension at row " + std::to_string(row) + " in " + path +
           ": got " + std::to_string(row_dim) + " expected " + std::to_string(dim));
    }
    float* dst = out.data.data() + static_cast<size_t>(row) * dim;
    if (ext == ".hvecs") {
      std::vector<uint16_t> half_row(dim);
      read_exact(input,
                 half_row.data(),
                 static_cast<size_t>(dim) * sizeof(uint16_t),
                 "hvecs row payload from " + path);
      convert_half_row_to_float(half_row.data(), dst, dim);
    } else {
      read_exact(input,
                 dst,
                 static_cast<size_t>(dim) * sizeof(float),
                 "fvecs row payload from " + path);
    }
  }
  return out;
}

VectorFileMetadata inspect_vector_file(const std::string& path, uint32_t row_limit);

HostMatrix load_host_matrix(const std::string& path, uint32_t row_limit)
{
  const std::string ext = lower_extension(path);
  HostMatrix out;
  if (ext == ".fbin") {
    const VectorFileMetadata metadata = inspect_vector_file(path, row_limit);
    std::ifstream input(path, std::ios::binary);
    if (!input) { fail("unable to open float vector file " + path); }
    input.seekg(static_cast<std::streamoff>(metadata.data_offset), std::ios::beg);
    out.rows = metadata.rows;
    out.cols = metadata.cols;
    out.scalar_type = InputScalarType::Float32;
    out.float_data.resize(checked_element_count(out.rows, out.cols, "fbin payload"));
    read_exact(input,
               out.float_data.data(),
               checked_size_product(out.float_data.size(), sizeof(float), "fbin payload"),
               "fbin payload from " + path);
    return out;
  }
  if (ext == ".fvecs" || ext == ".hvecs") {
    FloatMatrix input = load_fvecs(path, row_limit);
    out.rows = input.rows;
    out.cols = input.cols;
    out.scalar_type = ext == ".hvecs" ? InputScalarType::Float16 : InputScalarType::Float32;
    out.float_data = std::move(input.data);
    return out;
  }
  if (ext == ".bvecs" || ext == ".ibin" || ext == ".u8bin" || ext == ".bin") {
    Uint8Matrix input = load_uint8_matrix(path, row_limit);
    out.rows = input.rows;
    out.cols = input.cols;
    out.scalar_type = InputScalarType::Uint8;
    out.uint8_data = std::move(input.data);
    return out;
  }
  fail("unsupported vector format " + ext + " for " + path +
       " (supported: .bvecs, .u8bin, .ibin, .bin, .fbin, .fvecs, .hvecs)");
}

VectorFileMetadata inspect_vector_file(const std::string& path, uint32_t row_limit)
{
  const std::string ext = lower_extension(path);
  std::ifstream input(path, std::ios::binary);
  if (!input) { fail("unable to open vector file " + path); }

  VectorFileMetadata metadata;
  metadata.path = path;
  metadata.extension = ext;

  if (ext == ".fvecs" || ext == ".hvecs" || ext == ".bvecs") {
    uint32_t dim = 0;
    read_exact(input, &dim, sizeof(dim), "first vector dimension from " + path);
    if (dim == 0) { fail("invalid vector dimension 0 in " + path); }

    input.seekg(0, std::ios::end);
    const std::streamoff file_size = input.tellg();
    if (file_size <= 0) { fail("invalid or empty vector file " + path); }

    uint64_t payload_bytes = static_cast<uint64_t>(dim);
    if (ext == ".fvecs") {
      payload_bytes = checked_u64_product(dim, sizeof(float), "fvecs record");
    } else if (ext == ".hvecs") {
      payload_bytes = checked_u64_product(dim, sizeof(uint16_t), "hvecs record");
    }
    const uint64_t record_bytes =
      checked_u64_sum(sizeof(uint32_t), payload_bytes, "vector record");
    if (record_bytes == 0 || static_cast<uint64_t>(file_size) % record_bytes != 0) {
      fail("vector file size is not row-aligned for dim=" + std::to_string(dim) + " in " +
           path);
    }

    metadata.total_rows = static_cast<uint64_t>(file_size) / record_bytes;
    metadata.rows = effective_rows_from_limit(metadata.total_rows, row_limit, path);
    metadata.cols = dim;
    metadata.record_bytes = record_bytes;
    metadata.per_row_dim_header = true;
    if (ext == ".fvecs") {
      metadata.scalar_type = InputScalarType::Float32;
    } else if (ext == ".hvecs") {
      metadata.scalar_type = InputScalarType::Float16;
    } else {
      metadata.scalar_type = InputScalarType::Uint8;
    }
    return metadata;
  }

  if (ext == ".ibin" || ext == ".u8bin" || ext == ".bin" || ext == ".fbin") {
    const bool is_float = ext == ".fbin";
    uint32_t rows = 0;
    uint32_t cols = 0;
    read_exact(input, &rows, sizeof(rows), ext + " row count from " + path);
    read_exact(input, &cols, sizeof(cols), ext + " dimension from " + path);
    if (rows == 0 || cols == 0) { fail("invalid " + ext + " header in " + path); }

    input.seekg(0, std::ios::end);
    const std::streamoff file_size = input.tellg();
    const uint64_t record_bytes =
      checked_u64_product(cols, is_float ? sizeof(float) : sizeof(uint8_t), ext + " row");
    const uint64_t payload_bytes = checked_u64_product(rows, record_bytes, ext + " payload");
    const uint64_t expected_bytes = checked_u64_sum(2 * sizeof(uint32_t),
                                                   payload_bytes,
                                                   ext + " file");
    if (file_size < 0 || static_cast<uint64_t>(file_size) < expected_bytes) {
      fail(ext + " payload is truncated in " + path);
    }
    if (is_float && static_cast<uint64_t>(file_size) != expected_bytes) {
      fail("fbin file size does not match header in " + path);
    }

    metadata.total_rows = rows;
    metadata.rows = effective_rows_from_limit(metadata.total_rows, row_limit, path);
    metadata.cols = cols;
    metadata.data_offset = 2 * sizeof(uint32_t);
    metadata.record_bytes = record_bytes;
    metadata.per_row_dim_header = false;
    metadata.scalar_type = is_float ? InputScalarType::Float32 : InputScalarType::Uint8;
    return metadata;
  }

  fail("unsupported vector format " + ext + " for " + path +
       " (supported: .bvecs, .u8bin, .ibin, .bin, .fbin, .fvecs, .hvecs)");
}

uint64_t row_offset_bytes(const VectorFileMetadata& metadata, uint32_t row)
{
  return checked_u64_sum(metadata.data_offset,
                         checked_u64_product(row, metadata.record_bytes, "row offset"),
                         "row offset");
}

void require_finite(const std::vector<float>& values);

class StreamingVectorReader {
 public:
  explicit StreamingVectorReader(VectorFileMetadata metadata)
    : metadata_(std::move(metadata)), input_(metadata_.path, std::ios::binary)
  {
    if (!input_) { fail("unable to open vector file " + metadata_.path); }
  }

  void read_rows_as_float(uint32_t row_begin, uint32_t row_count, std::vector<float>& out)
  {
    if (row_begin > metadata_.rows || row_count > metadata_.rows - row_begin) {
      fail("requested row chunk is outside selected vector prefix");
    }
    out.resize(checked_element_count(row_count, metadata_.cols, "streaming row chunk"));
    if (row_count == 0) { return; }

    if (!metadata_.per_row_dim_header && metadata_.scalar_type == InputScalarType::Float32) {
      seek_to(row_offset_bytes(metadata_, row_begin), "fbin chunk from " + metadata_.path);
      read_exact(input_,
                 out.data(),
                 checked_size_product(out.size(), sizeof(float), "fbin chunk"),
                 "fbin chunk from " + metadata_.path);
      require_finite(out);
      return;
    }

    if (!metadata_.per_row_dim_header && metadata_.scalar_type == InputScalarType::Uint8) {
      const size_t bytes = checked_element_count(row_count, metadata_.cols, "u8bin chunk");
      std::vector<uint8_t> tmp(bytes);
      seek_to(row_offset_bytes(metadata_, row_begin), "u8bin chunk from " + metadata_.path);
      read_exact(input_, tmp.data(), tmp.size(), "u8bin chunk from " + metadata_.path);
      for (size_t i = 0; i < tmp.size(); ++i) { out[i] = static_cast<float>(tmp[i]); }
      return;
    }

    for (uint32_t local = 0; local < row_count; ++local) {
      read_row_as_float(row_begin + local,
                        out.data() + static_cast<size_t>(local) * metadata_.cols);
    }
    require_finite(out);
  }

  void read_selected_rows_as_float(const std::vector<uint32_t>& row_ids, std::vector<float>& out)
  {
    out.resize(checked_size_product(row_ids.size(), metadata_.cols, "selected row chunk"));
    for (size_t i = 0; i < row_ids.size(); ++i) {
      read_row_as_float(row_ids[i], out.data() + i * metadata_.cols);
    }
    require_finite(out);
  }

 private:
  void seek_to(uint64_t offset, const std::string& description)
  {
    input_.clear();
    if (offset > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max())) {
      fail(description + " offset exceeds streamoff range");
    }
    input_.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    if (!input_) { fail("failed to seek " + description); }
  }

  void read_row_as_float(uint32_t row, float* dst)
  {
    if (row >= metadata_.rows) {
      fail("requested vector row " + std::to_string(row) + " outside selected prefix");
    }

    seek_to(row_offset_bytes(metadata_, row), "row " + std::to_string(row) + " from " +
                                             metadata_.path);
    if (metadata_.per_row_dim_header) {
      uint32_t row_dim = 0;
      read_exact(input_, &row_dim, sizeof(row_dim), "row dimension from " + metadata_.path);
      if (row_dim != metadata_.cols) {
        fail("inconsistent vector dimension at row " + std::to_string(row) + " in " +
             metadata_.path + ": got " + std::to_string(row_dim) + " expected " +
             std::to_string(metadata_.cols));
      }
    }

    if (metadata_.scalar_type == InputScalarType::Float32) {
      read_exact(input_,
                 dst,
                 static_cast<size_t>(metadata_.cols) * sizeof(float),
                 "float32 row payload from " + metadata_.path);
      return;
    }
    if (metadata_.scalar_type == InputScalarType::Float16) {
      std::vector<uint16_t> half_row(metadata_.cols);
      read_exact(input_,
                 half_row.data(),
                 static_cast<size_t>(metadata_.cols) * sizeof(uint16_t),
                 "hvecs row payload from " + metadata_.path);
      convert_half_row_to_float(half_row.data(), dst, metadata_.cols);
      return;
    }

    std::vector<uint8_t> tmp(metadata_.cols);
    read_exact(input_, tmp.data(), tmp.size(), "uint8 row payload from " + metadata_.path);
    for (uint32_t d = 0; d < metadata_.cols; ++d) { dst[d] = static_cast<float>(tmp[d]); }
  }

  VectorFileMetadata metadata_;
  std::ifstream input_;
};

void validate_selected_row_dimensions(const VectorFileMetadata& metadata)
{
  if (!metadata.per_row_dim_header) { return; }

  std::ifstream input(metadata.path, std::ios::binary);
  if (!input) { fail("unable to open vector file " + metadata.path); }

  for (uint32_t row = 0; row < metadata.rows; ++row) {
    const uint64_t offset = row_offset_bytes(metadata, row);
    if (offset > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max())) {
      fail("row " + std::to_string(row) + " offset exceeds streamoff range in " + metadata.path);
    }
    input.clear();
    input.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    if (!input) {
      fail("failed to seek row " + std::to_string(row) + " in " + metadata.path);
    }
    uint32_t row_dim = 0;
    read_exact(input, &row_dim, sizeof(row_dim), "row dimension from " + metadata.path);
    if (row_dim != metadata.cols) {
      fail("inconsistent vector dimension at row " + std::to_string(row) + " in " +
           metadata.path + ": got " + std::to_string(row_dim) + " expected " +
           std::to_string(metadata.cols));
    }
  }
}

const char* scalar_type_name(InputScalarType type)
{
  switch (type) {
    case InputScalarType::Uint8: return "uint8";
    case InputScalarType::Float32: return "float32";
    case InputScalarType::Float16: return "float16";
  }
  return "unknown";
}


std::vector<uint32_t> sample_training_rows(uint32_t rows, uint32_t count, uint64_t seed)
{
  if (count == 0 || count > rows) { fail("invalid training row count"); }
  std::mt19937_64 rng(seed);
  std::unordered_set<uint32_t> selected;
  selected.reserve(count);
  for (uint64_t j = uint64_t(rows) - count; j < rows; ++j) {
    const uint32_t candidate = std::uniform_int_distribution<uint32_t>(0, j)(rng);
    if (!selected.insert(candidate).second) { selected.insert(static_cast<uint32_t>(j)); }
  }
  std::vector<uint32_t> result(selected.begin(), selected.end());
  std::sort(result.begin(), result.end());
  return result;
}

void require_finite(const std::vector<float>& values)
{
  for (float value : values) { if (!std::isfinite(value)) { fail("input contains NaN or infinity"); } }
}
}  // namespace go_fann_quantization
