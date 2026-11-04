#include "execution/async_io.h"
#include "data/dataloader.h"
#include "data/fp16_l2.h"
#include "data/vector_file_reader.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <fcntl.h>
#include <unistd.h>

AsyncIoConfig::AsyncIoConfig(uint32_t dim_value,
                             uint32_t degree_value,
                             uint32_t prefetch_hops_value,
                             size_t block_size_value,
                             size_t header_size_value)
    : dim(dim_value),
      degree(degree_value),
      prefetch_hops(prefetch_hops_value),
      block_size(block_size_value),
      header_size(header_size_value)
{
}

void AsyncIoConfig::set_reserve_vector_payload(bool reserve)
{
    reserve_vector_payload = reserve;
}

void build_prefetch_slices(GraphRuntime&, const AsyncIoConfig&)
{
}

namespace {

void require(bool condition, const char* message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_near(float actual, float expected, float tolerance, const char* message)
{
    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(message);
    }
}

template <typename Function>
void require_throws_with_message(Function&& function,
                                 const char* expected_text,
                                 const char* message)
{
    try {
        function();
    } catch (const std::exception& error) {
        require(std::string(error.what()).find(expected_text) != std::string::npos,
                message);
        return;
    }
    throw std::runtime_error(message);
}

std::string temporary_path(const char* suffix)
{
    char path[] = "/tmp/flashanns_rerank_store_XXXXXX";
    const int fd = mkstemp(path);
    require(fd >= 0, "mkstemp failed");
    close(fd);
    std::string result(path);
    std::remove(result.c_str());
    return result + suffix;
}

void write_hvecs_fixture(const std::string& path)
{
    constexpr uint32_t cols = 4;
    const std::array<std::array<uint16_t, cols>, 3> rows{{
        {{0x0000, 0x0000, 0x0000, 0x0000}},
        {{0x3c00, 0xc000, 0x3800, 0x4400}},
        {{0xbc00, 0x4200, 0xb800, 0x4000}},
    }};
    std::ofstream output(path, std::ios::binary);
    require(output.is_open(), "failed to create hvecs fixture");
    for (const auto& row : rows) {
        output.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
        output.write(reinterpret_cast<const char*>(row.data()),
                     static_cast<std::streamsize>(row.size() * sizeof(uint16_t)));
    }
    require(output.good(), "failed to write hvecs fixture");
}

void write_fvecs_fixture(const std::string& path, uint32_t cols)
{
    std::ofstream output(path, std::ios::binary);
    require(output.is_open(), "failed to create fvecs fixture");
    std::vector<float> row(cols);
    for (uint32_t row_id = 0; row_id < 3; ++row_id) {
        for (uint32_t col = 0; col < cols; ++col) {
            const int pattern = static_cast<int>((col * 17 + row_id * 11) % 61) - 30;
            row[col] = static_cast<float>(pattern) * 0.0625f +
                static_cast<float>(row_id) * 0.03125f;
        }
        output.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
        output.write(reinterpret_cast<const char*>(row.data()),
                     static_cast<std::streamsize>(row.size() * sizeof(float)));
    }
    require(output.good(), "failed to write fvecs fixture");
}

std::vector<uint8_t> read_all(const std::string& path)
{
    std::ifstream input(path, std::ios::binary);
    require(input.is_open(), "failed to open artifact for comparison");
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

void test_graph_artifact_format_detection()
{
    const auto path = temporary_path(".graph.bin");
    const auto legacy = temporary_path(".ibin");
    const std::vector<uint32_t> expected{1, 2, 0, 2, 0, 1};
    try {
        {
            go_fann::graph_file::GraphFileSequentialWriter writer(path, 3, 2);
            writer.append_rows(expected.data(), 3);
            writer.close();
        }
        auto graph = load_graph_matrix_auto(path);
        require(graph.rows == 3 && graph.cols == 2 && graph.data == expected,
                "arbitrary graph artifact filename did not load by magic");
        {
            std::ofstream out(legacy, std::ios::binary);
            const uint64_t rows = 3, cols = 2;
            out.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
            out.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
            out.write(reinterpret_cast<const char*>(expected.data()), expected.size() * sizeof(uint32_t));
        }
        graph = load_graph_matrix_auto(legacy);
        require(graph.rows == 3 && graph.cols == 2 && graph.data == expected,
                "legacy graph matrix loading changed");
        {
            std::fstream out(path, std::ios::binary | std::ios::in | std::ios::out);
            const uint32_t version = 99;
            out.seekp(sizeof(uint32_t));
            out.write(reinterpret_cast<const char*>(&version), sizeof(version));
        }
        require_throws_with_message([&] { (void)load_graph_matrix_auto(path); },
            "unsupported", "unknown graph version silently treated as matrix");
    } catch (...) { std::remove(path.c_str()); std::remove(legacy.c_str()); throw; }
    std::remove(path.c_str()); std::remove(legacy.c_str());
}

void test_streamed_vector_input()
{
    const auto half_path = temporary_path(".hvecs");
    const auto float_path = temporary_path(".fvecs");
    const auto byte_path = temporary_path(".u8bin");
    try {
        write_hvecs_fixture(half_path);
        write_fvecs_fixture(float_path, 4);
        {
            std::ofstream out(byte_path, std::ios::binary);
            uint32_t rows = 2, cols = 3;
            const uint8_t values[] = {1, 2, 3, 9, 8, 7};
            out.write(reinterpret_cast<const char*>(&rows), 4);
            out.write(reinterpret_cast<const char*>(&cols), 4);
            out.write(reinterpret_cast<const char*>(values), sizeof(values));
        }
        VectorFileReader half(inspect_vector_file(half_path));
        std::array<float, 4> values{};
        half.read_float_row(2, values.data(), 4);
        require(values == std::array<float, 4>{-1, 3, -0.5f, 2}, "streamed FP16 row wrong");
        half.read_float_row(1, values.data(), 4);
        require(values == std::array<float, 4>{1, -2, 0.5f, 4}, "streamed backward seek wrong");
        require_throws_with_message([&] { half.read_float_row(3, values.data(), 4); },
            "invalid row", "streamed invalid row accepted");
        require_throws_with_message([&] { half.read_float_row(0, values.data(), 3); },
            "invalid row", "streamed invalid dimension accepted");
        VectorFileReader floats(inspect_vector_file(float_path));
        const auto resident = load_float_matrix_auto(float_path);
        floats.read_float_row(2, values.data(), 4);
        for (size_t col = 0; col < 4; ++col)
            require(values[col] == resident.row_ptr(2)[col], "streamed FP32 differs from resident");
        VectorFileReader bytes(inspect_vector_file(byte_path));
        std::array<uint8_t, 3> row{};
        bytes.read_uint8_row(1, row.data(), 3);
        require(row == std::array<uint8_t, 3>{9, 8, 7}, "streamed uint8 header offset wrong");
        const auto metadata = inspect_vector_file(half_path);
        auto overflow = metadata;
        overflow.row_data_offset = UINT64_MAX - 4;
        require_throws_with_message([&] { VectorFileReader bad(overflow); },
            "offset overflow", "streamed overflow metadata accepted");
        std::filesystem::resize_file(half_path, 7);
        require_throws_with_message([&] { VectorFileReader bad(metadata); },
            "truncated", "streamed truncated file accepted");
        write_hvecs_fixture(half_path);
        {
            std::fstream corrupt(half_path, std::ios::binary | std::ios::in | std::ios::out);
            uint32_t invalid_dim = 3;
            corrupt.seekp(12);
            corrupt.write(reinterpret_cast<const char*>(&invalid_dim), 4);
        }
        VectorFileReader malformed(metadata);
        require_throws_with_message([&] { malformed.read_float_row(1, values.data(), 4); },
            "dimension mismatch", "streamed malformed prefix accepted");
    } catch (...) {
        std::remove(half_path.c_str()); std::remove(float_path.c_str()); std::remove(byte_path.c_str());
        throw;
    }
    std::remove(half_path.c_str()); std::remove(float_path.c_str()); std::remove(byte_path.c_str());
}

void test_fp16_mmap_and_memory_int8()
{
    const std::string source_path = temporary_path(".hvecs");
    const std::string artifact_a = temporary_path(".int8.bin");
    const std::string artifact_b = temporary_path(".int8.bin");
    const std::string truncated_artifact = temporary_path(".truncated.bin");
    const std::string malformed_source = temporary_path(".malformed.hvecs");
    const std::string malformed_artifact = temporary_path(".malformed.int8.bin");
    write_hvecs_fixture(source_path);

    try {
        const VectorFileInfo info = inspect_vector_file(source_path);
        require(info.rows == 3 && info.cols == 4,
                "hvecs fixture shape was inspected incorrectly");

        const auto fp16 = RerankVectorStore::map_fp16(info);
        require(fp16->mode() == "mmap_fp16", "FP16 store mode is incorrect");
        require(fp16->mapped_bytes() == std::filesystem::file_size(source_path),
                "FP16 mapping does not cover the source file");
        require(fp16->resident_payload_bytes() == 0,
                "FP16 mapping was reported as anonymous resident payload");
        std::array<float, 4> row{};
        fp16->read_float_row(1, row.data(), row.size());
        const std::array<float, 4> expected{1.0f, -2.0f, 0.5f, 4.0f};
        for (size_t col = 0; col < row.size(); ++col) {
            require_near(row[col], expected[col], 0.0f,
                         "FP16 mmap row conversion changed a value");
        }
        const std::array<float, 4> zero_query{};
        require_near(fp16->l2_float(1, zero_query.data(), zero_query.size()),
                     21.25f,
                     0.0f,
                     "FP16 mmap exact L2 is incorrect");

        // GPU-post gathers reordered candidate rows into bounded FP32 staging
        // even though the full resident FP32 matrix has never been loaded.
        GraphRuntime runtime;
        runtime.vector_scalar_type = DenseVectorScalarType::Float16;
        runtime.base_vectors.rows = fp16->rows();
        runtime.base_vectors.cols = fp16->cols();
        runtime.rerank_vector_store = fp16;
        const std::array<uint32_t, 4> candidates{2, 0, 1, 2};
        std::array<float, 18> staged{};
        staged.front() = staged.back() = -123.0f;
        for (size_t slot = 0; slot < candidates.size(); ++slot) {
            copy_base_float_vector_row(runtime, candidates[slot],
                                       staged.data() + 1 + slot * 4, 4);
            float squared_norm = 0.0f;
            for (size_t col = 0; col < 4; ++col) {
                const float value = staged[1 + slot * 4 + col];
                squared_norm += value * value;
            }
            require_near(squared_norm, fp16->l2_float(candidates[slot],
                zero_query.data(), zero_query.size()), 0.0f,
                "candidate staging changed the FP16 exact distance");
        }
        require(staged.front() == -123.0f && staged.back() == -123.0f,
                "candidate row gathering overwrote staging bounds");
        require(runtime.base_float_vectors.data.empty(),
                "candidate staging materialized a full FP32 matrix");
        require_throws_with_message([&] {
            copy_base_float_vector_row(runtime, 3, staged.data() + 1, 4);
        }, "row out of range", "out-of-range candidate was gathered");
        require_throws_with_message([&] {
            copy_base_float_vector_row(runtime, 1, staged.data() + 1, 3);
        }, "dimension mismatch", "wrong staging dimension was accepted");

        const Int8RerankBuildStats stats_a = build_int8_rerank_artifact(
            source_path, artifact_a);
        const Int8RerankBuildStats stats_b = build_int8_rerank_artifact(
            source_path, artifact_b);
        require(stats_a.rows == 3 && stats_a.cols == 4 &&
                    stats_a.artifact_bytes == stats_b.artifact_bytes,
                "INT8 artifact build statistics are incorrect");
        require(read_all(artifact_a) == read_all(artifact_b),
                "INT8 artifact build is not deterministic");

        const DenseVectorMetadata metadata{
            source_path, DenseVectorScalarType::Float16, 3, 4};
        const auto int8 = RerankVectorStore::load_memory_int8(artifact_a, metadata);
        require(int8->mode() == "memory_int8", "INT8 store mode is incorrect");
        require(int8->mapped_bytes() == 0,
                "INT8 anonymous memory was reported as a file mapping");
        require(int8->resident_payload_bytes() >= 3 * 4,
                "INT8 resident payload size is too small");
        int8->read_float_row(1, row.data(), row.size());
        const float quantization_tolerance = 4.0f / 127.0f / 2.0f + 1e-6f;
        float expected_l2 = 0.0f;
        for (size_t col = 0; col < row.size(); ++col) {
            require_near(row[col], expected[col], quantization_tolerance,
                         "INT8 dequantization error exceeds half a quantization step");
            expected_l2 += row[col] * row[col];
        }
        require_near(int8->l2_float(1, zero_query.data(), zero_query.size()),
                     expected_l2,
                     1e-5f,
                     "INT8 approximate L2 does not match its decoded row");
        int8->read_float_row(0, row.data(), row.size());
        for (float value : row) {
            require(value == 0.0f, "zero row did not survive INT8 quantization");
        }

        std::filesystem::copy_file(artifact_a, truncated_artifact);
        std::filesystem::resize_file(
            truncated_artifact, std::filesystem::file_size(truncated_artifact) - 1);
        require_throws_with_message(
            [&] {
                (void)RerankVectorStore::load_memory_int8(truncated_artifact, metadata);
            },
            "file size",
            "truncated INT8 artifact was accepted");

        write_hvecs_fixture(malformed_source);
        std::fstream malformed(malformed_source,
                               std::ios::binary | std::ios::in | std::ios::out);
        require(malformed.is_open(), "failed to open malformed hvecs fixture");
        constexpr uint32_t invalid_cols = 3;
        malformed.seekp(sizeof(uint32_t) + 4 * sizeof(uint16_t));
        malformed.write(reinterpret_cast<const char*>(&invalid_cols),
                        sizeof(invalid_cols));
        malformed.close();
        require_throws_with_message(
            [&] {
                (void)build_int8_rerank_artifact(malformed_source,
                                                 malformed_artifact);
            },
            "invalid dimension",
            "INT8 builder accepted a malformed row dimension");

        std::ofstream source_append(source_path, std::ios::binary | std::ios::app);
        source_append.put('\0');
        source_append.close();
        require_throws_with_message(
            [&] {
                (void)RerankVectorStore::load_memory_int8(artifact_a, metadata);
            },
            "source file identity",
            "INT8 artifact was accepted after its source file changed");
    } catch (...) {
        std::remove(source_path.c_str());
        std::remove(artifact_a.c_str());
        std::remove(artifact_b.c_str());
        std::remove(truncated_artifact.c_str());
        std::remove(malformed_source.c_str());
        std::remove(malformed_artifact.c_str());
        throw;
    }

    std::remove(source_path.c_str());
    std::remove(artifact_a.c_str());
    std::remove(artifact_b.c_str());
    std::remove(truncated_artifact.c_str());
    std::remove(malformed_source.c_str());
    std::remove(malformed_artifact.c_str());
}

void test_resident_vector_access()
{
    const auto uint8_path = temporary_path(".u8bin");
    const auto float_path = temporary_path(".fbin");
    try {
        const uint32_t rows = 2, cols = 3;
        const std::array<uint8_t, 6> bytes{0, 1, 255, 9, 7, 5};
        const std::array<float, 6> values{0.0f, 1.0f, -2.0f, 3.0f, -4.0f, 0.5f};
        {
            std::ofstream out(uint8_path, std::ios::binary);
            out.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
            out.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
            out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        }
        {
            std::ofstream out(float_path, std::ios::binary);
            out.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
            out.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
            out.write(reinterpret_cast<const char*>(values.data()), sizeof(values));
        }
        GraphRuntime u8;
        u8.base_vectors = load_uint8_matrix_auto(uint8_path);
        std::array<uint8_t, 3> byte_row{};
        copy_base_vector_row(u8, 0, byte_row.data(), cols);
        require(std::equal(byte_row.begin(), byte_row.end(), bytes.begin()),
                "resident uint8 row changed values");
        copy_lowfreq_base_vector_row(u8, 1, byte_row.data(), cols);
        require(std::equal(byte_row.begin(), byte_row.end(), bytes.begin() + cols),
                "low-frequency uint8 row changed values");
        require_throws_with_message([&] { copy_base_vector_row(u8, rows, byte_row.data(), cols); },
            "row out of range", "resident uint8 accepted invalid row");
        require_throws_with_message([&] { copy_base_vector_row(u8, 0, byte_row.data(), cols - 1); },
            "dimension mismatch", "resident uint8 accepted invalid dimension");

        GraphRuntime fp32;
        fp32.vector_scalar_type = DenseVectorScalarType::Float32;
        fp32.base_float_vectors = load_float_matrix_auto(float_path);
        std::array<float, 3> float_row{};
        copy_base_float_vector_row(fp32, 0, float_row.data(), cols);
        require(std::equal(float_row.begin(), float_row.end(), values.begin()),
                "resident float row changed values");
        copy_lowfreq_base_float_vector_row(fp32, 1, float_row.data(), cols);
        require(std::equal(float_row.begin(), float_row.end(), values.begin() + cols),
                "low-frequency float row changed values");
        const std::array<float, 3> query{1.0f, 2.0f, 3.0f};
        const auto prepared = prepare_base_float_vector_l2_query(fp32, query.data(), cols);
        require_near(compute_base_float_vector_l2(fp32, 0, prepared), 27.0f, 0.0f,
                     "resident float exact distance changed");
        require_throws_with_message([&] { (void)compute_base_float_vector_l2(fp32, rows, prepared); },
            "row out of range", "resident float scoring accepted invalid row");
        require_throws_with_message([&] { (void)compute_base_float_vector_l2(fp32, 0, {query.data(), 2, 0}); },
            "dimension mismatch", "resident float scoring accepted invalid dimension");
        fp32.base_float_vectors.data.clear();
        require_throws_with_message([&] { (void)compute_base_float_vector_l2(fp32, 0, prepared); },
            "payload is unavailable", "shape-only resident matrix was scored");
    } catch (...) {
        std::remove(uint8_path.c_str()); std::remove(float_path.c_str()); throw;
    }
    std::remove(uint8_path.c_str()); std::remove(float_path.c_str());

    GraphLoadConfig unsupported;
    unsupported.graph_storage_mode = "disk";
    require_throws_with_message([&] { (void)load_graph_runtime(unsupported); },
        "disk runtime is unsupported", "disk graph mode reached input loading");
}

void test_fp16_mapping_validation()
{
    const auto path = temporary_path(".hvecs");
    try {
        write_hvecs_fixture(path);
        const auto info = inspect_vector_file(path);
        auto malformed = info;
        malformed.cols += 1;
        require_throws_with_message([&] { (void)RerankVectorStore::map_fp16(malformed); },
            "requires a strided", "FP16 mapping accepted inconsistent dimension");
        malformed = info;
        malformed.rows += 1;
        require_throws_with_message([&] { (void)RerankVectorStore::map_fp16(malformed); },
            "file size", "FP16 mapping accepted an extent beyond the file");
        malformed = info;
        malformed.path += ".missing";
        require_throws_with_message([&] { (void)RerankVectorStore::map_fp16(malformed); },
            "unable to open", "FP16 mapping accepted a missing file");

        const auto limited = inspect_vector_file(path, 2);
        auto store = RerankVectorStore::map_fp16(limited);
        require(store->rows() == 2, "FP16 mapping lost row limit");
        GraphRuntime primary, secondary;
        primary.vector_scalar_type = secondary.vector_scalar_type = DenseVectorScalarType::Float16;
        primary.rerank_vector_store = store;
        secondary.rerank_vector_store = primary.rerank_vector_store;
        const std::array<float, 4> query{};
        const auto prepared = prepare_base_float_vector_l2_query(secondary, query.data(), query.size());
        require_near(compute_base_float_vector_l2(secondary, 1, prepared), 21.25f, 0.0f,
            "shared FP16 store changed exact distance");
        std::array<float, 4> row{};
        copy_lowfreq_base_float_vector_row(secondary, 1, row.data(), row.size());
        require(row[0] == 1.0f && row[3] == 4.0f,
            "shared FP16 low-frequency access changed row");
        require(secondary.base_float_vectors.data.empty(), "shared FP16 store loaded resident vectors");
        store.reset(); primary.rerank_vector_store.reset(); secondary.rerank_vector_store.reset();

        std::filesystem::resize_file(path, std::filesystem::file_size(path) - 1);
        require_throws_with_message([&] { (void)RerankVectorStore::map_fp16(info); },
            "file size", "FP16 mapping accepted truncated input after inspection");
        require_throws_with_message([&] { (void)inspect_vector_file(path); },
            "row-aligned", "vector inspection accepted truncated FP16 input");
    } catch (...) { std::remove(path.c_str()); throw; }
    std::remove(path.c_str());
}

void test_int8_prepared_query_dimensions()
{
    const std::array<uint32_t, 3> dimensions{37, 512, 4096};
    for (uint32_t cols : dimensions) {
        const std::string source_path = temporary_path(".fvecs");
        const std::string artifact_path = temporary_path(".int8.bin");
        write_fvecs_fixture(source_path, cols);
        try {
            (void)build_int8_rerank_artifact(source_path, artifact_path);
            const DenseVectorMetadata metadata{
                source_path, DenseVectorScalarType::Float32, 3, cols};
            const auto store = RerankVectorStore::load_memory_int8(
                artifact_path, metadata);
            require(store->distance_kernel() == "int8_scalar_l2" ||
                        store->distance_kernel() == "int8_avx512_dot",
                    "INT8 store selected an unknown distance kernel");

            std::vector<float> query(cols);
            for (uint32_t col = 0; col < cols; ++col) {
                const int pattern = static_cast<int>((col * 13 + 7) % 47) - 23;
                query[col] = static_cast<float>(pattern) * 0.078125f;
            }
            const PreparedFloatL2Query prepared = store->prepare_l2_query(
                query.data(), cols);
            require(prepared.values == query.data() && prepared.cols == cols,
                    "prepared INT8 query lost its source pointer or dimension");

            std::vector<float> decoded(cols);
            for (uint32_t row = 0; row < 3; ++row) {
                store->read_float_row(row, decoded.data(), cols);
                float scalar_l2 = 0.0f;
                for (uint32_t col = 0; col < cols; ++col) {
                    const float diff = query[col] - decoded[col];
                    scalar_l2 += diff * diff;
                }
                const float actual = store->l2_float(row, prepared);
                const float tolerance = std::max(1e-3f, std::abs(scalar_l2) * 2e-5f);
                require_near(actual,
                             scalar_l2,
                             tolerance,
                             "prepared INT8 L2 differs from decoded scalar L2");
            }
        } catch (...) {
            std::remove(source_path.c_str());
            std::remove(artifact_path.c_str());
            throw;
        }
        std::remove(source_path.c_str());
        std::remove(artifact_path.c_str());
    }
}

void test_fp16_batch_dispatch()
{
    struct SavedEnv {
        const char* key;
        std::optional<std::string> value;
        explicit SavedEnv(const char* name) : key(name) {
            if (const char* v = std::getenv(key)) value = v;
        }
        ~SavedEnv() {
            if (value) setenv(key, value->c_str(), 1);
            else unsetenv(key);
        }
    } kernel_env("GO_FANN_FP16_RERANK_KERNEL"), batch_env("GO_FANN_FP16_RERANK_BATCH"),
      prefetch_env("GO_FANN_FP16_RERANK_PREFETCH");
    const std::string path = temporary_path(".batch.hvecs");
    constexpr uint32_t cols = 37, count = 19;
    {
        std::ofstream file(path, std::ios::binary);
        for (uint32_t row = 0; row < count; ++row) {
            file.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
            for (uint32_t col = 0; col < cols; ++col) {
                const uint16_t bits = uint16_t(0x2000 + (row * 7 + col * 11) % 0x2400);
                file.write(reinterpret_cast<const char*>(&bits), sizeof(bits));
            }
        }
    }
    const auto info = inspect_vector_file(path);
    const std::array<uint32_t, 11> ids{7, 2, 7, 18, 1, 8, 12, 6, 13, 3, 14};
    const std::vector<float> query(cols, 0.125f);
    try {
        for (const std::string kernel : {"scalar", "avx2", "avx512"}) {
            using namespace go_fann::fp16_l2;
            if ((kernel == "avx2" && !supported(Kernel::Avx2)) ||
                (kernel == "avx512" && !supported(Kernel::Avx512))) continue;
            setenv(kernel_env.key, kernel.c_str(), 1);
            for (const char* width : {"1", "4", "8"}) {
                setenv(batch_env.key, width, 1);
                setenv(prefetch_env.key, "8", 1);
                const auto store = RerankVectorStore::map_fp16(info);
                const auto prepared = store->prepare_l2_query(query.data(), cols);
                std::array<float, ids.size()> distances{};
                store->l2_float_batch(ids.data(), ids.size(), prepared, distances.data());
                for (size_t i = 0; i < ids.size(); ++i)
                    require_near(distances[i], store->l2_float(ids[i], prepared), 0.0f,
                                 "FP16 batch altered row order or reduction");
                store->l2_float_batch(nullptr, 0, {}, nullptr);
                require_throws_with_message([&] { store->l2_float_batch(nullptr, 1, prepared, distances.data()); },
                    "invalid batched", "null batch input accepted");
                uint32_t invalid = count;
                require_throws_with_message([&] { store->l2_float_batch(&invalid, 1, prepared, distances.data()); },
                    "out of range", "invalid batch row accepted");
            }
        }
        setenv(batch_env.key, "3", 1);
        require_throws_with_message([&] { (void)RerankVectorStore::map_fp16(info); },
            "must be 1, 4 or 8", "invalid batch width accepted");
    } catch (...) { std::remove(path.c_str()); throw; }
    std::remove(path.c_str());
}

}  // namespace

int main()
{
    try {
        test_graph_artifact_format_detection();
        test_streamed_vector_input();
        test_resident_vector_access();
        test_fp16_mapping_validation();
        test_fp16_mmap_and_memory_int8();
        test_int8_prepared_query_dimensions();
        test_fp16_batch_dispatch();
        std::puts("rerank vector store tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "rerank vector store test failed: %s\n", error.what());
        return 1;
    }
}
