#include "index/graph_file_format.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace go_fann::graph_file {

namespace {

[[noreturn]] void fail_graph_file(const std::string& message)
{
    throw std::runtime_error("graph file format failed: " + message);
}

size_t compute_rows_per_block(uint32_t row_size_bytes)
{
    if (row_size_bytes == 0 || row_size_bytes > kGraphFileBlockSize) {
        return 0;
    }
    return kGraphFileBlockSize / row_size_bytes;
}

uint64_t checked_mul_u64(uint64_t lhs, uint64_t rhs, const std::string& context)
{
    if (rhs != 0 && lhs > std::numeric_limits<uint64_t>::max() / rhs) {
        fail_graph_file(context + " overflows uint64_t");
    }
    return lhs * rhs;
}

uint64_t checked_add_u64(uint64_t lhs, uint64_t rhs, const std::string& context)
{
    if (lhs > std::numeric_limits<uint64_t>::max() - rhs) {
        fail_graph_file(context + " overflows uint64_t");
    }
    return lhs + rhs;
}

uint64_t block_count_for_rows(const GraphFileHeader& header)
{
    if (header.rows == 0) {
        return 0;
    }
    return (header.rows / header.rows_per_block) +
        ((header.rows % header.rows_per_block) == 0 ? 0 : 1);
}

std::streamoff checked_stream_offset(uint64_t offset, const std::string& path)
{
    if (offset > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max())) {
        fail_graph_file("file offset exceeds streamoff range in " + path);
    }
    return static_cast<std::streamoff>(offset);
}

void validate_seekable_file_span(const GraphFileHeader& header, const std::string& path)
{
    const uint64_t block_count = block_count_for_rows(header);
    const uint64_t payload_bytes =
        checked_mul_u64(block_count, header.block_size, "graph file payload byte count");
    const uint64_t total_bytes =
        checked_add_u64(header.header_size, payload_bytes, "graph file file byte count");
    checked_stream_offset(total_bytes, path);
}

}

bool has_graph_file_magic(const std::string& path)
{
    std::ifstream input(path, std::ios::binary);
    uint32_t magic = 0;
    input.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (!input) fail_graph_file("unable to read format marker from " + path);
    return magic == kGraphFileMagic;
}

GraphFileHeader read_graph_file_header(const std::string& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        fail_graph_file("unable to open graph file " + path);
    }

    GraphFileHeader header;
    input.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!input) {
        fail_graph_file("failed to read graph file header from " + path);
    }

    if (header.magic != kGraphFileMagic) {
        fail_graph_file("invalid magic in graph file " + path);
    }
    if (header.version != kGraphFileVersion) {
        fail_graph_file("unsupported graph file version in " + path);
    }
    if (header.block_size != kGraphFileBlockSize || header.header_size != kGraphFileHeaderSize) {
        fail_graph_file("unexpected block/header size in " + path);
    }
    if (header.degree == 0) {
        fail_graph_file("degree must be > 0 in " + path);
    }
    if (header.row_size_bytes != header.degree * sizeof(uint32_t)) {
        fail_graph_file("row_size_bytes does not match degree in " + path);
    }
    const size_t rows_per_block = compute_rows_per_block(header.row_size_bytes);
    if (rows_per_block == 0 || header.rows_per_block != rows_per_block) {
        fail_graph_file("rows_per_block is invalid in " + path);
    }
    return header;
}

uint64_t graph_file_row_offset_bytes(const GraphFileHeader& header, uint64_t global_row)
{
    const uint64_t block_idx = global_row / header.rows_per_block;
    const uint64_t row_in_block = global_row % header.rows_per_block;
    return static_cast<uint64_t>(header.header_size) +
        block_idx * static_cast<uint64_t>(header.block_size) +
        row_in_block * static_cast<uint64_t>(header.row_size_bytes);
}

ContiguousGraphMatrix load_graph_file_as_contiguous_matrix(const std::string& path)
{
    const GraphFileHeader header = read_graph_file_header(path);
    validate_seekable_file_span(header, path);
    const uint64_t checked_rows = std::max<uint64_t>(header.rows, 1);
    if (header.degree > std::numeric_limits<size_t>::max() / checked_rows) {
        fail_graph_file("graph payload is too large to index in memory for " + path);
    }

    std::ifstream input(path, std::ios::binary);
    if (!input) {
        fail_graph_file("unable to open graph file " + path);
    }

    ContiguousGraphMatrix matrix;
    matrix.rows = header.rows;
    matrix.cols = header.degree;
    const size_t word_count = static_cast<size_t>(matrix.rows) * static_cast<size_t>(matrix.cols);
    matrix.data.resize(word_count);

    std::vector<char> block(kGraphFileBlockSize);
    uint64_t next_row = 0;
    while (next_row < header.rows) {
        const uint64_t block_idx = next_row / header.rows_per_block;
        const uint64_t block_row_begin = block_idx * header.rows_per_block;
        const uint64_t rows_in_block = std::min<uint64_t>(header.rows_per_block,
                                                         header.rows - block_row_begin);
        const uint64_t block_payload_offset =
            checked_mul_u64(block_idx, header.block_size, "graph file block offset");
        const uint64_t block_file_offset =
            checked_add_u64(header.header_size, block_payload_offset, "graph file block file offset");
        input.seekg(checked_stream_offset(block_file_offset, path), std::ios::beg);
        if (!input) {
            fail_graph_file("failed seeking graph file block in " + path);
        }
        input.read(block.data(), static_cast<std::streamsize>(block.size()));
        if (!input) {
            fail_graph_file("failed reading graph file block from " + path);
        }

        for (uint64_t row_in_block = 0; row_in_block < rows_in_block; ++row_in_block) {
            const uint64_t global_row = block_row_begin + row_in_block;
            const char* src = block.data() + row_in_block * header.row_size_bytes;
            uint32_t* dst = matrix.data.data() +
                static_cast<size_t>(global_row) * static_cast<size_t>(matrix.cols);
            std::memcpy(dst, src, header.row_size_bytes);
        }
        next_row = block_row_begin + rows_in_block;
    }

    return matrix;
}

void convert_legacy_graph_file(const std::string& legacy_graph_path,
                                  const std::string& graph_file_path)
{
    std::ifstream input(legacy_graph_path, std::ios::binary);
    if (!input) {
        fail_graph_file("unable to open legacy graph " + legacy_graph_path);
    }

    uint64_t rows = 0;
    uint64_t cols = 0;
    input.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    input.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    if (!input) {
        fail_graph_file("failed to read legacy graph header from " + legacy_graph_path);
    }
    if (cols == 0 || cols > UINT32_MAX) {
        fail_graph_file("invalid degree in legacy graph " + legacy_graph_path);
    }

    const uint32_t degree = static_cast<uint32_t>(cols);
    const uint32_t row_size_bytes = degree * sizeof(uint32_t);
    const size_t rows_per_block = compute_rows_per_block(row_size_bytes);
    if (rows_per_block == 0) {
        fail_graph_file("degree too large for 4KB block packing");
    }

    std::ofstream output(graph_file_path, std::ios::binary | std::ios::trunc);
    if (!output) {
        fail_graph_file("unable to open graph file output " + graph_file_path);
    }

    GraphFileHeader header;
    header.rows = rows;
    header.degree = degree;
    header.row_size_bytes = row_size_bytes;
    header.rows_per_block = rows_per_block;

    std::array<char, kGraphFileHeaderSize> header_block{};
    std::copy_n(reinterpret_cast<const char*>(&header), sizeof(header), header_block.data());
    output.write(header_block.data(), header_block.size());
    if (!output) {
        fail_graph_file("failed to write graph file header to " + graph_file_path);
    }

    std::vector<uint32_t> row(static_cast<size_t>(degree));
    std::vector<char> block(kGraphFileBlockSize, 0);
    size_t row_in_block = 0;

    for (uint64_t r = 0; r < rows; ++r) {
        input.read(reinterpret_cast<char*>(row.data()), static_cast<std::streamsize>(row.size() * sizeof(uint32_t)));
        if (!input) {
            fail_graph_file("failed reading row payload from " + legacy_graph_path);
        }

        std::memcpy(block.data() + row_in_block * row_size_bytes,
                    row.data(),
                    row_size_bytes);
        ++row_in_block;

        if (row_in_block == rows_per_block) {
            output.write(block.data(), static_cast<std::streamsize>(block.size()));
            if (!output) {
                fail_graph_file("failed writing packed block to " + graph_file_path);
            }
            std::fill(block.begin(), block.end(), 0);
            row_in_block = 0;
        }
    }

    if (row_in_block != 0) {
        output.write(block.data(), static_cast<std::streamsize>(block.size()));
        if (!output) {
            fail_graph_file("failed writing final packed block to " + graph_file_path);
        }
    }
}

struct GraphFileSequentialWriter::Impl {
    explicit Impl(const std::string& path, uint64_t rows, uint32_t degree)
        : path(path)
    {
        if (degree == 0) {
            fail_graph_file("degree must be > 0 for graph file output " + path);
        }
        if (degree > std::numeric_limits<uint32_t>::max() / sizeof(uint32_t)) {
            fail_graph_file("degree is too large for graph file output " + path);
        }

        header.rows = rows;
        header.degree = degree;
        header.row_size_bytes = degree * static_cast<uint32_t>(sizeof(uint32_t));
        const size_t computed_rows_per_block = compute_rows_per_block(header.row_size_bytes);
        if (computed_rows_per_block == 0) {
            fail_graph_file("degree too large for 4KB block packing in " + path);
        }
        header.rows_per_block = static_cast<uint64_t>(computed_rows_per_block);
        validate_seekable_file_span(header, path);

        output.open(path, std::ios::binary | std::ios::trunc);
        if (!output) {
            fail_graph_file("unable to open graph file output " + path);
        }

        std::array<char, kGraphFileHeaderSize> header_block{};
        std::copy_n(reinterpret_cast<const char*>(&header), sizeof(header), header_block.data());
        output.write(header_block.data(), static_cast<std::streamsize>(header_block.size()));
        if (!output) {
            fail_graph_file("failed to write graph file header to " + path);
        }
        block.assign(kGraphFileBlockSize, 0);
    }

    void append_rows(const uint32_t* rows, uint64_t row_count)
    {
        if (closed) {
            fail_graph_file("cannot append rows after closing graph file " + path);
        }
        if (row_count == 0) {
            return;
        }
        if (rows == nullptr) {
            fail_graph_file("null row payload for graph file " + path);
        }
        if (written_rows > header.rows || row_count > header.rows - written_rows) {
            fail_graph_file("too many rows appended to graph file " + path);
        }

        const uint32_t* cursor = rows;
        for (uint64_t i = 0; i < row_count; ++i) {
            std::memcpy(block.data() + rows_in_block * header.row_size_bytes,
                        cursor,
                        header.row_size_bytes);
            cursor += header.degree;
            ++rows_in_block;
            ++written_rows;
            if (rows_in_block == header.rows_per_block) {
                flush_block();
            }
        }
    }

    void close()
    {
        if (closed) {
            return;
        }
        if (written_rows != header.rows) {
            fail_graph_file("row count mismatch while closing graph file " + path);
        }
        if (rows_in_block != 0) {
            flush_block();
        }
        output.close();
        if (!output) {
            fail_graph_file("failed to close graph file output " + path);
        }
        closed = true;
    }

    void flush_block()
    {
        output.write(block.data(), static_cast<std::streamsize>(block.size()));
        if (!output) {
            fail_graph_file("failed writing packed block to " + path);
        }
        std::fill(block.begin(), block.end(), 0);
        rows_in_block = 0;
    }

    std::string path;
    GraphFileHeader header;
    std::ofstream output;
    std::vector<char> block;
    uint64_t written_rows = 0;
    uint64_t rows_in_block = 0;
    bool closed = false;
};

GraphFileSequentialWriter::GraphFileSequentialWriter(const std::string& path,
                                                     uint64_t rows,
                                                     uint32_t degree)
    : impl_(std::make_unique<Impl>(path, rows, degree))
{
}

GraphFileSequentialWriter::~GraphFileSequentialWriter() = default;

void GraphFileSequentialWriter::append_rows(const uint32_t* rows, uint64_t row_count)
{
    impl_->append_rows(rows, row_count);
}

void GraphFileSequentialWriter::close()
{
    impl_->close();
}

const GraphFileHeader& GraphFileSequentialWriter::header() const
{
    return impl_->header;
}

}
