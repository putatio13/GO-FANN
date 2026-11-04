#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace go_fann::graph_file {

constexpr uint32_t kGraphFileMagic = 0x47524446;  // "FDRG"
constexpr uint32_t kGraphFileVersion = 1;
constexpr size_t kGraphFileBlockSize = 4096;
constexpr size_t kGraphFileHeaderSize = 4096;

struct GraphFileHeader {
    uint32_t magic = kGraphFileMagic;
    uint32_t version = kGraphFileVersion;
    uint32_t block_size = static_cast<uint32_t>(kGraphFileBlockSize);
    uint32_t header_size = static_cast<uint32_t>(kGraphFileHeaderSize);
    uint64_t rows = 0;
    uint32_t degree = 0;
    uint32_t row_size_bytes = 0;
    uint64_t rows_per_block = 0;
};

struct ContiguousGraphMatrix {
    uint64_t rows = 0;
    uint64_t cols = 0;
    std::vector<uint32_t> data;
};

// Detect only the magic; the reader still validates version and full header.
bool has_graph_file_magic(const std::string& path);
GraphFileHeader read_graph_file_header(const std::string& path);
uint64_t graph_file_row_offset_bytes(const GraphFileHeader& header, uint64_t global_row);
ContiguousGraphMatrix load_graph_file_as_contiguous_matrix(const std::string& path);
void convert_legacy_graph_file(const std::string& legacy_graph_path,
                                  const std::string& graph_file_path);

class GraphFileSequentialWriter {
public:
    GraphFileSequentialWriter(const std::string& path, uint64_t rows, uint32_t degree);
    ~GraphFileSequentialWriter();

    GraphFileSequentialWriter(const GraphFileSequentialWriter&) = delete;
    GraphFileSequentialWriter& operator=(const GraphFileSequentialWriter&) = delete;

    void append_rows(const uint32_t* rows, uint64_t row_count);
    void close();
    const GraphFileHeader& header() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
