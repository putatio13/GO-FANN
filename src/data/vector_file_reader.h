#pragma once

#include "data/dataloader.h"
#include "data/fp16_l2.h"

#include <fstream>
#include <limits>
#include <stdexcept>
#include <vector>

// A worker-local input reader for streamed index construction. It owns no cache
// and is not used by the search host service. Each builder worker opens its own.
class VectorFileReader {
public:
    explicit VectorFileReader(const VectorFileInfo& info)
        : info_(info), input_(info.path, std::ios::binary)
    {
        if (!input_) fail("cannot open input");
        uint64_t scalar_bytes = 0;
        switch (info_.scalar_type) {
            case DenseVectorScalarType::Uint8: scalar_bytes = 1; break;
            case DenseVectorScalarType::Float16: scalar_bytes = 2; break;
            case DenseVectorScalarType::Float32: scalar_bytes = 4; break;
            default: fail("unsupported scalar type");
        }
        payload_bytes_ = uint64_t(info_.cols) * scalar_bytes;
        if (!info_.cols || !info_.rows ||
            (info_.row_prefix_bytes != 0 && info_.row_prefix_bytes != sizeof(uint32_t)) ||
            info_.row_data_offset < info_.row_prefix_bytes ||
            uint64_t(info_.row_stride_bytes) != payload_bytes_ + info_.row_prefix_bytes)
            fail("invalid row layout");
        const uint64_t row_span = uint64_t(info_.rows - 1) * info_.row_stride_bytes;
        const uint64_t limit = std::numeric_limits<uint64_t>::max();
        if (info_.row_data_offset > limit - row_span ||
            info_.row_data_offset + row_span > limit - payload_bytes_)
            fail("row offset overflow");
        const uint64_t required = info_.row_data_offset + row_span + payload_bytes_;
        input_.seekg(0, std::ios::end);
        const auto extent = input_.tellg();
        if (extent < 0 || uint64_t(extent) < required) fail("truncated input");
        if (info_.scalar_type == DenseVectorScalarType::Float16) half_row_.resize(info_.cols);
    }

    void read_uint8_row(uint32_t row, uint8_t* destination, uint32_t cols)
    {
        if (info_.scalar_type != DenseVectorScalarType::Uint8) fail("uint8 type mismatch");
        seek_row(row, destination, cols);
        read_exact(destination, payload_bytes_);
    }

    void read_float_row(uint32_t row, float* destination, uint32_t cols)
    {
        if (info_.scalar_type != DenseVectorScalarType::Float32 &&
            info_.scalar_type != DenseVectorScalarType::Float16) fail("float type mismatch");
        seek_row(row, destination, cols);
        if (info_.scalar_type == DenseVectorScalarType::Float32) {
            read_exact(destination, payload_bytes_);
        } else {
            read_exact(half_row_.data(), payload_bytes_);
            for (uint32_t col = 0; col < cols; ++col)
                destination[col] = go_fann::fp16_l2::decode(half_row_[col]);
        }
    }

private:
    [[noreturn]] void fail(const char* message) const
    {
        throw std::runtime_error("vector input reader: " + std::string(message) + ": " + info_.path);
    }
    void read_exact(void* destination, uint64_t bytes)
    {
        if (bytes > uint64_t(std::numeric_limits<std::streamsize>::max())) fail("row too large");
        input_.read(static_cast<char*>(destination), static_cast<std::streamsize>(bytes));
        if (!input_) fail("truncated row");
    }
    void seek_row(uint32_t row, const void* destination, uint32_t cols)
    {
        if (row >= info_.rows || cols != info_.cols || !destination) fail("invalid row request");
        const uint64_t offset = info_.row_data_offset + uint64_t(row) * info_.row_stride_bytes;
        if (offset > uint64_t(std::numeric_limits<std::streamoff>::max())) fail("offset too large");
        input_.clear();
        input_.seekg(static_cast<std::streamoff>(offset - info_.row_prefix_bytes));
        if (!input_) fail("seek failed");
        if (info_.row_prefix_bytes) {
            uint32_t dim = 0;
            read_exact(&dim, sizeof(dim));
            if (dim != cols) fail("row dimension mismatch");
        }
    }
    VectorFileInfo info_;
    std::ifstream input_;
    uint64_t payload_bytes_ = 0;
    std::vector<uint16_t> half_row_;
};
