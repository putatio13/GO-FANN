#include "index/range_posting_index.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

namespace go_fann::range_posting {

namespace {

[[noreturn]] void fail(const std::string& message)
{
    throw std::runtime_error("range posting index failed: " + message);
}

bool checked_region(uint64_t offset, uint64_t bytes, uint64_t file_bytes)
{
    return offset <= file_bytes && bytes <= file_bytes - offset;
}

template <typename T>
const T* mapped_array(const void* mapping,
                      uint64_t offset,
                      uint64_t bytes,
                      uint64_t count,
                      uint64_t file_bytes,
                      const char* name)
{
    if (count > std::numeric_limits<uint64_t>::max() / sizeof(T) ||
        bytes != count * sizeof(T)) {
        fail(std::string(name) + " byte count does not match element count");
    }
    if (offset % alignof(T) != 0 || !checked_region(offset, bytes, file_bytes)) {
        fail(std::string(name) + " region is unaligned or outside the file");
    }
    return reinterpret_cast<const T*>(
        static_cast<const uint8_t*>(mapping) + static_cast<size_t>(offset));
}

std::pair<uint64_t, uint64_t> slice_bounds(const DimensionView& dimension,
                                           uint8_t mode,
                                           float lower,
                                           float upper)
{
    const float* begin = dimension.unique_values;
    const float* end = begin + dimension.unique_count;
    const float* first = begin;
    const float* last = end;
    switch (mode) {
    case 1:
        first = std::upper_bound(begin, end, lower);
        break;
    case 2:
        last = std::lower_bound(begin, end, upper);
        break;
    case 3:
        first = std::lower_bound(begin, end, lower);
        last = std::lower_bound(first, end, upper);
        break;
    default:
        fail("unknown active query range mode " + std::to_string(mode));
    }
    const uint64_t first_idx = static_cast<uint64_t>(first - begin);
    const uint64_t last_idx = static_cast<uint64_t>(last - begin);
    return {dimension.offsets[first_idx], dimension.offsets[last_idx]};
}

}  // namespace

Mode parse_mode(const std::string& value)
{
    if (value == "off") {
        return Mode::Off;
    }
    if (value == "select") {
        return Mode::Select;
    }
    if (value == "hybrid") {
        return Mode::Hybrid;
    }
    fail("mode must be one of: off, select, hybrid");
}

const char* mode_name(Mode mode) noexcept
{
    switch (mode) {
    case Mode::Off:
        return "off";
    case Mode::Select:
        return "select";
    case Mode::Hybrid:
        return "hybrid";
    }
    return "unknown";
}

Index::~Index()
{
    reset();
}

Index::Index(Index&& other) noexcept
{
    *this = std::move(other);
}

Index& Index::operator=(Index&& other) noexcept
{
    if (this == &other) {
        return *this;
    }
    reset();
    fd_ = other.fd_;
    mapping_ = other.mapping_;
    mapped_bytes_ = other.mapped_bytes_;
    row_count_ = other.row_count_;
    dimensions_ = std::move(other.dimensions_);
    other.fd_ = -1;
    other.mapping_ = nullptr;
    other.mapped_bytes_ = 0;
    other.row_count_ = 0;
    return *this;
}

void Index::reset() noexcept
{
    dimensions_.clear();
    if (mapping_ != nullptr && mapped_bytes_ != 0) {
        munmap(mapping_, mapped_bytes_);
    }
    if (fd_ >= 0) {
        close(fd_);
    }
    fd_ = -1;
    mapping_ = nullptr;
    mapped_bytes_ = 0;
    row_count_ = 0;
}

Index Index::load(const std::string& path)
{
    Index index;
    index.fd_ = open(path.c_str(), O_RDONLY);
    if (index.fd_ < 0) {
        fail("could not open " + path + ": " + std::strerror(errno));
    }
    struct stat stat_buffer {};
    if (fstat(index.fd_, &stat_buffer) != 0 || stat_buffer.st_size < 0) {
        fail("could not stat " + path + ": " + std::strerror(errno));
    }
    const uint64_t file_bytes = static_cast<uint64_t>(stat_buffer.st_size);
    if (file_bytes < sizeof(FileHeader) || file_bytes > std::numeric_limits<size_t>::max()) {
        fail("file is too small or exceeds the host address range");
    }
    index.mapped_bytes_ = static_cast<size_t>(file_bytes);
    index.mapping_ = mmap(nullptr, index.mapped_bytes_, PROT_READ, MAP_PRIVATE, index.fd_, 0);
    if (index.mapping_ == MAP_FAILED) {
        index.mapping_ = nullptr;
        fail("could not mmap " + path + ": " + std::strerror(errno));
    }

    const auto* header = static_cast<const FileHeader*>(index.mapping_);
    if (std::memcmp(header->magic, kMagic, sizeof(kMagic)) != 0) {
        fail("magic mismatch");
    }
    if (header->version != kVersion) {
        fail("unsupported version " + std::to_string(header->version));
    }
    if (header->file_bytes != file_bytes) {
        fail("header file size does not match the actual file");
    }
    if (header->row_count == 0 || header->row_count > UINT32_MAX) {
        fail("row count must fit nonzero uint32 sample ids");
    }
    if (header->range_count == 0) {
        fail("range count must be nonzero");
    }
    const uint64_t descriptor_bytes =
        static_cast<uint64_t>(header->range_count) * sizeof(DimensionDescriptor);
    const uint64_t expected_header_bytes = sizeof(FileHeader) + descriptor_bytes;
    if (header->header_bytes != expected_header_bytes ||
        !checked_region(sizeof(FileHeader), descriptor_bytes, file_bytes)) {
        fail("descriptor table size is invalid");
    }

    const auto* descriptors = reinterpret_cast<const DimensionDescriptor*>(
        static_cast<const uint8_t*>(index.mapping_) + sizeof(FileHeader));
    index.row_count_ = header->row_count;
    index.dimensions_.reserve(header->range_count);
    for (uint32_t range_index = 0; range_index < header->range_count; ++range_index) {
        const DimensionDescriptor& descriptor = descriptors[range_index];
        if (descriptor.range_index != range_index || descriptor.reserved != 0) {
            fail("dimension descriptors must be dense, ordered, and reserved-zero");
        }
        if (descriptor.row_count != header->row_count || descriptor.unique_count == 0 ||
            descriptor.unique_count > descriptor.row_count) {
            fail("dimension row count mismatch or invalid value-domain size");
        }
        const float* values = mapped_array<float>(index.mapping_,
                                                  descriptor.values_offset,
                                                  descriptor.values_bytes,
                                                  descriptor.unique_count,
                                                  file_bytes,
                                                  "unique values");
        const uint64_t* offsets = mapped_array<uint64_t>(index.mapping_,
                                                         descriptor.offsets_offset,
                                                         descriptor.offsets_bytes,
                                                         descriptor.unique_count + 1,
                                                         file_bytes,
                                                         "value offsets");
        const uint32_t* sample_ids = mapped_array<uint32_t>(index.mapping_,
                                                            descriptor.sample_ids_offset,
                                                            descriptor.sample_ids_bytes,
                                                            descriptor.row_count,
                                                            file_bytes,
                                                            "sample ids");
        for (uint64_t value_idx = 0; value_idx < descriptor.unique_count; ++value_idx) {
            if (!std::isfinite(values[value_idx]) ||
                (value_idx != 0 && !(values[value_idx - 1] < values[value_idx]))) {
                fail("unique values must be finite and strictly increasing");
            }
        }
        if (offsets[0] != 0 || offsets[descriptor.unique_count] != descriptor.row_count) {
            fail("value offsets must begin at zero and end at row_count");
        }
        for (uint64_t offset_idx = 1; offset_idx <= descriptor.unique_count; ++offset_idx) {
            if (offsets[offset_idx] < offsets[offset_idx - 1] ||
                offsets[offset_idx] > descriptor.row_count) {
                fail("value offsets must be monotonic and in range");
            }
        }
        std::vector<uint64_t> seen_sample_ids(
            static_cast<size_t>((descriptor.row_count + 63) / 64), 0);
        for (uint64_t row = 0; row < descriptor.row_count; ++row) {
            if (sample_ids[row] >= descriptor.row_count) {
                fail("sample id is outside the base row range");
            }
            const uint32_t sample_id = sample_ids[row];
            const size_t word = static_cast<size_t>(sample_id / 64);
            const uint64_t mask = uint64_t{1} << (sample_id % 64);
            if ((seen_sample_ids[word] & mask) != 0) {
                fail("sample ids must form a permutation without duplicates");
            }
            seen_sample_ids[word] |= mask;
        }
        index.dimensions_.push_back(DimensionView{range_index,
                                                  descriptor.unique_count,
                                                  descriptor.row_count,
                                                  values,
                                                  offsets,
                                                  sample_ids});
    }
    return index;
}

const DimensionView& Index::dimension(uint32_t range_index) const
{
    if (range_index >= dimensions_.size()) {
        fail("range dimension " + std::to_string(range_index) + " is missing");
    }
    return dimensions_[range_index];
}

Slice choose_smallest_slice(const Index& index,
                            const QueryRangeTable& query_ranges,
                            uint32_t query_idx)
{
    if (query_idx >= query_ranges.rows) {
        fail("query index is outside the query range table");
    }
    if (query_ranges.range_count != index.range_count()) {
        fail("query and posting range counts do not match");
    }
    Slice best;
    best.count = UINT64_MAX;
    for (uint32_t range_index = 0; range_index < query_ranges.range_count; ++range_index) {
        const size_t flat = static_cast<size_t>(query_idx) * query_ranges.range_count +
            range_index;
        const uint8_t mode = query_ranges.modes[flat];
        if (mode == 0) {
            continue;
        }
        best.has_active_range = true;
        ++best.active_dimension_count;
        const DimensionView& dimension = index.dimension(range_index);
        const auto [slice_begin, slice_end] = slice_bounds(dimension,
                                                          mode,
                                                          query_ranges.lower_bounds[flat],
                                                          query_ranges.upper_bounds[flat]);
        const uint64_t slice_count = slice_end - slice_begin;
        if (slice_count < best.count ||
            (slice_count == best.count && range_index < best.range_index)) {
            best.range_index = range_index;
            best.begin = slice_begin;
            best.count = slice_count;
        }
    }
    if (!best.has_active_range) {
        best.count = 0;
        best.range_index = UINT32_MAX;
    }
    return best;
}

}  // namespace go_fann::range_posting
