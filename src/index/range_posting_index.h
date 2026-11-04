#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "data/dataloader.h"

namespace go_fann::range_posting {

inline constexpr char kMagic[8] = {'F', 'A', 'R', 'P', 'I', 'D', 'X', '1'};
inline constexpr uint32_t kVersion = 1;

enum class Mode : uint8_t {
    Off = 0,
    Select = 1,
    Hybrid = 2,
};

Mode parse_mode(const std::string& value);
const char* mode_name(Mode mode) noexcept;

#pragma pack(push, 1)
struct FileHeader {
    char magic[8];
    uint32_t version;
    uint32_t header_bytes;
    uint64_t file_bytes;
    uint64_t row_count;
    uint32_t range_count;
    uint32_t reserved;
};

struct DimensionDescriptor {
    uint32_t range_index;
    uint32_t reserved;
    uint64_t unique_count;
    uint64_t row_count;
    uint64_t values_offset;
    uint64_t offsets_offset;
    uint64_t sample_ids_offset;
    uint64_t values_bytes;
    uint64_t offsets_bytes;
    uint64_t sample_ids_bytes;
};
#pragma pack(pop)

static_assert(sizeof(FileHeader) == 40, "range posting file header layout changed");
static_assert(sizeof(DimensionDescriptor) == 72,
              "range posting dimension descriptor layout changed");

struct DimensionView {
    uint32_t range_index = 0;
    uint64_t unique_count = 0;
    uint64_t row_count = 0;
    const float* unique_values = nullptr;
    const uint64_t* offsets = nullptr;
    const uint32_t* sample_ids = nullptr;
};

class Index {
public:
    Index() = default;
    ~Index();
    Index(const Index&) = delete;
    Index& operator=(const Index&) = delete;
    Index(Index&& other) noexcept;
    Index& operator=(Index&& other) noexcept;

    static Index load(const std::string& path);

    uint64_t row_count() const noexcept { return row_count_; }
    uint32_t range_count() const noexcept
    {
        return static_cast<uint32_t>(dimensions_.size());
    }
    uint64_t file_bytes() const noexcept { return mapped_bytes_; }
    const std::vector<DimensionView>& dimensions() const noexcept { return dimensions_; }
    const DimensionView& dimension(uint32_t range_index) const;

private:
    void reset() noexcept;

    int fd_ = -1;
    void* mapping_ = nullptr;
    size_t mapped_bytes_ = 0;
    uint64_t row_count_ = 0;
    std::vector<DimensionView> dimensions_;
};

struct Slice {
    bool has_active_range = false;
    uint32_t active_dimension_count = 0;
    uint32_t range_index = UINT32_MAX;
    uint64_t begin = 0;
    uint64_t count = 0;
};

Slice choose_smallest_slice(const Index& index,
                            const QueryRangeTable& query_ranges,
                            uint32_t query_idx);

inline bool eligible_for_scan(const Slice& slice, uint64_t scan_threshold) noexcept
{
    return slice.has_active_range && slice.count <= scan_threshold;
}

}  // namespace go_fann::range_posting
