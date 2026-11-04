#include "index/range_posting_index.h"

#include <cassert>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace rp = go_fann::range_posting;

namespace {

constexpr uint64_t kAlignment = 64;

uint64_t align_up(uint64_t value)
{
    return ((value + kAlignment - 1) / kAlignment) * kAlignment;
}

struct TestDimension {
    std::vector<float> values;
    std::vector<uint64_t> offsets;
    std::vector<uint32_t> ids;
};

void write_zeroes(std::ofstream& out, uint64_t target)
{
    const uint64_t current = static_cast<uint64_t>(out.tellp());
    assert(current <= target);
    std::vector<char> zeroes(static_cast<size_t>(target - current), 0);
    out.write(zeroes.data(), static_cast<std::streamsize>(zeroes.size()));
}

void write_index(const std::filesystem::path& path,
                 const std::vector<TestDimension>& dimensions,
                 const std::function<void(rp::FileHeader&,
                                          std::vector<rp::DimensionDescriptor>&,
                                          std::vector<TestDimension>&)>& mutate = {})
{
    assert(!dimensions.empty());
    std::vector<TestDimension> payload = dimensions;
    const uint64_t row_count = payload.front().ids.size();
    const uint32_t range_count = static_cast<uint32_t>(payload.size());
    const uint32_t header_bytes = static_cast<uint32_t>(
        sizeof(rp::FileHeader) + payload.size() * sizeof(rp::DimensionDescriptor));
    uint64_t cursor = align_up(header_bytes);
    std::vector<rp::DimensionDescriptor> descriptors;
    for (uint32_t range_index = 0; range_index < range_count; ++range_index) {
        const auto& dimension = payload[range_index];
        rp::DimensionDescriptor descriptor{};
        descriptor.range_index = range_index;
        descriptor.unique_count = dimension.values.size();
        descriptor.row_count = dimension.ids.size();
        descriptor.values_offset = cursor;
        descriptor.values_bytes = dimension.values.size() * sizeof(float);
        cursor = align_up(cursor + descriptor.values_bytes);
        descriptor.offsets_offset = cursor;
        descriptor.offsets_bytes = dimension.offsets.size() * sizeof(uint64_t);
        cursor = align_up(cursor + descriptor.offsets_bytes);
        descriptor.sample_ids_offset = cursor;
        descriptor.sample_ids_bytes = dimension.ids.size() * sizeof(uint32_t);
        cursor = align_up(cursor + descriptor.sample_ids_bytes);
        descriptors.push_back(descriptor);
    }
    rp::FileHeader header{};
    std::memcpy(header.magic, rp::kMagic, sizeof(rp::kMagic));
    header.version = rp::kVersion;
    header.header_bytes = header_bytes;
    header.file_bytes = cursor;
    header.row_count = row_count;
    header.range_count = range_count;
    if (mutate) {
        mutate(header, descriptors, payload);
    }

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(&header), sizeof(header));
    out.write(reinterpret_cast<const char*>(descriptors.data()),
              static_cast<std::streamsize>(descriptors.size() * sizeof(descriptors.front())));
    for (size_t i = 0; i < payload.size(); ++i) {
        write_zeroes(out, descriptors[i].values_offset);
        out.write(reinterpret_cast<const char*>(payload[i].values.data()),
                  static_cast<std::streamsize>(payload[i].values.size() * sizeof(float)));
        write_zeroes(out, descriptors[i].offsets_offset);
        out.write(reinterpret_cast<const char*>(payload[i].offsets.data()),
                  static_cast<std::streamsize>(payload[i].offsets.size() * sizeof(uint64_t)));
        write_zeroes(out, descriptors[i].sample_ids_offset);
        out.write(reinterpret_cast<const char*>(payload[i].ids.data()),
                  static_cast<std::streamsize>(payload[i].ids.size() * sizeof(uint32_t)));
    }
    write_zeroes(out, header.file_bytes);
    assert(out.good());
}

QueryRangeTable make_ranges(uint32_t rows, uint32_t range_count)
{
    QueryRangeTable table;
    table.rows = rows;
    table.range_count = range_count;
    table.modes.assign(static_cast<size_t>(rows) * range_count, 0);
    table.lower_bounds.assign(static_cast<size_t>(rows) * range_count, 0.0f);
    table.upper_bounds.assign(static_cast<size_t>(rows) * range_count, 0.0f);
    return table;
}

bool throws_with(const std::function<void()>& fn, const std::string& needle)
{
    try {
        fn();
    } catch (const std::runtime_error& error) {
        return std::string(error.what()).find(needle) != std::string::npos;
    }
    return false;
}

}  // namespace

int main()
{
    const auto root = std::filesystem::temp_directory_path() /
        ("flashanns-range-posting-test-" + std::to_string(::getpid()));
    std::filesystem::create_directories(root);
    const std::vector<TestDimension> dimensions = {
        {{1.0f, 2.0f, 4.0f}, {0, 2, 4, 6}, {0, 1, 2, 3, 4, 5}},
        {{10.0f, 20.0f, 30.0f}, {0, 1, 3, 6}, {5, 3, 4, 0, 1, 2}},
    };
    const auto valid_path = root / "valid.bin";
    write_index(valid_path, dimensions);
    rp::Index index = rp::Index::load(valid_path.string());
    assert(index.row_count() == 6);
    assert(index.range_count() == 2);

    auto ranges = make_ranges(5, 2);
    const auto flat = [&ranges](uint32_t row, uint32_t range_index) {
        return static_cast<size_t>(row) * ranges.range_count + range_index;
    };
    ranges.modes[flat(0, 0)] = 1;
    ranges.lower_bounds[flat(0, 0)] = 2.0f;
    auto slice = rp::choose_smallest_slice(index, ranges, 0);
    assert(slice.range_index == 0 && slice.begin == 4 && slice.count == 2);
    assert(!rp::eligible_for_scan(slice, 1));
    assert(rp::eligible_for_scan(slice, 2));

    ranges.modes[flat(1, 1)] = 2;
    ranges.upper_bounds[flat(1, 1)] = 20.0f;
    slice = rp::choose_smallest_slice(index, ranges, 1);
    assert(slice.range_index == 1 && slice.begin == 0 && slice.count == 1);

    ranges.modes[flat(2, 0)] = 3;
    ranges.lower_bounds[flat(2, 0)] = 2.0f;
    ranges.upper_bounds[flat(2, 0)] = 4.0f;
    slice = rp::choose_smallest_slice(index, ranges, 2);
    assert(slice.range_index == 0 && slice.begin == 2 && slice.count == 2);

    ranges.modes[flat(3, 0)] = 3;
    ranges.lower_bounds[flat(3, 0)] = 1.0f;
    ranges.upper_bounds[flat(3, 0)] = 4.0f;
    ranges.modes[flat(3, 1)] = 3;
    ranges.lower_bounds[flat(3, 1)] = 10.0f;
    ranges.upper_bounds[flat(3, 1)] = 20.0f;
    slice = rp::choose_smallest_slice(index, ranges, 3);
    assert(slice.active_dimension_count == 2);
    assert(slice.range_index == 1 && slice.begin == 0 && slice.count == 1);

    slice = rp::choose_smallest_slice(index, ranges, 4);
    assert(!slice.has_active_range && slice.count == 0);
    assert(rp::parse_mode("off") == rp::Mode::Off);
    assert(rp::parse_mode("select") == rp::Mode::Select);
    assert(rp::parse_mode("hybrid") == rp::Mode::Hybrid);
    assert(throws_with([] { (void)rp::parse_mode("bad"); }, "off, select, hybrid"));

    const auto bad_magic = root / "bad-magic.bin";
    write_index(bad_magic, dimensions, [](auto& header, auto&, auto&) {
        header.magic[0] = 'X';
    });
    assert(throws_with([&] { (void)rp::Index::load(bad_magic.string()); }, "magic mismatch"));

    const auto bad_unique_count = root / "bad-unique-count.bin";
    write_index(bad_unique_count, dimensions, [](auto&, auto& descriptors, auto&) {
        descriptors[0].unique_count = descriptors[0].row_count + 1;
    });
    assert(throws_with([&] { (void)rp::Index::load(bad_unique_count.string()); },
                       "value-domain size"));

    const auto bad_id = root / "bad-id.bin";
    write_index(bad_id, dimensions, [](auto&, auto&, auto& payload) {
        payload[0].ids.back() = 6;
    });
    assert(throws_with([&] { (void)rp::Index::load(bad_id.string()); }, "sample id"));

    const auto duplicate_id = root / "duplicate-id.bin";
    write_index(duplicate_id, dimensions, [](auto&, auto&, auto& payload) {
        payload[0].ids.back() = payload[0].ids.front();
    });
    assert(throws_with([&] { (void)rp::Index::load(duplicate_id.string()); }, "permutation"));

    const auto truncated = root / "truncated.bin";
    std::filesystem::copy_file(valid_path, truncated);
    std::filesystem::resize_file(truncated, std::filesystem::file_size(truncated) - 1);
    assert(throws_with([&] { (void)rp::Index::load(truncated.string()); }, "file size"));

    std::filesystem::remove_all(root);
    return 0;
}
