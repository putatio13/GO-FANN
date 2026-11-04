#pragma once
#include "predicates/gpu_predicate_index.cuh"
#include <nlohmann/json.hpp>
#include <openssl/evp.h>
#include <algorithm>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace go_fann::gpu_predicate {
inline void require(bool ok, const std::string& why) {
    if (!ok) throw std::runtime_error("GPU predicate index: " + why);
}
inline std::string sha256_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    require(bool(f), "cannot open " + path);
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> ctx(EVP_MD_CTX_new(), EVP_MD_CTX_free);
    require(ctx && EVP_DigestInit_ex(ctx.get(), EVP_sha256(), nullptr) == 1, "SHA256 initialization");
    std::vector<char> buffer(1 << 20);
    while (f) { f.read(buffer.data(), buffer.size()); require(EVP_DigestUpdate(ctx.get(), buffer.data(), f.gcount()) == 1, "SHA256 update"); }
    require(f.eof(), "read error " + path);
    unsigned char bytes[32]; unsigned int size = 0;
    require(EVP_DigestFinal_ex(ctx.get(), bytes, &size) == 1 && size == 32, "SHA256 final");
    std::ostringstream out;
    for (auto byte : bytes) out << std::hex << std::setw(2) << std::setfill('0') << unsigned(byte);
    return out.str();
}
struct HostIndex {
    uint32_t rows = 0, chunks = 0, threshold = 0;
    std::vector<uint32_t> labels, directory;
    std::vector<Descriptor> descriptors;
    std::vector<uint16_t> payload;
    std::vector<uint64_t> cardinalities;
    template<class T> static std::vector<T> read(const std::string& root, const std::string& name, const nlohmann::json& manifest) {
        const auto path = root + "/" + name;
        const uint64_t bytes = manifest.at("files").at(name).at("bytes");
        require(bytes == std::filesystem::file_size(path) && bytes % sizeof(T) == 0, "size mismatch " + name);
        require(sha256_file(path) == manifest.at("files").at(name).at("sha256").get<std::string>(), "checksum mismatch " + name);
        std::vector<T> result(bytes / sizeof(T));
        std::ifstream f(path, std::ios::binary);
        f.read(reinterpret_cast<char*>(result.data()), bytes);
        require(bool(f), "short read " + name);
        return result;
    }
    static HostIndex load(const std::string& root, const std::string& source, uint32_t expected_rows, uint32_t expected_threshold) {
        std::ifstream f(root + "/manifest.json"); require(bool(f), "missing manifest");
        nlohmann::json m; f >> m;
        require(m.at("version") == 1 && m.at("format") == "flashanns_roaring_arrays", "unsupported version/container format");
        HostIndex h;
        h.rows = m.at("rows"); h.chunks = m.at("chunks"); h.threshold = m.at("threshold");
        require(h.rows == expected_rows && h.threshold == expected_threshold && h.chunks == (uint64_t(h.rows) + 65535) / 65536, "stale row count/threshold/chunks");
        require(std::filesystem::equivalent(source, m.at("source").at("path").get<std::string>()), "different source/sample ordering");
        require(std::filesystem::file_size(source) == m.at("source").at("bytes").get<uint64_t>() && sha256_file(source) == m.at("source").at("sha256").get<std::string>(), "stale source identity");
        h.labels = read<uint32_t>(root, "labels.bin", m);
        h.directory = read<uint32_t>(root, "directory.bin", m);
        h.descriptors = read<Descriptor>(root, "descriptors.bin", m);
        h.payload = read<uint16_t>(root, "payload.bin", m);
        require(h.labels.size() == m.at("labels").get<uint64_t>() && h.descriptors.size() == m.at("containers").get<uint64_t>() && h.payload.size()*2 == m.at("payload_bytes").get<uint64_t>(), "manifest pool count mismatch");
        h.validate(); return h;
    }
    void validate() {
        require(rows > 0 && chunks == (uint64_t(rows)+65535)/65536, "invalid domain");
        require(std::is_sorted(labels.begin(), labels.end()) && std::adjacent_find(labels.begin(), labels.end()) == labels.end(), "duplicate/unsorted labels");
        require(directory.size() == uint64_t(labels.size()) * chunks, "directory length");
        cardinalities.assign(labels.size(), 0);
        std::vector<uint8_t> seen(descriptors.size(), 0);
        uint64_t previous_end = 0;
        for (const auto& d : descriptors) {
            require(d.reserved == 0 && d.count > 0 && d.count <= 4096, "unsupported/non-array container");
            require(d.offset % 64 == 0 && d.offset >= previous_end && d.offset <= payload.size() && d.count <= payload.size()-d.offset, "invalid payload offset");
            previous_end = d.offset + d.count;
            const auto begin = payload.begin() + d.offset, end = begin + d.count;
            require(std::is_sorted(begin,end) && std::adjacent_find(begin,end) == end, "container not sorted unique");
        }
        for (size_t pos = 0; pos < directory.size(); ++pos) {
            const auto id = directory[pos]; if (id == UINT32_MAX) continue;
            require(id < descriptors.size() && !seen[id]++, "invalid/aliased descriptor");
            const auto& d = descriptors[id];
            require((uint64_t(pos % chunks) << 16) + payload[d.offset+d.count-1] < rows, "last chunk outside sample domain");
            cardinalities[pos/chunks] += d.count;
        }
        require(std::find(seen.begin(),seen.end(),0) == seen.end(), "unreferenced descriptor");
    }
};
}
