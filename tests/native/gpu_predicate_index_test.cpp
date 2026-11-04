#include "predicates/gpu_predicate_index.h"
#include <cassert>
#include <functional>
#include <iostream>
#include <unistd.h>
using namespace go_fann::gpu_predicate;
void rejects(const std::function<void()>& f) { bool caught=false; try {f();} catch(const std::runtime_error&) {caught=true;} assert(caught); }

void test_artifact_loading(const HostIndex& fixture) {
    struct TemporaryDirectory {
        std::filesystem::path path = std::filesystem::temp_directory_path() /
            ("gpu-predicate-index-test-" + std::to_string(getpid()));
        TemporaryDirectory() { assert(std::filesystem::create_directory(path)); }
        ~TemporaryDirectory() { std::filesystem::remove_all(path); }
    } temp;
    const auto write_bytes = [&](const char* name, const void* data, size_t size) {
        std::ofstream file(temp.path / name, std::ios::binary | std::ios::trunc);
        file.write(static_cast<const char*>(data), size);
        assert(file.good());
    };
    write_bytes("source.spmat", "original", 8);
    write_bytes("labels.bin", fixture.labels.data(), fixture.labels.size()*sizeof(uint32_t));
    write_bytes("directory.bin", fixture.directory.data(), fixture.directory.size()*sizeof(uint32_t));
    write_bytes("descriptors.bin", fixture.descriptors.data(), fixture.descriptors.size()*sizeof(Descriptor));
    write_bytes("payload.bin", fixture.payload.data(), fixture.payload.size()*sizeof(uint16_t));
    nlohmann::json manifest = {
        {"version", 1}, {"format", "flashanns_roaring_arrays"},
        {"rows", fixture.rows}, {"chunks", fixture.chunks}, {"threshold", fixture.threshold},
        {"labels", fixture.labels.size()}, {"containers", fixture.descriptors.size()},
        {"payload_bytes", fixture.payload.size()*sizeof(uint16_t)},
        {"source", {{"path", (temp.path / "source.spmat").string()}, {"bytes", 8},
                    {"sha256", sha256_file((temp.path / "source.spmat").string())}}}
    };
    for (const char* name : {"labels.bin", "directory.bin", "descriptors.bin", "payload.bin"}) {
        manifest["files"][name] = {{"bytes", std::filesystem::file_size(temp.path / name)},
                                    {"sha256", sha256_file((temp.path / name).string())}};
    }
    const auto save_manifest = [&] { std::ofstream(temp.path / "manifest.json") << manifest; };
    const auto load = [&] { return HostIndex::load(temp.path.string(),
        (temp.path / "source.spmat").string(), fixture.rows, fixture.threshold); };
    save_manifest();
    const auto loaded = load();
    assert(loaded.labels == fixture.labels && loaded.cardinalities == fixture.cardinalities);
    manifest["version"] = 2;
    save_manifest(); rejects([&] { load(); });
    manifest["version"] = 1;
    manifest["threshold"] = fixture.threshold + 1;
    save_manifest(); rejects([&] { load(); });
    manifest["threshold"] = fixture.threshold;
    save_manifest();
    write_bytes("source.spmat", "modified", 8);
    rejects([&] { load(); });
    write_bytes("source.spmat", "original", 8);
    auto damaged_payload = fixture.payload;
    damaged_payload[0] = 1;
    write_bytes("payload.bin", damaged_payload.data(), damaged_payload.size()*sizeof(uint16_t));
    rejects([&] { load(); });
}

int main() {
    HostIndex h; h.rows=65538; h.chunks=2; h.labels={7,9};
    h.directory={0,1,UINT32_MAX,2}; h.descriptors={{0,2,0},{64,2,0},{128,1,0}};
    h.payload.resize(192); h.payload[0]=0;h.payload[1]=65535;h.payload[64]=0;h.payload[65]=1;h.payload[128]=1;
    h.validate(); assert(h.cardinalities[0]==4 && h.cardinalities[1]==1);
    test_artifact_loading(h);
    uint32_t offsets[]={0,1,3,3}, handles[]={0,0,1};
    View v{h.directory.data(),h.descriptors.data(),h.payload.data(),offsets,handles,nullptr,h.rows,h.chunks,0};
    for(uint32_t sample=0;sample<=h.rows;++sample) {
        assert(matches(v,0,sample)==(sample==0||sample==65535||sample==65536||sample==65537));
        assert(matches(v,1,sample)==(sample==65537));
        assert(matches(v,2,sample)==(sample<h.rows));
    }
    auto bad=h; bad.descriptors[1].reserved=1; rejects([&]{bad.validate();});
    bad=h;bad.descriptors[1].offset=999;rejects([&]{bad.validate();});
    bad=h;bad.payload[65]=2;rejects([&]{bad.validate();});
    bad=h;bad.payload[1]=0;rejects([&]{bad.validate();});
    bad=h;bad.directory[1]=0;rejects([&]{bad.validate();});
    bad=h;bad.labels[1]=7;rejects([&]{bad.validate();});
    bool sat=false; assert(priority(-5,3,false,sat)==-2 && !sat);
    assert(priority(-5,3,true,sat)==-5); assert(priority(-5,0,false,sat)==-5);
    assert(priority(FLT_MAX/2,FLT_MAX,false,sat)==FLT_MAX && sat);
    sat=false;assert(priority(FLT_MAX,1,false,sat)==FLT_MAX && !sat);
    std::cout << "GPU predicate host membership/validation/priority tests passed\n";
}
