#include "app/config_loader.h"

#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unistd.h>

int main()
{
    const std::string path = "/tmp/rabitq-predicate-config-" +
        std::to_string(getpid()) + ".json";
    const auto load = [&](const char* text) {
        { std::ofstream out(path); out << text; }
        try {
            auto config = go_fann::load_config_file(path);
            std::remove(path.c_str());
            return config;
        } catch (...) {
            std::remove(path.c_str());
            throw;
        }
    };
    const auto absent = load("{}");
    if (absent.rabitq_predicate_mode || absent.rabitq_predicate_storage || absent.rabitq_predicate_index_path ||
        absent.rabitq_predicate_bias_alpha || absent.rabitq_predicate_diagnostics) {
        throw std::runtime_error("absent predicate keys must preserve defaults");
    }
    if (absent.pq_lowfreq_scan_mode || absent.posting_predicate_order)
        throw std::runtime_error("absent posting keys must preserve defaults");
    const auto posting = load(R"({"pq_lowfreq_scan_mode":"host", "pq":{"lowfreq_scan_mode":"resident"},
                                 "posting_predicate_order":"score_first"})");
    if (posting.pq_lowfreq_scan_mode != "resident" || posting.posting_predicate_order != "score_first")
        throw std::runtime_error("posting nested/root keys did not load");
    const auto posting_auto = load(R"({"pq_lowfreq_scan_mode":"resident", "pq":{"lowfreq_scan_mode":"auto"}})");
    if (posting_auto.pq_lowfreq_scan_mode != "auto")
        throw std::runtime_error("nested auto must survive loading to reset explicit root policy");
    const auto foreign = load(R"({"backend":"rabitq", "pq":{"lowfreq_scan_mode":"resident"}})");
    if (foreign.pq_lowfreq_scan_mode != "resident")
        throw std::runtime_error("explicit unsupported PQ policy must survive for validation");
    const auto simplified = load(R"({"rabitq":{"predicate_mode":"cpu_simplified"}})");
    if (simplified.rabitq_predicate_mode != "cpu_simplified" ||
        simplified.rabitq_predicate_index_path) {
        throw std::runtime_error("CPU simplification must not require a bitmap path");
    }
    const auto flat = load(R"({"rabitq_predicate_mode":"gpu_exact_bias",
        "rabitq_predicate_storage":"csr", "rabitq_predicate_index_path":"/index", "rabitq_predicate_bias_alpha":0.15,
        "rabitq_predicate_diagnostics":true})");
    if (flat.rabitq_predicate_storage != "csr" || flat.rabitq_predicate_mode != "gpu_exact_bias" ||
        flat.rabitq_predicate_index_path != "/index" ||
        flat.rabitq_predicate_bias_alpha != 0.15 ||
        flat.rabitq_predicate_diagnostics != true) {
        throw std::runtime_error("flat predicate configuration did not load");
    }
    const auto nested = load(R"({"rabitq_predicate_mode":"cpu",
        "rabitq_predicate_storage":"csr", "rabitq_predicate_index_path":"old", "rabitq_predicate_bias_alpha":0.3,
        "rabitq_predicate_diagnostics":true,
        "rabitq":{"predicate_mode":"gpu_exact", "predicate_storage":"roaring", "predicate_index_path":"new",
                  "predicate_bias_alpha":0, "predicate_diagnostics":false}})");
    if (nested.rabitq_predicate_storage != "roaring" || nested.rabitq_predicate_mode != "gpu_exact" ||
        nested.rabitq_predicate_index_path != "new" ||
        nested.rabitq_predicate_bias_alpha != 0.0 ||
        nested.rabitq_predicate_diagnostics != false) {
        throw std::runtime_error("nested predicate keys must override flat keys, including zero/false");
    }
    const auto neutral = load(R"({"backend":"pq", "rabitq_predicate_mode":"cpu",
        "rabitq":{"predicate_storage":"roaring"},
        "pq":{"predicate_mode":"gpu_exact", "predicate_bias_alpha":0},
        "predicate_mode":"auto", "predicate_storage":"csr",
        "predicate_diagnostics":false})");
    if (neutral.rabitq_predicate_mode != "auto" || neutral.rabitq_predicate_storage != "csr" ||
        neutral.rabitq_predicate_bias_alpha != 0.0 || neutral.rabitq_predicate_diagnostics != false) {
        throw std::runtime_error("neutral keys must override legacy aliases; PQ nested zero must survive");
    }
    for (const char* invalid : {
             R"({"pq_lowfreq_scan_mode":true})",
             R"({"pq":{"lowfreq_scan_mode":false}})",
             R"({"posting_predicate_order":0})",
             R"({"predicate_mode":true})",
             R"({"pq":{"predicate_bias_alpha":"0.15"}})",
             R"({"rabitq_predicate_mode":true})",
             R"({"rabitq_predicate_storage":false})",
             R"({"rabitq":{"predicate_storage":17}})",
             R"({"rabitq_predicate_index_path":42})",
             R"({"rabitq_predicate_bias_alpha":"0.15"})",
             R"({"rabitq_predicate_diagnostics":"false"})",
             R"({"rabitq":{"predicate_mode":null}})",
             R"({"rabitq":{"predicate_bias_alpha":[]}})"}) {
        bool rejected = false;
        try { load(invalid); } catch (const std::runtime_error&) { rejected = true; }
        if (!rejected) {
            throw std::runtime_error("mistyped predicate configuration silently accepted");
        }
    }
}
