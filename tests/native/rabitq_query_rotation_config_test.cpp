#include "app/config_loader.h"
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <unistd.h>

int main() {
    const std::string path="/tmp/rabitq-rotation-config-"+std::to_string(getpid())+".json";
    const auto load=[&](const char* text) {
        {std::ofstream out(path);out<<text;}
        auto config=go_fann::load_config_file(path);std::remove(path.c_str());return config;
    };
    if(load("{}").rabitq_query_rotation.has_value()) throw std::runtime_error("absent override");
    if(load(R"({"rabitq_query_rotation":"legacy"})").rabitq_query_rotation!="legacy")
        throw std::runtime_error("root override");
    if(load(R"({"rabitq_query_rotation":"legacy","rabitq":{"query_rotation":"cublas_fp32"}})").rabitq_query_rotation!="cublas_fp32")
        throw std::runtime_error("nested override precedence");
    if(load(R"({"rabitq_centroid_distances":"legacy","rabitq":{"centroid_distances":"cublas_fp32"}})").rabitq_centroid_distances!="cublas_fp32")
        throw std::runtime_error("centroid override precedence");
}
