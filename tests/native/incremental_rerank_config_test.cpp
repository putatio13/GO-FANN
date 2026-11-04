#include "app/config_loader.h"
#include <cassert>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <unistd.h>
using go_fann::load_config_file;
static go_fann::AppConfigValues load(const std::string& content) {
    char path[]="/tmp/incremental_config_XXXXXX";int fd=mkstemp(path);assert(fd>=0);close(fd);
    std::ofstream(path)<<content;
    try {auto value=load_config_file(path);std::remove(path);return value;}
    catch(...) {std::remove(path);throw;}
}
int main() {
    assert(!load("{}").rabitq_incremental_rerank_shortlist);
    assert(load(R"({"rabitq_incremental_rerank_shortlist":0})").rabitq_incremental_rerank_shortlist==0);
    assert(load(R"({"rabitq_incremental_rerank_shortlist":100})").rabitq_incremental_rerank_shortlist==100);
    for(const std::string value:{"-1","100.5","true","\"100\"","2147483648"}) {
        bool threw=false;try{load("{\"rabitq_incremental_rerank_shortlist\":"+value+"}");}
        catch(const std::runtime_error&){threw=true;}assert(threw);
    }
    std::puts("incremental config tests passed");
}
