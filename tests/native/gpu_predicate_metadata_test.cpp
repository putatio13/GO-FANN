#include "predicates/gpu_predicate_metadata.h"
#include <cassert>
#include <functional>
using namespace go_fann::gpu_predicate;
struct Metadata {
    uint32_t original_label_count=3, label_count=3;
    std::vector<uint32_t> group_posting_size{3,7,0}, cat_freq{3,7,0}, label_size{0,7,0};
    std::vector<uint32_t> original_label_cardinality;
};
struct Layout {std::vector<uint32_t> label_to_group{0,1,2}; std::vector<std::vector<uint32_t>> groups{{0},{1},{2}};};
void rejects(const std::function<void()>& f){bool caught=false;try{f();}catch(const std::runtime_error&){caught=true;}assert(caught);}
int main(){
    HostIndex h;h.rows=10;h.threshold=3;h.labels={1};h.cardinalities={7};
    Metadata m;Layout g;
    validate_identity_cardinalities(h,m,g,3); // Legacy sidecar without optional original counts.
    m.original_label_cardinality={3,7,0};validate_identity_cardinalities(h,m,g,3);
    // Identity means singleton mapping, not group_id == label_id.
    g.label_to_group={1,0,2};g.groups={{1},{0},{2}};
    m.group_posting_size=m.cat_freq={7,3,0};m.label_size={7,0,0};
    validate_identity_cardinalities(h,m,g,3);
    m.original_label_cardinality.clear();validate_identity_cardinalities(h,m,g,3);
    auto bad=m;bad.cat_freq[0]=6;rejects([&]{validate_identity_cardinalities(h,bad,g,3);});
    bad=m;bad.label_size[1]=3;rejects([&]{validate_identity_cardinalities(h,bad,g,3);});
    bad=m;bad.original_label_cardinality={3,6,0};rejects([&]{validate_identity_cardinalities(h,bad,g,3);});
    auto badg=g;badg.groups[0].push_back(2);rejects([&]{validate_identity_cardinalities(h,m,badg,3);});
    badg=g;badg.label_to_group[0]=0;rejects([&]{validate_identity_cardinalities(h,m,badg,3);});
    auto badh=h;badh.cardinalities={6};rejects([&]{validate_identity_cardinalities(badh,m,g,3);});
}
