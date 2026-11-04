#include "app/runtime_defaults.h"

#include <cassert>

int main()
{
    using namespace go_fann::defaults;

    assert(max_highfreq_rerank_shortlist == 512);
    assert(default_range_posting_shortlist == 200);
    assert(max_range_posting_shortlist == 400);
    assert(default_range_posting_postfilter_shortlist_multiplier == 10);
    assert(max_range_posting_postfilter_shortlist == 4096);
    assert(!default_lowfreq_rerank_merge_routes);
    assert(effective_highfreq_rerank_shortlist(100,
                                               512,
                                               highfreq_rerank_mode_cpu,
                                               highfreq_rerank_algorithm_heuristic) == 512);
    assert(effective_highfreq_rerank_shortlist(400,
                                               256,
                                               highfreq_rerank_mode_cpu,
                                               highfreq_rerank_algorithm_heuristic) == 400);
    assert(effective_highfreq_rerank_shortlist(100,
                                               512,
                                               highfreq_rerank_mode_cpu,
                                               highfreq_rerank_algorithm_exact) == 100);
    assert(effective_highfreq_rerank_shortlist(100,
                                               512,
                                               highfreq_rerank_mode_gpu,
                                               highfreq_rerank_algorithm_heuristic) == 100);
    return 0;
}
