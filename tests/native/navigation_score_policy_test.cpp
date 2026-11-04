#include "search/common/navigation_score_policy.h"

#include <cassert>
#include <iostream>

using namespace go_fann::navigation_score_policy;

int main()
{
    for (const auto* backend : {"pq", "rabitq"}) {
        for (const auto* operation : {"search", "benchmark"}) {
            assert(uses_raw_scores(backend, operation, "coupled", "mailbox", "memory"));
            assert(!uses_raw_scores(backend, operation, "finite", "mailbox", "memory"));
            assert(!uses_raw_scores(backend, operation, "coupled", "device_resident", "memory"));
            assert(!uses_raw_scores(backend, operation, "coupled", "mailbox", "disk"));
        }
        for (const auto* operation : {"latency", "build-graph", "build-rabitq"})
            assert(!uses_raw_scores(backend, operation, "coupled", "mailbox", "memory"));
    }
    assert(!uses_raw_scores("full_gpu", "search", "coupled", "mailbox", "memory"));
    validate_bonus_axis(true, 0);
    validate_bonus_axis(true, 1);
    validate_bonus_axis(false, 3);
    bool rejected = false;
    try { validate_bonus_axis(true, 2); }
    catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
    std::cout << "navigation score policy tests passed\n";
}
