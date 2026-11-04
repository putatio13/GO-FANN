#include "pq_resident_posting_fixture.h"
#include <iostream>

int main() {
    using namespace go_fann::pq_resident_posting;
    PqResidentFixture f;
    auto v = f.view();
    uint32_t out[9], begin = 99, count = 99;
    pq_resident_probe(v, out);
    pq_resident_expect(out);
    assert(!bounds(v, 3, begin, count) && begin == 0 && count == 0);
    f.groups[0] = UINT32_MAX;
    assert(!bounds(v, 0, begin, count));
    f.groups[0] = 1;
    f.offsets[1] = UINT32_MAX;
    assert(!bounds(v, 0, begin, count) && count == 0);
    f.offsets[1] = 3;
    f.sizes[1] = UINT32_MAX;
    assert(!bounds(v, 0, begin, count) && count == 0);
    assert(candidate(v, 3, 0) == UINT32_MAX);
    assert(candidate(v, 0, 6) == UINT32_MAX);
    f.ids[5] = 6;
    assert(candidate(v, 0, 5) == UINT32_MAX);
    v.base_ranges = nullptr; // Historic producer skips ranges when absent.
    assert(candidate(v, 1, 0) == 4);
    std::cout << "PQ resident route/group/sample, masks, numeric and bounds checks passed\n";
}
