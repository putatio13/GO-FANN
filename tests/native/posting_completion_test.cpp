#include "execution/posting_completion.h"

#include <cassert>
#include <future>
#include <numeric>

int main()
{
    using namespace go_fann::posting_completion;
    // Disjoint groups, out-of-order completion, and a 17-route group exceeding
    // both the batch hint and the historical eight-route assumption.
    std::vector<uint32_t> originals(20, 2);
    originals[1] = originals[19] = 0;
    originals[3] = 1;
    const auto groups = Groups::make(originals, 3);
    std::atomic<bool> cancelled{false};
    std::vector<unsigned> visits(20, 0);
    unsigned polls = 0, batches = 0;
    consume(groups, 8, cancelled,
            [&](size_t route) { ++polls; return route != 0 || polls > 40; },
            [&](const std::vector<size_t>& positions) {
                ++batches;
                for (size_t p : positions) {
                    ++visits[p];
                    for (size_t q = 0; q < originals.size(); ++q) {
                        if (originals[q] == originals[p])
                            assert(std::find(positions.begin(), positions.end(), q) != positions.end());
                    }
                }
            });
    assert(batches >= 2);
    for (unsigned count : visits) assert(count == 1);

    auto worker = std::async(std::launch::async, [&] {
        consume(groups, 8, cancelled, [](size_t) { return false; },
                [](const auto&) { assert(false); });
    });
    cancelled.store(true, std::memory_order_release);
    assert(worker.wait_for(std::chrono::seconds(1)) == std::future_status::ready);
    worker.get();
    bool rejected = false;
    try { Groups::make({4}, 4); }
    catch (const std::out_of_range&) { rejected = true; }
    assert(rejected);
}
