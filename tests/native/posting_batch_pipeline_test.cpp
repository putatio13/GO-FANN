#include "execution/posting_batch_pipeline.h"
#include "execution/posting_completion.h"

#include <cassert>
#include <future>
#include <string>

using go_fann::posting_completion::BatchPipeline;

void test_bounded_out_of_order_completion()
{
    std::atomic<bool> cancelled{false};
    BatchPipeline pipeline(2, cancelled);
    std::promise<void> first_started, second_started, release_first, release_second;
    auto first_gate = release_first.get_future().share();
    auto second_gate = release_second.get_future().share();
    size_t first_slot = SIZE_MAX, second_slot = SIZE_MAX, third_slot = SIZE_MAX;
    pipeline.submit([&](size_t slot) {
        first_slot = slot;
        first_started.set_value();
        first_gate.wait();
    });
    pipeline.submit([&](size_t slot) {
        second_slot = slot;
        second_started.set_value();
        second_gate.wait();
    });
    first_started.get_future().get();
    second_started.get_future().get();
    assert(first_slot != second_slot);
    auto submission = std::async(std::launch::async, [&] {
        return pipeline.submit([&](size_t slot) { third_slot = slot; });
    });
    assert(submission.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout);
    release_second.set_value();
    assert(submission.wait_for(std::chrono::seconds(2)) == std::future_status::ready);
    assert(submission.get()); // The slow first batch has not been released.
    release_first.set_value();
    pipeline.finish();
    assert(third_slot == second_slot && pipeline.peak() == 2);
}

void test_error_drain_and_cancellation()
{
    std::atomic<bool> cancelled{false};
    BatchPipeline pipeline(2, cancelled);
    std::promise<void> started, release;
    auto gate = release.get_future().share();
    pipeline.submit([&](size_t) { started.set_value(); gate.wait(); });
    started.get_future().get();
    pipeline.submit([](size_t) { throw std::runtime_error("posting test error"); });
    auto completion = std::async(std::launch::async, [&] {
        try { pipeline.finish(); }
        catch (const std::runtime_error& error) {
            return std::string(error.what()) == "posting test error";
        }
        return false;
    });
    assert(completion.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout);
    release.set_value();
    assert(completion.get() && cancelled.load());
    bool threw = false;
    try { pipeline.submit([](size_t) { assert(false); }); }
    catch (const std::runtime_error&) { threw = true; }
    assert(threw);
    std::atomic<bool> stopped{true};
    BatchPipeline inactive(2, stopped);
    assert(!inactive.submit([](size_t) { assert(false); }));
    inactive.finish();
}

void test_complete_groups_and_reused_slots()
{
    using namespace go_fann::posting_completion;
    std::vector<uint32_t> originals(70);
    for (size_t i = 0; i < originals.size(); ++i) originals[i] = i / 10;
    const auto groups = Groups::make(originals, 7);
    std::vector<unsigned> visits(originals.size(), 0);
    std::atomic<bool> cancelled{false};
    for (size_t repeat = 0; repeat < 8; ++repeat) {
        BatchPipeline pipeline(2, cancelled);
        consume(groups, 4, cancelled, [](size_t) { return true; },
            [&](const std::vector<size_t>& positions) {
                pipeline.submit([&, positions](size_t) {
                    assert(positions.size() == 10); // Never split an oversized group.
                    for (size_t p : positions) {
                        assert(originals[p] == originals[positions.front()]);
                        ++visits[p];
                    }
                });
            });
        pipeline.finish();
    }
    for (auto count : visits) assert(count == 8);
}

void test_environment_validation()
{
    using go_fann::posting_completion::stream_inflight_from_env;
    unsetenv("GO_FANN_POSTING_STREAM_INFLIGHT");
    assert(stream_inflight_from_env() == 0);
    for (const char* value : {"0", "1", "2", "8"}) {
        setenv("GO_FANN_POSTING_STREAM_INFLIGHT", value, 1);
        assert(stream_inflight_from_env() == static_cast<size_t>(std::stoi(value)));
    }
    for (const char* value : {"-1", "9", "2x", " 2", "999999999999999"}) {
        setenv("GO_FANN_POSTING_STREAM_INFLIGHT", value, 1);
        bool threw = false;
        try { stream_inflight_from_env(); }
        catch (const std::invalid_argument&) { threw = true; }
        assert(threw);
    }
    unsetenv("GO_FANN_POSTING_STREAM_INFLIGHT");
}

int main()
{
    test_bounded_out_of_order_completion();
    test_error_drain_and_cancellation();
    test_complete_groups_and_reused_slots();
    test_environment_validation();
}
