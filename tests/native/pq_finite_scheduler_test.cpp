#include "search/pq/pq_finite_scheduler.h"

#include <cstdio>
#include <atomic>
#include <chrono>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_original_query_id_validation()
{
    using namespace go_fann::backend_pq;
    require(validate_finite_query_ids({7, 2, 9}, 10).error ==
                PqFiniteQueryIdError::NONE,
            "sparse original query IDs must be accepted");
    require(validate_finite_query_ids({7, 10}, 10).error ==
                PqFiniteQueryIdError::OUT_OF_RANGE,
            "out-of-range original query ID must be rejected before launch");
    require(validate_finite_query_ids({7, 2, 7}, 10).error ==
                PqFiniteQueryIdError::DUPLICATE,
            "duplicate original query ID must be rejected before launch");
    require(validate_finite_query_ids({0, 1, 2}, 2).error ==
                PqFiniteQueryIdError::TOO_MANY,
            "active context count must not exceed context capacity");
}

void test_mailbox_identity_validation()
{
    using go_fann::backend_pq::finite_mailbox_identity_is_valid;
    require(finite_mailbox_identity_is_valid(123, 1, 9),
            "valid row/buffer/generation identity was rejected");
    require(!finite_mailbox_identity_is_valid(-1, 0, 1),
            "negative rows must be rejected");
    require(!finite_mailbox_identity_is_valid(1, 2, 1),
            "buffers outside the two serial slots must be rejected");
    require(!finite_mailbox_identity_is_valid(1, 0, 0),
            "generation zero must be rejected");
    require(!finite_mailbox_identity_is_valid(1, 0,
                                               static_cast<int64_t>(UINT32_MAX) + 1),
            "generations wider than the device token must be rejected");
}

void test_partial_batch_timeout()
{
    using go_fann::backend_pq::finite_ready_batch_should_flush;
    using go_fann::backend_pq::finite_partial_batch_remaining_wait_ns;
    require(!finite_ready_batch_should_flush(0, 1024, 100000, 50000),
            "empty batches must not launch");
    require(finite_ready_batch_should_flush(1024, 1024, 0, 50000),
            "full batches must launch immediately");
    require(!finite_ready_batch_should_flush(3, 1024, 49999, 50000),
            "partial batches must wait until the bounded timeout");
    require(finite_ready_batch_should_flush(3, 1024, 50000, 50000),
            "partial tail batches must flush at the timeout");
    require(finite_partial_batch_remaining_wait_ns(100, 100, 50) == 50,
            "a newly published batch must retain the full timeout");
    require(finite_partial_batch_remaining_wait_ns(100, 125, 50) == 25,
            "partial wait must be reduced by the oldest event age");
    require(finite_partial_batch_remaining_wait_ns(100, 150, 50) == 0,
            "an aged batch must not receive another full timeout");
    require(finite_partial_batch_remaining_wait_ns(100, 175, 50) == 0,
            "remaining wait must saturate at zero");
}

void test_host_demand_batch_flush_conditions()
{
    require(!host_demand_batch_should_flush(0, 16, 5000, 5000),
            "empty host demand batches must not flush");
    require(!host_demand_batch_should_flush(1, 16, 4999, 5000),
            "partial host demand batches must wait for the timeout");
    require(host_demand_batch_should_flush(1, 16, 5000, 5000),
            "host demand timeout must flush a partial batch");
    require(host_demand_batch_should_flush(16, 16, 0, 5000),
            "a full host demand batch must flush immediately");
}

void test_persistent_sequence_exhaustion()
{
    using go_fann::backend_pq::next_finite_persistent_sequence;
    require(next_finite_persistent_sequence(0) == 1,
            "persistent sequence must start at one");
    require(next_finite_persistent_sequence(UINT32_MAX - 2) == UINT32_MAX - 1,
            "last non-stop sequence must remain usable");
    bool rejected = false;
    try {
        next_finite_persistent_sequence(UINT32_MAX - 1);
    } catch (const std::overflow_error&) {
        rejected = true;
    }
    require(rejected, "persistent sequence exhaustion must fail before ABA wrap");
}

void test_ready_ring_wraparound_acknowledgements()
{
    using go_fann::backend_pq::finite_ready_ring_required_consumed_sequence;
    require(finite_ready_ring_required_consumed_sequence(0, 4) == 0,
            "the first ring cycle must not wait for an acknowledgement");
    require(finite_ready_ring_required_consumed_sequence(3, 4) == 0,
            "the final entry in the first ring cycle must remain immediately writable");
    require(finite_ready_ring_required_consumed_sequence(4, 4) == 1,
            "the first wrapped entry must wait for ticket zero acknowledgement");
    require(finite_ready_ring_required_consumed_sequence(7, 4) == 4,
            "the final entry in the second ring cycle has the wrong acknowledgement");
    require(finite_ready_ring_required_consumed_sequence(8, 4) == 5,
            "later ring wraps must advance the required acknowledgement monotonically");

    bool rejected = false;
    try {
        (void)finite_ready_ring_required_consumed_sequence(0, 0);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "zero-capacity ready rings must fail before modulo or waiting");
}

void test_ready_ring_packed_ownership_linearization()
{
    using namespace go_fann::backend_pq;
    std::atomic<uint64_t> query_state{0};

    require(finite_ready_ring_try_claim_ready(query_state, 3) ==
                PqFiniteReadyRingClaimResult::CLAIMED,
            "Ready did not claim a fresh generation");
    uint64_t state = query_state.load(std::memory_order_acquire);
    require(finite_ready_ring_query_generation(state) == 3,
            "Ready claim lost its generation");
    require(finite_ready_ring_query_publishing(state),
            "Ready claim did not protect its publication interval");
    require(finite_ready_ring_try_claim_ready(query_state, 3) ==
                PqFiniteReadyRingClaimResult::DUPLICATE,
            "duplicate Ready was accepted during publication");
    require(finite_ready_ring_try_claim_ready(query_state, 4) ==
                PqFiniteReadyRingClaimResult::PUBLISHING,
            "a later Ready overtook an in-progress publication");
    require(finite_ready_ring_try_claim_done(query_state) ==
                PqFiniteReadyRingClaimResult::PUBLISHING,
            "Done overtook an in-progress Ready publication");

    finite_ready_ring_finish_ready(query_state, 3);
    state = query_state.load(std::memory_order_acquire);
    require(!finite_ready_ring_query_publishing(state),
            "Ready publication did not release ownership");
    require(finite_ready_ring_try_claim_done(query_state) ==
                PqFiniteReadyRingClaimResult::CLAIMED,
            "Done did not linearize after Ready publication");
    state = query_state.load(std::memory_order_acquire);
    require(finite_ready_ring_query_done(state), "Done did not set the terminal bit");
    require(finite_ready_ring_query_generation(state) == 3,
            "Done did not preserve the accepted generation");
    require(finite_ready_ring_try_claim_ready(query_state, 4) ==
                PqFiniteReadyRingClaimResult::TERMINAL,
            "Ready was accepted after Done");
}

void test_ready_ring_concurrent_done_waits_for_ready_publication()
{
    using namespace go_fann::backend_pq;
    std::atomic<uint64_t> query_state{0};
    std::atomic<bool> ready_claimed{false};
    std::atomic<bool> done_saw_publication{false};
    std::atomic<bool> allow_ready_finish{false};
    std::atomic<bool> done_claimed{false};

    std::thread ready([&] {
        require(finite_ready_ring_try_claim_ready(query_state, 1) ==
                    PqFiniteReadyRingClaimResult::CLAIMED,
                "concurrent Ready failed to claim ownership");
        ready_claimed.store(true, std::memory_order_release);
        while (!allow_ready_finish.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        finite_ready_ring_finish_ready(query_state, 1);
    });
    std::thread done([&] {
        while (!ready_claimed.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        require(finite_ready_ring_try_claim_done(query_state) ==
                    PqFiniteReadyRingClaimResult::PUBLISHING,
                "concurrent Done did not observe the Ready publication interval");
        done_saw_publication.store(true, std::memory_order_release);
        while (!allow_ready_finish.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        while (finite_ready_ring_try_claim_done(query_state) ==
               PqFiniteReadyRingClaimResult::PUBLISHING) {
            std::this_thread::yield();
        }
        done_claimed.store(true, std::memory_order_release);
    });

    while (!done_saw_publication.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    require(!done_claimed.load(std::memory_order_acquire),
            "Done linearized before Ready publication was released");
    allow_ready_finish.store(true, std::memory_order_release);
    ready.join();
    done.join();
    require(done_claimed.load(std::memory_order_acquire),
            "Done did not complete after Ready publication");
    require(finite_ready_ring_query_done(query_state.load(std::memory_order_acquire)),
            "concurrent Done did not leave the query terminal");
}

void test_completion_queue_ordering_and_dedup()
{
    using namespace go_fann::backend_pq;
    PqFiniteCompletionQueue queue({7, 2, 9});
    require(queue.capacity() == 3, "queue capacity must equal active contexts");
    require(queue.publish_ready(7, 1, 70, 0), "first ready publish failed");
    require(queue.publish_ready(2, 3, 20, 1), "second ready publish failed");
    require(!queue.publish_ready(7, 1, 70, 0), "duplicate identity was enqueued");

    std::vector<HostCompletionEvent> events;
    require(queue.wait_pop_batch(events, 2, std::chrono::milliseconds(1)),
            "ordered batch was cancelled");
    require(events.size() == 2 && events[0].query_id == 7 && events[1].query_id == 2,
            "completion queue must preserve publication order");
    require(queue.publish_done(7), "first terminal publish failed");
    require(!queue.publish_done(7), "duplicate terminal publish was enqueued");
    require(!queue.publish_ready(7, 2, 71, 1), "ready was accepted after terminal");
    require(!queue.publish_ready(99, 1, 1, 0), "unknown query was accepted");
    const PqFiniteCompletionQueueStats stats = queue.stats();
    require(stats.duplicate_rejections == 2,
            "duplicate counter must include only duplicate ready/done publishes");
    require(stats.terminal_rejections == 1,
            "ready-after-Done must have a separate terminal counter");
    require(stats.invalid_rejections == 1,
            "invalid query identities must have a separate counter");
}

void test_completion_queue_batch_publication()
{
    using namespace go_fann::backend_pq;
    PqFiniteCompletionQueue queue({4, 8, 12, 16});
    const HostCompletionEvent invalid_batch[] = {
        {HostCompletionKind::Ready, 4, 1, 40, 0, 101},
        {HostCompletionKind::Ready, 99, 1, 99, 0, 102},
    };
    require(!queue.publish_ready_batch(invalid_batch, 2),
            "a batch with an unknown query identity was accepted");

    std::vector<HostCompletionEvent> events;
    require(queue.try_pop_batch(events, 4), "invalid batch unexpectedly cancelled queue");
    require(events.empty(), "invalid batch was partially enqueued");

    const HostCompletionEvent batch[] = {
        {HostCompletionKind::Ready, 8, 2, 80, 1, 201},
        {HostCompletionKind::Ready, 4, 3, 41, 0, 202},
        {HostCompletionKind::Ready, 16, 1, 160, 1, 203},
    };
    require(queue.publish_ready_batch(batch, 3), "valid batch publication failed");
    require(queue.try_pop_batch(events, 4), "valid batch dequeue failed");
    require(events.size() == 3 && events[0].query_id == 8 && events[1].query_id == 4 &&
                events[2].query_id == 16,
            "batch publication did not preserve event order");
    require(events[0].published_ns == 201 && events[2].published_ns == 203,
            "batch publication did not preserve published timestamps");
    require(queue.stats().invalid_rejections == 1,
            "invalid batch rejection was not recorded exactly once");
}

void test_completion_queue_partial_tail()
{
    using namespace go_fann::backend_pq;
    PqFiniteCompletionQueue queue({0, 1, 2, 3});
    require(queue.publish_ready(0, 1, 10, 0), "partial ready 0 failed");
    require(queue.publish_ready(1, 1, 11, 1), "partial ready 1 failed");
    std::vector<HostCompletionEvent> events;
    const auto start = std::chrono::steady_clock::now();
    require(queue.wait_pop_batch(events, 4, std::chrono::milliseconds(2)),
            "partial tail was cancelled");
    const auto elapsed = std::chrono::steady_clock::now() - start;
    require(events.size() == 2, "partial tail did not flush");
    require(elapsed >= std::chrono::milliseconds(1), "partial tail ignored timeout");
}

void test_completion_queue_nonblocking_drain()
{
    using namespace go_fann::backend_pq;
    PqFiniteCompletionQueue queue({4, 8, 12});
    std::vector<HostCompletionEvent> events;
    require(queue.try_pop_batch(events, 2), "empty nonblocking drain was cancelled");
    require(events.empty(), "empty nonblocking drain returned events");

    require(queue.publish_ready(8, 1, 19, 0), "nonblocking ready publish failed");
    require(queue.publish_done(4), "nonblocking done publish failed");
    require(queue.try_pop_batch(events, 1), "first nonblocking drain failed");
    require(events.size() == 1 && events[0].kind == HostCompletionKind::Ready &&
                events[0].query_id == 8,
            "nonblocking drain did not preserve ready order");
    require(queue.try_pop_batch(events, 2), "second nonblocking drain failed");
    require(events.size() == 1 && events[0].kind == HostCompletionKind::Done &&
                events[0].query_id == 4,
            "nonblocking drain did not preserve done order");

    queue.cancel();
    require(!queue.try_pop_batch(events, 2), "cancelled nonblocking drain stayed active");
}

void test_completion_queue_cancellation_wakes_backpressure()
{
    using namespace go_fann::backend_pq;
    PqFiniteCompletionQueue queue({0, 1});
    require(queue.publish_ready(0, 1, 10, 0), "queue fill 0 failed");
    require(queue.publish_ready(1, 1, 11, 0), "queue fill 1 failed");
    std::atomic<bool> returned{false};
    std::thread producer([&] {
        require(!queue.publish_done(0), "cancelled backpressured publish succeeded");
        returned.store(true, std::memory_order_release);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    queue.cancel();
    producer.join();
    require(returned.load(std::memory_order_acquire), "cancel did not wake producer");
    require(queue.stats().backpressure_waits == 1, "backpressure wait was not counted");
    require(queue.stats().cancelled_rejections == 1,
            "cancelled producer rejection was not counted separately");

    std::vector<HostCompletionEvent> events;
    require(!queue.wait_pop_batch(events, 2, std::chrono::milliseconds(1)),
            "cancelled consumer reported a batch");
}

void test_null_sink_is_noop()
{
    require(!publish_host_ready(nullptr, 1, 2, 3, 0), "null ready sink published");
    const HostCompletionEvent event{HostCompletionKind::Ready, 1, 2, 3, 0, 0};
    require(!publish_host_ready_batch(nullptr, &event, 1), "null batched ready sink published");
    require(!publish_host_done(nullptr, 1), "null done sink published");
}

}  // namespace

int main()
{
    try {
        test_original_query_id_validation();
        test_mailbox_identity_validation();
        test_partial_batch_timeout();
        test_host_demand_batch_flush_conditions();
        test_persistent_sequence_exhaustion();
        test_ready_ring_wraparound_acknowledgements();
        test_ready_ring_packed_ownership_linearization();
        test_ready_ring_concurrent_done_waits_for_ready_publication();
        test_completion_queue_ordering_and_dedup();
        test_completion_queue_batch_publication();
        test_completion_queue_partial_tail();
        test_completion_queue_nonblocking_drain();
        test_completion_queue_cancellation_wakes_backpressure();
        test_null_sink_is_noop();
        std::puts("PQ finite scheduler tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "PQ finite scheduler test failed: %s\n", error.what());
        return 1;
    }
}
