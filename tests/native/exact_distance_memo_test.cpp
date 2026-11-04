#include "execution/exact_distance_memo.h"
#include "execution/rabitq_cpu_worker_pool.h"
#include <cassert>
#include <future>
#include <iostream>
#include <numeric>
#include <thread>

using go_fann::ExactDistanceMemo;

int main() {
    int base, queries;
    const ExactDistanceMemo::Domain domain{4, 1000, 64, 2, &base, &queries};
    ExactDistanceMemo memo(domain);
    memo.require_domain(domain);
    auto bad = domain; ++bad.dimensions;
    try { memo.require_domain(bad); assert(false); } catch (const std::invalid_argument&) {}
    try { memo.evaluate(4, 0, []{return 0.f;}); assert(false); } catch (const std::out_of_range&) {}
    try { memo.evaluate(0, 1000, []{return 0.f;}); assert(false); } catch (const std::out_of_range&) {}

    std::atomic<int> calls{0};
    std::promise<void> entered, release;
    auto go = release.get_future().share();
    auto first = std::async(std::launch::async, [&] {
        return memo.evaluate(0, 7, [&] { ++calls; entered.set_value(); go.wait(); return 7.25f; });
    });
    entered.get_future().wait();
    std::vector<std::future<float>> waiters;
    for (int i=0; i<8; ++i) waiters.push_back(std::async(std::launch::async, [&] {
        return memo.evaluate(0, 7, [&] { ++calls; return -1.f; });
    }));
    release.set_value();
    assert(first.get() == 7.25f);
    for (auto& waiter : waiters) assert(waiter.get() == 7.25f);
    assert(calls == 1);

    // Batch misses are computed once, before resolving another producer's entries.
    std::promise<void> batch_entered, batch_release;
    auto batch_go = batch_release.get_future().share();
    auto batch = std::async(std::launch::async, [&] {
        uint32_t ids[]{3,4,3}; float scores[3];
        memo.evaluate_batch(1, ids, 3, scores, [&](const uint32_t* ids, size_t n, float* out) {
            assert(n==2); batch_entered.set_value(); batch_go.wait();
            for(size_t i=0;i<n;++i) out[i]=ids[i]+.5f;
        });
        assert(scores[0]==3.5f && scores[1]==4.5f && scores[2]==3.5f);
    });
    batch_entered.get_future().wait();
    auto other = std::async(std::launch::async, [&] {
        uint32_t ids[]{4,5,3}; float scores[3];
        memo.evaluate_batch(1, ids, 3, scores, [&](const uint32_t* ids, size_t n, float* out) {
            assert(n==1 && ids[0]==5); out[0]=5.5f; batch_release.set_value();
        });
        assert(scores[0]==4.5f && scores[1]==5.5f && scores[2]==3.5f);
    });
    batch.get(); other.get();
    auto throwing = [&] { return memo.evaluate(2, 1, []()->float {throw std::runtime_error("distance failed");}); };
    try { throwing(); assert(false); } catch(const std::runtime_error&) {}
    try { memo.evaluate(2,1,[]{assert(false);return 0.f;}); assert(false); } catch(const std::runtime_error&) {}
    uint32_t error_ids[]{2,3};float error_scores[2];
    try { memo.evaluate_batch(2,error_ids,2,error_scores,[](const uint32_t*,size_t,float*){throw std::runtime_error("batch failed");}); assert(false); } catch(const std::runtime_error&) {}
    for(auto id:error_ids) {
        try {memo.evaluate(2,id,[]{assert(false);return 0.f;});assert(false);}catch(const std::runtime_error&){}
    }
    try {
        memo.evaluate(2,4,[&]{return memo.evaluate(2,4,[]{return 1.f;});});assert(false);
    } catch(const std::logic_error&) {}

    // Distances cross providers; each provider still applies its own residual,
    // visits its original candidate order and independently retains its results.
    std::vector<uint32_t> candidates{9,2,7,9,4};
    std::vector<std::pair<float,uint32_t>> reference, reused;
    for(int provider=0;provider<3;++provider) {
        for(auto id:candidates) {
            if((id+provider)%3==0) continue;
            const float expected=id*.125f;
            reference.emplace_back(expected,id);
            reused.emplace_back(memo.evaluate(3,id,[&]{return expected;}),id);
        }
    }
    assert(reference==reused);
    assert(memo.stats().hits>0 && memo.stats().entries==memo.stats().computations);
    ExactDistanceMemo next(domain);
    assert(next.stats().entries==0);
    assert(next.evaluate(0,7,[]{return 99.f;})==99.f); // iteration isolation
    ExactDistanceMemo masked(domain,{0,1,0,0});
    int bypass=0;
    for(int i=0;i<2;++i) masked.evaluate(0,1,[&]{++bypass;return 1.f;});
    assert(bypass==2 && masked.stats().entries==0);

    // Bounded workers need no nested submissions or all-provider barrier.
    go_fann::backend_rabitq::RabitqCpuWorkerPool pool(2);
    auto work=pool.submit(16,[&](size_t worker){
        for(uint32_t i=0;i<100;++i) {
            const auto id=(i+worker)%100;
            assert(next.evaluate(1,id,[&]{return float(id);})==float(id));
        }
    });
    work->wait();
    assert(next.stats().computations<=65);
    assert(next.stats().entries<=65);

    // Low reuse stops caching after a bounded probation, and neither overflow
    // nor bypass changes computed scores. Stats count only cache admissions.
    ExactDistanceMemo economical(domain);
    for(uint32_t id=0;id<400;++id)
        assert(economical.evaluate(0,id,[&]{return float(id);})==float(id));
    assert(economical.stats().entries==64);
    assert(economical.stats().computations==64);
    assert(economical.stats().adaptive_bypassed_queries==1);
    assert(economical.evaluate(0,0,[]{return 0.f;})==0.f);
    assert(economical.stats().hits==0);

    // High reuse keeps a bounded cache, while keys outside it compute directly.
    ExactDistanceMemo useful(domain);
    for(int i=0;i<5;++i) assert(useful.evaluate(0,0,[]{return 0.f;})==0.f);
    uint32_t large_ids[100]; float large_scores[100];
    std::iota(large_ids,large_ids+100,0);
    useful.evaluate_batch(0,large_ids,100,large_scores,
        [](const uint32_t* ids,size_t n,float* out){for(size_t i=0;i<n;++i) out[i]=float(ids[i]);});
    for(size_t i=0;i<100;++i) assert(large_scores[i]==float(i));
    assert(useful.stats().entries==64 && useful.stats().adaptive_bypassed_queries==0);
    assert(useful.evaluate(0,7,[]{assert(false);return -1.f;})==7.f);
    int overflow=0;
    for(int i=0;i<2;++i) useful.evaluate(0,99,[&]{++overflow;return 99.f;});
    assert(overflow==2 && useful.stats().computations==64);

    // Filling probation while an earlier owner is blocked cannot bypass that
    // owner: a cached waiter still receives the owner's value/error.
    ExactDistanceMemo inflight(domain);
    std::promise<void> owner_started, owner_release;
    auto owner_go=owner_release.get_future().share();
    auto owner=std::async(std::launch::async,[&]{return inflight.evaluate(0,0,[&]{
        owner_started.set_value();owner_go.wait();return 12.f;});});
    owner_started.get_future().wait();
    for(uint32_t id=1;id<80;++id) inflight.evaluate(0,id,[&]{return float(id);});
    assert(inflight.stats().adaptive_bypassed_queries==0);
    auto follower=std::async(std::launch::async,[&]{return inflight.evaluate(0,0,[]{
        assert(false);return 0.f;});});
    // Ensure follower claimed its hit before publishing the first owner's score.
    while(inflight.stats().hits==0) std::this_thread::yield();
    owner_release.set_value();
    assert(owner.get()==12.f && follower.get()==12.f);
    assert(inflight.stats().adaptive_bypassed_queries==1);

    // Cached errors prevent adaptive bypass from hiding a failed computation.
    ExactDistanceMemo failed(domain);
    try {failed.evaluate(0,0,[]()->float{throw std::runtime_error("failed");});assert(false);}
    catch(const std::runtime_error&){}
    for(uint32_t id=1;id<80;++id) failed.evaluate(0,id,[&]{return float(id);});
    assert(failed.stats().adaptive_bypassed_queries==0);
    try {failed.evaluate(0,0,[]{assert(false);return 0.f;});assert(false);}
    catch(const std::runtime_error&){}
    std::cout << "exact distance memo tests passed\n";
}
