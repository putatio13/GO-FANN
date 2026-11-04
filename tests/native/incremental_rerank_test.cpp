#include "execution/incremental_rerank.h"
#include <cassert>
#include <iostream>
#include <future>
#include <atomic>
#include <map>
#include <random>
#include <set>
#include <thread>
using namespace go_fann::incremental_rerank;

static void semantic_cases() {
    QueryState q;
    std::vector<ScoredId> incoming{{100,11}};
    std::vector<uint32_t> issued;
    q.offer(incoming,1); q.issue(issued); assert(issued == std::vector<uint32_t>{11});
    q.complete(11,0.1f,1);
    incoming={{1,22}};q.offer(incoming,1);q.issue(issued);q.complete(22,5,1);
    assert(q.exact() == std::vector<ScoredId>({{0.1f,11}}));
    incoming={{0.5f,33}};q.offer(incoming,1);q.issue(issued);q.complete(33,0.05f,1);
    incoming={{0,11}};q.offer(incoming,1);q.issue(issued);assert(issued.empty());
    assert(q.exact() == std::vector<ScoredId>({{0.05f,33}}));
    bool threw=false;try {q.complete(11,0.1f,1);} catch(const std::logic_error&){threw=true;}
    assert(threw);
    QueryState inflight;
    incoming={{10,1}};inflight.offer(incoming,1);inflight.issue(issued);
    incoming={{1,2}};inflight.offer(incoming,1);inflight.issue(issued);
    inflight.complete(2,4,1);inflight.complete(1,0.1f,1);
    assert(inflight.exact().front().second==1);
    incoming={{0,1}};inflight.offer(incoming,1);inflight.issue(issued);assert(issued.empty());
    QueryState ties;
    incoming={{1,3},{1,2},{1,1},{0,UINT32_MAX},{NAN,7}};
    ties.offer(incoming,2);ties.issue(issued);assert(issued==std::vector<uint32_t>({1,2}));
    ties.complete(2,1,2);ties.complete(1,1,2);assert(ties.exact().front().second==1);
}

// Independent slow reference: map+sort for the full seen union and exact winners.
static void random_reference() {
    std::mt19937 rng(1234);
    for (size_t cap : {1,7,16,100,400,1024}) {
        QueryState q;
        std::map<uint32_t,float> all;
        std::set<uint32_t> evaluated;
        std::vector<ScoredId> exact;
        std::vector<uint32_t> issued;
        for (int batch=0;batch<80;++batch) {
            std::vector<ScoredId> incoming;
            for (int i=0;i<30;++i) {
                uint32_t id=rng()%1500;float score=float(rng()%10000);
                incoming.emplace_back(score,id);
                auto it=all.find(id);if(it==all.end())all[id]=score;else it->second=std::min(score,it->second);
            }
            q.offer(incoming,cap);
            std::vector<ScoredId> expected;
            for(auto [id,score]:all)expected.emplace_back(score,id);
            std::sort(expected.begin(),expected.end());if(expected.size()>cap)expected.resize(cap);
            assert(q.approximate()==expected);
            q.issue(issued);
            std::vector<uint32_t> missing;
            for(auto [score,id]:expected)if(evaluated.insert(id).second)missing.push_back(id);
            assert(issued==missing);
            std::reverse(issued.begin(),issued.end());
            for(uint32_t id:issued){float d=float((id*37)%977);q.complete(id,d,10);exact.emplace_back(d,id);}
            std::sort(exact.begin(),exact.end());if(exact.size()>10)exact.resize(10);
            assert(q.exact()==exact);
        }
        assert(q.stats().completed==evaluated.size());
    }
}
static void concurrent_producers() {
    State state(2,100,10);
    std::vector<std::thread> threads;
    for(int producer=0;producer<8;++producer)threads.emplace_back([&] {
        for(int batch=0;batch<30;++batch)state.with_query(0,[&](QueryState& q){
            std::vector<ScoredId> incoming;std::vector<uint32_t> ids;
            for(uint32_t id=0;id<100;++id)incoming.emplace_back(float(id),id);
            q.offer(incoming,state.cap);q.issue(ids);
            for(uint32_t id:ids)q.complete(id,float(100-id),state.topk);
        });
    });
    for(auto& t:threads)t.join();
    assert(state.stats().issued==100 && state.stats().completed==100);
    state.with_query(1,[&](QueryState& q){
        std::vector<ScoredId> in{{0,99}};std::vector<uint32_t> ids;
        q.offer(in,100);q.issue(ids);assert(ids==std::vector<uint32_t>{99});q.complete(99,9,10);
    });
    std::vector<std::vector<uint64_t>> ids;std::vector<std::vector<float>> distances;
    state.publish(ids,distances);assert(ids[0][0]==99 && ids[1][0]==99 && distances[1][0]==9);
    // Failed scoring cannot publish a partial success; the next iteration is fresh.
    State failed(1,10,1);
    try {failed.with_query(0,[](QueryState& q){std::vector<ScoredId> a{{1,2}};std::vector<uint32_t> b;
        q.offer(a,10);q.issue(b);throw std::runtime_error("scorer");});}catch(const std::runtime_error&){}
    bool threw=false;try{failed.publish(ids,distances);}catch(const std::logic_error&){threw=true;}assert(threw);
}
// Deterministic overlap: a second producer must merge while the first scorer
// is blocked OUTSIDE the lock. A now-evicted in-flight sample still wins exact K.
static void coalesced_scoring() {
    State state(1,1,1);
    std::promise<void> scoring_started, arrivals_done;
    auto started = scoring_started.get_future();
    auto arrived = arrivals_done.get_future();
    std::thread scorer([&] {
        std::vector<ScoredId> incoming{{10,1}};
        std::vector<uint32_t> ids;
        state.with_query(0,[&](auto& q){q.offer(incoming,1);assert(q.acquire(ids));});
        scoring_started.set_value();
        arrived.wait();
        assert(ids == std::vector<uint32_t>{1});
        state.with_query(0,[&](auto& q){q.complete(1,0.1f,1);assert(q.next(ids));});
        assert(ids == std::vector<uint32_t>{3}); // intermediate ID2 was coalesced away
        state.with_query(0,[&](auto& q){q.complete(3,5,1);assert(!q.next(ids));});
    });
    started.wait();
    for (uint32_t id : {2u,3u}) state.with_query(0,[&](auto& q){
        std::vector<ScoredId> incoming{{float(4-id),id}};
        std::vector<uint32_t> ids;
        q.offer(incoming,1);assert(!q.acquire(ids));assert(ids.empty());
    });
    arrivals_done.set_value();scorer.join();
    std::vector<std::vector<uint32_t>> ids;std::vector<std::vector<float>> distances;
    state.publish(ids,distances);assert(ids[0][0]==1 && distances[0][0]==0.1f);
    assert(state.stats().issued==2 && state.stats().coalesced==2 && state.stats().score_batches==2);
}

static void unlocked_producers() {
    State state(1,100,10);
    std::atomic<int> computed[100]{};
    std::vector<std::thread> threads;
    for(int worker=0;worker<8;++worker) threads.emplace_back([&]{
        for(int batch=0;batch<20;++batch){
            std::vector<ScoredId> incoming;
            for(uint32_t id=0;id<100;++id) incoming.emplace_back(float(id),id);
            std::vector<uint32_t> ids;bool owns=false;
            state.with_query(0,[&](auto& q){q.offer(incoming,100);owns=q.acquire(ids);});
            while(owns){
                for(auto id:ids){++computed[id];std::this_thread::yield();}
                state.with_query(0,[&](auto& q){for(auto id:ids)q.complete(id,float(id),10);owns=q.next(ids);});
            }
        }
    });
    for(auto& t:threads)t.join();
    for(auto& count:computed)assert(count==1);
    std::vector<std::vector<uint32_t>> ids;std::vector<std::vector<float>> distances;
    state.publish(ids,distances);assert(ids[0][0]==0 && ids[0][9]==9);
    // Completion state survives hash-table growth; duplicate completions fail.
    QueryState q;std::vector<uint32_t> batch;
    for(uint32_t id=0;id<200;++id){std::vector<ScoredId> in{{-float(id),id}};
        q.offer(in,1);q.issue(batch);q.complete(id,float(id),1);}
    bool threw=false;try{q.complete(0,0,1);}catch(const std::logic_error&){threw=true;}
    assert(threw);
}
// The trusted native entry must match the general path for the same delivery
// batches, including improved duplicates, tied thresholds and empty arrivals.
static void sorted_unique_admission() {
    std::mt19937 rng(4567);
    for (size_t cap : {1, 7, 100, 400, 1024}) {
        QueryState general, native;
        for (int batch = 0; batch < 100; ++batch) {
            std::map<uint32_t, float> best;
            for (int n = 0; n < 150; ++n) {
                const uint32_t id = rng() % 2000;
                const float value = float(rng() % 600);
                auto [it, inserted] = best.emplace(id, value);
                if (!inserted) it->second = std::min(it->second, value);
            }
            std::vector<ScoredId> incoming;
            for (auto [id, value] : best) incoming.emplace_back(value, id);
            std::sort(incoming.begin(), incoming.end());
            if (batch % 13 == 0) incoming.clear();
            general.offer(incoming, cap);
            native.offer_sorted_unique(incoming, cap);
            assert(native.approximate() == general.approximate());
            std::vector<uint32_t> a, b;
            assert(general.acquire(a) == native.acquire(b));
            assert(a == b);
            if (!a.empty()) {
                // No arrivals between acquisition and completion: next() must
                // release the owner without scanning/issuing the shortlist.
                for (auto id : a) { general.complete(id, float(id % 137), 10); native.complete(id, float(id % 137), 10); }
                assert(!general.next(a) && !native.next(b));
                assert(a.empty() && b.empty());
            }
            assert(general.exact() == native.exact());
            assert(general.stats().offered == native.stats().offered);
            assert(general.stats().repeated == native.stats().repeated);
        }
    }
}
int main(){semantic_cases();sorted_unique_admission();random_reference();concurrent_producers();coalesced_scoring();unlocked_producers();std::cout<<"incremental rerank tests passed\n";}
