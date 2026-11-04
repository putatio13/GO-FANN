#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace go_fann {

// Iteration-local, best-effort exact-distance reuse in one validated original
// query/vector/metric domain. Cached keys have one in-flight owner. Overflow and
// low-reuse queries compute directly; predicates/order/stopping remain caller-owned.
class ExactDistanceMemo {
public:
    struct Domain {
        uint32_t query_count;
        uint32_t sample_count;
        uint32_t dimensions;
        uint32_t scalar_tag;
        const void* base_identity;
        const void* query_identity;
    };
    struct Stats {
        uint64_t computations = 0, hits = 0, entries = 0;
        uint64_t adaptive_bypassed_queries = 0;
    };
    static constexpr size_t max_entries_per_query = 64;

    explicit ExactDistanceMemo(Domain domain, std::vector<uint8_t> enabled = {})
        : domain_(domain), queries_(domain.query_count), enabled_(std::move(enabled)) {
        if (!enabled_.empty() && enabled_.size() != domain.query_count)
            throw std::invalid_argument("exact distance memo enabled mask shape");
        if (!domain.dimensions || !domain.base_identity || !domain.query_identity)
            throw std::invalid_argument("exact distance memo requires a validated vector domain");
    }
    ExactDistanceMemo(const ExactDistanceMemo&) = delete;
    ExactDistanceMemo& operator=(const ExactDistanceMemo&) = delete;

    void require_domain(Domain domain) const {
        if (domain.query_count != domain_.query_count || domain.sample_count != domain_.sample_count ||
            domain.dimensions != domain_.dimensions || domain.scalar_tag != domain_.scalar_tag ||
            domain.base_identity != domain_.base_identity || domain.query_identity != domain_.query_identity)
            throw std::invalid_argument("exact distance memo domain mismatch");
    }
    // Only admitted cached computations/hits are counted. Direct computation is
    // deliberately uninstrumented to keep bypass equivalent to ordinary rerank.
    Stats stats() const {
        Stats result;
        for (const Query& q : queries_) {
            std::lock_guard<std::mutex> lock(q.mutex);
            result.computations += q.computations;
            result.hits += q.hits;
            result.entries += q.table ? q.table->used : 0;
            result.adaptive_bypassed_queries += q.bypass.load(std::memory_order_relaxed);
        }
        return result;
    }

    template<class Compute>
    float evaluate(uint32_t query, uint32_t sample, Compute&& compute) {
        if (query >= domain_.query_count || sample >= domain_.sample_count)
            throw std::out_of_range("exact distance memo key");
        reject_recursion();
        Query& state = queries_[query];
        if (bypassed(query, state)) { ComputeGuard guard; return compute(); }
        std::unique_lock<std::mutex> lock(state.mutex);
        if (state.bypass.load(std::memory_order_relaxed)) {
            lock.unlock(); ComputeGuard guard; return compute();
        }
        bool owned = false;
        Entry* entry = claim(state, sample, owned);
        if (!entry) {
            lock.unlock(); ComputeGuard guard; return compute();
        }
        if (owned) {
            lock.unlock();
            float distance = 0;
            std::exception_ptr error;
            try { ComputeGuard guard; distance = compute(); }
            catch (...) { error = std::current_exception(); }
            lock.lock();
            publish(state, *entry, distance, error);
            maybe_bypass(state);
            state.ready.notify_all();
        } else state.ready.wait(lock, [&] { return entry->ready; });
        if (entry->error) std::rethrow_exception(entry->error);
        return entry->distance;
    }

    template<class ComputeBatch>
    void evaluate_batch(uint32_t query, const uint32_t* ids, size_t count, float* out,
                        ComputeBatch&& compute) {
        if (query >= domain_.query_count) throw std::out_of_range("exact distance memo query");
        if (!count) return;
        for (size_t i = 0; i < count; ++i)
            if (ids[i] >= domain_.sample_count) throw std::out_of_range("exact distance memo sample");
        reject_recursion();
        Query& state = queries_[query];
        if (bypassed(query, state)) { ComputeGuard guard; compute(ids, count, out); return; }
        Scratch& buffers = scratch();
        auto& entries = buffers.entries;
        auto& missing = buffers.missing;
        auto& owned = buffers.owned;
        auto& distances = buffers.distances;
        auto& direct_index = buffers.direct_index;
        // Allocate all scratch before publishing any in-flight work. Once warm,
        // batch callers (including native FP16) reuse this worker-local storage.
        entries.resize(count); direct_index.resize(count); distances.resize(count);
        missing.clear(); owned.clear();
        missing.reserve(count); owned.reserve(count);
        {
            std::unique_lock<std::mutex> lock(state.mutex);
            if (state.bypass.load(std::memory_order_relaxed)) {
                lock.unlock(); ComputeGuard guard; compute(ids, count, out); return;
            }
            // The one table allocation happens before any claims. Remaining
            // insertion and pointer publication are allocation-free.
            if (!state.table) state.table = std::make_unique<Table>();
            for (size_t i = 0; i < count; ++i) {
                bool owner = false;
                entries[i] = claim(state, ids[i], owner);
                if (owner || !entries[i]) {
                    direct_index[i] = missing.size();
                    missing.push_back(ids[i]);
                    owned.push_back(entries[i]); // nullptr is uncached overflow
                }
            }
        }
        std::exception_ptr failure;
        if (!missing.empty()) {
            try { ComputeGuard guard; compute(missing.data(), missing.size(), distances.data()); }
            catch (...) { failure = std::current_exception(); }
            {
                std::lock_guard<std::mutex> lock(state.mutex);
                for (size_t i = 0; i < owned.size(); ++i)
                    if (owned[i]) publish(state, *owned[i], distances[i], failure);
                maybe_bypass(state);
            }
            state.ready.notify_all();
        }
        // Publish every owned score before waiting. Cached pointers never move
        // or expire, including after bypass, so reversed batches cannot deadlock.
        std::unique_lock<std::mutex> lock(state.mutex);
        state.ready.wait(lock, [&] {
            return std::all_of(entries.begin(), entries.end(), [](const Entry* e) { return !e || e->ready; });
        });
        if (failure) std::rethrow_exception(failure);
        for (size_t i = 0; i < count; ++i) {
            if (entries[i] && entries[i]->error) std::rethrow_exception(entries[i]->error);
            out[i] = entries[i] ? entries[i]->distance : distances[direct_index[i]];
        }
    }

private:
    struct Entry {
        uint32_t sample = 0;
        float distance = 0;
        bool ready = false;
        std::exception_ptr error;
    };
    struct Table {
        static constexpr size_t buckets = max_entries_per_query * 2;
        static constexpr uint8_t empty = UINT8_MAX;
        std::array<Entry, max_entries_per_query> entries;
        std::array<uint8_t, buckets> index;
        size_t used = 0;
        Table() { index.fill(empty); }
        Entry* find_or_insert(uint32_t sample, bool& inserted) {
            // Mix both high and low bits before probing the <=50%-full index.
            uint32_t hash = sample * 0x9e3779b9U;
            size_t bucket = (hash ^ (hash >> 16)) & (buckets - 1);
            while (index[bucket] != empty) {
                Entry& entry = entries[index[bucket]];
                if (entry.sample == sample) return &entry;
                bucket = (bucket + 1) & (buckets - 1);
            }
            if (used == max_entries_per_query) return nullptr;
            index[bucket] = static_cast<uint8_t>(used);
            Entry& entry = entries[used++];
            entry.sample = sample;
            inserted = true;
            return &entry;
        }
    };
    struct Query {
        mutable std::mutex mutex;
        std::condition_variable ready;
        std::unique_ptr<Table> table;
        uint64_t computations = 0, hits = 0;
        size_t in_flight = 0;
        bool failed = false;
        std::atomic<bool> bypass{false};
    };
    bool bypassed(uint32_t query, const Query& state) const {
        return (!enabled_.empty() && !enabled_[query]) || state.bypass.load(std::memory_order_acquire);
    }
    static Entry* claim(Query& state, uint32_t sample, bool& owned) {
        if (!state.table) state.table = std::make_unique<Table>();
        Entry* entry = state.table->find_or_insert(sample, owned);
        if (owned) { ++state.computations; ++state.in_flight; }
        else if (entry) ++state.hits;
        return entry;
    }
    static void publish(Query& state, Entry& entry, float distance, std::exception_ptr error) {
        entry.distance = distance;
        entry.error = error;
        entry.ready = true;
        state.failed |= error != nullptr;
        --state.in_flight;
    }
    static void maybe_bypass(Query& state) {
        // At most 64 speculative cached distances. Four hits justify retaining
        // this small cache; the measured 0.33%-reuse workload quickly bypasses.
        // Never bypass an outstanding owner or hide a published cached error.
        if (state.table->used == max_entries_per_query && state.hits < 4 &&
            state.in_flight == 0 && !state.failed)
            state.bypass.store(true, std::memory_order_release);
    }
    struct Scratch {
        std::vector<Entry*> entries, owned;
        std::vector<uint32_t> missing;
        std::vector<size_t> direct_index;
        std::vector<float> distances;
    };
    static Scratch& scratch() { static thread_local Scratch buffers; return buffers; }
    static bool& inside_compute() { static thread_local bool active = false; return active; }
    static void reject_recursion() {
        if (inside_compute()) throw std::logic_error("recursive exact distance computation");
    }
    struct ComputeGuard {
        ComputeGuard() { inside_compute() = true; }
        ~ComputeGuard() { inside_compute() = false; }
    };
    Domain domain_;
    std::vector<Query> queries_;
    std::vector<uint8_t> enabled_;
};

} // namespace go_fann
