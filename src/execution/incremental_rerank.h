#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace go_fann::incremental_rerank {

// ID membership only: no distance cache, no deletion, one lifetime per iteration.
class IdSet {
    static constexpr uint64_t empty = UINT64_MAX, completed = uint64_t(1) << 32;
    std::vector<uint64_t> slots_;
    size_t size_ = 0;
    size_t position(uint32_t id) const {
        size_t pos = (uint64_t(id) * 11400714819323198485ull) & (slots_.size() - 1);
        while (slots_[pos] != empty && uint32_t(slots_[pos]) != id)
            pos = (pos + 1) & (slots_.size() - 1);
        return pos;
    }
public:
    bool contains(uint32_t id) const {
        return !slots_.empty() && uint32_t(slots_[position(id)]) == id;
    }
    bool insert(uint32_t id) {
        if (id == UINT32_MAX) throw std::invalid_argument("incremental rerank sentinel ID");
        size_t pos = slots_.empty() ? 0 : position(id);
        if (!slots_.empty() && slots_[pos] != empty) return false;
        if (slots_.empty() || (size_ + 1) * 2 > slots_.size()) {
            auto old = std::move(slots_);
            slots_.assign(std::max<size_t>(16, old.size() * 2), empty);
            for (uint64_t value : old)
                if (value != empty) slots_[position(uint32_t(value))] = value;
            pos = position(id);
        }
        slots_[pos] = id;
        ++size_;
        return true;
    }
    bool mark_complete(uint32_t id) {
        if (slots_.empty()) return false;
        auto& slot = slots_[position(id)];
        if (slot == empty || (slot & completed)) return false;
        slot |= completed;
        return true;
    }
    size_t size() const { return size_; }
    void clear() { std::fill(slots_.begin(), slots_.end(), empty); size_ = 0; }
};

using ScoredId = std::pair<float, uint32_t>;
struct Stats {
    uint64_t batches = 0, offered = 0, repeated = 0, issued = 0, completed = 0;
    uint64_t coalesced = 0, score_batches = 0;
    Stats& operator+=(const Stats& o) {
        batches += o.batches; offered += o.offered; repeated += o.repeated;
        issued += o.issued; completed += o.completed;
        coalesced += o.coalesced; score_batches += o.score_batches; return *this;
    }
};

// Caller owns this query throughout offer/issue/complete. Exact top-k never
// depends on membership in approximate_: an evicted in-flight ID may still win.
class QueryState {
    std::vector<ScoredId> approximate_, merged_, exact_;
    IdSet merge_ids_, dispatched_;
    bool scoring_ = false;
    bool needs_issue_ = false;
    Stats stats_;
public:
    void offer(std::vector<ScoredId>& incoming, size_t cap) {
        if (!cap) throw std::invalid_argument("incremental shortlist must be positive");
        incoming.erase(std::remove_if(incoming.begin(), incoming.end(), [](const auto& c) {
            return c.second == UINT32_MAX || !std::isfinite(c.first);
        }), incoming.end());
        if (!std::is_sorted(incoming.begin(), incoming.end()))
            std::sort(incoming.begin(), incoming.end());
        offer_sorted(incoming, cap, false);
    }
    // Native route merging already emits finite, sorted, ID-unique candidates.
    // Keep the general offer() path for callers without that stronger contract.
    void offer_sorted_unique(const std::vector<ScoredId>& incoming, size_t cap) {
        if (!cap) throw std::invalid_argument("incremental shortlist must be positive");
        offer_sorted(incoming, cap, true);
    }
private:
    void offer_sorted(const std::vector<ScoredId>& incoming, size_t cap, bool unique) {
        ++stats_.batches;
        stats_.offered += incoming.size();
        for (const auto& c : incoming)
            if (dispatched_.contains(c.second)) ++stats_.repeated;
        // A full shortlist's threshold can only improve. Equal/worse arrivals
        // cannot change its ID union, including an already-dispatched duplicate.
        if (incoming.empty() && approximate_.size() <= cap) return;
        if (!incoming.empty() && approximate_.size() == cap &&
            !(incoming.front() < approximate_.back())) return;
        needs_issue_ = true;
        if (unique && approximate_.empty()) {
            approximate_.assign(incoming.begin(), incoming.begin() + std::min(cap, incoming.size()));
            return;
        }
        merge_ids_.clear();
        merged_.clear();
        merged_.reserve(cap);
        size_t a = 0, b = 0;
        while (merged_.size() < cap && (a < approximate_.size() || b < incoming.size())) {
            const auto candidate = b == incoming.size() ||
                (a < approximate_.size() && approximate_[a] < incoming[b])
                ? approximate_[a++] : incoming[b++];
            if (merge_ids_.insert(candidate.second)) merged_.push_back(candidate);
        }
        approximate_.swap(merged_);
    }
public:
    void issue(std::vector<uint32_t>& ids) {
        ids.clear();
        if (!needs_issue_) return;
        needs_issue_ = false;
        for (const auto& c : approximate_) {
            if (dispatched_.insert(c.second)) {
                ids.push_back(c.second);
                ++stats_.issued;
            }
        }
    }
    void complete(uint32_t id, float distance, size_t topk) {
        if (!topk || !std::isfinite(distance) || !dispatched_.mark_complete(id))
            throw std::logic_error("invalid or duplicate incremental exact completion");
        ++stats_.completed;
        const ScoredId result{distance, id};
        if (exact_.size() == topk && !(result < exact_.back())) return;
        const auto where = std::lower_bound(exact_.begin(), exact_.end(), result);
        if (size_t(where - exact_.begin()) < topk) {
            exact_.insert(where, result);
            if (exact_.size() > topk) exact_.pop_back();
        }
        // One ID-only table retains dispatch/completion state, never distances.
    }
    // Called under the query lock. A busy scorer owns all exact work for this
    // query; new arrivals update approximate_ and are coalesced until its return.
    bool acquire(std::vector<uint32_t>& ids) {
        ids.clear();
        if (scoring_) { ++stats_.coalesced; return false; }
        issue(ids);
        scoring_ = !ids.empty();
        if (scoring_) ++stats_.score_batches;
        return scoring_;
    }
    bool next(std::vector<uint32_t>& ids) {
        if (!scoring_) throw std::logic_error("incremental scorer has no ownership");
        if (stats_.issued != stats_.completed)
            throw std::logic_error("incremental batch has unfinished scoring");
        issue(ids);
        scoring_ = !ids.empty();
        if (scoring_) ++stats_.score_batches;
        return scoring_;
    }
    bool scoring() const { return scoring_; }
    const std::vector<ScoredId>& exact() const { return exact_; }
    const std::vector<ScoredId>& approximate() const { return approximate_; }
    const Stats& stats() const { return stats_; }
};

// The query lock protects admission and completion only. The current scorer
// owns a batch across unlocked vector access; other producers can merge arrivals.
class State {
    struct Entry { std::mutex mutex; QueryState query; };
    size_t count_;
    std::unique_ptr<Entry[]> entries_;
public:
    const size_t cap, topk;
    State(size_t count, size_t shortlist, size_t k)
        : count_(count), entries_(std::make_unique<Entry[]>(count)), cap(shortlist), topk(k) {
        if (!k || shortlist < k) throw std::invalid_argument("incremental shortlist below top-k");
    }
    template <typename Work> void with_query(size_t id, Work&& work) {
        if (id >= count_) throw std::out_of_range("incremental original query ID");
        auto& entry = entries_[id];
        std::lock_guard<std::mutex> lock(entry.mutex);
        work(entry.query);
    }
    // Call only after every producer/consumer has joined.
    template <typename Id> void publish(std::vector<std::vector<Id>>& ids,
                                       std::vector<std::vector<float>>& distances) const {
        ids.assign(count_, std::vector<Id>(topk, UINT32_MAX));
        distances.assign(count_, std::vector<float>(topk, std::numeric_limits<float>::max()));
        for (size_t q = 0; q < count_; ++q) {
            const auto& state = entries_[q].query;
            if (state.scoring() || state.stats().issued != state.stats().completed)
                throw std::logic_error("incremental rerank published with pending exact work");
            for (size_t rank = 0; rank < state.exact().size(); ++rank) {
                ids[q][rank] = state.exact()[rank].second;
                distances[q][rank] = state.exact()[rank].first;
            }
        }
    }
    Stats stats() const {
        Stats sum;
        for (size_t q = 0; q < count_; ++q) sum += entries_[q].query.stats();
        return sum;
    }
};
} // namespace go_fann::incremental_rerank
