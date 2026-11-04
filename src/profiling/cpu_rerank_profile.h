#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <nlohmann/json.hpp>

namespace go_fann::cpu_rerank_profile {

using Clock = std::chrono::steady_clock;
enum class Stage : size_t {
    Setup, Candidates, VectorPrepare, DistanceAccess, Topk, EarlyStop, Writeback,
    ReadyWait, WorkerWait, ThreadLaunch, ThreadJoin, Delegated,
    CandidateCollect, CandidateSortDedup, CandidateRank,
    QueryLockWait, IncrementalAdmit, IncrementalCommit, QueryPrepare, Count
};
inline constexpr std::array<const char*, static_cast<size_t>(Stage::Count)> names{
    "setup", "candidate_filter_dedup", "explicit_vector_prepare",
    "distance_and_vector_access", "topk_select", "early_stop", "writeback",
    "gpu_ready_poll_wait", "worker_permit_wait", "thread_launch", "thread_join",
    "delegated_work", "candidate_collect_filter_dedup", "candidate_sort_dedup",
    "candidate_rank_truncate", "query_state_lock_wait", "incremental_admission",
    "incremental_completion", "rerank_query_prepare"};

enum class Detail : size_t { Predicate, HashLookup, Count };
inline constexpr std::array<const char*, static_cast<size_t>(Detail::Count)> detail_names{
    "predicate_check", "hash_lookup"};

inline uint64_t now_ns()
{
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        Clock::now().time_since_epoch()).count());
}

struct Record {
    const char* path = nullptr;
    const char* role = nullptr;
    uint64_t thread = 0, start_ns = 0, end_ns = 0;
    std::array<uint64_t, names.size()> stage_ns{};
    uint64_t queries = 0, candidates = 0, exact_evaluations = 0, batches = 0;
    std::array<uint64_t, detail_names.size()> detail_calls{}, detail_samples{}, detail_ns{};
};

struct Session {
    std::mutex mutex;
    std::vector<Record> records;
    std::atomic<size_t> active_tasks{0};
    std::atomic<bool> record_failed{false};
};

// Benchmark execution is serialized. All asynchronous workers must join before
// the enclosing iteration is finished; the session outlives every Task.
inline std::atomic<Session*> current_session{nullptr};

class Task {
public:
    explicit Task(const char* path, const char* role = "worker")
        : session_(current_session.load(std::memory_order_acquire))
    {
        if (!session_) return;
        session_->active_tasks.fetch_add(1, std::memory_order_relaxed);
        record_.path = path;
        record_.role = role;
        record_.thread = std::hash<std::thread::id>{}(std::this_thread::get_id());
        record_.start_ns = last_ns_ = now_ns();
    }
    Task(const Task&) = delete;
    Task& operator=(const Task&) = delete;
    ~Task()
    {
        if (!session_) return;
        record_.end_ns = now_ns();
        record_.stage_ns[static_cast<size_t>(stage_)] += record_.end_ns - last_ns_;
        try {
            std::lock_guard<std::mutex> lock(session_->mutex);
            session_->records.push_back(record_);
        } catch (...) {
            session_->record_failed.store(true, std::memory_order_relaxed);
        }
        session_->active_tasks.fetch_sub(1, std::memory_order_release);
    }
    bool enabled() const { return session_ != nullptr; }
    void stage(Stage next)
    {
        if (!session_) return;
        const uint64_t at = now_ns();
        record_.stage_ns[static_cast<size_t>(stage_)] += at - last_ns_;
        stage_ = next;
        last_ns_ = at;
    }
    void query() { if (session_) ++record_.queries; }
    void candidates(size_t n) { if (session_) record_.candidates += n; }

    // Nested samples are not additional stages. Keep predicate and hash work
    // interleaved exactly as before; sample 1/64 calls without changing outputs.
    template <typename Operation>
    decltype(auto) sample(Detail detail, Operation&& operation)
    {
        if (!session_) return operation();
        const size_t index = static_cast<size_t>(detail);
        if ((record_.detail_calls[index]++ & 63U) != 0) return operation();
        const uint64_t begin = now_ns();
        decltype(auto) result = operation();
        record_.detail_ns[index] += now_ns() - begin;
        ++record_.detail_samples[index];
        return result;
    }

    // No per-candidate clocks. Only the diagnostic path separates a batch's
    // independent distance evaluations from its top-k updates. The normal
    // path retains the original interleaved order without allocating scratch.
    template <typename Evaluate, typename Select>
    void evaluate_batch(const std::vector<uint32_t>& ids, Evaluate&& evaluate, Select&& select)
    {
        if (!session_) {
            for (uint32_t id : ids) select(id, evaluate(id));
            return;
        }
        stage(Stage::Setup);
        distances_.resize(ids.size());
        ++record_.batches;
        record_.exact_evaluations += ids.size();
        stage(Stage::DistanceAccess);
        for (size_t i = 0; i < ids.size(); ++i) distances_[i] = evaluate(ids[i]);
        stage(Stage::Topk);
        for (size_t i = 0; i < ids.size(); ++i) select(ids[i], distances_[i]);
    }

    // Incremental scorers commit under a separate query lock. Keep distances
    // in caller-owned reusable scratch; profiling never changes tile boundaries.
    template <typename EvaluateBatch>
    void evaluate_into(const std::vector<uint32_t>& ids, std::vector<float>& scores,
                       EvaluateBatch&& evaluate) {
        stage(Stage::VectorPrepare);
        scores.resize(ids.size());
        if (session_) { ++record_.batches; record_.exact_evaluations += ids.size(); }
        stage(Stage::DistanceAccess);
        for (size_t begin = 0; begin < ids.size(); begin += 64)
            evaluate(ids.data() + begin, std::min<size_t>(64, ids.size() - begin), scores.data() + begin);
    }

    // Native batch kernels are also used without profiling. Bounded stack
    // scratch avoids per-query allocations and does not cap the candidate set.
    template <typename EvaluateBatch, typename Select>
    void evaluate_native_batch(const std::vector<uint32_t>& ids,
                               EvaluateBatch&& evaluate, Select&& select)
    {
        if (!session_) {
            std::array<float, 64> scores;
            for (size_t begin = 0; begin < ids.size(); begin += scores.size()) {
                const size_t count = std::min(scores.size(), ids.size() - begin);
                evaluate(ids.data() + begin, count, scores.data());
                for (size_t i = 0; i < count; ++i) select(ids[begin + i], scores[i]);
            }
            return;
        }
        stage(Stage::Setup);
        distances_.resize(ids.size());
        ++record_.batches;
        record_.exact_evaluations += ids.size();
        stage(Stage::DistanceAccess);
        evaluate(ids.data(), ids.size(), distances_.data());
        stage(Stage::Topk);
        for (size_t i = 0; i < ids.size(); ++i) select(ids[i], distances_[i]);
    }

private:
    Session* session_;
    Record record_;
    Stage stage_ = Stage::Setup;
    uint64_t last_ns_ = 0;
    std::vector<float> distances_;
};

class Reporter {
public:
    Reporter()
    {
        const char* value = std::getenv("GO_FANN_CPU_RERANK_PROFILE");
        if (!value || !*value) return;
        path_ = value;
        stream_.open(path_);
        if (!stream_) throw std::runtime_error("cannot open CPU rerank profile " + path_);
    }
    explicit Reporter(const std::string& path) : path_(path), stream_(path)
    {
        if (!stream_) throw std::runtime_error("cannot open CPU rerank profile " + path_);
    }
    bool enabled() const { return stream_.is_open(); }
    void write(nlohmann::json row)
    {
        row["schema_version"] = 1;
        stream_ << row.dump() << '\n';
        stream_.flush();
        if (!stream_) throw std::runtime_error("cannot write CPU rerank profile " + path_);
    }
private:
    std::string path_;
    std::ofstream stream_;
};

class Iteration {
public:
    Iteration(Reporter& reporter, const char* phase, int queue, int run,
              std::string point = "")
        : reporter_(reporter), phase_(phase), queue_(queue), run_(run), point_(std::move(point))
    {
        if (!reporter_.enabled()) return;
        Session* expected = nullptr;
        if (!current_session.compare_exchange_strong(expected, &session_))
            throw std::logic_error("CPU rerank profiling iterations must not overlap");
        start_ns_ = now_ns();
        active_ = true;
    }
    Iteration(const Iteration&) = delete;
    Iteration& operator=(const Iteration&) = delete;
    ~Iteration() { if (active_) current_session.store(nullptr, std::memory_order_release); }

    void finish()
    {
        if (!active_) return;
        const uint64_t end = now_ns();
        if (session_.active_tasks.load(std::memory_order_acquire) != 0)
            throw std::logic_error("CPU rerank profile finished before worker completion");
        current_session.store(nullptr, std::memory_order_release);
        active_ = false;
        if (session_.record_failed.load())
            throw std::runtime_error("CPU rerank profile could not retain all task records");
        std::array<uint64_t, names.size()> worker_ns{};
        uint64_t evaluations = 0;
        for (const Record& r : session_.records) {
            nlohmann::json stages = nlohmann::json::object();
            for (size_t i = 0; i < names.size(); ++i) {
                stages[names[i]] = r.stage_ns[i] / 1e6;
                if (std::string(r.role) == "worker") worker_ns[i] += r.stage_ns[i];
            }
            evaluations += r.exact_evaluations;
            nlohmann::json details = nlohmann::json::object();
            for (size_t i = 0; i < detail_names.size(); ++i) {
                details[detail_names[i]] = {{"calls", r.detail_calls[i]},
                    {"samples", r.detail_samples[i]}, {"sample_ns", r.detail_ns[i]},
                    {"sample_stride", 64}};
            }
            reporter_.write({{"record_type", "task"}, {"phase", phase_},
                {"queue", queue_}, {"run", run_}, {"point_id", point_},
                {"path", r.path}, {"role", r.role},
                {"thread", r.thread}, {"start_ns", r.start_ns}, {"end_ns", r.end_ns},
                {"wall_ms", (r.end_ns - r.start_ns) / 1e6}, {"stage_ms", stages},
                {"queries", r.queries}, {"candidate_slots", r.candidates},
                {"exact_evaluations", r.exact_evaluations}, {"batches", r.batches},
                {"nested_candidate_samples", details}});
        }
        nlohmann::json sums = nlohmann::json::object();
        for (size_t i = 0; i < names.size(); ++i) sums[names[i]] = worker_ns[i] / 1e6;
        reporter_.write({{"record_type", "iteration"}, {"phase", phase_},
            {"queue", queue_}, {"run", run_}, {"point_id", point_},
            {"start_ns", start_ns_}, {"end_ns", end},
            {"task_count", session_.records.size()}, {"worker_stage_sum_ms", sums},
            {"exact_evaluations", evaluations},
            {"interpretation", "worker sums are not additive batch wall times; mmap access is included in distance; diagnostic batch scoring precedes top-k"}});
    }
private:
    Reporter& reporter_;
    const char* phase_;
    int queue_, run_;
    std::string point_;
    Session session_;
    uint64_t start_ns_ = 0;
    bool active_ = false;
};

} // namespace go_fann::cpu_rerank_profile
