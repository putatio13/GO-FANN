#pragma once

#include <exception>
#include <future>
#include <stdexcept>
#include <string_view>

namespace go_fann::cpu_rerank_schedule {

enum class Mode { Native, NoOverlap, BatchOverlap, Streaming };

inline Mode parse(std::string_view value)
{
    if (value == "native") return Mode::Native;
    if (value == "no_overlap") return Mode::NoOverlap;
    if (value == "batch_overlap") return Mode::BatchOverlap;
    if (value == "streaming") return Mode::Streaming;
    throw std::invalid_argument(
        "cpu_rerank_schedule must be native, no_overlap, batch_overlap or streaming");
}

inline bool no_overlap(Mode mode) { return mode == Mode::NoOverlap; }

inline std::launch host_finalize_launch(Mode mode)
{
    return no_overlap(mode) ? std::launch::deferred : std::launch::async;
}

inline void apply(Mode mode, std::string_view backend, std::string_view operation,
                  std::string_view worker, std::string_view transport,
                  std::string_view rerank_mode, std::string_view algorithm,
                  bool& graph_streaming, bool& posting_streaming)
{
    if (mode == Mode::Native) return;
    if ((backend != "pq" && backend != "rabitq") ||
        (operation != "search" && operation != "benchmark") ||
        worker != "coupled" || transport != "mailbox" ||
        rerank_mode != "cpu" || algorithm != "exact") {
        throw std::invalid_argument(
            "cpu_rerank_schedule requires native PQ/RaBitQ search or benchmark "
            "with coupled mailbox workers and CPU exact rerank");
    }
    graph_streaming = posting_streaming = mode == Mode::Streaming;
}

// The primary provider publishes GPU completion, then waits here before CPU
// finalization. The secondary provider releases it after its own GPU phase.
class GpuPhaseGate {
public:
    GpuPhaseGate() : ready_(complete_.get_future().share()) {}
    void wait() const { ready_.get(); }
    void release() { complete_.set_value(); }
    void fail(std::exception_ptr error)
    {
        // A later CPU failure may arrive after successful GPU completion.
        try { complete_.set_exception(error); }
        catch (const std::future_error&) {}
    }

private:
    std::promise<void> complete_;
    std::shared_future<void> ready_;
};

}  // namespace go_fann::cpu_rerank_schedule
