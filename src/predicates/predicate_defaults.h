#pragma once

#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

namespace go_fann::predicate_defaults {

// Low-level backend options take concrete modes. The application resolves this
// policy once, after config and CLI overrides, and records the effective mode.
struct Context {
    std::string backend = "pq";
    std::string operation = "search";
    std::string worker = "coupled";
    std::string transport = "mailbox";
    std::string graph_storage = "memory";
    std::string early_stop = "adaptive";
    bool early_stop_explicit = true;
    bool cpu_exact_rerank = true;
    bool gpu_inline_exact_rerank = false;
    bool gpu_post_exact_rerank = false;
    bool full_filter = true;
    bool per_step_full_filter = true;
    bool graph_cache = false;
    bool persistent_queue = false;
    bool mailbox_profile = false;
};

struct Resolution {
    std::string mode;
    double alpha;
    std::string reason;
    std::string early_stop;
};

inline std::string unsupported_reason(const Context& c, const std::string& mode,
                                      const std::string& storage, bool diagnostics)
{
    if (c.backend != "pq" && c.backend != "rabitq") return "requires PQ or RabitQ backend";
    if (c.operation != "search" && c.operation != "benchmark") return "requires search or benchmark";
    if (c.worker != "coupled" || c.persistent_queue) return "requires coupled workers without persistent queue";
    if (c.transport != "mailbox") return "requires mailbox graph transport";
    if (c.graph_storage != "memory") return "requires memory graph";
    if (c.graph_cache) return "GPU graph row cache is unsupported";
    if (c.mailbox_profile) return "mailbox RTT instrumentation is unsupported";
    const bool inline_predicate = c.gpu_inline_exact_rerank &&
        (mode == "gpu_exact" || mode == "gpu_exact_bias");
    if (!c.cpu_exact_rerank && !inline_predicate &&
        !(c.backend == "rabitq" && c.gpu_post_exact_rerank))
        return "requires CPU, GPU-inline or RabitQ GPU-post exact high-frequency rerank";
    if (!c.full_filter || !c.per_step_full_filter) return "requires full-filter and per-step-full-filter";
    if (c.backend == "pq" && mode == "cpu_simplified") return "cpu_simplified requires RabitQ";
    if (c.backend == "pq" && storage != "csr") return "PQ GPU predicate requires CSR storage";
    if (c.backend == "pq" && diagnostics) return "PQ predicate diagnostics are unsupported";
    if (mode == "gpu_exact_bias" && c.early_stop != "off" && c.early_stop != "adaptive")
        return "bias requires search-early-stop-mode=off or adaptive (fixed mode is preserved)";
    return {};
}

inline Resolution resolve(const std::string& requested, const std::string& storage,
                          std::optional<double> specified_alpha, bool diagnostics,
                          const Context& context)
{
    if (requested != "auto" && requested != "cpu" && requested != "cpu_simplified" &&
        requested != "gpu_exact" && requested != "gpu_exact_bias")
        throw std::runtime_error("rabitq-predicate-mode must be one of: auto, cpu, cpu_simplified, gpu_exact, gpu_exact_bias");
    if (storage != "csr" && storage != "roaring")
        throw std::runtime_error("rabitq-predicate-storage must be one of: roaring, csr");
    if (specified_alpha && (!std::isfinite(*specified_alpha) || *specified_alpha < 0 ||
                            *specified_alpha > std::numeric_limits<float>::max()))
        throw std::runtime_error("rabitq-predicate-bias-alpha must be finite and >= 0 and fit float");
    if (context.graph_storage != "memory")
        throw std::runtime_error("graph storage must be memory");
    const std::string mode = requested == "auto" ? "gpu_exact_bias" : requested;
    const double alpha = specified_alpha.value_or(mode == "gpu_exact_bias" ? 0.15 : 0.0);
    if (alpha > 0 && mode != "gpu_exact_bias")
        throw std::runtime_error("positive rabitq-predicate-bias-alpha requires rabitq-predicate-mode=gpu_exact_bias");
    if (mode == "cpu") return {mode, alpha, "explicit CPU predicate", context.early_stop};
    Context effective = context;
    // Keep automatic policy selection stable; GPU-inline predicate admission is explicit.
    if (requested == "auto") effective.gpu_inline_exact_rerank = false;
    if (!context.early_stop_explicit && mode == "gpu_exact_bias") effective.early_stop = "adaptive";
    const auto reason = unsupported_reason(effective, mode, storage, diagnostics);
    if (!reason.empty()) {
        if (requested == "auto") return {"cpu", 0.0, reason, context.early_stop};
        throw std::runtime_error("predicate optimization unsupported: " + reason);
    }
    return {mode, alpha, requested == "auto" ? "supported native mailbox path" : "explicit predicate mode", effective.early_stop};
}

}  // namespace go_fann::predicate_defaults
