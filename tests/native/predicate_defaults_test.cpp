#include "predicates/predicate_defaults.h"
#include <cassert>
#include <functional>
#include <iostream>
using namespace go_fann::predicate_defaults;

int main()
{
    Context c;
    auto run = [&](std::string mode = "auto", std::optional<double> alpha = std::nullopt,
                   std::string storage = "csr", bool diagnostics = false) {
        return resolve(mode, storage, alpha, diagnostics, c);
    };
    auto rejects = [](auto action) {
        bool rejected = false;
        try { action(); } catch (const std::runtime_error&) { rejected = true; }
        assert(rejected);
    };
    for (const auto* backend : {"pq", "rabitq"}) {
        c.backend = backend;
        assert(run().mode == "gpu_exact_bias" && run().alpha == .15);
        assert(run("cpu").mode == "cpu" && run("cpu").alpha == 0);
        assert(run("gpu_exact").alpha == 0);
        assert(run("gpu_exact_bias", 0).alpha == 0);
        for (const auto& disable : std::initializer_list<std::function<void(Context&)>>{
                 [](auto& x) { x.operation = "latency"; },
                 [](auto& x) { x.worker = "finite"; },
                 [](auto& x) { x.transport = "device_resident"; },
                 [](auto& x) { x.graph_cache = true; },
                 [](auto& x) { x.persistent_queue = true; },
                 [](auto& x) { x.cpu_exact_rerank = false; },
                 [](auto& x) { x.full_filter = false; },
                 [](auto& x) { x.per_step_full_filter = false; },
                 [](auto& x) { x.mailbox_profile = true; },
                 [](auto& x) { x.early_stop = "fixed"; }}) {
            const auto saved = c;
            disable(c);
            assert(run().mode == "cpu" && run().alpha == 0 && !run().reason.empty());
            assert(run("cpu").mode == "cpu");
            rejects([&] { run("gpu_exact_bias"); });
            c = saved;
        }
        for (const auto* storage : {"disk", "unsupported"}) {
            c.graph_storage = storage;
            for (const auto* mode : {"auto", "cpu", "gpu_exact", "gpu_exact_bias"})
                rejects([&] { run(mode); });
        }
        c.graph_storage = "memory";
        c.early_stop = "off";
        assert(run().mode == "gpu_exact_bias");
        c.early_stop = "fixed";
        assert(run("gpu_exact").mode == "gpu_exact");
        rejects([&] { run("gpu_exact_bias", 0); }); // same stop policy for alpha-zero control
        c.early_stop = "adaptive";
    }
    // Explicit GPU-inline admission preserves the predicate/penalty, while
    // automatic selection retains the previous CPU-predicate fallback.
    for (const auto* backend : {"pq", "rabitq"}) {
        for (const auto* operation : {"search", "benchmark"}) {
            c = Context{};
            c.backend = backend;
            c.operation = operation;
            c.cpu_exact_rerank = false;
            c.gpu_inline_exact_rerank = true;
            assert(run().mode == "cpu" && run().alpha == 0);
            assert(run("gpu_exact", 0).mode == "gpu_exact");
            rejects([&] { run("cpu_simplified"); });
            for (double alpha : {0., .15, .3}) {
                const auto resolved = run("gpu_exact_bias", alpha);
                assert(resolved.mode == "gpu_exact_bias" && resolved.alpha == alpha);
            }
            for (const auto& disable : std::initializer_list<std::function<void(Context&)>>{
                     [](auto& x) { x.gpu_inline_exact_rerank = false; },
                     [](auto& x) { x.worker = "finite"; },
                     [](auto& x) { x.operation = "latency"; },
                     [](auto& x) { x.transport = "device_resident"; },
                     [](auto& x) { x.graph_cache = true; },
                     [](auto& x) { x.persistent_queue = true; },
                     [](auto& x) { x.full_filter = false; },
                     [](auto& x) { x.per_step_full_filter = false; },
                     [](auto& x) { x.mailbox_profile = true; },
                     [](auto& x) { x.early_stop = "fixed"; }}) {
                const auto saved = c;
                disable(c);
                rejects([&] { run("gpu_exact_bias", .15); });
                c = saved;
            }
            if (c.backend == "pq") {
                rejects([&] { run("gpu_exact", 0, "roaring"); });
                rejects([&] { run("gpu_exact", 0, "csr", true); });
            }
        }
    }
    c = Context{};
    c.backend = "rabitq";
    c.early_stop = "fixed";
    c.early_stop_explicit = false;
    assert(run().mode == "gpu_exact_bias" && run().early_stop == "adaptive");
    assert(run("cpu").early_stop == "fixed");
    c.full_filter = false;
    assert(run().mode == "cpu" && run().early_stop == "fixed");
    c.full_filter = true;
    c.early_stop_explicit = true;
    assert(run().mode == "cpu" && run().early_stop == "fixed");
    c.early_stop = "adaptive";
    rejects([&] { run("cpu", .15); });
    rejects([&] { run("gpu_exact", .15); });
    for (double alpha : {-1., std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::max()})
        rejects([&] { run("auto", alpha); });
    rejects([&] { run("typo"); });
    rejects([&] { run("auto", {}, "typo"); });
    c.backend = "pq";
    assert(run("auto", {}, "roaring").mode == "cpu");
    rejects([&] { run("gpu_exact", {}, "roaring"); });
    assert(run("auto", {}, "csr", true).mode == "cpu");
    rejects([&] { run("cpu_simplified"); });
    c.backend = "rabitq";
    assert(run("cpu_simplified").mode == "cpu_simplified");
    assert(run("auto", {}, "roaring", true).mode == "gpu_exact_bias");
    c.cpu_exact_rerank = false;
    c.gpu_post_exact_rerank = true;
    assert(run("gpu_exact_bias", 0).mode == "gpu_exact_bias");
    assert(run("gpu_exact_bias", .15).alpha == .15);
    c.backend = "pq";
    rejects([&] { run("gpu_exact_bias", 0); });
    c.backend = "rabitq";
    c.gpu_post_exact_rerank = false;
    rejects([&] { run("gpu_exact_bias", 0); });
    c.cpu_exact_rerank = true;
    c.backend = "full_gpu";
    assert(run().mode == "cpu");
    std::cout << "predicate defaults tests passed\n";
}
