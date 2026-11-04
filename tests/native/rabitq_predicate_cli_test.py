"""Exercise actual CLI/config validation without loading data or initializing CUDA.

Usage: python3 tests/native/rabitq_predicate_cli_test.py /absolute/path/to/go-fann
Every case must fail during option validation; valid predicate cases use an
invalid queue size as the deliberate terminal validation sentinel.
"""

import json
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    binary = str(Path(sys.argv[1]).resolve())
    with tempfile.TemporaryDirectory(prefix="rabitq-predicate-cli-") as directory:
        config_path = Path(directory) / "input.json"

        def check(args, expected, config=None):
            config_path.write_text(json.dumps({
                "graph_storage_mode": "memory", "full_filter": True,
                "highfreq_rerank_mode": "cpu", "highfreq_rerank_algorithm": "exact",
                "search_early_stop_mode": "adaptive", "lowfreq_full_filter_mode": "post",
                **(config or {})}))
            result = subprocess.run(
                [binary, f"--config={config_path}", "--backend=rabitq", "--mode=search",
                 "--search-queue-size=0", *args],
                cwd=directory, capture_output=True, text=True, timeout=20,
            )
            output = result.stdout + result.stderr
            expected_messages = (expected,) if isinstance(expected, str) else expected
            assert result.returncode != 0 and all(message in output for message in expected_messages), (args, output)

        sentinel = "search-queue-size"
        # Exercise real application capability selection, then require the later
        # queue validation sentinel. No case initializes CUDA or reads vectors.
        for backend in ("pq", "rabitq"):
            for operation in ("search", "benchmark"):
                inline = [f"--backend={backend}", f"--mode={operation}",
                          "--highfreq-rerank-mode=gpu", "--cpu-rerank-schedule=native",
                          "--highfreq-rerank-streaming=0"]
                for alpha in ("0", "0.15", "0.3"):
                    check([*inline, "--predicate-mode=gpu_exact_bias",
                           f"--predicate-bias-alpha={alpha}"],
                          (sentinel, f"effective=gpu_exact_bias storage=csr alpha={alpha}"))
                check([*inline, "--predicate-mode=gpu_exact", "--predicate-bias-alpha=0"],
                      (sentinel, "effective=gpu_exact storage=csr alpha=0"))
                check(inline, (sentinel, "effective=cpu storage=csr alpha=0"))
                check([*inline, "--predicate-mode=cpu_simplified"], "exact high-frequency rerank")
                check([*inline, "--predicate-mode=gpu_exact_bias", "--predicate-bias-alpha=0.15"],
                      (sentinel, "effective=gpu_exact_bias storage=csr alpha=0.15"),
                      {"rabitq": {"predicate_mode": "cpu", "predicate_bias_alpha": 0}})
                check(inline, (sentinel, "effective=gpu_exact_bias storage=csr alpha=0.3"),
                      {"rabitq_predicate_mode": "cpu", "rabitq": {
                          "predicate_mode": "gpu_exact_bias", "predicate_bias_alpha": 0.3}})
                for extra, reason in (
                    (["--highfreq-rerank-algorithm=heuristic"], "exact high-frequency rerank"),
                    (["--full-filter=0"], "full-filter and per-step-full-filter"),
                    (["--per-step-full-filter=0"], "full-filter and per-step-full-filter"),
                    ([f"--{backend}-query-worker-mode=finite"], "coupled workers"),
                    ([f"--{backend}-graph-transport=device_resident"], "mailbox graph transport"),
                    (["--search-early-stop-mode=fixed"], "bias requires"),
                    (["--highfreq-rerank-streaming=1"], "require --highfreq-rerank-mode=cpu"),
                    (["--cpu-rerank-schedule=no_overlap"], "CPU exact rerank"),
                ):
                    check([*inline, "--predicate-mode=gpu_exact_bias", *extra], reason)
            check([f"--backend={backend}", "--highfreq-rerank-mode=gpu",
                   "--predicate-mode=gpu_exact_bias", "--rerank-vector-storage-mode=mmap_fp16"],
                  "custom rerank vector storage")
        check(["--backend=pq", "--highfreq-rerank-mode=gpu_post", "--predicate-mode=cpu"],
              "gpu_post currently requires --backend=rabitq")
        check(["--backend=rabitq", "--highfreq-rerank-mode=gpu_post",
               "--predicate-mode=gpu_exact_bias", "--predicate-bias-alpha=0.15"],
              (sentinel, "effective=gpu_exact_bias storage=csr alpha=0.15"))
        check([], (sentinel, "[rerank-distance-reuse] enabled=0"))
        check(["--rerank-distance-reuse=1"], (sentinel, "[rerank-distance-reuse] enabled=1"))
        check(["--rerank_distance_reuse=0"], (sentinel, "[rerank-distance-reuse] enabled=0"),
              {"rerank_distance_reuse": True})
        check([], (sentinel, "[rerank-distance-reuse] enabled=1"), {"rerank_distance_reuse": True})
        check(["--rerank-distance-reuse=bad"], "invalid rerank-distance-reuse")
        check([], "invalid rerank_distance_reuse", {"rerank_distance_reuse": "false"})
        # Missing/auto PQ policy selects resident only after predicate resolution.
        # Require the terminal sentinel as well as the policy log, so an earlier
        # validation error cannot accidentally count as a supported default.
        for operation in ("search", "benchmark"):
            check(["--backend=pq", f"--mode={operation}"],
                  ("[pq-posting-policy] requested=auto effective=resident", sentinel))
        check(["--backend=pq", "--pq-lowfreq-scan-mode=host"],
              ("requested=host effective=host", sentinel))
        check(["--backend=pq", "--pq-lowfreq-scan-mode=auto"],
              ("requested=auto effective=resident", sentinel),
              {"pq_lowfreq_scan_mode": "host"})
        check(["--backend=pq", "--pq_lowfreq_scan_mode=auto"],
              ("requested=auto effective=resident", sentinel),
              {"pq": {"lowfreq_scan_mode": "host"}})
        check(["--backend=pq"], ("requested=auto effective=resident", sentinel),
              {"pq_lowfreq_scan_mode": "host", "pq": {"lowfreq_scan_mode": "auto"}})
        for args in (
            ["--predicate-mode=cpu"],
            ["--pq-query-worker-mode=finite"],
            ["--pq-graph-transport=device_resident", "--highfreq-rerank-mode=gpu"],
            ["--lowfreq-full-filter-mode=pre"],
            ["--search-early-stop-mode=fixed"],
        ):
            check(["--backend=pq", *args], ("requested=auto effective=host", sentinel))
        for args, reason in (
            (["--lowfreq-full-filter-mode=pre"], "host_prefilter"),
        ):
            check(["--backend=pq", "--pq-lowfreq-scan-mode=resident", *args],
                  ("resident/score-first posting requires", reason))
        check(["--backend=pq", "--benchmark-cache-warmup=1"], "unsupported retired runtime option")
        check(["--backend=pq", "--graph-storage-mode=disk"], "only memory graph storage is supported")
        check([], ("requested=auto effective=host reason=not_pq", sentinel))
        # This fixture requests CPU reranking, which full_gpu rejects before
        # the queue sentinel; posting auto must still resolve harmlessly.
        check(["--backend=full_gpu"],
              ("requested=auto effective=host reason=not_pq", "supported by --backend=pq|rabitq only"))
        check(["--backend=pq", "--pq-posting-streaming=1"],
              ("posting_streaming=1", sentinel))
        check(["--backend=pq"], ("posting_streaming=1", sentinel),
              {"pq_posting_streaming": False, "pq": {"posting_streaming": True}})
        check(["--backend=pq", "--pq-posting-streaming=0"],
              ("posting_streaming=0", sentinel), {"pq": {"posting_streaming": True}})
        for args in (["--backend=rabitq"],
                     ["--backend=pq", "--pq-lowfreq-scan-mode=host"],
                     ["--backend=pq", "--posting-predicate-order=score_first"],
                     ["--backend=pq", "--mode=latency"]):
            check([*args, "--pq-posting-streaming=1"], "pq-posting-streaming requires")
        check(["--backend=pq", "--pq-posting-streaming=bad"], "invalid pq-posting-streaming")
        for size in ("0", "1048577", "4294967297"):
            check(["--backend=pq", "--pq-posting-stream-batch-size=" + size],
                  "pq-posting-stream-batch-size must be")
        check(["--pq-lowfreq-scan-mode=invalid"], "pq-lowfreq-scan-mode must be")
        check(["--posting-predicate-order=invalid"], "posting-predicate-order must be")
        check(["--pq-lowfreq-scan-mode=resident"], "requires backend=pq")
        check(["--backend=pq", "--pq-lowfreq-scan-mode=resident"], "pq_lowfreq_scan_mode=resident")
        check(["--backend=pq", "--pq-lowfreq-scan-mode=resident", "--posting-predicate-order=score_first"],
              "predicate_order=score_first")
        check(["--posting_predicate_order=score_first"], "predicate_order=score_first")
        check(["--posting-predicate-order=filter_first"], "predicate_order=filter_first",
              {"posting_predicate_order":"score_first"})
        check(["--backend=pq", "--pq-lowfreq-scan-mode=host"], "pq_lowfreq_scan_mode=host",
              {"pq":{"lowfreq_scan_mode":"resident"}})
        check(["--backend=pq"], "pq_lowfreq_scan_mode=resident",
              {"pq":{"lowfreq_scan_mode":"resident"}, "posting_predicate_order":"score_first"})
        check(["--posting-predicate-order=score_first", "--predicate-mode=cpu"],
              "resident/score-first posting requires")
        check(["--backend=pq", "--pq-lowfreq-scan-mode=resident", "--pq-query-worker-mode=finite"],
              "resident/score-first posting requires")
        check(["--posting-predicate-order=score_first", "--predicate-storage=roaring"],
              "resident/score-first posting requires")
        check([], "invalid posting_predicate_order", {"posting_predicate_order":True})
        check([], "legacy_bonus=disabled")
        check(["--predicate-mode=cpu"], "score_stop=raw_approx")
        check(["--backend=pq", "--mode=benchmark", "--benchmark-quick-filter-bonuses=0.9,0.95"],
              "multiple values would repeat identical scoring")
        check(["--mode=benchmark", "--benchmark-quick-filter-bonuses=0.95,0.95"],
              "multiple values would repeat identical scoring")
        check(["--mode=benchmark", "--benchmark-quick-filter-bonuses=0.95"],
              "benchmark_bonus_axis=0.95")
        check(["--mode=benchmark", "--rabitq-query-worker-mode=finite",
               "--benchmark-quick-filter-bonuses=0.9,0.95"], sentinel)
        check(["--predicate-mode=auto"], "effective=gpu_exact_bias storage=csr alpha=0.15")
        check(["--predicate-mode=cpu"], "effective=cpu storage=csr alpha=0")
        check(["--predicate_mode=gpu_exact", "--predicate_storage=csr"], sentinel)
        check(["--predicate-bias-alpha=0"], "effective=gpu_exact_bias storage=csr alpha=0")
        check(["--predicate-mode=cpu", "--predicate-bias-alpha=0.15"], "positive rabitq-predicate-bias-alpha")
        check(["--predicate-mode=auto", "--search-early-stop-mode=fixed"], "effective=cpu")
        check(["--predicate-mode=gpu_exact", "--no-full-filter"], "requires full-filter")
        check(["--predicate-mode=auto", "--backend=full_gpu"], "effective=cpu")
        check(["--predicate-mode=gpu_exact", "--backend=pq", "--predicate-storage=roaring"], "requires CSR")
        check(["--predicate-mode=gpu_exact", "--backend=pq", "--predicate-diagnostics=1"], "diagnostics are unsupported")
        check(["--predicate-mode=cpu"], "effective=cpu", {"predicate_mode":"gpu_exact_bias"})

        check(["--rabitq-predicate-storage=bad"], "rabitq-predicate-storage must be")
        check(["--rabitq-predicate-mode=gpu_exact", "--rabitq-predicate-storage=csr"], sentinel)
        check(["--rabitq_predicate_storage=csr"], sentinel)
        check([], sentinel, {"rabitq_predicate_storage": "bad", "rabitq": {"predicate_storage": "csr"}})
        check(["--rabitq-predicate-storage=roaring"], sentinel, {"rabitq": {"predicate_storage": "bad"}})
        check([], "invalid rabitq_predicate_storage", {"rabitq_predicate_storage": True})
        check(["--rabitq-predicate-mode=typo"], "rabitq-predicate-mode must be")
        for value in ("-0.1", "nan", "inf", "-inf"):
            check([f"--rabitq-predicate-bias-alpha={value}"], "must be finite and >= 0")
        check(["--rabitq-predicate-bias-alpha=0.1garbage"], "invalid rabitq-predicate-bias-alpha")
        check(["--rabitq-predicate-diagnostics=maybe"], "invalid rabitq-predicate-diagnostics")
        check(["--rabitq-predicate-mode=gpu_exact"], sentinel)
        check(["--rabitq-predicate-mode=cpu_simplified"], sentinel)
        check(["--rabitq-predicate-mode=gpu_exact", "--backend=pq"], sentinel)
        check(["--rabitq-predicate-mode=cpu_simplified", "--backend=pq"], "requires RabitQ")
        check(["--rabitq-predicate-bias-alpha=0.15"], sentinel)
        check(["--rabitq-predicate-mode=gpu_exact_bias", "--rabitq-predicate-bias-alpha=0.15",
               "--search-early-stop-mode=fixed"],
              "bias requires search-early-stop-mode")
        check(["--rabitq-predicate-mode=gpu_exact_bias", "--rabitq-predicate-bias-alpha=0.15",
               "--search-early-stop-mode=adaptive"], sentinel)
        check(["--rabitq-predicate-mode=gpu_exact_bias", "--rabitq-predicate-index-path=/index",
               "--rabitq-predicate-bias-alpha=0.15", "--search-early-stop-mode=off"], sentinel)
        check(["--rabitq_predicate_mode=gpu_exact_bias", "--rabitq_predicate_index_path=/index",
               "--rabitq_predicate_bias_alpha=0", "--rabitq_predicate_diagnostics=off"], sentinel)
        check(["--rabitq-predicate-mode=cpu", "--rabitq-predicate-bias-alpha=0"], sentinel,
              {"rabitq": {"predicate_mode": "gpu_exact_bias", "predicate_bias_alpha": 0.3}})
        check([], sentinel, {"rabitq_predicate_mode": "bad", "rabitq": {"predicate_mode": "cpu"}})
        check([], "invalid rabitq_predicate_mode", {"rabitq_predicate_mode": True})
    print("RabitQ predicate CLI validation passed")


if __name__ == "__main__":
    main()
