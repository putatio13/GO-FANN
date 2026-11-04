#!/usr/bin/env python3

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2] / "src"
BACKEND = (ROOT / "search/rabitq/backend_rabitq.cu").read_text(encoding="utf-8")
DATALOADER = (ROOT / "data/dataloader.cpp").read_text(encoding="utf-8")


def require(needle: str, message: str) -> None:
    if needle not in BACKEND:
        raise AssertionError(message)


def require_dataloader(needle: str, message: str) -> None:
    if needle not in DATALOADER:
        raise AssertionError(message)


def main() -> None:
    primary = BACKEND.split(
        "std::future<RabitqIterationResult> primary_iteration_future = std::async(", 1
    )[1].split("        try {\n            primary_gpu_complete.get();", 1)[0]
    guarded_start = primary.index("                try {")
    device_setup = primary.index("check_cuda(cudaSetDevice(active_device)")
    failure_handler = primary.index("                } catch (...) {")
    if not guarded_start < device_setup < failure_handler:
        raise AssertionError("primary device setup must publish failure to its waiting caller")
    if "primary_gpu_complete_promise.set_exception(" not in primary[failure_handler:]:
        raise AssertionError("primary failure before GPU completion must release the caller")
    require("float highfreq_exact_l2_for_sample(",
            "shared high-frequency exact L2 helper is missing")
    require("compute_base_float_vector_l2(runtime,",
            "float rerank must use the shared vector-store-aware L2 helper")
    require_dataloader("runtime.rerank_vector_store->l2_float(sample_id, query)",
                       "custom float rerank store is not scored directly")
    require_dataloader("runtime.rerank_vector_store->prepare_l2_query(query, cols)",
                       "custom float rerank query is not prepared through its store")
    require_dataloader("runtime.base_float_vectors.row_ptr(sample_id)",
                       "resident float rerank must score the base row directly")
    require("runtime.base_vectors.row_ptr(sample_id)",
            "resident uint8 rerank must score the base row directly")
    for retired in ("base_vectors_on_disk", "DiskVectorCache", "DiskVectorBatch"):
        if retired in BACKEND or retired in DATALOADER:
            raise AssertionError("retired disk rerank dependency remains: " + retired)
    require("if (allow_early_stop &&\n                    !profile.sample(cpu_profile::Detail::HashLookup, [&] {\n                        return seen_sample_ids.insert",
            "exact rerank must not build the heuristic cross-batch dedup hash")
    require("std::vector<idx_t> prev_topk(allow_early_stop ? topk : 0",
            "exact rerank must not allocate heuristic stability state")
    if BACKEND.count("highfreq_exact_l2_for_sample(runtime,") < 2:
        raise AssertionError("single-route and merged-route rerank must share direct-row scoring")
    if BACKEND.count("compute_base_float_vector_l2(") < 3:
        raise AssertionError("high- and low-frequency float rerank paths must share vector-store scoring")
    if BACKEND.count("prepare_base_float_vector_l2_query(") < 4:
        raise AssertionError(
            "all high/low and single/merged float rerank paths must prepare query state once")

    print("RabitQ CPU rerank source contracts passed")


if __name__ == "__main__":
    main()
