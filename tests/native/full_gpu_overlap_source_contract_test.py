#!/usr/bin/env python3
"""Static contract for the optional overlap graph path.

The CUDA smoke is run only on an idle GPU with a materialized bridge artifact;
this test keeps the graph-context/sample-id invariants visible in CI.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "src"
backend = (ROOT / "search/full_gpu/backend_full_gpu.cu").read_text()
overlap = (ROOT / "search/full_gpu/full_gpu_overlap_search.cuh").read_text()
types_ = (ROOT / "search/full_gpu/full_gpu_overlap_types.cuh").read_text()

for needle in (
    "overlap_graph_path", "overlap_manifest_path", "overlap_max_bridges",
    "prepare_full_gpu_overlap", "launch_full_gpu_overlap",
):
    assert needle in backend or needle in overlap, needle
for needle in (
    "sample ID", "graph occurrence keys", "overlap_local_for_sample",
    "overlap_visit", "distance_reuses", "bridge_expansions",
):
    assert needle in overlap, needle
assert "FullGpuOverlapContext" in types_
assert "FullGpuOverlapQuery" in types_
assert "context.key_begin" in overlap
print("full_gpu_overlap_source_contract_test: PASS")
