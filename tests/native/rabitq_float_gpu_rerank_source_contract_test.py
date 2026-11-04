#!/usr/bin/env python3

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2] / "src"
BACKEND = (ROOT / "search/rabitq/backend_rabitq.cu").read_text(encoding="utf-8")
ASYNC_IO = (ROOT / "execution/async_io.cpp").read_text(encoding="utf-8")
DEFAULTS = (ROOT / "app/runtime_defaults.h").read_text(encoding="utf-8")
MAIN = (ROOT / "app/main.cu").read_text(encoding="utf-8")


def require(source: str, needle: str, message: str) -> None:
    if needle not in source:
        raise AssertionError(message)


def main() -> None:
    require(BACKEND, "highfreq_inline_gpu_exact_rerank_enabled(",
            "uint8 inline GPU rerank predicate is missing")
    require(BACKEND, "highfreq_staged_float_gpu_exact_rerank_enabled(",
            "float staged GPU rerank predicate is missing")
    require(DEFAULTS, 'highfreq_rerank_mode_gpu_post = "gpu_post"',
            "post-search GPU mode is not distinct from inline GPU mode")
    require(BACKEND, "highfreq_float_gpu_l2_kernel", "float GPU L2 kernel is missing")
    require(BACKEND, "highfreq_float_gpu_topk_kernel", "float GPU top-k kernel is missing")
    require(BACKEND, "transport=staged_fp32", "float GPU rerank transport is not diagnosed")
    require(BACKEND, "cudaMallocHost float GPU rerank candidate vectors",
            "float GPU rerank staging is not pinned and reusable")
    require(BACKEND, "copy_base_float_vector_row(runtime, sample_id,",
            "float GPU rerank does not use the typed resident/mmap row reader")
    require(BACKEND, "ctx.dim = runtime.base_vectors.cols;",
            "GPU-post depends on an absent mmap FP32 base matrix")
    require(BACKEND, 'runtime.rerank_vector_store->mode() != "mmap_fp16"',
            "GPU-post admits lossy custom stores without declared semantics")
    require(BACKEND, "source_values=%s device_values=fp32",
            "GPU-post source precision is not diagnosed")
    require(ASYNC_IO, "dense_vector_scalar_is_float(vector_payload_scalar_type)",
            "mailbox word layout is not scalar-aware")
    require(ASYNC_IO, "serialize_graph_float_node_row_words(",
            "FP32 rows are not serialized directly into the dynamic mailbox")
    require(ASYNC_IO, "copy_base_float_vector_row(*view.runtime,",
            "FP32 mailbox serialization does not use the typed base-row reader")
    require(BACKEND, "ctx.io_config.set_vector_payload_scalar_type(runtime.vector_scalar_type)",
            "RabitQ graph context does not propagate the payload scalar type")
    require(BACKEND, "const float* d_query_float,",
            "inline GPU scoring does not receive the FP32 query matrix")
    require(BACKEND, "rerank_payload_float",
            "inline GPU scoring does not distinguish uint8 and FP32 payloads")
    require(BACKEND, "__uint_as_float(",
            "FP32 mailbox words are not decoded by the graph kernel")
    require(BACKEND, "highfreq_result_width(options, runtime)",
            "result width does not account for the runtime scalar type")
    require(BACKEND, "base_vector_metadata.scalar_type",
            "dataloader prefetch mode does not inspect the dense scalar type")
    require(BACKEND, "load_config.prefetch_hops = 0;",
            "FP32 inline mode does not disable fixed-Node static prefetch")
    require(BACKEND, "const uint32_t mailbox_prefetch_hops =",
            "FP32 inline graph context does not mirror the loader prefetch policy")
    if "DiskVectorCache" in BACKEND or "base_vectors_on_disk" in BACKEND:
        raise AssertionError("GPU-post retains obsolete disk-cache branches")
    require(BACKEND, "validate_float_gpu_post_vector_storage(runtime);",
            "GPU-post does not check store mode and metadata")
    require(MAIN, "mmap_fp16 GPU-post search/benchmark",
            "CLI rejects the supported FP16 GPU-post search/benchmark mode")
    require(BACKEND, "CPU rerank or mmap_fp16 GPU-post",
            "native options reject the supported FP16 GPU-post mode")
    require(MAIN, "context.gpu_post_exact_rerank = !pq",
            "GPU-post cannot retain the main predicate-generation policy")
    require(BACKEND, "float RabitQ inline GPU rerank requires the default resident FP32 base matrix",
            "unsupported float inline vector storage is not rejected")
    require(BACKEND, "highfreq_rerank_mode=gpu_post currently requires float vectors",
            "uint8 gpu_post is not rejected before returning an unreranked shortlist")

    print("RabitQ float GPU rerank source contracts passed")


if __name__ == "__main__":
    main()
