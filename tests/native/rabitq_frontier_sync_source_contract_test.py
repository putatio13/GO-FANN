#!/usr/bin/env python3
"""Link the delayed-producer CUDA fixture to the real native call-site barrier."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2] / "src"


def strip_comments(source: str) -> str:
    return re.sub(r"//[^\n]*|/\*.*?\*/", "", source, flags=re.DOTALL)


def closing_brace(source: str, opening: int) -> int:
    depth = 0
    for pos in range(opening, len(source)):
        if source[pos] == "{":
            depth += 1
        elif source[pos] == "}":
            depth -= 1
            if depth == 0:
                return pos
    raise AssertionError("unbalanced producer block")


def validate(source: str) -> None:
    source = strip_comments(source)
    begin = source.index("__device__ void run_rabitq_search_one_block")
    end = source.index("void native_rabitq_search_kernel", begin)
    native = source[begin:end]
    insertion = native.rindex("smmh2::insert(q[cid], heap_size[cid], kp)")
    producer = native.rfind("if (subtid == 0)", 0, insertion)
    producer_end = closing_brace(native, native.index("{", producer))
    assert producer_end > insertion, "expected lane0 frontier producer"
    first_pickup = native.index("if (step_num == 1)", producer_end)
    gap = native[producer_end + 1:first_pickup]
    assert re.fullmatch(r"\s*__syncwarp\s*\(\s*\)\s*;\s*", gap), (
        "shared frontier publication needs an unconditional call-site __syncwarp "
        "after lane0 insertion and before evaluating cooperative pickup arguments"
    )
    call = native.index("pickup_next_unused_from_native_frontier<true>", first_pickup)
    call_end = native.index(";", call)
    assert "q[cid], heap_size[cid]" in native[call:call_end], (
        "the audited pickup must read the newly published shared heap size"
    )


def main() -> None:
    source = (ROOT / "search/rabitq/backend_rabitq.cu").read_text()
    validate(source)
    # Prove this guard rejects the original omission, rather than accidentally
    # accepting another unrelated barrier elsewhere in the native function.
    marker = "        if (step_num == 1) {"
    before = source.index(marker, source.index("__device__ void run_rabitq_search_one_block"))
    barrier = source.rfind("__syncwarp();", 0, before)
    without = source[:barrier] + source[barrier + len("__syncwarp();"):]
    try:
        validate(without)
    except AssertionError:
        pass
    else:
        raise AssertionError("source guard did not reject the missing producer barrier")
    print("[OK] native frontier producer barrier precedes cooperative argument loads")


if __name__ == "__main__":
    main()
