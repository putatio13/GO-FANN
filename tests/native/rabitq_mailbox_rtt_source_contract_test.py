#!/usr/bin/env python3
"""Structural contracts for opt-in coupled RabitQ mailbox RTT profiling."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2] / "src"
BACKEND = (ROOT / "search/rabitq/backend_rabitq.cu").read_text(encoding="utf-8")


def require(text: str, message: str) -> None:
    if text not in BACKEND:
        raise AssertionError(message)


def main() -> None:
    for text, message in (
        ("GO_FANN_RABITQ_MAILBOX_RTT_PROFILE", "profile environment gate is missing"),
        ("launch.template operator()<true, true, kUseInt8Score, false>()",
         "profiled request-ring scorer specialization is missing"),
        ("launch.template operator()<true, false, kUseInt8Score, false>()",
         "profiled scan scorer specialization is missing"),
        ("launch.template operator()<false, true, kUseInt8Score, false>()",
         "baseline request-ring scorer specialization is missing"),
        ("launch.template operator()<false, false, kUseInt8Score, false>()",
         "baseline scan scorer specialization is missing"),
        ("launch_request_mode.template operator()<true>()",
         "int8 scorer request-ring dispatch is missing"),
        ("launch_request_mode.template operator()<false>()",
         "FP32 scorer request-ring dispatch is missing"),
        ("mailbox RTT profiling currently requires coupled query workers",
         "finite/profile incompatibility guard is missing"),
        ("cudaMemset RabitQ mailbox RTT histogram", "per-run histogram reset is missing"),
        ("clock=gpu scope=publish-to-observe", "RTT clock and scope telemetry is missing"),
        ("kFixedWaitSleepUs = 100", "fixed wait interval must be named and auditable"),
        ("mailbox_rtt_profile=%d wait_policy=%s", "runtime wait-policy telemetry is missing"),
        ("? kInitialWaitSleepUs", "adaptive wait interval is missing from telemetry"),
        (": kFixedWaitSleepUs", "fixed wait interval is missing from telemetry"),
        ("mailbox_request_start_ns[cid] = warp_kernel::get_global_timer_ns()",
         "GPU request timestamp is missing"),
        ("warp_kernel::get_global_timer_ns() - mailbox_request_start_ns[cid]",
         "GPU reply observation timestamp is missing"),
    ):
        require(text, message)

    print("RabitQ mailbox RTT source contracts passed")


if __name__ == "__main__":
    main()
