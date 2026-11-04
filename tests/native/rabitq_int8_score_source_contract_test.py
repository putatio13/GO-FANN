#!/usr/bin/env python3

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2] / "src"


def require(source: str, needle: str, message: str) -> None:
    if needle not in source:
        raise AssertionError(message)


def main() -> None:
    backend = (ROOT / "search/rabitq/backend_rabitq.cu").read_text()
    options = (ROOT / "search/rabitq/backend_rabitq.cuh").read_text()
    config = (ROOT / "app/config_loader.cpp").read_text()
    config_header = (ROOT / "app/config_loader.h").read_text()
    cli = (ROOT / "app/main.cu").read_text()

    require(options, "bool use_int8_score = false;", "int8 scoring must remain opt-in")
    require(config_header, "rabitq_use_int8_score", "config value is not declared")
    require(config, '"use_int8_score"', "nested RabitQ config is not loaded")
    require(config, '"rabitq_use_int8_score"', "root compatibility config is not loaded")
    require(cli, '"--enable-rabitq-int8-score"', "enable CLI flag is missing")
    require(cli, '{"use_int8_score", options.rabitq_use_int8_score}',
            "effective nested config does not report int8 scoring")
    require(cli, "options.rabitq_bits_per_dim <= 2",
            "coupled memory b1/b2 must default to int8 scoring")

    require(backend, "quantize_rabitq_queries_int8_kernel", "query quantizer is missing")
    require(backend, "kRabitqDp4aNibbleMasks", "nibble expansion table is missing")
    require(backend, "__dp4a", "DP4A scoring is missing")
    require(backend, "rabitq_dp4a_binary_dot_lane<true>",
            "short-code DP4A bit-plane scoring is missing")
    require(backend, "rabitq_dp4a_binary_dot_lane<false>",
            "b2 extra-code DP4A bit-plane scoring is missing")
    require(backend, "long_ip = static_cast<float>(long_ip_int) * query_scale;",
            "b2 extra-code DP4A result is not scaled")
    require(backend, "template <bool kUseInt8Score = false>",
            "scorer must use compile-time specialization")
    require(backend, "score_rabitq_code_device<kUseInt8Score>",
            "native graph traversal does not select the specialized scorer")
    require(backend, "native_rabitq_search_kernel<QS,", "native kernel dispatch is missing")
    require(backend, "RabitQ int8 score mode currently requires a b1 or b2 artifact",
            "b1/b2 validation is missing")
    require(backend, "resolved.use_int8_score_auto && index.codec.bits_per_dim > 2u",
            "artifact-aware auto resolution must preserve b2 int8 scoring")
    require(backend, "RabitQ float LUT and int8 score modes are mutually exclusive",
            "LUT conflict validation is missing")
    require(backend, "RabitQ int8 score mode currently requires coupled query workers",
            "unsupported finite execution is not rejected")

    print("RabitQ int8 score source contract test passed")


if __name__ == "__main__":
    main()
