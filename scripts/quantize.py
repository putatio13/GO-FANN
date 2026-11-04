"""Train PQ or RaBitQ artifacts consumed by GO-FANN (python -m scripts.quantize)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def positive(value: str) -> int:
    number = int(value)
    if not 0 < number <= 2**31 - 1:
        raise argparse.ArgumentTypeError("expected an integer in [1, 2147483647]")
    return number


def nonnegative(value: str) -> int:
    number = int(value)
    if not 0 <= number < 2**64:
        raise argparse.ArgumentTypeError("expected an unsigned 64-bit integer")
    return number


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_subparsers(dest="backend", required=True)
    for backend in ("pq", "rabitq"):
        q = modes.add_parser(backend, help=f"Generate {backend} runtime artifacts")
        q.add_argument("--input", type=Path, required=True,
                       help=".u8bin/.ibin/.bin/.bvecs uint8; .fbin/.fvecs FP32; .hvecs FP16")
        q.add_argument("--output-dir", type=Path, required=True,
                       help="New artifact directory; existing directories are never overwritten")
        q.add_argument("--dim", type=positive, help="Expected dimension; inferred if omitted")
        q.add_argument("--row-limit", type=positive, help="Use first N rows; default all rows")
        q.add_argument("--training-rows", type=positive,
                       default=65536 if backend == "pq" else 262144,
                       help="Maximum training rows (RaBitQ rounds down to an anchor-count multiple)")
        q.add_argument("--chunk-rows", type=positive, default=8192,
                       help="Maximum encoding device rows (default: 8192)")
        q.add_argument("--iterations", type=positive, default=25 if backend == "pq" else 20)
        q.add_argument("--seed", type=nonnegative, default=137,
                       help="Training-row sampling seed; cuVS training/rotation retain library defaults")
        q.add_argument("--builder", type=Path, default=ROOT / "build/quantization" / f"{backend}-builder",
                       help="Native builder executable produced by make quantizers")
        if backend == "pq":
            q.add_argument("--pq-dim", type=int, choices=(16, 64), default=64,
                           help="Number of 8-bit PQ subspaces; must divide input dimension")
            q.add_argument("--verify", action="store_true",
                           help="Check exported reconstruction against cuVS inverse_transform")
        else:
            q.add_argument("--n-lists", type=positive, default=1024, help="Number of RaBitQ anchors")
            q.add_argument("--bits", type=int, choices=range(1, 10), nargs="+", default=[1])
            q.add_argument("--build-mode", choices=("streaming", "batch"), default="streaming",
                           help="Streaming bounds raw vector device residency by chunk-rows")
            q.add_argument("--runtime", type=Path, default=ROOT / "src/go-fann",
                           help="GO-FANN executable used to import cuVS artifacts")
            q.add_argument("--keep-cuvs", action="store_true", help="Also retain intermediate cuVS batch artifacts")
    return p


def run(command: list[str]) -> None:
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def generate(args: argparse.Namespace) -> Path:
    args.input = args.input.resolve(strict=True)
    args.builder = args.builder.resolve(strict=True)
    if not args.input.is_file():
        raise ValueError("input must be a regular file")
    if not os.access(args.builder, os.X_OK):
        raise ValueError("builder is not executable; run make quantizers")
    if args.backend == "pq" and args.training_rows < 256:
        raise ValueError("PQ requires --training-rows >= 256")
    if args.backend == "rabitq":
        args.runtime = args.runtime.resolve(strict=True)
        if not os.access(args.runtime, os.X_OK):
            raise ValueError("runtime is not executable; run make exe")
        if args.training_rows < args.n_lists:
            raise ValueError("training-rows must be at least n-lists")
        if len(set(args.bits)) != len(args.bits):
            raise ValueError("duplicate bit widths are not allowed")
    output = args.output_dir.absolute()
    if args.backend == "rabitq" and "," in str(output):
        raise ValueError("RaBitQ output directory cannot contain a comma (artifact-list separator)")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive directory creation prevents overwriting existing files or another run.
    output.mkdir()
    try:
        with tempfile.TemporaryDirectory(prefix=".building-", dir=output) as temp:
            stage = Path(temp)
            if args.backend == "pq":
                command = [str(args.builder), f"--input={args.input}",
                           f"--pivots={stage / 'pq.pivots.bin'}", f"--codes={stage / 'pq.codes.bin'}",
                           f"--pq-dim={args.pq_dim}", f"--training-rows={args.training_rows}",
                           f"--chunk-rows={args.chunk_rows}", f"--iterations={args.iterations}",
                           f"--seed={args.seed}"]
                if args.verify:
                    command.append("--verify")
            else:
                artifacts = [stage / f"rabitq.b{bits}.cuvs.bin" for bits in args.bits]
                command = [str(args.builder), f"--base-vector-path={args.input}",
                           "--output-list=" + ",".join(map(str, artifacts)),
                           "--bits-per-dim-list=" + ",".join(map(str, args.bits)),
                           f"--n-lists={args.n_lists}", f"--kmeans-iters={args.iterations}",
                           f"--max-train-points-per-cluster={args.training_rows // args.n_lists}",
                           f"--build-mode={args.build_mode}", f"--seed={args.seed}"]
                if args.build_mode == "streaming":
                    command.append(f"--chunk-rows={args.chunk_rows}")
            if args.dim:
                command.append(f"--dim={args.dim}")
            if args.row_limit:
                command.append(f"--row-limit={args.row_limit}")
            run(command)
            if args.backend == "rabitq":
                for bits, intermediate in zip(args.bits, artifacts):
                    command = [str(args.runtime), "--mode=build-rabitq",
                               f"--rabitq-cuvs-artifact-path={intermediate}",
                               f"--rabitq-artifact-path={stage / f'rabitq.b{bits}.bin'}"]
                    run(command)
                    if not args.keep_cuvs:
                        intermediate.unlink()
            expected = ["pq.pivots.bin", "pq.codes.bin"] if args.backend == "pq" else [
                f"rabitq.b{bits}.bin" for bits in args.bits]
            for name in expected:
                artifact = stage / name
                if not artifact.is_file() or artifact.stat().st_size == 0:
                    raise ValueError(f"builder did not produce {name}")
            settings = {key: str(value) if isinstance(value, Path) else value
                        for key, value in vars(args).items()}
            (stage / "parameters.json").write_text(json.dumps(settings, indent=2) + "\n")
            for artifact in stage.iterdir():
                artifact.rename(output / artifact.name)
    except BaseException:
        shutil.rmtree(output)
        raise
    return output


def main() -> None:
    p = parser()
    args = p.parse_args()
    try:
        print("Artifacts:", generate(args))
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        p.exit(1, f"quantize: {error}\n")


if __name__ == "__main__":
    main()
