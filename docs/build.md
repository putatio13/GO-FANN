# Build and quick start

## Environment

Use Linux x86-64, an NVIDIA GPU and a CUDA 12.8-compatible NVIDIA driver. The native runtime uses C++17. Install GNU Make and a system GCC/G++ compiler with OpenMP support; the tested host compiler is GCC 12.2.0.

Create the environment from the repository root:

```bash
conda env create -f environment.yml
conda activate go-fann
export CUDA_PATH="$CONDA_PREFIX"
export NVCC="$CONDA_PREFIX/bin/nvcc"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

[environment.yml](../environment.yml) includes the CUDA compiler, cuVS/RAFT/RMM libraries and headers, OpenSSL, nlohmann/json, NumPy, PyArrow, DuckDB and NLTK. It uses the public RAPIDS, conda-forge and NVIDIA channels; `nodefaults` follows the [RAPIDS channel guidance](https://docs.rapids.ai/install/). The native build uses the activated environment for CUDA and library dependencies and the system host compiler.

The tested component versions are:

| Component | Tested version |
| --- | --- |
| CUDA / nvcc | 12.8 / 12.8.93 |
| cuVS, RAFT, RMM | 26.02.00 |
| rapids_logger | 0.2.3 |
| Host GCC/G++ | 12.2.0 |
| GDRCopy | 2.5, with working gdrdrv |
| Python / NumPy | 3.12.12 / 2.2.5 |
| PyArrow / DuckDB / NLTK | 23.0.1 / 1.5.2 / 3.9.4 |
| PyTorch for exact GT | 2.10.0+cu126 |

For GPU ground-truth generation, install the CUDA 12.6 PyTorch wheel, which includes V100 (`sm_70`) support. PyTorch supplies its own CUDA runtime; native GO-FANN compilation continues to use CUDA 12.8 from Conda. The command follows the [official PyTorch 2.10 installation instructions](https://pytorch.org/get-started/previous-versions/#v2100):

```bash
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126
python -m nltk.downloader stopwords
```

The stopwords corpus is used by LAION caption preprocessing. Skip these two commands when using prepared inputs for search only.

## Host setup and compilation

Install [GDRCopy 2.5](https://github.com/NVIDIA/gdrcopy/tree/v2.5) using its instructions for the host's kernel and NVIDIA driver. Both `gdrapi.h`/`libgdrapi` and the `gdrdrv` kernel module are required. An administrator must install/load the module and provide access to `/dev/gdrdrv`; Conda does not install host GPU drivers or kernel modules. Check the host setup before running:

```bash
nvidia-smi
lsmod | grep '^gdrdrv'
ls -l /dev/gdrdrv
```

Build for the target GPU architecture (`sm_70` for V100):

```bash
make exe CUDA_CAPABILITY=sm_70
make quantizers CUDA_CAPABILITY=sm_70
./src/go-fann --help
```

Useful Make overrides: `CUDA_PATH`, `NVCC`, `CUDA_CAPABILITY`, `DEPENDENCY_PREFIX`, `CUVS_INCLUDE`, `RAFT_INCLUDE`, `CUVS_LIB_PATH`, `GDRAPI_INCLUDE`, `GDRAPI_LIB_PATH`, `MINICONDA_INCLUDE`. Default GDRCopy locations are `/usr/local/include` and `/usr/local/lib`.

Use `make clean` before changing toolchain, architecture or build options to avoid stale objects. GDRCopy remains required for the CPU–GPU host service channel.

The result is `src/go-fann`. The root Makefile delegates compilation to `src/Makefile`; run the binary from the repository root so the supplied configs resolve correctly. Source modules are described in [src/README.md](../src/README.md). Objects/stamps use `build/obj`, build-only test binaries use `build/tests`, and quantizer binaries/objects use `build/quantization`. `make clean` removes these generated paths plus the runtime binary.

## Quick synthetic run

After building, run a small generated dataset without downloading the paper datasets. Choose an idle physical GPU by its `nvidia-smi` index:

```bash
GPU=${GPU:-0}
python -m tests.tools.run_gpu_smoke --root=build/gpu-smoke --gpu="$GPU" --arch=sm_70
```

Use a new output directory on each run. The runner checks an idle physical GPU, serializes jobs, and writes compiler/runtime logs plus `VALIDATED.json`. It checks 12 CUDA components and builds/searches a small synthetic dataset. It sets CUDA library lookup paths from the activated conda prefix for its child tests.

After building both quantization producers with `make quantizers`, run their reader and generated-artifact checks:

```bash
make check-quantizers
python tests/native/quantization_smoke.py --root=build/quantization-smoke --gpu="$GPU"
```

The GPU check requires a new output directory. It generates small PQ/RaBitQ artifacts, checks their numerical/layout contracts, and searches with the native runtime. CPU, GPU-inline and supported GPU-post reranking are checked against exact filtered-neighbor calculations, including empty and short results. It also compares profiling on/off and converts real PQ/RaBitQ captures. Run it separately from other GPU jobs.

## CPU and preparation checks

```bash
python -m tests.tools.verify_release
make test-data
make test-cpu
python tests/native/rabitq_predicate_cli_test.py ./src/go-fann
```

The data suite exercises query generation, metadata, GT helpers, evaluation and figure inputs with small fixtures. The portable Python tests include trace conversion and independent overlap-accounting oracles. Host tests exercise the actual predicate policy, scheduling and trace recorder. The CLI check uses the built binary to validate both PQ and RaBitQ options without loading data or initializing CUDA. Source verification uses only the Python standard library. `make test-source-contracts` checks structural invariants in native source; numerical and execution checks are provided by the CPU and GPU tests above. Additional targets are listed in `src/Makefile`.

For profiling an experiment and plotting its stage breakdown, follow the
[capture and conversion commands](../REPRODUCE.md#search-window-breakdown).
