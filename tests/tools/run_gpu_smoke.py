"""Run bounded CUDA numerical/ID tests and synthetic CAGRA + FullGPU validation.

Activate the cuVS conda environment first. This does not train quantizers or
reproduce paper results. PQ/RaBitQ coverage here is component-level only.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from tests.tools.gpu_guard import run_on_idle_gpu


REPO = Path(__file__).resolve().parents[2]
CASES = (
    ('rabitq_query_rotation', 'rabitq_query_rotation_test.cu', ['-lcublas']),
    ('full_gpu_sorted_frontier', 'full_gpu_sorted_frontier_test.cu', []),
    ('pq_resident_posting', 'pq_resident_posting_cuda_test.cu', []),
    ('pq_predicate_scalar', 'pq_gpu_predicate_test.cu', ['-DGO_FANN_GPU_PREDICATE_PARALLEL=0']),
    ('pq_predicate_parallel', 'pq_gpu_predicate_test.cu', ['-DGO_FANN_GPU_PREDICATE_PARALLEL=1']),
    ('posting_predicate', 'posting_predicate_cuda_test.cu', []),
    ('posting_completion', 'posting_completion_cuda_test.cu', ['-Xcompiler=-pthread']),
    ('gpu_predicate', 'gpu_predicate_cuda_test.cu', []),
    ('gpu_predicate_scalar', 'gpu_predicate_parallel_test.cu', ['-DGO_FANN_GPU_PREDICATE_PARALLEL=0']),
    ('gpu_predicate_parallel', 'gpu_predicate_parallel_test.cu', ['-DGO_FANN_GPU_PREDICATE_PARALLEL=1']),
    ('gpu_predicate_csr_lookup', 'gpu_predicate_csr_lookup_test.cu', ['-DGO_FANN_GPU_PREDICATE_PARALLEL=1']),
    ('gpu_predicate_frontier_sync', 'gpu_predicate_frontier_sync_test.cu', []),
)


def checked(command, logfile, gpu=None):
    with logfile.open('w') as log:
        kwargs = dict(stdout=log, stderr=subprocess.STDOUT, text=True, timeout=300, cwd=REPO)
        result = (subprocess.run(command, **kwargs) if gpu is None
                  else run_on_idle_gpu(command, gpu, **kwargs))
    if result.returncode:
        raise RuntimeError(f'Command failed ({result.returncode}); see {logfile}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='New output directory; never overwritten')
    parser.add_argument('--gpu', type=int, default=2, help='Physical GPU index; must be idle')
    parser.add_argument('--binary', type=Path, default=REPO / 'src/go-fann')
    parser.add_argument('--arch', default='sm_70')
    parser.add_argument('--components-only', action='store_true', help='Skip CAGRA/FullGPU end-to-end checks')
    args = parser.parse_args()
    prefix = os.environ.get('CONDA_PREFIX')
    if not prefix:
        parser.error('Activate the cuVS conda environment (CONDA_PREFIX is required)')
    nvcc = Path(prefix) / 'bin/nvcc'
    if not nvcc.is_file():
        parser.error(f'CUDA compiler missing from activated environment: {nvcc}')
    binary = args.binary.resolve()
    if not args.components_only and not binary.is_file():
        parser.error(f'Build the executable first: {binary}')
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    passed = []
    for name, source, flags in CASES:
        executable = root / name
        checked([str(nvcc), '-ccbin=/usr/bin/g++', '-std=c++17', '-O2', f'-arch={args.arch}',
                 f'-I{REPO / "src"}', f'-I{REPO / "include"}',
                 '-DGO_FANN_GPU_PREDICATE_CSR_THRESHOLD=32',
                 str(REPO / 'tests/native' / source), *flags, '-o', str(executable)],
                root / f'{name}.build.log')
        checked([str(executable)], root / f'{name}.run.log', args.gpu)
        passed.append(name)
        print(f'PASS: {name}', flush=True)
    (root / 'COMPONENTS_VALIDATED.json').write_text(json.dumps(
        dict(component_tests=passed, physical_gpu=args.gpu, architecture=args.arch), indent=2) + '\n')
    if not args.components_only:
        checked([sys.executable, str(REPO / 'tests/native/cagra_dynamic_build_smoke.py'),
                 f'--binary={binary}', f'--root={root / "cagra"}', f'--gpu={args.gpu}'],
                root / 'cagra.log')
    summary = dict(scope='synthetic correctness validation; not paper performance replay',
                   component_tests=passed, physical_gpu=args.gpu, architecture=args.arch,
                   cagra_full_gpu_end_to_end=not args.components_only,
                   pq_rabitq_end_to_end=False)
    if not args.components_only:
        summary['binary_sha256'] = hashlib.sha256(binary.read_bytes()).hexdigest()
        summary['cagra_results'] = json.loads((root / 'cagra/VALIDATED.json').read_text())
    (root / 'VALIDATED.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
