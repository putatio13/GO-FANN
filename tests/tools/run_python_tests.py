#!/usr/bin/env python3
"""Run portable behavior regressions as modules without datasets or GPUs."""
from pathlib import Path
import argparse
import os
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[2]
TESTS = [
    'tests.python.test_reproduce_profile',
    'tests.python.test_evaluate',
    'tests.python.test_quantize',
    'tests.python.test_gpu_guard',
    'tests.python.test_verify_paper_inputs',
    'tests.python.range_posting_generator_test',
    'tests.python.range_hierarchy_generator_test',
]
DATA_TESTS = [
    'tests.python.test_reproduce_plots',
    'tests.python.test_laion_metadata',
    'tests.python.test_generate_laion_width_height_range_gt',
    'tests.python.test_combine_laion_range_gt',
    'tests.python.test_laion_pipeline',
    'tests.python.test_generate_laion_mixed_width_height_range_queries',
    'tests.python.test_generate_laion_pairwise_hybrid_query_sets',
    'tests.python.test_materialize_laion_query_runtime_ranges',
]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', action='store_true', help='Also test LAION preparation and exact GT (PyArrow, DuckDB and PyTorch).')
    args = parser.parse_args()
    tests = TESTS + (DATA_TESTS if args.data else [])
    env = os.environ.copy()
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    failures = []
    for test in tests:
        print(f'Running {test}', flush=True)
        result = subprocess.run([sys.executable, '-m', test], cwd=ROOT, env=env)
        if result.returncode: failures.append(test)
    print(f'{len(tests)-len(failures)}/{len(tests)} modules passed', flush=True)
    if failures: print('Failed: ' + ', '.join(failures), file=sys.stderr)
    return bool(failures)

if __name__ == '__main__':
    sys.exit(main())
