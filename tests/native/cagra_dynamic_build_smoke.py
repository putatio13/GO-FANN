#!/usr/bin/env python3
"""Bounded synthetic CAGRA4/8 and FullGPU search validation, not paper replay."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import struct
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests.tools.gpu_guard import run_on_idle_gpu


def run(binary, root, gpu):
    root.mkdir(parents=True, exist_ok=False)
    rows, dim, degree = 2048, 32, 16
    labels = [0, 1, 2, 3, 4, 160, 180, 199]
    rng = random.Random(17)
    base = root / 'base.u8bin'
    base.write_bytes(struct.pack('<II', rows, dim) + rng.randbytes(rows * dim))
    spmat = root / 'base.spmat'
    row_labels = [labels[row // 256] for row in range(rows)]
    spmat.write_bytes(struct.pack('<QQQ', rows, 200, rows)
                     + struct.pack(f'<{rows + 1}Q', *range(rows + 1))
                     + struct.pack(f'<{rows}I', *row_labels))
    groups = root / 'identity.groups'
    groups.write_text(''.join(f'{label}\n' for label in range(200)))
    results = []
    for workers in (4, 8):
        folder = root / f'workers{workers}'
        folder.mkdir()
        config = dict(base_vector_path=str(base), data_label_spmat_path=str(spmat),
                      group_path=str(groups), row_limit=rows, dim=dim, degree=degree,
                      specificity_threshold=128, mode='build-graph', run_analyze=False,
                      graph_builder=dict(builder='cagra', cagra_build_worker_count=4),
                      graph_path=str(folder / 'graph.bin'),
                      graph_file_path=str(folder / 'graph.serialized.bin'),
                      lowfreq_index_path=str(folder / 'lowfreq.bin'),
                      seed_sidecar_path=str(folder / 'graph.seed'),
                      original_label_seed_sidecar_path=str(folder / 'graph.orig_label.seed'),
                      metadata_sidecar_path=str(folder / 'graph.metadata'))
        path = folder / 'config.json'
        path.write_text(json.dumps(config, indent=2) + '\n')
        args = [str(binary), f'--config={path}', '--mode=build-graph']
        if workers == 8:
            args.append('--cagra-build-workers=8')  # Overrides nested config4.
        completed = run_on_idle_gpu(args, gpu, capture_output=True, text=True, timeout=180)
        log = completed.stdout + completed.stderr
        (folder / 'build.log').write_text(log)
        assert completed.returncode == 0, log[-4000:]
        assert 'Graph build complete.' in log and '[ERROR]' not in log
        assert f'workers={workers} labels=8' in log, 'worker config/CLI not consumed'
        stats = re.findall(r'\[graph-build-worker\] worker=(\d+) labels=(\d+) rows=(\d+)', log)
        assert len(stats) == workers and sum(int(s[1]) for s in stats) == len(labels)
        assert sum(int(s[2]) for s in stats) == rows
        assert sum(int(s[1]) > 0 for s in stats) >= 2, 'tasks remained on one worker'
        raw = (folder / 'graph.serialized.bin').read_bytes()
        magic, version, block, head, n, d, stride, per_block = struct.unpack('<IIIIQIIQ', raw[:40])
        assert (magic, version, block, head, n, d, stride, per_block) == (
            0x47524446, 1, 4096, 4096, rows, degree, degree * 4, 4096 // (degree * 4))
        assert len(raw) == head + ((rows + per_block - 1) // per_block) * block
        for row in range(rows):
            offset = head + (row // per_block) * block + (row % per_block) * stride
            neighbors = struct.unpack_from(f'<{degree}I', raw, offset)
            assert any(v < 256 for v in neighbors), ('empty graph row', row)
            assert all(v < 256 or v == 0xffffffff for v in neighbors), ('wrong local IDs', row)
        sidecars = {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                    for name in ['graph.seed', 'graph.orig_label.seed', 'graph.metadata', 'lowfreq.bin']}
        results.append(dict(workers=workers, worker_stats=stats, sidecars=sidecars))
        # Every query is an existing vector; its unique zero-distance neighbor
        # is a mathematical oracle independent of the graph or search code.
        query_ids = [group * 256 + offset for group in range(len(labels)) for offset in (17, 211)]
        payload = base.read_bytes()[8:]
        assert len({payload[row * dim:(row + 1) * dim] for row in range(rows)}) == rows
        queries = root / 'query.u8bin'
        queries.write_bytes(struct.pack('<II', len(query_ids), dim) + b''.join(
            payload[row * dim:(row + 1) * dim] for row in query_ids))
        query_labels = root / 'query.spmat'
        query_labels.write_bytes(struct.pack('<QQQ', len(query_ids), 200, len(query_ids))
                                 + struct.pack(f'<{len(query_ids) + 1}Q', *range(len(query_ids) + 1))
                                 + struct.pack(f'<{len(query_ids)}I', *(row_labels[row] for row in query_ids)))
        search = dict(config, mode='search', backend='full_gpu', topk=1,
                      graph_path=str(folder / 'graph.serialized.bin'), graph_storage_mode='memory',
                      query_vector_path=str(queries), query_label_spmat_path=str(query_labels),
                      full_filter=True, search_queue_size=512, search_early_stop_mode='off',
                      warmup_runs=0, timed_runs=1, output_csv_path=str(folder / 'results.csv'))
        search_path = folder / 'search.json'
        search_path.write_text(json.dumps(search, indent=2) + '\n')
        completed = run_on_idle_gpu([str(binary), f'--config={search_path}'], gpu,
                                    capture_output=True, text=True, timeout=180)
        (folder / 'search.log').write_text(completed.stdout + completed.stderr)
        assert completed.returncode == 0, (completed.stdout + completed.stderr)[-4000:]
        actual = [int(line.strip()) for line in (folder / 'results.csv').read_text().splitlines()]
        assert actual == query_ids, ('FullGPU filtered zero-distance ID mismatch', actual, query_ids)
        results[-1]['full_gpu_self_queries'] = len(query_ids)
        results[-1]['full_gpu_correct_ids'] = len(actual)
    assert results[0]['sidecars'] == results[1]['sidecars'], 'group/seed/posting mapping changed'
    for value in ['0', '-1', '4294967296']:
        completed = subprocess.run([str(binary), f'--cagra-build-workers={value}'],
                                   capture_output=True, text=True, timeout=30)
        assert completed.returncode != 0 and 'cagra-build-workers' in completed.stderr
    (root / 'VALIDATED.json').write_text(json.dumps(results, indent=2) + '\n')
    print('Synthetic CAGRA4/8, local IDs, sidecars, FullGPU exact self IDs and invalid CLI checks passed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=2)
    args = parser.parse_args()
    run(args.binary.resolve(), args.root.resolve(), args.gpu)
