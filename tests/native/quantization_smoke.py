#!/usr/bin/env python3
"""Small generated-artifact numerical, import and filtered-search oracles."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import struct
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests.tools.gpu_guard import run_on_idle_gpu


def pq_oracle(folder, vectors, chunks):
    raw = (folder / 'pq.pivots.bin').read_bytes()
    assert struct.unpack_from('<II', raw) == (4, 1)
    offsets = struct.unpack_from('<4Q', raw, 8)
    assert offsets[3] == len(raw)
    def matrix(offset, dtype):
        rows, cols = struct.unpack_from('<II', raw, offset)
        return np.frombuffer(raw, dtype, rows * cols, offset + 8).reshape(rows, cols)
    table = matrix(offsets[0], '<f4')
    mean = matrix(offsets[1], '<f4').ravel()
    boundaries = matrix(offsets[2], '<u4').ravel()
    n, d = vectors.shape
    assert table.shape == (256, d) and mean.shape == (d,) and not np.any(mean)
    assert np.array_equal(boundaries, np.arange(chunks + 1) * (d // chunks))
    data = (folder / 'pq.codes.bin').read_bytes()
    assert struct.unpack_from('<II', data) == (n, chunks) and len(data) == 8 + n * chunks
    codes = np.frombuffer(data, 'u1', offset=8).reshape(n, chunks)
    reconstructed = np.empty_like(vectors, dtype=np.float32)
    for m, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        subtable = table[:, start:end]
        dist = ((vectors[:, None, start:end].astype(np.float64) - subtable[None]) ** 2).sum(axis=2)
        selected = dist[np.arange(n), codes[:, m]]
        assert np.allclose(selected, dist.min(axis=1), atol=1e-4, rtol=1e-5), 'codes not nearest centers'
        reconstructed[:, start:end] = subtable[codes[:, m]]
    query = vectors[7].astype(np.float64)
    direct = ((reconstructed - query) ** 2).sum(axis=1)
    adc = sum(((table[codes[:, m], start:end] - query[start:end]) ** 2).sum(axis=1)
              for m, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])))
    assert np.allclose(direct, adc, rtol=1e-6, atol=1e-4)


def rabitq_oracle(cuvs_path, native_path):
    raw = cuvs_path.read_bytes()
    n, dim, anchors, extra = struct.unpack_from('<4Q', raw)
    assert raw[32] == 1 and extra <= 8
    padded = ((dim + 63) // 64) * 64
    short_bytes, long_bytes = padded // 8, padded * extra // 8
    offset = 33
    def read(dtype, count):
        nonlocal offset
        arr = np.frombuffer(raw, dtype, count, offset).copy()
        offset += count * np.dtype(dtype).itemsize
        return arr
    scales = read('<f4', 2)
    sizes = read('<u8', anchors)
    assert sizes.sum() == n
    rotation = read('<f4', padded * padded)
    centers = read('<f4', anchors * padded)
    codes = read('u1', n * short_bytes).reshape(n, short_bytes)
    factors = read('<f4', n * 3).reshape(n, 3)
    longcodes = read('u1', n * long_bytes).reshape(n, long_bytes)
    extra_factors = read('<f4', n * 2).reshape(n, 2)
    ids_offset = offset
    ids = read('<u4', n)
    assert offset == len(raw) and np.array_equal(np.sort(ids), np.arange(n))
    cluster_labels = np.repeat(np.arange(anchors), sizes.astype(int))
    order = np.argsort(ids)
    native = native_path.read_bytes()
    assert struct.unpack_from('<IIIIQ', native) == (2, dim, anchors, short_bytes, n)
    codec = struct.unpack_from('<7I', native, 24)
    assert codec[:4] == (padded, extra + 1, padded // 32, long_bytes)
    pos = 52
    for expected, dtype in [(centers, '<f4'), (rotation, '<f4'), (scales, '<f4'),
                            (cluster_labels[order], '<u4'), (codes[order].ravel(), 'u1'),
                            (longcodes[order].ravel(), 'u1'), (factors[order].ravel(), '<f4'),
                            (extra_factors[order].ravel(), '<f4')]:
        count, = struct.unpack_from('<Q', native, pos); pos += 8
        assert count == expected.size
        actual = np.frombuffer(native, dtype, count, pos); pos += count * np.dtype(dtype).itemsize
        assert np.array_equal(actual, expected, equal_nan=True), 'import did not preserve sample-major payload'
    assert pos == len(native)
    return ids_offset


def run(root, gpu, binary):
    root.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[2]
    checks = []
    def execute(name, command, success=True):
        (root / (name + '.command.json')).write_text(json.dumps(list(map(str, command))) + '\n')
        result = run_on_idle_gpu(list(map(str, command)), gpu, cwd=repo,
                                capture_output=True, text=True, timeout=300)
        (root / (name + '.log')).write_text(result.stdout + result.stderr)
        assert (result.returncode == 0) == success, (name, result.returncode, (result.stdout + result.stderr)[-3000:])
        checks.append(name)
        return result.stdout + result.stderr
    rng = np.random.default_rng(17)
    vectors = rng.integers(0, 256, (2048, 96), dtype=np.uint8)
    base = root / 'base.u8bin'
    base.write_bytes(struct.pack('<II', *vectors.shape) + vectors.tobytes())
    prefix = [sys.executable, '-m', 'scripts.quantize']
    pq_folder = root / 'pq'
    execute('pq-build', prefix + ['pq', '--input', base, '--output-dir', pq_folder,
            '--pq-dim', '16', '--training-rows', '1024', '--chunk-rows', '317', '--iterations', '5', '--verify'])
    pq_oracle(pq_folder, vectors, 16)
    execute('pq-overwrite-rejected', prefix + ['pq', '--input', base, '--output-dir', pq_folder], False)
    execute('pq-dim-rejected', prefix + ['pq', '--input', base, '--output-dir', root / 'bad-dim', '--dim', '95'], False)
    assert not (root / 'bad-dim').exists()
    # Fractional float and half inputs exercise the same encoding path and bounded row prefix.
    float_vectors = (rng.normal(size=(512, 32)) * 3.5).astype('<f4')
    for ext in ('fbin', 'hvecs'):
        path = root / ('fractional.' + ext)
        if ext == 'fbin':
            path.write_bytes(struct.pack('<II', *float_vectors.shape) + float_vectors.tobytes())
            expected = float_vectors
        else:
            half = float_vectors.astype('<f2')
            path.write_bytes(b''.join(struct.pack('<I', 32) + row.tobytes() for row in half))
            expected = half.astype('<f4')
        folder = root / ('pq-' + ext)
        execute('pq-' + ext, prefix + ['pq', '--input', path, '--output-dir', folder,
                '--pq-dim', '16', '--training-rows', '256', '--row-limit', '384',
                '--chunk-rows', '113', '--iterations', '3', '--verify'])
        pq_oracle(folder, expected[:384], 16)
    rabitq_folder = root / 'rabitq'
    execute('rabitq-stream', prefix + ['rabitq', '--input', base, '--output-dir', rabitq_folder,
            '--n-lists', '16', '--bits', '1', '2', '--training-rows', '512', '--chunk-rows', '311',
            '--iterations', '4', '--keep-cuvs', '--runtime', binary])
    for bits in (1, 2):
        rabitq_oracle(rabitq_folder / f'rabitq.b{bits}.cuvs.bin', rabitq_folder / f'rabitq.b{bits}.bin')
    # Wide FP32 fixture forces the padded 4096D path, with fractional data preserved.
    wide = rng.normal(size=(64, 4096)).astype('<f4')
    wide_path = root / 'wide.fbin'
    wide_path.write_bytes(struct.pack('<II', *wide.shape) + wide.tobytes())
    for mode, source in [('batch', root / 'fractional.fbin'), ('streaming', wide_path)]:
        folder = root / ('rabitq-' + mode)
        execute('rabitq-' + mode, prefix + ['rabitq', '--input', source, '--output-dir', folder,
                '--n-lists', '4', '--bits', '1', '2', '--training-rows', '32', '--chunk-rows', '23',
                '--iterations', '3', '--build-mode', mode, '--keep-cuvs', '--runtime', binary])
        for bits in (1, 2):
            rabitq_oracle(folder / f'rabitq.b{bits}.cuvs.bin', folder / f'rabitq.b{bits}.bin')
    # Corrupt a real valid emitted artifact, retaining independently valid remaining fields.
    good = rabitq_folder / 'rabitq.b1.cuvs.bin'
    ids_offset = rabitq_oracle(good, rabitq_folder / 'rabitq.b1.bin')
    for name in ('duplicate-id', 'out-of-range-id', 'not-batch', 'cluster-total', 'bad-bits'):
        data = bytearray(good.read_bytes())
        if name == 'duplicate-id': data[ids_offset:ids_offset + 4] = data[ids_offset + 4:ids_offset + 8]
        elif name == 'out-of-range-id': struct.pack_into('<I', data, ids_offset, len(vectors))
        elif name == 'not-batch': data[32] = 0
        elif name == 'cluster-total': struct.pack_into('<Q', data, 41, len(vectors) + 1)
        else: struct.pack_into('<Q', data, 24, 9)
        bad = root / (name + '.bin'); bad.write_bytes(data)
        execute(name, [binary, '--mode=build-rabitq', f'--rabitq-cuvs-artifact-path={bad}',
                      f'--rabitq-artifact-path={root / (name + ".native.bin")}'], False)
    search_oracle(root, binary, vectors, execute)
    rerank_oracle(root, binary, vectors, float_vectors, execute)
    (root / 'VALIDATED.json').write_text(json.dumps({
        'passed': checks, 'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
    }, indent=2) + '\n')
    print('Quantization numerical/import/search checks passed:', len(checks))


def search_oracle(root, binary, vectors, execute):
    n, dim = vectors.shape
    labels = np.arange(n, dtype=np.uint32) // 256
    def spmat(path, row_labels):
        count = len(row_labels)
        path.write_bytes(struct.pack('<QQQ', count, 8, count)
                         + np.arange(count + 1, dtype='<u8').tobytes()
                         + np.asarray(row_labels, dtype='<u4').tobytes())
    spmat(root / 'base.spmat', labels)
    (root / 'identity.groups').write_text(''.join(f'{i}\n' for i in range(8)))
    ids = [group * 256 + offset for group in range(8) for offset in (17, 211)]
    (root / 'query.u8bin').write_bytes(struct.pack('<II', len(ids), dim) + vectors[ids].tobytes())
    spmat(root / 'query.spmat', labels[ids])
    config = dict(base_vector_path=str(root / 'base.u8bin'), data_label_spmat_path=str(root / 'base.spmat'),
                  group_path=str(root / 'identity.groups'), row_limit=n, dim=dim, degree=16,
                  specificity_threshold=128, mode='build-graph', run_analyze=False,
                  graph_builder=dict(builder='cagra', cagra_build_worker_count=4),
                  graph_path=str(root / 'graph.bin'), graph_file_path=str(root / 'graph.serialized.bin'),
                  lowfreq_index_path=str(root / 'lowfreq.bin'), seed_sidecar_path=str(root / 'graph.seed'),
                  original_label_seed_sidecar_path=str(root / 'graph.orig_label.seed'),
                  metadata_sidecar_path=str(root / 'graph.metadata'))
    path = root / 'build.json'; path.write_text(json.dumps(config))
    execute('cagra-build', [binary, f'--config={path}', '--mode=build-graph'])
    for backend in ('pq', 'rabitq'):
        search = dict(config, mode='search', backend=backend, topk=1,
                      graph_path=str(root / 'graph.serialized.bin'), graph_storage_mode='memory',
                      query_vector_path=str(root / 'query.u8bin'), query_label_spmat_path=str(root / 'query.spmat'),
                      full_filter=True, per_step_full_filter=True, search_queue_size=256,
                      search_early_stop_mode='off', warmup_runs=0, timed_runs=1,
                      highfreq_rerank_mode='cpu', highfreq_rerank_algorithm='exact',
                      highfreq_rerank_shortlist=256, host_service_threads=4,
                      output_csv_path=str(root / f'{backend}.csv'))
        search['pq'] = dict(pq_dim=16, pivots_path=str(root / 'pq/pq.pivots.bin'),
                            compressed_path=str(root / 'pq/pq.codes.bin'), buffer_mode='double_slot_serial',
                            request_mode='batch_ring', request_batch_queries=16, query_worker_mode='coupled')
        if backend == 'rabitq':
            search.pop('pq')
            search.update(rabitq_artifact_path=str(root / 'rabitq/rabitq.b1.bin'),
                          rabitq_enable_highfreq_rerank=True, rabitq_query_worker_mode='coupled',
                          rabitq_request_mode='batch_ring', rabitq_request_batch_queries=16)
        path = root / f'{backend}-search.json'; path.write_text(json.dumps(search))
        execute(backend + '-search', [binary, f'--config={path}'])
        actual = [int(line.strip()) for line in (root / f'{backend}.csv').read_text().splitlines()]
        assert actual == ids, (backend, 'filtered self-query IDs differ', actual, ids)


def filtered_topk_oracle(path, vectors, queries, masks, k=5):
    """Exhaustive FP64 oracle, allowing only numerically tied ID substitutions."""
    assert vectors.ndim == queries.ndim == 2 and vectors.shape[1] == queries.shape[1]
    assert len(masks) == len(queries), 'one predicate mask is required per query'
    assert all(mask.dtype == np.bool_ and mask.shape == (len(vectors),) for mask in masks)
    exact_uint8 = vectors.dtype == np.uint8 and queries.dtype == np.uint8 and vectors.shape[1] * 255 ** 2 <= 2 ** 24
    with path.open() as source:
        rows = [[int(value) for value in row] for row in csv.reader(source)]
    assert len(rows) == len(queries), (path, 'query row count')
    for qi, (row, query, mask) in enumerate(zip(rows, queries, masks)):
        assert len(row) == k, (path, qi, 'result width', row)
        valid = [value for value in row if value not in (-1, 0xffffffff)]
        assert row[:len(valid)] == valid, (path, qi, 'interleaved sentinels')
        assert len(set(valid)) == len(valid), (path, qi, 'duplicate IDs')
        assert len(valid) == min(k, int(mask.sum())), (path, qi, 'missing matches', row)
        assert all(0 <= value < len(vectors) and mask[value] for value in valid), (path, qi, 'predicate')
        distances = np.square(vectors.astype(np.float64) - query.astype(np.float64)).sum(axis=1)
        expected = np.sort(distances[mask])[:k]
        actual = distances[valid]
        equal = np.array_equal(actual, expected) if exact_uint8 else np.allclose(actual, expected, rtol=2e-6, atol=2e-5)
        assert equal, (path, qi, row, actual, expected)
    return rows


def rerank_fixture(folder, base, vectors, queries, execute, binary):
    """An overlapping label graph with independent numeric/label residuals."""
    folder.mkdir()
    n, dim = vectors.shape
    ids = np.arange(n)
    memberships = [ids < 128, ids % 2 == 0, ids < 3, (ids >= 128) & (ids < 256)]
    rows = [[label for label, mask in enumerate(memberships) if mask[i]] for i in ids]
    indptr = np.concatenate(([0], np.cumsum([len(row) for row in rows]))).astype('<u8')
    labels = np.asarray([label for row in rows for label in row], dtype='<u4')
    (folder / 'base.spmat').write_bytes(struct.pack('<QQQ', n, 4, len(labels)) + indptr.tobytes() + labels.tobytes())
    (folder / 'identity.groups').write_text('0\n1\n2\n3\n')
    (folder / 'base_ranges.fbin').write_bytes(struct.pack('<II', n, 1) + ids.astype('<f4').tobytes())

    def atom(label, categorical=False):
        return dict(family='categorical' if categorical else 'string',
                    subtype='single' if categorical else 'exact',
                    predicates=[dict(field='caption_token', op='eq' if categorical else 'match',
                                     match_mode='' if categorical else 'exact', token_id=label,
                                     token_text=f'token{label}')])
    numeric = dict(family='range', subtype='width', predicates=[dict(
        column=0, field='original_width', op='between', lo=3.0, hi=5.0,
        inclusive_min=True, inclusive_max=True)])
    # OR deliberately overlaps the three posting IDs with a graph route, requiring dedup.
    cases = [('and', atom(1), memberships[0] & memberships[1]),
             ('or', atom(2), memberships[0] | memberships[2]),
             ('and', atom(2), memberships[0] & memberships[2]),
             ('and', atom(3), memberships[0] & memberships[3]),
             ('and', numeric, memberships[0] & (ids >= 3) & (ids <= 5))]
    manifests = []
    for i, (logic, right, _) in enumerate(cases):
        predicate = dict(op=logic, children=[atom(0, True), right])
        manifests.append(dict(family='hybrid', query_id=f'oracle-{i}', query_index=i,
                              hybrid_top_logic=logic, predicate_logic=logic.upper(),
                              predicate=predicate, predicates=[predicate]))
    (folder / 'queries.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in manifests))
    # JSON between is inclusive; native mode 3 uses an exclusive FP32 upper bound.
    upper = np.nextafter(np.float32(5.0), np.float32(np.inf))
    (folder / 'query_range.bin').write_bytes(struct.pack('<I', len(cases)) + b''.join(
        struct.pack('<Bff', 3 if i == 4 else 0, upper, 3.0) for i in range(len(cases))))
    query_path = folder / ('query' + base.suffix)
    if base.suffix == '.hvecs':
        query_path.write_bytes(b''.join(struct.pack('<I', dim) + row.astype('<f2').tobytes() for row in queries))
    else:
        dtype = 'u1' if base.suffix == '.u8bin' else '<f4'
        query_path.write_bytes(struct.pack('<II', *queries.shape) + queries.astype(dtype).tobytes())
    config = dict(base_vector_path=str(base), data_label_spmat_path=str(folder / 'base.spmat'),
                  group_path=str(folder / 'identity.groups'), row_limit=n, dim=dim, degree=16,
                  specificity_threshold=64, mode='build-graph', run_analyze=False,
                  graph_builder=dict(builder='cagra', cagra_build_worker_count=1),
                  graph_path=str(folder / 'graph.bin'), graph_file_path=str(folder / 'graph.serialized.bin'),
                  lowfreq_index_path=str(folder / 'lowfreq.bin'), seed_sidecar_path=str(folder / 'graph.seed'),
                  original_label_seed_sidecar_path=str(folder / 'graph.orig_label.seed'),
                  metadata_sidecar_path=str(folder / 'graph.metadata'))
    path = folder / 'build.json'; path.write_text(json.dumps(config))
    execute(folder.name + '-graph', [binary, f'--config={path}'])
    config.update(mode='benchmark', graph_path=str(folder / 'graph.serialized.bin'),
                  query_vector_path=str(query_path), query_manifest_jsonl_path=str(folder / 'queries.jsonl'),
                  base_range_fbin_path=str(folder / 'base_ranges.fbin'),
                  query_range_path=str(folder / 'query_range.bin'), range_posting_mode='off',
                  range_label_and_policy='auto', graph_storage_mode='memory',
                  topk=5, full_filter=True, per_step_full_filter=True, search_queue_size=256,
                  benchmark_queue_sizes='256', search_finish_count=256, search_early_stop_mode='off',
                  warmup_runs=1, timed_runs=2, highfreq_rerank_algorithm='exact',
                  highfreq_rerank_shortlist=256, highfreq_rerank_streaming=False,
                  host_service_threads=4, highfreq_rerank_threads=4,
                  lowfreq_full_filter_mode='post',
                  predicate_mode='gpu_exact_bias', predicate_bias_alpha=0.15,
                  predicate_storage='csr', quick_filter_bonus=1.0)
    return config, [mask for _, _, mask in cases]


def rerank_oracle(root, binary, uint8_vectors, float_vectors, execute):
    rng = np.random.default_rng(417)
    uint8_queries = rng.integers(0, 256, (5, uint8_vectors.shape[1]), dtype=np.uint8)
    float_queries = (rng.normal(size=(5, float_vectors.shape[1])) * 3.5).astype('<f4')
    half_vectors, half_queries = float_vectors.astype('<f2').astype('<f4'), float_queries.astype('<f2').astype('<f4')
    half_folder = root / 'rabitq-half'
    execute('rabitq-half-build', [sys.executable, '-m', 'scripts.quantize', 'rabitq',
            '--input', root / 'fractional.hvecs', '--output-dir', half_folder,
            '--n-lists', '4', '--bits', '1', '--training-rows', '32', '--chunk-rows', '113',
            '--iterations', '3', '--runtime', binary])
    fixtures = [('uint8', root / 'base.u8bin', uint8_vectors, uint8_queries),
                ('float32', root / 'fractional.fbin', float_vectors, float_queries),
                ('float16', root / 'fractional.hvecs', half_vectors, half_queries)]
    for dtype, base, vectors, queries in fixtures:
        # Non-self queries ensure exact scoring is exercised beyond a zero-distance hit.
        assert not any(np.any(np.all(vectors == q, axis=1)) for q in queries)
        config, masks = rerank_fixture(root / ('rerank-' + dtype), base, vectors, queries, execute, binary)
        for backend in (('pq', 'rabitq') if dtype == 'uint8' else ('rabitq',)):
            common = dict(config, backend=backend)
            if backend == 'pq':
                common['pq'] = dict(pq_dim=16, pivots_path=str(root / 'pq/pq.pivots.bin'),
                                    compressed_path=str(root / 'pq/pq.codes.bin'),
                                    buffer_mode='double_slot_serial', request_mode='batch_ring',
                                    request_batch_queries=16, query_worker_mode='coupled',
                                    lowfreq_scan_mode='resident', posting_streaming=False)
            else:
                artifact = {'uint8': 'rabitq/rabitq.b1.bin', 'float32': 'rabitq-batch/rabitq.b1.bin',
                            'float16': 'rabitq-half/rabitq.b1.bin'}[dtype]
                common.update(rabitq_artifact_path=str(root / artifact), rabitq_enable_highfreq_rerank=True,
                              rabitq_query_worker_mode='coupled', rabitq_request_mode='batch_ring',
                              rabitq_request_batch_queries=16, rabitq_lowfreq_gpu_posting=True,
                              rabitq_posting_streaming=False)
            variants = [('cpu', 'default'), ('gpu', 'default')]
            if dtype != 'uint8': variants.append(('gpu_post', 'default'))
            if dtype == 'float16': variants.extend([('cpu', 'mmap_fp16'), ('gpu_post', 'mmap_fp16')])
            for mode, storage in variants:
                name = f'oracle-{backend}-{dtype}-{mode}-{storage}'
                folder = root / name; folder.mkdir()
                options = dict(common, highfreq_rerank_mode=mode, rerank_vector_storage_mode=storage,
                               cpu_rerank_schedule='no_overlap' if mode == 'cpu' else 'native',
                               benchmark_output=str(folder / 'off.csv'))
                if dtype == 'float32' and mode == 'gpu_post': options['predicate_bias_alpha'] = 0.0
                if storage == 'mmap_fp16': options['rerank_vector_artifact_path'] = str(base)
                path = folder / 'config.json'; path.write_text(json.dumps(options))
                log = execute(name, ['env', '-u', 'GO_FANN_PERFORMANCE_TRACE', binary, f'--config={path}'])
                alpha = format(options['predicate_bias_alpha'], 'g')
                assert f'requested=gpu_exact_bias effective=gpu_exact_bias storage=csr alpha={alpha} ' in log, name
                results = list(folder.glob('off.*.results.csv')); assert len(results) == 1, name
                filtered_topk_oracle(results[0], vectors, queries, masks)
                if backend == 'rabitq' and mode == 'gpu':
                    words = vectors.shape[1] // 4 if dtype == 'uint8' else vectors.shape[1]
                    scalar = dtype
                    assert f'vector_words={words} vector_scalar={scalar}' in log, name
                if backend == 'pq' and mode == 'gpu':
                    assert f'vector_words={vectors.shape[1] // 4} ' in log, name
                if mode == 'gpu_post':
                    source = 'mmap_fp16' if storage == 'mmap_fp16' else 'resident_fp32'
                    assert 'float GPU rerank: transport=staged_fp32' in log and f'source={source}' in log, name
                    precision = 'fp16_exact' if storage == 'mmap_fp16' else 'fp32_exact'
                    assert f'source_values={precision} device_values=fp32' in log, name
                # One mixed graph/posting CPU pair per backend covers real profile call sites.
                if dtype == 'uint8' and mode == 'cpu':
                    profile_pair(folder, name, binary, path, results[0], execute)
            if dtype == 'uint8':
                bad = dict(common, highfreq_rerank_mode='gpu_post', cpu_rerank_schedule='native')
                if backend == 'pq':
                    # Reach the backend-mode guard without failing predicate admission first.
                    bad.update(predicate_mode='cpu', predicate_bias_alpha=0.0)
                    bad['pq'] = dict(bad['pq'], lowfreq_scan_mode='host')
                path = root / f'reject-{backend}-uint8-post.json'; path.write_text(json.dumps(bad))
                log = execute(f'reject-{backend}-uint8-post', [binary, f'--config={path}'], False)
                reason = 'requires --backend=rabitq' if backend == 'pq' else 'currently requires float vectors'
                assert reason in log, log[-2000:]
                if backend == 'rabitq':
                    bad = dict(common, highfreq_rerank_mode='gpu', cpu_rerank_schedule='native',
                               predicate_mode='cpu_simplified', predicate_bias_alpha=0.0)
                    path = root / 'reject-rabitq-simplified-inline.json'; path.write_text(json.dumps(bad))
                    log = execute('reject-rabitq-simplified-inline', [binary, f'--config={path}'], False)
                    assert 'exact high-frequency rerank' in log, log[-2000:]
            if dtype == 'float16':
                bad = dict(common, highfreq_rerank_mode='gpu', rerank_vector_storage_mode='mmap_fp16',
                           rerank_vector_artifact_path=str(base), cpu_rerank_schedule='native')
                path = root / 'reject-mmap-inline.json'; path.write_text(json.dumps(bad))
                log = execute('reject-mmap-inline', [binary, f'--config={path}'], False)
                assert 'custom rerank vector storage requires CPU rerank or mmap_fp16 GPU-post' in log, log[-2000:]


def profile_pair(folder, name, binary, config_path, off_results, execute):
    trace = folder / 'trace.jsonl'
    assert not trace.exists()
    execute(name + '-profile', ['env', f'GO_FANN_PERFORMANCE_TRACE={trace}', binary,
            f'--config={config_path}', f'--benchmark-output={folder / "on.csv"}'])
    on_results = list(folder.glob('on.*.results.csv')); assert len(on_results) == 1
    for suffix in ('results.csv', 'steps.csv', 'routes.csv'):
        off = Path(str(off_results).removesuffix('results.csv') + suffix)
        on = Path(str(on_results[0]).removesuffix('results.csv') + suffix)
        # Both runtimes expose route/step sidecars; require them rather than treating missing as parity.
        assert off.is_file() and on.is_file(), (name, suffix)
        assert off.read_bytes() == on.read_bytes(), (name, 'profiling changed', suffix)
    timed = list(folder.glob('on.*.timed_runs.csv')); assert len(timed) == 1
    repo = Path(__file__).resolve().parents[2]
    execute(name + '-convert', [sys.executable, repo / 'tools/reproduce/profile.py',
            '--input', 'hybrid', trace, timed[0], '--output', folder / 'window.csv'])
    records = [json.loads(line) for line in trace.read_text().splitlines()]
    assert sum(row['kind'] == 'complete' for row in records) == 2
    assert all(row['phase'] == 'timed' for row in records)
    spans = [row for row in records if row['kind'] == 'span' and row['end_ns'] > row['start_ns']]
    assert any(row['name'] == 'graph_launch_to_complete' for row in spans)
    assert any(row['name'] == 'lowfreq_kernel_launch_to_complete' for row in spans)
    for run in (0, 1):
        graph = [row for row in spans if row['run'] == run and row['name'] in
                 ('graph_launch_to_complete', 'lowfreq_kernel_launch_to_complete')]
        cpu = [row for row in spans if row['run'] == run and row['name'] in
               ('highfreq_cpu_rerank_batch', 'lowfreq_cpu_finalize')]
        assert graph and cpu, (name, run, 'missing GPU/CPU stage')
        assert all(min(a['end_ns'], b['end_ns']) <= max(a['start_ns'], b['start_ns'])
                   for a in graph for b in cpu), (name, run, 'serial CPU overlaps graph')
    with (folder / 'window.csv').open() as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 2
    for row in rows:
        phases = sum(float(value) for key, value in row.items() if key.endswith('_ms') and key != 'wall_ms')
        assert abs(phases - float(row['wall_ms'])) < 1e-6


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=2)
    p.add_argument('--binary', type=Path, default=Path('src/go-fann'))
    a = p.parse_args()
    run(a.root.resolve(), a.gpu, a.binary.resolve())
