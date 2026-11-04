"""Small file-based checks for the consolidated preparation pipeline."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import contextlib
import io
from unittest.mock import patch
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

from scripts.laion import assemble, base, hybrid_gt, query, range_gt, range_index, string_gt, string_index
from scripts.laion.categorical_gt import write_dist_fbin, write_gt_ibin, write_hvecs


class LaionPipelineTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    @staticmethod
    def write_jsonl(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))

    @staticmethod
    def run_main(module, arguments):
        with patch('sys.argv', [module.__name__, *map(str, arguments)]), contextlib.redirect_stdout(io.StringIO()):
            module.main()

    def test_npy_prefix_numeric_shards_and_partial_last_shard(self):
        images = self.root / 'images'
        images.mkdir()
        arrays = []
        for shard in range(12):
            array = np.full((1 if shard == 8 else 3, 2), shard, dtype='<f2')
            array[:, 1] += np.arange(len(array)) / 4
            np.save(images / f'img_emb_{shard}.npy', array)
            arrays.append(array)
        output = self.root / 'base.hvecs'
        expected = np.concatenate(arrays)[:30]
        batch_lengths = []
        writer = base.write_hvecs_batch
        def record_batch(handle, matrix):
            batch_lengths.append(len(matrix))
            writer(handle, matrix)
        with patch.object(base, 'write_hvecs_batch', side_effect=record_batch):
            self.run_main(base, ['export-vectors', '--input', images, '--output', output,
                                 '--rows', 30, '--chunk-rows', 2, '--dim', 2])
        expected_records = b''.join(np.asarray([2], '<u4').tobytes() + row.tobytes() for row in expected)
        self.assertEqual(output.read_bytes(), expected_records)
        self.assertLessEqual(max(batch_lengths), 2)
        self.assertEqual(sum(batch_lengths), 30)

    def test_npy_prefix_single_fp32_array_and_query_row_ids(self):
        source = self.root / 'all.npy'
        matrix = np.asfortranarray([[0.125, -0.0], [1.0002, 2.75], [-3.2, 4.5], [9, 10]], dtype='>f4')
        np.save(source, matrix)
        output = self.root / 'base.hvecs'
        base.export_npy_prefix(source, output, 3, 2, 2)
        expected = matrix[:3].astype('<f2')
        self.assertEqual(output.read_bytes(), b''.join(np.asarray([2], '<u4').tobytes() + row.tobytes() for row in expected))
        records = base.hvecs_records(output, base.inspect_hvecs(output))
        np.testing.assert_array_equal(records['vec'][[2, 0]], expected[[2, 0]])
        with self.assertRaises(FileExistsError):
            base.export_npy_prefix(source, output, 2, 2, 2)
        self.assertEqual(len(output.read_bytes()), 3 * 8)

    def test_npy_prefix_agrees_with_existing_row_map_materializer(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        images = self.root / 'images'
        images.mkdir()
        np.save(images / 'img_emb_0.npy', np.asarray([[1, 2], [3, 4]], '<f2'))
        np.save(images / 'img_emb_1.npy', np.asarray([[5, 6], [7, 8], [9, 10]], '<f2'))
        row_map = self.root / 'row_map.parquet'
        pq.write_table(pa.table({'row_id': [0, 1, 2, 3], 'metadata_shard_id': [0, 0, 1, 1],
                                 'shard_row_id': [0, 1, 0, 1]}), row_map)
        old, new = self.root / 'old.hvecs', self.root / 'new.hvecs'
        base.write_base_hvecs(old, row_map, self.root, 4, 2, False)
        base.export_npy_prefix(images, new, 4, 2, 2)
        self.assertEqual(old.read_bytes(), new.read_bytes())

    def test_npy_prefix_invalid_shape_dtype_limit_and_truncation(self):
        source, output = self.root / 'bad.npy', self.root / 'out/base.hvecs'
        for array in (np.ones(4, '<f2'), np.ones((4, 3), '<f2'), np.ones((4, 2), '<i4'),
                      np.ones((0, 2), '<f2'), np.ones((4, 2), object)):
            with self.subTest(dtype=array.dtype, shape=array.shape):
                np.save(source, array)
                with self.assertRaises(ValueError):
                    base.export_npy_prefix(source, output, 3, 2, 2)
                self.assertFalse(output.parent.exists())
        np.save(source, np.ones((4, 2), '<f2'))
        for rows, chunk, dim in ((5, 2, 2), (0, 2, 2), (-1, 2, 2), (3, 0, 2), (3, 2, 0)):
            with self.subTest(rows=rows, chunk=chunk, dim=dim), self.assertRaises(ValueError):
                base.export_npy_prefix(source, output, rows, chunk, dim)
        payload = source.read_bytes()
        for invalid in (payload[:-1], payload + b'x'):
            source.write_bytes(invalid)
            with self.assertRaises(ValueError):
                base.export_npy_prefix(source, output, 3, 2, 2)
        self.assertFalse(output.parent.exists())

    def test_npy_prefix_missing_or_duplicate_numeric_shards(self):
        images = self.root / 'images'
        images.mkdir()
        output = self.root / 'out/base.hvecs'
        np.save(images / 'img_emb_0.npy', np.ones((2, 2), '<f2'))
        np.save(images / 'img_emb_2.npy', np.ones((2, 2), '<f2'))
        with self.assertRaisesRegex(ValueError, 'missing image shard'):
            base.export_npy_prefix(images, output, 3, 2, 2)
        np.save(images / 'img_emb_00.npy', np.ones((2, 2), '<f2'))
        with self.assertRaisesRegex(ValueError, 'duplicate image shard'):
            base.export_npy_prefix(images, output, 3, 2, 2)
        self.assertFalse(output.parent.exists())

    def test_npy_prefix_failure_removes_partial_output(self):
        source, output = self.root / 'source.npy', self.root / 'out/base.hvecs'
        np.save(source, np.asarray([[1, 2], [3, 4], [np.nan, 5]], '<f4'))
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            base.export_npy_prefix(source, output, 3, 1, 2)
        self.assertEqual(list(output.parent.iterdir()), [])
        output.write_bytes(b'preserve existing')
        with self.assertRaises(FileExistsError):
            base.export_npy_prefix(source, output, 3, 1, 2)
        self.assertEqual(output.read_bytes(), b'preserve existing')

    def test_required_inputs_fail_before_output_creation(self):
        for module in (base, query, range_gt, string_gt):
            with self.subTest(module=module.__name__), patch('sys.argv', [module.__name__]), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    module.parse_args()
                self.assertEqual(error.exception.code, 2)
        self.assertEqual(list(self.root.iterdir()), [])

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'torch fixture dependency')
    def test_numeric_hierarchy_materialization_gt_and_assembly(self):
        vectors = np.asarray([[0, 0], [2, 0], [5, 0], [9, 0]], np.float16)
        hvecs = self.root / 'base.hvecs'
        write_hvecs(hvecs, vectors, False)
        numeric = np.asarray([[10, 40, 99], [20, 30, 98], [30, 20, 97], [40, 10, 96]], '<f4')
        source = self.root / 'source.fbin'
        source.write_bytes(np.asarray(numeric.shape, '<u4').tobytes() + numeric.tobytes())
        hierarchy = self.root / 'range'
        self.run_main(range_index, ['hierarchy', '--source-base-range-fbin', source, '--output-root', hierarchy,
                                    '--levels', '2', '--chunk-rows', '2'])
        np.testing.assert_array_equal(assemble.read_matrix(hierarchy / 'base_range.fbin', '<f4'), numeric[:, :2])
        self.assertTrue((hierarchy / 'base.virtual.spmat').is_file())
        self.assertTrue((hierarchy / 'manifest.json').is_file())
        # Source vector order is deliberately non-prefix, non-sorted.
        records = [dict(family='range', subtype='width', query_id=str(row), query_vector_id=row,
                        candidate_count=3, predicates=[dict(column=0, op='gt', lo=10)]) for row in (3, 1)]
        manifest = self.root / 'source.jsonl'
        self.write_jsonl(manifest, records)
        shards = self.root / 'shards'
        runtime = shards / 'width/runtime'
        self.run_main(base, ['--runtime-root', self.root, '--query-manifest', manifest, '--output-root', runtime,
                             '--base-vector-path', hvecs, '--data-label-spmat', hierarchy / 'base.virtual.spmat',
                             '--base-range-fbin', hierarchy / 'base_range.fbin', '--include-range',
                             '--range-label-manifest', hierarchy / 'manifest.json'])
        np.testing.assert_array_equal(base.hvecs_records(runtime / 'query.hvecs', base.inspect_hvecs(runtime / 'query.hvecs'))['vec'], vectors[[3, 1]])
        materialized = assemble.load_jsonl(runtime / 'query_manifest.flashanns.jsonl')
        self.assertEqual([row['query_vector_id'] for row in materialized], [3, 1])
        self.assertEqual([row['flashanns_query_index'] for row in materialized], [0, 1])
        self.assertNotIn('configs', json.loads((runtime / 'materialize_summary.json').read_text())['artifacts'])
        self.assertFalse(list(runtime.glob('*config*.json')))
        gt = shards / 'width/gt'
        arguments = ['--base-hvecs', hvecs, '--query-hvecs', runtime / 'query.hvecs',
                     '--base-range-fbin', hierarchy / 'base_range.fbin', '--query-range-bin', runtime / 'query_range.bin',
                     '--index-root', self.root / 'value-index', '--topk', '2', '--recall-k', '2', '--device', 'cpu']
        missing_result_output = self.root / 'bad-results-output'
        with self.assertRaises(FileNotFoundError):
            self.run_main(range_gt, [*arguments, '--output-root', missing_result_output,
                                    '--search-results-csv', self.root / 'missing.csv'])
        self.assertFalse(missing_result_output.exists())
        self.run_main(range_gt, [*arguments, '--output-root', gt])
        ids = assemble.read_matrix(gt / 'gt.range.top2.ibin', '<u4')
        np.testing.assert_array_equal(ids, [[3, 2], [1, 2]])
        summary = json.loads((gt / 'gt.range.summary.json').read_text())
        self.assertIsNone(summary['recall'])
        self.assertIsNone(summary['source']['search_results_csv'])
        self.assertEqual(summary['outputs']['recall_k_gt'], summary['outputs']['topk_gt'])
        csv = self.root / 'results.csv'
        np.savetxt(csv, ids, fmt='%d', delimiter=',')
        with_recall = self.root / 'gt-recall'
        self.run_main(range_gt, [*arguments, '--output-root', with_recall, '--search-results-csv', csv])
        recall = json.loads((with_recall / 'gt.range.summary.json').read_text())['recall']
        self.assertEqual(recall['recall_valid_gt_denominator'], 1.0)
        self.run_main(range_gt, [*arguments, '--output-root', self.root / 'gt-skipped',
                                '--search-results-csv', self.root / 'missing.csv', '--skip-recall'])
        output = self.root / 'assembled'
        assemble.combine(Namespace(all_query_manifest=manifest, by_subtype_root=shards, output_root=output,
                                   gt_tag='gt.range', topk=2, num_points=4, overwrite=False))
        self.assertEqual((output / 'gt.range.top2.ibin').read_bytes(), (gt / 'gt.range.top2.ibin').read_bytes())

    @unittest.skipUnless(importlib.util.find_spec('pyarrow'), 'parquet fixture dependency')
    def test_selected_query_output_and_token_postings_feed_exact_gt(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        vectors = np.asarray([[0, 0], [2, 0], [5, 0], [9, 0]], np.float16)
        hvecs = self.root / 'base.hvecs'
        write_hvecs(hvecs, vectors, False)
        tokens = self.root / 'token_map.parquet'
        pq.write_table(pa.table({'token_id': [0, 1], 'token': ['cat', 'dog']}), tokens)
        labels = self.root / 'base.spmat'
        base.write_spmat(labels, [[0], [1], [0], [1]], 2, False)
        postings = self.root / 'token_postings.bin'
        self.run_main(string_index, ['--base-spmat', labels, '--token-map', tokens, '--output-path', postings,
                                     '--max-row-id-exclusive', '4', '--spmat-row-batch', '2'])
        numeric = self.root / 'base_range.fbin'
        numeric.write_bytes(np.asarray([4, 2], '<u4').tobytes() + np.zeros((4, 2), '<f4').tobytes())
        records = [dict(family='string', subtype=subtype, query_id=str(i), query_vector_id=row,
                        predicates=[dict(token_id=token, token_text=text)])
                   for i, (subtype, row, token, text) in enumerate([('exact', 3, 1, 'dog'), ('exact', 0, 0, 'cat')])]
        manifest = self.root / 'selected-source.jsonl'
        self.write_jsonl(manifest, records)
        output = self.root / 'exact'
        self.run_main(query, ['--query-manifest', manifest, '--token-map', tokens, '--base-hvecs', hvecs,
                             '--base-range-fbin', numeric, '--output-root', output, '--subtypes', 'exact'])
        selected_path = output / 'query_manifest.jsonl'
        selected = query.load_selected_queries(selected_path, {'string'}, {'exact'}, 0, 0)
        self.assertEqual([q.source['query_vector_id'] for q in selected.queries], [3, 0])
        actual = base.hvecs_records(output / 'query.hvecs', base.inspect_hvecs(output / 'query.hvecs'))['vec']
        np.testing.assert_array_equal(actual, vectors[[3, 0]])
        provider = string_gt.BaseVectorProvider(base.hvecs_records(hvecs, base.inspect_hvecs(hvecs)))
        lookup = string_index.TokenIdPostingsBinaryLookup(postings, 4)
        for i, token in enumerate((1, 0)):
            candidates = lookup.rows_for_token_ids_union(np.asarray([token], np.uint32))
            ids, distances = string_gt.cpu_topk_for_query_vectors(provider, actual[i:i+1].astype(np.float32), candidates, 2, 2)
            np.testing.assert_array_equal(ids[0], [[3, 1], [0, 2]][i])
            self.assertEqual(distances[0, 0], 0)
        # Renaming the output file preserves the ngram payload and wire format.
        reference = self.root / 'reference.bin'
        string_index.write_ngram_index(reference, tokens, 3, 262144, False)
        self.assertEqual((output / 'token_ngram_index.bin').read_bytes(), reference.read_bytes())

    def test_filtered_gt_vectors_use_input_positions_not_prefix(self):
        vectors = np.asarray([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=np.float16)
        rows = [
            {'family': 'string', 'subtype': subtype, 'query_vector_id': 100 + i, 'query_id': str(i)}
            for i, subtype in enumerate(['fuzzy', 'exact', 'fuzzy', 'exact'])
        ]
        manifest = self.root / 'queries.jsonl'
        self.write_jsonl(manifest, rows)
        source = self.root / 'query.hvecs'
        write_hvecs(source, vectors, False)
        selected = query.load_selected_queries(manifest, {'string'}, {'exact'}, 0, 0)
        self.assertEqual(selected.source_indices, [1, 3])
        self.assertEqual([q.source['flashanns_query_index'] for q in selected.queries], [0, 1])
        target = self.root / 'selected.hvecs'
        string_gt.write_query_hvecs_subset(source, target, selected.source_indices, False)
        actual = base.hvecs_records(target, base.inspect_hvecs(target))['vec']
        np.testing.assert_array_equal(actual, vectors[[1, 3]])
        with self.assertRaises(ValueError):
            string_gt.write_query_hvecs_subset(source, self.root / 'bad.hvecs', [4], False)
        self.assertFalse((self.root / 'bad.hvecs').exists())

    def test_full_count_subset_writes_output_and_preserves_reordered_rows(self):
        source = self.root / 'query.hvecs'
        vectors = np.asarray([[0, 1], [2, 3]], dtype=np.float16)
        write_hvecs(source, vectors, False)
        target = self.root / 'selected.hvecs'
        string_gt.write_query_hvecs_subset(source, target, [1, 0], False)
        np.testing.assert_array_equal(base.hvecs_records(target, base.inspect_hvecs(target))['vec'], vectors[[1, 0]])
        with self.assertRaises(FileExistsError):
            string_gt.write_query_hvecs_subset(source, target, [0, 1], False)

    def test_split_and_hybrid_combine_restore_interleaved_query_ids(self):
        rows = [
            {'query_id': 'a0', 'subtype': 'a', 'flashanns_query_index': 91},
            {'query_id': 'b0', 'subtype': 'b', 'flashanns_query_index': 47},
            {'query_id': 'a1', 'subtype': 'a', 'flashanns_query_index': 83},
        ]
        manifest = self.root / 'queries.jsonl'
        self.write_jsonl(manifest, rows)
        vectors = np.asarray([[1, 2], [3, 4], [5, 6]], dtype=np.float16)
        hvecs = self.root / 'query.hvecs'
        write_hvecs(hvecs, vectors, False)
        split_root = self.root / 'split'
        assemble.split(Namespace(query_manifest=manifest, query_hvecs=hvecs, output_root=split_root, overwrite=False))
        split_rows = assemble.load_jsonl(split_root / 'a/query_manifest.selected.jsonl')
        self.assertEqual([r['flashanns_parent_query_index'] for r in split_rows], [91, 83])
        a_hvecs = split_root / 'a/query.hvecs'
        np.testing.assert_array_equal(base.hvecs_records(a_hvecs, base.inspect_hvecs(a_hvecs))['vec'], vectors[[0, 2]])
        for subtype, order, ids in [('a', [2, 0], [[5, 6], [1, 2]]), ('b', [1], [[3, 4]])]:
            gt_root = split_root / subtype / 'gt'
            gt_root.mkdir()
            write_gt_ibin(gt_root / 'gt.hybrid.top2.ibin', np.asarray(ids, np.uint32), False)
            write_dist_fbin(gt_root / 'gt.hybrid.top2.dist.fbin', np.asarray([[0., 1.]] * len(order), np.float32), False)
            self.write_jsonl(gt_root / 'gt.hybrid.rows.jsonl', [dict(rows[i], candidate_count=7 + i) for i in order])
        output = self.root / 'combined'
        args = Namespace(all_query_manifest=manifest, by_subtype_root=split_root, output_root=output, topk=2, overwrite=False)
        assemble.combine_hybrid(args)
        np.testing.assert_array_equal(assemble.read_matrix(output / 'gt.hybrid.top2.ibin', '<u4'), [[1, 2], [3, 4], [5, 6]])
        combined_rows = assemble.load_jsonl(output / 'gt.hybrid.rows.jsonl')
        self.assertEqual([r['flashanns_parent_query_index'] for r in combined_rows], [91, 47, 83])
        self.assertEqual([r['candidate_count'] for r in combined_rows], [7, 8, 9])
        self.write_jsonl(split_root / 'a/gt/gt.hybrid.rows.jsonl', [dict(rows[0]), dict(rows[0])])
        with self.assertRaisesRegex(ValueError, 'unique'):
            assemble.combine_hybrid(args)

    def test_matrix_reader_rejects_truncated_and_trailing_payload(self):
        path = self.root / 'matrix.ibin'
        header = np.asarray([1, 2], dtype='<u4').tobytes()
        for values in ([3], [3, 4, 5]):
            path.write_bytes(header + np.asarray(values, dtype='<u4').tobytes())
            with self.assertRaisesRegex(ValueError, 'size mismatch'):
                assemble.read_matrix(path, '<u4')

    def test_range_gt_rejects_sentinels_out_of_range_and_unsorted_distances(self):
        for ids, distances, message in [
            ([[1, 0xFFFFFFFF]], [[0., 1.]], 'sentinel'),
            ([[1, 20]], [[0., 1.]], 'outside'),
            ([[1, 2]], [[1., 0.]], 'nondecreasing'),
        ]:
            with self.assertRaisesRegex(ValueError, message):
                assemble.validate_gt(np.asarray(ids, np.uint32), np.asarray(distances, np.float32), 20)

    def test_numpy_token_postings_and_exact_topk_need_no_parquet(self):
        postings = self.root / 'postings'
        postings.mkdir()
        np.save(postings / 'token_0.npy', np.asarray([0, 2, 3], np.uint32))
        np.save(postings / 'token_1.npy', np.asarray([1, 2], np.uint32))
        self.assertEqual(string_index.detect_token_postings_layout(postings), 'token_id_npy')
        lookup = string_index.TokenIdPostingsNpyLookup(postings, 4)
        rows = lookup.rows_for_token_ids_union(np.asarray([0, 1], np.uint32))
        np.testing.assert_array_equal(rows, [0, 1, 2, 3])
        vectors = np.asarray([[0, 0], [1, 2], [3, 0], [6, 0]], dtype=np.float16)
        hvecs = self.root / 'base.hvecs'
        write_hvecs(hvecs, vectors, False)
        provider = string_gt.BaseVectorProvider(base.hvecs_records(hvecs, base.inspect_hvecs(hvecs)))
        queries = np.asarray([[1, 0], [5, 0]], np.float32)
        ids, distances = string_gt.cpu_topk_for_query_vectors(provider, queries, rows, 3, 2)
        oracle = np.sum((queries[:, None] - vectors.astype(np.float32)[None, :]) ** 2, axis=2)
        expected = np.argsort(oracle, axis=1, kind='stable')[:, :3]
        np.testing.assert_array_equal(ids, expected)
        np.testing.assert_array_equal(distances, np.take_along_axis(oracle, expected, axis=1))

    @unittest.skipUnless(importlib.util.find_spec('pyarrow') and importlib.util.find_spec('duckdb'), 'parquet fixture dependencies')
    def test_embedding_shards_follow_manifest_order_and_reject_invalid_rows(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        first = np.asarray([[1, 2], [3, 4]], np.float16)
        second = np.asarray([[5, 6], [7, 8]], np.float16)
        np.save(self.root / 'img_emb_0.npy', first)
        np.save(self.root / 'img_emb_1.npy', second)
        row_map = self.root / 'rows.parquet'
        pq.write_table(pa.table({'row_id': [8, 2, 9], 'shard_id': [1, 0, 1], 'shard_row_id': [1, 0, 7]}), row_map)
        lookup = base.RowMapLookup(row_map, workers=1, batch_size=2)
        self.addCleanup(lookup.con.close)
        queries = [base.RuntimeQuery({'query_vector_id': row}, [], []) for row in [8, 2, 8]]
        output = self.root / 'query.hvecs'
        query.write_query_hvecs(output, queries, lookup, self.root, None, 2, False)
        np.testing.assert_array_equal(base.hvecs_records(output, base.inspect_hvecs(output))['vec'], [second[1], first[0], second[1]])
        with self.assertRaisesRegex(IndexError, 'outside embedding shard'):
            query.write_query_hvecs(self.root / 'bad.hvecs', [base.RuntimeQuery({'query_vector_id': 9}, [], [])], lookup, self.root, None, 2, False)

    @unittest.skipUnless(importlib.util.find_spec('pyarrow') and importlib.util.find_spec('torch'), 'parquet and torch fixture dependencies')
    def test_string_ngram_and_hybrid_gt_file_pipeline_cpu_oracle(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        import torch

        token_map = self.root / 'tokens.parquet'
        pq.write_table(pa.table({'token_id': [0, 1, 2], 'token': ['Cat!', 'cats', 'dog']}), token_map)
        ngram = self.root / 'ngram.bin'
        labels, postings = self.root / 'base.spmat', self.root / 'postings.bin'
        base.write_spmat(labels, [[0], [1], [0, 1], [0], [2]], 3, False)
        argv = ['--base-spmat', labels, '--token-map', token_map, '--output-path', postings,
                '--ngram-output', ngram, '--max-row-id-exclusive', '5', '--spmat-row-batch', '2']
        self.run_main(string_index, argv)
        info = json.loads(postings.with_suffix('.bin.summary.json').read_text())['ngram_index']
        payload = ngram.read_bytes()
        self.assertEqual(hashlib.sha256(payload[string_index.NGRAM_HEADER.size:]).hexdigest(), info['checksum_sha256'])
        index = string_index.load_token_text_index_from_ngram_binary(ngram)
        np.testing.assert_array_equal(string_index.fuzzy_matching_token_ids('prefix', 'cat', index), [0, 1])
        # Token rows overlap at row 2; OR must admit this row only once.
        before = postings.read_bytes()
        with self.assertRaises(FileExistsError):
            self.run_main(string_index, argv)
        self.assertEqual(postings.read_bytes(), before)
        self.assertEqual(ngram.read_bytes(), payload)
        with self.assertRaisesRegex(ValueError, 'must differ'):
            self.run_main(string_index, ['--base-spmat', labels, '--token-map', token_map,
                '--output-path', self.root / 'new.bin', '--ngram-output', token_map,
                '--max-row-id-exclusive', '5', '--overwrite'])
        self.assertFalse((self.root / 'new.bin').exists())
        operator = self.root / 'runtime-operator.bin'
        operator.write_bytes(b'FANNSOPLEXIDX'.ljust(16, b'\0') + payload[16:])
        with self.assertRaisesRegex(ValueError, 'runtime operator index'):
            string_index.load_token_text_index_from_ngram_binary(operator)
        self.assertEqual(ngram.read_bytes(), payload)
        ranges = self.root / 'range.fbin'
        ranges.write_bytes(np.asarray([5, 1], '<u4').tobytes() + np.asarray([10, 20, 30, 40, 50], '<f4').tobytes())
        provider = hybrid_gt.RowSetProvider(token_postings_bin=postings, token_ngram_index=ngram, range_fbin=ranges, row_scope=5, load_range_values=False, rowset_cache_bytes=1024)
        fuzzy = {'family': 'string', 'subtype': 'fuzzy', 'predicates': [{'match_mode': 'prefix', 'pattern': 'cat'}]}
        range_term = {'family': 'range', 'subtype': 'width', 'predicates': [{'column': 0, 'op': 'gt', 'lo': 30}]}
        vectors = np.asarray([[0, 0], [1, 2], [3, 0], [6, 0], [8, 0]], np.float16)
        hvecs = self.root / 'base.hvecs'
        write_hvecs(hvecs, vectors, False)
        base_provider = string_gt.BaseVectorProvider(base.hvecs_records(hvecs, base.inspect_hvecs(hvecs)))
        query_vectors = np.asarray([[2, 0], [7, 1]], np.float32)
        tensors = torch.from_numpy(query_vectors)
        for op, expected_rows in [('and', [3]), ('or', [0, 1, 2, 3, 4])]:
            parts, count = hybrid_gt.candidate_row_parts({'predicate': {'op': op, 'children': [fuzzy, range_term]}}, provider)
            self.assertEqual(count, len(expected_rows))
            np.testing.assert_array_equal(np.sort(np.concatenate(parts)), expected_rows)
            ids, distances = hybrid_gt.topk_for_candidate_parts(base_provider=base_provider, query_gpu=tensors, query_norm=(tensors * tensors).sum(1), candidate_parts=parts, topk=1, device=torch.device('cpu'), row_batch_size=2)
            oracle = np.sum((query_vectors[:, None] - vectors[expected_rows].astype(np.float32)[None, :]) ** 2, axis=2)
            ranks = np.argmin(oracle, axis=1)
            np.testing.assert_array_equal(ids[:, 0], np.asarray(expected_rows)[ranks])
            np.testing.assert_allclose(distances[:, 0], oracle[np.arange(2), ranks])


if __name__ == '__main__':
    unittest.main()
