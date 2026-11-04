#!/usr/bin/env python3
"""Integration guards; semantic DNF/admission cases live in posting CUDA/host fixtures."""
from pathlib import Path
import unittest

SOURCE = (Path(__file__).resolve().parents[2] / "src" / 'search/pq/backend_pq.cu').read_text()


def function(name):
    start = SOURCE.index(name + '(')
    # Ignore brace-initialized default arguments in the signature.
    body = SOURCE.index('\n{', start) + 1
    depth = 1
    end = body + 1
    while depth:
        depth += (SOURCE[end] == '{') - (SOURCE[end] == '}')
        end += 1
    return SOURCE[start:end]


class PqPostingPredicateContract(unittest.TestCase):
    def test_filter_precedes_score_and_shortlist_on_original_sample(self):
        scan = function('run_lowfreq_pq_scan_one_block')
        match = scan.index('gpu_predicate::matches(posting_predicate, query_idx, sample_id)')
        self.assertLess(match, scan.index('const uint32_t code = d_data['))
        self.assertLess(match, scan.index('posting_shortlist_heap::insert'))
        self.assertIn(': d_candidate_ids[begin + idx]', scan)
        self.assertIn('pq_resident_posting::candidate(resident, query_idx, begin + idx)', scan)
        self.assertNotIn('priority(', scan)

    def test_score_first_admits_only_after_valid_heap_and_predicate(self):
        scan = function('run_lowfreq_pq_scan_one_block')
        score = scan.index('const uint32_t code = d_data[')
        guard = scan.index('if (score_first)', score)
        gate = scan.index('posting_shortlist_heap::can_enter', guard)
        predicate = scan.index('gpu_predicate::matches', gate)
        insert = scan.index('posting_shortlist_heap::insert', predicate)
        self.assertLess(score, gate)
        self.assertLess(gate, predicate)
        self.assertLess(predicate, insert)
        self.assertIn('if (!score_first && gpu_predicate::enabled', scan)

    def test_resident_metadata_uploaded_at_init_and_host_stream_preserved(self):
        init = function('init_lowfreq_context')
        for field in ('index.label_offset', 'index.label_size', 'index.sample_ids',
                      'index.row_filter_mask', 'runtime.query_group_ids', 'runtime.query_filter_bits'):
            self.assertIn('upload(' + field, init)
        run = function('run_lowfreq_pq_search_with_context')
        self.assertIn('if (!ctx.resident.active)', run)
        self.assertIn('ctx.resident, options.posting_predicate_order == "score_first"', run)
        self.assertNotIn('resident_storage->upload', run)

    def test_posting_plan_does_not_borrow_graph_entailment_or_bias(self):
        prepare = function('prepare_pq_posting_predicate')
        self.assertIn('route_predicate::compile(runtime, route_predicate::Producer::LabelPosting)', prepare)
        self.assertNotIn('route_predicate::compile(runtime)', prepare)
        self.assertNotIn('v.alpha =', prepare)
        self.assertNotIn('shortlist_verified =', prepare)
        self.assertIn('runtime.base_range_values', prepare)

    def test_graph_and_posting_acquire_one_immutable_csr(self):
        for name in ('prepare_pq_predicate', 'prepare_pq_posting_predicate'):
            self.assertIn('acquire_pq_predicate_csr(runtime, options, csr_cache, index_reused)', function(name))
        acquire = function('acquire_pq_predicate_csr')
        self.assertIn('validate_csr_offsets', acquire)
        self.assertIn('csr_cache.find(key)', acquire)
        self.assertIn('cudaGetDevice', acquire)

    def test_low_only_queries_and_context_lifetime(self):
        init = function('init_benchmark_query_contexts')
        self.assertLess(init.index('init_lowfreq_context('), init.index('if (!has_potential_high)'))
        self.assertIn('prepare_pq_posting_predicate(runtime, options, immutable_state.predicate_csr_cache)',
                      function('init_lowfreq_context'))
        self.assertIn('ctx = PqLowfreqSearchContext{};', function('cleanup_lowfreq_context'))

    def test_legacy_and_host_prefilter_are_explicit(self):
        prepare = function('prepare_pq_posting_predicate')
        for guard in ('options.query_worker_mode == "coupled"',
                      'options.graph_transport == "mailbox"', '!options.latency_persistent_queue',
                      'if (!supported || host_prefilter)'):
            self.assertIn(guard, prepare)
        self.assertNotIn('runtime.graph_on_disk', prepare)
        self.assertIn('mode=legacy reason=', prepare)
        self.assertIn('gpu_predicate::View posting_predicate = {}',
                      function('run_lowfreq_pq_scan_one_block'))

    def test_cpu_bypass_is_route_scoped_and_keeps_sample_bounds(self):
        rerank = function('lowfreq_host_heuristic_rerank_range')
        bounds = rerank.index('sample_id == UINT32_MAX || sample_id >= runtime.base_vectors.rows')
        coverage = rerank.index('predicate_verified(verified_routes, query_indices[q])')
        predicate = rerank.index('!sample_matches_lowfreq_full_labels')
        self.assertLess(bounds, coverage)
        self.assertLess(coverage, predicate)
        self.assertIn('if (filter_full_labels &&', rerank)
        run = function('run_lowfreq_pq_search_with_context')
        # Both the ready-batch and synchronized paths pass this provider's mask.
        self.assertEqual(run.count('ctx.posting_predicate ? &ctx.posting_predicate->verified_routes : nullptr'), 2)
        self.assertIn('cpu_final_validation=unverified_routes_only', SOURCE)

    def test_stream_publishes_before_cpu_consumes_and_drains(self):
        kernel = function('lowfreq_pq_scan_kernel')
        self.assertLess(kernel.index('mapped_result['), kernel.index('__threadfence_system()'))
        self.assertLess(kernel.index('__threadfence_system()'), kernel.index('mapped_ready + qid'))
        run = function('run_lowfreq_pq_search_with_context')
        self.assertIn('__atomic_load_n(ctx.streaming->host_ready + pos, __ATOMIC_ACQUIRE)', run)
        self.assertLess(run.index('consumer = std::async'), run.index('cudaDeviceSynchronize(),'))
        self.assertIn('if (consumer.valid()) consumer.wait();', run)

    def test_launch_and_cpu_safety_validation(self):
        run = function('run_lowfreq_pq_search_with_context')
        self.assertIn('ctx.posting_predicate ? ctx.posting_predicate->view : gpu_predicate::View{}', run)
        self.assertIn('defaults::lowfreq_full_filter_mode_is_post(options.lowfreq_full_filter_mode) ||\n'
                      '        ctx.posting_predicate != nullptr', run)
        self.assertIn('!sample_matches_lowfreq_full_labels(runtime, query_indices[q], sample)', run)
        # Existing host prefilter remains prior to appending/uploading candidates.
        self.assertLess(run.index('if (pre_filter_full_labels &&'), run.index('candidate_ids.push_back(sample_id)'))


if __name__ == '__main__':
    unittest.main()
