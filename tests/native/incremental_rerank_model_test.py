"""Executable specification only: this model is not wired into native search."""
import math
import unittest


class QueryState:
    def __init__(self, shortlist_size, topk):
        if shortlist_size <= 0 or topk <= 0:
            raise ValueError('positive capacities required')
        self.limit = shortlist_size
        self.k = topk
        self.approximate = []
        self.exact = []
        self.evaluated_ids = set()
        self.in_flight_ids = set()

    def offer(self, sample, score, predicate_valid=True):
        if not predicate_valid:
            return
        score = score if math.isfinite(score) else float('inf')
        # Reference selection, not the proposed native two-way implementation.
        candidates = {identifier: distance for distance, identifier in self.approximate}
        candidates[sample] = min(candidates.get(sample, float('inf')), score)
        self.approximate = sorted((distance, identifier) for identifier, distance in candidates.items())[:self.limit]

    def issue(self):
        samples = [sample for _, sample in self.approximate
                   if sample not in self.evaluated_ids and sample not in self.in_flight_ids]
        self.in_flight_ids.update(samples)
        return samples

    def complete(self, sample, exact_distance):
        if sample not in self.in_flight_ids:
            raise ValueError('unissued or duplicate completion')
        if not math.isfinite(exact_distance):
            raise ValueError('invalid exact distance')
        self.in_flight_ids.remove(sample)
        self.evaluated_ids.add(sample)
        # Exact top-K depends only on exact distance, never approximate retention.
        self.exact = sorted(self.exact + [(exact_distance, sample)])[:self.k]


class IncrementalRerankContract(unittest.TestCase):
    def test_approximate_eviction_does_not_remove_exact_winner(self):
        state = QueryState(1, 1)
        state.offer(11, 100)
        self.assertEqual(state.issue(), [11])
        state.complete(11, 0.1)
        state.offer(22, 1)
        self.assertEqual(state.issue(), [22])
        state.complete(22, 5)
        self.assertEqual(state.approximate, [(1, 22)])
        self.assertEqual(state.exact, [(0.1, 11)])
        state.offer(33, 0.5)
        state.issue()
        state.complete(33, 0.05)
        self.assertEqual(state.exact, [(0.05, 33)])

    def test_evicted_in_flight_result_still_enters_exact_topk(self):
        state = QueryState(1, 1)
        state.offer(1, 10)
        state.issue()
        state.offer(2, 1)
        self.assertEqual(state.issue(), [2])
        state.complete(2, 4)
        state.complete(1, 0.1)
        self.assertEqual(state.exact, [(0.1, 1)])

    def test_evaluated_sample_absent_from_both_arrays_is_never_recomputed(self):
        state = QueryState(1, 1)
        state.offer(1, 10)
        state.issue()
        state.complete(1, 10)
        state.offer(2, 1)
        state.issue()
        state.complete(2, 1)
        state.offer(1, 0.1)  # another producer, improved approximate score
        self.assertEqual(state.issue(), [])
        self.assertEqual(state.exact, [(1, 2)])
        with self.assertRaises(ValueError):
            state.complete(1, 10)

    def test_in_flight_duplicates_predicates_and_query_isolation(self):
        a, b = QueryState(2, 1), QueryState(2, 1)
        a.offer(1, 0, predicate_valid=False)
        self.assertEqual(a.issue(), [])
        a.offer(1, 2)
        self.assertEqual(a.issue(), [1])
        a.offer(1, 1)
        self.assertEqual(a.issue(), [])
        a.complete(1, 3)
        b.offer(1, 1)
        self.assertEqual(b.issue(), [1])
        b.complete(1, 0.5)
        self.assertEqual(a.exact, [(3, 1)])
        self.assertEqual(b.exact, [(0.5, 1)])

    def test_cutoff_ties_and_topk_equal_distance_ids(self):
        state = QueryState(2, 2)
        state.offer(3, 1)
        state.offer(2, 1)
        state.offer(1, 1)
        state.offer(4, 2)
        self.assertEqual(state.issue(), [1, 2])
        state.complete(2, 1)
        state.complete(1, 1)
        self.assertEqual(state.exact, [(1, 1), (1, 2)])

    def test_nonfinite_empty_and_invalid_completion(self):
        state = QueryState(2, 1)
        self.assertEqual(state.issue(), [])
        state.offer(1, math.nan)
        state.offer(2, 0)
        self.assertEqual(state.issue(), [2, 1])
        with self.assertRaises(ValueError):
            state.complete(1, math.nan)
        with self.assertRaises(ValueError):
            state.complete(3, 1)


if __name__ == '__main__':
    unittest.main()
