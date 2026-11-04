"""Connect the host operation policy to native score production and stopping.

Supplement host policy and CUDA score/heap tests; these are integration guards,
not numerical or performance evidence.
"""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[2] / "src"


class NavigationScoreSourceContract(unittest.TestCase):
    maxDiff = 1000

    def test_operation_policy_reaches_both_backends(self):
        main = (ROOT / "app/main.cu").read_text()
        self.assertEqual(main.count(
            "search_options.raw_navigation_scores = uses_raw_navigation_scores(options);"), 2)
        self.assertLess(main.index("go_fann::resolve_navigation_score_policy(options);"),
                        main.index("go_fann::validate_app_options(options);"))
        for backend in ("pq", "rabitq"):
            header = (ROOT / f"search/{backend}/backend_{backend}.cuh").read_text()
            self.assertIn("bool raw_navigation_scores = false;", header)

    def test_all_native_launches_normalize_both_factors(self):
        for backend, launches in (("pq", 2), ("rabitq", 1)):
            source = (ROOT / f"search/{backend}/backend_{backend}.cu").read_text()
            for factor in ("bit", "range"):
                pattern = (r"\(options\.raw_navigation_scores && !kDeviceResident\)"
                           r"\s*\? 1\.0f : options\."
                           + factor + r"_filter_bonus")
                self.assertEqual(len(re.findall(pattern, source)), launches)
            benchmark = source[source.index("int run_benchmark_sweep("):]
            self.assertLess(benchmark.index("quick_filter_bonuses.size() > 1"),
                            benchmark.index("load_"))

    def test_partial_bonus_work_is_guarded_and_base_precedes_penalty(self):
        for backend, producers in (("pq", 2), ("rabitq", 1)):
            source = (ROOT / f"search/{backend}/backend_{backend}.cu").read_text()
            pattern = re.compile(
                r"kp\.first = dist_list\[cid\]\[i\];\s*"
                r"if \(bit_filter_bonus != 1\.0f \|\| range_filter_bonus != 1\.0f\) "
                r"\{(.*?)\}\s*if constexpr \(kPredicateBias\) \{\s*kp\.base = kp\.first;",
                re.S)
            bodies = pattern.findall(source)
            self.assertEqual(len(bodies), producers)
            for body in bodies:
                self.assertIn("count_matching_active_ranges(", body)
                self.assertIn("apply_filter_bonuses(", body)
            self.assertIn("gpu_predicate::priority(kp.base, predicate_penalty, pass, saturated)", source)
            selected_base = ("gpu_predicate::base_score(lane0_state[cid].now)" if backend == "pq"
                             else "if constexpr (kPredicateBias) return lane0_state[cid].now.base;")
            self.assertTrue(selected_base in source, backend)
            self.assertTrue("gpu_predicate::minimum_pending_base(" in source, backend)


if __name__ == "__main__":
    unittest.main()
