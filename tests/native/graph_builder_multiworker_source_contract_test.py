#!/usr/bin/env python3

import re
import unittest
from pathlib import Path


SOURCE = (Path(__file__).resolve().parents[2] / "src" / "index/graph_builder.cu").read_text()


def function_body(name: str) -> str:
    match = re.search(rf"\b{name}\s*\([^)]*\)\s*\{{", SOURCE)
    if match is None:
        raise AssertionError(f"missing function {name}")
    depth = 1
    cursor = match.end()
    while cursor < len(SOURCE) and depth:
        if SOURCE[cursor] == "{":
            depth += 1
        elif SOURCE[cursor] == "}":
            depth -= 1
        cursor += 1
    if depth != 0:
        raise AssertionError(f"unterminated function {name}")
    return SOURCE[match.end() : cursor - 1]


class GraphBuilderMultiworkerSourceContractTest(unittest.TestCase):
    def test_parallel_cagra_keeps_host_dataset(self) -> None:
        body = function_body("build_parallel_stream_graph_chunk")
        self.assertIn("build_cagra_graph_chunk<ScalarT>", body)
        self.assertNotIn("make_device_matrix", body)
        self.assertNotIn("build_cagra_graph_chunk_device_staged", SOURCE)

    def test_parallel_path_controls_connectivity(self) -> None:
        body = function_body("build_cagra_graph_chunk")
        self.assertIn("params.guarantee_connectivity = guarantee_connectivity", body)

    def test_serial_builds_share_one_raft_handle(self) -> None:
        body = function_body("run_graph_builder_streamed_parallel")
        isolated_resource = body.index("raft::resources serial_res;")
        worker_resources = body.index("shared_resources::configured_raft_resources main_res")
        self.assertLess(isolated_resource, worker_resources)
        self.assertIn("cagra_label_uses_ace(label_size)", body)
        self.assertIn("isolated_labels", body)
        self.assertIn("worker_labels", body)
        self.assertIn("raft::resources& serial_res = main_res", body)
        self.assertRegex(body, r'process_label\(label,\s*0,\s*"isolated"')
        self.assertRegex(body, r'process_label\(label,\s*worker_id,\s*"serialized"')


if __name__ == "__main__":
    unittest.main()
