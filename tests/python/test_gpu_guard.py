"""CPU-only regression checks for the GPU smoke preflight."""

import subprocess
import tempfile
import unittest
from unittest.mock import patch

from tests.tools.gpu_guard import idle_gpu, require_idle_gpu


class GpuGuardTests(unittest.TestCase):
    def probe(self, status, processes=''):
        outputs = [subprocess.CompletedProcess([], 0, status),
                   subprocess.CompletedProcess([], 0, processes)]
        with patch('tests.tools.gpu_guard.subprocess.run', side_effect=outputs):
            return require_idle_gpu(2)

    def test_idle_target_ignores_other_gpu_process(self):
        self.assertEqual(self.probe('GPU-target, 0, 0', 'GPU-other, 1234'), 'GPU-target')

    def test_target_compute_process_blocks_even_without_reported_memory(self):
        with self.assertRaisesRegex(RuntimeError, 'occupied'):
            self.probe('GPU-target, 0, 0', 'GPU-target, 1234')

    def test_busy_memory_or_utilization_blocks(self):
        for status in ('GPU-target, 80, 0', 'GPU-target, 0, 5'):
            with self.subTest(status=status), self.assertRaisesRegex(RuntimeError, 'occupied'):
                self.probe(status)

    def test_unknown_device_and_missing_metrics_fail_closed(self):
        with self.assertRaises(ValueError):
            require_idle_gpu(-1)
        with self.assertRaises(RuntimeError):
            self.probe('')
        with self.assertRaises(ValueError):
            self.probe('GPU-target, N/A, 0')

    def test_child_uses_checked_uuid_and_conda_libraries(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch('tests.tools.gpu_guard.tempfile.gettempdir', return_value=folder), \
                    patch('tests.tools.gpu_guard.require_idle_gpu', return_value='GPU-target'), \
                    patch.dict('os.environ', {'CONDA_PREFIX': '/env', 'LD_LIBRARY_PATH': '/extra',
                                              'CUDA_VISIBLE_DEVICES': '7'}, clear=True):
                with idle_gpu(2) as env:
                    self.assertEqual(env['CUDA_VISIBLE_DEVICES'], 'GPU-target')
                    self.assertEqual(env['LD_LIBRARY_PATH'], '/env/lib:/env/targets/x86_64-linux/lib:/extra')
                    self.assertEqual(env['OMP_DYNAMIC'], 'FALSE')
                    with self.assertRaisesRegex(RuntimeError, 'Another GO-FANN validation'):
                        with idle_gpu(2):
                            self.fail('Concurrent GPU owner was accepted')


if __name__ == '__main__':
    unittest.main()
