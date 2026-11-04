"""Artifact publication safety, independent of CUDA availability."""
import tempfile
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

from scripts.quantize import generate, parser


class QuantizePublicationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.input = self.root / 'input.u8bin'
        self.input.write_bytes(b'fixture')
        self.builder = self.root / 'builder'
        self.builder.write_text('#!/bin/sh\nexit 1\n')
        self.builder.chmod(0o755)
        self.output = self.root / 'result'

    def args(self):
        return parser().parse_args(['pq', '--input', str(self.input), '--output-dir', str(self.output),
                                    '--builder', str(self.builder)])

    def test_existing_artifact_directory_preserved(self):
        self.output.mkdir()
        sentinel = self.output / 'important.bin'
        sentinel.write_bytes(b'preserve')
        with self.assertRaises(FileExistsError):
            generate(self.args())
        self.assertEqual(sentinel.read_bytes(), b'preserve')

    def test_failed_native_build_removes_partial_output(self):
        def fail(command):
            dest = Path(next(item.split('=', 1)[1] for item in command if item.startswith('--pivots=')))
            dest.write_bytes(b'partial')
            raise subprocess.CalledProcessError(1, command)
        with patch('scripts.quantize.run', side_effect=fail):
            with self.assertRaises(subprocess.CalledProcessError):
                generate(self.args())
        self.assertFalse(self.output.exists())
        self.assertEqual(self.input.read_bytes(), b'fixture')

    def test_interruption_removes_partial_output(self):
        with patch('scripts.quantize.run', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                generate(self.args())
        self.assertFalse(self.output.exists())

    def test_missing_builder_does_not_create_output(self):
        self.builder.unlink()
        with self.assertRaises(FileNotFoundError):
            generate(self.args())
        self.assertFalse(self.output.exists())

    def test_invalid_training_cap_does_not_create_output(self):
        args = self.args(); args.training_rows = 255
        with self.assertRaisesRegex(ValueError, '256'):
            generate(args)
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
