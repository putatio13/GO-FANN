"""Raw parquet preparation preserves vector rows and predicate semantics."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.laion import metadata, base, string_index


def read_csr(path):
    with path.open('rb') as f:
        rows, cols, nnz = struct.unpack('<qqq', f.read(24))
        ptr = np.fromfile(f, '<i8', rows + 1)
        ids = np.fromfile(f, '<i4', nnz)
        assert not f.read(1)
    return cols, [ids[ptr[i]:ptr[i+1]].tolist() for i in range(rows)]


class MetadataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.meta, self.images = self.root/'metadata', self.root/'images'
        self.meta.mkdir(); self.images.mkdir()

    def shard(self, number, captions, **columns):
        values = dict(caption=captions, similarity=[.5]*len(captions), **columns)
        pq.write_table(pa.table(values), self.meta/f'metadata_{number}.parquet', row_group_size=2)
        np.save(self.images/f'img_emb_{number}.npy', np.ones((len(captions), 2), '<f2'))

    def run_prepare(self, name='out', rows=4, **kwargs):
        out = self.root/name
        metadata.prepare(self.meta, self.images, out, rows=rows, batch_rows=2, **kwargs)
        return out

    def test_raw_parquet_to_predicate_readers(self):
        self.shard(0, ['The grey bike RED red', None], original_width=[-1,300], width=[20,30], height=[40,50],key=['33','44.0'])
        self.shard(1, ['Shoes wooden plane İSTANBUL café 123', 'red red'], width=[60,70], height=[80,90],key=['invalid',None])
        out = self.run_prepare()
        table = pq.read_table(out/'token_map.parquet').to_pydict()
        ids = dict(zip(table['token'],table['token_id']))
        self.assertEqual(ids['black'],0)
        self.assertEqual(ids['tesla'],198)
        self.assertNotIn('the',ids); self.assertNotIn('123',ids)
        self.assertEqual(table['df'][ids['red']],2)
        self.assertEqual(table['df'][ids['gray']],1)
        self.assertFalse(table['is_string_token'][ids['gray']])
        _, rows = read_csr(out/'base.spmat')
        expected = [{'grey','gray','bike','bicycle','red'},set(),
                    {'shoes','shoe','wooden','wood','plane','airplane','istanbul','caf'}, {'red'}]
        self.assertEqual(rows,[sorted(ids[t] for t in tokens) for tokens in expected])
        ranges = np.fromfile(out/'base_ranges.fbin','<f4',offset=8).reshape(-1,3)
        np.testing.assert_array_equal(ranges[:,0],[20,300,60,70])
        self.assertEqual(base.read_fbin_header(out/'base_ranges.fbin'),(4,3))
        self.assertEqual(base.read_spmat_header(out/'base.spmat')[:2],(4,len(ids)))
        rowmap = pq.read_table(out/'row_map.parquet').to_pydict()
        self.assertEqual(rowmap['row_id'],[0,1,2,3])
        self.assertEqual(rowmap['sample_id'],[33,44,2,3])
        self.assertEqual(rowmap['metadata_shard_id'],[0,0,1,1])
        self.assertEqual(rowmap['shard_row_id'],[0,1,0,1])
        string_index.write_ngram_index(self.root/'operators.bin', out/'token_map.parquet', 3, 2, False)
        self.assertGreater((self.root/'operators.bin').stat().st_size,128)

    def test_numeric_shard_order_and_partial_prefix_cli(self):
        for shard in reversed(range(12)):
            self.shard(shard,[f'token{shard}'],width=[shard+1],height=[1])
        output=self.root/'cli'
        result=subprocess.run([sys.executable,'-m','scripts.laion.metadata','--metadata',str(self.meta),
            '--images',str(self.images),'--output',str(output),'--rows','11','--batch-rows','3'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        rows=pq.read_table(output/'row_map.parquet').to_pydict()
        self.assertEqual(rows['metadata_shard_id'],list(range(11)))
        np.testing.assert_array_equal(np.fromfile(output/'base_ranges.fbin','<f4',offset=8).reshape(-1,3)[:,0],np.arange(1,12))

    def test_reuse_vocabulary_and_unknown_rejection(self):
        self.shard(0,['cat rareword','dog', 'blue','red'])
        first=self.run_prepare()
        second=self.run_prepare('reuse',token_map=first/'token_map.parquet')
        self.assertEqual((first/'base.spmat').read_bytes(),(second/'base.spmat').read_bytes())
        self.shard(0,['newunknown','dog','blue','red'])
        with self.assertRaisesRegex(ValueError,'lacks caption token'):
            self.run_prepare('bad',token_map=first/'token_map.parquet')
        self.assertFalse((self.root/'bad').exists())
        self.assertFalse(list(self.root.glob('.laion-metadata-*')))

    def test_numeric_invalid_and_fallback(self):
        batch=pa.record_batch({'caption':['a']*4,'original_width':[None,-1.,float('inf'),12.],
            'width':[10.,None,0.,5.],'height':[None,5.,-2.,3.],'similarity':[None,-1.,1.1,0.9]})
        actual=metadata.numeric_rows(batch)
        np.testing.assert_allclose(actual,[[10,np.nan,np.nan],[np.nan,5,np.nan],
            [np.nan,np.nan,np.nan],[12,3,.9]],equal_nan=True)

    def test_shard_alignment_limits_and_protected_output(self):
        self.shard(0,['cat','dog']); self.shard(2,['red','blue'])
        with self.assertRaisesRegex(ValueError,'missing metadata shard'): self.run_prepare()
        (self.meta/'metadata_2.parquet').rename(self.meta/'metadata_1.parquet')
        (self.images/'img_emb_2.npy').rename(self.images/'img_emb_1.npy')
        out=self.run_prepare()
        before=(out/'base.spmat').read_bytes()
        with self.assertRaises(FileExistsError):self.run_prepare()
        self.assertEqual((out/'base.spmat').read_bytes(),before)
        np.save(self.images/'img_emb_1.npy',np.ones((3,2),'<f2'))
        with self.assertRaisesRegex(ValueError,'shapes/types'):self.run_prepare('mismatch')
        self.assertFalse((self.root/'mismatch').exists())
        with self.assertRaises(ValueError):self.run_prepare('zero',rows=0)
        with self.assertRaises(ValueError):self.run_prepare('negative',rows=-1)

    def test_duplicate_shard_and_source_failure_cleanup(self):
        self.shard(0,['cat','dog','red','blue'])
        (self.meta/'metadata_00.parquet').write_bytes((self.meta/'metadata_0.parquet').read_bytes())
        with self.assertRaisesRegex(ValueError,'duplicate'):self.run_prepare()
        (self.meta/'metadata_00.parquet').unlink()
        with patch.object(metadata,'numeric_rows',side_effect=RuntimeError('injected read failure')):
            with self.assertRaisesRegex(RuntimeError,'injected'):self.run_prepare()
        self.assertFalse((self.root/'out').exists())
        self.assertFalse(list(self.root.glob('.laion-metadata-*')))


if __name__=='__main__':
    unittest.main()
