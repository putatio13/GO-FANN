"""Prepare aligned caption labels, token IDs and numeric attributes from LAION parquet."""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path
import re
import shutil
import struct
import tempfile

import numpy as np

# Ordered categorical vocabulary used by the LAION workloads. Caption tokens
# retain their own IDs; aliases add the canonical label to the same row.
CATEGORIES = {
    'color': 'black white blue red gold green silver pink brown yellow gray orange navy purple beige ivory teal maroon',
    'product': 'shirt dress bag book ring iphone watch jacket poster shoe necklace earring furniture bracelet backpack wallet sneakers helmet charger perfume',
    'place': 'house city beach room kitchen park bedroom school york road street garden office hotel island lake mountain bridge forest church restaurant airport francisco',
    'material': 'leather glass cotton steel wood metal paper fabric stone silk plastic denim wool ceramic linen acrylic velvet marble bamboo',
    'activity_event': 'wedding party birthday travel football running festival basketball soccer yoga meeting cooking dancing celebration cycling concert camping hiking graduation skiing surfing',
    'object': 'table chair pillow camera phone cup lamp toy laptop bottle plate sofa clock mirror mouse basket desk stool vase keyboard',
    'animal': 'dog cat bear chicken fish bird horse butterfly fox tiger eagle lion elephant rabbit wolf deer monkey shark panda cow',
    'vehicle': 'car bicycle truck van train boat motorcycle ship jeep bus airplane scooter helicopter yacht subway taxi ferry tram',
    'food': 'cake chocolate tea wine recipe fruit salad beer breakfast dinner bread lunch pizza cookie vegetable pasta dessert sandwich burger sushi',
    'brand': 'nike samsung honda adidas bmw mercedes toyota lego sony yamaha canon porsche chanel ikea ferrari gucci rolex dior prada tesla',
}
ALIASES = {'grey': 'gray', 'shoes': 'shoe', 'earrings': 'earring', 'mountains': 'mountain',
           'wooden': 'wood', 'bike': 'bicycle', 'plane': 'airplane'}
# These two curated categories were selected through their plural spellings.
CATEGORY_ALIAS_ONLY = ('shoe', 'earring')
STOPWORDS_SHA256 = '47608d511aa4fec95139d41e487109ab4a260313745d397210dbf966b1d3c225'
RAW_TOKENS_SQL = """
SELECT DISTINCT row_idx, token FROM input_batch,
UNNEST(regexp_extract_all(lower(coalesce(caption, '')), '[a-z][a-z0-9]{2,}')) AS t(token)
WHERE token NOT IN (SELECT word FROM stopwords)
"""
OUTPUTS = ('base.spmat', 'base_ranges.fbin', 'row_map.parquet',
           'token_map.parquet', 'category_canonicalization.csv')


def english_stopwords():
    from nltk.corpus import stopwords
    try:
        words = sorted(set(stopwords.words('english')))
    except LookupError as exc:
        raise ValueError('Install the corpus with: python3 -m nltk.downloader stopwords') from exc
    if hashlib.sha256('\n'.join(words).encode()).hexdigest() != STOPWORDS_SHA256:
        raise ValueError('English stopwords differ from the expected 198-word vocabulary')
    return words


def plan_prefix(metadata: Path, images: Path, rows: int):
    import pyarrow.parquet as pq
    found = {}
    for path in metadata.glob('metadata_*.parquet'):
        match = re.fullmatch(r'metadata_(\d+)\.parquet', path.name)
        if not match or int(match[1]) in found:
            raise ValueError(f'invalid or duplicate metadata shard: {path.name}')
        found[int(match[1])] = path
    selected, remaining = [], rows
    for expected, (shard, path) in enumerate(sorted(found.items())):
        if shard != expected:
            raise ValueError(f'missing metadata shard {expected}')
        pf = pq.ParquetFile(path)
        names = pf.schema_arrow.names
        if not {'caption', 'similarity'} <= set(names):
            raise ValueError(f'{path}: caption and similarity columns are required')
        array = np.load(images / f'img_emb_{shard}.npy', mmap_mode='r', allow_pickle=False)
        if (not isinstance(array, np.memmap) or array.ndim != 2 or array.shape[1] <= 0
                or array.dtype.kind != 'f' or array.dtype.itemsize not in (2, 4)
                or array.shape[0] != pf.metadata.num_rows or array.shape[0] == 0):
            raise ValueError(f'{path}: image and metadata shard shapes/types do not match')
        if (images / f'img_emb_{shard}.npy').stat().st_size != array.offset + array.nbytes:
            raise ValueError(f'{path}: invalid image NPY payload size')
        take = min(remaining, pf.metadata.num_rows)
        selected.append((shard, path, take))
        remaining -= take
        del array
        if not remaining:
            break
    if remaining:
        raise ValueError(f'metadata has only {rows - remaining} rows; requested {rows}')
    return selected


def batches(plan, batch_rows):
    import pyarrow.parquet as pq
    for shard, path, take in plan:
        pf = pq.ParquetFile(path)
        columns = [n for n in ('caption', 'similarity', 'key', 'original_width',
                               'original_height', 'width', 'height') if n in pf.schema_arrow.names]
        offset = 0
        for batch in pf.iter_batches(batch_size=batch_rows, columns=columns):
            batch = batch.slice(0, min(batch.num_rows, take - offset))
            if batch.num_rows:
                yield shard, offset, batch
                offset += batch.num_rows
            if offset == take:
                break


def register_tokens(con, batch):
    import pyarrow as pa
    con.register('input_batch', pa.table({'row_idx': np.arange(batch.num_rows, dtype=np.int64),
                                        'caption': batch.column('caption')}))
    con.execute('CREATE OR REPLACE TEMP TABLE raw_tokens AS ' + RAW_TOKENS_SQL)
    con.unregister('input_batch')


def create_vocabulary(con, rows):
    """Assign fixed category IDs first, then descending raw DF / lexical order."""
    con.execute('''CREATE TABLE vocabulary AS
        SELECT ordinal::INTEGER AS token_id, canonical_label AS token FROM categories
        UNION ALL SELECT (row_number() OVER (ORDER BY df DESC, token) - 1 +
            (SELECT count(*) FROM categories))::INTEGER, token FROM counts
        WHERE token NOT IN (SELECT canonical_label FROM categories)''')
    con.execute('''CREATE TABLE token_map AS WITH category_counts AS (
        SELECT m.canonical_label, sum(c.df)::BIGINT AS df
        FROM (SELECT canonical_label AS original_token, canonical_label FROM categories
              WHERE canonical_label NOT IN (?, ?)
              UNION SELECT original_token, canonical_label FROM aliases) m
        JOIN counts c ON c.token=m.original_token GROUP BY m.canonical_label)
        SELECT v.token_id, v.token, coalesce(c.df, least(?, cc.df), 0)::BIGINT AS df,
            coalesce(c.df, 0)::BIGINT AS string_df, coalesce(cc.df, 0)::BIGINT AS category_df_sum,
            coalesce(c.df, least(?, cc.df), 0)::DOUBLE / ? AS selectivity,
            (c.token IS NOT NULL) AS is_string_token, coalesce(g.category_group, '') AS categorical_groups
        FROM vocabulary v LEFT JOIN counts c USING(token)
        LEFT JOIN categories g ON v.token=g.canonical_label
        LEFT JOIN category_counts cc ON v.token=cc.canonical_label''', [*CATEGORY_ALIAS_ONLY, rows, rows, rows])


def numeric_value(value):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return float('nan')
    return number if math.isfinite(number) else float('nan')


def numeric_rows(batch):
    values = batch.to_pydict()
    missing = [None] * batch.num_rows
    dimensions = [(values.get('original_' + name, missing), values.get(name, missing))
                  for name in ('width', 'height')]
    result = np.empty((batch.num_rows, 3), dtype='<f4')
    for i in range(batch.num_rows):
        for col, (original, fallback_values) in enumerate(dimensions):
            primary = numeric_value(original[i])
            fallback = numeric_value(fallback_values[i])
            result[i, col] = primary if primary > 0 else fallback if fallback > 0 else np.nan
        similarity = numeric_value(values['similarity'][i])
        result[i, 2] = similarity if 0 <= similarity <= 1 else np.nan
    return result


def prepare(metadata: Path, images: Path, output: Path, *, rows=100000000,
            batch_rows=65536, token_map: Path | None = None, memory_limit='8GB'):
    import duckdb
    import pyarrow as pa
    import pyarrow.parquet as pq
    if not 0 < rows <= np.iinfo(np.uint32).max or batch_rows <= 0:
        raise ValueError('rows must fit uint32 and batch-rows must be positive')
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'{output} already exists; choose a new output directory')
    plan = plan_prefix(metadata, images, rows)
    words = english_stopwords()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.laion-metadata-', dir=output.parent) as temp:
        work = Path(temp)
        con = duckdb.connect(str(work / 'work.duckdb'))
        try:
            con.execute('SET memory_limit=?', [memory_limit])
            con.execute('SET threads=4')
            con.execute('SET temp_directory=?', [str(work / 'spill')])
            con.execute('CREATE TABLE stopwords(word VARCHAR)')
            con.executemany('INSERT INTO stopwords VALUES (?)', [(w,) for w in words])
            category_rows = [(i, token, group) for i, (token, group) in enumerate(
                (token, group) for group, tokens in CATEGORIES.items() for token in tokens.split())]
            con.execute('CREATE TABLE categories(ordinal INTEGER, canonical_label VARCHAR, category_group VARCHAR)')
            con.executemany('INSERT INTO categories VALUES (?, ?, ?)', category_rows)
            con.execute('CREATE TABLE aliases(original_token VARCHAR, canonical_label VARCHAR)')
            con.executemany('INSERT INTO aliases VALUES (?, ?)', list(ALIASES.items()))
            if token_map is None:
                con.execute('CREATE TABLE counts(token VARCHAR PRIMARY KEY, df BIGINT)')
                done = 0
                for _, _, batch in batches(plan, batch_rows):
                    register_tokens(con, batch)
                    con.execute('''INSERT INTO counts SELECT token, count(*) FROM raw_tokens GROUP BY token
                        ON CONFLICT(token) DO UPDATE SET df=counts.df + excluded.df''')
                    done += batch.num_rows
                    if done % 5000000 < batch.num_rows or done == rows:
                        print(f'Counted caption tokens: {done:,}/{rows:,} rows', flush=True)
                assert done == rows
                create_vocabulary(con, rows)
            else:
                schema = pq.read_schema(token_map)
                if 'token_id' not in schema.names or not pa.types.is_integer(schema.field('token_id').type):
                    raise ValueError('token_id must be an integer column')
                con.execute('CREATE TABLE token_map AS SELECT * FROM read_parquet(?)', [str(token_map)])
            count, unique_ids, unique_tokens, low, high, nulls = con.execute('''
                SELECT count(*),count(DISTINCT token_id),count(DISTINCT token),min(token_id),max(token_id),
                    count(*) FILTER(WHERE token_id IS NULL OR token IS NULL OR token='') FROM token_map''').fetchone()
            if not count or count != unique_ids or count != unique_tokens or low != 0 or high != count-1 or nulls or count > 2**31-1:
                raise ValueError('token map must have unique tokens and contiguous integer IDs from zero')
            if con.execute("SELECT count(*) FROM token_map WHERE token_id != trunc(token_id)").fetchone()[0]:
                raise ValueError('token IDs must be integers')
            con.execute('COPY (SELECT * FROM token_map ORDER BY token_id) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)',
                        [str(work / 'token_map.parquet')])
            with (work / 'category_canonicalization.csv').open('w', newline='') as f:
                writer = csv.writer(f); writer.writerow(['group', 'original_token', 'canonical_label', 'token_id'])
                for ordinal, canonical, group in category_rows:
                    result = con.execute('SELECT token_id FROM token_map WHERE token=?', [canonical]).fetchone()
                    if result is None:
                        raise ValueError(f'token map lacks category {canonical}')
                    originals = ([] if canonical in CATEGORY_ALIAS_ONLY else [canonical])
                    originals += [a for a, c in ALIASES.items() if c == canonical]
                    for original in originals:
                        writer.writerow([group, original, canonical, result[0]])
            row_schema = pa.schema([('row_id', pa.int64()), ('sample_id', pa.int64()),
                                    ('metadata_shard_id', pa.int32()), ('shard_row_id', pa.int64())])
            row_count = nnz = 0
            with ((work / 'indptr').open('wb') as pointers, (work / 'indices').open('wb') as indices,
                  (work / 'base_ranges.fbin').open('wb') as ranges,
                  pq.ParquetWriter(work / 'row_map.parquet', row_schema, compression='zstd') as row_writer):
                pointers.write(struct.pack('<q', 0)); ranges.write(struct.pack('<II', rows, 3))
                for shard, offset, batch in batches(plan, batch_rows):
                    register_tokens(con, batch)
                    con.execute('''CREATE OR REPLACE TEMP TABLE expanded AS
                        SELECT row_idx,token FROM raw_tokens UNION
                        SELECT r.row_idx,a.canonical_label FROM raw_tokens r JOIN aliases a ON r.token=a.original_token''')
                    missing = con.execute('SELECT token FROM expanded ANTI JOIN token_map USING(token) LIMIT 1').fetchone()
                    if missing:
                        raise ValueError(f'token map lacks caption token {missing[0]!r}')
                    pairs = con.execute('''SELECT e.row_idx,t.token_id FROM expanded e JOIN token_map t USING(token)
                        ORDER BY e.row_idx,t.token_id''').fetchnumpy()
                    ids = np.asarray(pairs['token_id'], dtype='<i4')
                    counts = np.bincount(np.asarray(pairs['row_idx'], dtype=np.int64), minlength=batch.num_rows)
                    np.asarray(np.cumsum(counts, dtype=np.int64) + nnz, dtype='<i8').tofile(pointers)
                    ids.tofile(indices); nnz += len(ids)
                    numeric_rows(batch).tofile(ranges)
                    row_ids = np.arange(row_count, row_count + batch.num_rows, dtype=np.int64)
                    keys = batch.column('key').to_pylist() if 'key' in batch.schema.names else [None]*batch.num_rows
                    samples = []
                    for key, fallback in zip(keys, row_ids):
                        try:
                            try:
                                sample = int(key)
                            except (TypeError, ValueError):
                                sample = int(float(key))
                            if not -(2**63) <= sample < 2**63: raise ValueError()
                        except (TypeError, ValueError, OverflowError): sample = int(fallback)
                        samples.append(sample)
                    row_writer.write_table(pa.Table.from_arrays([
                        pa.array(row_ids), pa.array(samples, type=pa.int64()),
                        pa.array(np.full(batch.num_rows, shard, dtype=np.int32)),
                        pa.array(np.arange(offset, offset + batch.num_rows, dtype=np.int64))], schema=row_schema))
                    row_count += batch.num_rows
                    if row_count % 5000000 < batch.num_rows or row_count == rows:
                        print(f'Prepared attributes: {row_count:,}/{rows:,} rows', flush=True)
                if row_count != rows: raise ValueError('metadata row count changed during preparation')
            with (work / 'base.spmat').open('wb') as f:
                f.write(struct.pack('<qqq', rows, count, nnz))
                for name in ('indptr', 'indices'):
                    with (work / name).open('rb') as part: shutil.copyfileobj(part, f, 1024*1024)
        finally:
            con.close()
        # Publish only after all producers and validation have succeeded.
        output.mkdir()
        installed = []
        try:
            for name in OUTPUTS:
                (work / name).rename(output / name)
                installed.append(output / name)
        except BaseException:
            for path in installed: path.unlink()
            output.rmdir()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, required=True, help='Directory of metadata_<id>.parquet')
    parser.add_argument('--images', type=Path, required=True, help='Matching img_emb_<id>.npy directory')
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    parser.add_argument('--rows', type=int, default=100000000)
    parser.add_argument('--batch-rows', type=int, default=65536)
    parser.add_argument('--memory-limit', default='8GB', help='DuckDB memory budget; temporary data can spill to disk')
    parser.add_argument('--token-map', type=Path, help='Reuse token IDs when preparing inputs for existing queries/indexes')
    args = parser.parse_args()
    prepare(args.metadata, args.images, args.output, rows=args.rows, batch_rows=args.batch_rows,
            token_map=args.token_map, memory_limit=args.memory_limit)
    print(f'Prepared {args.rows} metadata rows in {args.output}')


if __name__ == '__main__':
    main()
