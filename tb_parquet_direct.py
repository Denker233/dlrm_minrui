#!/usr/bin/env python3
"""Direct parquet -> numpy for Criteo Terabyte, replacing the parquet->TSV->python-loop path.

The reference parser (data_utils.py process_one_file) does, per row:
    line = line.split("\t"); pad to 40; "" or "\n" -> "0"
    y      = int64(line[0])
    X_int  = np.array(line[1:14],  dtype=int64)
    X_cat  = np.array([int(x,16) % max_ind_range for x in line[14:]], dtype=int64)

This reproduces exactly that, vectorised, straight from the parquet buffers:
  - null / empty  -> 0   (the TSV path writes "" which becomes "0")
  - cat values    -> int(hex,16) % max_ind_range
  - X_int negatives are left alone (processCriteoAdData zeroes them later)
"""
import numpy as np, pyarrow as pa, pyarrow.parquet as pq

MAX_IND_RANGE = 10_000_000
_LUT = np.zeros(256, dtype=np.uint64)
for _i, _c in enumerate(b'0123456789'): _LUT[_c] = _i
for _i, _c in enumerate(b'abcdef'):     _LUT[_c] = 10 + _i
for _i, _c in enumerate(b'ABCDEF'):     _LUT[_c] = 10 + _i

def _chunk(tbl, name):
    c = tbl.column(name).combine_chunks()
    return c.chunk(0) if isinstance(c, pa.ChunkedArray) else c

def hex_col(arr, n):
    """8-char hex strings -> uint64. Null/empty -> 0.
    Valid values are contiguous in the value buffer (nulls occupy 0 bytes)."""
    offs = np.frombuffer(arr.buffers()[1], dtype=np.int32, count=n + 1)
    lens = offs[1:] - offs[:-1]
    valid = lens == 8
    nv = int(valid.sum())
    out = np.zeros(n, dtype=np.uint64)
    if nv:
        buf = np.frombuffer(arr.buffers()[2], dtype=np.uint8)
        b = buf[:nv * 8].reshape(nv, 8)
        v = _LUT[b[:, 0]]
        for k in range(1, 8):
            v = (v << np.uint64(4)) | _LUT[b[:, k]]
        out[valid] = v
    if not valid.all() and (lens[~valid] != 0).any():
        raise ValueError("unexpected hex length (not 0 or 8)")
    return out

def read_part(path, max_ind_range=MAX_IND_RANGE):
    """returns y (i4), X_int (n,13 i4), X_cat (n,26 i4)"""
    cols = (["label"] + [f"integer_feature_{i}" for i in range(1, 14)]
                      + [f"categorical_feature_{j}" for j in range(1, 27)])
    tbl = pq.read_table(path, columns=cols)
    n = tbl.num_rows
    y = tbl.column("label").fill_null(0).combine_chunks().to_numpy(zero_copy_only=False).astype(np.int32)
    X_int = np.empty((n, 13), dtype=np.int32)
    for i in range(13):
        X_int[:, i] = tbl.column(f"integer_feature_{i+1}").fill_null(0) \
                         .combine_chunks().to_numpy(zero_copy_only=False).astype(np.int32)
    X_cat = np.empty((n, 26), dtype=np.int32)
    for j in range(26):
        v = hex_col(_chunk(tbl, f"categorical_feature_{j+1}"), n)
        X_cat[:, j] = (v % np.uint64(max_ind_range)).astype(np.int32) if max_ind_range > 0 \
                      else v.astype(np.int32)
    return y, X_int, X_cat
