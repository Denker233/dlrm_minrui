#!/usr/bin/env python3
"""Convert one HuggingFace criteo/CriteoClickLogs parquet part to the original
Criteo Terabyte TSV line format expected by data_utils.getCriteoAdData():

    label \t int_1..int_13 \t cat_1..cat_26 \n

Missing values are written as empty fields (the repo's parser maps "" -> "0").
Usage: tb_convert_part.py <in.parquet> <out.txt>
"""
import sys
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

COLS = (["label"]
        + [f"integer_feature_{i}" for i in range(1, 14)]
        + [f"categorical_feature_{i}" for i in range(1, 27)])

def main(src, dst):
    tbl = pq.read_table(src, columns=COLS)
    df = tbl.to_pandas(types_mapper={pa.int32(): pd.Int32Dtype()}.get)
    df = df[COLS]
    df.to_csv(dst, sep="\t", header=False, index=False, na_rep="",
              lineterminator="\n")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
