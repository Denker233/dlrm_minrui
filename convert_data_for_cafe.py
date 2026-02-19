#!/usr/bin/env python3
"""Convert Kaggle .npz data to CAFE+ binary memmap format."""

import numpy as np
import os
import sys

def main():
    npz_path = "./input/kaggleAdDisplayChallenge_processed.npz"
    output_dir = "/home/cc/expr/CAFE_plus/criteo_kaggle"

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading {npz_path}...")
    data = np.load(npz_path, allow_pickle=True)

    print("Keys:", list(data.keys()))
    for k in data.keys():
        print(f"  {k}: shape={data[k].shape}, dtype={data[k].dtype}")

    X_cat = data['X_cat']   # (45840617, 26), float64
    X_int = data['X_int']   # (45840617, 13), int32
    y = data['y']           # (45840617,), int32
    counts = data['counts'] # (26,), int64

    # 1. Sparse features: convert to int32
    print("Converting sparse features to int32...")
    sparse_path = os.path.join(output_dir, "kaggle_processed_sparse_sep.bin")
    X_cat_int32 = X_cat.astype(np.int32)
    print(f"  X_cat range: [{X_cat_int32.min()}, {X_cat_int32.max()}]")
    X_cat_int32.tofile(sparse_path)
    print(f"  Saved: {sparse_path} ({os.path.getsize(sparse_path)} bytes)")

    # 2. Dense features: apply log(x+1) transform and save as float32
    print("Converting dense features with log(x+1) transform to float32...")
    dense_path = os.path.join(output_dir, "kaggle_processed_dense.bin")
    X_int_float = np.log(X_int.astype(np.float32) + 1.0)
    print(f"  X_int raw range: [{X_int.min()}, {X_int.max()}]")
    print(f"  X_int log-transformed range: [{X_int_float.min():.4f}, {X_int_float.max():.4f}]")
    X_int_float.tofile(dense_path)
    print(f"  Saved: {dense_path} ({os.path.getsize(dense_path)} bytes)")

    # 3. Labels: already int32
    print("Saving labels as int32...")
    label_path = os.path.join(output_dir, "kaggle_processed_label.bin")
    y_int32 = y.astype(np.int32)
    print(f"  Labels: {np.unique(y_int32, return_counts=True)}")
    y_int32.tofile(label_path)
    print(f"  Saved: {label_path} ({os.path.getsize(label_path)} bytes)")

    # 4. Counts: CAFE+ expects cumulative counts with shape (27,)
    # count[0] = 0, count[i+1] = count[i] + per_feature_count[i]
    print("Converting counts to cumulative format (27,) int32...")
    cum_counts = np.zeros(27, dtype=np.int32)
    for i in range(26):
        cum_counts[i + 1] = cum_counts[i] + int(counts[i])
    count_path = os.path.join(output_dir, "kaggle_processed_count.bin")
    print(f"  Per-feature counts: {counts.astype(int).tolist()}")
    print(f"  Cumulative counts: {cum_counts.tolist()}")
    cum_counts.tofile(count_path)
    print(f"  Saved: {count_path} ({os.path.getsize(count_path)} bytes)")

    # 5. Verify by loading back
    print("\nVerifying files...")
    v_cat = np.memmap(sparse_path, dtype=np.int32, mode='r', shape=(45840617, 26))
    v_dense = np.memmap(dense_path, dtype=np.float32, mode='r', shape=(45840617, 13))
    v_label = np.memmap(label_path, dtype=np.int32, mode='r', shape=(45840617,))
    v_count = np.memmap(count_path, dtype=np.int32, mode='r', shape=(27,))

    # Verify shapes
    assert v_cat.shape == (45840617, 26), f"Bad cat shape: {v_cat.shape}"
    assert v_dense.shape == (45840617, 13), f"Bad dense shape: {v_dense.shape}"
    assert v_label.shape == (45840617,), f"Bad label shape: {v_label.shape}"
    assert v_count.shape == (27,), f"Bad count shape: {v_count.shape}"

    # Verify first few samples match
    assert np.allclose(v_cat[0], X_cat_int32[0]), "Cat mismatch at index 0"
    assert np.allclose(v_dense[0], X_int_float[0]), "Dense mismatch at index 0"
    assert v_label[0] == y_int32[0], "Label mismatch at index 0"

    # Verify derived counts
    new_count = np.zeros(26)
    for i in range(26):
        new_count[i] = v_count[i + 1] - v_count[i]
    assert np.allclose(new_count, counts.astype(np.float64)), "Count mismatch"

    print("All verifications passed!")
    print(f"\nCAFE+ data ready at: {output_dir}")
    print(f"  --cat-path=\"{sparse_path}\"")
    print(f"  --dense-path=\"{dense_path}\"")
    print(f"  --label-path=\"{label_path}\"")
    print(f"  --count-path=\"{count_path}\"")

if __name__ == "__main__":
    main()
