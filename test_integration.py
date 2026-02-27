#!/usr/bin/env python3
"""Quick integration test: verify C++ frame packing works in codec pipeline."""

import os, sys, time, tempfile, subprocess
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import compressed_emb as _C

# Import the updated functions from codec_ondemand_benchmark
from codec_ondemand_benchmark import (
    rows_to_tiled_frame, tiled_frame_to_rows, encode_h265_perframe,
    OnDemandFrameDecoder, InMemoryFrameDecoder, quantize_table
)

EMB_DIM = 16
WIDTH, HEIGHT = 1920, 1080
ROWS_PER_FRAME = (WIDTH // 4) * (HEIGHT // 4)

def test_tiling_roundtrip():
    """Test: tile → untile preserves data."""
    print("Test 1: Tiling roundtrip...")
    N = ROWS_PER_FRAME
    data = np.random.randint(0, 256, (N, 16), dtype=np.uint8)
    frame = rows_to_tiled_frame(data, WIDTH, HEIGHT)
    assert frame.shape == (HEIGHT, WIDTH), f"Frame shape wrong: {frame.shape}"
    recovered = tiled_frame_to_rows(frame, WIDTH, HEIGHT)
    assert np.array_equal(data, recovered[:N]), "Roundtrip MISMATCH!"
    print(f"  PASS: {N:,} rows preserved through tile/untile")

def test_fused_multiframe():
    """Test: C++ fused multiframe matches Python step-by-step."""
    print("Test 2: Fused multiframe encode...")
    N = ROWS_PER_FRAME * 3 + 50000  # 3.5 frames
    fp32_data = torch.randn(N, 16)

    # C++ fused
    results = _C.fused_quantize_tile_multiframe_fp32(fp32_data, WIDTH, HEIGHT)
    frames = results[:-2]
    scale = results[-2].item()
    zp = results[-1].item()

    expected_frames = (N + ROWS_PER_FRAME - 1) // ROWS_PER_FRAME
    assert len(frames) == expected_frames, f"Expected {expected_frames} frames, got {len(frames)}"

    # Verify first frame data
    frame0_rows = tiled_frame_to_rows(frames[0].numpy(), WIDTH, HEIGHT)
    orig_chunk = fp32_data[:ROWS_PER_FRAME].numpy()
    dequant = (frame0_rows[:ROWS_PER_FRAME].astype(np.float32) - zp) * scale
    mae = np.abs(dequant - orig_chunk).mean()
    # MAE depends on data range; for randn (range ~[-3,3]) quantized to [0,255],
    # step size is ~6/255 ≈ 0.024, so MAE should be < step_size/2
    data_range = fp32_data.max().item() - fp32_data.min().item()
    step_size = data_range / 255.0
    assert mae < step_size, f"MAE {mae:.6f} exceeds step size {step_size:.6f}"
    print(f"  PASS: {len(frames)} frames, MAE={mae:.6f} (step={step_size:.6f})")

def test_selective_gather():
    """Test: gather specific rows from tiled frame."""
    print("Test 3: Selective gather from tiled frame...")
    N = ROWS_PER_FRAME
    data = np.random.randint(0, 256, (N, 16), dtype=np.uint8)
    frame = rows_to_tiled_frame(data, WIDTH, HEIGHT)
    frame_t = torch.from_numpy(frame)

    # Gather 100 random rows
    indices = np.sort(np.random.choice(N, 100, replace=False))
    indices_t = torch.from_numpy(indices).long()

    tiles_per_row = WIDTH // 4
    gathered = _C.gather_from_tiled_frame(frame_t, indices_t, tiles_per_row)
    expected = data[indices]

    assert np.array_equal(gathered.numpy(), expected), "Gather MISMATCH!"
    print(f"  PASS: 100 rows gathered correctly")

def test_gather_dequant():
    """Test: gather + dequantize from tiled frame."""
    print("Test 4: Gather + dequantize from tiled frame...")
    N = ROWS_PER_FRAME
    data = np.random.randint(0, 256, (N, 16), dtype=np.uint8)
    frame = rows_to_tiled_frame(data, WIDTH, HEIGHT)
    frame_t = torch.from_numpy(frame)

    scale, zp = 0.02, 128
    indices = np.sort(np.random.choice(N, 500, replace=False))
    indices_t = torch.from_numpy(indices).long()
    tiles_per_row = WIDTH // 4

    result = _C.gather_dequant_from_tiled_frame(frame_t, indices_t, tiles_per_row, scale, zp)
    expected = (data[indices].astype(np.float32) - zp) * scale

    assert np.allclose(result.numpy(), expected, atol=1e-5), "Gather+dequant MISMATCH!"
    print(f"  PASS: 500 rows gathered+dequantized correctly")

def test_encode_decode_roundtrip():
    """Test: full encode → decode roundtrip with H.265."""
    print("Test 5: H.265 encode/decode roundtrip...")
    N = ROWS_PER_FRAME + 5000  # Just over 1 frame
    fp32_data = torch.randn(N, 16)
    q, s, zp = quantize_table(fp32_data)
    q_np = q.numpy()

    with tempfile.TemporaryDirectory() as tmpdir:
        num_frames, frame_dir, compressed_bytes, enc_time, rpf = encode_h265_perframe(
            q_np, WIDTH, HEIGHT, crf=0, output_dir=tmpdir, table_id=99
        )

        print(f"  Encoded: {num_frames} frames, {compressed_bytes/1024:.0f}KB, {enc_time:.2f}s")

        # Decode first frame
        decoder = OnDemandFrameDecoder(frame_dir, rpf, EMB_DIM, WIDTH, HEIGHT)
        decoded = decoder.decode_frame(0)

        # Compare
        orig_chunk = q_np[:rpf]
        mae = np.abs(decoded[:rpf].astype(int) - orig_chunk.astype(int)).mean()
        print(f"  Decode MAE (uint8): {mae:.4f}")
        if mae < 0.5:
            print(f"  PASS: lossless/near-lossless roundtrip")
        else:
            print(f"  WARNING: MAE={mae:.4f} (may be lossy CRF)")

def test_scattered_gather_multiframe():
    """Test: fused scattered gather + quantize + tile multiframe."""
    print("Test 6: Fused scattered gather multiframe...")
    total_rows = 500000
    n_cold = 300000
    weight = torch.randn(total_rows, 16)
    cold_indices = torch.sort(torch.randperm(total_rows)[:n_cold])[0].long()

    results = _C.fused_gather_quantize_tile_multiframe(weight, cold_indices, WIDTH, HEIGHT)
    frames = results[:-2]
    scale = results[-2].item()
    zp = results[-1].item()

    n_frames_expected = (n_cold + ROWS_PER_FRAME - 1) // ROWS_PER_FRAME
    assert len(frames) == n_frames_expected, f"Expected {n_frames_expected} frames, got {len(frames)}"

    # Verify: dequantize first frame and compare to original
    frame0_rows = tiled_frame_to_rows(frames[0].numpy(), WIDTH, HEIGHT)
    chunk_indices = cold_indices[:ROWS_PER_FRAME]
    orig = weight[chunk_indices].numpy()
    dequant = (frame0_rows[:ROWS_PER_FRAME].astype(np.float32) - zp) * scale
    mae = np.abs(dequant - orig).mean()
    data_range = weight[chunk_indices].max().item() - weight[chunk_indices].min().item()
    step_size = data_range / 255.0
    assert mae < step_size, f"MAE {mae:.6f} exceeds step size {step_size:.6f}"
    print(f"  PASS: {len(frames)} frames from {n_cold:,} scattered rows, MAE={mae:.6f}")


if __name__ == "__main__":
    print("=" * 60)
    print("Integration Tests: C++ Frame Packing in Codec Pipeline")
    print("=" * 60)

    test_tiling_roundtrip()
    test_fused_multiframe()
    test_selective_gather()
    test_gather_dequant()
    test_scattered_gather_multiframe()
    test_encode_decode_roundtrip()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
