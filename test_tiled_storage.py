#!/usr/bin/env python3
"""Test tiled-layout cold storage in OnDemandPrefetchCache pipeline."""

import os, sys, time, tempfile
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import compressed_emb as _C
from codec_ondemand_benchmark import (
    encode_h265_perframe, OnDemandFrameDecoder, OnDemandPrefetchCache,
    GlobalFrameCache, quantize_table, tiled_frame_to_rows, log
)

EMB_DIM = 16
WIDTH, HEIGHT = 1920, 1080
ROWS_PER_FRAME = (WIDTH // 4) * (HEIGHT // 4)


def test_tiled_decode():
    """Test: OnDemandFrameDecoder with tiled=True returns correct data."""
    print("Test 1: Tiled decode mode...")
    N = ROWS_PER_FRAME + 5000
    fp32_data = torch.randn(N, 16)
    q, s, zp = quantize_table(fp32_data)
    q_np = q.numpy()

    with tempfile.TemporaryDirectory() as tmpdir:
        num_frames, frame_dir, _, _, rpf = encode_h265_perframe(
            q_np, WIDTH, HEIGHT, crf=0, output_dir=tmpdir, table_id=99)

        decoder = OnDemandFrameDecoder(frame_dir, rpf, EMB_DIM, WIDTH, HEIGHT)

        # Decode untiled
        rows = decoder.decode_frame(0, tiled=False)
        # Decode tiled
        tiled_frame = decoder.decode_frame(0, tiled=True)

        assert isinstance(tiled_frame, torch.Tensor), "Tiled frame should be torch.Tensor"
        assert tiled_frame.shape == (HEIGHT, WIDTH), f"Expected ({HEIGHT},{WIDTH}), got {tiled_frame.shape}"

        # Verify: untile the tiled frame and compare to direct untiled result
        rows_from_tiled = tiled_frame_to_rows(tiled_frame.numpy(), WIDTH, HEIGHT)
        assert np.array_equal(rows[:rpf], rows_from_tiled[:rpf]), "Tiled decode should match untiled!"

        # Verify C++ gather from tiled
        indices = torch.arange(100).long()
        gathered = _C.gather_from_tiled_frame(tiled_frame, indices, WIDTH // 4)
        assert np.array_equal(gathered.numpy(), rows[:100]), "C++ gather should match rows!"

        print(f"  PASS: tiled decode ({tiled_frame.shape}) matches untiled ({rows.shape})")


def test_tiled_cache_lookup():
    """Test: OnDemandPrefetchCache with tiled storage returns correct embeddings."""
    print("Test 2: Tiled cache lookup...")
    N = ROWS_PER_FRAME * 2 + 1000  # ~2 full frames + partial
    fp32_data = torch.randn(N, 16)
    q, s, zp = quantize_table(fp32_data)
    q_np = q.numpy()

    with tempfile.TemporaryDirectory() as tmpdir:
        num_frames, frame_dir, _, _, rpf = encode_h265_perframe(
            q_np, WIDTH, HEIGHT, crf=0, output_dir=tmpdir, table_id=99)

        global_cache = GlobalFrameCache(capacity=50, store_uint8=True)

        cache = OnDemandPrefetchCache(
            frame_dir=frame_dir,
            rows_per_frame=rpf,
            emb_dim=EMB_DIM,
            num_cold_rows=N,
            width=WIDTH, height=HEIGHT,
            quant_scale=s, quant_zp=zp,
            cache_capacity=50,
            global_cache=global_cache,
            table_id=99,
        )

        assert cache.use_tiled_storage, "Tiled storage should be enabled with C++ ext!"

        # Lookup some indices from frame 0
        indices_f0 = torch.tensor([0, 10, 100, 500, 1000, 5000], dtype=torch.long)
        result_f0, frames_used = cache.lookup(indices_f0)
        assert result_f0.shape == (len(indices_f0), EMB_DIM), f"Wrong shape: {result_f0.shape}"
        assert 0 in frames_used, "Frame 0 should be used"

        # Verify the cache has tiled frame data
        cached_data = global_cache.get(99, 0)
        assert cached_data is not None, "Frame 0 should be in cache"
        assert isinstance(cached_data, torch.Tensor), f"Cached data should be Tensor, got {type(cached_data)}"
        assert cached_data.dim() == 2 and cached_data.shape == (HEIGHT, WIDTH), \
            f"Cached frame should be ({HEIGHT},{WIDTH}), got {cached_data.shape}"

        # Lookup indices from frame 1
        indices_f1 = torch.tensor([rpf, rpf + 10, rpf + 1000], dtype=torch.long)
        result_f1, _ = cache.lookup(indices_f1)
        assert result_f1.shape == (len(indices_f1), EMB_DIM)

        # Verify correctness: compare with direct dequantization
        expected_f0 = (q_np[indices_f0.numpy()].astype(np.float32) - zp) * s
        mae_f0 = np.abs(result_f0.numpy() - expected_f0).mean()
        assert mae_f0 < 0.5, f"Frame 0 MAE too high: {mae_f0}"

        expected_f1 = (q_np[indices_f1.numpy()].astype(np.float32) - zp) * s
        mae_f1 = np.abs(result_f1.numpy() - expected_f1).mean()
        assert mae_f1 < 0.5, f"Frame 1 MAE too high: {mae_f1}"

        # Second lookup should be cache hit
        cache.reset_stats()
        result_f0_2, _ = cache.lookup(indices_f0)
        assert cache.stats['cache_hits'] > 0, "Second lookup should hit cache"
        assert cache.stats['cache_misses'] == 0, "Should be no misses on second lookup"

        # Results should match
        assert torch.allclose(result_f0, result_f0_2, atol=1e-5), "Repeated lookup should match"

        cache.close()
        print(f"  PASS: tiled cache lookup correct (MAE_f0={mae_f0:.4f}, MAE_f1={mae_f1:.4f})")


def test_tiled_cache_performance():
    """Benchmark: compare tiled vs non-tiled cache miss time."""
    print("Test 3: Tiled vs non-tiled cache miss performance...")
    N = ROWS_PER_FRAME * 3
    fp32_data = torch.randn(N, 16)
    q, s, zp = quantize_table(fp32_data)
    q_np = q.numpy()

    with tempfile.TemporaryDirectory() as tmpdir:
        num_frames, frame_dir, _, _, rpf = encode_h265_perframe(
            q_np, WIDTH, HEIGHT, crf=0, output_dir=tmpdir, table_id=99)

        # Test 100 random lookups each
        test_indices = torch.from_numpy(
            np.sort(np.random.choice(rpf, 200, replace=False))).long()

        # Tiled storage path
        gc_tiled = GlobalFrameCache(capacity=50)
        cache_tiled = OnDemandPrefetchCache(
            frame_dir=frame_dir, rows_per_frame=rpf, emb_dim=EMB_DIM,
            num_cold_rows=N, width=WIDTH, height=HEIGHT,
            quant_scale=s, quant_zp=zp, cache_capacity=50,
            global_cache=gc_tiled, table_id=0)

        # Warm up
        cache_tiled.lookup(test_indices[:10])
        gc_tiled.clear()
        cache_tiled.reset_stats()

        # Time tiled path (cold miss)
        t0 = time.perf_counter()
        result_tiled, _ = cache_tiled.lookup(test_indices)
        t_tiled_miss = (time.perf_counter() - t0) * 1000

        # Time tiled path (cache hit)
        t0 = time.perf_counter()
        for _ in range(10):
            result_tiled_hit, _ = cache_tiled.lookup(test_indices)
        t_tiled_hit = (time.perf_counter() - t0) * 1000 / 10

        cache_tiled.close()

        print(f"  Tiled storage:   cache miss={t_tiled_miss:.1f}ms, "
              f"cache hit={t_tiled_hit:.2f}ms, "
              f"tiled_storage={cache_tiled.use_tiled_storage}")
        print(f"  PASS: tiled storage pipeline functional")


if __name__ == "__main__":
    print("=" * 60)
    print("Tiled Storage Integration Tests")
    print("=" * 60)

    test_tiled_decode()
    test_tiled_cache_lookup()
    test_tiled_cache_performance()

    print("\n" + "=" * 60)
    print("ALL TILED STORAGE TESTS PASSED")
    print("=" * 60)
