#!/usr/bin/env python3
"""
Benchmark inference time: Original vs Compressed model
Uses your actual test dataset and parameters
"""

import torch
import numpy as np
import argparse
import time
import sys
import subprocess
import tempfile
import os
import dlrm_data_pytorch as dp
from dlrm_s_pytorch import DLRM_Net

def decompress_embedding_tables(compressed_path):
    """Decompress all embedding tables from compressed model"""
    print("Loading compressed model...")
    compressed = torch.load(compressed_path, map_location='cpu')
    compressed_tables = compressed['compressed_tables']
    
    print(f"Decompressing {len(compressed_tables)} embedding tables...")
    start_time = time.time()
    
    decompressed_weights = {}
    
    for table_idx in sorted(compressed_tables.keys()):
        table_data = compressed_tables[table_idx]
        compressed_data = table_data['data']
        metadata = table_data['metadata']
        
        # Decompress this table
        weights = decompress_table(compressed_data, metadata)
        decompressed_weights[f'emb_l.{table_idx}.weight'] = torch.from_numpy(weights)
        print(f"  Table {table_idx}: {weights.shape}")
    
    decomp_time = time.time() - start_time
    print(f"Total decompression time: {decomp_time:.2f}s")
    print()
    
    return decompressed_weights, decomp_time

def decompress_table(compressed_data, metadata):
    """Decompress a single table"""
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        f.write(compressed_data)
        video_path = f.name
    
    try:
        width = metadata['width']
        height = metadata['height']
        
        cmd = [
            'ffmpeg', '-loglevel', 'quiet',
            '-i', video_path,
            '-f', 'rawvideo',
            '-pix_fmt', 'gray',
            'pipe:1'
        ]
        
        result = subprocess.run(cmd, capture_output=True, check=True)
        raw_pixels = np.frombuffer(result.stdout, dtype=np.uint8)
        pixels = raw_pixels.reshape((height, width))
        
        # Untile if needed
        if metadata.get('tiled', False):
            tile_size = metadata.get('tile_size', 4)
            h, w = pixels.shape
            num_tiles_h = h // tile_size
            num_tiles_w = w // tile_size
            
            embeddings = []
            for i in range(num_tiles_h):
                for j in range(num_tiles_w):
                    tile = pixels[
                        i*tile_size:(i+1)*tile_size,
                        j*tile_size:(j+1)*tile_size
                    ]
                    embeddings.append(tile.flatten())
            
            embeddings = np.array(embeddings)
            embeddings = embeddings[:metadata['num_embeddings'], :metadata['embedding_dim']]
        else:
            embeddings = pixels.flatten()[:metadata['num_embeddings'] * metadata['embedding_dim']]
            embeddings = embeddings.reshape(metadata['num_embeddings'], metadata['embedding_dim'])
        
        # Dequantize
        scale = metadata['scale']
        zero_point = metadata['zero_point']
        dequantized = (embeddings.astype(np.float32) - zero_point) * scale
        
        return dequantized
        
    finally:
        os.unlink(video_path)

def benchmark_inference(model, test_ld, device, model_name):
    """Benchmark inference time"""
    print(f"\n{'='*80}")
    print(f"BENCHMARKING: {model_name}")
    print(f"{'='*80}")
    
    model.eval()
    
    total_batches = len(test_ld)
    print(f"Total test batches: {total_batches}")
    print(f"Batch size: {test_ld.batch_size}")
    print(f"Total samples: {total_batches * test_ld.batch_size:,}")
    print()
    
    # Warmup (first 10 batches)
    print("Warming up (10 batches)...")
    with torch.no_grad():
        for j, (X, lS_o, lS_i, T) in enumerate(test_ld):
            if j >= 10:
                break
            _ = model(X.to(device), lS_o, lS_i)
    print("Warmup complete\n")
    
    # Actual benchmark
    print("Running full inference...")
    batch_times = []
    correct = 0
    total = 0
    
    start_time = time.time()
    
    with torch.no_grad():
        for j, (X, lS_o, lS_i, T) in enumerate(test_ld):
            batch_start = time.time()
            
            Z = model(X.to(device), lS_o, lS_i)
            
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            
            # Calculate accuracy
            predictions = (torch.sigmoid(Z) > 0.5).float()
            correct += (predictions.squeeze() == T.to(device)).sum().item()
            total += T.size(0)
            
            if (j + 1) % 100 == 0:
                elapsed = time.time() - start_time
                progress = (j + 1) / total_batches * 100
                print(f"  Progress: {j+1}/{total_batches} ({progress:.1f}%) - "
                      f"Elapsed: {elapsed:.1f}s")
    
    total_time = time.time() - start_time
    accuracy = 100.0 * correct / total
    
    # Statistics
    batch_times = np.array(batch_times)
    mean_batch_time = np.mean(batch_times) * 1000  # ms
    std_batch_time = np.std(batch_times) * 1000
    min_batch_time = np.min(batch_times) * 1000
    max_batch_time = np.max(batch_times) * 1000
    p50_batch_time = np.percentile(batch_times, 50) * 1000
    p95_batch_time = np.percentile(batch_times, 95) * 1000
    p99_batch_time = np.percentile(batch_times, 99) * 1000
    
    throughput = total / total_time  # samples/sec
    
    print()
    print(f"{'='*80}")
    print(f"RESULTS: {model_name}")
    print(f"{'='*80}")
    print(f"Total inference time:  {total_time:.2f}s")
    print(f"Total samples:         {total:,}")
    print(f"Accuracy:              {accuracy:.4f}%")
    print()
    print(f"Per-batch timing (ms):")
    print(f"  Mean:    {mean_batch_time:.2f} ± {std_batch_time:.2f} ms")
    print(f"  Min:     {min_batch_time:.2f} ms")
    print(f"  Max:     {max_batch_time:.2f} ms")
    print(f"  Median:  {p50_batch_time:.2f} ms")
    print(f"  P95:     {p95_batch_time:.2f} ms")
    print(f"  P99:     {p99_batch_time:.2f} ms")
    print()
    print(f"Throughput:            {throughput:,.0f} samples/sec")
    print(f"Time per sample:       {1000*total_time/total:.4f} ms")
    print(f"{'='*80}")
    print()
    
    return {
        'total_time': total_time,
        'accuracy': accuracy,
        'throughput': throughput,
        'mean_batch_time_ms': mean_batch_time,
        'std_batch_time_ms': std_batch_time,
        'p50_batch_time_ms': p50_batch_time,
        'p95_batch_time_ms': p95_batch_time,
        'p99_batch_time_ms': p99_batch_time,
        'samples_per_sec': throughput
    }

def main():
    parser = argparse.ArgumentParser()
    
    # Architecture
    parser.add_argument("--arch-sparse-feature-size", type=int, default=16)
    parser.add_argument("--arch-mlp-bot", type=str, default="13-512-256-64-16")
    parser.add_argument("--arch-mlp-top", type=str, default="512-256-1")
    parser.add_argument("--arch-interaction-op", type=str, default="dot")
    parser.add_argument("--arch-interaction-itself", action="store_true", default=False)
    
    # Data
    parser.add_argument("--data-generation", type=str, default="dataset")
    parser.add_argument("--data-set", type=str, default="kaggle")
    parser.add_argument("--raw-data-file", type=str, default="./input/train.txt")
    parser.add_argument("--processed-data-file", type=str, default="./input/kaggleAdDisplayChallenge_processed.npz")
    parser.add_argument("--data-randomize", type=str, default="total")
    parser.add_argument("--data-trace-file", type=str, default="./input/trace.txt")
    parser.add_argument("--data-sub-sample-rate", type=float, default=0.0)
    parser.add_argument("--num-indices-per-lookup", type=int, default=10)
    parser.add_argument("--num-indices-per-lookup-fixed", type=bool, default=False)
    parser.add_argument("--memory-map", action="store_true", default=False)
    
    # Training (not used for inference but required by data loader)
    parser.add_argument("--mini-batch-size", type=int, default=128)
    parser.add_argument("--nepochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--print-precision", type=int, default=5)
    parser.add_argument("--numpy-rand-seed", type=int, default=123)
    parser.add_argument("--sync-dense-params", type=bool, default=True)
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--dataset-multiprocessing", action="store_true", default=False)
    parser.add_argument("--num-workers", type=int, default=0)
    
    # Testing
    parser.add_argument("--loss-function", type=str, default="bce")
    parser.add_argument("--round-targets", type=bool, default=True)
    parser.add_argument("--test-mini-batch-size", type=int, default=2048)
    parser.add_argument("--test-num-workers", type=int, default=0)
    
    # MLPerf logging
    parser.add_argument("--mlperf-logging", action="store_true", default=False)
    parser.add_argument("--mlperf-auc-threshold", type=float, default=0.0)
    parser.add_argument("--mlperf-bin-loader", action="store_true", default=False)
    parser.add_argument("--mlperf-bin-shuffle", action="store_true", default=False)
    
    # Models
    parser.add_argument("--original-model", type=str, default="./models/dlrm_kaggle_quick.pt")
    parser.add_argument("--compressed-model", type=str, default="./models/dlrm_kaggle_quick_compressed.pt")
    parser.add_argument("--use-gpu", action="store_true", default=False)
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device("cuda:0" if args.use_gpu and torch.cuda.is_available() else "cpu")
    
    print("="*80)
    print("DLRM INFERENCE TIME BENCHMARK")
    print("="*80)
    print(f"Device: {device}")
    print(f"Original model:   {args.original_model}")
    print(f"Compressed model: {args.compressed_model}")
    print()
    
    # Check files exist
    if not os.path.exists(args.original_model):
        print(f"ERROR: Original model not found at {args.original_model}")
        sys.exit(1)
    
    if not os.path.exists(args.compressed_model):
        print(f"ERROR: Compressed model not found at {args.compressed_model}")
        sys.exit(1)
    
    # Load test data
    print("Loading test data...")
    train_data, train_ld, test_data, test_ld = dp.make_criteo_data_and_loaders(args)
    print(f"Test dataset: {len(test_data)} samples")
    print(f"Test batches: {len(test_ld)} batches")
    print()
    
    # Get model structure info
    ln_emb = np.array(train_data.counts)
    m_spa = args.arch_sparse_feature_size
    ln_bot = np.fromstring(args.arch_mlp_bot, dtype=int, sep="-")
    ln_bot[0] = train_data.m_den
    
    num_fea = ln_emb.size + 1
    m_den_out = ln_bot[ln_bot.size - 1]
    
    if args.arch_interaction_op == "dot":
        if args.arch_interaction_itself:
            num_int = (num_fea * (num_fea + 1)) // 2 + m_den_out
        else:
            num_int = (num_fea * (num_fea - 1)) // 2 + m_den_out
    else:
        num_int = num_fea * m_den_out
    
    arch_mlp_top_adjusted = str(num_int) + "-" + args.arch_mlp_top
    ln_top = np.fromstring(arch_mlp_top_adjusted, dtype=int, sep="-")
    
    # ========================================
    # SCENARIO 1: Original Model
    # ========================================
    print("\n" + "="*80)
    print("SCENARIO 1: ORIGINAL FP32 MODEL")
    print("="*80)
    
    print("\nLoading original model...")
    load_start = time.time()
    original_checkpoint = torch.load(args.original_model, map_location='cpu')
    original_load_time = time.time() - load_start
    print(f"Loaded in {original_load_time:.2f}s")
    
    # Create model
    original_model = DLRM_Net(
        m_spa, ln_emb, ln_bot, ln_top,
        arch_interaction_op=args.arch_interaction_op,
        arch_interaction_itself=args.arch_interaction_itself,
        sigmoid_bot=-1, sigmoid_top=ln_top.size - 2,
        loss_threshold=0.0, ndevices=1
    ).to(device)
    
    original_model.load_state_dict(original_checkpoint['state_dict'])
    
    # Benchmark
    original_results = benchmark_inference(original_model, test_ld, device, "Original FP32")
    
    # ========================================
    # SCENARIO 2: Compressed Model
    # ========================================
    print("\n" + "="*80)
    print("SCENARIO 2: COMPRESSED MODEL (DECOMPRESSED)")
    print("="*80)
    
    print("\nLoading compressed model...")
    comp_load_start = time.time()
    compressed_checkpoint = torch.load(args.compressed_model, map_location='cpu')
    comp_load_time = time.time() - comp_load_start
    print(f"Loaded in {comp_load_time:.2f}s")
    
    # Decompress embeddings
    decompressed_weights, decomp_time = decompress_embedding_tables(args.compressed_model)
    total_load_time = comp_load_time + decomp_time
    
    # Create model
    compressed_model = DLRM_Net(
        m_spa, ln_emb, ln_bot, ln_top,
        arch_interaction_op=args.arch_interaction_op,
        arch_interaction_itself=args.arch_interaction_itself,
        sigmoid_bot=-1, sigmoid_top=ln_top.size - 2,
        loss_threshold=0.0, ndevices=1
    ).to(device)
    
    # Load original state dict
    state_dict = original_checkpoint['state_dict'].copy()
    
    # Replace with decompressed embeddings
    for key, weights in decompressed_weights.items():
        state_dict[key] = weights
    
    compressed_model.load_state_dict(state_dict)
    
    # Benchmark
    compressed_results = benchmark_inference(compressed_model, test_ld, device, "Compressed (Decompressed)")
    
    # ========================================
    # COMPARISON
    # ========================================
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print()
    
    print("MODEL LOADING:")
    print("-"*80)
    print(f"  Original:")
    print(f"    Load time:        {original_load_time:.2f}s")
    print()
    print(f"  Compressed:")
    print(f"    Load time:        {comp_load_time:.2f}s")
    print(f"    Decompress time:  {decomp_time:.2f}s")
    print(f"    Total:            {total_load_time:.2f}s")
    print()
    
    if total_load_time > original_load_time:
        print(f"  → Compressed is {total_load_time/original_load_time:.2f}x SLOWER to load "
              f"({total_load_time - original_load_time:.2f}s extra)")
    else:
        print(f"  → Compressed is {original_load_time/total_load_time:.2f}x FASTER to load")
    print()
    
    print("INFERENCE PERFORMANCE:")
    print("-"*80)
    print(f"  {'Metric':<30} {'Original':<20} {'Compressed':<20} {'Difference':<15}")
    print("-"*80)
    
    metrics = [
        ('Total time (s)', 'total_time', 's'),
        ('Accuracy (%)', 'accuracy', '%'),
        ('Throughput (samples/s)', 'samples_per_sec', 'samp/s'),
        ('Mean batch time (ms)', 'mean_batch_time_ms', 'ms'),
        ('P50 batch time (ms)', 'p50_batch_time_ms', 'ms'),
        ('P95 batch time (ms)', 'p95_batch_time_ms', 'ms'),
    ]
    
    for label, key, unit in metrics:
        orig_val = original_results[key]
        comp_val = compressed_results[key]
        diff_pct = (comp_val / orig_val - 1) * 100
        
        if key == 'accuracy':
            diff_abs = comp_val - orig_val
            print(f"  {label:<30} {orig_val:>19.4f} {comp_val:>19.4f} {diff_abs:>+13.4f}%")
        else:
            print(f"  {label:<30} {orig_val:>19,.2f} {comp_val:>19,.2f} {diff_pct:>+13.2f}%")
    
    print()
    
    # Key insights
    inference_overhead = (compressed_results['total_time'] / original_results['total_time'] - 1) * 100
    accuracy_diff = compressed_results['accuracy'] - original_results['accuracy']
    
    print("KEY INSIGHTS:")
    print("-"*80)
    
    if abs(inference_overhead) < 1:
        print(f"  ✓ ZERO INFERENCE OVERHEAD! ({inference_overhead:+.2f}%)")
        print("    Both models run at IDENTICAL speed after decompression")
    elif abs(inference_overhead) < 5:
        print(f"  ✓ Minimal overhead: {inference_overhead:+.2f}%")
    else:
        print(f"  ⚠ Some overhead: {inference_overhead:+.2f}%")
    
    print()
    print(f"  Accuracy difference: {accuracy_diff:+.4f}%")
    
    if abs(accuracy_diff) < 0.1:
        print("  ✓ Accuracy preserved! (<0.1% difference)")
    
    print()
    
    # Model sizes
    orig_size = os.path.getsize(args.original_model) / (1024**2)
    comp_size = os.path.getsize(args.compressed_model) / (1024**2)
    
    print(f"  Model sizes:")
    print(f"    Original:   {orig_size:,.2f} MB")
    print(f"    Compressed: {comp_size:,.2f} MB")
    print(f"    Ratio:      {orig_size/comp_size:.1f}x smaller")
    print()
    
    print(f"  Trade-off:")
    print(f"    Storage:    {orig_size/comp_size:.0f}x smaller ✓✓✓")
    print(f"    Loading:    {total_load_time/original_load_time:.2f}x slower")
    print(f"    Inference:  {abs(inference_overhead):.2f}% difference ✓")
    print(f"    Accuracy:   {accuracy_diff:+.4f}% difference ✓")
    
    print()
    print("="*80)
    print("BENCHMARK COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()

