#!/usr/bin/env python3
"""
Benchmark DLRM inference with batch size 16384
Comparing original FP32 vs compressed (decompressed) model
"""

import torch
import numpy as np
import time
import sys
import subprocess
import tempfile
import os
from pathlib import Path


class DLRMInferenceBenchmark:
    """Benchmark DLRM inference at batch size 16384"""
    
    def __init__(self, original_path, compressed_path, batch_size=16384):
        self.original_path = original_path
        self.compressed_path = compressed_path
        self.batch_size = batch_size
        
        print("="*80)
        print("DLRM INFERENCE BENCHMARK - BATCH SIZE 16384")
        print("="*80)
        print()
        print(f"Original model:   {original_path}")
        print(f"Compressed model: {compressed_path}")
        print(f"Batch size:       {batch_size}")
        print()
    
    def load_original_model(self):
        """Load original FP32 model"""
        print("="*80)
        print("LOADING ORIGINAL FP32 MODEL")
        print("="*80)
        print()
        
        start = time.time()
        checkpoint = torch.load(self.original_path, map_location='cpu')
        load_time = time.time() - start
        
        print(f"Loaded in {load_time:.3f} seconds")
        
        # Get model size
        file_size = os.path.getsize(self.original_path) / (1024**2)
        print(f"File size: {file_size:.2f} MB")
        print()
        
        return checkpoint, load_time
    
    def load_and_decompress_model(self):
        """Load compressed model and decompress all embeddings"""
        print("="*80)
        print("LOADING AND DECOMPRESSING MODEL")
        print("="*80)
        print()
        
        # Load compressed file
        print("Step 1: Loading compressed file...")
        load_start = time.time()
        compressed = torch.load(self.compressed_path, map_location='cpu')
        load_time = time.time() - load_start
        
        comp_file_size = os.path.getsize(self.compressed_path) / (1024**2)
        print(f"  Loaded in {load_time:.3f} seconds")
        print(f"  File size: {comp_file_size:.2f} MB")
        print()
        
        # Load original for structure
        original = torch.load(self.original_path, map_location='cpu')
        
        # Decompress embeddings
        print("Step 2: Decompressing embedding tables...")
        decomp_start = time.time()
        
        decompressed = {'state_dict': {}}
        
        # Copy non-embedding layers
        for key in original['state_dict'].keys():
            if 'emb_l' not in key:
                decompressed['state_dict'][key] = original['state_dict'][key]
        
        # Get compressed tables (format: compressed_tables[table_idx] = {'data': bytes, 'metadata': dict})
        compressed_tables = compressed['compressed_tables']
        
        total_original_size = 0
        total_compressed_size = 0
        
        for table_idx in sorted(compressed_tables.keys()):
            table_data = compressed_tables[table_idx]
            compressed_data = table_data['data']
            metadata = table_data['metadata']
            
            table_start = time.time()
            weights = self._decompress_table(compressed_data, metadata)
            table_time = time.time() - table_start
            
            decompressed['state_dict'][f'emb_l.{table_idx}.weight'] = torch.from_numpy(weights)
            
            orig_size = weights.nbytes / (1024**2)
            comp_size = len(compressed_data) / (1024**2)
            total_original_size += orig_size
            total_compressed_size += comp_size
            
            print(f"  Table {table_idx:2d}: {weights.shape[0]:8d}×{weights.shape[1]:2d} "
                  f"({orig_size:7.2f} MB → {comp_size:6.3f} MB) in {table_time:.3f}s")
        
        decomp_time = time.time() - decomp_start
        total_time = time.time() - load_start
        
        print()
        print(f"Decompression complete in {decomp_time:.3f} seconds")
        print(f"Total load+decompress: {total_time:.3f} seconds")
        print(f"Total size: {total_original_size:.2f} MB (decompressed) from {total_compressed_size:.2f} MB (compressed)")
        if total_compressed_size > 0:
            print(f"Compression ratio: {total_original_size/total_compressed_size:.1f}x")
        print()
        
        return decompressed, load_time, decomp_time, total_time
    
    def _decompress_table(self, compressed_data, metadata):
        """Decompress a single embedding table"""
        # Write compressed to temp file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            f.write(compressed_data)
            video_path = f.name
        
        try:
            # Decode video
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
                embeddings = self._untile(pixels, tile_size, 
                                         metadata['num_embeddings'],
                                         metadata['embedding_dim'])
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
    
    def _untile(self, tiled_image, tile_size, num_embeddings, embedding_dim):
        """Untile image back to embeddings"""
        h, w = tiled_image.shape
        num_tiles_h = h // tile_size
        num_tiles_w = w // tile_size
        
        embeddings = []
        for i in range(num_tiles_h):
            for j in range(num_tiles_w):
                tile = tiled_image[
                    i*tile_size:(i+1)*tile_size,
                    j*tile_size:(j+1)*tile_size
                ]
                embeddings.append(tile.flatten())
        
        embeddings = np.array(embeddings)
        return embeddings[:num_embeddings, :embedding_dim]
    
    def create_synthetic_batch(self, emb_tables):
        """Create synthetic test batch"""
        # Random categorical indices
        sparse_indices = []
        for table in emb_tables:
            num_embeddings = table.shape[0]
            indices = torch.randint(0, num_embeddings, (self.batch_size,))
            sparse_indices.append(indices)
        
        # Random dense features (13 features)
        dense_x = torch.randn(self.batch_size, 13)
        
        return dense_x, sparse_indices
    
    def run_inference_benchmark(self, model, num_iterations=100):
        """
        Run inference benchmark
        
        Args:
            model: Model checkpoint with state_dict
            num_iterations: Number of inference iterations
        """
        print("="*80)
        print("RUNNING INFERENCE BENCHMARK")
        print("="*80)
        print()
        print(f"Batch size:  {self.batch_size}")
        print(f"Iterations:  {num_iterations}")
        print(f"Total samples: {self.batch_size * num_iterations:,}")
        print()
        
        # Extract embedding tables
        state_dict = model['state_dict']
        emb_tables = []
        for i in range(26):
            key = f'emb_l.{i}.weight'
            if key in state_dict:
                emb_tables.append(state_dict[key])
        
        print(f"Found {len(emb_tables)} embedding tables")
        print()
        
        # Warmup
        print("Warming up (10 iterations)...")
        for _ in range(10):
            dense_x, sparse_indices = self.create_synthetic_batch(emb_tables)
            _ = self._forward_embeddings(emb_tables, dense_x, sparse_indices)
        print("Warmup complete")
        print()
        
        # Actual benchmark
        print(f"Running {num_iterations} inference iterations...")
        latencies = []
        
        for i in range(num_iterations):
            dense_x, sparse_indices = self.create_synthetic_batch(emb_tables)
            
            start = time.time()
            _ = self._forward_embeddings(emb_tables, dense_x, sparse_indices)
            latency = (time.time() - start) * 1000  # Convert to ms
            latencies.append(latency)
            
            if (i + 1) % 10 == 0:
                print(f"  Completed {i+1}/{num_iterations} iterations...")
        
        print()
        
        # Calculate statistics
        latencies = np.array(latencies)
        results = {
            'batch_size': self.batch_size,
            'num_iterations': num_iterations,
            'mean_latency_ms': np.mean(latencies),
            'std_latency_ms': np.std(latencies),
            'min_latency_ms': np.min(latencies),
            'max_latency_ms': np.max(latencies),
            'p50_latency_ms': np.percentile(latencies, 50),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'throughput_samples_per_sec': (self.batch_size * 1000) / np.mean(latencies),
            'latencies': latencies
        }
        
        return results
    
    def _forward_embeddings(self, emb_tables, dense_x, sparse_indices):
        """Forward pass through embeddings (simplified DLRM)"""
        # Embedding lookups
        sparse_embeddings = []
        for i, indices in enumerate(sparse_indices):
            emb = torch.nn.functional.embedding(indices, emb_tables[i])
            sparse_embeddings.append(emb)
        
        # Concatenate all embeddings
        result = torch.cat(sparse_embeddings, dim=1)
        return result
    
    def print_results(self, results, label):
        """Print benchmark results"""
        print(f"Results for {label}:")
        print(f"  Batch size:        {results['batch_size']:,}")
        print(f"  Mean latency:      {results['mean_latency_ms']:.2f} ± {results['std_latency_ms']:.2f} ms")
        print(f"  Min latency:       {results['min_latency_ms']:.2f} ms")
        print(f"  Max latency:       {results['max_latency_ms']:.2f} ms")
        print(f"  Median (P50):      {results['p50_latency_ms']:.2f} ms")
        print(f"  P95 latency:       {results['p95_latency_ms']:.2f} ms")
        print(f"  P99 latency:       {results['p99_latency_ms']:.2f} ms")
        print(f"  Throughput:        {results['throughput_samples_per_sec']:,.0f} samples/sec")
        print(f"  Time per sample:   {results['mean_latency_ms'] / results['batch_size']:.4f} ms")
        print()
    
    def run_full_benchmark(self, num_iterations=100):
        """Run complete benchmark"""
        
        # Scenario 1: Original model
        print()
        print("╔"+"="*78+"╗")
        print("║" + " "*20 + "SCENARIO 1: ORIGINAL FP32 MODEL" + " "*27 + "║")
        print("╚"+"="*78+"╝")
        print()
        
        original_model, original_load_time = self.load_original_model()
        original_results = self.run_inference_benchmark(original_model, num_iterations)
        self.print_results(original_results, "Original FP32 Model")
        
        # Scenario 2: Compressed model
        print()
        print("╔"+"="*78+"╗")
        print("║" + " "*15 + "SCENARIO 2: COMPRESSED MODEL (DECOMPRESSED)" + " "*20 + "║")
        print("╚"+"="*78+"╝")
        print()
        
        decompressed_model, comp_load_time, decomp_time, total_load_time = \
            self.load_and_decompress_model()
        compressed_results = self.run_inference_benchmark(decompressed_model, num_iterations)
        self.print_results(compressed_results, "Compressed Model (Decompressed)")
        
        # Comparison
        print()
        print("╔"+"="*78+"╗")
        print("║" + " "*25 + "COMPARISON SUMMARY" + " "*35 + "║")
        print("╚"+"="*78+"╝")
        print()
        
        print("MODEL LOADING:")
        print("-"*80)
        print(f"  Original model:")
        print(f"    Load time:              {original_load_time:.3f}s")
        print()
        print(f"  Compressed model:")
        print(f"    Load time:              {comp_load_time:.3f}s")
        print(f"    Decompression time:     {decomp_time:.3f}s")
        print(f"    Total time:             {total_load_time:.3f}s")
        print()
        
        if total_load_time > original_load_time:
            slowdown = total_load_time / original_load_time
            print(f"  → Compressed is {slowdown:.2f}x SLOWER to load ({total_load_time-original_load_time:.2f}s extra)")
        else:
            speedup = original_load_time / total_load_time
            print(f"  → Compressed is {speedup:.2f}x FASTER to load!")
        print()
        
        print("INFERENCE PERFORMANCE (After Loading):")
        print("-"*80)
        print(f"  {'Metric':<30} {'Original':<20} {'Compressed':<20} {'Difference':<15}")
        print("-"*80)
        
        metrics = [
            ('Mean latency (ms)', 'mean_latency_ms', 'ms'),
            ('P50 latency (ms)', 'p50_latency_ms', 'ms'),
            ('P95 latency (ms)', 'p95_latency_ms', 'ms'),
            ('P99 latency (ms)', 'p99_latency_ms', 'ms'),
            ('Throughput (samples/s)', 'throughput_samples_per_sec', 'samp/s'),
        ]
        
        for label, key, unit in metrics:
            orig_val = original_results[key]
            comp_val = compressed_results[key]
            
            if 'throughput' in key.lower():
                diff = comp_val - orig_val
                diff_pct = (comp_val / orig_val - 1) * 100
            else:
                diff = comp_val - orig_val
                diff_pct = (comp_val / orig_val - 1) * 100
            
            print(f"  {label:<30} {orig_val:>19,.2f} {comp_val:>19,.2f} {diff_pct:>+13.2f}%")
        
        print()
        
        # Key insight
        inference_overhead = (compressed_results['mean_latency_ms'] / 
                             original_results['mean_latency_ms'] - 1) * 100
        
        print("KEY INSIGHTS:")
        print("-"*80)
        
        if abs(inference_overhead) < 1:
            print("  ✓ ZERO INFERENCE OVERHEAD!")
            print(f"    Difference: {inference_overhead:+.2f}%")
            print("    → After decompression, both models run at IDENTICAL speed")
        elif abs(inference_overhead) < 5:
            print(f"  ✓ Minimal inference overhead: {inference_overhead:+.2f}%")
            print("    → Negligible performance difference")
        else:
            print(f"  ⚠ Some inference overhead: {inference_overhead:+.2f}%")
        
        print()
        print(f"  Model sizes:")
        orig_size = os.path.getsize(self.original_path) / (1024**2)
        comp_size = os.path.getsize(self.compressed_path) / (1024**2)
        print(f"    Original:   {orig_size:,.2f} MB")
        print(f"    Compressed: {comp_size:,.2f} MB")
        print(f"    Ratio:      {orig_size/comp_size:.1f}x smaller")
        
        print()
        print(f"  Trade-off:")
        print(f"    Storage:  {orig_size/comp_size:.0f}x smaller ✓✓✓")
        print(f"    Loading:  {total_load_time/original_load_time:.2f}x slower")
        print(f"    Inference: {abs(inference_overhead):.2f}% difference ✓")
        
        print()
        
        return {
            'original': {
                'load_time': original_load_time,
                'inference': original_results
            },
            'compressed': {
                'load_time': comp_load_time,
                'decomp_time': decomp_time,
                'total_load_time': total_load_time,
                'inference': compressed_results
            }
        }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark DLRM inference with batch size 16384')
    parser.add_argument('--original-model', type=str, default='./models/dlrm_kaggle_quick.pt',
                       help='Path to original model')
    parser.add_argument('--compressed-model', type=str, default='./models/dlrm_kaggle_quick_compressed.pt',
                       help='Path to compressed model')
    parser.add_argument('--batch-size', type=int, default=16384,
                       help='Batch size for inference (default: 16384)')
    parser.add_argument('--num-iterations', type=int, default=100,
                       help='Number of inference iterations (default: 100)')
    
    args = parser.parse_args()
    
    # Check files exist
    if not Path(args.original_model).exists():
        print(f"Error: Original model not found at {args.original_model}")
        sys.exit(1)
    
    if not Path(args.compressed_model).exists():
        print(f"Error: Compressed model not found at {args.compressed_model}")
        sys.exit(1)
    
    # Run benchmark
    benchmark = DLRMInferenceBenchmark(
        args.original_model,
        args.compressed_model,
        batch_size=args.batch_size
    )
    
    results = benchmark.run_full_benchmark(num_iterations=args.num_iterations)
    
    print()
    print("="*80)
    print("BENCHMARK COMPLETE!")
    print("="*80)
    print()

