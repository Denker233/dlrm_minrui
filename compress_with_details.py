#!/usr/bin/env python3
"""
Detailed compression with verbose output and progress tracking
"""

import argparse
import torch
import time
import os
import sys
import numpy as np

# Your imports
from dlrm_s_codecs import QSVEmbeddingCompressor

class VerboseCompressor:
    """Wrapper around QSVEmbeddingCompressor with detailed logging"""
    
    def __init__(self, codec='hevc_qsv', quality=23, quantization='asymmetric'):
        self.compressor = QSVEmbeddingCompressor(codec, quality, quantization, bits=8)
        self.stats = {
            'quantize_times': [],
            'encode_times': [],
            'total_times': [],
            'ratios': [],
            'original_sizes': [],
            'compressed_sizes': []
        }
    
    def compress_table_verbose(self, weights, table_idx):
        """Compress with detailed timing"""
        num_emb, emb_dim = weights.shape
        original_size_mb = (weights.element_size() * weights.numel()) / (1024**2)
        
        print(f"\n{'─'*80}")
        print(f"TABLE {table_idx} (C{table_idx+1})")
        print(f"{'─'*80}")
        print(f"  Shape:          {num_emb:,} rows × {emb_dim} cols")
        print(f"  Original size:  {original_size_mb:.2f} MB ({weights.element_size()} bytes/elem)")
        print(f"  Data range:     [{weights.min():.6f}, {weights.max():.6f}]")
        print(f"  Mean/Std:       {weights.mean():.6f} / {weights.std():.6f}")
        
        # Step 1: Quantization
        print(f"\n  [1/3] Quantizing (Float32 → uint8)...", end=' ', flush=True)
        quant_start = time.time()
        
        quantize_fn = self.compressor.quantizers[self.compressor.quantization][0]
        pixels_uint8, quant_metadata = quantize_fn(weights, bits=8)
        
        quant_time = time.time() - quant_start
        quantized_size_mb = pixels_uint8.numel() / (1024**2)
        
        print(f"done ({quant_time*1000:.1f}ms)")
        print(f"      Quantized size: {quantized_size_mb:.2f} MB")
        print(f"      Reduction: {original_size_mb/quantized_size_mb:.2f}x (expected 4x for float32→uint8)")
        
        # Step 2: QSV Encoding
        print(f"\n  [2/3] QSV encoding ({self.compressor.codec})...", end=' ', flush=True)
        encode_start = time.time()
        
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_file = os.path.join(tmpdir, 'pixels.raw')
            video_file = os.path.join(tmpdir, 'compressed.mp4')
            
            # Write raw
            pixels_uint8.numpy().tofile(raw_file)
            raw_size = os.path.getsize(raw_file) / (1024**2)
            
            # Encode
            self.compressor._encode_qsv(raw_file, video_file, emb_dim, num_emb)
            
            # Read compressed
            with open(video_file, 'rb') as f:
                compressed_data = f.read()
            
            compressed_size = len(compressed_data)
        
        encode_time = time.time() - encode_start
        compressed_size_mb = compressed_size / (1024**2)
        
        print(f"done ({encode_time*1000:.1f}ms)")
        print(f"      Raw file size: {raw_size:.2f} MB")
        print(f"      Compressed size: {compressed_size_mb:.2f} MB")
        print(f"      Codec compression: {raw_size/compressed_size_mb:.2f}x")
        
        # Step 3: Metadata
        metadata = {
            'shape': (num_emb, emb_dim),
            'quantization': self.compressor.quantization,
            'quant_params': quant_metadata,
            'bits': 8,
            'codec': self.compressor.codec,
            'quality': self.compressor.quality
        }
        
        # Overall stats
        total_time = quant_time + encode_time
        overall_ratio = original_size_mb / compressed_size_mb
        
        print(f"\n  [3/3] Summary:")
        print(f"      Total time:      {total_time*1000:.1f}ms")
        print(f"      Overall ratio:   {overall_ratio:.2f}x")
        print(f"      Throughput:      {original_size_mb/total_time:.1f} MB/s")
        
        # Store stats
        self.stats['quantize_times'].append(quant_time)
        self.stats['encode_times'].append(encode_time)
        self.stats['total_times'].append(total_time)
        self.stats['ratios'].append(overall_ratio)
        self.stats['original_sizes'].append(original_size_mb)
        self.stats['compressed_sizes'].append(compressed_size_mb)
        
        return compressed_data, metadata
    
    def print_summary(self):
        """Print overall compression summary"""
        print(f"\n{'='*80}")
        print(f"COMPRESSION SUMMARY")
        print(f"{'='*80}")
        
        total_orig = sum(self.stats['original_sizes'])
        total_comp = sum(self.stats['compressed_sizes'])
        total_time = sum(self.stats['total_times'])
        avg_ratio = np.mean(self.stats['ratios'])
        
        print(f"Tables processed:        {len(self.stats['ratios'])}")
        print(f"\nTotal size:")
        print(f"  Original:              {total_orig:.2f} MB")
        print(f"  Compressed:            {total_comp:.2f} MB")
        print(f"  Saved:                 {total_orig - total_comp:.2f} MB")
        print(f"\nCompression:")
        print(f"  Average ratio:         {avg_ratio:.2f}x")
        print(f"  Min ratio:             {min(self.stats['ratios']):.2f}x")
        print(f"  Max ratio:             {max(self.stats['ratios']):.2f}x")
        print(f"\nTiming:")
        print(f"  Total time:            {total_time:.2f}s")
        print(f"  Avg per table:         {total_time/len(self.stats['ratios']):.2f}s")
        print(f"  Time breakdown:")
        print(f"    Quantization:        {sum(self.stats['quantize_times']):.2f}s ({sum(self.stats['quantize_times'])/total_time*100:.1f}%)")
        print(f"    QSV encoding:        {sum(self.stats['encode_times']):.2f}s ({sum(self.stats['encode_times'])/total_time*100:.1f}%)")
        print(f"  Overall throughput:    {total_orig/total_time:.1f} MB/s")
        print(f"{'='*80}\n")


def compress_with_verification(model_path, output_path, **kwargs):
    """Compress and verify accuracy"""
    
    # Load model
    print(f"\n{'='*80}")
    print(f"LOADING MODEL")
    print(f"{'='*80}")
    print(f"File: {model_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    
    # Extract embeddings
    emb_tables = {}
    for key, value in state_dict.items():
        if 'emb_l' in key and 'weight' in key:
            parts = key.split('.')
            if len(parts) >= 2 and parts[1].isdigit():
                emb_tables[int(parts[1])] = value
    
    print(f"Found {len(emb_tables)} embedding tables")
    total_params = sum(v.numel() for v in emb_tables.values())
    total_size = sum(v.element_size() * v.numel() for v in emb_tables.values()) / (1024**2)
    print(f"Total parameters: {total_params:,}")
    print(f"Total size: {total_size:.2f} MB")
    
    # Create compressor
    compressor = VerboseCompressor(**kwargs)
    
    # Compress each table
    print(f"\n{'='*80}")
    print(f"COMPRESSING TABLES")
    print(f"{'='*80}")
    
    compressed_tables = {}
    accuracy_results = {}
    
    for table_idx in sorted(emb_tables.keys()):
        original_weights = emb_tables[table_idx]
        
        # Compress
        compressed_data, metadata = compressor.compress_table_verbose(
            original_weights, table_idx
        )
        
        # Verify by decompressing
        print(f"\n  [Verification] Decompressing to check accuracy...", end=' ', flush=True)
        verify_start = time.time()
        
        decompressed = compressor.compressor.decompress_table(compressed_data, metadata)
        
        # Calculate error
        mse = ((original_weights - decompressed) ** 2).mean().item()
        mae = (original_weights - decompressed).abs().mean().item()
        max_err = (original_weights - decompressed).abs().max().item()
        
        # Relative error
        relative_err = (original_weights - decompressed).norm().item() / original_weights.norm().item() * 100
        
        verify_time = time.time() - verify_start
        
        print(f"done ({verify_time*1000:.1f}ms)")
        print(f"      MSE:             {mse:.8f}")
        print(f"      MAE:             {mae:.8f}")
        print(f"      Max error:       {max_err:.8f}")
        print(f"      Relative error:  {relative_err:.4f}%")
        
        # Store
        compressed_tables[table_idx] = {
            'data': compressed_data,
            'metadata': metadata
        }
        
        accuracy_results[table_idx] = {
            'mse': mse,
            'mae': mae,
            'max_error': max_err,
            'relative_error': relative_err
        }
    
    # Print summary
    compressor.print_summary()
    
    # Accuracy summary
    print(f"{'='*80}")
    print(f"ACCURACY VERIFICATION")
    print(f"{'='*80}")
    avg_mse = np.mean([r['mse'] for r in accuracy_results.values()])
    avg_relative = np.mean([r['relative_error'] for r in accuracy_results.values()])
    max_relative = max([r['relative_error'] for r in accuracy_results.values()])
    
    print(f"Average MSE:           {avg_mse:.8f}")
    print(f"Average relative err:  {avg_relative:.4f}%")
    print(f"Max relative err:      {max_relative:.4f}%")
    print(f"{'='*80}\n")
    
    # Save compressed model
    print(f"Saving compressed model to: {output_path}")
    
    compressed_checkpoint = {
        'compressed_tables': compressed_tables,
        'compression_info': {
            'codec': kwargs.get('codec', 'hevc_qsv'),
            'quality': kwargs.get('quality', 23),
            'quantization': kwargs.get('quantization', 'asymmetric'),
            'stats': compressor.stats,
            'accuracy': accuracy_results
        }
    }
    
    torch.save(compressed_checkpoint, output_path)
    
    saved_size = os.path.getsize(output_path) / (1024**2)
    original_size = os.path.getsize(model_path) / (1024**2)
    
    print(f"\nFile sizes:")
    print(f"  Original:   {original_size:.2f} MB")
    print(f"  Compressed: {saved_size:.2f} MB")
    print(f"  Reduction:  {(1 - saved_size/original_size)*100:.1f}%")
    print(f"\nDone!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Trained model path')
    parser.add_argument('--output', required=True, help='Output compressed model path')
    parser.add_argument('--codec', default='libx265', choices=['h264_qsv', 'hevc_qsv', 'libx264', 'libx265'])
    parser.add_argument('--quality', type=int, default=23, help='18-28, lower=better')
    parser.add_argument('--quantization', default='asymmetric', 
                       choices=['asymmetric', 'per_row'])
    
    args = parser.parse_args()
    
    compress_with_verification(
        args.model,
        args.output,
        codec=args.codec,
        quality=args.quality,
        quantization=args.quantization
    )
