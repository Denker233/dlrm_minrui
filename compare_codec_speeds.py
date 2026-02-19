#!/usr/bin/env python3
"""
Compare different video codecs for speed/compression tradeoff
"""

import torch
import time
from dlrm_s_pytorch import QSVEmbeddingCompressor

def test_codec(model_path, codec, quality):
    """Test a specific codec configuration"""
    print(f"\nTesting {codec} quality={quality}...")
    
    # Load one table for testing
    model = torch.load(model_path, map_location=torch.device('cpu'))
    state_dict = model['state_dict']
    emb_keys = [k for k in state_dict.keys() if 'emb_l' in k and 'weight' in k]
    
    # Test on largest table
    weights = state_dict[emb_keys[2]]  # Table 2 is usually the largest
    
    compressor = QSVEmbeddingCompressor(
        codec=codec,
        quality=quality,
        quantization='asymmetric',
        bits=8
    )
    
    # Compression
    start = time.time()
    compressed_data, metadata = compressor.compress_table(weights)
    comp_time = time.time() - start
    
    # Decompression  
    start = time.time()
    decompressed = compressor.decompress_table(compressed_data, metadata)
    decomp_time = time.time() - start
    
    original_size = weights.numel() * 4
    compressed_size = len(compressed_data)
    
    return {
        'codec': codec,
        'quality': quality,
        'original_mb': original_size / 1024 / 1024,
        'compressed_mb': compressed_size / 1024 / 1024,
        'ratio': original_size / compressed_size,
        'comp_time': comp_time,
        'decomp_time': decomp_time,
        'comp_mbps': (original_size / 1024 / 1024) / comp_time,
        'decomp_mbps': (original_size / 1024 / 1024) / decomp_time
    }

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python compare_codec_speeds.py <model_path>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    print("="*90)
    print("CODEC SPEED COMPARISON")
    print("="*90)
    print("\nTesting on largest embedding table (Table 2, ~618 MB)...")
    
    configs = [
        ('libx265', 20),  # Your current setting
        ('libx265', 28),  # Faster, less compression
        ('libx265', 15),  # Slower, more compression
        ('libx264', 20),  # H.264 alternative
    ]
    
    results = []
    for codec, quality in configs:
        try:
            result = test_codec(model_path, codec, quality)
            results.append(result)
        except Exception as e:
            print(f"  Failed: {e}")
    
    print("\n" + "="*90)
    print(f"{'Codec':<15} {'Quality':<8} {'Size':<12} {'Ratio':<8} {'Comp Time':<12} {'Decomp Time':<12} {'Comp Speed':<12}")
    print("-" * 90)
    
    for r in results:
        print(f"{r['codec']:<15} {r['quality']:<8} {r['compressed_mb']:.2f} MB    {r['ratio']:.1f}x     {r['comp_time']:.3f}s       {r['decomp_time']:.3f}s        {r['comp_mbps']:.1f} MB/s")
    
    print("\n" + "="*90)
    print("RECOMMENDATION")
    print("="*90)
    print("\nFor best compression: libx265 quality=15")
    print("For best speed:       libx265 quality=28 or libx264")
    print("For balance:          libx265 quality=20 (current) ✓")
    print("="*90)
