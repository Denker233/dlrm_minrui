#!/usr/bin/env python3
"""
Convert .npz files to binary files and measure decompression time.

This script measures the overhead of loading .npz (compressed numpy) files
vs raw binary files to understand if data format affects inference benchmarks.

Usage:
    python convert_npz_to_binary.py --input-dir /path/to/npz --output-dir /path/to/binary
"""

import numpy as np
import os
import sys
import time
import argparse
from pathlib import Path


def measure_npz_load_time(npz_path: str) -> tuple:
    """
    Load an .npz file and measure the time it takes.
    Returns (load_time_seconds, data_dict, file_size_bytes)
    """
    file_size = os.path.getsize(npz_path)
    
    start_time = time.perf_counter()
    data = np.load(npz_path)
    # Force loading all arrays (np.load is lazy by default)
    loaded_data = {key: data[key] for key in data.keys()}
    end_time = time.perf_counter()
    
    return end_time - start_time, loaded_data, file_size


def measure_binary_load_time(bin_path: str, shape: tuple, dtype: np.dtype) -> tuple:
    """
    Load a binary file using memmap and measure the time.
    Returns (load_time_seconds, data_array, file_size_bytes)
    """
    file_size = os.path.getsize(bin_path)
    
    start_time = time.perf_counter()
    data = np.memmap(bin_path, dtype=dtype, mode='r', shape=shape)
    # Force loading into memory (simulate actual usage)
    _ = data.sum()  # Access data to ensure it's loaded
    end_time = time.perf_counter()
    
    return end_time - start_time, data, file_size


def convert_npz_to_binary(npz_path: str, output_dir: str, verbose: bool = True) -> dict:
    """
    Convert a single .npz file to binary files.
    Returns dict with timing and size information.
    """
    results = {
        'npz_file': npz_path,
        'npz_size_mb': 0,
        'npz_load_time_sec': 0,
        'binary_files': [],
        'binary_total_size_mb': 0,
        'binary_load_time_sec': 0,
        'arrays': {}
    }
    
    # Measure NPZ load time
    if verbose:
        print(f"\n{'='*60}")
        print(f"Processing: {os.path.basename(npz_path)}")
        print(f"{'='*60}")
    
    npz_load_time, loaded_data, npz_size = measure_npz_load_time(npz_path)
    results['npz_size_mb'] = npz_size / (1024 * 1024)
    results['npz_load_time_sec'] = npz_load_time
    
    if verbose:
        print(f"\nNPZ file size: {results['npz_size_mb']:.2f} MB")
        print(f"NPZ load time: {npz_load_time*1000:.2f} ms")
        print(f"\nArrays in NPZ file:")
    
    # Get base name for output files
    base_name = Path(npz_path).stem
    
    total_binary_size = 0
    total_binary_load_time = 0
    
    for key, arr in loaded_data.items():
        if verbose:
            print(f"  - {key}: shape={arr.shape}, dtype={arr.dtype}")
        
        # Save as binary
        bin_filename = f"{base_name}_{key}.bin"
        bin_path = os.path.join(output_dir, bin_filename)
        
        # Write binary file
        start_write = time.perf_counter()
        arr.tofile(bin_path)
        write_time = time.perf_counter() - start_write
        
        bin_size = os.path.getsize(bin_path)
        total_binary_size += bin_size
        
        # Measure binary load time
        bin_load_time, _, _ = measure_binary_load_time(bin_path, arr.shape, arr.dtype)
        total_binary_load_time += bin_load_time
        
        results['arrays'][key] = {
            'shape': arr.shape,
            'dtype': str(arr.dtype),
            'binary_file': bin_path,
            'binary_size_mb': bin_size / (1024 * 1024),
            'binary_load_time_sec': bin_load_time
        }
        results['binary_files'].append(bin_path)
        
        if verbose:
            print(f"    -> Saved: {bin_filename} ({bin_size/(1024*1024):.2f} MB)")
            print(f"       Binary load time: {bin_load_time*1000:.2f} ms")
    
    results['binary_total_size_mb'] = total_binary_size / (1024 * 1024)
    results['binary_load_time_sec'] = total_binary_load_time
    
    return results


def process_directory(input_dir: str, output_dir: str, pattern: str = "*.npz") -> dict:
    """
    Process all .npz files in a directory.
    """
    import glob
    
    os.makedirs(output_dir, exist_ok=True)
    
    npz_files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    
    if not npz_files:
        print(f"No .npz files found in {input_dir}")
        return {}
    
    print(f"\nFound {len(npz_files)} .npz files to convert")
    print(f"Output directory: {output_dir}")
    
    all_results = {
        'files': [],
        'total_npz_size_mb': 0,
        'total_npz_load_time_sec': 0,
        'total_binary_size_mb': 0,
        'total_binary_load_time_sec': 0
    }
    
    for npz_file in npz_files:
        result = convert_npz_to_binary(npz_file, output_dir)
        all_results['files'].append(result)
        all_results['total_npz_size_mb'] += result['npz_size_mb']
        all_results['total_npz_load_time_sec'] += result['npz_load_time_sec']
        all_results['total_binary_size_mb'] += result['binary_total_size_mb']
        all_results['total_binary_load_time_sec'] += result['binary_load_time_sec']
    
    return all_results


def print_summary(results: dict):
    """Print summary statistics."""
    print("\n" + "="*70)
    print("SUMMARY: NPZ vs Binary File Loading Performance")
    print("="*70)
    
    n_files = len(results['files'])
    
    print(f"\nTotal files processed: {n_files}")
    print(f"\nNPZ Format:")
    print(f"  Total size:      {results['total_npz_size_mb']:.2f} MB")
    print(f"  Total load time: {results['total_npz_load_time_sec']*1000:.2f} ms")
    print(f"  Avg per file:    {results['total_npz_load_time_sec']/n_files*1000:.2f} ms")
    
    print(f"\nBinary Format:")
    print(f"  Total size:      {results['total_binary_size_mb']:.2f} MB")
    print(f"  Total load time: {results['total_binary_load_time_sec']*1000:.2f} ms")
    print(f"  Avg per file:    {results['total_binary_load_time_sec']/n_files*1000:.2f} ms")
    
    # Calculate speedup and compression
    size_ratio = results['total_binary_size_mb'] / results['total_npz_size_mb']
    time_speedup = results['total_npz_load_time_sec'] / results['total_binary_load_time_sec']
    
    print(f"\nComparison:")
    print(f"  Size ratio (binary/npz): {size_ratio:.2f}x")
    print(f"  Load time speedup:       {time_speedup:.2f}x faster with binary")
    print(f"  Time saved per load:     {(results['total_npz_load_time_sec'] - results['total_binary_load_time_sec'])*1000:.2f} ms")
    
    print("\n" + "="*70)
    print("CONCLUSION:")
    print("="*70)
    print(f"\nBinary files are {time_speedup:.1f}x faster to load than .npz files.")
    print(f"This is because .npz uses ZIP compression internally, requiring")
    print(f"decompression on every load, while binary files can be memory-mapped")
    print(f"directly without any decompression overhead.")
    print(f"\nFor inference benchmarks, this means CAFE (using binary files) has")
    print(f"a data loading advantage over original DLRM (using .npz files).")
    print(f"This difference should be accounted for in fair comparisons.")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Convert .npz files to binary and measure load times'
    )
    parser.add_argument(
        '--input-dir', '-i',
        type=str,
        default=os.path.expanduser('~/expr/dlrm_minrui/input'),
        help='Directory containing .npz files'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=None,
        help='Output directory for binary files (default: input_dir/binary)'
    )
    parser.add_argument(
        '--pattern', '-p',
        type=str,
        default='*.npz',
        help='Glob pattern for .npz files (default: *.npz)'
    )
    parser.add_argument(
        '--single-file', '-f',
        type=str,
        default=None,
        help='Process a single .npz file instead of directory'
    )
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.join(args.input_dir, 'binary')
    
    print("="*70)
    print("NPZ to Binary Converter with Timing Analysis")
    print("="*70)
    
    if args.single_file:
        os.makedirs(args.output_dir, exist_ok=True)
        result = convert_npz_to_binary(args.single_file, args.output_dir)
        
        # Create a summary-compatible structure
        results = {
            'files': [result],
            'total_npz_size_mb': result['npz_size_mb'],
            'total_npz_load_time_sec': result['npz_load_time_sec'],
            'total_binary_size_mb': result['binary_total_size_mb'],
            'total_binary_load_time_sec': result['binary_load_time_sec']
        }
    else:
        results = process_directory(args.input_dir, args.output_dir, args.pattern)
    
    if results and results['files']:
        print_summary(results)
    
    return results


if __name__ == '__main__':
    main()