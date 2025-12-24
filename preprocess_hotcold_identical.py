#!/usr/bin/env python3
"""
Hot/Cold Preprocessor with IDENTICAL Remapping Logic to Runtime

This script uses the EXACT SAME remapping logic as the runtime version in dlrm_hot.py
to ensure 100% consistency between preprocessed and runtime-remapped results.

Key differences from preprocess_simple_hotcold.py:
- Uses identical hot_map/cold_map lookup logic
- Applies the same filtering for -1 values
- Same offset calculation method

Usage:
    python preprocess_hotcold_identical.py \
        --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz \
        --profile-file=./profiles/kaggle_profile_P80_analyzed.pkl \
        --output-dir=./preprocessed_hotcold_identical \
        --batch-size=16384
"""

import argparse
import os
import sys
import time
import pickle
import numpy as np
import torch


def load_criteo_data(processed_file, memory_map=True):
    """Load the processed Criteo dataset."""
    print(f"Loading processed data from {processed_file}...")
    
    mmap_mode = 'r' if memory_map else None
    data = np.load(processed_file, mmap_mode=mmap_mode)
    
    X_int = data['X_int']
    X_cat = data['X_cat']
    y = data['y']
    
    print(f"  Dense features shape: {X_int.shape}")
    print(f"  Sparse features shape: {X_cat.shape}")
    print(f"  Labels shape: {y.shape}")
    
    return X_int, X_cat, y


def load_hotcold_profile(profile_file):
    """Load the analyzed hot/cold profile."""
    print(f"Loading hot/cold profile from {profile_file}...")
    
    with open(profile_file, 'rb') as f:
        profiles = pickle.load(f)
    
    print(f"  Hot/cold tables: {list(profiles.keys())}")
    for table_idx in sorted(profiles.keys()):
        p = profiles[table_idx]
        print(f"    C{table_idx+1}: hot={p['hot_size']:,}, cold={p['cold_size']:,}")
    
    return profiles


def build_remap_tables(profiles):
    """
    Build remapping tables IDENTICAL to runtime version.
    
    Extracts the EXACT SAME hot_map and cold_map tensors used by
    the runtime remapping code in dlrm_hot.py
    
    Returns:
        Dict mapping table_idx -> {
            'hot_map': NumPy array [original_size], -1 if not in hot
            'cold_map': NumPy array [original_size], -1 if not in cold
            'hot_size': number of hot embeddings
            'cold_size': number of cold embeddings
        }
    """
    remap_tables = {}
    
    for table_idx, profile in profiles.items():
        hot_size = profile['hot_size']
        cold_size = profile['cold_size']
        
        # Get the SAME remap tensors used by runtime version
        hot_map_tensor = profile['hot_map_tensor']
        cold_map_tensor = profile['cold_map_tensor']
        
        # Convert to numpy (handle torch tensor, numpy array, or list)
        if hasattr(hot_map_tensor, 'numpy'):
            hot_map = hot_map_tensor.cpu().numpy().astype(np.int64)
        elif isinstance(hot_map_tensor, np.ndarray):
            hot_map = hot_map_tensor.astype(np.int64)
        else:
            hot_map = np.array(hot_map_tensor, dtype=np.int64)
        
        if hasattr(cold_map_tensor, 'numpy'):
            cold_map = cold_map_tensor.cpu().numpy().astype(np.int64)
        elif isinstance(cold_map_tensor, np.ndarray):
            cold_map = cold_map_tensor.astype(np.int64)
        else:
            cold_map = np.array(cold_map_tensor, dtype=np.int64)
        
        remap_tables[table_idx] = {
            'hot_map': hot_map,      # NumPy array: [original_size], -1 if not in hot
            'cold_map': cold_map,    # NumPy array: [original_size], -1 if not in cold
            'hot_size': hot_size,
            'cold_size': cold_size,
        }
        
        print(f"  C{table_idx+1}: Loaded remap tables (hot={hot_size:,}, cold={cold_size:,})")
    
    return remap_tables


def preprocess_and_save(X_int, X_cat, y, remap_tables, output_dir, batch_size, split_name, split_ratio=0.9):
    """
    Preprocess sparse features using IDENTICAL remapping logic to runtime version.
    
    The remapping logic here is 100% identical to the runtime version in dlrm_hot.py:
    
    1. hot_indices_mapped = hot_map[original_indices]
    2. cold_indices_mapped = cold_map[original_indices]
    3. hot_mask = (hot_indices_mapped >= 0)
    4. cold_mask = (cold_indices_mapped >= 0)
    5. Filter using masks
    6. Build offsets from cumsum of masks
    """
    num_samples = X_cat.shape[0]
    num_tables = X_cat.shape[1]
    
    # Determine train/test split
    if split_name == 'train':
        start_idx = 0
        end_idx = int(num_samples * split_ratio)
    else:  # test
        start_idx = int(num_samples * split_ratio)
        end_idx = num_samples
    
    split_samples = end_idx - start_idx
    num_batches = (split_samples + batch_size - 1) // batch_size
    
    print(f"\nProcessing {split_name} data...")
    print(f"  Samples: {split_samples:,} (indices {start_idx:,} to {end_idx:,})")
    print(f"  Batches: {num_batches:,}")
    print(f"  Using IDENTICAL remapping logic to runtime version")
    
    # Create output directory
    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)
    
    # Which tables are hot/cold split?
    hotcold_mask = [table_idx in remap_tables for table_idx in range(num_tables)]
    
    # Process each batch
    for batch_idx in range(num_batches):
        if batch_idx % 100 == 0:
            print(f"  Batching {split_name}: {batch_idx}/{num_batches} ({100*batch_idx/num_batches:.1f}%)")
        
        batch_start = start_idx + batch_idx * batch_size
        batch_end = min(batch_start + batch_size, end_idx)
        actual_batch_size = batch_end - batch_start
        
        # Extract data for this batch
        batch_X_int = np.array(X_int[batch_start:batch_end], dtype=np.float32)
        batch_X_cat = np.array(X_cat[batch_start:batch_end], dtype=np.int64)
        batch_y = np.array(y[batch_start:batch_end], dtype=np.float32)
        
        # Process each table - IDENTICAL LOGIC TO RUNTIME REMAPPING
        lS_o_hot = []
        lS_i_hot = []
        lS_o_cold = []
        lS_i_cold = []
        
        for table_idx in range(num_tables):
            # Get original indices for this batch [batch_size]
            original_indices = batch_X_cat[:, table_idx]
            
            if table_idx in remap_tables:
                # ============================================================
                # HOT/COLD SPLIT TABLE
                # IDENTICAL REMAPPING LOGIC TO dlrm_hot.py runtime version
                # ============================================================
                remap = remap_tables[table_idx]
                hot_map = remap['hot_map']    # [original_size], -1 if not in hot
                cold_map = remap['cold_map']  # [original_size], -1 if not in cold
                
                # Step 1: Apply remapping - EXACTLY like runtime
                hot_indices_mapped = hot_map[original_indices]  # -1 if not in hot
                cold_indices_mapped = cold_map[original_indices]  # -1 if not in cold
                
                # Step 2: Filter out -1 values - EXACTLY like runtime
                hot_mask = (hot_indices_mapped >= 0)
                cold_mask = (cold_indices_mapped >= 0)
                
                # Step 3: Extract valid indices - EXACTLY like runtime
                hot_indices = hot_indices_mapped[hot_mask]
                cold_indices = cold_indices_mapped[cold_mask]
                
                # Step 4: Build offsets for EmbeddingBag
                # Each sample contributes 0 or 1 index to hot/cold
                hot_offsets = np.zeros(actual_batch_size + 1, dtype=np.int64)
                hot_offsets[1:] = np.cumsum(hot_mask.astype(np.int64))
                
                cold_offsets = np.zeros(actual_batch_size + 1, dtype=np.int64)
                cold_offsets[1:] = np.cumsum(cold_mask.astype(np.int64))

                if table_idx == 2 and batch_idx == 0:  # First batch, table C3
                    print(f"\n[DEBUG PREPROCESSING C3]:")
                    print(f"  Original indices[:10]: {original_indices[:10].tolist()}")
                    print(f"  Cold map min/max: {cold_map.min()}, {cold_map.max()}")
                    print(f"  Cold indices mapped[:10]: {cold_indices_mapped[:10].tolist()}")
                    print(f"  Cold mask sum: {cold_mask.sum()}")
                    print(f"  Cold indices (after filter)[:10]: {cold_indices[:10].tolist()}")
                
            else:
                # ============================================================
                # STANDARD TABLE - NO REMAPPING
                # ============================================================
                # Put in hot list, leave cold empty
                hot_indices = original_indices
                hot_offsets = np.arange(actual_batch_size + 1, dtype=np.int64)
                
                cold_indices = np.array([], dtype=np.int64)
                cold_offsets = np.zeros(actual_batch_size + 1, dtype=np.int64)
            
            # Convert to tensors
            lS_o_hot.append(torch.from_numpy(hot_offsets))
            lS_i_hot.append(torch.from_numpy(hot_indices.copy()))
            lS_o_cold.append(torch.from_numpy(cold_offsets))
            lS_i_cold.append(torch.from_numpy(cold_indices.copy()))
        
        # Save batch as TUPLE (7 elements)
        batch_data = (
            torch.from_numpy(batch_X_int.copy()),  # [0] Dense features
            torch.from_numpy(batch_y.copy()),      # [1] Targets
            lS_o_hot,                               # [2] Hot offsets
            lS_i_hot,                               # [3] Hot indices
            lS_o_cold,                              # [4] Cold offsets
            lS_i_cold,                              # [5] Cold indices
            hotcold_mask,                           # [6] Hot/cold mask
        )
        
        batch_file = os.path.join(split_dir, f'batch_{batch_idx:06d}.pt')
        torch.save(batch_data, batch_file)
    
    print(f"  Batching {split_name}: {num_batches}/{num_batches} (100.0%)")
    return num_batches, split_samples


def main():
    parser = argparse.ArgumentParser(
        description='Hot/Cold Preprocessor with IDENTICAL Remapping Logic to Runtime'
    )
    parser.add_argument('--processed-data-file', type=str, required=True,
                        help='Path to processed .npz file')
    parser.add_argument('--profile-file', type=str, required=True,
                        help='Path to analyzed hot/cold profile .pkl file')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for preprocessed data')
    parser.add_argument('--batch-size', type=int, default=16384,
                        help='Batch size (default: 16384)')
    parser.add_argument('--split-ratio', type=float, default=0.9,
                        help='Train/test split ratio (default: 0.9)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Hot/Cold Preprocessor - IDENTICAL to Runtime Remapping")
    print("=" * 60)
    print(f"Input: {args.processed_data_file}")
    print(f"Profile: {args.profile_file}")
    print(f"Output: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Remapping logic: 100% IDENTICAL to dlrm_hot.py runtime")
    
    start_time = time.time()
    
    # Load data
    X_int, X_cat, y = load_criteo_data(args.processed_data_file)
    num_tables = X_cat.shape[1]
    
    # Get table sizes
    table_sizes = [int(X_cat[:, i].max()) + 1 for i in range(num_tables)]
    print(f"\nTable sizes: {table_sizes}")
    
    # Load profile
    profiles = load_hotcold_profile(args.profile_file)
    
    # Build remapping tables
    print("\nBuilding remapping tables (IDENTICAL to runtime)...")
    remap_tables = build_remap_tables(profiles)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process train and test
    train_batches, train_samples = preprocess_and_save(
        X_int, X_cat, y, remap_tables, args.output_dir, args.batch_size, 'train', args.split_ratio
    )
    
    test_batches, test_samples = preprocess_and_save(
        X_int, X_cat, y, remap_tables, args.output_dir, args.batch_size, 'test', args.split_ratio
    )
    
    # Build table configs for model creation
    table_configs = []
    hotcold_mask = []
    hot_sizes = {}
    cold_sizes = {}
    
    for table_idx in range(num_tables):
        if table_idx in remap_tables:
            table_configs.append({
                'type': 'hotcold',
                'hot_size': remap_tables[table_idx]['hot_size'],
                'cold_size': remap_tables[table_idx]['cold_size'],
                'original_size': table_sizes[table_idx],
            })
            hotcold_mask.append(True)
            hot_sizes[table_idx] = remap_tables[table_idx]['hot_size']
            cold_sizes[table_idx] = remap_tables[table_idx]['cold_size']
        else:
            table_configs.append({
                'type': 'standard',
                'size': table_sizes[table_idx],
            })
            hotcold_mask.append(False)
    
    # Save metadata
    metadata = {
        'batch_size': args.batch_size,
        'train_batches': train_batches,
        'test_batches': test_batches,
        'train_samples': train_samples,
        'test_samples': test_samples,
        'num_tables': num_tables,
        'table_sizes': table_sizes,
        'table_configs': table_configs,
        'hotcold_mask': hotcold_mask,
        'hotcold_tables': list(remap_tables.keys()),
        'hot_cold_profiles': profiles,
        'hot_sizes': hot_sizes,
        'cold_sizes': cold_sizes,
        'profile_file': args.profile_file,
        'per_sample_format': False,
        'remapping_logic': 'identical_to_runtime',  # Flag to indicate this version
    }
    
    metadata_file = os.path.join(args.output_dir, 'metadata.pkl')
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE!")
    print("=" * 60)
    print(f"Time: {elapsed:.1f}s")
    print(f"Train: {train_batches} batches ({train_samples:,} samples)")
    print(f"Test: {test_batches} batches ({test_samples:,} samples)")
    print(f"Output: {args.output_dir}")
    print(f"\nRemapping logic: IDENTICAL to runtime version")
    print(f"Batch format: (X, T, lS_o_hot, lS_i_hot, lS_o_cold, lS_i_cold, hotcold_mask)")
    print(f"\nUsage:")
    print(f"  python dlrm_hot.py \\")
    print(f"      --use-hotcold-preprocessed \\")
    print(f"      --hotcold-preprocessed-dir {args.output_dir} \\")
    print(f"      --mini-batch-size {args.batch_size} \\")
    print(f"      --inference-only --num-batches 1000")


if __name__ == '__main__':
    main()
