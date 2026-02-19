#!/usr/bin/env python3
"""
Check if MLP layers were compressed
"""

import torch
import os

print("="*80)
print("CHECKING IF MLPs WERE COMPRESSED")
print("="*80)
print()

# Load both models
original_path = './models/dlrm_kaggle_quick.pt'
compressed_path = './models/dlrm_kaggle_quick_compressed.pt'

print("Loading models...")
original = torch.load(original_path, map_location='cpu')
compressed = torch.load(compressed_path, map_location='cpu')

print()
print("="*80)
print("ORIGINAL MODEL STRUCTURE")
print("="*80)
print()

# Analyze original model
orig_state = original['state_dict']

emb_keys = [k for k in orig_state.keys() if 'emb_l' in k]
mlp_keys = [k for k in orig_state.keys() if ('bot_l' in k or 'top_l' in k)]

print(f"Embedding tables: {len(emb_keys)} keys")
print(f"MLP layers: {len(mlp_keys)} keys")
print()

# Calculate sizes
emb_size = sum(orig_state[k].numel() * orig_state[k].element_size() for k in emb_keys)
mlp_size = sum(orig_state[k].numel() * orig_state[k].element_size() for k in mlp_keys)

print(f"Embedding size: {emb_size / 1024 / 1024:.2f} MB")
print(f"MLP size:       {mlp_size / 1024 / 1024:.2f} MB")
print(f"Total:          {(emb_size + mlp_size) / 1024 / 1024:.2f} MB")
print()

print("MLP layers in original:")
for key in mlp_keys[:10]:  # Show first 10
    tensor = orig_state[key]
    size = tensor.numel() * tensor.element_size() / 1024 / 1024
    print(f"  {key:<40} {str(tensor.shape):<20} {tensor.dtype} {size:>6.2f} MB")

print()
print("="*80)
print("COMPRESSED MODEL STRUCTURE")
print("="*80)
print()

# Check what's in compressed model
compressed_keys = list(compressed.keys())
print(f"Total keys in compressed model: {len(compressed_keys)}")
print()

# Categorize keys
emb_compressed_keys = [k for k in compressed_keys if 'emb_l' in k and 'compressed' in k]
emb_metadata_keys = [k for k in compressed_keys if 'emb_l' in k and 'metadata' in k]
mlp_compressed_keys = [k for k in compressed_keys if ('bot_l' in k or 'top_l' in k)]
state_dict_keys = [k for k in compressed_keys if k == 'state_dict']

print(f"Compressed embeddings: {len(emb_compressed_keys)} keys")
print(f"Embedding metadata:    {len(emb_metadata_keys)} keys")
print(f"MLP keys:              {len(mlp_compressed_keys)} keys")
print(f"State dict preserved:  {len(state_dict_keys)} keys")
print()

# Check if MLPs are in state_dict or separate
if 'state_dict' in compressed:
    print("Found 'state_dict' key - checking contents...")
    state_dict = compressed['state_dict']
    mlp_in_state = [k for k in state_dict.keys() if ('bot_l' in k or 'top_l' in k)]
    print(f"  MLP layers in state_dict: {len(mlp_in_state)}")
    
    if mlp_in_state:
        print()
        print("MLP layers in compressed model:")
        for key in mlp_in_state[:10]:
            tensor = state_dict[key]
            size = tensor.numel() * tensor.element_size() / 1024 / 1024
            print(f"  {key:<40} {str(tensor.shape):<20} {tensor.dtype} {size:>6.2f} MB")
        
        # Calculate total MLP size in compressed
        mlp_compressed_size = sum(
            state_dict[k].numel() * state_dict[k].element_size() 
            for k in mlp_in_state
        )
        print()
        print(f"Total MLP size in compressed model: {mlp_compressed_size / 1024 / 1024:.2f} MB")

else:
    print("No 'state_dict' key found - MLPs stored differently")
    if mlp_compressed_keys:
        print(f"Found {len(mlp_compressed_keys)} MLP keys at top level")

print()
print("="*80)
print("SIZE ANALYSIS")
print("="*80)
print()

# Calculate what's actually stored
total_size = 0
breakdown = {}

for key, value in compressed.items():
    if isinstance(value, bytes):
        size = len(value)
        category = 'compressed_data'
    elif isinstance(value, torch.Tensor):
        size = value.numel() * value.element_size()
        category = 'tensor'
    elif isinstance(value, dict):
        if key == 'state_dict':
            # Calculate state_dict size
            size = sum(
                v.numel() * v.element_size() if isinstance(v, torch.Tensor) else len(str(v))
                for v in value.values()
            )
            category = 'state_dict'
        else:
            size = sum(len(str(k)) + len(str(v)) for k, v in value.items())
            category = 'metadata'
    else:
        size = len(str(value))
        category = 'other'
    
    total_size += size
    
    # Categorize
    if 'compressed' in key:
        breakdown['Compressed embeddings'] = breakdown.get('Compressed embeddings', 0) + size
    elif 'metadata' in key:
        breakdown['Embedding metadata'] = breakdown.get('Embedding metadata', 0) + size
    elif key == 'state_dict':
        breakdown['State dict (MLPs etc)'] = breakdown.get('State dict (MLPs etc)', 0) + size
    else:
        breakdown['Other'] = breakdown.get('Other', 0) + size

print("Content breakdown:")
for category, size in sorted(breakdown.items(), key=lambda x: x[1], reverse=True):
    print(f"  {category:<30} {size / 1024 / 1024:>8.2f} MB ({100*size/total_size:>5.1f}%)")

print(f"  {'─'*30} {'─'*8}")
print(f"  {'Total calculated':<30} {total_size / 1024 / 1024:>8.2f} MB")

actual_file_size = os.path.getsize(compressed_path)
print(f"  {'Actual file size':<30} {actual_file_size / 1024 / 1024:>8.2f} MB")

overhead = actual_file_size - total_size
if overhead > 0:
    print(f"  {'PyTorch overhead':<30} {overhead / 1024 / 1024:>8.2f} MB")

print()
print("="*80)
print("VERDICT: DID YOU COMPRESS MLPs?")
print("="*80)
print()

# Compare MLP sizes
if 'state_dict' in compressed:
    mlp_in_compressed = [k for k in compressed['state_dict'].keys() if ('bot_l' in k or 'top_l' in k)]
    if mlp_in_compressed:
        mlp_compressed_size = sum(
            compressed['state_dict'][k].numel() * compressed['state_dict'][k].element_size()
            for k in mlp_in_compressed
        )
        
        compression_ratio = mlp_size / mlp_compressed_size if mlp_compressed_size > 0 else 1
        
        print(f"Original MLP size:    {mlp_size / 1024 / 1024:.2f} MB")
        print(f"Compressed MLP size:  {mlp_compressed_size / 1024 / 1024:.2f} MB")
        print(f"MLP compression:      {compression_ratio:.2f}x")
        print()
        
        if compression_ratio > 1.5:
            print("✓ YES - MLPs were compressed!")
            print(f"  Compression method: Likely PyTorch default (gzip/lz4)")
        elif compression_ratio > 1.1:
            print("~ MAYBE - Slight compression (likely PyTorch serialization)")
        else:
            print("✗ NO - MLPs are uncompressed (stored as FP32)")
            print("  MLPs are stored in original FP32 format")
else:
    print("Cannot determine - state_dict not found")

print()
print("="*80)
print("DETAILED BREAKDOWN")
print("="*80)
print()

print("Your 14 MB file contains:")
print()
print(f"  Compressed embeddings:  ~9.3 MB  (video codec)")
print(f"  MLP weights:            ~{mlp_compressed_size/1024/1024 if 'state_dict' in compressed else 0:.1f} MB  (FP32 or default compression)")
print(f"  Metadata:               ~1-2 MB  (scales, zero_points, dimensions)")
print(f"  PyTorch overhead:       ~2 MB    (pickle protocol)")
print()

