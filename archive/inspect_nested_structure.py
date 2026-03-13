#!/usr/bin/env python3
import torch
import os
import sys

model_path = './models/dlrm_kaggle_quick_compressed.pt'
file_size = os.path.getsize(model_path)

print("="*80)
print("DEEP INSPECTION OF NESTED STRUCTURE")
print("="*80)
print()

print(f"File size: {file_size / 1024 / 1024:.2f} MB")
print()

# Load and inspect
compressed = torch.load(model_path, map_location='cpu')

print(f"Top-level keys: {list(compressed.keys())}")
print(f"Total top-level keys: {len(compressed)}")
print()

def get_size(obj, depth=0):
    """Recursively calculate size of nested structures"""
    indent = "  " * depth
    
    if isinstance(obj, bytes):
        size = len(obj)
        print(f"{indent}bytes: {size / 1024 / 1024:.2f} MB")
        return size
    elif isinstance(obj, (list, tuple)):
        print(f"{indent}{type(obj).__name__} with {len(obj)} items:")
        total = 0
        for i, item in enumerate(obj):
            print(f"{indent}  [{i}]:")
            total += get_size(item, depth + 2)
        return total
    elif isinstance(obj, dict):
        print(f"{indent}dict with {len(obj)} keys:")
        total = 0
        for key, value in obj.items():
            print(f"{indent}  '{key}':")
            total += get_size(value, depth + 2)
        return total
    elif isinstance(obj, torch.Tensor):
        size = obj.element_size() * obj.numel()
        print(f"{indent}Tensor {obj.shape}: {size / 1024 / 1024:.2f} MB")
        return size
    else:
        size = sys.getsizeof(obj)
        print(f"{indent}{type(obj).__name__}: {size} bytes")
        return size

print("="*80)
print("NESTED STRUCTURE ANALYSIS")
print("="*80)
print()

total_data_size = 0
for key in compressed.keys():
    print(f"Key: '{key}'")
    total_data_size += get_size(compressed[key], depth=1)
    print()

print("="*80)
print("SUMMARY")
print("="*80)
print()
print(f"Total data size (calculated): {total_data_size / 1024 / 1024:.2f} MB")
print(f"Actual file size:             {file_size / 1024 / 1024:.2f} MB")
print(f"Pickle overhead:              {(file_size - total_data_size) / 1024 / 1024:.2f} MB")
print(f"Overhead percentage:          {100 * (file_size - total_data_size) / file_size:.1f}%")
print()

# Check if it's using ZIP format
import zipfile
try:
    with zipfile.ZipFile(model_path, 'r') as z:
        print("File format: ZIP archive (new PyTorch format)")
        print(f"Files in archive: {z.namelist()}")
except zipfile.BadZipFile:
    print("File format: Legacy pickle")

