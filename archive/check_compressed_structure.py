#!/usr/bin/env python3
import torch

print("Checking compressed model structure...")
compressed = torch.load('./models/dlrm_kaggle_quick_compressed.pt', map_location='cpu')

print("\n1. compressed_tables:")
compressed_tables = compressed['compressed_tables']
print(f"   Type: {type(compressed_tables)}")
if isinstance(compressed_tables, dict):
    print(f"   Keys: {list(compressed_tables.keys())}")
    # Show first table
    first_key = list(compressed_tables.keys())[0]
    print(f"\n   Example table (key={first_key}):")
    print(f"   Type: {type(compressed_tables[first_key])}")
    if isinstance(compressed_tables[first_key], dict):
        print(f"   Sub-keys: {list(compressed_tables[first_key].keys())}")

print("\n2. compression_info:")
compression_info = compressed['compression_info']
print(f"   Type: {type(compression_info)}")
if isinstance(compression_info, dict):
    for key, value in compression_info.items():
        print(f"   {key}: {value}")

