#!/usr/bin/env python3
import torch

compressed = torch.load('./models/dlrm_kaggle_quick_compressed.pt', map_location='cpu')
compressed_tables = compressed['compressed_tables']

# Check first table's metadata
first_table = compressed_tables[0]
print("First table structure:")
print(f"  Keys: {first_table.keys()}")
print()

if 'metadata' in first_table:
    metadata = first_table['metadata']
    print("Metadata keys:")
    for key, value in metadata.items():
        if isinstance(value, (int, float, str, bool)):
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {type(value).__name__}")

