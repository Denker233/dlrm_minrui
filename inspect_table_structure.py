#!/usr/bin/env python3
import torch

compressed = torch.load('./models/dlrm_kaggle_quick_compressed.pt')
compressed_tables = compressed['compressed_tables']

print("Inspecting table 0 structure:")
table_0 = compressed_tables[0]
print(f"Type: {type(table_0)}")

if isinstance(table_0, dict):
    print(f"Keys: {list(table_0.keys())}")
    for key in table_0.keys():
        print(f"  {key}: {type(table_0[key])}")
elif isinstance(table_0, tuple):
    print(f"Tuple length: {len(table_0)}")
    for i, item in enumerate(table_0):
        print(f"  Item {i}: {type(item)}")
        if isinstance(item, dict):
            print(f"    Keys: {list(item.keys())}")
elif isinstance(table_0, bytes):
    print(f"Bytes length: {len(table_0)}")
else:
    print(f"Other type: {table_0}")
