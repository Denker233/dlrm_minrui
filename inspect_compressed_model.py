#!/usr/bin/env python3
import torch

compressed = torch.load('./models/dlrm_kaggle_quick_compressed.pt')

print("Keys in compressed model:")
for key in compressed.keys():
    print(f"  {key}: {type(compressed[key])}")
    if isinstance(compressed[key], dict):
        print(f"    Sub-keys: {list(compressed[key].keys())[:5]}...")
    elif isinstance(compressed[key], list):
        print(f"    Length: {len(compressed[key])}")
        if len(compressed[key]) > 0:
            print(f"    First item type: {type(compressed[key][0])}")
