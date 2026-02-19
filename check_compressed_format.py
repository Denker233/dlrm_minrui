#!/usr/bin/env python3
import torch

print("Checking compressed model format...")
compressed = torch.load('./models/dlrm_kaggle_quick_compressed.pt', map_location='cpu')

print(f"\nTop-level keys in compressed model:")
for key in list(compressed.keys())[:20]:
    print(f"  {key}")

print(f"\nTotal keys: {len(compressed.keys())}")

# Check for compressed embeddings
compressed_keys = [k for k in compressed.keys() if 'compressed' in k]
print(f"\nKeys with 'compressed': {len(compressed_keys)}")
for key in compressed_keys[:5]:
    print(f"  {key}")

# Check for metadata
metadata_keys = [k for k in compressed.keys() if 'metadata' in k]
print(f"\nKeys with 'metadata': {len(metadata_keys)}")
for key in metadata_keys[:5]:
    print(f"  {key}")

# Check if it has state_dict
if 'state_dict' in compressed:
    print(f"\nHas 'state_dict' key")
    state_dict = compressed['state_dict']
    emb_keys = [k for k in state_dict.keys() if 'emb_l' in k]
    print(f"Embedding keys in state_dict: {len(emb_keys)}")
    for key in emb_keys[:5]:
        print(f"  {key}")

