#!/usr/bin/env python3
import torch
import os

model_path = './models/dlrm_kaggle_quick_compressed.pt'
file_size = os.path.getsize(model_path)

print(f"File size: {file_size / 1024 / 1024:.2f} MB")
print()

# Load and inspect
compressed = torch.load(model_path, map_location='cpu')

print(f"Keys in file: {list(compressed.keys())[:5]}...")
print(f"Total keys: {len(compressed)}")
print()

# Calculate actual compressed data size
video_data_size = 0
metadata_size = 0

for key, value in compressed.items():
    if isinstance(value, bytes):
        size = len(value)
        video_data_size += size
        print(f"{key}: {size / 1024 / 1024:.2f} MB")
    elif isinstance(value, dict):
        # Rough metadata size
        metadata_size += 100  # Estimate

print()
print(f"Total video data: {video_data_size / 1024 / 1024:.2f} MB")
print(f"Total metadata: {metadata_size / 1024:.2f} KB")
print(f"File overhead: {(file_size - video_data_size) / 1024 / 1024:.2f} MB")
print()

