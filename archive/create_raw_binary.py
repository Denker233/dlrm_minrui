#!/usr/bin/env python3
import torch
import struct
import os

print("="*80)
print("CREATING RAW BINARY FORMAT")
print("="*80)
print()

# Load compressed model
model_path = './models/dlrm_kaggle_quick_compressed.pt'
compressed = torch.load(model_path, map_location='cpu')

print(f"Loaded model with keys: {list(compressed.keys())}")
print()

# Extract structure
if 'compressed_tables' in compressed and 'compression_info' in compressed:
    tables = compressed['compressed_tables']
    info = compressed['compression_info']
    
    print(f"Number of tables: {len(tables)}")
    print(f"Type of first table: {type(tables[0])}")
    print(f"Size of first table: {len(tables[0]) / 1024 / 1024:.2f} MB")
    print()
    
    # Create raw binary file
    output_path = './models/dlrm_kaggle_quick_compressed.bin'
    
    with open(output_path, 'wb') as f:
        # Magic number
        f.write(b'DLRM')
        
        # Version
        f.write(struct.pack('I', 1))
        
        # Number of tables
        f.write(struct.pack('I', len(tables)))
        
        # For each table
        total_video_size = 0
        for i in range(len(tables)):
            # Get data
            video_data = tables[i]
            metadata = info[i]
            
            total_video_size += len(video_data)
            
            # Write table ID
            f.write(struct.pack('I', i))
            
            # Write metadata
            f.write(struct.pack('I', metadata['shape'][0]))
            f.write(struct.pack('I', metadata['shape'][1]))
            f.write(struct.pack('f', metadata['scale']))
            f.write(struct.pack('f', metadata['zero_point']))
            f.write(struct.pack('f', metadata['quantization_min']))
            f.write(struct.pack('f', metadata['quantization_max']))
            
            # Write frame dimensions
            f.write(struct.pack('I', metadata['frame_shape'][0]))
            f.write(struct.pack('I', metadata['frame_shape'][1]))
            
            # Write video data length
            f.write(struct.pack('I', len(video_data)))
            
            # Write video data
            f.write(video_data)
    
    raw_size = os.path.getsize(output_path)
    pytorch_size = os.path.getsize(model_path)
    
    print(f"Total compressed video data: {total_video_size / 1024 / 1024:.2f} MB")
    print()
    print(f"PyTorch .pt file:  {pytorch_size / 1024 / 1024:.2f} MB")
    print(f"Raw binary file:   {raw_size / 1024 / 1024:.2f} MB")
    print(f"Overhead savings:  {(pytorch_size - raw_size) / 1024 / 1024:.2f} MB ({100*(pytorch_size-raw_size)/pytorch_size:.1f}%)")
    print()
    print(f"Saved to: {output_path}")
    print()
    
    # Also compare compression ratios
    original_emb_size = 2060.70  # MB
    print("Compression ratios:")
    print(f"  PyTorch format: {original_emb_size / (pytorch_size/1024/1024):.1f}x")
    print(f"  Binary format:  {original_emb_size / (raw_size/1024/1024):.1f}x")
    
else:
    print("Unexpected structure!")
    print(f"Keys: {compressed.keys()}")
    for key in compressed.keys():
        print(f"  {key}: {type(compressed[key])}")

