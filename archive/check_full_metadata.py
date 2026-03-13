#!/usr/bin/env python3
import torch
import subprocess
import tempfile

compressed = torch.load('./models/dlrm_kaggle_quick_compressed.pt', map_location='cpu')
compressed_tables = compressed['compressed_tables']

# Check first table
first_table = compressed_tables[0]
metadata = first_table['metadata']
compressed_data = first_table['data']

print("Full metadata for table 0:")
print(f"  shape: {metadata['shape']}")
print(f"  quantization: {metadata['quantization']}")
print(f"  bits: {metadata['bits']}")
print(f"  codec: {metadata['codec']}")
print(f"  quality: {metadata['quality']}")
print()

print("quant_params:")
for key, value in metadata['quant_params'].items():
    print(f"  {key}: {value}")
print()

print(f"Compressed data size: {len(compressed_data)} bytes")
print()

# Try to decode the video to get dimensions
with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
    f.write(compressed_data)
    video_path = f.name

try:
    # Use ffprobe to get video dimensions
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height',
        '-of', 'csv=s=x:p=0',
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    dims = result.stdout.strip()
    width, height = map(int, dims.split('x'))
    
    print(f"Video dimensions: {width}x{height}")
    print(f"Total pixels: {width * height}")
    print(f"Original shape was: {metadata['shape']}")
    print(f"Original elements: {metadata['shape'][0] * metadata['shape'][1]}")
    
finally:
    import os
    os.unlink(video_path)

