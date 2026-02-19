#!/usr/bin/env python3
import sys, torch, time
from dlrm_s_pytorch import QSVEmbeddingCompressor

if len(sys.argv) != 3:
    print("Usage: python3 decompress_codec.py <compressed.pt> <output.pt>")
    sys.exit(1)

compressed_path, output_path = sys.argv[1], sys.argv[2]

print("="*80)
print(f"Decompressing: {compressed_path}")
print("="*80)

# Load compressed checkpoint
checkpoint = torch.load(compressed_path, map_location='cpu')
compressed_tables = checkpoint['compressed_tables']
info = checkpoint.get('compression_info', {})

print(f"Found {len(compressed_tables)} compressed embedding tables")
print(f"Codec: {info.get('codec', 'unknown')}")
print(f"Quality: {info.get('quality', 'unknown')}")

# Get the original model path (replace model_compressed_qXX.pt with model.pt)
original_model_path = compressed_path.replace('model_compressed_q', 'model_temp_').replace('.pt', '.pt')
original_model_path = original_model_path.replace('model_temp_', 'model').replace('_q18', '').replace('_q23', '').replace('_q28', '')
# Simpler: just get the directory and use model.pt
import os
model_dir = os.path.dirname(compressed_path)
original_model_path = os.path.join(model_dir, 'model.pt')

print(f"\nLoading MLP weights from original model: {original_model_path}")

# Load original model to get MLP weights
try:
    original_checkpoint = torch.load(original_model_path, map_location='cpu')
    original_state_dict = original_checkpoint.get('state_dict', original_checkpoint)
    print(f"Loaded original model successfully")
except Exception as e:
    print(f"ERROR: Could not load original model: {e}")
    sys.exit(1)

# Create compressor
compressor = QSVEmbeddingCompressor(
    info.get('codec', 'libx265'),
    info.get('quality', 23),
    info.get('quantization', 'asymmetric'),
    bits=8
)

# Decompress embedding tables
print(f"\nDecompressing embedding tables...")
decompressed_state_dict = {}

for idx in sorted(compressed_tables.keys()):
    weights = compressor.decompress_table(
        compressed_tables[idx]['data'],
        compressed_tables[idx]['metadata']
    )
    key = f'emb_l.{idx}.weight'
    decompressed_state_dict[key] = weights
    print(f"  Table {idx}: {weights.shape}")

# Copy MLP weights from original model
print(f"\nCopying MLP weights from original model...")
mlp_keys = 0
for key, value in original_state_dict.items():
    if 'bot_l' in key or 'top_l' in key:
        decompressed_state_dict[key] = value
        mlp_keys += 1

print(f"  Copied {mlp_keys} MLP weight tensors")

# Save complete model
print(f"\nSaving complete decompressed model...")
output_checkpoint = {
    'state_dict': decompressed_state_dict,
    'compression_info': info
}

torch.save(output_checkpoint, output_path)

total_keys = len(decompressed_state_dict)
print(f"\n✓ Complete!")
print(f"  Total keys in state_dict: {total_keys}")
print(f"  - Embedding tables: {len(compressed_tables)}")
print(f"  - MLP weights: {mlp_keys}")
print(f"  Output: {output_path}")
print("="*80)
