#!/usr/bin/env python3
"""
Decompress models created by compress_with_details.py
"""

import argparse
import torch
import time
import os
from dlrm_s_pytorch import QSVEmbeddingCompressor

def decompress_model(compressed_path, output_path):
    """
    Decompress a model compressed by compress_with_details.py
    Combines decompressed embeddings with original MLP weights
    """
    
    print("="*80)
    print("DECOMPRESSING MODEL")
    print("="*80)
    print(f"Input:  {compressed_path}")
    print(f"Output: {output_path}")
    print()
    
    start_time = time.time()
    
    # Load compressed checkpoint
    print("[1/3] Loading compressed model...")
    compressed_checkpoint = torch.load(compressed_path, map_location='cpu')
    
    if 'compressed_tables' not in compressed_checkpoint:
        print("ERROR: Not a compressed model (missing 'compressed_tables')")
        return False
    
    compressed_tables = compressed_checkpoint['compressed_tables']
    compression_info = compressed_checkpoint.get('compression_info', {})
    
    print(f"  Tables: {len(compressed_tables)}")
    print(f"  Codec: {compression_info.get('codec', 'unknown')}")
    print(f"  Quality: {compression_info.get('quality', 'unknown')}")
    
    # Load original model for MLP weights and metadata
    model_dir = os.path.dirname(compressed_path)
    original_model_path = os.path.join(model_dir, 'model.pt')
    
    print(f"\n[2/3] Loading original model for MLP weights...")
    print(f"  Path: {original_model_path}")
    
    try:
        original_checkpoint = torch.load(original_model_path, map_location='cpu')
        print(f"  ✓ Loaded successfully")
    except Exception as e:
        print(f"ERROR: Could not load original model: {e}")
        return False
    
    # Create compressor for decompression
    codec = compression_info.get('codec', 'libx265')
    quality = compression_info.get('quality', 23)
    quantization = compression_info.get('quantization', 'asymmetric')
    
    compressor = QSVEmbeddingCompressor(codec, quality, quantization, bits=8)
    
    # Start with complete original checkpoint (preserves all metadata)
    output_checkpoint = dict(original_checkpoint)
    
    # Get original state_dict
    original_state_dict = original_checkpoint.get('state_dict', original_checkpoint)
    
    # Create new state_dict
    new_state_dict = {}
    
    # Decompress embedding tables
    print(f"\n[3/3] Decompressing embedding tables...")
    for table_idx in sorted(compressed_tables.keys()):
        compressed_data = compressed_tables[table_idx]['data']
        metadata = compressed_tables[table_idx]['metadata']
        
        decompressed_weights = compressor.decompress_table(compressed_data, metadata)
        key = f'emb_l.{table_idx}.weight'
        new_state_dict[key] = decompressed_weights
        
        print(f"  Table {table_idx}: {decompressed_weights.shape}")
    
    # Copy MLP weights from original model
    print(f"\nCopying MLP weights from original model...")
    mlp_count = 0
    for key, value in original_state_dict.items():
        if 'bot_l' in key or 'top_l' in key:
            new_state_dict[key] = value
            mlp_count += 1
    
    print(f"  Copied {mlp_count} MLP tensors")
    
    # Update state_dict in checkpoint (preserves iter, epoch, etc.)
    output_checkpoint['state_dict'] = new_state_dict
    
    # Save complete checkpoint
    print(f"\nSaving decompressed model...")
    torch.save(output_checkpoint, output_path)
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"DECOMPRESSION COMPLETE")
    print(f"{'='*80}")
    print(f"Total time:       {total_time:.2f}s")
    print(f"State dict keys:  {len(new_state_dict)}")
    print(f"  - Embeddings:   {len(compressed_tables)}")
    print(f"  - MLP weights:  {mlp_count}")
    print(f"  - Metadata:     Preserved from original (iter, epoch, etc.)")
    print(f"Output:           {output_path}")
    print(f"{'='*80}")
    
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Decompress models created by compress_with_details.py'
    )
    parser.add_argument('--compressed', required=True, 
                       help='Path to compressed model')
    parser.add_argument('--output', required=True, 
                       help='Path to save decompressed model')
    
    args = parser.parse_args()
    
    success = decompress_model(args.compressed, args.output)
    
    if not success:
        exit(1)
