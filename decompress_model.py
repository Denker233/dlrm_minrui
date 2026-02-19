#!/usr/bin/env python3
"""
Decompress compressed DLRM model back to standard PyTorch format
"""

import torch
from dlrm_s_pytorch import QSVEmbeddingCompressor

def decompress_model(compressed_path, original_path, output_path):
    """
    Load compressed model and decompress embeddings
    """
    print("Loading compressed model...")
    compressed = torch.load(compressed_path)
    
    print("Loading original model structure...")
    original = torch.load(original_path)
    
    # Get compression info
    comp_info = compressed['compression_info']
    
    # Create decompressor
    compressor = QSVEmbeddingCompressor(
        codec=comp_info['codec'],
        quality=comp_info['quality'],
        quantization=comp_info['quantization']
    )
    
    # Get compressed tables dict
    compressed_tables = compressed['compressed_tables']
    num_tables = len(compressed_tables)
    
    print(f"\nDecompressing {num_tables} embedding tables...")
    print("-" * 80)
    
    # Decompress each table
    decompressed_embs = []
    for i in range(num_tables):
        table_data = compressed_tables[i]
        compressed_data = table_data['data']  # Changed from 'compressed' to 'data'
        metadata = table_data['metadata']
        
        shape = metadata['shape']
        size_mb = (shape[0] * shape[1] * 4) / (1024*1024)
        print(f"Table {i:2d}: {shape[0]:>10,} × {shape[1]:>2} = {size_mb:>7.2f} MB", end=' ... ')
        
        weights = compressor.decompress_table(compressed_data, metadata)
        decompressed_embs.append(weights)
        print("✓")
    
    print("-" * 80)
    
    # Put decompressed embeddings back into model state dict
    state_dict = original['state_dict'].copy()
    
    emb_idx = 0
    for key in list(state_dict.keys()):
        if 'emb_l' in key and 'weight' in key:
            print(f"Replacing {key} with decompressed table {emb_idx}")
            state_dict[key] = decompressed_embs[emb_idx]
            emb_idx += 1
    
    print(f"\nReplaced {emb_idx} embedding tables")
    
    # Create decompressed model
    decompressed_model = {
        'state_dict': state_dict,
        'iter': original.get('iter', 0),
        'epoch': original.get('epoch', 0),
        'nepochs': original.get('nepochs', 1),
        'nbatches': original.get('nbatches', 0),
        'nbatches_test': original.get('nbatches_test', 0),
        'train_loss': original.get('train_loss', 0),
        'total_loss': original.get('total_loss', 0),
        'test_acc': original.get('test_acc', 0),
    }
    
    # Save decompressed model
    print(f"\nSaving decompressed model to: {output_path}")
    torch.save(decompressed_model, output_path)
    
    # Print compression summary
    print("\n" + "=" * 80)
    print("COMPRESSION SUMMARY")
    print("=" * 80)
    print(f"Codec:        {comp_info['codec']}")
    print(f"Quality:      {comp_info['quality']}")
    print(f"Quantization: {comp_info['quantization']}")
    if 'stats' in comp_info:
        stats = comp_info['stats']
        print(f"\nOriginal size:   {stats.get('original_size_mb', 'N/A')} MB")
        print(f"Compressed size: {stats.get('compressed_size_mb', 'N/A')} MB")
        print(f"Compression:     {stats.get('compression_ratio', 'N/A')}x")
    print("=" * 80)
    print("Done!")
    
    return output_path

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 4:
        print("Usage: python decompress_model.py <compressed_model> <original_model> <output_path>")
        sys.exit(1)
    
    compressed_path = sys.argv[1]
    original_path = sys.argv[2]
    output_path = sys.argv[3]
    
    decompress_model(compressed_path, original_path, output_path)
