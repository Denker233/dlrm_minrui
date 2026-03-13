#!/bin/bash

# Quick fix: Create the decompress_infer_autoarch.py script

cat > decompress_infer_autoarch.py << 'PYTHON_SCRIPT'
#!/usr/bin/env python3
"""
Decompress and infer - with AUTOMATIC architecture detection
"""

import torch
import argparse
import sys
import numpy as np
import sklearn.metrics
import time
import subprocess
import tempfile
import os

# Import DLRM modules
try:
    import dlrm_s_pytorch as dlrm_module
    import dlrm_data_pytorch as dp
except ImportError as e:
    print(f"Error: Could not import DLRM modules.")
    sys.exit(1)

def decompress_table(compressed_data, metadata):
    """Decompress embedding table using FFmpeg"""
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        f.write(compressed_data)
        video_path = f.name
    
    try:
        cmd = ['ffmpeg', '-loglevel', 'quiet', '-i', video_path,
               '-f', 'rawvideo', '-pix_fmt', 'gray', 'pipe:1']
        result = subprocess.run(cmd, capture_output=True, check=True)
        raw_pixels = np.frombuffer(result.stdout, dtype=np.uint8)
        
        num_embeddings, embedding_dim = metadata['shape']
        total_elements = num_embeddings * embedding_dim
        quantized = raw_pixels[:total_elements].reshape(num_embeddings, embedding_dim)
        
        # Dequantize
        quant_params = metadata['quant_params']
        if 'scales' in quant_params:
            # Per-row quantization
            scales = quant_params['scales']
            zero_points = quant_params['zero_points']
            dequantized = np.zeros((num_embeddings, embedding_dim), dtype=np.float32)
            for i in range(num_embeddings):
                dequantized[i] = (quantized[i].astype(np.float32) - zero_points[i]) * scales[i]
        else:
            # Global quantization
            scale = quant_params['scale']
            zero_point = quant_params['zero_point']
            dequantized = (quantized.astype(np.float32) - zero_point) * scale
        
        return torch.from_numpy(dequantized)
    finally:
        if os.path.exists(video_path):
            os.unlink(video_path)

def detect_architecture(state_dict):
    """
    Automatically detect DLRM architecture from state_dict
    Returns: (m_spa, ln_bot, ln_top, num_emb_tables)
    """
    print("\nDetecting model architecture from checkpoint...")
    
    # Get embedding dimensions
    emb_keys = [k for k in state_dict.keys() if k.startswith('emb_l.') and k.endswith('.weight')]
    num_emb_tables = len(emb_keys)
    m_spa = state_dict[emb_keys[0]].shape[1] if emb_keys else 16
    
    print(f"  Embedding dim: {m_spa}")
    print(f"  Number of embedding tables: {num_emb_tables}")
    
    # Get bottom MLP dimensions
    bot_keys = sorted([k for k in state_dict.keys() if k.startswith('bot_l.') and '.weight' in k])
    ln_bot = []
    if bot_keys:
        # First layer input dimension
        first_layer = state_dict[bot_keys[0]]
        ln_bot.append(first_layer.shape[1])
        
        # Hidden and output dimensions
        for key in bot_keys:
            layer = state_dict[key]
            ln_bot.append(layer.shape[0])
    
    print(f"  Bottom MLP: {'-'.join(map(str, ln_bot))}")
    
    # Get top MLP dimensions
    top_keys = sorted([k for k in state_dict.keys() if k.startswith('top_l.') and '.weight' in k])
    ln_top = []
    if top_keys:
        # First layer input dimension
        first_layer = state_dict[top_keys[0]]
        ln_top.append(first_layer.shape[1])
        
        # Hidden and output dimensions
        for key in top_keys:
            layer = state_dict[key]
            ln_top.append(layer.shape[0])
    
    print(f"  Top MLP: {'-'.join(map(str, ln_top))}")
    
    return m_spa, np.array(ln_bot), np.array(ln_top), num_emb_tables

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compressed-model', type=str, required=True)
    parser.add_argument('--original-model', type=str, default=None)
    parser.add_argument('--data-set', type=str, default='kaggle')
    parser.add_argument('--raw-data-file', type=str, default='./input/train.txt')
    parser.add_argument('--processed-data-file', type=str, 
                       default='./input/kaggleAdDisplayChallenge_processed.npz')
    parser.add_argument('--test-mini-batch-size', type=int, default=2048)
    parser.add_argument('--test-num-workers', type=int, default=8)
    args = parser.parse_args()
    
    device = torch.device('cpu')
    
    print("="*80)
    print("COMPRESSED DLRM: DECOMPRESS + INFERENCE")
    print("="*80)
    print(f"\nCompressed: {args.compressed_model}")
    if args.original_model:
        print(f"Original:   {args.original_model}")
    
    # ========================================================================
    # STEP 1: LOAD MODELS
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 1: LOADING MODELS")
    print("="*80)
    
    compressed_checkpoint = torch.load(args.compressed_model, map_location='cpu')
    print("✓ Compressed model loaded")
    
    if 'compressed_tables' not in compressed_checkpoint:
        print("\n✗ Error: Not a compressed model!")
        sys.exit(1)
    
    original_checkpoint = None
    orig_acc = None
    orig_auc = None
    
    if args.original_model:
        original_checkpoint = torch.load(args.original_model, map_location='cpu')
        print("✓ Original model loaded")
        orig_acc = original_checkpoint.get('test_acc')
        orig_auc = original_checkpoint.get('test_auc')
        if orig_acc or orig_auc:
            print(f"\nOriginal Metrics:")
            if orig_acc:
                print(f"  Accuracy: {orig_acc * 100:.3f}%")
            if orig_auc:
                print(f"  AUC:      {orig_auc:.4f}")
    
    # ========================================================================
    # STEP 2: DECOMPRESS
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 2: DECOMPRESSING EMBEDDING TABLES")
    print("="*80)
    
    compressed_tables = compressed_checkpoint['compressed_tables']
    print(f"\nFound {len(compressed_tables)} compressed embedding tables")
    
    if 'state_dict' in compressed_checkpoint:
        state_dict = compressed_checkpoint['state_dict'].copy()
    elif original_checkpoint and 'state_dict' in original_checkpoint:
        state_dict = original_checkpoint['state_dict'].copy()
    else:
        print("✗ No state_dict found!")
        sys.exit(1)
    
    total_compressed = 0
    total_decompressed = 0
    total_time = 0
    
    print("\nDecompressing...")
    for idx in sorted(compressed_tables.keys()):
        table_data = compressed_tables[idx]
        compressed_data = table_data['data']
        metadata = table_data['metadata']
        
        print(f"  Table {idx}: {metadata['shape']}", end='', flush=True)
        
        start = time.time()
        weights = decompress_table(compressed_data, metadata)
        elapsed = time.time() - start
        total_time += elapsed
        
        compressed_size = len(compressed_data)
        decompressed_size = weights.numel() * 4
        ratio = decompressed_size / compressed_size
        
        total_compressed += compressed_size
        total_decompressed += decompressed_size
        
        print(f" → {ratio:.1f}x in {elapsed:.2f}s")
        
        state_dict[f'emb_l.{idx}.weight'] = weights
    
    overall_ratio = total_decompressed / total_compressed
    print(f"\n{'='*80}")
    print(f"Summary:")
    print(f"  Compressed:   {total_compressed / 1024**2:.2f} MB")
    print(f"  Decompressed: {total_decompressed / 1024**2:.2f} MB")
    print(f"  Overall:      {overall_ratio:.1f}x")
    print(f"  Time:         {total_time:.2f}s")
    print(f"{'='*80}")
    
    # ========================================================================
    # STEP 3: DETECT ARCHITECTURE
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 3: DETECTING ARCHITECTURE")
    print("="*80)
    
    m_spa, ln_bot, ln_top, num_emb_tables = detect_architecture(state_dict)
    
    # ========================================================================
    # STEP 4: LOAD DATA
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 4: LOADING TEST DATA")
    print("="*80)
    
    class DataArgs:
        pass
    
    data_args = DataArgs()
    data_args.arch_sparse_feature_size = m_spa
    data_args.data_generation = 'dataset'
    data_args.data_set = args.data_set
    data_args.raw_data_file = args.raw_data_file
    data_args.processed_data_file = args.processed_data_file
    data_args.mini_batch_size = 128
    data_args.test_mini_batch_size = args.test_mini_batch_size
    data_args.test_num_workers = args.test_num_workers
    data_args.num_workers = 0
    data_args.nepochs = 1
    data_args.numpy_rand_seed = 123
    data_args.data_randomize = "total"
    data_args.data_trace_enable_padding = False
    data_args.max_ind_range = -1
    data_args.data_sub_sample_rate = 0.0
    data_args.num_indices_per_lookup = 10
    data_args.num_indices_per_lookup_fixed = False
    data_args.memory_map = False
    data_args.mlperf_bin_loader = False
    data_args.mlperf_bin_shuffle = False
    data_args.mlperf_logging = False
    data_args.dataset_multiprocessing = False
    
    _, _, test_data, test_ld = dp.make_criteo_data_and_loaders(data_args)
    ln_emb = test_data.counts
    
    print(f"\n✓ Data loaded: {len(test_ld)} batches")
    
    # ========================================================================
    # STEP 5: CREATE MODEL
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 5: CREATING MODEL")
    print("="*80)
    
    model = dlrm_module.DLRM_Net(
        m_spa, ln_emb, ln_bot, ln_top,
        arch_interaction_op="dot",
        arch_interaction_itself=False,
        sigmoid_bot=-1,
        sigmoid_top=ln_top.size - 2,
        sync_dense_params=True,
        loss_threshold=0.0,
        ndevices=-1,
        qr_flag=False,
        md_flag=False,
        weighted_pooling=None,
        loss_function="bce",
    )
    
    print("\nLoading weights...")
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print("✓ Model ready")
    
    # ========================================================================
    # STEP 6: INFERENCE
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 6: RUNNING INFERENCE")
    print("="*80)
    
    all_scores = []
    all_targets = []
    
    print(f"\nProcessing {len(test_ld)} batches...")
    start = time.time()
    
    with torch.no_grad():
        for i, batch in enumerate(test_ld):
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start
                rate = (i + 1) / elapsed
                eta = (len(test_ld) - i - 1) / rate if rate > 0 else 0
                print(f"  {i+1}/{len(test_ld)} ({rate:.1f} batch/s, ETA: {eta:.0f}s)", end='\r', flush=True)
            
            if len(batch) == 6:
                X, lS_o, lS_i, T, W, CBPP = batch
            else:
                X, lS_o, lS_i, T = batch[:4]
            
            X = X.to(device)
            if isinstance(lS_o, list):
                lS_o = [s.to(device) for s in lS_o]
            else:
                lS_o = lS_o.to(device)
            if isinstance(lS_i, list):
                lS_i = [s.to(device) for s in lS_i]
            else:
                lS_i = lS_i.to(device)
            T = T.to(device)
            
            Z = model(X, lS_o, lS_i)
            all_scores.append(Z.detach().cpu().numpy())
            all_targets.append(T.detach().cpu().numpy())
    
    total = time.time() - start
    print(f"\n✓ Completed in {total:.1f}s ({len(test_ld)/total:.1f} batch/s)")
    
    all_scores = np.concatenate(all_scores)
    all_targets = np.concatenate(all_targets)
    
    metrics = {
        "accuracy": sklearn.metrics.accuracy_score(all_targets, np.round(all_scores)),
        "roc_auc": sklearn.metrics.roc_auc_score(all_targets, all_scores),
        "recall": sklearn.metrics.recall_score(all_targets, np.round(all_scores)),
        "precision": sklearn.metrics.precision_score(all_targets, np.round(all_scores)),
        "f1": sklearn.metrics.f1_score(all_targets, np.round(all_scores)),
        "average_precision": sklearn.metrics.average_precision_score(all_targets, all_scores),
    }
    
    # ========================================================================
    # RESULTS
    # ========================================================================
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    print(f"\nDecompressed Model:")
    print(f"  Accuracy:   {metrics['accuracy'] * 100:.3f}%")
    print(f"  AUC:        {metrics['roc_auc']:.4f}")
    print(f"  Recall:     {metrics['recall']:.4f}")
    print(f"  Precision:  {metrics['precision']:.4f}")
    print(f"  F1:         {metrics['f1']:.4f}")
    print(f"  Avg Prec:   {metrics['average_precision']:.4f}")
    
    if orig_acc or orig_auc:
        print("\n" + "="*80)
        print("COMPRESSION IMPACT")
        print("="*80)
        
        if orig_auc:
            loss = orig_auc - metrics['roc_auc']
            loss_pct = (loss / orig_auc) * 100
            print(f"\nAUC:")
            print(f"  Original:     {orig_auc:.4f}")
            print(f"  Decompressed: {metrics['roc_auc']:.4f}")
            print(f"  Loss:         {loss:.4f} ({loss_pct:.2f}%)")
        
        if orig_acc:
            loss = (orig_acc - metrics['accuracy']) * 100
            print(f"\nAccuracy:")
            print(f"  Original:     {orig_acc * 100:.3f}%")
            print(f"  Decompressed: {metrics['accuracy'] * 100:.3f}%")
            print(f"  Loss:         {loss:.3f} pp")
        
        print("\n" + "="*80)
        print("VS CAFE BASELINE")
        print("="*80)
        
        cafe_loss = 0.11441
        
        print(f"\nCAFE (525x):")
        print(f"  Loss: 0.1144 (14.8%)")
        
        if orig_auc:
            auc_loss = orig_auc - metrics['roc_auc']
            auc_loss_pct = (auc_loss / orig_auc) * 100
            print(f"\nYours ({overall_ratio:.0f}x):")
            print(f"  Loss: {auc_loss:.4f} ({auc_loss_pct:.2f}%)")
            
            if auc_loss < cafe_loss:
                improvement = cafe_loss - auc_loss
                print(f"\n✓ BETTER THAN CAFE!")
                print(f"  Improvement: {improvement:.4f} AUC points")
            elif auc_loss < cafe_loss * 1.1:
                print(f"\n≈ COMPETITIVE WITH CAFE")
            else:
                print(f"\n✗ CAFE is better")
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)

if __name__ == '__main__':
    main()
PYTHON_SCRIPT

chmod +x decompress_infer_autoarch.py
echo "✓ Created decompress_infer_autoarch.py"