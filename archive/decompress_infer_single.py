#!/usr/bin/env python3
"""
Single Script: Decompress Compressed DLRM Model and Run Inference
This combines decompression + inference into one streamlined process
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
from pathlib import Path

# Import DLRM modules
try:
    import dlrm_s_pytorch as dlrm_module
    import dlrm_data_pytorch as dp
except ImportError as e:
    print(f"Error: Could not import DLRM modules.")
    print(f"Make sure dlrm_s_pytorch.py is in the same directory.")
    sys.exit(1)

def decompress_table(compressed_data, metadata):
    """
    Decompress a single embedding table using FFmpeg
    Uses the proven decompression logic from benchmark script
    """
    # Write compressed data to temporary file
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        f.write(compressed_data)
        video_path = f.name
    
    try:
        # Decompress using FFmpeg
        cmd = [
            'ffmpeg', '-loglevel', 'quiet',
            '-i', video_path,
            '-f', 'rawvideo',
            '-pix_fmt', 'gray',
            'pipe:1'
        ]
        
        result = subprocess.run(cmd, capture_output=True, check=True)
        raw_pixels = np.frombuffer(result.stdout, dtype=np.uint8)
        
        # Reshape to original dimensions
        num_embeddings, embedding_dim = metadata['shape']
        total_elements = num_embeddings * embedding_dim
        
        # Handle tiling if used
        if metadata.get('tiling', {}).get('tiled', False):
            tiling = metadata['tiling']
            grid_size = tiling['grid_size']
            tile_size = tiling['tile_size']
            image_size = grid_size * tile_size
            
            # Extract the tiled image
            image = raw_pixels[:image_size*image_size].reshape(image_size, image_size)
            
            # Untile to get back embeddings
            from dlrm_s_pytorch import untile_embeddings
            quantized = untile_embeddings(
                image, embedding_dim, num_embeddings,
                grid_size, tiling['tiles_per_emb'], tile_size
            ).reshape(num_embeddings, embedding_dim)
        else:
            quantized = raw_pixels[:total_elements].reshape(num_embeddings, embedding_dim)
        
        # Dequantize - handle both asymmetric and per-row quantization
        quant_params = metadata['quant_params']
        quantization_type = metadata.get('quantization', 'asymmetric')
        
        if quantization_type == 'per_row' or 'scales' in quant_params:
            # Per-row quantization
            scales = quant_params['scales']
            zero_points = quant_params['zero_points']
            dequantized = np.zeros((num_embeddings, embedding_dim), dtype=np.float32)
            
            for i in range(num_embeddings):
                scale = scales[i]
                zero_point = zero_points[i]
                dequantized[i] = (quantized[i].astype(np.float32) - zero_point) * scale
        else:
            # Asymmetric (global) quantization
            scale = quant_params['scale']
            zero_point = quant_params['zero_point']
            dequantized = (quantized.astype(np.float32) - zero_point) * scale
        
        return torch.from_numpy(dequantized)
        
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr if e.stderr else str(e)}")
        raise
    finally:
        if os.path.exists(video_path):
            os.unlink(video_path)

def main():
    parser = argparse.ArgumentParser(description='Decompress and test compressed DLRM model')
    
    # Model paths
    parser.add_argument('--compressed-model', type=str, required=True,
                       help='Path to compressed model file')
    parser.add_argument('--original-model', type=str, default=None,
                       help='Path to original uncompressed model (for comparison)')
    
    # Architecture (must match training)
    parser.add_argument('--arch-sparse-feature-size', type=int, default=16)
    parser.add_argument('--arch-mlp-bot', type=str, default="13-512-256-64-16")
    parser.add_argument('--arch-mlp-top', type=str, default="512-256-1")
    parser.add_argument('--arch-interaction-op', type=str, default="dot")
    parser.add_argument('--arch-interaction-itself', action='store_true', default=False)
    
    # Data
    parser.add_argument('--data-generation', type=str, default='dataset')
    parser.add_argument('--data-set', type=str, default='kaggle')
    parser.add_argument('--raw-data-file', type=str, default='./input/train.txt')
    parser.add_argument('--processed-data-file', type=str, 
                       default='./input/kaggleAdDisplayChallenge_processed.npz')
    parser.add_argument('--test-mini-batch-size', type=int, default=2048)
    parser.add_argument('--test-num-workers', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=0)
    
    # Embedding sizes (auto-detected if not provided)
    parser.add_argument('--ln-emb', nargs='+', type=int, default=None)
    
    args = parser.parse_args()
    
    device = torch.device('cpu')
    use_gpu = False
    
    print("="*80)
    print("COMPRESSED DLRM: DECOMPRESS + INFERENCE")
    print("="*80)
    print(f"\nCompressed model: {args.compressed_model}")
    if args.original_model:
        print(f"Original model:   {args.original_model}")
    print("")
    
    # ========================================================================
    # STEP 1: LOAD MODELS
    # ========================================================================
    print("="*80)
    print("STEP 1: LOADING MODELS")
    print("="*80)
    
    print("\nLoading compressed model...")
    compressed_checkpoint = torch.load(args.compressed_model, map_location='cpu')
    print("✓ Compressed model loaded")
    
    # Check format
    if 'compressed_tables' not in compressed_checkpoint:
        print("\n✗ Error: This doesn't appear to be a compressed model!")
        print(f"Keys found: {list(compressed_checkpoint.keys())}")
        sys.exit(1)
    
    # Load original model if provided
    original_checkpoint = None
    orig_acc = None
    orig_auc = None
    
    if args.original_model:
        print("Loading original model for comparison...")
        original_checkpoint = torch.load(args.original_model, map_location='cpu')
        print("✓ Original model loaded")
        
        orig_acc = original_checkpoint.get('test_acc', None)
        orig_auc = original_checkpoint.get('test_auc', None)
        
        if orig_acc or orig_auc:
            print(f"\nOriginal Model Metrics (from checkpoint):")
            if orig_acc:
                print(f"  Accuracy: {orig_acc * 100:.3f}%")
            if orig_auc:
                print(f"  AUC:      {orig_auc:.4f}")
    
    # ========================================================================
    # STEP 2: DECOMPRESS EMBEDDING TABLES
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 2: DECOMPRESSING EMBEDDING TABLES")
    print("="*80)
    
    compressed_tables = compressed_checkpoint['compressed_tables']
    print(f"\nFound {len(compressed_tables)} compressed embedding tables")
    
    # Start with state_dict from checkpoint
    if 'state_dict' in compressed_checkpoint:
        state_dict = compressed_checkpoint['state_dict'].copy()
    elif original_checkpoint and 'state_dict' in original_checkpoint:
        state_dict = original_checkpoint['state_dict'].copy()
        print("Using state_dict from original checkpoint as base")
    else:
        print("✗ Error: No state_dict found!")
        sys.exit(1)
    
    total_compressed_size = 0
    total_decompressed_size = 0
    total_decomp_time = 0
    
    print("\nDecompressing tables...")
    for table_idx in sorted(compressed_tables.keys()):
        table_data = compressed_tables[table_idx]
        compressed_data = table_data['data']
        metadata = table_data['metadata']
        
        print(f"\n  Table {table_idx}:")
        print(f"    Shape: {metadata['shape']}")
        print(f"    Codec: {metadata['codec']}, Quality: {metadata['quality']}")
        print(f"    Compressed: {len(compressed_data):,} bytes")
        
        # Decompress
        start_time = time.time()
        decompressed_weights = decompress_table(compressed_data, metadata)
        decomp_time = time.time() - start_time
        total_decomp_time += decomp_time
        
        print(f"    Decompressed: {decompressed_weights.shape} in {decomp_time:.3f}s")
        
        # Calculate sizes
        compressed_size = len(compressed_data)
        decompressed_size = decompressed_weights.numel() * 4  # float32
        compression_ratio = decompressed_size / compressed_size
        
        total_compressed_size += compressed_size
        total_decompressed_size += decompressed_size
        
        print(f"    Compression ratio: {compression_ratio:.1f}x")
        
        # Update state_dict
        emb_key = f'emb_l.{table_idx}.weight'
        state_dict[emb_key] = decompressed_weights
        
        # Compare with original if available
        if original_checkpoint and 'state_dict' in original_checkpoint:
            original_weights = original_checkpoint['state_dict'].get(emb_key)
            if original_weights is not None:
                mse = torch.mean((decompressed_weights - original_weights) ** 2).item()
                max_diff = torch.max(torch.abs(decompressed_weights - original_weights)).item()
                print(f"    MSE vs original: {mse:.6f}")
                print(f"    Max diff: {max_diff:.6f}")
    
    # Overall statistics
    overall_ratio = total_decompressed_size / total_compressed_size
    print(f"\n{'='*80}")
    print(f"Decompression Summary:")
    print(f"  Total compressed:   {total_compressed_size / 1024**2:.2f} MB")
    print(f"  Total decompressed: {total_decompressed_size / 1024**2:.2f} MB")
    print(f"  Overall ratio:      {overall_ratio:.1f}x")
    print(f"  Total time:         {total_decomp_time:.2f}s")
    print(f"{'='*80}")
    
    # ========================================================================
    # STEP 3: LOAD TEST DATA
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 3: LOADING TEST DATA")
    print("="*80)
    
    class DataArgs:
        pass
    
    data_args = DataArgs()
    for key, value in vars(args).items():
        setattr(data_args, key, value)
    
    # Set required args
    data_args.mini_batch_size = 128
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
    data_args.mlperf_logging = False  # ADD THIS
    data_args.dataset_multiprocessing = False
    
    try:
        _, _, test_data, test_ld = dp.make_criteo_data_and_loaders(data_args)
        
        if args.ln_emb is None:
            args.ln_emb = test_data.counts
            print(f"\n✓ Auto-detected {len(args.ln_emb)} embedding tables")
        
    except Exception as e:
        print(f"\n✗ Error loading data: {e}")
        sys.exit(1)
    
    print(f"✓ Test data loaded: {len(test_ld)} batches")
    
    # ========================================================================
    # STEP 4: CREATE MODEL WITH DECOMPRESSED WEIGHTS
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 4: CREATING MODEL")
    print("="*80)
    
    m_spa = args.arch_sparse_feature_size
    ln_emb = np.array(args.ln_emb)
    ln_bot = np.fromstring(args.arch_mlp_bot, dtype=int, sep="-")
    ln_top = np.fromstring(args.arch_mlp_top, dtype=int, sep="-")
    
    print(f"\nCreating DLRM model...")
    model = dlrm_module.DLRM_Net(
        m_spa, ln_emb, ln_bot, ln_top,
        arch_interaction_op=args.arch_interaction_op,
        arch_interaction_itself=args.arch_interaction_itself,
        sigmoid_bot=-1,
        sigmoid_top=ln_top.size - 2,
        sync_dense_params=True,
        loss_threshold=0.0,
        ndevices=-1,
        qr_flag=False,
        qr_operation="mult",
        qr_collisions=4,
        qr_threshold=200,
        md_flag=False,
        md_threshold=200,
        weighted_pooling=None,
        loss_function="bce",
    )
    
    print("Loading decompressed weights...")
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print("✓ Model ready for inference")
    
    # ========================================================================
    # STEP 5: RUN INFERENCE
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 5: RUNNING INFERENCE")
    print("="*80)
    
    all_scores = []
    all_targets = []
    
    print(f"\nProcessing {len(test_ld)} batches...")
    start_time = time.time()
    
    with torch.no_grad():
        for i, testBatch in enumerate(test_ld):
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (len(test_ld) - i - 1) / rate if rate > 0 else 0
                print(f"  {i+1}/{len(test_ld)} batches ({rate:.1f} batch/s, ETA: {eta:.0f}s)...", 
                      end='\r')
            
            # Unpack batch
            if len(testBatch) == 6:
                X_test, lS_o_test, lS_i_test, T_test, W_test, CBPP_test = testBatch
            else:
                X_test, lS_o_test, lS_i_test, T_test = testBatch[:4]
            
            # Move to device
            X_test = X_test.to(device)
            if isinstance(lS_o_test, list):
                lS_o_test = [s.to(device) for s in lS_o_test]
            else:
                lS_o_test = lS_o_test.to(device)
            
            if isinstance(lS_i_test, list):
                lS_i_test = [s.to(device) for s in lS_i_test]
            else:
                lS_i_test = lS_i_test.to(device)
            
            T_test = T_test.to(device)
            
            # Forward pass
            Z_test = model(X_test, lS_o_test, lS_i_test)
            
            # Collect results
            scores = Z_test.detach().cpu().numpy()
            targets = T_test.detach().cpu().numpy()
            
            all_scores.append(scores)
            all_targets.append(targets)
    
    total_time = time.time() - start_time
    print(f"\n✓ Completed in {total_time:.1f}s ({len(test_ld)/total_time:.1f} batch/s)")
    
    # Concatenate all batches
    all_scores = np.concatenate(all_scores, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = {
        "accuracy": sklearn.metrics.accuracy_score(
            y_true=all_targets, y_pred=np.round(all_scores)
        ),
        "roc_auc": sklearn.metrics.roc_auc_score(
            y_true=all_targets, y_score=all_scores
        ),
        "recall": sklearn.metrics.recall_score(
            y_true=all_targets, y_pred=np.round(all_scores)
        ),
        "precision": sklearn.metrics.precision_score(
            y_true=all_targets, y_pred=np.round(all_scores)
        ),
        "f1": sklearn.metrics.f1_score(
            y_true=all_targets, y_pred=np.round(all_scores)
        ),
        "average_precision": sklearn.metrics.average_precision_score(
            y_true=all_targets, y_score=all_scores
        ),
    }
    
    # ========================================================================
    # STEP 6: REPORT RESULTS
    # ========================================================================
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    print(f"\nDecompressed Model Metrics:")
    print(f"  Accuracy:          {metrics['accuracy'] * 100:.3f}%")
    print(f"  AUC:               {metrics['roc_auc']:.4f}")
    print(f"  Recall:            {metrics['recall']:.4f}")
    print(f"  Precision:         {metrics['precision']:.4f}")
    print(f"  F1:                {metrics['f1']:.4f}")
    print(f"  Average Precision: {metrics['average_precision']:.4f}")
    
    # Compare with original if available
    if orig_acc or orig_auc:
        print("\n" + "="*80)
        print("COMPRESSION IMPACT")
        print("="*80)
        
        if orig_auc:
            auc_loss = orig_auc - metrics['roc_auc']
            auc_loss_pct = (auc_loss / orig_auc) * 100
            
            print(f"\nAUC Comparison:")
            print(f"  Original (uncompressed): {orig_auc:.4f}")
            print(f"  Decompressed:            {metrics['roc_auc']:.4f}")
            print(f"  Loss:                    {auc_loss:.4f} ({auc_loss_pct:.2f}%)")
        
        if orig_acc:
            acc_loss = (orig_acc - metrics['accuracy']) * 100
            acc_loss_pct = (acc_loss / (orig_acc * 100)) * 100
            
            print(f"\nAccuracy Comparison:")
            print(f"  Original (uncompressed): {orig_acc * 100:.3f}%")
            print(f"  Decompressed:            {metrics['accuracy'] * 100:.3f}%")
            print(f"  Loss:                    {acc_loss:.3f} pp ({acc_loss_pct:.2f}%)")
        
        # Compare with CAFE baseline
        print("\n" + "="*80)
        print("COMPARISON TO CAFE BASELINE")
        print("="*80)
        
        cafe_baseline_auc = 0.77268
        cafe_compressed_auc = 0.65827
        cafe_loss = cafe_baseline_auc - cafe_compressed_auc
        cafe_loss_pct = (cafe_loss / cafe_baseline_auc) * 100
        
        print(f"\nCAFE (525x compression):")
        print(f"  Baseline AUC:    {cafe_baseline_auc:.4f}")
        print(f"  Compressed AUC:  {cafe_compressed_auc:.4f}")
        print(f"  Loss:            {cafe_loss:.4f} ({cafe_loss_pct:.2f}%)")
        
        if orig_auc:
            print(f"\nYour Method ({overall_ratio:.0f}x compression):")
            print(f"  Baseline AUC:    {orig_auc:.4f}")
            print(f"  Compressed AUC:  {metrics['roc_auc']:.4f}")
            print(f"  Loss:            {auc_loss:.4f} ({auc_loss_pct:.2f}%)")
            
            print(f"\nComparison:")
            if auc_loss < cafe_loss:
                improvement = cafe_loss - auc_loss
                improvement_pct = (improvement / cafe_loss) * 100
                print(f"  ✓ YOUR METHOD IS BETTER!")
                print(f"  Improvement: {improvement:.4f} AUC points ({improvement_pct:.1f}% less loss)")
            elif auc_loss < cafe_loss * 1.1:
                print(f"  ≈ YOUR METHOD IS COMPETITIVE!")
            else:
                worse = auc_loss - cafe_loss
                print(f"  ✗ CAFE performs better by {worse:.4f} AUC points")
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)

if __name__ == '__main__':
    main()