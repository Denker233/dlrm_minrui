#!/usr/bin/env python3
"""Test quantization accuracy"""

import torch
import numpy as np
import argparse
import sys
import dlrm_data_pytorch as dp
from dlrm_s_pytorch import DLRM_Net

def quantize_to_n_bits(weights, n_bits):
    """Quantize weights to n-bit precision"""
    w_min, w_max = weights.min(), weights.max()
    num_levels = 2**n_bits
    scale = (w_max - w_min) / (num_levels - 1) if w_max > w_min else 1.0
    zero_point = -(w_min / scale).round() if scale > 0 else 0
    
    quantized = ((weights / scale).round() + zero_point).clamp(0, num_levels - 1)
    quantized = quantized.to(torch.uint8)
    
    # Dequantize immediately for inference
    dequantized = (quantized.float() - zero_point) * scale
    
    return dequantized

def test_quantization_accuracy(args, n_bits):
    """Test model accuracy with n-bit quantization"""
    
    print(f"\n{'='*80}")
    print(f"TESTING INT{n_bits} QUANTIZATION ACCURACY")
    print(f"{'='*80}")
    
    # Add missing attributes
    args.dataset_multiprocessing = False
    
    # Load data
    print("Loading data...")
    train_data, train_ld, test_data, test_ld = dp.make_criteo_data_and_loaders(args)
    
    ln_emb = np.array(train_data.counts)
    m_spa = args.arch_sparse_feature_size
    ln_bot = np.fromstring(args.arch_mlp_bot, dtype=int, sep="-")
    ln_bot[0] = train_data.m_den
    
    # Calculate interaction size
    num_fea = ln_emb.size + 1
    m_den_out = ln_bot[ln_bot.size - 1]
    if args.arch_interaction_op == "dot":
        if args.arch_interaction_itself:
            num_int = (num_fea * (num_fea + 1)) // 2 + m_den_out
        else:
            num_int = (num_fea * (num_fea - 1)) // 2 + m_den_out
    else:
        num_int = num_fea * m_den_out
    
    arch_mlp_top_adjusted = str(num_int) + "-" + args.arch_mlp_top
    ln_top = np.fromstring(arch_mlp_top_adjusted, dtype=int, sep="-")
    
    # Create model
    print("Creating model...")
    dlrm = DLRM_Net(
        m_spa, ln_emb, ln_bot, ln_top,
        arch_interaction_op=args.arch_interaction_op,
        arch_interaction_itself=args.arch_interaction_itself,
        sigmoid_bot=-1,
        sigmoid_top=ln_top.size - 2,
        loss_function=args.loss_function,
    )
    
    # Load weights
    print(f"Loading model from: {args.load_model}")
    ld_model = torch.load(args.load_model, map_location=torch.device("cpu"))
    dlrm.load_state_dict(ld_model['state_dict'])
    
    # Quantize embeddings
    print(f"Quantizing embeddings to INT{n_bits}...")
    original_size = 0
    quantized_size = 0
    
    for i, emb in enumerate(dlrm.emb_l):
        original_weights = emb.weight.data
        original_size += original_weights.numel() * 4  # float32 = 4 bytes
        
        # Quantize and dequantize
        quantized_weights = quantize_to_n_bits(original_weights, n_bits)
        emb.weight.data = quantized_weights
        
        quantized_size += original_weights.numel() * n_bits / 8
    
    compression_ratio = original_size / quantized_size
    print(f"  Original size:  {original_size/1024/1024:.2f} MB")
    print(f"  Quantized size: {quantized_size/1024/1024:.2f} MB (theoretical)")
    print(f"  Compression:    {compression_ratio:.2f}x")
    
    # Run inference
    print("\nRunning inference...")
    dlrm.eval()
    
    test_accu = 0
    test_samp = 0
    
    with torch.no_grad():
        for i, testBatch in enumerate(test_ld):
            if i % 100 == 0:
                print(f"  Batch {i}/{len(test_ld)}", end='\r')
            
            X_test, lS_o_test, lS_i_test, T_test = testBatch[0], testBatch[1], testBatch[2], testBatch[3]
            
            # Forward pass
            Z_test = dlrm(X_test, lS_o_test, lS_i_test)
            
            # Compute accuracy
            S_test = Z_test.detach().cpu().numpy()
            T_test_np = T_test.detach().cpu().numpy()
            
            mbs_test = T_test_np.shape[0]
            A_test = np.sum((np.round(S_test, 0) == T_test_np).astype(np.uint8))
            
            test_accu += A_test
            test_samp += mbs_test
    
    accuracy = test_accu / test_samp
    
    print(f"\n{'='*80}")
    print(f"RESULTS FOR INT{n_bits}")
    print(f"{'='*80}")
    print(f"Accuracy:    {accuracy*100:.4f}%")
    print(f"Compression: {compression_ratio:.2f}x")
    print(f"Size:        {quantized_size/1024/1024:.2f} MB")
    print(f"{'='*80}\n")
    
    return {
        'bits': n_bits,
        'accuracy': accuracy,
        'compression': compression_ratio,
        'size_mb': quantized_size/1024/1024
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-model", type=str, required=True)
    parser.add_argument("--bits", type=int, choices=[2, 3, 4, 8], required=True)
    parser.add_argument("--arch-sparse-feature-size", type=int, default=16)
    parser.add_argument("--arch-mlp-bot", type=str, default="13-512-256-64-16")
    parser.add_argument("--arch-mlp-top", type=str, default="512-256-1")
    parser.add_argument("--arch-interaction-op", type=str, default="dot")
    parser.add_argument("--arch-interaction-itself", action="store_true", default=False)
    parser.add_argument("--data-generation", type=str, default="dataset")
    parser.add_argument("--data-set", type=str, default="kaggle")
    parser.add_argument("--raw-data-file", type=str, default="./input/train.txt")
    parser.add_argument("--processed-data-file", type=str, default="./input/kaggleAdDisplayChallenge_processed.npz")
    parser.add_argument("--loss-function", type=str, default="bce")
    parser.add_argument("--max-ind-range", type=int, default=-1)
    parser.add_argument("--test-mini-batch-size", type=int, default=2048)
    parser.add_argument("--test-num-workers", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--mlperf-logging", action="store_true", default=False)
    parser.add_argument("--memory-map", action="store_true", default=False)
    parser.add_argument("--data-randomize", type=str, default="total")
    parser.add_argument("--data-trace-enable-padding", type=bool, default=False)
    parser.add_argument("--data-sub-sample-rate", type=float, default=0.0)
    parser.add_argument("--num-indices-per-lookup", type=int, default=10)
    parser.add_argument("--num-indices-per-lookup-fixed", type=bool, default=False)
    parser.add_argument("--mini-batch-size", type=int, default=128)
    parser.add_argument("--round-targets", type=bool, default=False)
    parser.add_argument("--mlperf-bin-loader", action="store_true", default=False)
    parser.add_argument("--mlperf-bin-shuffle", action="store_true", default=False)
    
    args = parser.parse_args()
    result = test_quantization_accuracy(args, args.bits)
