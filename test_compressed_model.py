#!/usr/bin/env python3
"""
Test accuracy of compressed DLRM model
"""

import torch
import numpy as np
import argparse
from dlrm_s_pytorch import QSVEmbeddingCompressor
import dlrm_data_pytorch as dp

def decompress_model(compressed_path, original_model_path):
    """
    Load compressed model and decompress embeddings
    """
    print("Loading compressed model...")
    compressed = torch.load(compressed_path)
    
    print("Loading original model structure...")
    original = torch.load(original_model_path)
    
    # Create decompressor
    compressor = QSVEmbeddingCompressor(
        codec=compressed['compression_info']['codec'],
        quality=compressed['compression_info']['quality'],
        quantization=compressed['compression_info']['quantization']
    )
    
    print(f"Decompressing {len(compressed['compressed_tables'])} embedding tables...")
    
    # Decompress each table
    decompressed_embs = []
    for i, (compressed_data, metadata) in enumerate(compressed['compressed_tables']):
        print(f"  Table {i}: {metadata['shape']}", end=' ')
        weights = compressor.decompress_table(compressed_data, metadata)
        decompressed_embs.append(weights)
        print("✓")
    
    # Put decompressed embeddings back into model state dict
    state_dict = original['state_dict'].copy()
    
    emb_idx = 0
    for key in state_dict.keys():
        if 'emb_l' in key and 'weight' in key:
            state_dict[key] = decompressed_embs[emb_idx]
            emb_idx += 1
    
    # Update model with decompressed embeddings
    decompressed_model = original.copy()
    decompressed_model['state_dict'] = state_dict
    
    return decompressed_model

def test_accuracy(model_path, args):
    """
    Test model accuracy on test set
    """
    import sys
    sys.path.insert(0, '.')
    
    # Import DLRM
    from dlrm_s_pytorch import DLRM_Net
    
    print("\n" + "="*80)
    print("LOADING TEST DATA")
    print("="*80)
    
    # Load data (reuse from training)
    train_data, train_ld, test_data, test_ld = dp.make_criteo_data_and_loaders(args)
    
    ln_emb = train_data.counts
    if args.max_ind_range > 0:
        ln_emb = np.array(
            list(map(lambda x: x if x < args.max_ind_range else args.max_ind_range, ln_emb))
        )
    else:
        ln_emb = np.array(ln_emb)
    
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
    
    print("\n" + "="*80)
    print("CREATING MODEL")
    print("="*80)
    
    # Create model
    dlrm = DLRM_Net(
        m_spa, ln_emb, ln_bot, ln_top,
        arch_interaction_op=args.arch_interaction_op,
        arch_interaction_itself=args.arch_interaction_itself,
        sigmoid_bot=-1,
        sigmoid_top=ln_top.size - 2,
        loss_function=args.loss_function,
    )
    
    # Load model weights
    print(f"Loading model from: {model_path}")
    ld_model = torch.load(model_path, map_location=torch.device("cpu"))
    dlrm.load_state_dict(ld_model['state_dict'])
    dlrm.eval()
    
    print("\n" + "="*80)
    print("RUNNING INFERENCE")
    print("="*80)
    
    # Run inference
    device = torch.device("cpu")
    test_accu = 0
    test_samp = 0
    
    with torch.no_grad():
        for i, testBatch in enumerate(test_ld):
            if i % 100 == 0:
                print(f"Batch {i}/{len(test_ld)}", end='\r')
            
            # Unpack batch
            X_test, lS_o_test, lS_i_test, T_test, W_test, CBPP_test = testBatch[0], testBatch[1], testBatch[2], testBatch[3], torch.ones(testBatch[3].size()), None
            
            # Forward pass
            Z_test = dlrm(X_test.to(device), lS_o_test, lS_i_test)
            
            # Compute accuracy
            S_test = Z_test.detach().cpu().numpy()
            T_test_np = T_test.detach().cpu().numpy()
            
            mbs_test = T_test_np.shape[0]
            A_test = np.sum((np.round(S_test, 0) == T_test_np).astype(np.uint8))
            
            test_accu += A_test
            test_samp += mbs_test
    
    accuracy = test_accu / test_samp
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Test Accuracy: {accuracy*100:.4f}%")
    print(f"Test Samples:  {test_samp}")
    print("="*80)
    
    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Model paths
    parser.add_argument("--compressed-model", type=str, required=True)
    parser.add_argument("--original-model", type=str, required=True)
    parser.add_argument("--decompress-first", action="store_true", help="Decompress model first")
    
    # Data args (must match training)
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
    
    # Additional required args for data loader
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
    
    if args.decompress_first:
        # Decompress and save
        decompressed = decompress_model(args.compressed_model, args.original_model)
        temp_path = args.compressed_model.replace('.pt', '_decompressed.pt')
        torch.save(decompressed, temp_path)
        print(f"Saved decompressed model to: {temp_path}")
        model_path = temp_path
    else:
        model_path = args.original_model
    
    # Test accuracy
    accuracy = test_accuracy(model_path, args)
