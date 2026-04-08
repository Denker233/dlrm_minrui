#!/usr/bin/env python3
"""
Compress CAFE's hash tables using video codec (H.265/HEVC)

This script proves that codec compression can be applied on top of CAFE's
already-compressed hash tables for additional compression.

Pipeline:
  Original DLRM (540MB) → CAFE (~3.6MB) → Video Codec (???MB)

Usage:
  python compress_cafe_tables.py --model path/to/cafe_model.pt --output-dir ./cafe_compressed
"""

import argparse
import os
import sys
import time
import tempfile
import subprocess
import numpy as np
import torch
from pathlib import Path


class VideoCodecCompressor:
    """
    Compress embedding tables using H.265 (HEVC) video codec.
    Treats embedding tables as 2D grayscale images.
    """
    
    def __init__(self, quality=23, codec='libx265', bit_depth=8, frame_size=16):
        """
        Args:
            quality: CRF value (0-51, lower = better quality, larger file)
                     Recommended: 15-25 for embeddings
            codec: 'libx265' (software) or 'hevc_qsv' (Intel hardware)
            bit_depth: 8 or 16 bit encoding
            frame_size: Size of each video frame (e.g., 4, 16, 64)
        """
        self.quality = quality
        self.codec = codec
        self.bit_depth = bit_depth
        self.frame_size = frame_size
        self._check_ffmpeg()
    
    def _check_ffmpeg(self):
        """Verify ffmpeg is available"""
        try:
            result = subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True, text=True, check=True
            )
            print(f"[Codec] ffmpeg available ✓")
        except Exception as e:
            raise RuntimeError(f"ffmpeg not found: {e}")
    
    def compress(self, embedding_table):
        """
        Compress an embedding table.
        
        Args:
            embedding_table: numpy array of shape (num_embeddings, embedding_dim)
        
        Returns:
            dict with compressed data and metadata for decompression
        """
        if isinstance(embedding_table, torch.Tensor):
            embedding_table = embedding_table.detach().cpu().numpy()
        
        original_shape = embedding_table.shape
        original_dtype = embedding_table.dtype
        original_size = embedding_table.nbytes
        
        # Store normalization parameters
        min_val = embedding_table.min()
        max_val = embedding_table.max()
        
        # Normalize to 0-255 for 8-bit encoding
        if max_val - min_val > 1e-8:
            normalized = (embedding_table - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(embedding_table)
        
        if self.bit_depth == 8:
            pixels = (normalized * 255).astype(np.uint8)
            pix_fmt = 'gray'
        else:  # 16-bit
            pixels = (normalized * 65535).astype(np.uint16)
            pix_fmt = 'gray16le'
        
        # Reshape into fixed frame size (e.g., 16x16) and create video with multiple frames
        height, width = pixels.shape
        
        # Use frame_size x frame_size blocks as video frames
        frame_size = self.frame_size  # e.g., 16x16
        
        # Flatten the data
        flat = pixels.flatten()
        total_pixels = len(flat)
        
        # Calculate how many pixels per frame
        pixels_per_frame = frame_size * frame_size
        
        # Pad to make it divisible by pixels_per_frame
        pad_needed = (pixels_per_frame - (total_pixels % pixels_per_frame)) % pixels_per_frame
        if pad_needed > 0:
            flat = np.pad(flat, (0, pad_needed), mode='constant', constant_values=0)
        
        total_pixels_padded = len(flat)
        num_frames = total_pixels_padded // pixels_per_frame
        
        # Reshape into frames: (num_frames, frame_size, frame_size)
        frames = flat.reshape(num_frames, frame_size, frame_size)
        
        reshaped_dims = (num_frames, frame_size, frame_size)
        padded_shape = reshaped_dims  # Same in this case
        
        # Compress using ffmpeg (multiple frames as video)
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = os.path.join(tmpdir, 'input.raw')
            compressed_path = os.path.join(tmpdir, 'output.mp4')
            
            # Write raw frames (all frames concatenated)
            frames.tofile(raw_path)
            
            # Build ffmpeg command for multi-frame video
            cmd = [
                'ffmpeg', '-y',
                '-f', 'rawvideo',
                '-pix_fmt', pix_fmt,
                '-s', f'{frame_size}x{frame_size}',
                '-r', '25',  # framerate
                '-i', raw_path,
                '-c:v', self.codec,
                '-crf', str(self.quality),
                compressed_path
            ]
            
            # Add codec-specific params
            if self.codec == 'libx265':
                cmd.insert(-1, '-x265-params')
                cmd.insert(-1, 'log-level=error')
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg compression failed: {result.stderr}")
            
            # Read compressed data
            with open(compressed_path, 'rb') as f:
                compressed_data = f.read()
        
        compressed_size = len(compressed_data)
        compression_ratio = original_size / compressed_size
        
        return {
            'compressed_data': compressed_data,
            'original_shape': original_shape,
            'reshaped_dims': reshaped_dims,
            'padded_shape': padded_shape,
            'original_dtype': str(original_dtype),
            'min_val': float(min_val),
            'max_val': float(max_val),
            'bit_depth': self.bit_depth,
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_ratio': compression_ratio,
        }
    
    def decompress(self, compressed_info):
        """
        Decompress an embedding table.
        
        Args:
            compressed_info: dict returned by compress()
        
        Returns:
            numpy array of shape (num_embeddings, embedding_dim)
        """
        compressed_data = compressed_info['compressed_data']
        original_shape = compressed_info['original_shape']
        reshaped_dims = compressed_info['reshaped_dims']  # (num_frames, frame_size, frame_size)
        min_val = compressed_info['min_val']
        max_val = compressed_info['max_val']
        bit_depth = compressed_info['bit_depth']
        
        if bit_depth == 8:
            pix_fmt = 'gray'
            dtype = np.uint8
            max_pixel = 255.0
        else:
            pix_fmt = 'gray16le'
            dtype = np.uint16
            max_pixel = 65535.0
        
        num_frames, frame_h, frame_w = reshaped_dims
        
        with tempfile.TemporaryDirectory() as tmpdir:
            compressed_path = os.path.join(tmpdir, 'input.mp4')
            raw_path = os.path.join(tmpdir, 'output.raw')
            
            # Write compressed data
            with open(compressed_path, 'wb') as f:
                f.write(compressed_data)
            
            # Decompress using ffmpeg
            cmd = [
                'ffmpeg', '-y',
                '-i', compressed_path,
                '-f', 'rawvideo',
                '-pix_fmt', pix_fmt,
                raw_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg decompression failed: {result.stderr}")
            
            # Read raw pixels
            pixels = np.fromfile(raw_path, dtype=dtype)
        
        # Reshape from frames back to flat
        frames = pixels.reshape(num_frames, frame_h, frame_w)
        flat = frames.flatten()
        
        # Extract original size
        original_size = original_shape[0] * original_shape[1]
        flat = flat[:original_size]
        
        # Reshape to original dimensions
        pixels = flat.reshape(original_shape)
        
        # Denormalize
        normalized = pixels.astype(np.float32) / max_pixel
        embedding_table = normalized * (max_val - min_val) + min_val
        
        return embedding_table


def extract_cafe_tables(model_path):
    """
    Extract embedding tables from a CAFE model.
    
    Returns:
        dict mapping table names to numpy arrays
    """
    print(f"\n[1] Loading CAFE model from: {model_path}")
    
    # Load model checkpoint (handle newer PyTorch versions)
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    except Exception:
        # Fallback for models saved with numpy arrays
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint.state_dict() if hasattr(checkpoint, 'state_dict') else checkpoint
    
    tables = {}
    total_size = 0
    
    print(f"\n[2] Extracting embedding tables...")
    print(f"{'Table Name':<50} {'Shape':<20} {'Size (KB)':<15}")
    print("-" * 85)
    
    for name, param in state_dict.items():
        # Look for embedding-related weights
        # CAFE uses: weight_h (hot), weight_hash (hash table)
        # Also check for standard embedding weights
        if any(keyword in name.lower() for keyword in ['weight_h', 'weight_hash', 'emb', 'embedding']):
            if isinstance(param, torch.Tensor) and param.dim() == 2:
                arr = param.detach().cpu().numpy()
                size_kb = arr.nbytes / 1024
                total_size += arr.nbytes
                tables[name] = arr
                print(f"{name:<50} {str(arr.shape):<20} {size_kb:<15.2f}")
    
    print("-" * 85)
    print(f"{'TOTAL':<50} {'':<20} {total_size/1024:<15.2f}")
    
    return tables, total_size


def compress_all_tables(tables, compressor, output_dir=None):
    """
    Compress all extracted tables and report results.
    """
    print(f"\n[3] Compressing tables with video codec (quality={compressor.quality})...")
    print(f"{'Table Name':<50} {'Original (KB)':<15} {'Compressed (KB)':<15} {'Ratio':<10}")
    print("-" * 90)
    
    results = {}
    total_original = 0
    total_compressed = 0
    
    for name, table in tables.items():
        try:
            start_time = time.time()
            compressed = compressor.compress(table)
            compress_time = time.time() - start_time
            
            orig_kb = compressed['original_size'] / 1024
            comp_kb = compressed['compressed_size'] / 1024
            ratio = compressed['compression_ratio']
            
            total_original += compressed['original_size']
            total_compressed += compressed['compressed_size']
            
            results[name] = compressed
            
            print(f"{name:<50} {orig_kb:<15.2f} {comp_kb:<15.2f} {ratio:<10.2f}x")
            
            # Save compressed data if output_dir specified
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                safe_name = name.replace('.', '_').replace('/', '_')
                
                # Save compressed video
                video_path = os.path.join(output_dir, f"{safe_name}.mp4")
                with open(video_path, 'wb') as f:
                    f.write(compressed['compressed_data'])
                
                # Save metadata
                meta_path = os.path.join(output_dir, f"{safe_name}_meta.npz")
                np.savez(meta_path,
                         original_shape=compressed['original_shape'],
                         padded_shape=compressed['padded_shape'],
                         min_val=compressed['min_val'],
                         max_val=compressed['max_val'],
                         bit_depth=compressed['bit_depth'])
        
        except Exception as e:
            print(f"{name:<50} FAILED: {e}")
            continue
    
    print("-" * 90)
    overall_ratio = total_original / total_compressed if total_compressed > 0 else 0
    print(f"{'TOTAL':<50} {total_original/1024:<15.2f} {total_compressed/1024:<15.2f} {overall_ratio:<10.2f}x")
    
    return results, total_original, total_compressed


def verify_compression(tables, results, compressor):
    """
    Verify decompression accuracy.
    """
    print(f"\n[4] Verifying decompression accuracy...")
    print(f"{'Table Name':<50} {'MSE':<15} {'Max Error':<15} {'PSNR (dB)':<10}")
    print("-" * 90)
    
    for name, compressed in results.items():
        try:
            original = tables[name]
            reconstructed = compressor.decompress(compressed)
            
            mse = np.mean((original - reconstructed) ** 2)
            max_err = np.max(np.abs(original - reconstructed))
            
            # PSNR calculation
            data_range = original.max() - original.min()
            if data_range > 0 and mse > 0:
                psnr = 20 * np.log10(data_range / np.sqrt(mse))
            else:
                psnr = float('inf')
            
            print(f"{name:<50} {mse:<15.2e} {max_err:<15.4f} {psnr:<10.2f}")
        
        except Exception as e:
            print(f"{name:<50} FAILED: {e}")


def main():
    parser = argparse.ArgumentParser(description='Compress CAFE embedding tables with video codec')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to CAFE model checkpoint (.pt file)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save compressed tables')
    parser.add_argument('--quality', type=int, default=23,
                        help='Compression quality (CRF): 0-51, lower=better quality (default: 23)')
    parser.add_argument('--codec', type=str, default='libx265',
                        choices=['libx265', 'hevc_qsv', 'libx264', 'h264_qsv'],
                        help='Video codec to use (default: libx265)')
    parser.add_argument('--bit-depth', type=int, default=8, choices=[8, 16],
                        help='Bit depth for encoding (default: 8)')
    parser.add_argument('--frame-size', type=int, default=16,
                        help='Frame size for video encoding, e.g., 4, 16, 64 (default: 16)')
    parser.add_argument('--verify', action='store_true',
                        help='Verify decompression accuracy')
    parser.add_argument('--quality-sweep', action='store_true',
                        help='Test multiple quality levels')
    args = parser.parse_args()
    
    print("=" * 90)
    print("CAFE Hash Table Compression with Video Codec")
    print("=" * 90)
    
    # Extract tables
    tables, original_total = extract_cafe_tables(args.model)
    
    if not tables:
        print("ERROR: No embedding tables found in model!")
        sys.exit(1)
    
    if args.quality_sweep:
        # Test multiple quality levels
        print("\n" + "=" * 90)
        print("QUALITY SWEEP")
        print("=" * 90)
        
        quality_levels = [15, 20, 23, 28, 35, 45]
        
        print(f"\n{'Quality (CRF)':<15} {'Compressed (KB)':<20} {'Ratio':<15} {'Notes':<30}")
        print("-" * 80)
        
        for q in quality_levels:
            compressor = VideoCodecCompressor(quality=q, codec=args.codec, bit_depth=args.bit_depth, frame_size=args.frame_size)
            _, _, total_compressed = compress_all_tables(tables, compressor)
            ratio = original_total / total_compressed if total_compressed > 0 else 0
            
            notes = ""
            if q <= 18:
                notes = "Near-lossless"
            elif q <= 23:
                notes = "High quality"
            elif q <= 28:
                notes = "Medium quality"
            else:
                notes = "Lower quality"
            
            print(f"{q:<15} {total_compressed/1024:<20.2f} {ratio:<15.2f}x {notes:<30}")
    else:
        # Single compression run
        compressor = VideoCodecCompressor(
            quality=args.quality,
            codec=args.codec,
            bit_depth=args.bit_depth,
            frame_size=args.frame_size
        )
        
        results, total_original, total_compressed = compress_all_tables(
            tables, compressor, args.output_dir
        )
        
        if args.verify:
            verify_compression(tables, results, compressor)
        
        # Summary
        print("\n" + "=" * 90)
        print("SUMMARY")
        print("=" * 90)
        print(f"Original DLRM size (estimated):  540 MB")
        print(f"CAFE compressed size:            {total_original/1024/1024:.2f} MB")
        print(f"Codec compressed size:           {total_compressed/1024/1024:.2f} MB")
        print(f"")
        print(f"CAFE compression ratio:          {540*1024*1024/total_original:.2f}x (from original)")
        print(f"Codec compression ratio:         {total_original/total_compressed:.2f}x (from CAFE)")
        print(f"Total compression ratio:         {540*1024*1024/total_compressed:.2f}x (from original)")
        
        if args.output_dir:
            print(f"\nCompressed files saved to: {args.output_dir}")


if __name__ == '__main__':
    main()