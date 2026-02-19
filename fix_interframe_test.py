#!/usr/bin/env python3
"""Fix inter-frame test by using proper frame dimensions"""
import os
import sys
import time
import json
import tempfile
import subprocess
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = "./models/dlrm_kaggle_quick.pt"
RESULTS_DIR = os.path.expanduser("~/experiment-control")


def run_interframe_test():
    """Test inter-frame prediction with proper frame dimensions"""
    state_dict = torch.load(MODEL_PATH, map_location='cpu')['state_dict']
    emb_keys = [k for k in state_dict.keys() if 'emb_l' in k and 'weight' in k]

    # Quantize all tables
    all_quantized = []
    raw_uint8_size = 0
    float32_size = 0

    for key in emb_keys:
        weights = state_dict[key]
        w_min, w_max = weights.min(), weights.max()
        scale = (w_max - w_min) / 255.0
        zero_point = -(w_min / scale).round() if scale > 0 else 0.0
        quantized = ((weights / scale).round() + zero_point).clamp(0, 255).to(torch.uint8)
        all_quantized.append(quantized.numpy())
        raw_uint8_size += quantized.numel()
        float32_size += weights.numel() * 4

    # For multi-frame: each table becomes one frame
    # All frames must have same dimensions
    # Use a fixed frame size and pad/truncate tables
    FRAME_WIDTH = 512
    FRAME_HEIGHT = 1024
    FRAME_SIZE = FRAME_WIDTH * FRAME_HEIGHT

    # Build uniform frames
    frames = []
    for q in all_quantized:
        flat = q.reshape(-1)
        if len(flat) > FRAME_SIZE:
            # Split into multiple frames
            n_frames = (len(flat) + FRAME_SIZE - 1) // FRAME_SIZE
            padded = np.zeros(n_frames * FRAME_SIZE, dtype=np.uint8)
            padded[:len(flat)] = flat
            for j in range(n_frames):
                frames.append(padded[j*FRAME_SIZE:(j+1)*FRAME_SIZE])
        else:
            frame = np.zeros(FRAME_SIZE, dtype=np.uint8)
            frame[:len(flat)] = flat
            frames.append(frame)

    num_frames = len(frames)
    raw_multiframe = b''.join(f.tobytes() for f in frames)

    print(f"Frames: {num_frames}, each {FRAME_WIDTH}x{FRAME_HEIGHT}")
    print(f"Total raw data: {len(raw_multiframe)} bytes ({len(raw_multiframe)/1024/1024:.2f} MB)")
    print(f"Original uint8 size: {raw_uint8_size} bytes ({raw_uint8_size/1024/1024:.2f} MB)")
    print(f"Original float32 size: {float32_size} bytes ({float32_size/1024/1024:.2f} MB)")

    configs = [
        {
            'name': 'Multi-frame: I-frames only (keyint=1)',
            'args': ['-c:v', 'libx265', '-crf', '23', '-preset', 'ultrafast',
                    '-x265-params', 'log-level=error:allow-non-conformance=1:keyint=1:min-keyint=1:scenecut=0'],
        },
        {
            'name': 'Multi-frame: with inter-frame (keyint=250)',
            'args': ['-c:v', 'libx265', '-crf', '23', '-preset', 'ultrafast',
                    '-x265-params', 'log-level=error:allow-non-conformance=1:keyint=250'],
        },
        {
            'name': 'Multi-frame: with inter-frame (keyint=999)',
            'args': ['-c:v', 'libx265', '-crf', '23', '-preset', 'ultrafast',
                    '-x265-params', 'log-level=error:allow-non-conformance=1:keyint=999'],
        },
    ]

    results = []
    for config in configs:
        name = config['name']
        print(f"\nTesting: {name}")

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                raw_file = os.path.join(tmpdir, 'multi.raw')
                video_file = os.path.join(tmpdir, 'multi.mp4')
                decoded_file = os.path.join(tmpdir, 'decoded.raw')

                with open(raw_file, 'wb') as f:
                    f.write(raw_multiframe)

                cmd = [
                    'ffmpeg', '-y',
                    '-f', 'rawvideo', '-pix_fmt', 'gray',
                    '-s', f'{FRAME_WIDTH}x{FRAME_HEIGHT}',
                    '-r', '30',
                    '-i', raw_file,
                ] + config['args'] + [video_file]

                print(f"  CMD: {' '.join(cmd[:20])}...")
                t0 = time.time()
                r = subprocess.run(cmd, capture_output=True, text=True, check=False)
                ct = time.time() - t0

                if r.returncode != 0:
                    print(f"  FAILED: {r.stderr[:500]}")
                    results.append({
                        'name': name,
                        'compressed_size': 0,
                        'compress_time': ct,
                        'decompress_time': 0,
                        'error': r.stderr[:200],
                    })
                    continue

                compressed_size = os.path.getsize(video_file)

                # Decompress
                cmd_dec = ['ffmpeg', '-y', '-i', video_file,
                          '-pix_fmt', 'gray', '-f', 'rawvideo', decoded_file]
                t0 = time.time()
                subprocess.run(cmd_dec, capture_output=True, text=True, check=False)
                dt = time.time() - t0

                bits_per_value = (compressed_size * 8) / raw_uint8_size
                comp_vs_uint8 = raw_uint8_size / compressed_size
                comp_vs_float32 = float32_size / compressed_size

                results.append({
                    'name': name,
                    'compressed_size': compressed_size,
                    'comp_ratio_vs_uint8': comp_vs_uint8,
                    'comp_ratio_vs_float32': comp_vs_float32,
                    'bits_per_value': bits_per_value,
                    'compress_time': ct,
                    'decompress_time': dt,
                    'failures': 0,
                })

                print(f"  Size: {compressed_size} bytes ({compressed_size/1024/1024:.2f} MB)")
                print(f"  Ratio vs uint8: {comp_vs_uint8:.2f}x, vs float32: {comp_vs_float32:.2f}x")
                print(f"  Bits/value: {bits_per_value:.4f}")
                print(f"  Compress: {ct:.2f}s, Decompress: {dt:.2f}s")

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                'name': name,
                'compressed_size': 0,
                'compress_time': 0,
                'decompress_time': 0,
                'error': str(e),
            })

    # Print summary
    print("\n" + "=" * 80)
    print("INTER-FRAME TEST RESULTS")
    print("=" * 80)
    for r in results:
        print(f"  {r['name']}:")
        if r.get('compressed_size', 0) > 0:
            print(f"    Size: {r['compressed_size']} ({r['compressed_size']/1024/1024:.2f} MB)")
            print(f"    Ratio vs uint8: {r['comp_ratio_vs_uint8']:.2f}x")
            print(f"    Bits/value: {r['bits_per_value']:.4f}")
            print(f"    Times: compress={r['compress_time']:.2f}s decompress={r['decompress_time']:.2f}s")
        else:
            print(f"    FAILED: {r.get('error', 'unknown')}")

    # Calculate inter-frame contribution
    iframes = next((r for r in results if 'I-frames only' in r['name'] and r.get('compressed_size', 0) > 0), None)
    inter = next((r for r in results if 'keyint=250' in r['name'] and r.get('compressed_size', 0) > 0), None)

    if iframes and inter:
        savings = iframes['compressed_size'] - inter['compressed_size']
        print(f"\n  Inter-frame savings: {savings} bytes ({savings/1024/1024:.2f} MB)")
        print(f"  Inter-frame reduces size by {savings/iframes['compressed_size']*100:.1f}%")

    return results


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    results = run_interframe_test()

    # Save results as JSON
    with open(os.path.join(RESULTS_DIR, 'interframe_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_DIR}/interframe_results.json")
