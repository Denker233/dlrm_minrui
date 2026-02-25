# CLAUDE.md — Project Context for Claude Code

## Project
DLRM (Deep Learning Recommendation Model) embedding table compression using H.265 video codecs on CPU.

## Branch
`cpu_codecs` — all work happens here.

## Key Files
- `prefetch_benchmark_v10_h265.py` — core H.265 encode/decode, hot/cold split, quantization
- `prefetch_benchmark_v11_sequential.py` — frequency-sorted layout, LRU cache, multi-batch grouping
- `benchmark_frame_reorder.py` — frame mapping and batch analysis
- `dlrm_s_pytorch.py` — base DLRM model and inference loop

## Dataset
Kaggle/Criteo at `~/input/train.txt` or `~/input/kaggleAdDisplayChallenge_processed.npz`

## Critical Rules
1. **PyAV for decoding** — `import av`, NOT `subprocess.run(['ffmpeg', ...])`. PyAV is 5-10x faster.
2. **NEVER decompress all cold frames** — bounded LRU cache only. No full-table tensors.
3. **Vectorized ops only** — no Python for-loops over embedding indices.
4. **Check for existing work** — look for `results/profiling/.done`, `results/hotcold/.done`, `results/reorder/.done` markers before re-running expensive phases.

## Working Directories
- `results/profiling/` — access frequency, CDF, co-occurrence data
- `results/hotcold/` — hot/cold split tensors
- `results/reorder/` — reordered tables + compressed .mp4 files
- `logs/` — experiment logs