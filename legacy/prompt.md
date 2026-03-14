# CPU Video Codec Compression for DLRM Embedding Tables
# With Hypergraph Co-Access Reordering + Multi-Batch Lookahead Prefetching

## Project Context

This project implements H.265 video codec compression of DLRM embedding tables on CPU with on-demand frame decompression, co-access reordering, and multi-batch prefetching. The codebase is at `~/expr/dlrm_minrui` on branch `cpu_codecs`.

**Read the existing CPU implementation first** before writing any code:
- `prefetch_benchmark_v10_h265.py` — core H.265 encode/decode, hot/cold split, quantization
- `prefetch_benchmark_v11_sequential.py` — frequency-sorted layout, LRU cache, multi-batch grouping
- `benchmark_frame_reorder.py` — frame mapping and batch analysis
- `dlrm_s_pytorch.py` — base DLRM model and inference loop
- Any `CLAUDE.md` or documentation files in the repo

Use the Kaggle/Criteo dataset at `~/input/train.txt` or `~/input/kaggleAdDisplayChallenge_processed.npz`. Find the trained model: `find ~ -name "*.pt" -path "*/model*" 2>/dev/null | head -20`. If no model exists, train one first.

---

## Core Architecture Rules (NON-NEGOTIABLE)

These rules define the architecture. All optimizations must be built on top of them, NEVER violate them. Claude Code implementations have repeatedly violated Rule 2 by decompressing all cold frames into memory at startup. THIS MUST NOT HAPPEN.

### Rule 1: Hot/Cold Splitting
- Large embedding tables MUST be split into hot and cold partitions based on access frequency.
- Hot embeddings (frequently accessed, top ~0.05-7% by frequency covering ~76-80% of accesses) stay as fp32 tensors in memory permanently.
- Cold embeddings are stored as H.265 compressed video frames on disk or in a compressed byte buffer in memory.
- Use access frequency profiling over training/test batches to determine the hot/cold threshold.

### Rule 2: On-Demand Frame Decompression (NO Full Pre-Decode) ⚠️ CRITICAL ⚠️
- You MUST NOT decompress all cold embedding frames into memory at once.
- You MUST NOT hold a tensor of shape (num_cold_rows, emb_dim) in memory at any point during inference.
- Decompression happens on-demand: only decode the frames actually needed for the current (and prefetched) batches.
- Compressed frames live on disk (or as a compressed byte buffer); decoded frames exist ONLY inside the bounded LRU cache.
- **Memory assertion:** At every batch boundary, assert that the total memory used by cached decoded frames <= `cache_capacity * frame_size * emb_dim * 4 bytes`. If this assertion fails, STOP and fix the leak.
- **Anti-patterns to watch for:**
  - `cold_weights = decompress_all_frames(...)` ← FORBIDDEN
  - `nn.Embedding(num_cold, dim)` for cold table ← FORBIDDEN (this allocates the full table)
  - Loading compressed file then immediately decompressing to a full tensor ← FORBIDDEN
  - Any code path that accumulates decoded frames without eviction ← FORBIDDEN

### Rule 3: Prefetching with Bounded Cache
- Look ahead at the next K batches (K=1 to 10) to identify which frames will be needed.
- Use PyAV CPU decoding to decompress needed frames ahead of time in background threads.
- Maintain a priority-aware LRU cache of decoded frames with a **strictly bounded size** (e.g., 60-200 frames).
- The cache must NEVER grow beyond its configured capacity — enforce with assertion.
- Track and log cache hit rate, demand decompressions, and prefetch accuracy per batch.
- **Memory budget formula:** `cache_memory_bytes = cache_capacity × frame_size × emb_dim × 4`
  - Example: 100 frames × 4096 rows × 16 dims × 4 bytes = 25 MB ← this is fine
  - Example: 500 frames × 4096 rows × 16 dims × 4 bytes = 125 MB ← still fine vs 2 GB original
  - If cache_memory > 500 MB, the cache is too large — reduce capacity

### Rule 4: Batch Index Sorting for Frame Locality
- Before looking up cold embeddings, sort the embedding indices within each batch.
- Sorting groups accesses to the same frame together, reducing the number of unique frames touched per batch.
- This directly improves cache hit rate and reduces decode operations.

### Rule 5: Vectorized Operations
- All embedding-to-frame and frame-to-embedding transformations must use vectorized operations (PyTorch tensor ops or numpy vectorized).
- No Python for-loops over individual rows or indices in the hot path.
- Use `torch.gather`, `torch.index_select`, advanced indexing, boolean masking.
- Hot/cold classification MUST be vectorized: `hot_mask = is_hot[indices]`, not a Python loop.

### Rule 6: Comprehensive Metrics
Every experiment must measure and report:
- **AUC** — compare against uncompressed baseline
- **Total wall-clock time** — end-to-end including all overhead
- **Inference time** — pure model forward pass (embedding lookup + MLP + interaction)
- **Embedding lookup time** — broken down: hot lookup, cold cache-hit, cold demand-decode
- **Compression ratio** — original table size vs compressed size on disk/in-memory
- **Peak RSS memory** — via `psutil.Process().memory_info().rss` or `/proc/self/status`
- **Cache hit rate** — per-batch and overall average
- **Demand decompressions per batch** — THE key metric: frames decoded on critical path
- **Prefetch accuracy** — % of actually-needed frames that were prefetched in time
- **Decode throughput** — frames/second via PyAV CPU decode
- **AUC degradation** — difference from fp32 baseline (target: < 0.15pp)

### Rule 7: CPU-Specific Decompression via PyAV
- Use `PyAV` (Python binding for libav/FFmpeg) for H.265 frame decoding. No GPU/NVDEC. No subprocess calls to ffmpeg.
- PyAV gives direct in-process access to the decoder — no subprocess overhead, no pipe serialization, no shell spawning per frame.
- Install if needed: `pip install av --break-system-packages`
- For ALL-INTRA video (keyint=1), every frame is a keyframe, so seeking to any frame is instant.
- Use `container.seek(frame_id, stream=video_stream)` or iterate with `container.decode(video=0)` and cache the frame index.
- **Do NOT use `subprocess.run(['ffmpeg', ...])` for decoding.** PyAV is 5-10x faster for single-frame random access because it avoids process creation overhead.
- Measure per-frame decode latency to establish the prefetch time budget.

---

## CHECKPOINT SYSTEM — Skip Completed Work

Before starting ANY phase, check if its outputs already exist. If they do, load them and skip to the next phase. This avoids re-running expensive profiling, training, or encoding steps.

```python
import os

def check_done(marker_file):
    """Check if a phase was already completed."""
    return os.path.exists(marker_file)

def mark_done(marker_file):
    """Mark a phase as completed."""
    os.makedirs(os.path.dirname(marker_file), exist_ok=True)
    with open(marker_file, 'w') as f:
        f.write(f"Completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
```

**At the start of each phase, check these markers and SKIP if present:**

| Phase | Marker File | Key Outputs to Check | Skip Condition |
|-------|-------------|---------------------|----------------|
| 1. Profiling | `results/profiling/.done` | `access_frequency_*.npy`, `cooccurrence_*.npz`, `frame_overlap_*.npy` | All output files exist |
| 2. Hot/Cold Split | `results/hotcold/.done` | `hot_indices_*.pt`, `cold_indices_*.pt`, `is_hot_*.pt`, `orig_to_hot_*.pt`, `orig_to_cold_*.pt` | All mapping tensors exist |
| 3. Reordering | `results/reorder/.done` | `cold_order_*.npy`, `reordered_orig_to_cold_*.pt`, `compressed_cold_*.mp4` | Reordered + compressed files exist |
| 4. Inference | — | — | Never skip (this is what we're benchmarking) |

**Implementation pattern for every phase:**

```python
PROFILING_DONE = 'results/profiling/.done'
if check_done(PROFILING_DONE):
    print("Phase 1 SKIPPED — loading existing profiling data")
    freq = {t: np.load(f'results/profiling/freq_table_{t}.npy') for t in range(26)}
    # ... load all profiling outputs
else:
    print("Phase 1 — Running access pattern profiling")
    # ... do the profiling work
    mark_done(PROFILING_DONE)

HOTCOLD_DONE = 'results/hotcold/.done'
if check_done(HOTCOLD_DONE):
    print("Phase 2 SKIPPED — loading existing hot/cold split")
    is_hot = {t: torch.load(f'results/hotcold/is_hot_{t}.pt') for t in range(26)}
    orig_to_hot = {t: torch.load(f'results/hotcold/orig_to_hot_{t}.pt') for t in range(26)}
    orig_to_cold = {t: torch.load(f'results/hotcold/orig_to_cold_{t}.pt') for t in range(26)}
    hot_emb_weight = {t: torch.load(f'results/hotcold/hot_weight_{t}.pt') for t in range(26)}
    # ... load all mappings
else:
    print("Phase 2 — Building hot/cold split")
    # ... do the split
    mark_done(HOTCOLD_DONE)

REORDER_DONE = 'results/reorder/.done'
if check_done(REORDER_DONE):
    print("Phase 3 SKIPPED — loading existing reorder + compressed files")
    orig_to_cold_reordered = {t: torch.load(f'results/reorder/orig_to_cold_reordered_{t}.pt') for t in range(26)}
    # compressed .mp4 files already on disk
else:
    print("Phase 3 — Hypergraph partitioning + H.265 encoding")
    # ... do the reordering and encoding
    mark_done(REORDER_DONE)

# Phase 4: ALWAYS run — this is the benchmark
print("Phase 4 — Running inference benchmarks")
```

**IMPORTANT:** When loading checkpointed data, verify it's consistent:
- Check that hot_indices + cold_indices cover the full table
- Check that compressed .mp4 files are valid (PyAV can open them)
- If any checkpoint file is corrupted or missing, re-run that phase from scratch

Also check for existing work from previous runs of THIS script and from the existing v10/v11 benchmarks. For example:
- If `prefetch_benchmark_v10_h265.py` already generated access frequency data, reuse it
- If a trained model checkpoint exists, don't retrain
- If the dataset is already preprocessed as `.npz`, use it directly

---

## Phase 1: Access Pattern Profiling

This data feeds everything else. Do this first.

1. **Run inference on the test set** and log, for EVERY batch:
   - Which embedding indices are accessed in each of the 26 tables
   - Save as compact format: list of (table_idx, set_of_indices) per batch
   - Use batch_size=2048, process the entire test set

2. **Compute per-index access frequency** for each table:
   - `freq[table_idx][emb_idx] = count of batches that access this index`
   - Save the full frequency arrays

3. **Compute access CDF** for each table:
   - Sort indices by frequency (descending)
   - Compute cumulative fraction of total accesses
   - Find and report: how many indices (and what % of table) cover 50%, 80%, 90%, 95% of accesses
   - Save CDF data and generate plots (matplotlib) for the top 5 largest tables

4. **Compute co-access statistics for cold embeddings:**
   - For each table, define "hot" = indices covering top 80% of accesses
   - For the cold indices only: build co-occurrence counts
   - For each pair of cold indices (i, j), count batches containing BOTH
   - Expensive for large tables — use sampling: 10,000 cold indices, 20,000 batches. Build scipy.sparse co-occurrence matrix.
   - Report: average co-occurrence density, distribution

5. **Compute temporal frame access patterns (for prefetch predictor):**
   - After defining frames (blocks of FRAME_SIZE rows), log the ORDERED sequence of frame sets per batch
   - For consecutive batch pairs (T, T+1), compute frame overlap: `|frames_T ∩ frames_{T+1}| / |frames_{T+1}|`
   - Report: average overlap, median, distribution
   - This tells us if prefetching is viable (>70% overlap = good)
   - Save the full frame access sequence for predictor training

6. **Measure the prefetch time budget:**
   - Run baseline inference and time separately: embedding lookup, bottom MLP, feature interaction, top MLP
   - `prefetch_budget_ms = time(bot_mlp + interact + top_mlp)` — this is free time for background decode
   - Time single-frame PyAV CPU decode latency
   - Report: `prefetch_budget_ms`, `decode_per_frame_ms`, `max_free_decodes_per_batch`

Save all profiling results to `results/profiling/`.

## Phase 2: Hot/Cold Split Implementation

Build a tiered embedding lookup that is FAST. Everything must be vectorized.

```python
# === AT INIT (once) ===
# Determine hot threshold from access frequency
hot_threshold = find_threshold(freq, coverage=0.80)
hot_indices = torch.where(freq >= hot_threshold)[0]
num_hot = len(hot_indices)

# Create hot embedding table — ONLY the hot rows, as fp32
hot_emb_weight = original_weights[hot_indices].clone()  # shape: (num_hot, emb_dim)

# Create mapping tensors (vectorized, no Python loops)
is_hot = torch.zeros(num_embeddings, dtype=torch.bool)
is_hot[hot_indices] = True

orig_to_hot = torch.full((num_embeddings,), -1, dtype=torch.long)
orig_to_hot[hot_indices] = torch.arange(num_hot)

# Cold indices: everything not hot
cold_indices = torch.where(~is_hot)[0]
num_cold = len(cold_indices)
orig_to_cold = torch.full((num_embeddings,), -1, dtype=torch.long)
orig_to_cold[cold_indices] = torch.arange(num_cold)

# === AT LOOKUP (every batch, fully vectorized) ===
def tiered_lookup(indices, offsets):
    hot_mask = is_hot[indices]
    
    # Hot path: direct tensor lookup (fast, cache-resident)
    hot_idx = orig_to_hot[indices[hot_mask]]
    hot_result = hot_emb_weight[hot_idx]
    
    # Cold path: lookup from frame cache (bounded, on-demand decode)
    cold_mapped = orig_to_cold[indices[~hot_mask]]
    cold_result = frame_cache.lookup(cold_mapped)  # uses bounded LRU cache
    
    # Merge results
    result = torch.zeros(len(indices), emb_dim)
    result[hot_mask] = hot_result
    result[~hot_mask] = cold_result
    
    # Pool with offsets (EmbeddingBag-style)
    return pool_embeddings(result, offsets)
```

**Verification:**
- AUC must be IDENTICAL to baseline when cold is uncompressed (lossless split)
- Hot buffer memory should fit in L3 cache — check with `lscpu | grep cache`
- Measure classification overhead — must be < 0.1ms per batch

## Phase 3: Hypergraph Co-Access Partitioning of Cold Region

Reorder cold embeddings so co-accessed ones are adjacent, then pack into codec frames.

1. **Build co-access graph for cold embeddings of each large table:**
   ```python
   # For tables with num_cold > 10,000:
   # Sample 20,000 batches, build sparse co-occurrence matrix
   from scipy.sparse import lil_matrix
   from collections import defaultdict
   
   cold_to_batches = defaultdict(set)
   for batch_id, batch_cold_indices in enumerate(sampled_batches):
       for idx in batch_cold_indices:
           cold_to_batches[idx].add(batch_id)
   # Build co-occurrence via inverted index intersection
   ```

2. **Partition cold embeddings into frame-sized groups:**

   Target: groups of `FRAME_SIZE` cold embeddings (e.g., 4096 per group). Each group = one codec frame.

   **Greedy clustering:**
   ```
   1. Start with highest total co-access count cold embedding
   2. Greedily add embeddings with highest co-access to current group
   3. When group reaches FRAME_SIZE, close it and start new group
   4. Repeat until all assigned
   ```

   **Spectral ordering (alternative):**
   ```
   1. Build co-occurrence Laplacian matrix (sampled)
   2. Compute Fiedler vector (2nd smallest eigenvector)
   3. Sort cold embeddings by Fiedler vector value
   4. Chunk into groups of FRAME_SIZE
   ```

   Implement BOTH, compare which gives better frame locality.

3. **Create reordered cold embedding table and compress:**
   ```python
   reordered_cold_weights = cold_weights[cold_order]
   # Update mapping: orig_to_cold_reordered[original_idx] = position in reordered table
   ```

4. **Compress reordered cold table with H.265:**
   - Multi-frame video: each frame = one group of FRAME_SIZE rows
   - ALL-INTRA encoding (keyint=1) — each frame independently decodable
   - Encoding is a one-time offline step: can use ffmpeg subprocess or PyAV for encoding
   - **Decoding at runtime MUST use PyAV** (in-process, no subprocess overhead)
   - CRF sweep: 18, 23, 28
   - INT8 quantization (per-row min/max scale) applied BEFORE encoding
   - 4x4 block reshaping for spatial correlation if embedding dim allows
   - **Store compressed file on disk. DO NOT decompress to memory.**
   - Optionally also pre-split into individual frame files for faster random access

## Phase 4: Frame Cache with Multi-Batch Lookahead Prefetching

This is the core runtime system. The cold compressed data stays on disk. Only the bounded cache holds decoded frames. Background threads decode ahead.

### 4.1: Frame Predictors (Multi-Batch Lookahead)

All predictors return `{frame_id: priority}` where priority encodes confidence AND temporal distance. Higher priority = prefetch first, evict last.

**Strategy A — Last-Batch Heuristic (baseline):**
```python
class LastBatchPredictor:
    """T+1 gets T's frames (priority=1.0), T+2 gets union (priority=0.5)."""
    def __init__(self):
        self.last_frames = set()
        self.second_last_frames = set()
    
    def predict(self, current_frames, budget_frames=30):
        predictions = {}
        for fid in self.last_frames:
            predictions[fid] = max(predictions.get(fid, 0), 1.0)
        for fid in self.second_last_frames:
            predictions[fid] = max(predictions.get(fid, 0), 0.5)
        self.second_last_frames = self.last_frames.copy()
        self.last_frames = set(current_frames)
        return predictions
```

**Strategy B — EMA with depth:**
```python
class EMAPredictor:
    """Exponential decay scores naturally support multi-batch lookahead.
    High-scoring frames are likely needed at T+1, T+2, T+3..."""
    
    def __init__(self, num_frames, alpha=0.3):
        self.scores = np.zeros(num_frames)
        self.alpha = alpha
    
    def predict(self, current_frames, budget_frames=30):
        self.scores *= (1 - self.alpha)
        for fid in current_frames:
            self.scores[fid] += self.alpha
        top_ids = np.argsort(self.scores)[-budget_frames:]
        return {int(fid): float(self.scores[fid]) 
                for fid in top_ids if self.scores[fid] > 0.01}
```

**Strategy C — Markov with multi-step lookahead:**
```python
class MarkovPredictor:
    """T+1: direct transitions (priority=1.0).
    T+2: 2-step transitions (priority=0.5).
    T+3+: EMA fallback (priority=0.25)."""
    
    def __init__(self, num_frames, lookahead_depth=3):
        self.transition = defaultdict(Counter)
        self.last_frames = set()
        self.lookahead_depth = lookahead_depth
        self.ema_scores = np.zeros(num_frames)
        self.ema_alpha = 0.2
        self.warmup_batches = 0
    
    def observe(self, current_frames):
        current_set = set(current_frames)
        for prev in self.last_frames:
            for curr in current_set:
                self.transition[prev][curr] += 1
        self.last_frames = current_set
        self.warmup_batches += 1
        self.ema_scores *= (1 - self.ema_alpha)
        for fid in current_frames:
            self.ema_scores[fid] += self.ema_alpha
    
    def predict(self, current_frames, budget_frames=50):
        self.observe(current_frames)
        predictions = {}
        
        # T+1: direct Markov (highest priority)
        step1 = Counter()
        for fid in current_frames:
            for nxt, cnt in self.transition[fid].most_common(budget_frames):
                step1[nxt] += cnt
        if step1:
            mx = max(step1.values())
            for fid, s in step1.most_common(budget_frames):
                predictions[fid] = 1.0 * (s / mx)
        
        # T+2: 2-step transitions (medium priority)
        if self.lookahead_depth >= 2:
            step1_top = [f for f, _ in step1.most_common(20)]
            step2 = Counter()
            for fid in step1_top:
                for nxt, cnt in self.transition[fid].most_common(budget_frames):
                    step2[nxt] += cnt
            if step2:
                mx2 = max(step2.values())
                for fid, s in step2.most_common(budget_frames):
                    if fid not in predictions:
                        predictions[fid] = 0.5 * (s / mx2)
        
        # T+3+: EMA fallback (lowest priority)
        if self.lookahead_depth >= 3:
            for fid in np.argsort(self.ema_scores)[-budget_frames:]:
                fid = int(fid)
                if fid not in predictions and self.ema_scores[fid] > 0.01:
                    predictions[fid] = 0.25 * self.ema_scores[fid]
        
        return predictions
```

**Strategy D — Oracle Lookahead (offline inference only):**

Pre-scans the ENTIRE test set before inference to know exactly which frames each future batch needs. This is the upper bound — no predictor can beat it.

```python
class OracleLookaheadPredictor:
    """Pre-scan data loader, build per-batch frame schedule.
    Memory cost: ~4 bytes/frame-id/batch ≈ 1.2 MB for 10K batches × 30 frames."""
    
    def __init__(self, data_loader, hot_masks, orig_to_cold_reordered,
                 frame_size, lookahead_depth=5):
        self.lookahead_depth = lookahead_depth
        self.batch_frames = []  # batch_frames[i] = set of frame_ids
        self.current_batch_idx = 0
        
        print(f"Oracle: pre-scanning data loader...")
        for batch_idx, (X, lS_o, lS_i, T) in enumerate(data_loader):
            batch_set = set()
            for table_idx in range(len(lS_i)):
                indices = lS_i[table_idx]
                cold_mask = ~hot_masks[table_idx][indices]
                if cold_mask.any():
                    cold_mapped = orig_to_cold_reordered[table_idx][indices[cold_mask]]
                    fids = (cold_mapped // frame_size).unique().tolist()
                    batch_set.update(fids)
            self.batch_frames.append(batch_set)
        
        avg_frames = np.mean([len(f) for f in self.batch_frames])
        print(f"Oracle: {len(self.batch_frames)} batches, avg {avg_frames:.1f} frames/batch")
    
    def predict(self, current_frames, budget_frames=50):
        predictions = {}
        idx = self.current_batch_idx
        for d in range(1, self.lookahead_depth + 1):
            future = idx + d
            if future >= len(self.batch_frames):
                break
            priority = 1.0 - (d - 1) * (0.8 / self.lookahead_depth)
            for fid in self.batch_frames[future]:
                predictions[fid] = max(predictions.get(fid, 0), priority)
        self.current_batch_idx += 1
        return predictions
    
    def get_optimal_cache_size(self, target_hit_rate=0.95):
        """Simulate cache at each size, report minimum for target hit rate."""
        from collections import OrderedDict
        for cache_size in [25, 50, 75, 100, 150, 200, 300, 500]:
            cache = OrderedDict()
            hits = total = 0
            for batch_idx, frame_set in enumerate(self.batch_frames):
                # Prefetch future frames
                for d in range(1, self.lookahead_depth + 1):
                    future = batch_idx + d
                    if future < len(self.batch_frames):
                        for fid in self.batch_frames[future]:
                            cache[fid] = True
                            cache.move_to_end(fid)
                            while len(cache) > cache_size:
                                cache.popitem(last=False)
                # Check hits
                for fid in frame_set:
                    total += 1
                    if fid in cache:
                        hits += 1
                        cache.move_to_end(fid)
                    else:
                        cache[fid] = True
                        while len(cache) > cache_size:
                            cache.popitem(last=False)
            rate = hits / total if total else 0
            print(f"  Cache {cache_size:>4d}: hit={rate:.3f} {'✓' if rate >= target_hit_rate else '✗'}")
            if rate >= target_hit_rate:
                return cache_size
        return 500
```

### 4.2: Prefetch Frame Cache Engine

```python
import threading, time, psutil
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict

class PrefetchFrameCache:
    """Bounded frame cache with async multi-batch prefetching.
    
    MEMORY INVARIANT: At all times, len(self.cache) <= self.cache_capacity.
    The cache holds ONLY decoded frame tensors. Compressed data stays on disk.
    
    Architecture:
    - Main thread: embedding lookups from cached frames
    - Background workers: decode predicted frames via PyAV CPU
    - Priority-aware eviction: demand-hit > near-future > far-future > speculative
    """
    
    def __init__(self, compressed_path, frame_size, emb_dim, num_cold_rows,
                 cache_capacity=100, predictor_type='ema',
                 num_prefetch_workers=2, quantize_params=None):
        self.compressed_path = compressed_path  # path to .mp4 on DISK, not in memory
        self.frame_size = frame_size
        self.emb_dim = emb_dim
        self.num_cold_rows = num_cold_rows
        self.num_frames = (num_cold_rows + frame_size - 1) // frame_size
        self.quantize_params = quantize_params  # per-row min/max for dequantization
        
        # ===== PyAV CONTAINER (opened once, seeked per frame) =====
        # Each prefetch worker needs its own container handle (PyAV is not thread-safe)
        # Main thread gets one, workers create their own in _init_worker_container()
        import av
        self.av = av
        self._main_container = av.open(compressed_path)
        self._main_stream = self._main_container.streams.video[0]
        # Pre-decode all frames into an index for fast seeking
        # For ALL-INTRA (keyint=1), every frame is a keyframe
        self._frame_pts = []  # pts[i] = presentation timestamp of frame i
        self._build_frame_index()
        
        # Thread-local containers for prefetch workers
        self._thread_containers = {}  # thread_id -> (container, stream)
        self._container_lock = threading.Lock()
        
        # ===== BOUNDED CACHE (thread-safe) =====
        self.cache = {}              # frame_id -> decoded fp32 tensor (frame_size, emb_dim)
        self.cache_priority = {}     # frame_id -> priority score
        self.cache_capacity = cache_capacity
        self.lock = threading.Lock()
        
        # Memory budget check
        frame_bytes = frame_size * emb_dim * 4  # fp32
        total_cache_bytes = cache_capacity * frame_bytes
        print(f"Cache memory budget: {cache_capacity} frames × {frame_bytes/1024:.1f} KB "
              f"= {total_cache_bytes/1024/1024:.1f} MB")
        assert total_cache_bytes < 500 * 1024 * 1024, \
            f"Cache too large: {total_cache_bytes/1024/1024:.0f} MB > 500 MB limit"
        
        # ===== PREFETCH INFRASTRUCTURE =====
        self.executor = ThreadPoolExecutor(max_workers=num_prefetch_workers)
        self.pending = set()
        self.prefetch_lock = threading.Lock()
        
        # ===== PREDICTOR =====
        if predictor_type == 'last':
            self.predictor = LastBatchPredictor()
        elif predictor_type == 'ema':
            self.predictor = EMAPredictor(num_frames=self.num_frames)
        elif predictor_type == 'markov':
            self.predictor = MarkovPredictor(num_frames=self.num_frames)
        # oracle predictor is set externally after construction
        
        # ===== STATS =====
        self.stats = {
            'cache_hits': 0, 'cache_misses': 0,
            'prefetch_hits': 0, 'demand_decomps': 0,
            'prefetch_submissions': 0, 'prefetch_wastes': 0,
            'total_demand_ms': 0, 'total_prefetch_ms': 0,
        }
    
    def _build_frame_index(self):
        """Pre-scan the video to map frame_id → pts for fast seeking."""
        self._frame_pts = []
        for frame in self._main_container.decode(video=0):
            self._frame_pts.append(frame.pts)
        # Reset to beginning
        self._main_container.seek(0)
        print(f"PyAV: indexed {len(self._frame_pts)} frames")
    
    def _get_thread_container(self):
        """Get or create a per-thread PyAV container (PyAV is not thread-safe)."""
        tid = threading.get_ident()
        with self._container_lock:
            if tid not in self._thread_containers:
                container = self.av.open(self.compressed_path)
                stream = container.streams.video[0]
                self._thread_containers[tid] = (container, stream)
            return self._thread_containers[tid]
    
    def _decode_frame_pyav(self, frame_id, container=None, stream=None):
        """Decode ONE frame using PyAV (in-process, no subprocess overhead).
        
        For ALL-INTRA video (keyint=1), every frame is a keyframe,
        so seeking to any frame is instant with no decode chain.
        
        Returns fp32 tensor of shape (actual_rows, emb_dim)."""
        
        if container is None:
            container = self._main_container
            stream = self._main_stream
        
        # Seek to the target frame's timestamp
        target_pts = self._frame_pts[frame_id]
        container.seek(target_pts, stream=stream)
        
        # Decode the next frame (should be our target)
        frame = None
        for f in container.decode(video=0):
            frame = f
            break
        
        assert frame is not None, f"PyAV: failed to decode frame {frame_id}"
        
        # Convert to numpy array (grayscale Y plane)
        arr = frame.to_ndarray(format='gray')  # shape: (height, width), dtype=uint8
        
        # Reshape from 2D frame back to embedding rows
        # (inverse of whatever reshaping was used during encoding)
        frame_uint8 = arr.reshape(-1, self.emb_dim)[:self.frame_size]
        
        # Dequantize: uint8 → fp32 using stored per-row min/max
        if self.quantize_params is not None:
            row_start = frame_id * self.frame_size
            row_end = min(row_start + self.frame_size, self.num_cold_rows)
            actual_rows = row_end - row_start
            mins = self.quantize_params['min'][row_start:row_end]
            maxs = self.quantize_params['max'][row_start:row_end]
            frame_fp32 = mins + (frame_uint8[:actual_rows].astype(np.float32) / 255.0) * (maxs - mins)
        else:
            frame_fp32 = frame_uint8.astype(np.float32)
        
        return torch.from_numpy(frame_fp32)
    
    def _prefetch_worker(self, frame_id, priority):
        """Background thread: decode via PyAV and cache with priority.
        Each worker thread uses its own PyAV container handle."""
        container, stream = self._get_thread_container()
        
        t0 = time.time()
        frame_data = self._decode_frame_pyav(frame_id, container, stream)
        elapsed = (time.time() - t0) * 1000
        
        with self.lock:
            if frame_id not in self.cache:
                self.cache[frame_id] = frame_data
                self.cache_priority[frame_id] = priority
                self._evict_if_needed()
            self.stats['total_prefetch_ms'] += elapsed
        
        with self.prefetch_lock:
            self.pending.discard(frame_id)
    
    def _evict_if_needed(self):
        """Priority-aware eviction. Must hold self.lock.
        Evict lowest-priority frames first."""
        while len(self.cache) > self.cache_capacity:
            # Find lowest priority
            evict_id = min(self.cache_priority, key=self.cache_priority.get)
            del self.cache[evict_id]
            del self.cache_priority[evict_id]
            self.stats['prefetch_wastes'] += 1
        # INVARIANT CHECK
        assert len(self.cache) <= self.cache_capacity, \
            f"CACHE OVERFLOW: {len(self.cache)} > {self.cache_capacity}"
    
    def launch_prefetch(self, current_batch_frames):
        """Called AFTER batch T's embedding lookup.
        Predicts frames for T+1, T+2, ... and starts background decode.
        Runs during MLP compute phase (free time).
        
        Memory control: fills up to 70% of cache capacity, reserving
        30% for demand misses. This prevents deep speculation from 
        thrashing frames that are still needed."""
        
        predictions = self.predictor.predict(
            current_batch_frames,
            budget_frames=self.cache_capacity
        )
        
        with self.lock:
            cached = set(self.cache.keys())
            cur_size = len(self.cache)
        with self.prefetch_lock:
            in_flight = self.pending.copy()
        
        candidates = {fid: pri for fid, pri in predictions.items()
                      if fid not in cached and fid not in in_flight}
        sorted_cands = sorted(candidates.items(), key=lambda x: -x[1])
        
        # Budget: don't exceed 70% of cache to leave room for demand loads
        budget = max(0, int(self.cache_capacity * 0.7) - cur_size - len(in_flight))
        
        submitted = 0
        for fid, pri in sorted_cands:
            if submitted >= budget:
                break
            with self.prefetch_lock:
                self.pending.add(fid)
            self.executor.submit(self._prefetch_worker, fid, pri)
            submitted += 1
        self.stats['prefetch_submissions'] += submitted
    
    def lookup(self, cold_indices_reordered):
        """Main thread: lookup cold embeddings from bounded frame cache.
        Missing frames are decoded ON DEMAND (blocking).
        
        Returns (result_tensor, set_of_frame_ids_used)."""
        
        # Sort indices for frame locality (Rule 4)
        sorted_order = torch.argsort(cold_indices_reordered)
        sorted_indices = cold_indices_reordered[sorted_order]
        
        frame_ids = sorted_indices // self.frame_size
        unique_frames = torch.unique(frame_ids).tolist()
        
        for fid in unique_frames:
            with self.lock:
                if fid in self.cache:
                    self.cache_priority[fid] = 2.0  # boost: confirmed needed
                    self.stats['cache_hits'] += 1
                    continue
            
            # Wait briefly for in-flight prefetch
            for _ in range(10):
                with self.prefetch_lock:
                    if fid not in self.pending:
                        break
                time.sleep(0.0005)
            
            with self.lock:
                if fid in self.cache:
                    self.stats['prefetch_hits'] += 1
                    self.cache_priority[fid] = 2.0
                    continue
            
            # DEMAND DECODE — critical path, this is what we minimize
            t0 = time.time()
            frame_data = self._decode_frame_pyav(fid)  # uses main thread's container
            elapsed = (time.time() - t0) * 1000
            
            with self.lock:
                self.cache[fid] = frame_data
                self.cache_priority[fid] = 2.0
                self._evict_if_needed()
                self.stats['demand_decomps'] += 1
                self.stats['total_demand_ms'] += elapsed
            self.stats['cache_misses'] += 1
        
        # Vectorized gather from cached frames (Rule 5)
        results = torch.zeros(len(sorted_indices), self.emb_dim)
        with self.lock:
            for fid in unique_frames:
                mask = (frame_ids == fid)
                offsets = sorted_indices[mask] % self.frame_size
                frame_data = self.cache[fid]
                actual_rows = frame_data.shape[0]
                # Clamp offsets for last frame which may be smaller
                safe_offsets = torch.clamp(offsets, max=actual_rows - 1)
                results[mask] = frame_data[safe_offsets]
        
        # Unsort back to original order
        final = torch.zeros_like(results)
        final[sorted_order] = results
        
        # MEMORY INVARIANT CHECK (every batch)
        with self.lock:
            assert len(self.cache) <= self.cache_capacity, \
                f"LEAK: cache has {len(self.cache)} frames, limit is {self.cache_capacity}"
        
        return final, set(unique_frames)
    
    def report_stats(self):
        total = self.stats['cache_hits'] + self.stats['cache_misses']
        hit_rate = self.stats['cache_hits'] / total if total else 0
        print(f"Cache: {hit_rate:.1%} hit rate | "
              f"Demand decomps: {self.stats['demand_decomps']} | "
              f"Prefetch hits: {self.stats['prefetch_hits']} | "
              f"Demand time: {self.stats['total_demand_ms']:.0f}ms | "
              f"Cache size: {len(self.cache)}/{self.cache_capacity}")
```

### 4.3: Inference Loop Integration

```python
# === SETUP ===
# Oracle: pre-scan the data loader (optional, ~30s)
if predictor_type == 'oracle':
    predictor = OracleLookaheadPredictor(
        data_loader=test_loader, hot_masks=is_hot,
        orig_to_cold_reordered=orig_to_cold_reordered,
        frame_size=FRAME_SIZE, lookahead_depth=LOOKAHEAD_DEPTH
    )
    optimal_cache = predictor.get_optimal_cache_size(0.95)
    print(f"Oracle: need cache={optimal_cache} for 95% hit rate at depth={LOOKAHEAD_DEPTH}")

cache = PrefetchFrameCache(
    compressed_path='results/cold_compressed.mp4',  # ON DISK, NOT IN MEMORY
    frame_size=FRAME_SIZE, emb_dim=16,
    num_cold_rows=NUM_COLD, cache_capacity=CACHE_CAPACITY,
    predictor_type=predictor_type, num_prefetch_workers=2,
    quantize_params=quant_params
)
if predictor_type == 'oracle':
    cache.predictor = predictor

# === INFERENCE LOOP ===
for batch_idx, (X, lS_o, lS_i, T) in enumerate(test_loader):
    
    # --- EMBEDDING PHASE ---
    embeddings = []
    batch_cold_frames = set()
    for table_idx in range(26):
        indices = lS_i[table_idx]
        hot_mask = is_hot[table_idx][indices]
        
        # Hot lookup (instant, cache-resident)
        hot_result = hot_emb_weight[table_idx][orig_to_hot[table_idx][indices[hot_mask]]]
        
        # Cold lookup (bounded frame cache + on-demand decode)
        if (~hot_mask).any():
            cold_mapped = orig_to_cold_reordered[table_idx][indices[~hot_mask]]
            cold_result, frames_used = cache.lookup(cold_mapped)
            batch_cold_frames.update(frames_used)
        
        # Merge and pool
        combined = merge_and_pool(hot_result, cold_result, hot_mask, offsets)
        embeddings.append(combined)
    
    # --- COMPUTE PHASE (MLP) ---
    # Launch prefetch BEFORE compute — background threads decode during MLP
    cache.launch_prefetch(batch_cold_frames)
    
    x = bot_mlp(dense_x)
    z = interact_features(x, embeddings)
    p = top_mlp(z)
    # By the time next batch's lookup starts, prefetched frames are in cache
    
    # --- PERIODIC MEMORY CHECK ---
    if batch_idx % 100 == 0:
        rss = psutil.Process().memory_info().rss / 1024 / 1024
        cache.report_stats()
        print(f"Batch {batch_idx}: RSS={rss:.0f} MB")
```

### 4.4: Warmup for Markov Predictor

```python
if predictor_type == 'markov':
    print("Warming up Markov predictor...")
    for batch_idx, (X, lS_o, lS_i, T) in enumerate(test_loader):
        if batch_idx >= 500:
            break
        cold_frames = get_cold_frames_for_batch(lS_i)
        cache.predictor.observe(cold_frames)
    print(f"Learned transitions from {cache.predictor.warmup_batches} batches")
```

---

## Phase 5: Experiments

Run these IN ORDER. Clear caches between ALL runs: `sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'`. Run each config 3 times, report mean ± std.

### Experiment A: Baselines
```
1. Baseline DLRM inference (no compression, all embeddings fp32): AUC, time, embedding time
2. Full-decompression codec (decompress all → inference, FOR COMPARISON ONLY): AUC, time
   ↳ This violates Rule 2 but establishes the "codec overhead = 0" reference point
3. Measure per-batch timing: embedding_lookup_ms, bot_mlp_ms, interact_ms, top_mlp_ms
```

### Experiment B: Hot/cold split only
```
1. Hot/cold split, cold as plain nn.Embedding (no codec)
2. Measure: classification overhead, hot/cold lookup times
3. AUC MUST match baseline exactly
4. Sweep coverage: 70%, 80%, 90%, 95% → report hot buffer size at each
```

### Experiment C: Hot/cold + UNORDERED cold codec (no prefetch)
```
1. Cold region compressed as H.265 multi-frame, original row order
2. On-demand decode with bounded LRU cache, NO prefetching
3. Sweep cache sizes: 10, 25, 50, 100, 200 frames
4. Per-batch: frames needed, cache hit rate, demand decomps, decode time
5. Report AUC, total time, peak RSS memory
```

### Experiment D: Hot/cold + REORDERED cold codec (no prefetch)
```
1. Cold region reordered by co-access partitioning, then compressed
2. Same LRU cache, NO prefetching
3. KEY METRIC: frames/batch before vs after reordering (should drop significantly)
4. Test greedy clustering AND spectral ordering
5. Test FRAME_SIZE: 1024, 2048, 4096, 8192
6. Compare cache hit rates vs Experiment C at same cache sizes
```

### Experiment E: REORDERED + PREFETCHING (full system)
```
This is the main event.

1. Measure prefetch budget: MLP compute time vs per-frame decode time

2. Compare all 4 predictors at depth=1, 3, 5:
   - Last-batch, EMA, Markov, Oracle
   - Metrics per predictor: prediction accuracy, coverage, waste, demand decomps, 
     cache hit rate, inference time

3. Sweep lookahead depth (THE key sweep):
   For EACH predictor: depth = 1, 2, 3, 5, 8, 10
   Report per depth:
   - Demand decomps per batch (should decrease)
   - Cache hit rate
   - Prefetch waste rate (increases with depth)
   - Required cache size for this depth
   - Cache memory (MB) = cache_capacity × frame_size × emb_dim × 4
   - Inference time
   
   Find sweet spot: depth where demand decomps ≈ 0 without excessive memory.

4. For oracle: run get_optimal_cache_size() at each depth for 95% hit rate.

5. Sweep cache_capacity: 25, 50, 100, 200, 500
   Plot: cache memory (MB) vs demand decomps/batch
   The curve should show diminishing returns — find the knee.

6. Per-batch latency breakdown:
   hot_lookup_ms, cold_cache_hit_ms, cold_demand_decode_ms, mlp_ms, prefetch_overlap_ms
```

### Experiment F: Ablation — incremental benefit of each layer
```
| Configuration                          | Depth | Demand/Batch | Hit % | Cache MB | Time  | Speedup |
|----------------------------------------|-------|-------------|-------|----------|-------|---------|
| C. Unordered + no prefetch             | -     | ?           | ?%    | ?        | ?s    | 1.0x    |
| D. Reordered + no prefetch             | -     | ?           | ?%    | ?        | ?s    | ?x      |
| E1. Reordered + last-batch (depth=1)   | 1     | ?           | ?%    | ?        | ?s    | ?x      |
| E2. Reordered + EMA (depth=3)          | 3     | ?           | ?%    | ?        | ?s    | ?x      |
| E3. Reordered + Markov (depth=1)       | 1     | ?           | ?%    | ?        | ?s    | ?x      |
| E3. Reordered + Markov (depth=3)       | 3     | ?           | ?%    | ?        | ?s    | ?x      |
| E3. Reordered + Markov (depth=5)       | 5     | ?           | ?%    | ?        | ?s    | ?x      |
| E4. Reordered + Oracle (depth=1)       | 1     | ?           | ?%    | ?        | ?s    | ?x      |
| E4. Reordered + Oracle (depth=3)       | 3     | ?           | ?%    | ?        | ?s    | ?x      |
| E4. Reordered + Oracle (depth=5)       | 5     | ?           | ?%    | ?        | ?s    | ?x      |
| E4. Reordered + Oracle (depth=10)      | 10    | ?           | ?%    | ?        | ?s    | ?x      |
| A. Baseline (uncompressed, full table) | -     | 0           | N/A   | 2061     | ?s    | ?x      |

KEY: Find the depth where Markov ≈ Oracle. That's the practical ceiling.
```

### Experiment G: Per-table ablation
```
For each of 26 tables:
- Table size, hot coverage at 80%, cold co-access density
- Frames/batch before and after reordering
- Oracle: min cache for 95% hit at depth=5
- Markov: demand decomps at depth=3, cache=100
- Batch-to-batch frame overlap (predictability metric)
```

---

## Output Files

```
results/
├── profiling/
│   ├── access_frequency_cdf.png
│   ├── cooccurrence_stats.md
│   ├── temporal_frame_overlap.md
│   └── prefetch_budget.md
├── experiment_results.md                  — main comparison table
├── frames_per_batch_comparison.png        — histogram C vs D
├── cache_hit_rate_vs_size.png            — curves C vs D vs E
├── prefetch_accuracy_comparison.png       — 4 predictors compared
├── latency_breakdown_stacked.png         — stacked bars per config
├── demand_decomp_over_time.png           — per-batch over time
├── depth_sweep_demand_decomp.png         — demand vs depth, all predictors
├── depth_sweep_cache_memory.png          — cache MB vs depth
├── oracle_cache_size_analysis.md         — min cache per depth for 95%
├── memory_trace.png                      — RSS over time (must be flat/bounded!)
└── src/                                  — all code
```

---

## What NOT To Do

- ⛔ Do NOT decode all cold frames into memory at startup (Rule 2)
- ⛔ Do NOT create a full-size tensor for cold embeddings (`nn.Embedding(num_cold, dim)`)
- ⛔ Do NOT use unbounded caches or dicts that grow without eviction (Rule 3)
- ⛔ Do NOT use Python for-loops for row extraction/reshaping in the hot path (Rule 5)
- ⛔ Do NOT skip metrics — every experiment needs ALL Rule 6 metrics
- ⛔ Do NOT modify the base DLRM model architecture
- ⛔ Do NOT use lossy quantization below INT8
- ⛔ Do NOT use GPU/NVDEC — this is the CPU version (Rule 7)
- ⛔ Do NOT use `subprocess.run(['ffmpeg', ...])` for decoding — use PyAV in-process (Rule 7)
- ⛔ Do NOT share a single PyAV container across threads — each thread needs its own handle
- ⛔ Do NOT load the compressed .mp4 file into a Python bytes object in memory — open it via PyAV which reads from disk on demand
- ⛔ Do NOT store `cold_weights` tensor anywhere after encoding — the whole point is it doesn't exist in memory

## Critical Rules

- Do NOT ask for permission. Execute immediately.
- Launch long-running processes in tmux.
- If something fails, debug and retry. Log errors to `logs/`.
- Write clean code. This goes into a paper.
- **PRIMARY METRIC: demand decompressions per batch** — the number of frames decoded on the critical path. If reordering + prefetch drives this to ≈0, the codec adds no runtime overhead.
- **SECONDARY METRIC: frames per batch before vs after reordering** — determines how much prefetching needs to cover.
- **MEMORY INVARIANT: RSS must be stable/flat over time.** If RSS grows linearly with batches, there is a memory leak. Plot RSS over time and verify flatness.
- Start with the LARGEST table only for prototyping. Once working, generalize to all 26.
- Run Oracle first at every depth to establish the upper bound. Then compare Markov/EMA.
- Oracle pre-scan: ~30s, ~1 MB memory. Worth it for ground truth.
- Watch cache memory: `cache_capacity × frame_size × emb_dim × 4 bytes`. At 200 frames × 4096 × 16 × 4 = 50 MB. Good. At 500 frames = 125 MB. Still fine vs 2 GB. Only worry if >500 MB.
- Pin prefetch threads to different cores than main inference thread if possible (`os.sched_setaffinity`).
- For Markov transition matrix: use sparse representation (defaultdict(Counter)), keep only top-K transitions.

GO.