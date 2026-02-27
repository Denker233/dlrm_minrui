# Figure Summaries for Presentation

Dataset: Kaggle/Criteo, 26 embedding tables, emb_dim=16, CPU (40 threads)

---

## Fig 0 — Overview Composite (single-slide summary)

This is the full picture of our embedding compression system. Top-left: row reordering alone gives 1.29x speedup from CPU cache locality. Top-right: our H.265 codec pipeline achieves 2.3x speedup with 10x memory reduction. Bottom-left: reordering hot embeddings doesn't help because they already fit in L3 cache. Bottom-right: Rec-AD's graph-based community detection gives identical results to our simpler batch-affinity method because most cold indices only appear in a single batch — there's no co-occurrence structure to exploit.

---

## Fig 1 — Cold Row Reordering Speedup

We compared four row orderings on uncompressed fp32 embedding tables. The baseline uses the original row order from training. Frequency-sort places the most-accessed rows first, giving 1.17x speedup. Our batch-affinity method sorts rows by their first access batch, grouping co-accessed rows together — this gives 1.29x speedup purely from CPU cache locality. Rec-AD's Louvain community detection gives the same 1.29x because on Kaggle/Criteo, cold rows are too sparse for graph structure to matter. All methods preserve identical AUC since we're only rearranging rows, not modifying values.

---

## Fig 2 — End-to-End Codec Results

Left panel: our H.265 codec pipeline reduces inference time from 8.8s to 3.8s — a 2.3x speedup. This is counterintuitive because we're adding decompression overhead, but the memory reduction improves cache behavior so much that it more than compensates. Right panel: embedding memory drops from 2061MB to ~200MB, a 10x reduction. The compressed cold embeddings are stored as H.265 video frames on disk and decoded on-demand. AUC loss is negligible (0.802497 vs 0.802489).

---

## Fig 3 — Latency Breakdown

In the baseline, embedding lookup takes 1.5ms (23% of each batch). After compression, the hot/cold split means we only do fp32 lookups on 22MB of hot rows instead of 2GB — embedding time drops to near zero. The MLP forward pass (5.1ms) is unchanged and now dominates at 98% of latency. This explains why further optimizing embedding lookups (like hot reordering) has diminishing returns — the bottleneck has shifted to the MLP.

---

## Fig 4 — Hot Embedding Reordering (Negative Result)

We tested whether reordering hot embedding rows helps within the codec pipeline. It doesn't — all three methods perform within noise of each other (3.86s vs 3.97s vs 4.05s). The reason is that the hot embedding tensor is only 22MB across all 8 large tables, which fits entirely in L3 cache. Row layout within a cache-resident tensor has minimal impact. Additionally, reordering breaks compatibility with our bitmap-rank data structure, which uses hardware popcount for O(1) index lookups, forcing a fallback to a slower mapping tensor.

---

## Fig 5 — P99 Tail Latency

Left: row reordering reduces P99 tail latency by 34% (7.5ms to 4.9ms) because the worst-case cache miss scenario improves dramatically when related rows are adjacent. Right: the full codec pipeline reduces P99 from 7.9ms to 3.5ms at 4K resolution. Tail latency matters for serving SLA compliance — our system improves both mean and worst-case latency.

---

## Fig 6 — Memory Breakdown

The baseline stores all embeddings as fp32: 2061MB. Our codec pipeline breaks this into four components: hot embeddings (22MB fp32 for the 4.3% most-accessed rows), index mapping (129-135MB for hot/cold lookup), compressed cold frames (48-58MB of H.265 encoded data on disk), and an LRU frame cache (34-63MB for recently decoded frames). The mapping tensor is the largest runtime component — the bitmap-rank optimization reduces this by replacing per-row int64 mappings with a 1-bit-per-row bitmap plus popcount.

---

## Fig 7 — Why Rec-AD = Batch-Affinity

Left: we measured how many batches each cold index appears in. For 5 of the 8 large tables (red bars), the average is exactly 1.0 — each cold row is accessed in only one batch. This means there are zero co-occurrence edges for Rec-AD's graph to work with, so it falls back to batch-affinity sorting. Only Table 9 (8.2 batches/idx) has rich enough co-occurrence for Louvain to find meaningful communities, but it's small (41K rows) and doesn't affect overall timing. Right: the largest tables by cold row count are exactly the ones with no co-occurrence. On datasets with denser access patterns, Rec-AD could potentially outperform batch-affinity.
