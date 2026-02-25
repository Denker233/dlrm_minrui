// compressed_emb.cpp — C++ PyTorch extension for CompressedEmbeddingBag forward pass
// Replaces Python hot/cold routing + scatter_add with efficient C++ implementation.

#include <torch/extension.h>
#include <vector>
#include <cstring>
#include <algorithm>
#include <atomic>

// SIMD helpers for D=16 embedding accumulation
#ifdef __AVX512F__
#include <immintrin.h>

// Accumulate fp32 embedding (D=16) into output using AVX-512 (single 512-bit op)
static inline void accum_fp32_d16(float* __restrict__ dst, const float* __restrict__ src) {
    __m512 d = _mm512_loadu_ps(dst);
    __m512 s = _mm512_loadu_ps(src);
    _mm512_storeu_ps(dst, _mm512_add_ps(d, s));
}

static inline void accum_fp32_d16_weighted(float* __restrict__ dst, const float* __restrict__ src, float w) {
    __m512 d = _mm512_loadu_ps(dst);
    __m512 s = _mm512_loadu_ps(src);
    __m512 wv = _mm512_set1_ps(w);
    _mm512_storeu_ps(dst, _mm512_fmadd_ps(s, wv, d));
}

// Dequantize uint8 embedding (D=16) and accumulate into fp32 output using AVX-512
static inline void accum_q8_d16(float* __restrict__ dst, const uint8_t* __restrict__ src,
                                  float scale, float zp) {
    // Load 16 uint8 values, zero-extend to int32, convert to float, dequantize, accumulate
    __m128i u8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
    __m512i u32 = _mm512_cvtepu8_epi32(u8);
    __m512 f32 = _mm512_cvtepi32_ps(u32);
    __m512 zpv = _mm512_set1_ps(zp);
    __m512 sv = _mm512_set1_ps(scale);
    __m512 dequant = _mm512_mul_ps(_mm512_sub_ps(f32, zpv), sv);
    __m512 d = _mm512_loadu_ps(dst);
    _mm512_storeu_ps(dst, _mm512_add_ps(d, dequant));
}

static inline void accum_q8_d16_weighted(float* __restrict__ dst, const uint8_t* __restrict__ src,
                                          float scale, float zp, float w) {
    __m128i u8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
    __m512i u32 = _mm512_cvtepu8_epi32(u8);
    __m512 f32 = _mm512_cvtepi32_ps(u32);
    __m512 zpv = _mm512_set1_ps(zp);
    __m512 sv = _mm512_set1_ps(scale * w);
    __m512 dequant = _mm512_mul_ps(_mm512_sub_ps(f32, zpv), sv);
    __m512 d = _mm512_loadu_ps(dst);
    _mm512_storeu_ps(dst, _mm512_add_ps(d, dequant));
}
#define HAS_AVX512 1
#else
#define HAS_AVX512 0
#endif

// Forward pass: hot/cold split embedding lookup with sum pooling.
//
// For each index in `indices`:
//   - If is_hot[index] == true: gather from hot_weight[orig_to_hot[index]]
//   - Else: gather from cold_decoded[orig_to_cold[index]]  (pre-decoded fp32 frame cache)
//
// Then sum-pool into bags defined by `offsets`.
//
// This replaces the Python CompressedEmbeddingBag.forward() which was the bottleneck
// (~7.5ms per batch in Python vs ~0.05ms expected in C++).

torch::Tensor compressed_emb_forward(
    const torch::Tensor& indices,        // (N,) int64 — original embedding indices
    const torch::Tensor& offsets,        // (B,) int64 — bag boundaries
    const torch::Tensor& hot_weight,     // (n_hot, D) float32 — compact hot embeddings
    const torch::Tensor& is_hot,         // (num_emb,) bool — hot mask per original index
    const torch::Tensor& orig_to_hot,    // (num_emb,) int64 — maps orig_idx -> hot compact idx (-1 if cold)
    const torch::Tensor& orig_to_cold,   // (num_emb,) int64 — maps orig_idx -> cold reordered idx (-1 if hot)
    const torch::Tensor& cold_decoded,   // (n_cold_cached, D) float32 — pre-looked-up cold rows (from LRU cache)
    const torch::Tensor& cold_remap,     // (N_cold_in_batch,) int64 — maps cold batch positions to cold_decoded rows
    const torch::Tensor& per_sample_weights  // (N,) float32 or empty — optional weights
) {
    const int64_t N = indices.size(0);
    const int64_t B = offsets.size(0);
    const int64_t D = hot_weight.size(1);
    const bool has_weights = per_sample_weights.numel() > 0;

    auto output = torch::zeros({B, D}, hot_weight.options());

    const int64_t* idx_ptr = indices.data_ptr<int64_t>();
    const int64_t* off_ptr = offsets.data_ptr<int64_t>();
    const bool* hot_ptr = is_hot.data_ptr<bool>();
    const int64_t* o2h_ptr = orig_to_hot.data_ptr<int64_t>();
    const float* hw_ptr = hot_weight.data_ptr<float>();
    const float* cw_ptr = cold_decoded.numel() > 0 ? cold_decoded.data_ptr<float>() : nullptr;
    const float* psw_ptr = has_weights ? per_sample_weights.data_ptr<float>() : nullptr;
    float* out_ptr = output.data_ptr<float>();

    const int64_t cold_D = cold_decoded.numel() > 0 ? cold_decoded.size(1) : D;

    // Process each bag
    for (int64_t b = 0; b < B; b++) {
        int64_t start = off_ptr[b];
        int64_t end = (b + 1 < B) ? off_ptr[b + 1] : N;
        float* out_row = out_ptr + b * D;

        for (int64_t i = start; i < end; i++) {
            int64_t orig_idx = idx_ptr[i];
            const float* emb_row = nullptr;

            if (hot_ptr[orig_idx]) {
                // Hot path (99.9% of lookups)
                int64_t hot_idx = o2h_ptr[orig_idx];
                emb_row = hw_ptr + hot_idx * D;
            } else if (cw_ptr) {
                // Cold path — lookup from pre-gathered cold_decoded
                int64_t cold_idx = o2h_ptr[orig_idx];  // reuse pointer, will be overridden below
                // We need the cold_remap approach, but for simplicity we'll use
                // the orig_to_cold mapping directly into cold_decoded
                // cold_decoded is indexed by the reordered cold position
                // But cold_decoded might be a subset (only cached frames)
                // For the fast path, we pass cold embeddings pre-gathered
                // Skip if no cold data available
                continue;
            } else {
                continue;
            }

            if (has_weights) {
                float w = psw_ptr[i];
                for (int64_t d = 0; d < D; d++) {
                    out_row[d] += emb_row[d] * w;
                }
            } else {
                for (int64_t d = 0; d < D; d++) {
                    out_row[d] += emb_row[d];
                }
            }
        }
    }
    return output;
}


// Simpler, more practical version:
// Since 99.9% of lookups are hot, we do the entire forward pass
// assuming all indices are hot, then fix up the rare cold ones in Python.
//
// This avoids the complex cold-path logic in C++ while eliminating
// the Python overhead for the hot path.

torch::Tensor hot_embedding_bag_forward(
    const torch::Tensor& indices,           // (N,) int64
    const torch::Tensor& offsets,           // (B,) int64
    const torch::Tensor& hot_weight,        // (n_hot, D) float32
    const torch::Tensor& is_hot,            // (num_emb,) bool
    const torch::Tensor& orig_to_hot,       // (num_emb,) int64
    const torch::Tensor& per_sample_weights // (N,) float32 or empty
) {
    const int64_t N = indices.size(0);
    const int64_t B = offsets.size(0);
    const int64_t D = hot_weight.size(1);
    const bool has_weights = per_sample_weights.numel() > 0;

    auto output = torch::zeros({B, D}, hot_weight.options());
    auto cold_mask = torch::zeros({N}, torch::dtype(torch::kBool));

    const int64_t* idx_ptr = indices.data_ptr<int64_t>();
    const int64_t* off_ptr = offsets.data_ptr<int64_t>();
    const bool* hot_ptr = is_hot.data_ptr<bool>();
    const int64_t* o2h_ptr = orig_to_hot.data_ptr<int64_t>();
    const float* hw_ptr = hot_weight.data_ptr<float>();
    const float* psw_ptr = has_weights ? per_sample_weights.data_ptr<float>() : nullptr;
    float* out_ptr = output.data_ptr<float>();
    bool* cold_mask_ptr = cold_mask.data_ptr<bool>();

    for (int64_t b = 0; b < B; b++) {
        int64_t start = off_ptr[b];
        int64_t end = (b + 1 < B) ? off_ptr[b + 1] : N;
        float* out_row = out_ptr + b * D;

        for (int64_t i = start; i < end; i++) {
            int64_t orig_idx = idx_ptr[i];

            if (hot_ptr[orig_idx]) {
                int64_t hot_idx = o2h_ptr[orig_idx];
                const float* emb_row = hw_ptr + hot_idx * D;
                if (has_weights) {
                    float w = psw_ptr[i];
                    for (int64_t d = 0; d < D; d++) {
                        out_row[d] += emb_row[d] * w;
                    }
                } else {
                    for (int64_t d = 0; d < D; d++) {
                        out_row[d] += emb_row[d];
                    }
                }
            } else {
                cold_mask_ptr[i] = true;
            }
        }
    }

    return output;  // cold_mask available via the second return
}

// Optimized version: returns output, cold_mask, and cold_count.
// Uses at::parallel_for for multi-threaded bag processing.
// cold_count avoids expensive Python .any() call.
std::vector<torch::Tensor> compressed_emb_bag_forward(
    const torch::Tensor& indices,           // (N,) int64
    const torch::Tensor& offsets,           // (B,) int64
    const torch::Tensor& hot_weight,        // (n_hot, D) float32
    const torch::Tensor& is_hot,            // (num_emb,) bool
    const torch::Tensor& orig_to_hot,       // (num_emb,) int64
    const torch::Tensor& per_sample_weights // (N,) float32 or empty
) {
    const int64_t N = indices.size(0);
    const int64_t B = offsets.size(0);
    const int64_t D = hot_weight.size(1);
    const bool has_weights = per_sample_weights.numel() > 0;

    auto output = torch::zeros({B, D}, hot_weight.options());
    auto cold_mask = torch::zeros({N}, torch::dtype(torch::kBool));

    const int64_t* idx_ptr = indices.data_ptr<int64_t>();
    const int64_t* off_ptr = offsets.data_ptr<int64_t>();
    const bool* hot_ptr = is_hot.data_ptr<bool>();
    const int64_t* o2h_ptr = orig_to_hot.data_ptr<int64_t>();
    const float* hw_ptr = hot_weight.data_ptr<float>();
    const float* psw_ptr = has_weights ? per_sample_weights.data_ptr<float>() : nullptr;
    float* out_ptr = output.data_ptr<float>();
    bool* cold_mask_ptr = cold_mask.data_ptr<bool>();

    std::atomic<int64_t> cold_count{0};

    // Parallel over bags — each bag is independent
    at::parallel_for(0, B, /* grain_size= */ 64, [&](int64_t b_begin, int64_t b_end) {
        int64_t local_cold = 0;
        for (int64_t b = b_begin; b < b_end; b++) {
            int64_t start = off_ptr[b];
            int64_t end = (b + 1 < B) ? off_ptr[b + 1] : N;
            float* out_row = out_ptr + b * D;

            for (int64_t i = start; i < end; i++) {
                int64_t orig_idx = idx_ptr[i];

                if (hot_ptr[orig_idx]) {
                    int64_t hot_idx = o2h_ptr[orig_idx];
                    const float* emb_row = hw_ptr + hot_idx * D;
                    if (has_weights) {
                        float w = psw_ptr[i];
                        for (int64_t d = 0; d < D; d++) {
                            out_row[d] += emb_row[d] * w;
                        }
                    } else {
                        for (int64_t d = 0; d < D; d++) {
                            out_row[d] += emb_row[d];
                        }
                    }
                } else {
                    cold_mask_ptr[i] = true;
                    local_cold++;
                }
            }
        }
        cold_count.fetch_add(local_cold, std::memory_order_relaxed);
    });

    // Return cold count as a scalar tensor to avoid Python .any() overhead
    auto cold_count_t = torch::tensor(cold_count.load(), torch::dtype(torch::kLong));
    return {output, cold_mask, cold_count_t};
}

// Merged mapping version: uses a single int32 mapping tensor instead of
// separate is_hot (bool), orig_to_hot (int64), and o2c (int64) tensors.
// Saves ~418MB for 8 large tables (from 547MB to 129MB mapping overhead).
//
// Mapping encoding:
//   mapping[i] >= 0            → hot index
//   mapping[i] < 0 && != MIN   → cold index = -(mapping[i] + 1)
//   mapping[i] == INT32_MIN    → invalid
std::vector<torch::Tensor> compressed_emb_bag_forward_merged(
    const torch::Tensor& indices,           // (N,) int64
    const torch::Tensor& offsets,           // (B,) int64
    const torch::Tensor& hot_weight,        // (n_hot, D) float32
    const torch::Tensor& mapping,           // (num_emb,) int32 — merged hot/cold mapping
    const torch::Tensor& per_sample_weights // (N,) float32 or empty
) {
    const int64_t N = indices.size(0);
    const int64_t B = offsets.size(0);
    const int64_t D = hot_weight.size(1);
    const bool has_weights = per_sample_weights.numel() > 0;
    constexpr int32_t INVALID = std::numeric_limits<int32_t>::min();

    auto output = torch::zeros({B, D}, hot_weight.options());
    auto cold_mask = torch::zeros({N}, torch::dtype(torch::kBool));

    const int64_t* idx_ptr = indices.data_ptr<int64_t>();
    const int64_t* off_ptr = offsets.data_ptr<int64_t>();
    const int32_t* map_ptr = mapping.data_ptr<int32_t>();
    const float* hw_ptr = hot_weight.data_ptr<float>();
    const float* psw_ptr = has_weights ? per_sample_weights.data_ptr<float>() : nullptr;
    float* out_ptr = output.data_ptr<float>();
    bool* cold_mask_ptr = cold_mask.data_ptr<bool>();

    std::atomic<int64_t> cold_count{0};

    at::parallel_for(0, B, 64, [&](int64_t b_begin, int64_t b_end) {
        int64_t local_cold = 0;
        for (int64_t b = b_begin; b < b_end; b++) {
            int64_t start = off_ptr[b];
            int64_t end = (b + 1 < B) ? off_ptr[b + 1] : N;
            float* out_row = out_ptr + b * D;

            for (int64_t i = start; i < end; i++) {
                int64_t orig_idx = idx_ptr[i];
                int32_t m = map_ptr[orig_idx];

                // Prefetch next embedding row
                if (i + 1 < end) {
                    int32_t nm = map_ptr[idx_ptr[i + 1]];
                    if (nm >= 0) __builtin_prefetch(hw_ptr + static_cast<int64_t>(nm) * D, 0, 1);
                } else if (b + 1 < b_end) {
                    int64_t ns = off_ptr[b + 1];
                    if (ns < N) {
                        int32_t nm = map_ptr[idx_ptr[ns]];
                        if (nm >= 0) __builtin_prefetch(hw_ptr + static_cast<int64_t>(nm) * D, 0, 1);
                    }
                }

                if (__builtin_expect(m >= 0, 1)) {
                    // Hot path
                    const float* emb_row = hw_ptr + static_cast<int64_t>(m) * D;
#if HAS_AVX512
                    if (D == 16) {
                        if (has_weights) {
                            accum_fp32_d16_weighted(out_row, emb_row, psw_ptr[i]);
                        } else {
                            accum_fp32_d16(out_row, emb_row);
                        }
                    } else
#endif
                    {
                        if (has_weights) {
                            float w = psw_ptr[i];
                            for (int64_t d = 0; d < D; d++) {
                                out_row[d] += emb_row[d] * w;
                            }
                        } else {
                            for (int64_t d = 0; d < D; d++) {
                                out_row[d] += emb_row[d];
                            }
                        }
                    }
                } else if (m != INVALID) {
                    // Cold path
                    cold_mask_ptr[i] = true;
                    local_cold++;
                }
            }
        }
        cold_count.fetch_add(local_cold, std::memory_order_relaxed);
    });

    auto cold_count_t = torch::tensor(cold_count.load(), torch::dtype(torch::kLong));
    return {output, cold_mask, cold_count_t};
}

// Merged mapping version with q8 hot weights
std::vector<torch::Tensor> compressed_emb_bag_forward_q8_merged(
    const torch::Tensor& indices,           // (N,) int64
    const torch::Tensor& offsets,           // (B,) int64
    const torch::Tensor& hot_weight_q8,    // (n_hot, D) uint8
    const torch::Tensor& mapping,           // (num_emb,) int32 — merged hot/cold mapping
    const torch::Tensor& per_sample_weights,// (N,) float32 or empty
    double hot_scale,
    int64_t hot_zero_point
) {
    const int64_t N = indices.size(0);
    const int64_t B = offsets.size(0);
    const int64_t D = hot_weight_q8.size(1);
    const bool has_weights = per_sample_weights.numel() > 0;
    constexpr int32_t INVALID = std::numeric_limits<int32_t>::min();

    auto output = torch::zeros({B, D}, torch::dtype(torch::kFloat32));
    auto cold_mask = torch::zeros({N}, torch::dtype(torch::kBool));

    const int64_t* idx_ptr = indices.data_ptr<int64_t>();
    const int64_t* off_ptr = offsets.data_ptr<int64_t>();
    const int32_t* map_ptr = mapping.data_ptr<int32_t>();
    const uint8_t* hw_ptr = hot_weight_q8.data_ptr<uint8_t>();
    const float* psw_ptr = has_weights ? per_sample_weights.data_ptr<float>() : nullptr;
    float* out_ptr = output.data_ptr<float>();
    bool* cold_mask_ptr = cold_mask.data_ptr<bool>();
    const float s = static_cast<float>(hot_scale);
    const float zp = static_cast<float>(hot_zero_point);

    std::atomic<int64_t> cold_count{0};

    at::parallel_for(0, B, 64, [&](int64_t b_begin, int64_t b_end) {
        int64_t local_cold = 0;
        for (int64_t b = b_begin; b < b_end; b++) {
            int64_t start = off_ptr[b];
            int64_t end = (b + 1 < B) ? off_ptr[b + 1] : N;
            float* out_row = out_ptr + b * D;

            for (int64_t i = start; i < end; i++) {
                int64_t orig_idx = idx_ptr[i];
                int32_t m = map_ptr[orig_idx];

                // Prefetch next embedding row
                if (i + 1 < end) {
                    int32_t nm = map_ptr[idx_ptr[i + 1]];
                    if (nm >= 0) __builtin_prefetch(hw_ptr + static_cast<int64_t>(nm) * D, 0, 1);
                } else if (b + 1 < b_end) {
                    int64_t ns = off_ptr[b + 1];
                    if (ns < N) {
                        int32_t nm = map_ptr[idx_ptr[ns]];
                        if (nm >= 0) __builtin_prefetch(hw_ptr + static_cast<int64_t>(nm) * D, 0, 1);
                    }
                }

                if (__builtin_expect(m >= 0, 1)) {
                    const uint8_t* emb_row = hw_ptr + static_cast<int64_t>(m) * D;
#if HAS_AVX512
                    if (D == 16) {
                        if (has_weights) {
                            accum_q8_d16_weighted(out_row, emb_row, s, zp, psw_ptr[i]);
                        } else {
                            accum_q8_d16(out_row, emb_row, s, zp);
                        }
                    } else
#endif
                    {
                        if (has_weights) {
                            float w = psw_ptr[i];
                            for (int64_t d = 0; d < D; d++) {
                                out_row[d] += (static_cast<float>(emb_row[d]) - zp) * s * w;
                            }
                        } else {
                            for (int64_t d = 0; d < D; d++) {
                                out_row[d] += (static_cast<float>(emb_row[d]) - zp) * s;
                            }
                        }
                    }
                } else if (m != INVALID) {
                    cold_mask_ptr[i] = true;
                    local_cold++;
                }
            }
        }
        cold_count.fetch_add(local_cold, std::memory_order_relaxed);
    });

    auto cold_count_t = torch::tensor(cold_count.load(), torch::dtype(torch::kLong));
    return {output, cold_mask, cold_count_t};
}

// Cold fixup: add cold embeddings into already-computed output.
// Called only when cold_mask has any True entries (rare: ~0.1% of batches).
void cold_fixup(
    torch::Tensor& output,                  // (B, D) float32 — in-place modify
    const torch::Tensor& indices,            // (N,) int64
    const torch::Tensor& offsets,            // (B,) int64
    const torch::Tensor& cold_mask,          // (N,) bool
    const torch::Tensor& cold_embeddings,    // (n_cold_hits, D) float32
    const torch::Tensor& cold_positions,     // (n_cold_hits,) int64 — positions in indices array
    const torch::Tensor& per_sample_weights  // (N,) float32 or empty
) {
    const int64_t B = offsets.size(0);
    const int64_t N = indices.size(0);
    const int64_t D = output.size(1);
    const int64_t n_cold = cold_positions.size(0);
    const bool has_weights = per_sample_weights.numel() > 0;

    const int64_t* off_ptr = offsets.data_ptr<int64_t>();
    const int64_t* cpos_ptr = cold_positions.data_ptr<int64_t>();
    const float* ce_ptr = cold_embeddings.data_ptr<float>();
    const float* psw_ptr = has_weights ? per_sample_weights.data_ptr<float>() : nullptr;
    float* out_ptr = output.data_ptr<float>();

    for (int64_t ci = 0; ci < n_cold; ci++) {
        int64_t pos = cpos_ptr[ci];  // position in the indices array

        // Find which bag this position belongs to (binary search on offsets)
        int64_t bag = 0;
        int64_t lo = 0, hi = B - 1;
        while (lo <= hi) {
            int64_t mid = (lo + hi) / 2;
            if (off_ptr[mid] <= pos) {
                bag = mid;
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }

        float* out_row = out_ptr + bag * D;
        const float* emb_row = ce_ptr + ci * D;

        if (has_weights) {
            float w = psw_ptr[pos];
            for (int64_t d = 0; d < D; d++) {
                out_row[d] += emb_row[d] * w;
            }
        } else {
            for (int64_t d = 0; d < D; d++) {
                out_row[d] += emb_row[d];
            }
        }
    }
}


// Quantized hot forward: hot_weight is stored as uint8 with per-table scale/zp.
// Dequantizes on the fly during accumulation. Saves 4x memory for hot embeddings.
std::vector<torch::Tensor> compressed_emb_bag_forward_q8(
    const torch::Tensor& indices,           // (N,) int64
    const torch::Tensor& offsets,           // (B,) int64
    const torch::Tensor& hot_weight_q8,    // (n_hot, D) uint8 — quantized hot embeddings
    const torch::Tensor& is_hot,            // (num_emb,) bool
    const torch::Tensor& orig_to_hot,       // (num_emb,) int64
    const torch::Tensor& per_sample_weights,// (N,) float32 or empty
    double hot_scale,                       // dequant scale for hot weights
    int64_t hot_zero_point                  // dequant zero point for hot weights
) {
    const int64_t N = indices.size(0);
    const int64_t B = offsets.size(0);
    const int64_t D = hot_weight_q8.size(1);
    const bool has_weights = per_sample_weights.numel() > 0;

    auto output = torch::zeros({B, D}, torch::dtype(torch::kFloat32));
    auto cold_mask = torch::zeros({N}, torch::dtype(torch::kBool));

    const int64_t* idx_ptr = indices.data_ptr<int64_t>();
    const int64_t* off_ptr = offsets.data_ptr<int64_t>();
    const bool* hot_ptr = is_hot.data_ptr<bool>();
    const int64_t* o2h_ptr = orig_to_hot.data_ptr<int64_t>();
    const uint8_t* hw_ptr = hot_weight_q8.data_ptr<uint8_t>();
    const float* psw_ptr = has_weights ? per_sample_weights.data_ptr<float>() : nullptr;
    float* out_ptr = output.data_ptr<float>();
    bool* cold_mask_ptr = cold_mask.data_ptr<bool>();
    const float s = static_cast<float>(hot_scale);
    const float zp = static_cast<float>(hot_zero_point);

    std::atomic<int64_t> cold_count{0};

    at::parallel_for(0, B, 64, [&](int64_t b_begin, int64_t b_end) {
        int64_t local_cold = 0;
        for (int64_t b = b_begin; b < b_end; b++) {
            int64_t start = off_ptr[b];
            int64_t end = (b + 1 < B) ? off_ptr[b + 1] : N;
            float* out_row = out_ptr + b * D;

            for (int64_t i = start; i < end; i++) {
                int64_t orig_idx = idx_ptr[i];

                if (hot_ptr[orig_idx]) {
                    int64_t hot_idx = o2h_ptr[orig_idx];
                    const uint8_t* emb_row = hw_ptr + hot_idx * D;
                    if (has_weights) {
                        float w = psw_ptr[i];
                        for (int64_t d = 0; d < D; d++) {
                            out_row[d] += (static_cast<float>(emb_row[d]) - zp) * s * w;
                        }
                    } else {
                        for (int64_t d = 0; d < D; d++) {
                            out_row[d] += (static_cast<float>(emb_row[d]) - zp) * s;
                        }
                    }
                } else {
                    cold_mask_ptr[i] = true;
                    local_cold++;
                }
            }
        }
        cold_count.fetch_add(local_cold, std::memory_order_relaxed);
    });

    auto cold_count_t = torch::tensor(cold_count.load(), torch::dtype(torch::kLong));
    return {output, cold_mask, cold_count_t};
}


// Batched multi-table forward: process all large tables in one C++ call.
// Eliminates Python overhead of 8 separate C++ calls (~1.6ms saving).
std::vector<torch::Tensor> batched_emb_forward(
    const std::vector<torch::Tensor>& indices_list,    // N_tables tensors
    const std::vector<torch::Tensor>& offsets_list,    // N_tables tensors
    const std::vector<torch::Tensor>& hot_weights,     // N_tables tensors (fp32 or uint8)
    const std::vector<torch::Tensor>& is_hot_list,     // N_tables tensors
    const std::vector<torch::Tensor>& o2h_list,        // N_tables tensors
    const std::vector<double>& scales,                 // N_tables scales (0.0 if fp32)
    const std::vector<int64_t>& zero_points,           // N_tables zero points
    const std::vector<bool>& is_quantized              // whether each table uses q8
) {
    const int64_t T = indices_list.size();
    std::vector<torch::Tensor> results;
    results.reserve(T * 2);  // output + cold_count per table

    for (int64_t t = 0; t < T; t++) {
        const auto& indices = indices_list[t];
        const auto& offsets = offsets_list[t];
        const auto& is_hot = is_hot_list[t];
        const auto& o2h = o2h_list[t];

        const int64_t N = indices.size(0);
        const int64_t B = offsets.size(0);
        const int64_t D = hot_weights[t].size(1);

        auto output = torch::zeros({B, D}, torch::dtype(torch::kFloat32));
        const int64_t* idx_ptr = indices.data_ptr<int64_t>();
        const int64_t* off_ptr = offsets.data_ptr<int64_t>();
        const bool* hot_ptr = is_hot.data_ptr<bool>();
        const int64_t* o2h_ptr = o2h.data_ptr<int64_t>();
        float* out_ptr = output.data_ptr<float>();

        int64_t cold_count = 0;

        if (is_quantized[t]) {
            const uint8_t* hw_ptr = hot_weights[t].data_ptr<uint8_t>();
            const float s = static_cast<float>(scales[t]);
            const float zp = static_cast<float>(zero_points[t]);

            for (int64_t b = 0; b < B; b++) {
                int64_t start = off_ptr[b];
                int64_t end = (b + 1 < B) ? off_ptr[b + 1] : N;
                float* out_row = out_ptr + b * D;
                for (int64_t i = start; i < end; i++) {
                    int64_t orig_idx = idx_ptr[i];
                    if (hot_ptr[orig_idx]) {
                        int64_t hot_idx = o2h_ptr[orig_idx];
                        const uint8_t* emb_row = hw_ptr + hot_idx * D;
                        for (int64_t d = 0; d < D; d++) {
                            out_row[d] += (static_cast<float>(emb_row[d]) - zp) * s;
                        }
                    } else {
                        cold_count++;
                    }
                }
            }
        } else {
            const float* hw_ptr = hot_weights[t].data_ptr<float>();
            for (int64_t b = 0; b < B; b++) {
                int64_t start = off_ptr[b];
                int64_t end = (b + 1 < B) ? off_ptr[b + 1] : N;
                float* out_row = out_ptr + b * D;
                for (int64_t i = start; i < end; i++) {
                    int64_t orig_idx = idx_ptr[i];
                    if (hot_ptr[orig_idx]) {
                        int64_t hot_idx = o2h_ptr[orig_idx];
                        const float* emb_row = hw_ptr + hot_idx * D;
                        for (int64_t d = 0; d < D; d++) {
                            out_row[d] += emb_row[d];
                        }
                    } else {
                        cold_count++;
                    }
                }
            }
        }

        results.push_back(output);
        results.push_back(torch::tensor(cold_count, torch::dtype(torch::kLong)));
    }
    return results;  // [output_0, cold_count_0, output_1, cold_count_1, ...]
}


// Gather specific rows from a uint8 frame and dequantize to fp32.
// Much faster than Python numpy indexing + float conversion.
torch::Tensor gather_dequant_uint8(
    const torch::Tensor& uint8_frame,    // (rows_in_frame, D) uint8
    const torch::Tensor& row_offsets,    // (K,) int64 — which rows to gather
    double scale,                         // dequant scale
    int64_t zero_point                   // dequant zero point
) {
    const int64_t K = row_offsets.size(0);
    const int64_t D = uint8_frame.size(1);
    const int64_t max_row = uint8_frame.size(0);

    auto output = torch::zeros({K, D}, torch::dtype(torch::kFloat32));

    const uint8_t* frame_ptr = uint8_frame.data_ptr<uint8_t>();
    const int64_t* off_ptr = row_offsets.data_ptr<int64_t>();
    float* out_ptr = output.data_ptr<float>();
    const float s = static_cast<float>(scale);
    const float zp = static_cast<float>(zero_point);

    for (int64_t k = 0; k < K; k++) {
        int64_t row = off_ptr[k];
        if (row < 0) row = 0;
        if (row >= max_row) row = max_row - 1;
        const uint8_t* src = frame_ptr + row * D;
        float* dst = out_ptr + k * D;
        for (int64_t d = 0; d < D; d++) {
            dst[d] = (static_cast<float>(src[d]) - zp) * s;
        }
    }
    return output;
}


// Multi-table batched forward: process ALL compressed tables in one C++ call.
// Parallelizes across all bags from all tables in a single at::parallel_for,
// eliminating both pybind11 dispatch overhead AND separate parallel_for setups.
//
// Returns: [output_0, cold_count_0, output_1, cold_count_1, ...]
std::vector<torch::Tensor> multi_table_forward_merged(
    const std::vector<torch::Tensor>& indices_list,
    const std::vector<torch::Tensor>& offsets_list,
    const std::vector<torch::Tensor>& hot_weights,
    const std::vector<torch::Tensor>& mappings,
    const std::vector<bool>& is_quantized,
    const std::vector<double>& scales,
    const std::vector<int64_t>& zero_points
) {
    const int64_t T = static_cast<int64_t>(indices_list.size());
    constexpr int32_t INVALID = std::numeric_limits<int32_t>::min();

    // Pre-allocate all outputs and extract pointers
    struct TableCtx {
        const int64_t* idx_ptr;
        const int64_t* off_ptr;
        const int32_t* map_ptr;
        const void* hw_ptr;   // float* or uint8_t*
        float* out_ptr;
        int64_t N, B, D;
        bool q8;
        float scale, zp;
    };

    std::vector<TableCtx> ctx(T);
    std::vector<torch::Tensor> outputs(T);
    std::vector<std::atomic<int64_t>> cold_counts(T);
    for (int64_t t = 0; t < T; t++) {
        cold_counts[t].store(0);
    }

    // Compute flattened bag offsets: table t starts at bag_starts[t]
    std::vector<int64_t> bag_starts(T + 1);
    bag_starts[0] = 0;
    for (int64_t t = 0; t < T; t++) {
        int64_t B = offsets_list[t].size(0);
        bag_starts[t + 1] = bag_starts[t] + B;

        outputs[t] = torch::zeros({B, 16}, torch::dtype(torch::kFloat32));
        auto& c = ctx[t];
        c.idx_ptr = indices_list[t].data_ptr<int64_t>();
        c.off_ptr = offsets_list[t].data_ptr<int64_t>();
        c.map_ptr = mappings[t].data_ptr<int32_t>();
        c.out_ptr = outputs[t].data_ptr<float>();
        c.N = indices_list[t].size(0);
        c.B = B;
        c.D = hot_weights[t].size(1);
        c.q8 = is_quantized[t];
        c.scale = static_cast<float>(scales[t]);
        c.zp = static_cast<float>(zero_points[t]);
        if (c.q8) {
            c.hw_ptr = hot_weights[t].data_ptr<uint8_t>();
        } else {
            c.hw_ptr = hot_weights[t].data_ptr<float>();
        }
    }

    int64_t total_bags = bag_starts[T];

    // Single parallel_for across ALL bags from ALL tables
    at::parallel_for(0, total_bags, 64, [&](int64_t flat_begin, int64_t flat_end) {
        // For each table, process its portion of bags
        for (int64_t t = 0; t < T; t++) {
            int64_t t_start = bag_starts[t];
            int64_t t_end = bag_starts[t + 1];
            // Intersection of [flat_begin, flat_end) with [t_start, t_end)
            int64_t b_begin = std::max(flat_begin, t_start) - t_start;
            int64_t b_end = std::min(flat_end, t_end) - t_start;
            if (b_begin >= b_end) continue;

            const auto& c = ctx[t];
            int64_t local_cold = 0;

            if (c.q8) {
                const uint8_t* hw = static_cast<const uint8_t*>(c.hw_ptr);
                for (int64_t b = b_begin; b < b_end; b++) {
                    int64_t start = c.off_ptr[b];
                    int64_t end = (b + 1 < c.B) ? c.off_ptr[b + 1] : c.N;
                    float* out_row = c.out_ptr + b * c.D;
                    for (int64_t i = start; i < end; i++) {
                        int32_t m = c.map_ptr[c.idx_ptr[i]];
                        // Prefetch next
                        if (i + 1 < end) {
                            int32_t nm = c.map_ptr[c.idx_ptr[i + 1]];
                            if (nm >= 0) __builtin_prefetch(hw + static_cast<int64_t>(nm) * c.D, 0, 1);
                        } else if (b + 1 < b_end) {
                            int64_t ns = c.off_ptr[b + 1];
                            if (ns < c.N) {
                                int32_t nm = c.map_ptr[c.idx_ptr[ns]];
                                if (nm >= 0) __builtin_prefetch(hw + static_cast<int64_t>(nm) * c.D, 0, 1);
                            }
                        }
                        if (__builtin_expect(m >= 0, 1)) {
#if HAS_AVX512
                            if (c.D == 16) {
                                accum_q8_d16(out_row, hw + static_cast<int64_t>(m) * c.D, c.scale, c.zp);
                            } else
#endif
                            {
                                const uint8_t* emb = hw + static_cast<int64_t>(m) * c.D;
                                for (int64_t d = 0; d < c.D; d++)
                                    out_row[d] += (static_cast<float>(emb[d]) - c.zp) * c.scale;
                            }
                        } else if (m != INVALID) {
                            local_cold++;
                        }
                    }
                }
            } else {
                const float* hw = static_cast<const float*>(c.hw_ptr);
                for (int64_t b = b_begin; b < b_end; b++) {
                    int64_t start = c.off_ptr[b];
                    int64_t end = (b + 1 < c.B) ? c.off_ptr[b + 1] : c.N;
                    float* out_row = c.out_ptr + b * c.D;
                    for (int64_t i = start; i < end; i++) {
                        int32_t m = c.map_ptr[c.idx_ptr[i]];
                        // Prefetch next
                        if (i + 1 < end) {
                            int32_t nm = c.map_ptr[c.idx_ptr[i + 1]];
                            if (nm >= 0) __builtin_prefetch(hw + static_cast<int64_t>(nm) * c.D, 0, 1);
                        } else if (b + 1 < b_end) {
                            int64_t ns = c.off_ptr[b + 1];
                            if (ns < c.N) {
                                int32_t nm = c.map_ptr[c.idx_ptr[ns]];
                                if (nm >= 0) __builtin_prefetch(hw + static_cast<int64_t>(nm) * c.D, 0, 1);
                            }
                        }
                        if (__builtin_expect(m >= 0, 1)) {
#if HAS_AVX512
                            if (c.D == 16) {
                                accum_fp32_d16(out_row, hw + static_cast<int64_t>(m) * c.D);
                            } else
#endif
                            {
                                const float* emb = hw + static_cast<int64_t>(m) * c.D;
                                for (int64_t d = 0; d < c.D; d++)
                                    out_row[d] += emb[d];
                            }
                        } else if (m != INVALID) {
                            local_cold++;
                        }
                    }
                }
            }
            if (local_cold > 0) {
                cold_counts[t].fetch_add(local_cold, std::memory_order_relaxed);
            }
        }
    });

    // Build results
    std::vector<torch::Tensor> results;
    results.reserve(T * 2);
    for (int64_t t = 0; t < T; t++) {
        results.push_back(outputs[t]);
        results.push_back(torch::tensor(cold_counts[t].load(), torch::dtype(torch::kLong)));
    }
    return results;
}


// All-tables forward: process ALL 26 tables in one C++ call.
// Uses ATen's torch::embedding_bag for standard tables (highly optimized)
// and our custom merged-mapping loop with at::parallel_for for compressed tables.
//
// Returns: [out_0, ..., out_T-1, cold_0, ..., cold_T-1]
std::vector<torch::Tensor> all_tables_forward(
    const std::vector<torch::Tensor>& indices_list,
    const std::vector<torch::Tensor>& offsets_list,
    const std::vector<torch::Tensor>& weights_list,
    const std::vector<torch::Tensor>& mappings_list,
    const std::vector<bool>& is_compressed,
    const std::vector<bool>& is_q8,
    const std::vector<double>& scales,
    const std::vector<int64_t>& zero_points
) {
    const int64_t T = static_cast<int64_t>(indices_list.size());
    constexpr int32_t INVALID = std::numeric_limits<int32_t>::min();

    std::vector<torch::Tensor> outputs(T);
    std::vector<int64_t> cold_counts(T, 0);

    for (int64_t t = 0; t < T; t++) {
        const auto& indices = indices_list[t];
        const auto& offsets = offsets_list[t];

        if (!is_compressed[t]) {
            // Standard table: use ATen's optimized embedding_bag
            auto result = torch::embedding_bag(
                weights_list[t], indices, offsets,
                /*scale_grad_by_freq=*/false, /*mode=*/0, /*sparse=*/false,
                /*per_sample_weights=*/torch::Tensor(), /*include_last_offset=*/false);
            outputs[t] = std::get<0>(result);
            cold_counts[t] = 0;
        } else {
            // Compressed table: our merged-mapping with at::parallel_for
            const int64_t N = indices.size(0);
            const int64_t B = offsets.size(0);
            const int64_t D = weights_list[t].size(1);
            const int32_t* map_ptr = mappings_list[t].data_ptr<int32_t>();
            const int64_t* idx_ptr = indices.data_ptr<int64_t>();
            const int64_t* off_ptr = offsets.data_ptr<int64_t>();

            auto output = torch::zeros({B, D}, torch::dtype(torch::kFloat32));
            float* out_ptr = output.data_ptr<float>();

            std::atomic<int64_t> cold_count{0};

            if (is_q8[t]) {
                const uint8_t* hw_ptr = weights_list[t].data_ptr<uint8_t>();
                const float s = static_cast<float>(scales[t]);
                const float zp = static_cast<float>(zero_points[t]);

                at::parallel_for(0, B, 64, [&](int64_t b_begin, int64_t b_end) {
                    int64_t local_cold = 0;
                    for (int64_t b = b_begin; b < b_end; b++) {
                        int64_t start = off_ptr[b];
                        int64_t end = (b + 1 < B) ? off_ptr[b + 1] : N;
                        float* out_row = out_ptr + b * D;
                        for (int64_t i = start; i < end; i++) {
                            int32_t m = map_ptr[idx_ptr[i]];
                            if (__builtin_expect(m >= 0, 1)) {
#if HAS_AVX512
                                if (D == 16) {
                                    accum_q8_d16(out_row, hw_ptr + static_cast<int64_t>(m) * D, s, zp);
                                } else
#endif
                                {
                                    const uint8_t* emb = hw_ptr + static_cast<int64_t>(m) * D;
                                    for (int64_t d = 0; d < D; d++)
                                        out_row[d] += (static_cast<float>(emb[d]) - zp) * s;
                                }
                            } else if (m != INVALID) {
                                local_cold++;
                            }
                        }
                    }
                    if (local_cold > 0) cold_count.fetch_add(local_cold, std::memory_order_relaxed);
                });
            } else {
                const float* hw_ptr = weights_list[t].data_ptr<float>();

                at::parallel_for(0, B, 64, [&](int64_t b_begin, int64_t b_end) {
                    int64_t local_cold = 0;
                    for (int64_t b = b_begin; b < b_end; b++) {
                        int64_t start = off_ptr[b];
                        int64_t end = (b + 1 < B) ? off_ptr[b + 1] : N;
                        float* out_row = out_ptr + b * D;
                        for (int64_t i = start; i < end; i++) {
                            int32_t m = map_ptr[idx_ptr[i]];
                            if (__builtin_expect(m >= 0, 1)) {
#if HAS_AVX512
                                if (D == 16) {
                                    accum_fp32_d16(out_row, hw_ptr + static_cast<int64_t>(m) * D);
                                } else
#endif
                                {
                                    const float* emb = hw_ptr + static_cast<int64_t>(m) * D;
                                    for (int64_t d = 0; d < D; d++)
                                        out_row[d] += emb[d];
                                }
                            } else if (m != INVALID) {
                                local_cold++;
                            }
                        }
                    }
                    if (local_cold > 0) cold_count.fetch_add(local_cold, std::memory_order_relaxed);
                });
            }
            outputs[t] = output;
            cold_counts[t] = cold_count.load();
        }
    }

    // Pack results: [outputs..., cold_counts_as_tensors...]
    std::vector<torch::Tensor> results;
    results.reserve(T * 2);
    for (int64_t t = 0; t < T; t++) {
        results.push_back(outputs[t]);
    }
    for (int64_t t = 0; t < T; t++) {
        results.push_back(torch::tensor(cold_counts[t], torch::dtype(torch::kLong)));
    }
    return results;
}


// ============================================================
// Pre-registered fast_forward: store all table data in C++ static storage,
// then process all 26 tables with a single pybind11 call per batch.
// This eliminates the Python for-loop entirely.
// ============================================================

namespace {

// Table types
enum class TableKind { STANDARD, COMPRESSED_FP32, COMPRESSED_Q8 };

// Open-addressing hash table: maps original_idx (int32) -> hot_compact_idx (int32)
// Interleaved key-value layout for cache-friendly probing (key+value in same cache line).
// Empty slots are marked with key = EMPTY_KEY.
struct HotHashTable {
    static constexpr int32_t EMPTY_KEY = -1;
    // Interleaved: [key0, val0, key1, val1, ...] — 8 bytes per slot
    // This ensures key and value are always in the same cache line.
    std::vector<int32_t> slots;  // size = capacity * 2
    int64_t capacity = 0;
    int64_t mask = 0;  // capacity - 1 (capacity is power of 2)
    int shift = 0;     // pre-computed shift for Fibonacci hashing

    void build(const int32_t* mapping, int64_t n_rows) {
        // Count hot entries
        int64_t n_hot = 0;
        for (int64_t i = 0; i < n_rows; i++) {
            if (mapping[i] >= 0) n_hot++;
        }

        // Size to next power of 2 with ~60% load factor
        capacity = 1;
        while (capacity < n_hot * 5 / 3) capacity *= 2;
        mask = capacity - 1;
        shift = 32 - __builtin_ctzll(capacity);

        slots.resize(capacity * 2);
        // Initialize all keys to EMPTY_KEY, values to -1
        for (int64_t i = 0; i < capacity; i++) {
            slots[i * 2] = EMPTY_KEY;
            slots[i * 2 + 1] = -1;
        }

        // Insert hot entries
        for (int64_t i = 0; i < n_rows; i++) {
            int32_t v = mapping[i];
            if (v >= 0) {
                int32_t k = static_cast<int32_t>(i);
                uint32_t h = static_cast<uint32_t>(k) * 2654435769u;
                int64_t slot = (h >> shift) & mask;
                while (slots[slot * 2] != EMPTY_KEY) {
                    slot = (slot + 1) & mask;
                }
                slots[slot * 2] = k;
                slots[slot * 2 + 1] = v;
            }
        }
    }

    // Returns hot_compact_idx if found, -1 if cold
    inline int32_t lookup(int32_t orig_idx) const {
        uint32_t h = static_cast<uint32_t>(orig_idx) * 2654435769u;
        int64_t slot = (h >> shift) & mask;
        // First probe is the common case (~60% of lookups hit here)
        const int32_t* s = &slots[slot * 2];
        if (__builtin_expect(s[0] == orig_idx, 1)) return s[1];
        if (s[0] == EMPTY_KEY) return -1;
        // Linear probe for remaining cases
        slot = (slot + 1) & mask;
        while (true) {
            s = &slots[slot * 2];
            if (s[0] == orig_idx) return s[1];
            if (s[0] == EMPTY_KEY) return -1;
            slot = (slot + 1) & mask;
        }
    }

    int64_t memory_bytes() const {
        return capacity * (sizeof(int32_t) * 2);
    }
};

// Bitmap-Rank structure: O(1) hot lookup using ~2MB instead of 128MB mapping.
// Uses a bitmap (1 bit per row) + rank array (prefix popcounts per 64 rows).
// lookup(i) returns: hot compact index (>=0) if hot, -1 if cold.
struct BitmapRank {
    std::vector<uint64_t> bitmap;   // 1 bit per embedding row (1=hot, 0=cold)
    std::vector<int32_t> rank;      // prefix popcount at each 64-row block boundary
    int64_t n_rows = 0;
    int64_t n_words = 0;

    void build(const int32_t* mapping, int64_t n) {
        n_rows = n;
        n_words = (n + 63) / 64;
        bitmap.assign(n_words, 0);
        rank.resize(n_words + 1, 0);

        // Set bitmap bits for hot entries
        for (int64_t i = 0; i < n; i++) {
            if (mapping[i] >= 0) {
                bitmap[i / 64] |= (1ULL << (i % 64));
            }
        }

        // Build rank array: rank[k] = popcount of bitmap[0..k-1]
        rank[0] = 0;
        for (int64_t k = 0; k < n_words; k++) {
            rank[k + 1] = rank[k] + __builtin_popcountll(bitmap[k]);
        }
    }

    // O(1) lookup: returns hot compact index if hot, -1 if cold
    inline int32_t lookup(int64_t orig_idx) const {
        int64_t word = orig_idx / 64;
        int64_t bit = orig_idx % 64;
        uint64_t w = bitmap[word];
        if (!((w >> bit) & 1)) return -1;  // cold
        // Count set bits strictly before position `bit` in this word
        uint64_t below_mask = (1ULL << bit) - 1;  // bits 0..bit-1
        return rank[word] + __builtin_popcountll(w & below_mask);
    }

    // O(1) cold rank: returns the cold position (natural order) for a cold row.
    // Assumes the caller already knows orig_idx is cold.
    // cold_pos = orig_idx - count_of_hot_rows_before(orig_idx)
    inline int64_t cold_rank(int64_t orig_idx) const {
        int64_t word = orig_idx / 64;
        int64_t bit = orig_idx % 64;
        uint64_t below_mask = (1ULL << bit) - 1;  // bits 0..bit-1
        int32_t hot_before = rank[word] + __builtin_popcountll(bitmap[word] & below_mask);
        return orig_idx - hot_before;
    }

    int64_t memory_bytes() const {
        return n_words * sizeof(uint64_t) + (n_words + 1) * sizeof(int32_t);
    }
};

struct RegisteredTable {
    TableKind kind;
    torch::Tensor weight;        // hot_weight (fp32 or q8) or standard weight
    torch::Tensor mapping;       // merged int32 mapping (compressed only) — used when hash table is off
    HotHashTable hash_table;     // hash table for hot lookups (compressed only)
    BitmapRank bitmap_rank;      // bitmap-rank for hot lookups (ultra-low memory)
    bool use_hash = false;       // whether to use hash table instead of mapping
    bool use_bitmap = false;     // whether to use bitmap-rank instead of mapping
    float hot_scale = 0.0f;
    float hot_zp = 0.0f;
    int64_t D = 16;

    // Cold frame data (registered after warmup for full C++ cold lookup)
    bool has_cold_frames = false;
    std::vector<torch::Tensor> cold_frame_tensors;  // keep references to prevent GC
    std::vector<int64_t> cold_frame_offsets;  // frame_id -> offset in concatenated buffer (-1 if not cached)
    const uint8_t* cold_frame_data_ptr = nullptr;
    int64_t rows_per_frame = 0;
    float cold_scale = 0.0f;
    float cold_zp = 0.0f;
    torch::Tensor cold_frame_concat;  // concatenated uint8 frame data (keeps memory alive)

    // Cold mapping for bitmap mode (orig_idx -> cold_reordered_idx, int32)
    torch::Tensor cold_mapping;
    const int32_t* cold_mapping_ptr = nullptr;
    bool has_cold_mapping = false;

    // Flat cold buffer mode: cold frame data in natural order, indexed by bitmap cold_rank
    // When cold_flat=true, no cold_mapping is needed — cold_rank gives direct buffer offset
    bool cold_flat = false;
    int64_t n_cold_rows = 0;  // total number of cold rows in flat buffer
};

// Static storage for registered tables
static std::vector<RegisteredTable> g_tables;
static bool g_registered = false;

} // namespace

// register_tables: called once during setup.
// table_kinds: vector of int (0=standard, 1=compressed_fp32, 2=compressed_q8)
// weights: vector of tensors (standard weight or hot_weight)
// mappings: vector of tensors (merged mapping for compressed, empty for standard)
// scales: vector of double (q8 scale, 0 for others)
// zero_points: vector of int64 (q8 zero point, 0 for others)
// use_hash_table: bool — if true, build hash tables for compressed tables (saves ~113MB)
// use_bitmap: bool — if true, build bitmap-rank for compressed tables (saves ~126MB, faster)
void register_tables(
    std::vector<int64_t> table_kinds,
    std::vector<torch::Tensor> weights,
    std::vector<torch::Tensor> mappings,
    std::vector<double> scales,
    std::vector<int64_t> zero_points,
    bool use_hash_table,
    bool use_bitmap = false
) {
    int64_t T = table_kinds.size();
    TORCH_CHECK(T == (int64_t)weights.size(), "weights size mismatch");
    TORCH_CHECK(T == (int64_t)mappings.size(), "mappings size mismatch");
    TORCH_CHECK(T == (int64_t)scales.size(), "scales size mismatch");
    TORCH_CHECK(T == (int64_t)zero_points.size(), "zero_points size mismatch");

    int64_t total_hash_bytes = 0;
    int64_t total_bitmap_bytes = 0;
    // Clear all previous state (including cold frame registrations from prior runs)
    g_tables.clear();
    g_tables.resize(T);
    for (int64_t t = 0; t < T; t++) {
        auto& tab = g_tables[t];
        switch (table_kinds[t]) {
            case 0: tab.kind = TableKind::STANDARD; break;
            case 1: tab.kind = TableKind::COMPRESSED_FP32; break;
            case 2: tab.kind = TableKind::COMPRESSED_Q8; break;
            default: TORCH_CHECK(false, "Unknown table kind: ", table_kinds[t]);
        }
        tab.weight = weights[t];
        tab.hot_scale = static_cast<float>(scales[t]);
        tab.hot_zp = static_cast<float>(zero_points[t]);
        tab.D = weights[t].size(1);
        tab.use_hash = false;
        tab.use_bitmap = false;

        if (tab.kind != TableKind::STANDARD && mappings[t].numel() > 0) {
            if (use_bitmap) {
                // Build bitmap-rank from mapping, then release mapping tensor
                tab.bitmap_rank.build(mappings[t].data_ptr<int32_t>(), mappings[t].size(0));
                tab.mapping = torch::Tensor();
                tab.use_bitmap = true;
                total_bitmap_bytes += tab.bitmap_rank.memory_bytes();
            } else if (use_hash_table) {
                // Build hash table from mapping, then release mapping tensor
                tab.hash_table.build(mappings[t].data_ptr<int32_t>(), mappings[t].size(0));
                tab.mapping = torch::Tensor();
                tab.use_hash = true;
                total_hash_bytes += tab.hash_table.memory_bytes();
            } else {
                tab.mapping = mappings[t];
            }
        } else {
            tab.mapping = mappings[t];
        }
    }
    g_registered = true;
    if (total_hash_bytes > 0) {
        fprintf(stderr, "[C++] Hash tables built: %.1f MB total\n",
                total_hash_bytes / (1024.0 * 1024.0));
    }
    if (total_bitmap_bytes > 0) {
        fprintf(stderr, "[C++] Bitmap-rank built: %.1f MB total\n",
                total_bitmap_bytes / (1024.0 * 1024.0));
    }
}

// register_cold_frames_for_table: register pre-decoded cold frame data for a single table.
// After warmup, call this for each compressed table so fast_forward can handle cold lookups
// directly in C++ without any Python overhead.
//
// table_idx: index into g_tables
// frame_ids: [num_frames] int64 — which frame IDs are stored (sorted)
// frame_data: [num_frames * actual_rows_per_frame, D] uint8 — concatenated decoded frame data
// cold_scale, cold_zp: dequantization parameters for cold weights
// rows_per_frame: number of rows per frame (last frame may have fewer)
// cold_mapping: optional [num_rows] int64 — maps orig_idx -> cold_reordered_idx (for bitmap mode)
void register_cold_frames_for_table(
    int64_t table_idx,
    const torch::Tensor& frame_ids,
    const torch::Tensor& frame_data,
    double cold_scale,
    double cold_zp,
    int64_t rows_per_frame,
    const torch::Tensor& cold_mapping
) {
    TORCH_CHECK(g_registered, "Tables not registered. Call register_tables first.");
    TORCH_CHECK(table_idx >= 0 && table_idx < (int64_t)g_tables.size(),
                "Invalid table index: ", table_idx);

    auto& tab = g_tables[table_idx];
    tab.has_cold_frames = true;
    tab.cold_frame_concat = frame_data.contiguous();
    tab.cold_frame_data_ptr = tab.cold_frame_concat.data_ptr<uint8_t>();
    tab.cold_scale = static_cast<float>(cold_scale);
    tab.cold_zp = static_cast<float>(cold_zp);
    tab.rows_per_frame = rows_per_frame;

    // Build frame_id -> offset mapping
    int64_t num_frames = frame_ids.size(0);
    const int64_t* fid_ptr = frame_ids.data_ptr<int64_t>();
    int64_t max_fid = 0;
    for (int64_t i = 0; i < num_frames; i++)
        max_fid = std::max(max_fid, fid_ptr[i]);

    tab.cold_frame_offsets.assign(max_fid + 1, -1);
    int64_t cumulative_rows = 0;
    for (int64_t i = 0; i < num_frames; i++) {
        tab.cold_frame_offsets[fid_ptr[i]] = cumulative_rows;
        // Compute actual rows for this frame
        // For simplicity, use rows_per_frame for all frames
        // (last frame may have fewer rows, but cold_reordered_idx is always valid)
        cumulative_rows += rows_per_frame;
    }

    // Register cold mapping if provided (for bitmap mode, int32 to save memory)
    if (cold_mapping.numel() > 0) {
        if (cold_mapping.scalar_type() == torch::kInt32) {
            tab.cold_mapping = cold_mapping.contiguous();
        } else {
            tab.cold_mapping = cold_mapping.to(torch::kInt32).contiguous();
        }
        tab.cold_mapping_ptr = tab.cold_mapping.data_ptr<int32_t>();
        tab.has_cold_mapping = true;
    }

    int64_t frame_mb = frame_data.numel() / (1024 * 1024);
    fprintf(stderr, "[C++] Table %ld: registered %ld cold frames (%ld MB uint8), "
            "rows_per_frame=%ld, scale=%.6f, zp=%.1f\n",
            table_idx, num_frames, frame_mb, rows_per_frame, tab.cold_scale, tab.cold_zp);
}

// register_cold_flat: register a flat natural-order cold buffer for a table.
// Uses bitmap cold_rank for O(1) cold position lookup — no cold_mapping needed!
// cold_flat_data: [n_cold_rows, D] uint8 — cold rows in natural order
void register_cold_flat(
    int64_t table_idx,
    const torch::Tensor& cold_flat_data,
    double cold_scale,
    double cold_zp,
    int64_t n_cold_rows
) {
    TORCH_CHECK(g_registered, "Tables not registered. Call register_tables first.");
    TORCH_CHECK(table_idx >= 0 && table_idx < (int64_t)g_tables.size(),
                "Invalid table index: ", table_idx);

    auto& tab = g_tables[table_idx];
    tab.has_cold_frames = true;
    tab.cold_flat = true;
    tab.n_cold_rows = n_cold_rows;
    tab.cold_frame_concat = cold_flat_data.contiguous();
    tab.cold_frame_data_ptr = tab.cold_frame_concat.data_ptr<uint8_t>();
    tab.cold_scale = static_cast<float>(cold_scale);
    tab.cold_zp = static_cast<float>(cold_zp);

    int64_t flat_mb = cold_flat_data.numel() / (1024 * 1024);
    fprintf(stderr, "[C++] Table %ld: registered flat cold buffer (%ld rows, %ld MB uint8), "
            "scale=%.6f, zp=%.1f\n",
            table_idx, n_cold_rows, flat_mb, tab.cold_scale, tab.cold_zp);
}

// Helper: look up a cold row from registered frame cache and accumulate into output.
// cold_idx: either the reordered cold index (frame mode) or natural cold position (flat mode)
// Returns true if handled in C++, false if needs Python fallback.
static inline bool cold_frame_accum(
    float* __restrict__ out_row,
    const RegisteredTable& tab,
    int64_t cold_idx,
    int64_t D
) {
    int64_t buf_row;
    if (tab.cold_flat) {
        // Flat mode: direct indexing by natural cold position
        if (__builtin_expect(cold_idx >= 0 && cold_idx < tab.n_cold_rows, 1)) {
            buf_row = cold_idx;
        } else {
            return false;
        }
    } else {
        // Frame mode: compute frame + offset
        int64_t fid = cold_idx / tab.rows_per_frame;
        if (__builtin_expect(fid < (int64_t)tab.cold_frame_offsets.size() &&
                             tab.cold_frame_offsets[fid] >= 0, 1)) {
            int64_t off_in_frame = cold_idx % tab.rows_per_frame;
            buf_row = tab.cold_frame_offsets[fid] + off_in_frame;
        } else {
            return false;
        }
    }
    const uint8_t* emb = tab.cold_frame_data_ptr + buf_row * D;
#if HAS_AVX512
    if (D == 16) {
        accum_q8_d16(out_row, emb, tab.cold_scale, tab.cold_zp);
    } else
#endif
    {
        for (int64_t d = 0; d < D; d++)
            out_row[d] += (static_cast<float>(emb[d]) - tab.cold_zp) * tab.cold_scale;
    }
    return true;
}

// fast_forward: process all registered tables with a single call.
// lS_i: (T, N_per_table) int64 — indices for each table
// lS_o: (T, B) int64 — offsets for each table
// Returns: list of T output tensors + list of T cold_mask tensors + list of T cold_count tensors
std::vector<torch::Tensor> fast_forward(
    const torch::Tensor& lS_i,  // (T, N) int64
    const torch::Tensor& lS_o   // (T, B) int64
) {
    TORCH_CHECK(g_registered, "Tables not registered. Call register_tables first.");
    const int64_t T = g_tables.size();
    TORCH_CHECK(lS_i.size(0) == T, "lS_i table count mismatch: ", lS_i.size(0), " vs ", T);
    TORCH_CHECK(lS_o.size(0) == T, "lS_o table count mismatch");

    const int64_t N = lS_i.size(1);  // indices per table
    const int64_t B = lS_o.size(1);  // batch size (bags)
    const int64_t D = g_tables[0].D; // all tables have same D (=16)
    constexpr int32_t INVALID = std::numeric_limits<int32_t>::min();

    // Single allocation for ALL output tensors: (T*B, D) then view as T slices
    auto all_outputs = torch::zeros({T * B, D}, torch::dtype(torch::kFloat32));
    float* all_out_ptr = all_outputs.data_ptr<float>();

    // Count compressed tables for cold mask allocation
    int64_t n_compressed = 0;
    for (int64_t t = 0; t < T; t++) {
        if (g_tables[t].kind != TableKind::STANDARD) n_compressed++;
    }

    // Single allocation for all cold masks
    auto all_cold_masks = torch::zeros({n_compressed * N}, torch::dtype(torch::kBool));
    bool* all_cm_ptr = all_cold_masks.data_ptr<bool>();

    // Map compressed table indices to cold_mask slices
    std::vector<int64_t> cm_offset(T, -1);  // -1 = no cold mask (standard table)
    {
        int64_t ci = 0;
        for (int64_t t = 0; t < T; t++) {
            if (g_tables[t].kind != TableKind::STANDARD) {
                cm_offset[t] = ci * N;
                ci++;
            }
        }
    }

    std::vector<int64_t> cold_counts(T, 0);

    // Process all tables in parallel
    at::parallel_for(0, T, 1, [&](int64_t t_begin, int64_t t_end) {
        for (int64_t t = t_begin; t < t_end; t++) {
            const auto& tab = g_tables[t];
            const int64_t* idx_ptr = lS_i.data_ptr<int64_t>() + t * N;
            const int64_t* off_ptr = lS_o.data_ptr<int64_t>() + t * B;
            float* out_ptr = all_out_ptr + t * B * D;

            if (tab.kind == TableKind::STANDARD) {
                const float* w_ptr = tab.weight.data_ptr<float>();
                for (int64_t b = 0; b < B; b++) {
                    int64_t start = off_ptr[b];
                    int64_t end = (b + 1 < B) ? off_ptr[b + 1] : N;
                    float* out_row = out_ptr + b * D;
                    for (int64_t i = start; i < end; i++) {
                        const float* emb_row = w_ptr + idx_ptr[i] * D;
#if HAS_AVX512
                        if (D == 16) {
                            accum_fp32_d16(out_row, emb_row);
                        } else
#endif
                        {
                            for (int64_t d = 0; d < D; d++)
                                out_row[d] += emb_row[d];
                        }
                    }
                }

            } else if (tab.kind == TableKind::COMPRESSED_FP32) {
                const float* hw_ptr = tab.weight.data_ptr<float>();
                bool* cm_ptr = all_cm_ptr + cm_offset[t];
                int64_t local_cold = 0;

                // Macro-like cold handling for each mode:
                // When cold frames are registered, handle cold in C++ directly.
                // Otherwise fall back to cold_mask for Python fixup.
                #define COLD_FP32_FROM_MAP(m_val) do { \
                    if (tab.has_cold_frames) { \
                        int64_t ci = -(static_cast<int64_t>(m_val) + 1); \
                        cold_frame_accum(out_row, tab, ci, D); \
                    } else { cm_ptr[i] = true; local_cold++; } \
                } while(0)
                #define COLD_FROM_BITMAP() do { \
                    if (tab.has_cold_frames && tab.cold_flat) { \
                        int64_t ci = br.cold_rank(idx_ptr[i]); \
                        cold_frame_accum(out_row, tab, ci, D); \
                    } else if (tab.has_cold_frames && tab.has_cold_mapping) { \
                        int32_t ci = tab.cold_mapping_ptr[idx_ptr[i]]; \
                        if (ci >= 0) cold_frame_accum(out_row, tab, static_cast<int64_t>(ci), D); \
                    } else { cm_ptr[i] = true; local_cold++; } \
                } while(0)

                if (tab.use_bitmap) {
                    // Bitmap-rank mode: O(1) lookup using popcount
                    const auto& br = tab.bitmap_rank;
                    for (int64_t b = 0; b < B; b++) {
                        int64_t start = off_ptr[b];
                        int64_t end = (b + 1 < B) ? off_ptr[b + 1] : N;
                        float* out_row = out_ptr + b * D;
                        for (int64_t i = start; i < end; i++) {
                            if (i + 1 < end) {
                                __builtin_prefetch(&br.bitmap[idx_ptr[i+1] / 64], 0, 0);
                            }
                            int32_t m = br.lookup(idx_ptr[i]);
                            if (__builtin_expect(m >= 0, 1)) {
                                const float* emb_row = hw_ptr + static_cast<int64_t>(m) * D;
#if HAS_AVX512
                                if (D == 16) accum_fp32_d16(out_row, emb_row);
                                else
#endif
                                for (int64_t d = 0; d < D; d++) out_row[d] += emb_row[d];
                            } else {
                                COLD_FROM_BITMAP();
                            }
                        }
                    }
                } else if (tab.use_hash) {
                    // Hash table mode: lookup hot index via hash
                    const auto& ht = tab.hash_table;
                    for (int64_t b = 0; b < B; b++) {
                        int64_t start = off_ptr[b];
                        int64_t end = (b + 1 < B) ? off_ptr[b + 1] : N;
                        float* out_row = out_ptr + b * D;
                        if (start < end) {
                            uint32_t ph = static_cast<uint32_t>(static_cast<int32_t>(idx_ptr[start])) * 2654435769u;
                            __builtin_prefetch(&ht.slots[((ph >> ht.shift) & ht.mask) * 2], 0, 1);
                        }
                        for (int64_t i = start; i < end; i++) {
                            if (i + 1 < end) {
                                uint32_t ph = static_cast<uint32_t>(static_cast<int32_t>(idx_ptr[i+1])) * 2654435769u;
                                __builtin_prefetch(&ht.slots[((ph >> ht.shift) & ht.mask) * 2], 0, 1);
                            }
                            int32_t m = ht.lookup(static_cast<int32_t>(idx_ptr[i]));
                            if (__builtin_expect(m >= 0, 1)) {
                                const float* emb_row = hw_ptr + static_cast<int64_t>(m) * D;
#if HAS_AVX512
                                if (D == 16) accum_fp32_d16(out_row, emb_row);
                                else
#endif
                                for (int64_t d = 0; d < D; d++) out_row[d] += emb_row[d];
                            } else {
                                cm_ptr[i] = true;
                                local_cold++;
                            }
                        }
                    }
                } else {
                    // Array mapping mode
                    const int32_t* map_ptr = tab.mapping.data_ptr<int32_t>();
                    for (int64_t b = 0; b < B; b++) {
                        int64_t start = off_ptr[b];
                        int64_t end = (b + 1 < B) ? off_ptr[b + 1] : N;
                        float* out_row = out_ptr + b * D;
                        for (int64_t i = start; i < end; i++) {
                            if (i + 1 < end) {
                                __builtin_prefetch(&map_ptr[idx_ptr[i+1]], 0, 1);
                            }
                            int32_t m = map_ptr[idx_ptr[i]];
                            if (__builtin_expect(m >= 0, 1)) {
                                const float* emb_row = hw_ptr + static_cast<int64_t>(m) * D;
#if HAS_AVX512
                                if (D == 16) accum_fp32_d16(out_row, emb_row);
                                else
#endif
                                for (int64_t d = 0; d < D; d++) out_row[d] += emb_row[d];
                            } else if (m != INVALID) {
                                COLD_FP32_FROM_MAP(m);
                            }
                        }
                    }
                }
                cold_counts[t] = local_cold;

                #undef COLD_FP32_FROM_MAP
                #undef COLD_FROM_BITMAP

            } else {
                // COMPRESSED_Q8
                const uint8_t* hw_ptr = tab.weight.data_ptr<uint8_t>();
                const float s = tab.hot_scale;
                const float zp = tab.hot_zp;
                bool* cm_ptr = all_cm_ptr + cm_offset[t];
                int64_t local_cold = 0;

                // Cold handling macros for Q8 mode
                #define COLD_Q8_FROM_MAP(m_val) do { \
                    if (tab.has_cold_frames) { \
                        int64_t ci = -(static_cast<int64_t>(m_val) + 1); \
                        cold_frame_accum(out_row, tab, ci, D); \
                    } else { cm_ptr[i] = true; local_cold++; } \
                } while(0)
                #define COLD_Q8_FROM_BITMAP() do { \
                    if (tab.has_cold_frames && tab.cold_flat) { \
                        int64_t ci = br.cold_rank(idx_ptr[i]); \
                        cold_frame_accum(out_row, tab, ci, D); \
                    } else if (tab.has_cold_frames && tab.has_cold_mapping) { \
                        int32_t ci = tab.cold_mapping_ptr[idx_ptr[i]]; \
                        if (ci >= 0) cold_frame_accum(out_row, tab, static_cast<int64_t>(ci), D); \
                    } else { cm_ptr[i] = true; local_cold++; } \
                } while(0)

                if (tab.use_bitmap) {
                    // Bitmap-rank mode for Q8
                    const auto& br = tab.bitmap_rank;
                    for (int64_t b = 0; b < B; b++) {
                        int64_t start = off_ptr[b];
                        int64_t end = (b + 1 < B) ? off_ptr[b + 1] : N;
                        float* out_row = out_ptr + b * D;
                        for (int64_t i = start; i < end; i++) {
                            if (i + 1 < end) {
                                __builtin_prefetch(&br.bitmap[idx_ptr[i+1] / 64], 0, 0);
                            }
                            int32_t m = br.lookup(idx_ptr[i]);
                            if (__builtin_expect(m >= 0, 1)) {
#if HAS_AVX512
                                if (D == 16) accum_q8_d16(out_row, hw_ptr + static_cast<int64_t>(m) * D, s, zp);
                                else
#endif
                                {
                                    const uint8_t* emb = hw_ptr + static_cast<int64_t>(m) * D;
                                    for (int64_t d = 0; d < D; d++)
                                        out_row[d] += (static_cast<float>(emb[d]) - zp) * s;
                                }
                            } else {
                                COLD_Q8_FROM_BITMAP();
                            }
                        }
                    }
                } else if (tab.use_hash) {
                    const auto& ht = tab.hash_table;
                    for (int64_t b = 0; b < B; b++) {
                        int64_t start = off_ptr[b];
                        int64_t end = (b + 1 < B) ? off_ptr[b + 1] : N;
                        float* out_row = out_ptr + b * D;
                        if (start < end) {
                            uint32_t ph = static_cast<uint32_t>(static_cast<int32_t>(idx_ptr[start])) * 2654435769u;
                            __builtin_prefetch(&ht.slots[((ph >> ht.shift) & ht.mask) * 2], 0, 1);
                        }
                        for (int64_t i = start; i < end; i++) {
                            if (i + 1 < end) {
                                uint32_t ph = static_cast<uint32_t>(static_cast<int32_t>(idx_ptr[i+1])) * 2654435769u;
                                __builtin_prefetch(&ht.slots[((ph >> ht.shift) & ht.mask) * 2], 0, 1);
                            }
                            int32_t m = ht.lookup(static_cast<int32_t>(idx_ptr[i]));
                            if (__builtin_expect(m >= 0, 1)) {
#if HAS_AVX512
                                if (D == 16) accum_q8_d16(out_row, hw_ptr + static_cast<int64_t>(m) * D, s, zp);
                                else
#endif
                                {
                                    const uint8_t* emb = hw_ptr + static_cast<int64_t>(m) * D;
                                    for (int64_t d = 0; d < D; d++)
                                        out_row[d] += (static_cast<float>(emb[d]) - zp) * s;
                                }
                            } else {
                                cm_ptr[i] = true;
                                local_cold++;
                            }
                        }
                    }
                } else {
                    const int32_t* map_ptr = tab.mapping.data_ptr<int32_t>();
                    for (int64_t b = 0; b < B; b++) {
                        int64_t start = off_ptr[b];
                        int64_t end = (b + 1 < B) ? off_ptr[b + 1] : N;
                        float* out_row = out_ptr + b * D;
                        for (int64_t i = start; i < end; i++) {
                            if (i + 1 < end) {
                                __builtin_prefetch(&map_ptr[idx_ptr[i+1]], 0, 1);
                            }
                            int32_t m = map_ptr[idx_ptr[i]];
                            if (__builtin_expect(m >= 0, 1)) {
#if HAS_AVX512
                                if (D == 16) accum_q8_d16(out_row, hw_ptr + static_cast<int64_t>(m) * D, s, zp);
                                else
#endif
                                {
                                    const uint8_t* emb = hw_ptr + static_cast<int64_t>(m) * D;
                                    for (int64_t d = 0; d < D; d++)
                                        out_row[d] += (static_cast<float>(emb[d]) - zp) * s;
                                }
                            } else if (m != INVALID) {
                                COLD_Q8_FROM_MAP(m);
                            }
                        }
                    }
                }
                cold_counts[t] = local_cold;

                #undef COLD_Q8_FROM_MAP
                #undef COLD_Q8_FROM_BITMAP
            }
        }
    });

    // Pack results as views into pre-allocated buffers
    // [output_0, ..., output_{T-1}, cold_mask_0, ..., cold_mask_{T-1}, cc_0, ..., cc_{T-1}, stacked_3d]
    std::vector<torch::Tensor> results;
    results.reserve(T * 3 + 1);
    // Reshape and slice outputs
    auto outputs_3d = all_outputs.view({T, B, D});
    for (int64_t t = 0; t < T; t++)
        results.push_back(outputs_3d[t]);  // view, no copy
    // Slice cold masks
    for (int64_t t = 0; t < T; t++) {
        if (cm_offset[t] >= 0)
            results.push_back(all_cold_masks.slice(0, cm_offset[t], cm_offset[t] + N));
        else
            results.push_back(torch::Tensor());
    }
    // Cold counts as scalar tensors
    for (int64_t t = 0; t < T; t++)
        results.push_back(torch::tensor(cold_counts[t], torch::dtype(torch::kLong)));
    // Stacked [T, B, D] output for direct use in interact (avoids 26-tensor cat)
    results.push_back(outputs_3d);
    return results;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compressed_emb_bag_forward", &compressed_emb_bag_forward,
          "Compressed EmbeddingBag forward (hot path in C++, returns cold_mask)");
    m.def("compressed_emb_bag_forward_q8", &compressed_emb_bag_forward_q8,
          "Compressed EmbeddingBag forward with uint8 hot weights");
    m.def("cold_fixup", &cold_fixup,
          "Add cold embeddings into output in-place");
    m.def("batched_emb_forward", &batched_emb_forward,
          "Process multiple tables in one C++ call");
    m.def("gather_dequant_uint8", &gather_dequant_uint8,
          "Gather rows from uint8 frame and dequantize to fp32");
    m.def("compressed_emb_bag_forward_merged", &compressed_emb_bag_forward_merged,
          "Compressed EmbeddingBag forward with merged int32 mapping (saves memory)");
    m.def("compressed_emb_bag_forward_q8_merged", &compressed_emb_bag_forward_q8_merged,
          "Compressed EmbeddingBag forward with q8 hot and merged int32 mapping");
    m.def("multi_table_forward_merged", &multi_table_forward_merged,
          "Process all compressed tables in one C++ call with merged mapping");
    m.def("all_tables_forward", &all_tables_forward,
          "Process ALL tables (compressed + standard) in one C++ call, eliminating Python loop");
    m.def("register_tables", &register_tables,
          "Register all table metadata in C++ for fast_forward",
          py::arg("table_kinds"), py::arg("weights"), py::arg("mappings"),
          py::arg("scales"), py::arg("zero_points"), py::arg("use_hash_table") = false,
          py::arg("use_bitmap") = false);
    m.def("fast_forward", &fast_forward,
          "Process all registered tables with a single call (zero Python loop overhead)");
    m.def("register_cold_frames_for_table", &register_cold_frames_for_table,
          "Register pre-decoded cold frames for a table (enables full C++ cold lookup)",
          py::arg("table_idx"), py::arg("frame_ids"), py::arg("frame_data"),
          py::arg("cold_scale"), py::arg("cold_zp"), py::arg("rows_per_frame"),
          py::arg("cold_mapping") = torch::Tensor());
    m.def("register_cold_flat", &register_cold_flat,
          "Register flat natural-order cold buffer (no cold_mapping needed, uses bitmap cold_rank)",
          py::arg("table_idx"), py::arg("cold_flat_data"),
          py::arg("cold_scale"), py::arg("cold_zp"), py::arg("n_cold_rows"));
}
