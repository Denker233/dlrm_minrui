// compressed_emb.cpp — C++ PyTorch extension for CompressedEmbeddingBag forward pass
// Replaces Python hot/cold routing + scatter_add with efficient C++ implementation.

#include <torch/extension.h>
#include <vector>
#include <cstring>
#include <algorithm>
#include <atomic>

// FFmpeg/libav headers for direct H.265 decode (avoids PyAV Python overhead)
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
}

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


// ============================================================
// Frame packing/unpacking optimizations
// Reduce copy overhead when converting between row layout and tiled video frames.
// ============================================================

// Tile embedding rows into a video frame in a single C++ pass.
// Replaces Python: reshape(tc,tr,4,4).transpose(0,2,1,3).reshape(H,W)
// which forces a copy due to non-contiguous transpose.
//
// Input: emb_uint8 (N, 16) uint8 — quantized embedding rows (contiguous)
// Output: frame (height, width) uint8 — tiled frame ready for H.265 encode
//
// Layout: each 16-byte embedding becomes a 4x4 tile.
// Row r → tile grid (r / tiles_per_row, r % tiles_per_row)
// 4 bytes per tile-row written contiguously in the frame.
torch::Tensor tile_rows_to_frame(
    const torch::Tensor& emb_uint8,  // (N, 16) uint8
    int64_t width,
    int64_t height
) {
    TORCH_CHECK(emb_uint8.scalar_type() == torch::kUInt8, "Expected uint8 input");
    TORCH_CHECK(emb_uint8.is_contiguous(), "Input must be contiguous");
    const int64_t N = emb_uint8.size(0);
    const int64_t D = emb_uint8.size(1);
    TORCH_CHECK(D == 16, "Expected D=16, got ", D);
    TORCH_CHECK(width % 4 == 0 && height % 4 == 0, "Width and height must be multiples of 4");

    const int64_t tiles_per_row = width / 4;
    const int64_t rows_per_frame = tiles_per_row * (height / 4);
    TORCH_CHECK(N <= rows_per_frame, "Too many rows (", N, ") for frame ", width, "x", height,
                " (max ", rows_per_frame, ")");

    auto frame = torch::zeros({height, width}, torch::dtype(torch::kUInt8));
    const uint8_t* src = emb_uint8.data_ptr<uint8_t>();
    uint8_t* dst = frame.data_ptr<uint8_t>();

    at::parallel_for(0, N, 512, [&](int64_t begin, int64_t end) {
        for (int64_t r = begin; r < end; r++) {
            const int64_t ty = r / tiles_per_row;
            const int64_t tx = r % tiles_per_row;
            const uint8_t* row = src + r * 16;
            // Write 4 tile-rows of 4 bytes each
            for (int ly = 0; ly < 4; ly++) {
                std::memcpy(dst + (ty * 4 + ly) * width + tx * 4, row + ly * 4, 4);
            }
        }
    });

    return frame;
}

// Untile a video frame back to embedding rows in a single C++ pass.
// Replaces Python: reshape(tc,4,tr,4).transpose(0,2,1,3).reshape(N,16)
// which forces a copy due to non-contiguous transpose.
//
// Input: frame (height, width) uint8 — tiled frame from H.265 decode
// Output: emb (num_rows, 16) uint8 — embedding rows
torch::Tensor untile_frame_to_rows(
    const torch::Tensor& frame,  // (height, width) uint8
    int64_t num_rows             // actual valid rows (<= rows_per_frame)
) {
    TORCH_CHECK(frame.scalar_type() == torch::kUInt8, "Expected uint8 frame");
    const int64_t height = frame.size(0);
    const int64_t width = frame.size(1);
    TORCH_CHECK(width % 4 == 0 && height % 4 == 0);

    const int64_t tiles_per_row = width / 4;
    const int64_t rows_per_frame = tiles_per_row * (height / 4);
    if (num_rows <= 0) num_rows = rows_per_frame;
    TORCH_CHECK(num_rows <= rows_per_frame);

    const uint8_t* src = frame.contiguous().data_ptr<uint8_t>();
    auto output = torch::empty({num_rows, 16}, torch::dtype(torch::kUInt8));
    uint8_t* dst = output.data_ptr<uint8_t>();

    at::parallel_for(0, num_rows, 512, [&](int64_t begin, int64_t end) {
        for (int64_t r = begin; r < end; r++) {
            const int64_t ty = r / tiles_per_row;
            const int64_t tx = r % tiles_per_row;
            uint8_t* row = dst + r * 16;
            for (int ly = 0; ly < 4; ly++) {
                std::memcpy(row + ly * 4, src + (ty * 4 + ly) * width + tx * 4, 4);
            }
        }
    });

    return output;
}

// Gather specific rows from a tiled frame WITHOUT untiling the whole frame.
// For a cache miss that only needs K << rows_per_frame rows, this avoids
// copying the entire frame just to index a few rows.
//
// Input: frame (height, width) uint8, row_indices (K,) int64
// Output: gathered (K, 16) uint8
torch::Tensor gather_from_tiled_frame(
    const torch::Tensor& frame,        // (height, width) uint8
    const torch::Tensor& row_indices,  // (K,) int64
    int64_t tiles_per_row              // width / 4
) {
    const int64_t K = row_indices.size(0);
    const int64_t width = frame.size(1);

    auto output = torch::empty({K, 16}, torch::dtype(torch::kUInt8));
    const uint8_t* src = frame.contiguous().data_ptr<uint8_t>();
    const int64_t* idx = row_indices.data_ptr<int64_t>();
    uint8_t* dst = output.data_ptr<uint8_t>();

    at::parallel_for(0, K, 128, [&](int64_t begin, int64_t end) {
        for (int64_t k = begin; k < end; k++) {
            const int64_t r = idx[k];
            const int64_t ty = r / tiles_per_row;
            const int64_t tx = r % tiles_per_row;
            uint8_t* row = dst + k * 16;
            for (int ly = 0; ly < 4; ly++) {
                std::memcpy(row + ly * 4, src + (ty * 4 + ly) * width + tx * 4, 4);
            }
        }
    });

    return output;
}

// Gather from tiled frame AND dequantize to fp32 in one pass.
// Eliminates the intermediate uint8 buffer entirely.
// Particularly useful for on-demand cold lookups where we decode a frame
// and only need a few specific rows.
//
// Input: frame (height, width) uint8, row_indices (K,) int64
// Output: gathered_fp32 (K, 16) float32
torch::Tensor gather_dequant_from_tiled_frame(
    const torch::Tensor& frame,
    const torch::Tensor& row_indices,
    int64_t tiles_per_row,
    double scale,
    int64_t zero_point
) {
    const int64_t K = row_indices.size(0);
    const int64_t width = frame.size(1);
    const float s = static_cast<float>(scale);
    const float zp = static_cast<float>(zero_point);

    auto output = torch::empty({K, 16}, torch::dtype(torch::kFloat32));
    const uint8_t* src = frame.contiguous().data_ptr<uint8_t>();
    const int64_t* idx = row_indices.data_ptr<int64_t>();
    float* dst = output.data_ptr<float>();

    at::parallel_for(0, K, 128, [&](int64_t begin, int64_t end) {
        for (int64_t k = begin; k < end; k++) {
            const int64_t r = idx[k];
            const int64_t ty = r / tiles_per_row;
            const int64_t tx = r % tiles_per_row;
            float* out_row = dst + k * 16;

            for (int ly = 0; ly < 4; ly++) {
                const uint8_t* tile_row = src + (ty * 4 + ly) * width + tx * 4;
                float* out_seg = out_row + ly * 4;
                for (int lx = 0; lx < 4; lx++) {
                    out_seg[lx] = (static_cast<float>(tile_row[lx]) - zp) * s;
                }
            }
        }
    });

    return output;
}

// Fused: take fp32 embedding weights + cold index list → produce tiled uint8 frame
// in one pass. Combines: gather scattered rows + quantize + tile.
// Eliminates 3 intermediate buffers that the Python path creates.
//
// Returns: [frame (H,W) uint8, scale_tensor, zp_tensor]
std::vector<torch::Tensor> fused_gather_quantize_tile(
    const torch::Tensor& weight,        // (total_rows, 16) fp32 — full embedding table
    const torch::Tensor& cold_indices,  // (N_cold,) int64 — which rows are cold
    int64_t width,
    int64_t height
) {
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32);
    const int64_t N = cold_indices.size(0);
    const int64_t D = weight.size(1);
    TORCH_CHECK(D == 16);
    TORCH_CHECK(width % 4 == 0 && height % 4 == 0);

    const int64_t tiles_per_row = width / 4;
    const int64_t rows_per_frame = tiles_per_row * (height / 4);
    TORCH_CHECK(N <= rows_per_frame, "Need multiple frames for ", N, " rows");

    const float* w_ptr = weight.data_ptr<float>();
    const int64_t* idx_ptr = cold_indices.data_ptr<int64_t>();

    // Pass 1: parallel min/max for quantization parameters
    int num_threads = at::get_num_threads();
    std::vector<float> thr_min(num_threads, std::numeric_limits<float>::max());
    std::vector<float> thr_max(num_threads, std::numeric_limits<float>::lowest());

    at::parallel_for(0, N, 512, [&](int64_t begin, int64_t end) {
        int tid = at::get_thread_num();
        float lmin = thr_min[tid], lmax = thr_max[tid];
        for (int64_t i = begin; i < end; i++) {
            const float* row = w_ptr + idx_ptr[i] * D;
#if HAS_AVX512
            __m512 v = _mm512_loadu_ps(row);
            float row_min = _mm512_reduce_min_ps(v);
            float row_max = _mm512_reduce_max_ps(v);
            if (row_min < lmin) lmin = row_min;
            if (row_max > lmax) lmax = row_max;
#else
            for (int64_t d = 0; d < D; d++) {
                if (row[d] < lmin) lmin = row[d];
                if (row[d] > lmax) lmax = row[d];
            }
#endif
        }
        thr_min[tid] = lmin;
        thr_max[tid] = lmax;
    });

    float gmin = thr_min[0], gmax = thr_max[0];
    for (int i = 1; i < num_threads; i++) {
        if (thr_min[i] < gmin) gmin = thr_min[i];
        if (thr_max[i] > gmax) gmax = thr_max[i];
    }

    float scale = (gmax - gmin) / 255.0f;
    if (scale == 0.0f) scale = 1.0f;
    float inv_scale = 1.0f / scale;
    int32_t zero_point = static_cast<int32_t>(std::round(-gmin * inv_scale));

    // Pass 2: gather + quantize + tile in one pass (no intermediate buffers)
    auto frame = torch::zeros({height, width}, torch::dtype(torch::kUInt8));
    uint8_t* frame_ptr = frame.data_ptr<uint8_t>();

    at::parallel_for(0, N, 512, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; i++) {
            const float* row = w_ptr + idx_ptr[i] * D;
            const int64_t ty = i / tiles_per_row;
            const int64_t tx = i % tiles_per_row;

#if HAS_AVX512
            // Load 16 fp32, quantize to int32, clamp to [0,255]
            __m512 v = _mm512_loadu_ps(row);
            __m512 scaled = _mm512_mul_ps(v, _mm512_set1_ps(inv_scale));
            __m512i rounded = _mm512_cvtps_epi32(scaled);  // rounds to nearest
            __m512i with_zp = _mm512_add_epi32(rounded, _mm512_set1_epi32(zero_point));
            __m512i clamped = _mm512_max_epi32(_mm512_setzero_si512(),
                              _mm512_min_epi32(with_zp, _mm512_set1_epi32(255)));
            // Extract to temp array, write 4 bytes per tile-row
            alignas(64) int32_t vals[16];
            _mm512_store_epi32(vals, clamped);
            for (int ly = 0; ly < 4; ly++) {
                uint8_t* dst = frame_ptr + (ty * 4 + ly) * width + tx * 4;
                for (int lx = 0; lx < 4; lx++) {
                    dst[lx] = static_cast<uint8_t>(vals[ly * 4 + lx]);
                }
            }
#else
            for (int d = 0; d < 16; d++) {
                int32_t q = static_cast<int32_t>(std::round(row[d] * inv_scale)) + zero_point;
                if (q < 0) q = 0;
                if (q > 255) q = 255;
                int ly = d / 4, lx = d % 4;
                frame_ptr[(ty * 4 + ly) * width + tx * 4 + lx] = static_cast<uint8_t>(q);
            }
#endif
        }
    });

    return {frame,
            torch::tensor(static_cast<double>(scale)),
            torch::tensor(static_cast<int64_t>(zero_point))};
}

// Fused quantize + tile: already-gathered uint8 rows → tiled frame.
// Similar to tile_rows_to_frame but also handles fp32 input with quantization.
// This version takes fp32 rows (already gathered/contiguous) and produces a tiled frame.
//
// Returns: [frame (H,W) uint8, scale_tensor, zp_tensor]
std::vector<torch::Tensor> fused_quantize_tile(
    const torch::Tensor& rows_fp32,  // (N, 16) fp32 — contiguous cold rows
    int64_t width,
    int64_t height
) {
    TORCH_CHECK(rows_fp32.scalar_type() == torch::kFloat32);
    TORCH_CHECK(rows_fp32.is_contiguous());
    const int64_t N = rows_fp32.size(0);
    const int64_t D = rows_fp32.size(1);
    TORCH_CHECK(D == 16);

    const int64_t tiles_per_row = width / 4;
    const int64_t rows_per_frame = tiles_per_row * (height / 4);
    TORCH_CHECK(N <= rows_per_frame);

    const float* src = rows_fp32.data_ptr<float>();

    // Pass 1: min/max
    int num_threads = at::get_num_threads();
    std::vector<float> thr_min(num_threads, std::numeric_limits<float>::max());
    std::vector<float> thr_max(num_threads, std::numeric_limits<float>::lowest());

    at::parallel_for(0, N, 512, [&](int64_t begin, int64_t end) {
        int tid = at::get_thread_num();
        float lmin = thr_min[tid], lmax = thr_max[tid];
        for (int64_t i = begin; i < end; i++) {
            const float* row = src + i * D;
#if HAS_AVX512
            __m512 v = _mm512_loadu_ps(row);
            float rmin = _mm512_reduce_min_ps(v);
            float rmax = _mm512_reduce_max_ps(v);
            if (rmin < lmin) lmin = rmin;
            if (rmax > lmax) lmax = rmax;
#else
            for (int64_t d = 0; d < D; d++) {
                if (row[d] < lmin) lmin = row[d];
                if (row[d] > lmax) lmax = row[d];
            }
#endif
        }
        thr_min[tid] = lmin;
        thr_max[tid] = lmax;
    });

    float gmin = thr_min[0], gmax = thr_max[0];
    for (int i = 1; i < num_threads; i++) {
        if (thr_min[i] < gmin) gmin = thr_min[i];
        if (thr_max[i] > gmax) gmax = thr_max[i];
    }

    float scale = (gmax - gmin) / 255.0f;
    if (scale == 0.0f) scale = 1.0f;
    float inv_scale = 1.0f / scale;
    int32_t zero_point = static_cast<int32_t>(std::round(-gmin * inv_scale));

    // Pass 2: quantize + tile
    auto frame = torch::zeros({height, width}, torch::dtype(torch::kUInt8));
    uint8_t* frame_ptr = frame.data_ptr<uint8_t>();

    at::parallel_for(0, N, 512, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; i++) {
            const float* row = src + i * D;
            const int64_t ty = i / tiles_per_row;
            const int64_t tx = i % tiles_per_row;

#if HAS_AVX512
            __m512 v = _mm512_loadu_ps(row);
            __m512 scaled = _mm512_mul_ps(v, _mm512_set1_ps(inv_scale));
            __m512i rounded = _mm512_cvtps_epi32(scaled);
            __m512i with_zp = _mm512_add_epi32(rounded, _mm512_set1_epi32(zero_point));
            __m512i clamped = _mm512_max_epi32(_mm512_setzero_si512(),
                              _mm512_min_epi32(with_zp, _mm512_set1_epi32(255)));
            alignas(64) int32_t vals[16];
            _mm512_store_epi32(vals, clamped);
            for (int ly = 0; ly < 4; ly++) {
                uint8_t* dst = frame_ptr + (ty * 4 + ly) * width + tx * 4;
                for (int lx = 0; lx < 4; lx++) {
                    dst[lx] = static_cast<uint8_t>(vals[ly * 4 + lx]);
                }
            }
#else
            for (int d = 0; d < 16; d++) {
                int32_t q = static_cast<int32_t>(std::round(row[d] * inv_scale)) + zero_point;
                if (q < 0) q = 0;
                if (q > 255) q = 255;
                int ly = d / 4, lx = d % 4;
                frame_ptr[(ty * 4 + ly) * width + tx * 4 + lx] = static_cast<uint8_t>(q);
            }
#endif
        }
    });

    return {frame,
            torch::tensor(static_cast<double>(scale)),
            torch::tensor(static_cast<int64_t>(zero_point))};
}

// Multi-frame version: tile N rows across multiple frames.
// Returns list of frame tensors + scale + zp.
// Useful for large tables that need multiple frames.
std::vector<torch::Tensor> fused_quantize_tile_multiframe(
    const torch::Tensor& rows_uint8,  // (N, 16) uint8 — already quantized
    int64_t width,
    int64_t height
) {
    TORCH_CHECK(rows_uint8.scalar_type() == torch::kUInt8);
    TORCH_CHECK(rows_uint8.is_contiguous());
    const int64_t N = rows_uint8.size(0);

    const int64_t tiles_per_row = width / 4;
    const int64_t rows_per_frame = tiles_per_row * (height / 4);
    const int64_t num_frames = (N + rows_per_frame - 1) / rows_per_frame;

    const uint8_t* src = rows_uint8.data_ptr<uint8_t>();

    // Pre-allocate all frames
    std::vector<torch::Tensor> frames(num_frames);
    std::vector<uint8_t*> frame_ptrs(num_frames);
    for (int64_t f = 0; f < num_frames; f++) {
        frames[f] = torch::zeros({height, width}, torch::dtype(torch::kUInt8));
        frame_ptrs[f] = frames[f].data_ptr<uint8_t>();
    }

    // Parallelize across ALL rows from ALL frames
    at::parallel_for(0, N, 512, [&](int64_t begin, int64_t end) {
        for (int64_t r = begin; r < end; r++) {
            const int64_t f = r / rows_per_frame;
            const int64_t local_r = r - f * rows_per_frame;
            const int64_t ty = local_r / tiles_per_row;
            const int64_t tx = local_r % tiles_per_row;
            const uint8_t* row = src + r * 16;
            uint8_t* frame_ptr = frame_ptrs[f];
            for (int ly = 0; ly < 4; ly++) {
                std::memcpy(frame_ptr + (ty * 4 + ly) * width + tx * 4, row + ly * 4, 4);
            }
        }
    });

    return frames;
}

// Multi-frame fused: fp32 cold rows → quantize → tile across multiple frames.
// Single call replaces the entire Python encode prep pipeline for large tables.
// Returns: [frame_0, frame_1, ..., frame_N-1, scale_tensor, zp_tensor]
std::vector<torch::Tensor> fused_quantize_tile_multiframe_fp32(
    const torch::Tensor& rows_fp32,  // (N, 16) fp32 — contiguous cold rows
    int64_t width,
    int64_t height
) {
    TORCH_CHECK(rows_fp32.scalar_type() == torch::kFloat32);
    TORCH_CHECK(rows_fp32.is_contiguous());
    const int64_t N = rows_fp32.size(0);
    const int64_t D = rows_fp32.size(1);
    TORCH_CHECK(D == 16);

    const int64_t tiles_per_row = width / 4;
    const int64_t rows_per_frame = tiles_per_row * (height / 4);
    const int64_t num_frames = (N + rows_per_frame - 1) / rows_per_frame;

    const float* src = rows_fp32.data_ptr<float>();

    // Pass 1: min/max for quantization
    int nthreads = at::get_num_threads();
    std::vector<float> thr_min(nthreads, std::numeric_limits<float>::max());
    std::vector<float> thr_max(nthreads, std::numeric_limits<float>::lowest());

    at::parallel_for(0, N, 512, [&](int64_t begin, int64_t end) {
        int tid = at::get_thread_num();
        float lmin = thr_min[tid], lmax = thr_max[tid];
        for (int64_t i = begin; i < end; i++) {
            const float* row = src + i * D;
#if HAS_AVX512
            __m512 v = _mm512_loadu_ps(row);
            float rmin = _mm512_reduce_min_ps(v);
            float rmax = _mm512_reduce_max_ps(v);
            if (rmin < lmin) lmin = rmin;
            if (rmax > lmax) lmax = rmax;
#else
            for (int64_t d = 0; d < D; d++) {
                if (row[d] < lmin) lmin = row[d];
                if (row[d] > lmax) lmax = row[d];
            }
#endif
        }
        thr_min[tid] = lmin;
        thr_max[tid] = lmax;
    });

    float gmin = thr_min[0], gmax = thr_max[0];
    for (int i = 1; i < nthreads; i++) {
        if (thr_min[i] < gmin) gmin = thr_min[i];
        if (thr_max[i] > gmax) gmax = thr_max[i];
    }

    float scale = (gmax - gmin) / 255.0f;
    if (scale == 0.0f) scale = 1.0f;
    float inv_scale = 1.0f / scale;
    int32_t zero_point = static_cast<int32_t>(std::round(-gmin * inv_scale));

    // Pre-allocate all frames
    std::vector<torch::Tensor> frames(num_frames);
    std::vector<uint8_t*> frame_ptrs(num_frames);
    for (int64_t f = 0; f < num_frames; f++) {
        frames[f] = torch::zeros({height, width}, torch::dtype(torch::kUInt8));
        frame_ptrs[f] = frames[f].data_ptr<uint8_t>();
    }

    // Pass 2: quantize + tile across all frames in parallel
    at::parallel_for(0, N, 512, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; i++) {
            const float* row = src + i * D;
            const int64_t f = i / rows_per_frame;
            const int64_t local_r = i - f * rows_per_frame;
            const int64_t ty = local_r / tiles_per_row;
            const int64_t tx = local_r % tiles_per_row;
            uint8_t* frame_ptr = frame_ptrs[f];

#if HAS_AVX512
            __m512 v = _mm512_loadu_ps(row);
            __m512 scaled = _mm512_mul_ps(v, _mm512_set1_ps(inv_scale));
            __m512i rounded = _mm512_cvtps_epi32(scaled);
            __m512i with_zp = _mm512_add_epi32(rounded, _mm512_set1_epi32(zero_point));
            __m512i clamped = _mm512_max_epi32(_mm512_setzero_si512(),
                              _mm512_min_epi32(with_zp, _mm512_set1_epi32(255)));
            alignas(64) int32_t vals[16];
            _mm512_store_epi32(vals, clamped);
            for (int ly = 0; ly < 4; ly++) {
                uint8_t* dst = frame_ptr + (ty * 4 + ly) * width + tx * 4;
                for (int lx = 0; lx < 4; lx++) {
                    dst[lx] = static_cast<uint8_t>(vals[ly * 4 + lx]);
                }
            }
#else
            for (int d = 0; d < 16; d++) {
                int32_t q = static_cast<int32_t>(std::round(row[d] * inv_scale)) + zero_point;
                if (q < 0) q = 0;
                if (q > 255) q = 255;
                int ly = d / 4, lx = d % 4;
                frame_ptr[(ty * 4 + ly) * width + tx * 4 + lx] = static_cast<uint8_t>(q);
            }
#endif
        }
    });

    // Pack results: [frame_0, ..., frame_{N-1}, scale, zp]
    std::vector<torch::Tensor> results;
    results.reserve(num_frames + 2);
    for (int64_t f = 0; f < num_frames; f++) {
        results.push_back(frames[f]);
    }
    results.push_back(torch::tensor(static_cast<double>(scale)));
    results.push_back(torch::tensor(static_cast<int64_t>(zero_point)));
    return results;
}

// Multi-frame fused gather from scattered fp32 + quantize + tile.
// Like fused_gather_quantize_tile but handles tables larger than one frame.
// Returns: [frame_0, ..., frame_{N-1}, scale, zp]
std::vector<torch::Tensor> fused_gather_quantize_tile_multiframe(
    const torch::Tensor& weight,        // (total_rows, 16) fp32
    const torch::Tensor& cold_indices,  // (N_cold,) int64
    int64_t width,
    int64_t height
) {
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32);
    const int64_t N = cold_indices.size(0);
    const int64_t D = weight.size(1);
    TORCH_CHECK(D == 16);

    const int64_t tiles_per_row = width / 4;
    const int64_t rows_per_frame = tiles_per_row * (height / 4);
    const int64_t num_frames = (N + rows_per_frame - 1) / rows_per_frame;

    const float* w_ptr = weight.data_ptr<float>();
    const int64_t* idx_ptr = cold_indices.data_ptr<int64_t>();

    // Pass 1: min/max
    int nthreads = at::get_num_threads();
    std::vector<float> thr_min(nthreads, std::numeric_limits<float>::max());
    std::vector<float> thr_max(nthreads, std::numeric_limits<float>::lowest());

    at::parallel_for(0, N, 512, [&](int64_t begin, int64_t end) {
        int tid = at::get_thread_num();
        float lmin = thr_min[tid], lmax = thr_max[tid];
        for (int64_t i = begin; i < end; i++) {
            const float* row = w_ptr + idx_ptr[i] * D;
#if HAS_AVX512
            __m512 v = _mm512_loadu_ps(row);
            float rmin = _mm512_reduce_min_ps(v);
            float rmax = _mm512_reduce_max_ps(v);
            if (rmin < lmin) lmin = rmin;
            if (rmax > lmax) lmax = rmax;
#else
            for (int64_t d = 0; d < D; d++) {
                if (row[d] < lmin) lmin = row[d];
                if (row[d] > lmax) lmax = row[d];
            }
#endif
        }
        thr_min[tid] = lmin;
        thr_max[tid] = lmax;
    });

    float gmin = thr_min[0], gmax = thr_max[0];
    for (int i = 1; i < nthreads; i++) {
        if (thr_min[i] < gmin) gmin = thr_min[i];
        if (thr_max[i] > gmax) gmax = thr_max[i];
    }

    float scale = (gmax - gmin) / 255.0f;
    if (scale == 0.0f) scale = 1.0f;
    float inv_scale = 1.0f / scale;
    int32_t zero_point = static_cast<int32_t>(std::round(-gmin * inv_scale));

    // Pre-allocate frames
    std::vector<torch::Tensor> frames(num_frames);
    std::vector<uint8_t*> frame_ptrs(num_frames);
    for (int64_t f = 0; f < num_frames; f++) {
        frames[f] = torch::zeros({height, width}, torch::dtype(torch::kUInt8));
        frame_ptrs[f] = frames[f].data_ptr<uint8_t>();
    }

    // Pass 2: gather + quantize + tile
    at::parallel_for(0, N, 512, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; i++) {
            const float* row = w_ptr + idx_ptr[i] * D;
            const int64_t f = i / rows_per_frame;
            const int64_t local_r = i - f * rows_per_frame;
            const int64_t ty = local_r / tiles_per_row;
            const int64_t tx = local_r % tiles_per_row;
            uint8_t* frame_ptr = frame_ptrs[f];

#if HAS_AVX512
            __m512 v = _mm512_loadu_ps(row);
            __m512 scaled = _mm512_mul_ps(v, _mm512_set1_ps(inv_scale));
            __m512i rounded = _mm512_cvtps_epi32(scaled);
            __m512i with_zp = _mm512_add_epi32(rounded, _mm512_set1_epi32(zero_point));
            __m512i clamped = _mm512_max_epi32(_mm512_setzero_si512(),
                              _mm512_min_epi32(with_zp, _mm512_set1_epi32(255)));
            alignas(64) int32_t vals[16];
            _mm512_store_epi32(vals, clamped);
            for (int ly = 0; ly < 4; ly++) {
                uint8_t* dst = frame_ptr + (ty * 4 + ly) * width + tx * 4;
                for (int lx = 0; lx < 4; lx++) {
                    dst[lx] = static_cast<uint8_t>(vals[ly * 4 + lx]);
                }
            }
#else
            for (int d = 0; d < 16; d++) {
                int32_t q = static_cast<int32_t>(std::round(row[d] * inv_scale)) + zero_point;
                if (q < 0) q = 0;
                if (q > 255) q = 255;
                int ly = d / 4, lx = d % 4;
                frame_ptr[(ty * 4 + ly) * width + tx * 4 + lx] = static_cast<uint8_t>(q);
            }
#endif
        }
    });

    std::vector<torch::Tensor> results;
    results.reserve(num_frames + 2);
    for (int64_t f = 0; f < num_frames; f++) {
        results.push_back(frames[f]);
    }
    results.push_back(torch::tensor(static_cast<double>(scale)));
    results.push_back(torch::tensor(static_cast<int64_t>(zero_point)));
    return results;
}

// Get frame bytes ready for pipe to ffmpeg. Returns raw bytes as a uint8 1D tensor.
// This avoids Python .tobytes() overhead by giving direct access to frame memory.
torch::Tensor frame_to_bytes(const torch::Tensor& frame) {
    TORCH_CHECK(frame.scalar_type() == torch::kUInt8);
    auto contiguous = frame.contiguous();
    return contiguous.view({-1});
}


// ============================================================
// DIRECT H.265 ENCODE via libavcodec (avoids subprocess ffmpeg overhead)
// ============================================================

/**
 * encode_h265_frame - Encode a single grayscale frame to H.265 in-memory or to file.
 *
 * This replaces the subprocess ffmpeg encode path:
 *   subprocess.Popen(['ffmpeg', ...], stdin=PIPE) → proc.stdin.write(frame.tobytes())
 *
 * Benefits:
 * 1. No subprocess fork+exec overhead (~1-2ms)
 * 2. No tobytes() copy (~0.1ms for 1080p)
 * 3. Direct frame buffer access (no pipe I/O)
 * 4. Can be called from C++ threads for parallel multi-frame encode
 */

// File-scope struct for memory I/O in encode functions (avoids incomplete-type warnings)
struct EncOutputBuf {
    std::vector<uint8_t>* data;
};

/*
 *
 * Args:
 *   frame: (H, W) uint8 tensor (raw grayscale frame)
 *   output_path: file path to write .h265/.mkv output (empty string = return bytes)
 *   lossless: if true, use lossless encoding
 *   crf: quality parameter (0=lossless, 28=default, 51=worst)
 *
 * Returns: if output_path is empty, returns uint8 tensor of compressed bytes.
 *          if output_path is set, writes to file and returns empty tensor.
 */
torch::Tensor encode_h265_frame(
    const torch::Tensor& frame,
    const std::string& output_path,
    bool lossless,
    int crf)
{
    TORCH_CHECK(frame.scalar_type() == torch::kUInt8, "Frame must be uint8");
    TORCH_CHECK(frame.dim() == 2, "Frame must be 2D (H, W)");

    auto frame_c = frame.contiguous();
    int width = frame_c.size(1);
    int height = frame_c.size(0);
    const uint8_t* src = frame_c.data_ptr<uint8_t>();

    // Open output (file or memory)
    AVFormatContext* fmt_ctx = nullptr;
    bool to_memory = output_path.empty();

    if (to_memory) {
        avformat_alloc_output_context2(&fmt_ctx, nullptr, "matroska", nullptr);
    } else {
        avformat_alloc_output_context2(&fmt_ctx, nullptr, nullptr, output_path.c_str());
    }
    TORCH_CHECK(fmt_ctx != nullptr, "Failed to allocate output context");

    // Find H.265 encoder
    const AVCodec* codec = avcodec_find_encoder_by_name("libx265");
    if (!codec) codec = avcodec_find_encoder(AV_CODEC_ID_HEVC);
    TORCH_CHECK(codec != nullptr, "H.265 encoder not found");

    // Create stream
    AVStream* stream = avformat_new_stream(fmt_ctx, codec);
    TORCH_CHECK(stream != nullptr, "Failed to create stream");

    // Configure encoder
    AVCodecContext* enc_ctx = avcodec_alloc_context3(codec);
    TORCH_CHECK(enc_ctx != nullptr, "Failed to alloc encoder context");

    enc_ctx->width = width;
    enc_ctx->height = height;
    enc_ctx->pix_fmt = AV_PIX_FMT_GRAY8;
    enc_ctx->time_base = {1, 1};
    enc_ctx->framerate = {1, 1};
    enc_ctx->gop_size = 1;  // ALL-INTRA
    enc_ctx->thread_count = 1;  // single-frame encode, threading overhead not worth it

    // Set x265 options
    if (lossless || crf == 0) {
        av_opt_set(enc_ctx->priv_data, "preset", "ultrafast", 0);
        av_opt_set(enc_ctx->priv_data, "x265-params",
                   "lossless=1:log-level=error", 0);
    } else {
        av_opt_set(enc_ctx->priv_data, "preset", "ultrafast", 0);
        char params[128];
        snprintf(params, sizeof(params), "crf=%d:log-level=error", crf);
        av_opt_set(enc_ctx->priv_data, "x265-params", params, 0);
    }

    if (fmt_ctx->oformat->flags & AVFMT_GLOBALHEADER)
        enc_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    int ret = avcodec_open2(enc_ctx, codec, nullptr);
    TORCH_CHECK(ret >= 0, "Failed to open encoder");

    avcodec_parameters_from_context(stream->codecpar, enc_ctx);

    // Open I/O
    uint8_t* avio_buffer = nullptr;
    AVIOContext* avio_ctx = nullptr;
    std::vector<uint8_t> output_bytes;
    void* ob_ptr = nullptr;  // track for cleanup

    if (to_memory) {
        // Memory output using a growing buffer
        auto* ob = new EncOutputBuf{&output_bytes};
        ob_ptr = ob;

        avio_buffer = static_cast<uint8_t*>(av_malloc(32768));
        avio_ctx = avio_alloc_context(avio_buffer, 32768, 1, ob,
            nullptr,
            // write callback
            [](void* opaque, uint8_t* buf, int buf_size) -> int {
                auto* b = static_cast<EncOutputBuf*>(opaque);
                b->data->insert(b->data->end(), buf, buf + buf_size);
                return buf_size;
            },
            // seek callback
            [](void* opaque, int64_t offset, int whence) -> int64_t {
                auto* b = static_cast<EncOutputBuf*>(opaque);
                if (whence == AVSEEK_SIZE) return b->data->size();
                if (whence == SEEK_SET) {
                    if (offset > static_cast<int64_t>(b->data->size()))
                        b->data->resize(offset, 0);
                    return offset;
                }
                return -1;
            });
        fmt_ctx->pb = avio_ctx;
    } else {
        ret = avio_open(&fmt_ctx->pb, output_path.c_str(), AVIO_FLAG_WRITE);
        TORCH_CHECK(ret >= 0, "Failed to open output file: ", output_path);
    }

    // Write header
    ret = avformat_write_header(fmt_ctx, nullptr);
    TORCH_CHECK(ret >= 0, "Failed to write header");

    // Create AVFrame and copy data
    AVFrame* avframe = av_frame_alloc();
    avframe->format = AV_PIX_FMT_GRAY8;
    avframe->width = width;
    avframe->height = height;
    avframe->pts = 0;
    av_frame_get_buffer(avframe, 0);

    // Copy frame data (handle linesize padding)
    if (avframe->linesize[0] == width) {
        std::memcpy(avframe->data[0], src, width * height);
    } else {
        for (int y = 0; y < height; y++) {
            std::memcpy(avframe->data[0] + y * avframe->linesize[0],
                       src + y * width, width);
        }
    }

    // Encode
    AVPacket* pkt = av_packet_alloc();

    ret = avcodec_send_frame(enc_ctx, avframe);
    TORCH_CHECK(ret >= 0, "Failed to send frame to encoder");

    while (ret >= 0) {
        ret = avcodec_receive_packet(enc_ctx, pkt);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
        TORCH_CHECK(ret >= 0, "Failed to receive packet from encoder");
        pkt->stream_index = stream->index;
        av_interleaved_write_frame(fmt_ctx, pkt);
        av_packet_unref(pkt);
    }

    // Flush encoder
    avcodec_send_frame(enc_ctx, nullptr);
    while (true) {
        ret = avcodec_receive_packet(enc_ctx, pkt);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
        if (ret >= 0) {
            pkt->stream_index = stream->index;
            av_interleaved_write_frame(fmt_ctx, pkt);
            av_packet_unref(pkt);
        }
    }

    // Write trailer
    av_write_trailer(fmt_ctx);

    // Cleanup
    av_frame_free(&avframe);
    av_packet_free(&pkt);
    avcodec_free_context(&enc_ctx);

    if (to_memory) {
        if (avio_ctx) {
            avio_flush(fmt_ctx->pb);
            // Free the output buffer struct
            if (ob_ptr) {
                delete static_cast<EncOutputBuf*>(ob_ptr);
            }
            av_freep(&avio_ctx->buffer);
            avio_context_free(&avio_ctx);
        }
    } else {
        avio_closep(&fmt_ctx->pb);
    }
    avformat_free_context(fmt_ctx);

    if (to_memory) {
        auto result = torch::empty({static_cast<int64_t>(output_bytes.size())}, torch::kUInt8);
        std::memcpy(result.data_ptr<uint8_t>(), output_bytes.data(), output_bytes.size());
        return result;
    }
    return torch::empty({0}, torch::kUInt8);
}


/**
 * encode_h265_frame_to_file - Convenience wrapper: encode frame directly to file.
 */
void encode_h265_frame_to_file(
    const torch::Tensor& frame,
    const std::string& output_path,
    bool lossless)
{
    encode_h265_frame(frame, output_path, lossless, lossless ? 0 : 28);
}


/**
 * batch_encode_h265_frames - Encode multiple frames to files in parallel.
 *
 * Each frame is encoded independently on its own thread.
 * Returns total compressed bytes.
 */
int64_t batch_encode_h265_frames(
    const std::vector<torch::Tensor>& frames,
    const std::string& output_dir,
    bool lossless)
{
    const int64_t num_frames = frames.size();
    std::vector<std::thread> threads;
    std::vector<std::string> errors(num_frames);
    std::vector<int64_t> sizes(num_frames, 0);

    for (int64_t i = 0; i < num_frames; i++) {
        threads.emplace_back([&, i]() {
            try {
                char fname[64];
                snprintf(fname, sizeof(fname), "frame_%05ld.h265", i);
                std::string path = output_dir + "/" + fname;
                encode_h265_frame(frames[i], path, lossless, lossless ? 0 : 28);
                // Get file size
                FILE* f = fopen(path.c_str(), "rb");
                if (f) {
                    fseek(f, 0, SEEK_END);
                    sizes[i] = ftell(f);
                    fclose(f);
                }
            } catch (const std::exception& e) {
                errors[i] = e.what();
            }
        });
    }
    for (auto& t : threads) t.join();

    for (int64_t i = 0; i < num_frames; i++) {
        TORCH_CHECK(errors[i].empty(),
                    "Failed to encode frame ", i, ": ", errors[i]);
    }

    int64_t total = 0;
    for (auto s : sizes) total += s;
    return total;
}


// ============================================================
// MULTI-CODEC ENCODE/DECODE (H.264, H.265, FFV1)
// ============================================================

/**
 * encode_frame_codec - Encode a single grayscale frame using a specified codec.
 *
 * Supports: "h265" (libx265), "h264" (libx264), "ffv1" (FFV1 lossless).
 * Returns compressed bytes tensor (if output_path is empty) or empty tensor (if writing to file).
 */
torch::Tensor encode_frame_codec(
    const torch::Tensor& frame,
    const std::string& output_path,
    const std::string& codec_name,
    bool lossless,
    int crf)
{
    TORCH_CHECK(frame.scalar_type() == torch::kUInt8, "Frame must be uint8");
    TORCH_CHECK(frame.dim() == 2, "Frame must be 2D (H, W)");

    // Suppress FFmpeg log noise
    av_log_set_level(AV_LOG_ERROR);

    auto frame_c = frame.contiguous();
    int width = frame_c.size(1);
    int height = frame_c.size(0);
    const uint8_t* src = frame_c.data_ptr<uint8_t>();

    // Find the requested encoder
    const AVCodec* codec = nullptr;
    AVPixelFormat pix_fmt = AV_PIX_FMT_GRAY8;
    std::string container_fmt;
    std::string file_ext;

    if (codec_name == "h265" || codec_name == "hevc") {
        codec = avcodec_find_encoder_by_name("libx265");
        if (!codec) codec = avcodec_find_encoder(AV_CODEC_ID_HEVC);
        container_fmt = "matroska";
        file_ext = ".h265";
    } else if (codec_name == "h264" || codec_name == "avc") {
        codec = avcodec_find_encoder_by_name("libx264");
        if (!codec) codec = avcodec_find_encoder(AV_CODEC_ID_H264);
        container_fmt = "matroska";
        file_ext = ".h264";
        // libx264 doesn't support gray8 directly, use yuv420p and convert
        pix_fmt = AV_PIX_FMT_YUV420P;
    } else if (codec_name == "ffv1") {
        codec = avcodec_find_encoder(AV_CODEC_ID_FFV1);
        container_fmt = "matroska";
        file_ext = ".mkv";
    } else {
        TORCH_CHECK(false, "Unsupported codec: ", codec_name,
                     ". Supported: h265, h264, ffv1");
    }
    TORCH_CHECK(codec != nullptr, "Encoder not found for codec: ", codec_name);

    // Allocate output context
    AVFormatContext* fmt_ctx = nullptr;
    bool to_memory = output_path.empty();

    if (to_memory) {
        avformat_alloc_output_context2(&fmt_ctx, nullptr, container_fmt.c_str(), nullptr);
    } else {
        // Always force container format to avoid extension-detection issues
        avformat_alloc_output_context2(&fmt_ctx, nullptr, container_fmt.c_str(), output_path.c_str());
    }
    TORCH_CHECK(fmt_ctx != nullptr, "Failed to allocate output context");

    // Create stream
    AVStream* stream = avformat_new_stream(fmt_ctx, codec);
    TORCH_CHECK(stream != nullptr, "Failed to create stream");

    // Configure encoder
    AVCodecContext* enc_ctx = avcodec_alloc_context3(codec);
    TORCH_CHECK(enc_ctx != nullptr, "Failed to alloc encoder context");

    enc_ctx->width = width;
    enc_ctx->height = height;
    enc_ctx->pix_fmt = pix_fmt;
    enc_ctx->time_base = {1, 1};
    enc_ctx->framerate = {1, 1};
    enc_ctx->gop_size = 1;  // ALL-INTRA
    enc_ctx->thread_count = 1;

    // Codec-specific options
    if (codec_name == "h265" || codec_name == "hevc") {
        if (lossless || crf == 0) {
            av_opt_set(enc_ctx->priv_data, "preset", "ultrafast", 0);
            av_opt_set(enc_ctx->priv_data, "x265-params",
                       "lossless=1:log-level=error", 0);
        } else {
            av_opt_set(enc_ctx->priv_data, "preset", "ultrafast", 0);
            char params[512];
            snprintf(params, sizeof(params), "crf=%d:log-level=error", crf);
            av_opt_set(enc_ctx->priv_data, "x265-params", params, 0);
        }
    } else if (codec_name == "h264" || codec_name == "avc") {
        av_opt_set(enc_ctx->priv_data, "preset", "ultrafast", 0);
        if (lossless) {
            // x264 lossless: qp=0
            av_opt_set(enc_ctx->priv_data, "qp", "0", 0);
        } else {
            char crf_str[16];
            snprintf(crf_str, sizeof(crf_str), "%d", crf);
            av_opt_set(enc_ctx->priv_data, "crf", crf_str, 0);
        }
        // Suppress log noise
        av_log_set_level(AV_LOG_ERROR);
    } else if (codec_name == "ffv1") {
        // FFV1 is always lossless, no special options needed
        // Use higher compression level for better ratio
        enc_ctx->level = 3;  // FFV1 version 3
    }

    if (fmt_ctx->oformat->flags & AVFMT_GLOBALHEADER)
        enc_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    int ret = avcodec_open2(enc_ctx, codec, nullptr);
    TORCH_CHECK(ret >= 0, "Failed to open encoder for codec: ", codec_name);

    avcodec_parameters_from_context(stream->codecpar, enc_ctx);

    // Open I/O
    uint8_t* avio_buffer = nullptr;
    AVIOContext* avio_ctx = nullptr;
    std::vector<uint8_t> output_bytes;
    void* ob_ptr = nullptr;

    if (to_memory) {
        auto* ob = new EncOutputBuf{&output_bytes};
        ob_ptr = ob;

        avio_buffer = static_cast<uint8_t*>(av_malloc(32768));
        avio_ctx = avio_alloc_context(avio_buffer, 32768, 1, ob,
            nullptr,
            [](void* opaque, uint8_t* buf, int buf_size) -> int {
                auto* b = static_cast<EncOutputBuf*>(opaque);
                b->data->insert(b->data->end(), buf, buf + buf_size);
                return buf_size;
            },
            [](void* opaque, int64_t offset, int whence) -> int64_t {
                auto* b = static_cast<EncOutputBuf*>(opaque);
                if (whence == AVSEEK_SIZE) return b->data->size();
                if (whence == SEEK_SET) {
                    if (offset > static_cast<int64_t>(b->data->size()))
                        b->data->resize(offset, 0);
                    return offset;
                }
                return -1;
            });
        fmt_ctx->pb = avio_ctx;
    } else {
        ret = avio_open(&fmt_ctx->pb, output_path.c_str(), AVIO_FLAG_WRITE);
        TORCH_CHECK(ret >= 0, "Failed to open output file: ", output_path);
    }

    // Write header
    ret = avformat_write_header(fmt_ctx, nullptr);
    TORCH_CHECK(ret >= 0, "Failed to write header");

    // Allocate frame
    AVFrame* av_frame = av_frame_alloc();
    av_frame->format = pix_fmt;
    av_frame->width = width;
    av_frame->height = height;
    ret = av_frame_get_buffer(av_frame, 0);
    TORCH_CHECK(ret >= 0, "Failed to allocate frame buffer");

    // Copy grayscale data to frame
    if (pix_fmt == AV_PIX_FMT_GRAY8) {
        for (int y = 0; y < height; y++) {
            std::memcpy(av_frame->data[0] + y * av_frame->linesize[0],
                       src + y * width, width);
        }
    } else if (pix_fmt == AV_PIX_FMT_YUV420P) {
        // Convert gray8 to YUV420P: Y=gray, U=V=128 (neutral chroma)
        for (int y = 0; y < height; y++) {
            std::memcpy(av_frame->data[0] + y * av_frame->linesize[0],
                       src + y * width, width);
        }
        // Fill U and V planes with 128 (neutral)
        int chroma_h = (height + 1) / 2;
        int chroma_w = (width + 1) / 2;
        for (int y = 0; y < chroma_h; y++) {
            std::memset(av_frame->data[1] + y * av_frame->linesize[1], 128, chroma_w);
            std::memset(av_frame->data[2] + y * av_frame->linesize[2], 128, chroma_w);
        }
    }

    av_frame->pts = 0;

    // Encode
    AVPacket* pkt = av_packet_alloc();
    ret = avcodec_send_frame(enc_ctx, av_frame);
    TORCH_CHECK(ret >= 0, "Failed to send frame to encoder");

    // Flush
    avcodec_send_frame(enc_ctx, nullptr);

    while (true) {
        ret = avcodec_receive_packet(enc_ctx, pkt);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
        TORCH_CHECK(ret >= 0, "Failed to receive packet");
        pkt->stream_index = stream->index;
        av_interleaved_write_frame(fmt_ctx, pkt);
        av_packet_unref(pkt);
    }

    av_write_trailer(fmt_ctx);

    // Cleanup
    av_packet_free(&pkt);
    av_frame_free(&av_frame);
    avcodec_free_context(&enc_ctx);

    if (to_memory) {
        if (avio_ctx) {
            av_freep(&avio_ctx->buffer);
            avio_context_free(&avio_ctx);
        }
        if (ob_ptr) delete static_cast<EncOutputBuf*>(ob_ptr);
        avformat_free_context(fmt_ctx);
        // Return bytes as tensor
        auto result = torch::empty({static_cast<int64_t>(output_bytes.size())}, torch::kUInt8);
        std::memcpy(result.data_ptr<uint8_t>(), output_bytes.data(), output_bytes.size());
        return result;
    } else {
        avio_closep(&fmt_ctx->pb);
        avformat_free_context(fmt_ctx);
        return torch::empty({0}, torch::kUInt8);
    }
}

/**
 * encode_frame_with_params - Like encode_frame_codec but accepts custom codec params string.
 *
 * For H.265, extra_params is appended to x265-params (e.g., "slices=4:wpp=1").
 * For H.264, extra_params is appended to x264-params.
 */
void encode_frame_with_params(
    const torch::Tensor& frame,
    const std::string& output_path,
    const std::string& codec_name,
    bool lossless,
    int crf,
    const std::string& extra_params)
{
    TORCH_CHECK(frame.scalar_type() == torch::kUInt8, "Frame must be uint8");
    TORCH_CHECK(frame.dim() == 2, "Frame must be 2D (H, W)");

    av_log_set_level(AV_LOG_ERROR);

    auto frame_c = frame.contiguous();
    int width = frame_c.size(1);
    int height = frame_c.size(0);
    const uint8_t* src = frame_c.data_ptr<uint8_t>();

    const AVCodec* codec = nullptr;
    AVPixelFormat pix_fmt = AV_PIX_FMT_GRAY8;

    if (codec_name == "h265" || codec_name == "hevc") {
        codec = avcodec_find_encoder_by_name("libx265");
        if (!codec) codec = avcodec_find_encoder(AV_CODEC_ID_HEVC);
    } else if (codec_name == "h264" || codec_name == "avc") {
        codec = avcodec_find_encoder_by_name("libx264");
        if (!codec) codec = avcodec_find_encoder(AV_CODEC_ID_H264);
        pix_fmt = AV_PIX_FMT_YUV420P;
    } else if (codec_name == "ffv1") {
        codec = avcodec_find_encoder(AV_CODEC_ID_FFV1);
    }
    TORCH_CHECK(codec != nullptr, "Encoder not found for codec: ", codec_name);

    AVFormatContext* fmt_ctx = nullptr;
    avformat_alloc_output_context2(&fmt_ctx, nullptr, "matroska", output_path.c_str());
    TORCH_CHECK(fmt_ctx != nullptr, "Failed to allocate output context");

    AVStream* stream = avformat_new_stream(fmt_ctx, codec);
    TORCH_CHECK(stream != nullptr, "Failed to create stream");

    AVCodecContext* enc_ctx = avcodec_alloc_context3(codec);
    enc_ctx->width = width;
    enc_ctx->height = height;
    enc_ctx->pix_fmt = pix_fmt;
    enc_ctx->time_base = {1, 1};
    enc_ctx->framerate = {1, 1};
    enc_ctx->gop_size = 1;
    enc_ctx->thread_count = 1;

    if (codec_name == "h265" || codec_name == "hevc") {
        av_opt_set(enc_ctx->priv_data, "preset", "ultrafast", 0);
        char params[512];
        if (lossless || crf == 0) {
            snprintf(params, sizeof(params), "lossless=1:log-level=error");
        } else {
            snprintf(params, sizeof(params), "crf=%d:log-level=error", crf);
        }
        if (!extra_params.empty()) {
            strncat(params, ":", sizeof(params) - strlen(params) - 1);
            strncat(params, extra_params.c_str(), sizeof(params) - strlen(params) - 1);
        }
        av_opt_set(enc_ctx->priv_data, "x265-params", params, 0);
    } else if (codec_name == "h264" || codec_name == "avc") {
        av_opt_set(enc_ctx->priv_data, "preset", "ultrafast", 0);
        if (lossless) {
            av_opt_set(enc_ctx->priv_data, "qp", "0", 0);
        } else {
            char crf_str[16];
            snprintf(crf_str, sizeof(crf_str), "%d", crf);
            av_opt_set(enc_ctx->priv_data, "crf", crf_str, 0);
        }
        if (!extra_params.empty()) {
            av_opt_set(enc_ctx->priv_data, "x264-params", extra_params.c_str(), 0);
        }
    } else if (codec_name == "ffv1") {
        enc_ctx->level = 3;
        enc_ctx->slices = 4;  // FFV1 supports multi-slice
    }

    if (fmt_ctx->oformat->flags & AVFMT_GLOBALHEADER)
        enc_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    int ret = avcodec_open2(enc_ctx, codec, nullptr);
    TORCH_CHECK(ret >= 0, "Failed to open encoder");

    avcodec_parameters_from_context(stream->codecpar, enc_ctx);

    ret = avio_open(&fmt_ctx->pb, output_path.c_str(), AVIO_FLAG_WRITE);
    TORCH_CHECK(ret >= 0, "Failed to open output file: ", output_path);

    ret = avformat_write_header(fmt_ctx, nullptr);
    TORCH_CHECK(ret >= 0, "Failed to write header");

    AVFrame* av_frame = av_frame_alloc();
    av_frame->format = pix_fmt;
    av_frame->width = width;
    av_frame->height = height;
    ret = av_frame_get_buffer(av_frame, 0);

    if (pix_fmt == AV_PIX_FMT_GRAY8) {
        for (int y = 0; y < height; y++) {
            std::memcpy(av_frame->data[0] + y * av_frame->linesize[0],
                       src + y * width, width);
        }
    } else if (pix_fmt == AV_PIX_FMT_YUV420P) {
        for (int y = 0; y < height; y++) {
            std::memcpy(av_frame->data[0] + y * av_frame->linesize[0],
                       src + y * width, width);
        }
        int chroma_h = (height + 1) / 2;
        int chroma_w = (width + 1) / 2;
        for (int y = 0; y < chroma_h; y++) {
            std::memset(av_frame->data[1] + y * av_frame->linesize[1], 128, chroma_w);
            std::memset(av_frame->data[2] + y * av_frame->linesize[2], 128, chroma_w);
        }
    }

    av_frame->pts = 0;
    AVPacket* pkt = av_packet_alloc();
    avcodec_send_frame(enc_ctx, av_frame);
    avcodec_send_frame(enc_ctx, nullptr);

    while (true) {
        ret = avcodec_receive_packet(enc_ctx, pkt);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
        pkt->stream_index = stream->index;
        av_interleaved_write_frame(fmt_ctx, pkt);
        av_packet_unref(pkt);
    }

    av_write_trailer(fmt_ctx);
    av_packet_free(&pkt);
    av_frame_free(&av_frame);
    avcodec_free_context(&enc_ctx);
    avio_closep(&fmt_ctx->pb);
    avformat_free_context(fmt_ctx);
}


/**
 * batch_encode_frames_codec - Encode multiple frames to files in parallel using specified codec.
 *
 * Returns total compressed bytes.
 */
int64_t batch_encode_frames_codec(
    const std::vector<torch::Tensor>& frames,
    const std::string& output_dir,
    const std::string& codec_name,
    bool lossless)
{
    const int64_t num_frames = frames.size();

    // Determine file extension
    std::string ext = ".h265";
    if (codec_name == "h264" || codec_name == "avc") ext = ".h264";
    else if (codec_name == "ffv1") ext = ".mkv";

    std::vector<std::thread> threads;
    std::vector<std::string> errors(num_frames);
    std::vector<int64_t> sizes(num_frames, 0);

    for (int64_t i = 0; i < num_frames; i++) {
        threads.emplace_back([&, i]() {
            try {
                char fname[64];
                snprintf(fname, sizeof(fname), "frame_%05ld%s", i, ext.c_str());
                std::string path = output_dir + "/" + fname;
                encode_frame_codec(frames[i], path, codec_name, lossless, lossless ? 0 : 28);
                FILE* f = fopen(path.c_str(), "rb");
                if (f) {
                    fseek(f, 0, SEEK_END);
                    sizes[i] = ftell(f);
                    fclose(f);
                }
            } catch (const std::exception& e) {
                errors[i] = e.what();
            }
        });
    }
    for (auto& t : threads) t.join();

    for (int64_t i = 0; i < num_frames; i++) {
        TORCH_CHECK(errors[i].empty(),
                    "Failed to encode frame ", i, " with ", codec_name, ": ", errors[i]);
    }

    int64_t total = 0;
    for (auto s : sizes) total += s;
    return total;
}


// ============================================================
// DIRECT H.265 DECODE via libavcodec (avoids PyAV Python overhead)
// ============================================================

/**
 * decode_h265_frame_from_file - Decode a single H.265 frame from a file.
 *
 * Returns: torch::Tensor (height, width) uint8, the decoded grayscale frame.
 *
 * This is ~2-5ms faster than PyAV for single-frame decode because:
 * 1. No Python interpreter overhead in the decode loop
 * 2. Direct memory copy from AVFrame to torch tensor (no numpy intermediate)
 * 3. No Python GIL contention
 */
torch::Tensor decode_h265_frame_from_file(const std::string& path) {
    // Open input file
    AVFormatContext* fmt_ctx = nullptr;
    int ret = avformat_open_input(&fmt_ctx, path.c_str(), nullptr, nullptr);
    TORCH_CHECK(ret >= 0, "Failed to open file: ", path);

    ret = avformat_find_stream_info(fmt_ctx, nullptr);
    TORCH_CHECK(ret >= 0, "Failed to find stream info");

    // Find video stream
    int video_stream = -1;
    for (unsigned i = 0; i < fmt_ctx->nb_streams; i++) {
        if (fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream = i;
            break;
        }
    }
    TORCH_CHECK(video_stream >= 0, "No video stream found");

    AVCodecParameters* codecpar = fmt_ctx->streams[video_stream]->codecpar;
    const AVCodec* codec = avcodec_find_decoder(codecpar->codec_id);
    TORCH_CHECK(codec != nullptr, "Codec not found");

    AVCodecContext* codec_ctx = avcodec_alloc_context3(codec);
    TORCH_CHECK(codec_ctx != nullptr, "Failed to alloc codec context");
    avcodec_parameters_to_context(codec_ctx, codecpar);

    // Use auto-threading (thread_count=0 lets libavcodec choose optimal count)
    // For H.265 slice-based threading: splits frame into horizontal slices
    codec_ctx->thread_count = 0;
    codec_ctx->thread_type = FF_THREAD_SLICE;

    ret = avcodec_open2(codec_ctx, codec, nullptr);
    TORCH_CHECK(ret >= 0, "Failed to open codec");

    AVPacket* pkt = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();

    // Read and decode first frame
    torch::Tensor result;
    bool decoded = false;

    while (av_read_frame(fmt_ctx, pkt) >= 0 && !decoded) {
        if (pkt->stream_index == video_stream) {
            ret = avcodec_send_packet(codec_ctx, pkt);
            if (ret >= 0) {
                ret = avcodec_receive_frame(codec_ctx, frame);
                if (ret >= 0) {
                    int h = frame->height;
                    int w = frame->width;

                    // Allocate output tensor
                    result = torch::empty({h, w}, torch::kUInt8);
                    uint8_t* dst = result.data_ptr<uint8_t>();

                    // Copy frame data (handle linesize != width for padded frames)
                    if (frame->linesize[0] == w) {
                        std::memcpy(dst, frame->data[0], h * w);
                    } else {
                        for (int y = 0; y < h; y++) {
                            std::memcpy(dst + y * w,
                                       frame->data[0] + y * frame->linesize[0], w);
                        }
                    }
                    decoded = true;
                }
            }
        }
        av_packet_unref(pkt);
    }

    // Flush decoder
    if (!decoded) {
        avcodec_send_packet(codec_ctx, nullptr);
        ret = avcodec_receive_frame(codec_ctx, frame);
        if (ret >= 0) {
            int h = frame->height;
            int w = frame->width;
            result = torch::empty({h, w}, torch::kUInt8);
            uint8_t* dst = result.data_ptr<uint8_t>();
            if (frame->linesize[0] == w) {
                std::memcpy(dst, frame->data[0], h * w);
            } else {
                for (int y = 0; y < h; y++) {
                    std::memcpy(dst + y * w,
                               frame->data[0] + y * frame->linesize[0], w);
                }
            }
            decoded = true;
        }
    }

    av_frame_free(&frame);
    av_packet_free(&pkt);
    avcodec_free_context(&codec_ctx);
    avformat_close_input(&fmt_ctx);

    TORCH_CHECK(decoded, "Failed to decode any frame from: ", path);
    return result;
}

/**
 * decode_h265_frame_from_bytes - Decode a single H.265 frame from in-memory bytes.
 *
 * Takes a uint8 tensor of compressed bytes and returns (height, width) uint8 frame.
 * Uses AVIOContext with custom read callback to avoid writing to disk.
 */
struct MemoryBuffer {
    const uint8_t* data;
    int64_t size;
    int64_t pos;
};

static int mem_read_packet(void* opaque, uint8_t* buf, int buf_size) {
    MemoryBuffer* mb = static_cast<MemoryBuffer*>(opaque);
    int64_t remaining = mb->size - mb->pos;
    if (remaining <= 0) return AVERROR_EOF;
    int to_read = std::min(static_cast<int64_t>(buf_size), remaining);
    std::memcpy(buf, mb->data + mb->pos, to_read);
    mb->pos += to_read;
    return to_read;
}

static int64_t mem_seek(void* opaque, int64_t offset, int whence) {
    MemoryBuffer* mb = static_cast<MemoryBuffer*>(opaque);
    if (whence == AVSEEK_SIZE) return mb->size;
    if (whence == SEEK_SET) mb->pos = offset;
    else if (whence == SEEK_CUR) mb->pos += offset;
    else if (whence == SEEK_END) mb->pos = mb->size + offset;
    return mb->pos;
}

torch::Tensor decode_h265_frame_from_bytes(const torch::Tensor& compressed_bytes) {
    TORCH_CHECK(compressed_bytes.scalar_type() == torch::kUInt8,
                "compressed_bytes must be uint8");
    auto cdata = compressed_bytes.contiguous();
    const uint8_t* data = cdata.data_ptr<uint8_t>();
    int64_t data_size = cdata.numel();

    // Set up memory I/O
    MemoryBuffer mb = {data, data_size, 0};
    const int avio_buf_size = 32768;
    uint8_t* avio_buf = static_cast<uint8_t*>(av_malloc(avio_buf_size));
    AVIOContext* avio_ctx = avio_alloc_context(
        avio_buf, avio_buf_size, 0, &mb, mem_read_packet, nullptr, mem_seek);

    AVFormatContext* fmt_ctx = avformat_alloc_context();
    fmt_ctx->pb = avio_ctx;

    int ret = avformat_open_input(&fmt_ctx, nullptr, nullptr, nullptr);
    TORCH_CHECK(ret >= 0, "Failed to open in-memory stream");

    ret = avformat_find_stream_info(fmt_ctx, nullptr);
    TORCH_CHECK(ret >= 0, "Failed to find stream info");

    int video_stream = -1;
    for (unsigned i = 0; i < fmt_ctx->nb_streams; i++) {
        if (fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream = i;
            break;
        }
    }
    TORCH_CHECK(video_stream >= 0, "No video stream found");

    AVCodecParameters* codecpar = fmt_ctx->streams[video_stream]->codecpar;
    const AVCodec* codec = avcodec_find_decoder(codecpar->codec_id);
    TORCH_CHECK(codec != nullptr, "Codec not found");

    AVCodecContext* codec_ctx = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(codec_ctx, codecpar);
    codec_ctx->thread_count = 0;
    codec_ctx->thread_type = FF_THREAD_SLICE;
    ret = avcodec_open2(codec_ctx, codec, nullptr);
    TORCH_CHECK(ret >= 0, "Failed to open codec");

    AVPacket* pkt = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();

    torch::Tensor result;
    bool decoded = false;

    while (av_read_frame(fmt_ctx, pkt) >= 0 && !decoded) {
        if (pkt->stream_index == video_stream) {
            ret = avcodec_send_packet(codec_ctx, pkt);
            if (ret >= 0) {
                ret = avcodec_receive_frame(codec_ctx, frame);
                if (ret >= 0) {
                    int h = frame->height;
                    int w = frame->width;
                    result = torch::empty({h, w}, torch::kUInt8);
                    uint8_t* dst = result.data_ptr<uint8_t>();
                    if (frame->linesize[0] == w) {
                        std::memcpy(dst, frame->data[0], h * w);
                    } else {
                        for (int y = 0; y < h; y++) {
                            std::memcpy(dst + y * w,
                                       frame->data[0] + y * frame->linesize[0], w);
                        }
                    }
                    decoded = true;
                }
            }
        }
        av_packet_unref(pkt);
    }

    if (!decoded) {
        avcodec_send_packet(codec_ctx, nullptr);
        ret = avcodec_receive_frame(codec_ctx, frame);
        if (ret >= 0) {
            int h = frame->height;
            int w = frame->width;
            result = torch::empty({h, w}, torch::kUInt8);
            uint8_t* dst = result.data_ptr<uint8_t>();
            if (frame->linesize[0] == w) {
                std::memcpy(dst, frame->data[0], h * w);
            } else {
                for (int y = 0; y < h; y++) {
                    std::memcpy(dst + y * w,
                               frame->data[0] + y * frame->linesize[0], w);
                }
            }
            decoded = true;
        }
    }

    av_frame_free(&frame);
    av_packet_free(&pkt);
    avcodec_free_context(&codec_ctx);
    avformat_close_input(&fmt_ctx);
    av_freep(&avio_ctx->buffer);
    avio_context_free(&avio_ctx);

    TORCH_CHECK(decoded, "Failed to decode frame from bytes");
    return result;
}

/**
 * decode_h265_gather_dequant - Decode H.265 frame and gather+dequant specific rows.
 *
 * This is the fused decode+gather+dequant path that combines:
 * 1. H.265 decode from file → tiled (H,W) frame
 * 2. C++ gather specific rows from tiled layout
 * 3. Dequantize to fp32
 *
 * All in one C++ call, avoiding Python/PyAV overhead entirely.
 */
torch::Tensor decode_h265_gather_dequant(
    const std::string& path,
    const torch::Tensor& row_indices,
    int64_t tiles_per_row,
    double scale, int64_t zero_point)
{
    // Step 1: decode
    torch::Tensor tiled_frame = decode_h265_frame_from_file(path);

    // Step 2+3: gather + dequant (reuse existing function)
    return gather_dequant_from_tiled_frame(tiled_frame, row_indices, tiles_per_row,
                                            scale, zero_point);
}

// Forward declaration for auto-detect (defined near batch_decode_frames)
static std::string auto_detect_frame_ext(const std::string& frame_dir);

/**
 * batch_decode_gather_dequant - Decode multiple frames in parallel and gather+dequant.
 *
 * This is the key optimization for batch inference: when a batch needs rows from
 * N different frames, we decode all N frames in parallel threads and then gather
 * the needed rows with dequantization.
 *
 * Args:
 *   frame_dir: directory containing frame_XXXXX.h265 files
 *   frame_ids: tensor of frame IDs to decode (unique)
 *   all_cold_indices: tensor of all cold reordered indices for this batch
 *   rows_per_frame: number of embedding rows per frame
 *   tiles_per_row: tiles per row in the tiled frame
 *   scale, zero_point: dequantization parameters
 *
 * Returns: fp32 tensor (len(all_cold_indices), 16) of dequantized embeddings,
 *          in the same order as all_cold_indices.
 */
torch::Tensor batch_decode_gather_dequant(
    const std::string& frame_dir,
    const torch::Tensor& frame_ids,     // unique frame IDs
    const torch::Tensor& cold_indices,   // cold reordered indices (not sorted requirement)
    int64_t rows_per_frame,
    int64_t tiles_per_row,
    double scale, int64_t zero_point)
{
    const int64_t num_frames = frame_ids.size(0);
    const int64_t num_indices = cold_indices.size(0);
    const int64_t D = 16;
    const auto fid_ptr = frame_ids.data_ptr<int64_t>();
    const auto idx_ptr = cold_indices.data_ptr<int64_t>();

    // Step 1: Decode all frames in parallel (auto-detect codec from file extension)
    std::string ext = auto_detect_frame_ext(frame_dir);
    std::vector<torch::Tensor> decoded_frames(num_frames);

    // Use std::thread for parallel decode (at::parallel_for may not work well with
    // blocking I/O operations like file read + codec decode)
    std::vector<std::thread> threads;
    std::vector<std::string> errors(num_frames);

    for (int64_t i = 0; i < num_frames; i++) {
        threads.emplace_back([&, i]() {
            try {
                char fname[64];
                snprintf(fname, sizeof(fname), "frame_%05ld%s", fid_ptr[i], ext.c_str());
                std::string path = frame_dir + "/" + fname;
                decoded_frames[i] = decode_h265_frame_from_file(path);
            } catch (const std::exception& e) {
                errors[i] = e.what();
            }
        });
    }
    for (auto& t : threads) t.join();

    // Check for errors
    for (int64_t i = 0; i < num_frames; i++) {
        TORCH_CHECK(errors[i].empty(),
                    "Failed to decode frame ", fid_ptr[i], ": ", errors[i]);
    }

    // Step 2: Build a map from frame_id to decoded frame index
    // (frame_ids may not be contiguous)
    int64_t max_fid = 0;
    for (int64_t i = 0; i < num_frames; i++) {
        max_fid = std::max(max_fid, fid_ptr[i]);
    }
    std::vector<int64_t> fid_to_idx(max_fid + 1, -1);
    for (int64_t i = 0; i < num_frames; i++) {
        fid_to_idx[fid_ptr[i]] = i;
    }

    // Step 3: Gather + dequant for all indices
    auto output = torch::zeros({num_indices, D});
    float* out_ptr = output.data_ptr<float>();
    float fscale = static_cast<float>(scale);
    float fzp = static_cast<float>(zero_point);

    // Get frame data pointers
    struct FrameInfo {
        const uint8_t* data;
        int64_t width;
    };
    std::vector<FrameInfo> frame_infos(num_frames);
    for (int64_t i = 0; i < num_frames; i++) {
        frame_infos[i].data = decoded_frames[i].data_ptr<uint8_t>();
        frame_infos[i].width = decoded_frames[i].size(1);
    }

    at::parallel_for(0, num_indices, 64, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; i++) {
            int64_t cold_idx = idx_ptr[i];
            int64_t fid = cold_idx / rows_per_frame;
            int64_t row_in_frame = cold_idx % rows_per_frame;

            // Find the decoded frame
            TORCH_CHECK(fid <= max_fid && fid_to_idx[fid] >= 0,
                        "Frame ID ", fid, " not in decoded set");
            int64_t fidx = fid_to_idx[fid];
            const uint8_t* frame_data = frame_infos[fidx].data;
            int64_t width = frame_infos[fidx].width;

            // Compute tile position
            int64_t ty = row_in_frame / tiles_per_row;
            int64_t tx = row_in_frame % tiles_per_row;

            // Gather 4x4 tile and dequantize
            float* dst = out_ptr + i * D;
            for (int ly = 0; ly < 4; ly++) {
                const uint8_t* src = frame_data + (ty * 4 + ly) * width + tx * 4;
#ifdef __AVX512F__
                // Can't use full AVX-512 for 4 bytes, use scalar
#endif
                for (int lx = 0; lx < 4; lx++) {
                    dst[ly * 4 + lx] = (static_cast<float>(src[lx]) - fzp) * fscale;
                }
            }
        }
    });

    return output;
}


/**
 * auto_detect_frame_ext - Detect frame file extension in a directory.
 *
 * Checks for frame_00000 with extensions: .h265, .h264, .mkv
 * Returns the detected extension string, defaults to ".h265".
 */
static std::string auto_detect_frame_ext(const std::string& frame_dir) {
    const char* exts[] = {".h265", ".h264", ".mkv"};
    for (const char* ext : exts) {
        std::string path = frame_dir + "/frame_00000" + ext;
        FILE* f = fopen(path.c_str(), "rb");
        if (f) {
            fclose(f);
            return ext;
        }
    }
    return ".h265";  // default
}

/**
 * batch_decode_frames - Decode multiple frames in parallel, return tiled frames.
 *
 * Auto-detects codec from file extension (.h265, .h264, .mkv).
 * Useful when caller wants to cache the decoded frames.
 * Returns: vector of torch::Tensor, each (H, W) uint8.
 */
std::vector<torch::Tensor> batch_decode_frames(
    const std::string& frame_dir,
    const torch::Tensor& frame_ids)
{
    const int64_t num_frames = frame_ids.size(0);
    const auto fid_ptr = frame_ids.data_ptr<int64_t>();

    // Auto-detect file extension
    std::string ext = auto_detect_frame_ext(frame_dir);

    std::vector<torch::Tensor> decoded_frames(num_frames);
    std::vector<std::thread> threads;
    std::vector<std::string> errors(num_frames);

    for (int64_t i = 0; i < num_frames; i++) {
        threads.emplace_back([&, i]() {
            try {
                char fname[64];
                snprintf(fname, sizeof(fname), "frame_%05ld%s", fid_ptr[i], ext.c_str());
                std::string path = frame_dir + "/" + fname;
                decoded_frames[i] = decode_h265_frame_from_file(path);
            } catch (const std::exception& e) {
                errors[i] = e.what();
            }
        });
    }
    for (auto& t : threads) t.join();

    for (int64_t i = 0; i < num_frames; i++) {
        TORCH_CHECK(errors[i].empty(),
                    "Failed to decode frame ", fid_ptr[i], ": ", errors[i]);
    }

    return decoded_frames;
}


// scan_needed_frames: given batch indices, find which cold frames are needed per table.
// Replaces the Python loop over compressed tables with a single fused C++ call.
// Returns a list of K int64 tensors (one per compressed table), each containing
// the unique frame IDs needed for this batch.
std::vector<torch::Tensor> scan_needed_frames(
    const std::vector<torch::Tensor>& lS_i_list,      // K tensors, each [N] int64
    const std::vector<torch::Tensor>& is_hot_list,     // K bool tensors, each [num_emb] bool
    const std::vector<torch::Tensor>& o2c_map_list,    // K int32 tensors, each [num_emb] int32
    int64_t rows_per_frame
) {
    const int64_t K = lS_i_list.size();
    TORCH_CHECK((int64_t)is_hot_list.size() == K, "is_hot_list size mismatch");
    TORCH_CHECK((int64_t)o2c_map_list.size() == K, "o2c_map_list size mismatch");

    std::vector<torch::Tensor> results(K);

    // Process tables in parallel
    at::parallel_for(0, K, 1, [&](int64_t k_begin, int64_t k_end) {
        for (int64_t k = k_begin; k < k_end; k++) {
            const auto& indices = lS_i_list[k];
            const auto& is_hot = is_hot_list[k];
            const auto& o2c = o2c_map_list[k];

            const int64_t N = indices.size(0);
            const int64_t* idx_ptr = indices.data_ptr<int64_t>();
            const bool* hot_ptr = is_hot.data_ptr<bool>();
            const int32_t* o2c_ptr = o2c.data_ptr<int32_t>();

            // Collect unique frame IDs using a small set
            std::unordered_set<int64_t> frame_set;
            for (int64_t i = 0; i < N; i++) {
                int64_t idx = idx_ptr[i];
                if (!hot_ptr[idx]) {
                    int32_t cold_idx = o2c_ptr[idx];
                    if (cold_idx >= 0) {
                        int64_t fid = static_cast<int64_t>(cold_idx) / rows_per_frame;
                        frame_set.insert(fid);
                    }
                }
            }

            if (frame_set.empty()) {
                results[k] = torch::empty({0}, torch::dtype(torch::kInt64));
            } else {
                auto out = torch::empty({(int64_t)frame_set.size()}, torch::dtype(torch::kInt64));
                int64_t* out_ptr = out.data_ptr<int64_t>();
                int64_t j = 0;
                for (int64_t fid : frame_set) {
                    out_ptr[j++] = fid;
                }
                results[k] = out;
            }
        }
    });

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
    // Frame packing/unpacking optimizations
    m.def("tile_rows_to_frame", &tile_rows_to_frame,
          "Tile uint8 embedding rows (N,16) into a 2D frame (H,W) for H.265 encoding");
    m.def("untile_frame_to_rows", &untile_frame_to_rows,
          "Untile a 2D frame (H,W) back to embedding rows (N,16)");
    m.def("gather_from_tiled_frame", &gather_from_tiled_frame,
          "Gather specific rows from a tiled frame without untiling the whole frame");
    m.def("gather_dequant_from_tiled_frame", &gather_dequant_from_tiled_frame,
          "Gather rows from tiled frame and dequantize to fp32 in one pass");
    m.def("fused_gather_quantize_tile", &fused_gather_quantize_tile,
          "Fused: gather scattered fp32 rows + quantize + tile into frame");
    m.def("fused_quantize_tile", &fused_quantize_tile,
          "Fused: quantize contiguous fp32 rows + tile into frame");
    m.def("fused_quantize_tile_multiframe", &fused_quantize_tile_multiframe,
          "Tile pre-quantized uint8 rows across multiple frames");
    m.def("fused_quantize_tile_multiframe_fp32", &fused_quantize_tile_multiframe_fp32,
          "Fused quantize + tile fp32 rows across multiple frames");
    m.def("fused_gather_quantize_tile_multiframe", &fused_gather_quantize_tile_multiframe,
          "Fused gather scattered fp32 + quantize + tile across multiple frames");
    m.def("frame_to_bytes", &frame_to_bytes,
          "Get frame as contiguous 1D bytes (avoids Python .tobytes())");
    // Direct H.265 decode (avoids PyAV Python overhead)
    m.def("decode_h265_frame_from_file", &decode_h265_frame_from_file,
          "Decode single H.265 frame from file, returns (H,W) uint8 tensor");
    m.def("decode_h265_frame_from_bytes", &decode_h265_frame_from_bytes,
          "Decode single H.265 frame from in-memory bytes, returns (H,W) uint8 tensor");
    m.def("decode_h265_gather_dequant", &decode_h265_gather_dequant,
          "Fused: decode H.265 from file + gather specific rows + dequant to fp32");
    m.def("batch_decode_gather_dequant", &batch_decode_gather_dequant,
          "Batch: decode multiple H.265 frames in parallel + vectorized gather + dequant");
    m.def("batch_decode_frames", &batch_decode_frames,
          "Decode multiple H.265 frames in parallel, return list of tiled (H,W) tensors");
    // Direct H.265 encode (avoids subprocess ffmpeg overhead)
    m.def("encode_h265_frame", &encode_h265_frame,
          "Encode single grayscale frame to H.265 (in-memory or file)",
          py::arg("frame"), py::arg("output_path") = "",
          py::arg("lossless") = true, py::arg("crf") = 0);
    m.def("encode_h265_frame_to_file", &encode_h265_frame_to_file,
          "Encode single grayscale frame to H.265 file");
    m.def("batch_encode_h265_frames", &batch_encode_h265_frames,
          "Encode multiple frames to H.265 files in parallel",
          py::arg("frames"), py::arg("output_dir"), py::arg("lossless") = true);
    // Multi-codec encode
    m.def("encode_frame_codec", &encode_frame_codec,
          "Encode single grayscale frame with specified codec (h265/h264/ffv1)",
          py::arg("frame"), py::arg("output_path") = "",
          py::arg("codec_name") = "h265",
          py::arg("lossless") = true, py::arg("crf") = 0);
    m.def("batch_encode_frames_codec", &batch_encode_frames_codec,
          "Encode multiple frames in parallel with specified codec",
          py::arg("frames"), py::arg("output_dir"),
          py::arg("codec_name") = "h265", py::arg("lossless") = true);
    m.def("encode_frame_with_params", &encode_frame_with_params,
          "Encode single frame with custom codec params (e.g., slices, wpp)",
          py::arg("frame"), py::arg("output_path"),
          py::arg("codec_name") = "h265", py::arg("lossless") = true,
          py::arg("crf") = 0, py::arg("extra_params") = "");
    // Fused scan for needed frames
    m.def("scan_needed_frames", &scan_needed_frames,
          "Scan batch indices to find needed cold frames per table (fused C++)",
          py::arg("lS_i_list"), py::arg("is_hot_list"),
          py::arg("o2c_map_list"), py::arg("rows_per_frame"));
}
