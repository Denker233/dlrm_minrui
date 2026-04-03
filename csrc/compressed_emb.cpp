// compressed_emb.cpp — C++ PyTorch extension for CompressedEmbeddingBag forward pass
// Replaces Python hot/cold routing + scatter_add with efficient C++ implementation.

#include <torch/extension.h>
#include <vector>
#include <cstring>
#include <algorithm>
#include <atomic>
#include <mutex>
#include <queue>
#include <thread>
#include <future>
#include <unordered_map>
#include <unordered_set>
#include <unistd.h>

// FFmpeg/libav headers for direct H.265 decode (avoids PyAV Python overhead)
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
}

#include <zstd.h>

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
enum class TableKind { STANDARD, COMPRESSED_FP32, COMPRESSED_Q8, DCT_DOMAIN };

// ====================================================================
// DCT-Domain Embedding: compressed-domain lookup without full decode.
//
// Storage: per block (4 embedding rows), store only DC coefficient (int16).
// At step>=8 on CRF=30-equivalent quality, 100% of blocks are DC-only.
// Lookup: output[d] += DC_value * step_size * weight[row_in_block][d]
// Pre-computed weights: IDCT basis functions for each (row_in_block, dim).
// ====================================================================
static constexpr int DCT_BLOCK = 8;
static constexpr int DCT_TILE = 4;
static constexpr int DCT_ROWS_PER_BLOCK = 4;  // (8/4)^2

struct DctDomainTable {
    // DC values: one uint8 per block (quantized DC coefficient, fits in 0-255 for step>=8)
    std::vector<uint8_t> dc_values;           // (n_blocks,) — quantized DC coefficient per block
    std::vector<std::vector<std::pair<uint8_t, int16_t>>> ac_coeffs;  // per-block sparse AC
    int64_t n_blocks = 0;
    int64_t n_rows = 0;
    int block_size = 8;           // 4, 8, or 16
    int rows_per_block = 4;       // 1, 4, or 16
    float step_size = 16.0f;
    float quant_scale = 0.0f;     // uint8 dequant scale
    float quant_zp = 0.0f;        // uint8 dequant zero point

    // DC weight: uniform for ortho-normalized DCT
    // For NxN block: C(0) = 1/sqrt(N), dc_weight = C(0)^2 = 1/N
    float dc_weight_uniform;      // = 1/block_size

    // Full weight matrix for AC coefficients: weights[row_in_block * EMB_DIM + d][u * 8 + v]
    // Flattened for cache-friendly access
    std::vector<float> full_weights;  // (4 * 16, 64) = (64, 64) = 4096 floats = 16KB

    void init_weights(int D, int blk_size, int rpb) {
        block_size = blk_size;
        rows_per_block = rpb;
        dc_weight_uniform = 1.0f / block_size;  // C(0)^2 = 1/N

        // Full weights for AC coefficients
        int BS = block_size;
        full_weights.resize(rpb * D * BS * BS);
        for (int r = 0; r < rpb; r++) {
            for (int d = 0; d < D; d++) {
                int py, px;
                if (BS == 4) {
                    // 4×4: 1 row per block, row IS the 4×4 block
                    py = d / DCT_TILE;
                    px = d % DCT_TILE;
                } else if (BS == 8) {
                    // 8×8: 2×2 grid of 4×4 tiles
                    py = (r / 2) * DCT_TILE + d / DCT_TILE;
                    px = (r % 2) * DCT_TILE + d % DCT_TILE;
                } else {
                    // 16×16: 4×4 grid of 4×4 tiles
                    py = (r / 4) * DCT_TILE + d / DCT_TILE;
                    px = (r % 4) * DCT_TILE + d % DCT_TILE;
                }
                for (int u = 0; u < BS; u++) {
                    for (int v = 0; v < BS; v++) {
                        float cu = (u == 0) ? (1.0f / std::sqrt((float)BS))
                                             : std::sqrt(2.0f / BS);
                        float cv = (v == 0) ? (1.0f / std::sqrt((float)BS))
                                             : std::sqrt(2.0f / BS);
                        float w = cu * cv
                            * std::cos(M_PI * (2*py + 1) * u / (2.0f * BS))
                            * std::cos(M_PI * (2*px + 1) * v / (2.0f * BS));
                        int idx = (r * D + d) * (BS * BS) + u * BS + v;
                        full_weights[idx] = w;
                    }
                }
            }
        }
    }

    // Accumulate contribution of one cold row into output (D floats)
    // DC-only fast path: ~D multiply-adds
    inline void accum_row(float* __restrict__ out, int64_t cold_row_idx, int D) const {
        int64_t block_id = cold_row_idx / rows_per_block;
        int row_in_block = cold_row_idx % rows_per_block;

        if (block_id >= n_blocks) return;

        float dc_val = static_cast<float>(dc_values[block_id]) * step_size;  // uint8 * step

        // DC contribution: dc_val * dc_weight_uniform for each pixel in this row
        // But dc_weight is the same for all pixels (0.125), so:
        float dc_contribution = dc_val * dc_weight_uniform;

        // For DC-only blocks (common case), just add dc_contribution to each dim
        // But this gives the uint8 domain value. We need to dequantize:
        // fp32_val = (uint8_val - quant_zp) * quant_scale
        // Each dim gets dc_contribution, then sum is: n_dims_in_row × dc_contribution... no.
        // For SUM mode: output[d] += (pixel_value_at(row, d) - quant_zp) * quant_scale
        // pixel_value = dc_val * weight[row_in_block, d, 0, 0] + sum_ac(...)
        // For DC-only: pixel_value = dc_val * dc_weight_uniform (same for all d)
        // But wait — the weight is NOT uniform across dimensions because each dimension
        // maps to a DIFFERENT pixel position (py, px).
        //
        // For u=0, v=0: basis(u=0,v=0,x,y) = C(0)*C(0)*cos(0)*cos(0) = 1/8
        // This IS uniform — every pixel gets DC * 1/8 regardless of position.
        // So for DC-only: every dimension of this row = dc_val * 1/8.

        if (block_id < (int64_t)ac_coeffs.size() && !ac_coeffs[block_id].empty()) {
            // Has AC coefficients — use full weight lookup
            int BS2 = block_size * block_size;
            const float* w_base = &full_weights[(row_in_block * D) * BS2];
            for (int d = 0; d < D; d++) {
                float val = dc_contribution;  // DC part
                const float* w_d = w_base + d * BS2;
                // Add AC contributions
                for (const auto& [pos, coeff] : ac_coeffs[block_id]) {
                    val += static_cast<float>(coeff) * step_size * w_d[pos];
                }
                out[d] += (val - quant_zp) * quant_scale;
            }
        } else {
            // DC-only: all dimensions get the same uint8 value
            float uint8_val = dc_contribution;
            float fp32_val = (uint8_val - quant_zp) * quant_scale;
#if HAS_AVX512
            if (D == 16) {
                __m512 v = _mm512_set1_ps(fp32_val);
                __m512 o = _mm512_loadu_ps(out);
                _mm512_storeu_ps(out, _mm512_add_ps(o, v));
            } else
#endif
            {
                for (int d = 0; d < D; d++) out[d] += fp32_val;
            }
        }
    }
};

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

    // Sparse flat mode: only a subset of cold rows are cached (LRU after warmup)
    // Uses a second bitmap+rank to map cold_rank -> dense buffer position
    bool cold_sparse = false;
    std::vector<uint64_t> cold_valid_bitmap;  // 1 = cached, indexed by cold_rank
    std::vector<int32_t> cold_valid_rank;     // prefix popcount for O(1) dense index
    int64_t n_cold_total = 0;   // total cold rows (bitmap size)
    int64_t n_cold_cached = 0;  // number of cached rows (dense buffer size)

    // Dynamic frame mode: per-frame pointers for O(1) add/remove (pipelined decode)
    // cold_frame_accum uses this when cold_dynamic=true
    bool cold_dynamic = false;
    bool dyn_tiled = false;    // if true, frames stored in tiled (H,W) layout (skip untiling)
    int64_t dyn_width = 0;     // frame width (for tile coordinate computation)
    std::vector<const uint8_t*> dyn_frame_ptrs;  // frame_id -> pointer to frame data (null = not cached)
    std::vector<torch::Tensor> dyn_frame_tensors; // keep alive (frame_id -> tensor)

    // Row-level cache: cache individual cold rows, not entire frames.
    // Memory: ~80KB for Kaggle (vs 40MB for frame cache)
    // Uses open-addressing hash table for cache-friendly O(1) lookup.
    bool cold_row_cache = false;
    torch::Tensor rc_data;             // (capacity, D) uint8 — row data
    uint8_t* rc_data_ptr = nullptr;
    int32_t rc_next_slot = 0;          // next free slot in rc_data
    int32_t rc_data_capacity = 0;      // max rows in rc_data

    // Open-addressing hash table: cold_rank → slot in rc_data
    // Power-of-2 size, linear probing, contiguous memory (cache-friendly)
    std::vector<int64_t> rc_ht_keys;   // -1 = empty
    std::vector<int32_t> rc_ht_vals;   // slot in rc_data
    int64_t rc_ht_mask = 0;            // ht_size - 1
    int64_t rc_ht_count = 0;           // number of entries

    inline int64_t rc_hash(int64_t key) const {
        // Mix bits for better distribution
        uint64_t h = static_cast<uint64_t>(key);
        h ^= h >> 16;
        h *= 0x45d9f3b;
        h ^= h >> 16;
        return static_cast<int64_t>(h & rc_ht_mask);
    }

    inline int32_t rc_find(int64_t cold_rank) const {
        if (rc_ht_mask == 0) return -1;
        int64_t h = rc_hash(cold_rank);
        for (int64_t probe = 0; probe <= rc_ht_mask; probe++) {
            if (rc_ht_keys[h] == cold_rank) return rc_ht_vals[h];
            if (rc_ht_keys[h] == -1) return -1;
            h = (h + 1) & rc_ht_mask;
        }
        return -1;  // table full (shouldn't happen)
    }

    inline void rc_insert(int64_t cold_rank, int32_t slot) {
        // Rehash if load > 50%
        if (rc_ht_count * 2 >= (int64_t)rc_ht_keys.size()) {
            int64_t new_size = std::max((int64_t)rc_ht_keys.size() * 2, (int64_t)1024);
            std::vector<int64_t> old_keys = std::move(rc_ht_keys);
            std::vector<int32_t> old_vals = std::move(rc_ht_vals);
            rc_ht_keys.assign(new_size, -1);
            rc_ht_vals.resize(new_size);
            rc_ht_mask = new_size - 1;
            rc_ht_count = 0;
            for (size_t i = 0; i < old_keys.size(); i++) {
                if (old_keys[i] != -1) {
                    rc_insert(old_keys[i], old_vals[i]);
                }
            }
        }
        int64_t h = rc_hash(cold_rank);
        while (rc_ht_keys[h] != -1) {
            h = (h + 1) & rc_ht_mask;
        }
        rc_ht_keys[h] = cold_rank;
        rc_ht_vals[h] = slot;
        rc_ht_count++;
    }

    inline bool rc_contains(int64_t cold_rank) const {
        return rc_find(cold_rank) >= 0;
    }

    // DCT-domain cold lookup (no frame decode needed)
    DctDomainTable dct_cold;
    bool has_dct_cold = false;
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

// register_cold_sparse_flat: register a sparse cold buffer for a table.
// Only a subset of cold rows (those whose frames are cached) are stored.
// Uses a validity bitmap + rank for O(1) lookup: cold_rank -> dense buffer position.
// valid_cold_ranks: [n_cached] int64 — which cold_rank positions have data (sorted ascending)
// cold_data: [n_cached, D] uint8 — the actual cached row data (dense, in valid_cold_ranks order)
void register_cold_sparse_flat(
    int64_t table_idx,
    const torch::Tensor& cold_data,
    const torch::Tensor& valid_cold_ranks,
    double cold_scale,
    double cold_zp,
    int64_t n_cold_total
) {
    TORCH_CHECK(g_registered, "Tables not registered. Call register_tables first.");
    TORCH_CHECK(table_idx >= 0 && table_idx < (int64_t)g_tables.size(),
                "Invalid table index: ", table_idx);

    auto& tab = g_tables[table_idx];
    tab.has_cold_frames = true;
    tab.cold_flat = true;
    tab.cold_sparse = true;
    tab.n_cold_total = n_cold_total;
    tab.n_cold_cached = valid_cold_ranks.size(0);
    tab.cold_frame_concat = cold_data.contiguous();
    tab.cold_frame_data_ptr = tab.cold_frame_concat.data_ptr<uint8_t>();
    tab.cold_scale = static_cast<float>(cold_scale);
    tab.cold_zp = static_cast<float>(cold_zp);
    // Also set n_cold_rows for compatibility with dense flat path checks
    tab.n_cold_rows = n_cold_total;

    // Build validity bitmap + rank from valid_cold_ranks
    int64_t n_words = (n_cold_total + 63) / 64;
    tab.cold_valid_bitmap.assign(n_words, 0);
    tab.cold_valid_rank.resize(n_words + 1);

    const int64_t* vr_ptr = valid_cold_ranks.data_ptr<int64_t>();
    for (int64_t i = 0; i < tab.n_cold_cached; i++) {
        int64_t cr = vr_ptr[i];
        tab.cold_valid_bitmap[cr / 64] |= (1ULL << (cr % 64));
    }
    tab.cold_valid_rank[0] = 0;
    for (int64_t k = 0; k < n_words; k++) {
        tab.cold_valid_rank[k + 1] = tab.cold_valid_rank[k] +
            __builtin_popcountll(tab.cold_valid_bitmap[k]);
    }

    int64_t data_mb = cold_data.numel() / (1024 * 1024);
    int64_t bm_kb = (n_words * 8 + (n_words + 1) * 4) / 1024;
    fprintf(stderr, "[C++] Table %ld: registered sparse flat cold buffer "
            "(%ld/%ld cached rows, %ld MB uint8, %ld KB bitmap+rank), "
            "scale=%.6f, zp=%.1f\n",
            table_idx, tab.n_cold_cached, n_cold_total, data_mb, bm_kb,
            tab.cold_scale, tab.cold_zp);
}

// register_cold_dct: register DCT-domain cold data for a table.
// dc_values: (n_blocks,) int16 — DC coefficient per block
// ac_positions: list of (n_blocks) tensors, each (n_ac,) uint8 — position of non-zero AC coeffs
// ac_values: list of (n_blocks) tensors, each (n_ac,) int16 — values of non-zero AC coeffs
// For DC-only mode (step>=8 on most data): ac lists can be empty.
void register_cold_dct(
    int64_t table_idx,
    const torch::Tensor& dc_values_t,     // (n_blocks,) int16
    const torch::Tensor& ac_data_t,       // (total_ac,) int16 — concatenated AC values
    const torch::Tensor& ac_positions_t,  // (total_ac,) uint8 — concatenated AC positions
    const torch::Tensor& ac_block_offsets_t, // (n_blocks+1,) int64 — start offset per block in ac_data
    double step_size,
    double quant_scale,
    double quant_zp,
    int64_t n_cold_rows,
    int64_t block_size_param = 8   // 4, 8, or 16
) {
    TORCH_CHECK(g_registered, "Tables not registered. Call register_tables first.");
    TORCH_CHECK(table_idx >= 0 && table_idx < (int64_t)g_tables.size(),
                "Invalid table index: ", table_idx);

    auto& tab = g_tables[table_idx];
    auto& dct = tab.dct_cold;

    int64_t n_blocks = dc_values_t.size(0);
    dct.n_blocks = n_blocks;
    dct.n_rows = n_cold_rows;
    dct.step_size = static_cast<float>(step_size);
    dct.quant_scale = static_cast<float>(quant_scale);
    dct.quant_zp = static_cast<float>(quant_zp);

    // Copy DC values — accept int16 from Python, store as uint8
    // At step>=8, quantized DC fits in uint8 (range 0-255)
    dct.dc_values.resize(n_blocks);
    const int16_t* dc_ptr = dc_values_t.data_ptr<int16_t>();
    for (int64_t i = 0; i < n_blocks; i++) {
        int16_t v = dc_ptr[i];
        dct.dc_values[i] = static_cast<uint8_t>(std::max(0, std::min(255, (int)v)));
    }

    // Copy AC coefficients (sparse)
    dct.ac_coeffs.resize(n_blocks);
    if (ac_data_t.numel() > 0) {
        const int16_t* ac_val_ptr = ac_data_t.data_ptr<int16_t>();
        const uint8_t* ac_pos_ptr = ac_positions_t.data_ptr<uint8_t>();
        const int64_t* ac_off_ptr = ac_block_offsets_t.data_ptr<int64_t>();

        for (int64_t bi = 0; bi < n_blocks; bi++) {
            int64_t start = ac_off_ptr[bi];
            int64_t end = ac_off_ptr[bi + 1];
            dct.ac_coeffs[bi].clear();
            for (int64_t j = start; j < end; j++) {
                dct.ac_coeffs[bi].emplace_back(ac_pos_ptr[j], ac_val_ptr[j]);
            }
        }
    }

    // Initialize weight matrices with appropriate block size
    int rpb = 1;
    if (block_size_param == 4) rpb = 1;
    else if (block_size_param == 8) rpb = 4;
    else if (block_size_param == 16) rpb = 16;
    else rpb = (block_size_param / DCT_TILE) * (block_size_param / DCT_TILE);
    dct.init_weights(tab.D, static_cast<int>(block_size_param), rpb);

    tab.has_dct_cold = true;
    tab.has_cold_frames = true;  // Signal that cold lookup is handled in C++

    int64_t total_ac = ac_data_t.numel();
    int64_t dc_kb = n_blocks / 1024;  // uint8: 1 byte per block
    int64_t ac_kb = total_ac * 3 / 1024;  // 3 bytes per AC (pos + val)
    fprintf(stderr, "[C++] Table %ld: registered DCT-domain cold "
            "(%ld blocks, %ld rows, DC=%ldKB, AC=%ldKB, step=%.0f, "
            "scale=%.6f, zp=%.1f)\n",
            table_idx, n_blocks, n_cold_rows, dc_kb, ac_kb,
            dct.step_size, dct.quant_scale, dct.quant_zp);
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
    // DCT-domain: compute directly from DCT coefficients (no decode/cache needed)
    if (tab.has_dct_cold) {
        tab.dct_cold.accum_row(out_row, cold_idx, D);
        return true;
    }

    int64_t buf_row;
    if (tab.cold_flat) {
        if (tab.cold_sparse) {
            // Sparse flat: check validity bitmap, then rank for dense index
            if (__builtin_expect(cold_idx >= 0 && cold_idx < tab.n_cold_total, 1)) {
                int64_t word = cold_idx / 64;
                int64_t bit = cold_idx % 64;
                uint64_t w = tab.cold_valid_bitmap[word];
                if (!((w >> bit) & 1)) return false;  // not cached
                uint64_t below_mask = (1ULL << bit) - 1;
                buf_row = tab.cold_valid_rank[word] +
                    __builtin_popcountll(w & below_mask);
            } else {
                return false;
            }
        } else {
            // Dense flat: direct indexing by natural cold position
            if (__builtin_expect(cold_idx >= 0 && cold_idx < tab.n_cold_rows, 1)) {
                buf_row = cold_idx;
            } else {
                return false;
            }
        }
    } else if (tab.cold_row_cache) {
        // Row-level cache: open-addressing hash lookup
        int32_t slot = tab.rc_find(cold_idx);
        if (__builtin_expect(slot >= 0, 1)) {
            const uint8_t* emb = tab.rc_data_ptr + slot * D;
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
        return false;
    } else if (tab.cold_dynamic) {
        // Dynamic frame mode: per-frame pointers, O(1) add/remove
        int64_t fid = cold_idx / tab.rows_per_frame;
        if (__builtin_expect(fid < (int64_t)tab.dyn_frame_ptrs.size() &&
                             tab.dyn_frame_ptrs[fid] != nullptr, 1)) {
            int64_t off_in_frame = cold_idx % tab.rows_per_frame;
            const uint8_t* frame_ptr = tab.dyn_frame_ptrs[fid];

            if (tab.dyn_tiled && D == 16) {
                // Tiled (H,W) layout: compute tile coordinates directly
                // Each row is a 4×4 tile in the frame. Row r → tile at (ty, tx).
                int64_t tiles_per_row = tab.dyn_width / 4;
                int64_t ty = off_in_frame / tiles_per_row;
                int64_t tx = off_in_frame % tiles_per_row;
                float scale = tab.cold_scale;
                float zp = tab.cold_zp;
                // Read 4 rows of 4 pixels each from tiled frame
                for (int ly = 0; ly < 4; ly++) {
                    const uint8_t* pixel = frame_ptr + (ty * 4 + ly) * tab.dyn_width + tx * 4;
                    for (int lx = 0; lx < 4; lx++) {
                        out_row[ly * 4 + lx] += (static_cast<float>(pixel[lx]) - zp) * scale;
                    }
                }
                return true;
            }

            // Flat (rpf, D) layout: contiguous row read
            const uint8_t* emb = frame_ptr + off_in_frame * D;
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
        return false;
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
                    if (tab.has_dct_cold) { \
                        int64_t ci = br.cold_rank(idx_ptr[i]); \
                        tab.dct_cold.accum_row(out_row, ci, D); \
                    } else if (tab.has_cold_frames && tab.cold_flat) { \
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

/**
 * fast_forward_seq — Same as fast_forward but sequential tables,
 * parallel batch elements within each table.
 * Matches baseline nn.EmbeddingBag parallelization.
 */
std::vector<torch::Tensor> fast_forward_seq(
    const torch::Tensor& lS_i,
    const torch::Tensor& lS_o)
{
    TORCH_CHECK(g_registered, "Tables not registered.");
    const int64_t T = g_tables.size();
    const int64_t N = lS_i.size(1);
    const int64_t B = lS_o.size(1);
    const int64_t D = g_tables[0].D;
    constexpr int32_t INVALID = std::numeric_limits<int32_t>::min();

    auto all_outputs = torch::zeros({T * B, D}, torch::kFloat32);
    float* all_out_ptr = all_outputs.data_ptr<float>();

    int64_t n_compressed = 0;
    for (int64_t t = 0; t < T; t++)
        if (g_tables[t].kind != TableKind::STANDARD) n_compressed++;

    auto all_cold_masks = torch::zeros({n_compressed * N}, torch::kBool);
    bool* all_cm_ptr = all_cold_masks.data_ptr<bool>();

    std::vector<int64_t> cm_offset(T, -1);
    { int64_t ci = 0;
      for (int64_t t = 0; t < T; t++)
          if (g_tables[t].kind != TableKind::STANDARD)
              cm_offset[t] = ci++ * N;
    }
    std::vector<int64_t> cold_counts(T, 0);

    // Sequential over tables, parallel over batches within each table
    for (int64_t t = 0; t < T; t++) {
        const auto& tab = g_tables[t];
        const int64_t* idx_ptr = lS_i.data_ptr<int64_t>() + t * N;
        const int64_t* off_ptr = lS_o.data_ptr<int64_t>() + t * B;
        float* out_ptr = all_out_ptr + t * B * D;

        if (tab.kind == TableKind::STANDARD) {
            const float* w_ptr = tab.weight.data_ptr<float>();
            at::parallel_for(0, B, 1, [&](int64_t b_begin, int64_t b_end) {
                for (int64_t b = b_begin; b < b_end; b++) {
                    int64_t start = off_ptr[b];
                    int64_t end = (b + 1 < B) ? off_ptr[b + 1] : N;
                    float* out_row = out_ptr + b * D;
                    for (int64_t i = start; i < end; i++) {
                        const float* emb_row = w_ptr + idx_ptr[i] * D;
#if HAS_AVX512
                        if (D == 16) accum_fp32_d16(out_row, emb_row);
                        else
#endif
                        for (int64_t d = 0; d < D; d++) out_row[d] += emb_row[d];
                    }
                }
            });
        } else {
            bool is_q8 = (tab.kind == TableKind::COMPRESSED_Q8);
            const uint8_t* hw_q8 = is_q8 ? tab.weight.data_ptr<uint8_t>() : nullptr;
            const float* hw_fp32 = !is_q8 ? tab.weight.data_ptr<float>() : nullptr;
            bool* cm_ptr = all_cm_ptr + cm_offset[t];
            std::atomic<int64_t> atomic_cold{0};

            at::parallel_for(0, B, 1, [&](int64_t b_begin, int64_t b_end) {
                int64_t local_cold = 0;
                for (int64_t b = b_begin; b < b_end; b++) {
                    int64_t start = off_ptr[b];
                    int64_t end = (b + 1 < B) ? off_ptr[b + 1] : N;
                    float* out_row = out_ptr + b * D;

                    for (int64_t i = start; i < end; i++) {
                        int32_t m;
                        if (tab.use_bitmap) {
                            m = tab.bitmap_rank.lookup(idx_ptr[i]);
                        } else {
                            m = tab.mapping.data_ptr<int32_t>()[idx_ptr[i]];
                            if (m == INVALID) { cm_ptr[i] = true; local_cold++; continue; }
                        }

                        if (m >= 0) {
                            if (is_q8) {
                                const uint8_t* emb = hw_q8 + static_cast<int64_t>(m) * D;
#if HAS_AVX512
                                if (D == 16) accum_q8_d16(out_row, emb, tab.hot_scale, tab.hot_zp);
                                else
#endif
                                for (int64_t d = 0; d < D; d++)
                                    out_row[d] += (static_cast<float>(emb[d]) - tab.hot_zp) * tab.hot_scale;
                            } else {
                                const float* emb = hw_fp32 + static_cast<int64_t>(m) * D;
#if HAS_AVX512
                                if (D == 16) accum_fp32_d16(out_row, emb);
                                else
#endif
                                for (int64_t d = 0; d < D; d++) out_row[d] += emb[d];
                            }
                        } else {
                            if (tab.has_cold_frames) {
                                int64_t ci;
                                if (tab.use_bitmap && tab.cold_flat) {
                                    ci = tab.bitmap_rank.cold_rank(idx_ptr[i]);
                                } else if (tab.use_bitmap && tab.has_cold_mapping) {
                                    ci = tab.cold_mapping_ptr[idx_ptr[i]];
                                } else {
                                    ci = -(static_cast<int64_t>(m) + 1);
                                }
                                if (ci >= 0) cold_frame_accum(out_row, tab, ci, D);
                            } else {
                                cm_ptr[i] = true; local_cold++;
                            }
                        }
                    }
                }
                atomic_cold += local_cold;
            });
            cold_counts[t] = atomic_cold.load();
        }
    }

    std::vector<torch::Tensor> results;
    results.reserve(T * 3 + 1);
    auto outputs_3d = all_outputs.view({T, B, D});
    for (int64_t t = 0; t < T; t++) results.push_back(outputs_3d[t]);
    for (int64_t t = 0; t < T; t++) {
        if (cm_offset[t] >= 0)
            results.push_back(all_cold_masks.slice(0, cm_offset[t], cm_offset[t] + N));
        else results.push_back(torch::Tensor());
    }
    for (int64_t t = 0; t < T; t++)
        results.push_back(torch::tensor(cold_counts[t], torch::kLong));
    results.push_back(outputs_3d);
    return results;
}



// Pipeline structs (needed by fast_forward_seq_pipelined, defined early for access)
struct PipelineTableInfo {
    std::string frame_dir;
    std::string frame_ext;
    int64_t rpf;
    float cold_scale, cold_zp;
    int64_t n_cold;
    int64_t D;
    torch::Tensor is_hot;
    torch::Tensor o2c_map;
};

struct PipelineState {
    std::vector<PipelineTableInfo> table_info;
    std::vector<int64_t> compressed_tables;
    int64_t budget = 9999;
    bool initialized = false;
    std::future<void> bg_future;

    // Pre-loaded compressed frames in memory (eliminates file I/O during decode)
    // Key: (pipeline_k, frame_id) → compressed bytes tensor
    std::unordered_map<int64_t, torch::Tensor> inmem_frames;  // key = k * 100000 + fid
};

static PipelineState g_pipeline;

// Forward declarations for pipeline/decode functions defined later
static void pipeline_add_frame(int64_t table_idx, int64_t frame_id, torch::Tensor frame_data);
static void pipeline_remove_frame(int64_t table_idx, int64_t frame_id);
torch::Tensor decode_hevc_file_fast(const std::string& path, int mode,
    bool skip_loop_filter, bool skip_idct, bool fast_decode);
torch::Tensor decode_h265_frame_from_file(const std::string& path, int num_threads,
    bool skip_loop_filter, bool skip_idct, bool fast_decode);
std::vector<torch::Tensor> batch_decode_fast(
    const std::vector<std::string>& paths, int mode, int max_parallel,
    bool skip_loop_filter, bool skip_idct, bool fast_decode);

/**
 * fast_forward_seq_pipelined — Sequential tables with 2-table frame LRU cache.
 *
 * Processes tables sequentially. Before each compressed table:
 *   1. Decode its needed frames (parallel threads, one per frame)
 *   2. If cache > 2 tables, evict oldest table's frames
 *   3. Run inference using dynamic frame pointers
 *
 * Peak memory: ~2 tables' decoded frames (~8-16MB vs 40MB for all 8 tables)
 * Re-decodes frames each batch since different tables' frames get evicted.
 */
std::vector<torch::Tensor> fast_forward_seq_pipelined(
    const torch::Tensor& lS_i,
    const torch::Tensor& lS_o)
{
    TORCH_CHECK(g_registered, "Tables not registered.");
    TORCH_CHECK(g_pipeline.initialized, "Pipeline not initialized.");
    const int64_t T = g_tables.size();
    const int64_t N = lS_i.size(1);
    const int64_t B = lS_o.size(1);
    const int64_t D = g_tables[0].D;
    constexpr int32_t INVALID = std::numeric_limits<int32_t>::min();
    const int64_t K = g_pipeline.compressed_tables.size();

    auto all_outputs = torch::zeros({T * B, D}, torch::kFloat32);
    float* all_out_ptr = all_outputs.data_ptr<float>();

    int64_t n_compressed = 0;
    for (int64_t t = 0; t < T; t++)
        if (g_tables[t].kind != TableKind::STANDARD) n_compressed++;

    auto all_cold_masks = torch::zeros({n_compressed * N}, torch::kBool);
    bool* all_cm_ptr = all_cold_masks.data_ptr<bool>();

    std::vector<int64_t> cm_offset(T, -1);
    { int64_t ci = 0;
      for (int64_t t = 0; t < T; t++)
          if (g_tables[t].kind != TableKind::STANDARD) cm_offset[t] = ci++ * N;
    }
    std::vector<int64_t> cold_counts(T, 0);

    // Map table_idx -> pipeline index k
    std::unordered_map<int64_t, int64_t> table_to_k;
    for (int64_t k = 0; k < K; k++)
        table_to_k[g_pipeline.compressed_tables[k]] = k;

    // LRU: track which tables have frames loaded (oldest first)
    std::deque<int64_t> cached_tables;  // pipeline k indices, front = oldest

    // --- Process tables sequentially ---
    for (int64_t t = 0; t < T; t++) {
        const auto& tab = g_tables[t];
        const int64_t* idx_ptr = lS_i.data_ptr<int64_t>() + t * N;
        const int64_t* off_ptr = lS_o.data_ptr<int64_t>() + t * B;
        float* out_ptr = all_out_ptr + t * B * D;

        auto kit = table_to_k.find(t);
        if (kit != table_to_k.end()) {
            int64_t k = kit->second;
            auto& info = g_pipeline.table_info[k];

            // Evict oldest table if cache full (keep max 2)
            while ((int64_t)cached_tables.size() >= 2) {
                int64_t evict_k = cached_tables.front();
                cached_tables.pop_front();
                int64_t evict_t = g_pipeline.compressed_tables[evict_k];
                auto& evict_tab = g_tables[evict_t];
                // Clear all frame pointers for this table
                for (size_t fi = 0; fi < evict_tab.dyn_frame_ptrs.size(); fi++) {
                    evict_tab.dyn_frame_ptrs[fi] = nullptr;
                    evict_tab.dyn_frame_tensors[fi] = torch::Tensor();
                }
            }

            // Scan which frames this table needs
            const bool* hot_ptr = info.is_hot.data_ptr<bool>();
            const int32_t* o2c_ptr = info.o2c_map.data_ptr<int32_t>();
            std::unordered_set<int64_t> needed_fids;
            for (int64_t i = 0; i < N; i++) {
                int64_t orig = idx_ptr[i];
                if (!hot_ptr[orig]) {
                    int32_t ci = o2c_ptr[orig];
                    if (ci >= 0) needed_fids.insert(ci / info.rpf);
                }
            }

            // Decode missing frames using batch_decode_fast (pool-accelerated)
            std::vector<std::string> paths_to_decode;
            std::vector<int64_t> fids_to_decode;
            auto& dyn = g_tables[t].dyn_frame_ptrs;
            for (int64_t fid : needed_fids) {
                if (fid >= (int64_t)dyn.size() || dyn[fid] == nullptr) {
                    char fname[128];
                    snprintf(fname, sizeof(fname), "frame_%05ld%s",
                             (long)fid, info.frame_ext.c_str());
                    paths_to_decode.push_back(info.frame_dir + "/" + fname);
                    fids_to_decode.push_back(fid);
                }
            }

            if (!paths_to_decode.empty()) {
                auto decoded = batch_decode_fast(
                    paths_to_decode, 2, paths_to_decode.size(),
                    true, false, false);
                for (size_t i = 0; i < fids_to_decode.size(); i++) {
                    if (decoded[i].defined())
                        pipeline_add_frame(t, fids_to_decode[i], decoded[i]);
                }
            }

            cached_tables.push_back(k);
        }

        // --- Run inference for this table ---
        if (tab.kind == TableKind::STANDARD) {
            const float* w_ptr = tab.weight.data_ptr<float>();
            at::parallel_for(0, B, 1, [&](int64_t b_begin, int64_t b_end) {
                for (int64_t b = b_begin; b < b_end; b++) {
                    int64_t start = off_ptr[b];
                    int64_t end = (b + 1 < B) ? off_ptr[b + 1] : N;
                    float* out_row = out_ptr + b * D;
                    for (int64_t i = start; i < end; i++) {
                        const float* emb_row = w_ptr + idx_ptr[i] * D;
#if HAS_AVX512
                        if (D == 16) accum_fp32_d16(out_row, emb_row);
                        else
#endif
                        for (int64_t d = 0; d < D; d++) out_row[d] += emb_row[d];
                    }
                }
            });
        } else {
            bool is_q8 = (tab.kind == TableKind::COMPRESSED_Q8);
            const uint8_t* hw_q8 = is_q8 ? tab.weight.data_ptr<uint8_t>() : nullptr;
            const float* hw_fp32 = !is_q8 ? tab.weight.data_ptr<float>() : nullptr;
            bool* cm_ptr = all_cm_ptr + cm_offset[t];
            std::atomic<int64_t> atomic_cold{0};

            at::parallel_for(0, B, 1, [&](int64_t b_begin, int64_t b_end) {
                int64_t local_cold = 0;
                for (int64_t b = b_begin; b < b_end; b++) {
                    int64_t start = off_ptr[b];
                    int64_t end = (b + 1 < B) ? off_ptr[b + 1] : N;
                    float* out_row = out_ptr + b * D;
                    for (int64_t i = start; i < end; i++) {
                        int32_t m;
                        if (tab.use_bitmap) {
                            m = tab.bitmap_rank.lookup(idx_ptr[i]);
                        } else {
                            m = tab.mapping.data_ptr<int32_t>()[idx_ptr[i]];
                            if (m == INVALID) { cm_ptr[i] = true; local_cold++; continue; }
                        }
                        if (m >= 0) {
                            if (is_q8) {
                                const uint8_t* emb = hw_q8 + static_cast<int64_t>(m) * D;
#if HAS_AVX512
                                if (D == 16) accum_q8_d16(out_row, emb, tab.hot_scale, tab.hot_zp);
                                else
#endif
                                for (int64_t d = 0; d < D; d++)
                                    out_row[d] += (static_cast<float>(emb[d]) - tab.hot_zp) * tab.hot_scale;
                            } else {
                                const float* emb = hw_fp32 + static_cast<int64_t>(m) * D;
#if HAS_AVX512
                                if (D == 16) accum_fp32_d16(out_row, emb);
                                else
#endif
                                for (int64_t d = 0; d < D; d++) out_row[d] += emb[d];
                            }
                        } else {
                            if (tab.has_cold_frames) {
                                int64_t ci;
                                if (tab.use_bitmap && tab.cold_flat) {
                                    ci = tab.bitmap_rank.cold_rank(idx_ptr[i]);
                                } else if (tab.use_bitmap && tab.has_cold_mapping) {
                                    ci = tab.cold_mapping_ptr[idx_ptr[i]];
                                } else {
                                    ci = -(static_cast<int64_t>(m) + 1);
                                }
                                if (ci >= 0) cold_frame_accum(out_row, tab, ci, D);
                            } else {
                                cm_ptr[i] = true; local_cold++;
                            }
                        }
                    }
                }
                atomic_cold += local_cold;
            });
            cold_counts[t] = atomic_cold.load();
        }
    }

    std::vector<torch::Tensor> results;
    results.reserve(T * 3 + 1);
    auto outputs_3d = all_outputs.view({T, B, D});
    for (int64_t t = 0; t < T; t++) results.push_back(outputs_3d[t]);
    for (int64_t t = 0; t < T; t++) {
        if (cm_offset[t] >= 0)
            results.push_back(all_cold_masks.slice(0, cm_offset[t], cm_offset[t] + N));
        else results.push_back(torch::Tensor());
    }
    for (int64_t t = 0; t < T; t++)
        results.push_back(torch::tensor(cold_counts[t], torch::kLong));
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
// CODEC CONTEXT POOL — reuse AVCodecContext across decodes
// ============================================================

/**
 * HevcDecoderPool — Thread-safe pool of pre-initialized H.265 decoder contexts.
 *
 * Avoids ~2ms overhead of avcodec_alloc_context3 + avcodec_open2 per decode.
 * Contexts are flushed (avcodec_flush_buffers) between uses, which is safe
 * for ALL-INTRA frames with no inter-frame dependencies.
 */
class HevcDecoderPool {
public:
    struct DecoderCtx {
        AVCodecContext* codec_ctx;
        AVPacket* pkt;
        AVFrame* frame;
    };

    static HevcDecoderPool& instance() {
        static HevcDecoderPool pool;
        return pool;
    }

    /**
     * acquire — Get a decoder context from the pool or create one.
     *
     * The caller must provide AVCodecParameters from the stream so that
     * newly created contexts are properly initialized with SPS/PPS extradata.
     */
    DecoderCtx acquire(AVCodecParameters* codecpar,
                       bool skip_loop_filter = false,
                       bool skip_idct = false,
                       bool fast_decode = false) {
        std::lock_guard<std::mutex> lock(mu_);
        uint8_t key = (skip_loop_filter ? 1 : 0) |
                      (skip_idct ? 2 : 0) |
                      (fast_decode ? 4 : 0);
        auto& q = pools_[key];
        if (!q.empty()) {
            auto ctx = q.front();
            q.pop();
            avcodec_flush_buffers(ctx.codec_ctx);
            return ctx;
        }
        // Create new context from stream parameters
        return create_new(codecpar, skip_loop_filter, skip_idct, fast_decode);
    }

    void release(DecoderCtx ctx, bool skip_loop_filter = false,
                 bool skip_idct = false, bool fast_decode = false) {
        std::lock_guard<std::mutex> lock(mu_);
        uint8_t key = (skip_loop_filter ? 1 : 0) |
                      (skip_idct ? 2 : 0) |
                      (fast_decode ? 4 : 0);
        pools_[key].push(ctx);
    }

private:
    HevcDecoderPool() = default;

    DecoderCtx create_new(AVCodecParameters* codecpar,
                          bool skip_loop_filter, bool skip_idct, bool fast_decode) {
        const AVCodec* codec = avcodec_find_decoder(codecpar->codec_id);
        TORCH_CHECK(codec != nullptr, "Decoder not found for codec");

        AVCodecContext* codec_ctx = avcodec_alloc_context3(codec);
        TORCH_CHECK(codec_ctx != nullptr, "Failed to alloc codec context");

        // Copy ALL parameters from stream (including extradata/SPS/PPS)
        avcodec_parameters_to_context(codec_ctx, codecpar);

        codec_ctx->thread_count = 1;
        codec_ctx->thread_type = FF_THREAD_SLICE;

        if (skip_loop_filter) codec_ctx->skip_loop_filter = AVDISCARD_ALL;
        if (skip_idct) codec_ctx->skip_idct = AVDISCARD_ALL;
        if (fast_decode) codec_ctx->flags2 |= AV_CODEC_FLAG2_FAST;

        int ret = avcodec_open2(codec_ctx, codec, nullptr);
        TORCH_CHECK(ret >= 0, "Failed to open codec");

        AVPacket* pkt = av_packet_alloc();
        AVFrame* frame = av_frame_alloc();

        return {codec_ctx, pkt, frame};
    }

    std::mutex mu_;
    std::unordered_map<uint8_t, std::queue<DecoderCtx>> pools_;
};


/**
 * decode_hevc_file_fast — Fast single-frame decode with format hint + pool.
 *
 * Optimizations vs the original decode_h265_frame_from_file:
 *   1. Format hint: tells avformat the file is HEVC, skips format probing
 *   2. Skip find_stream_info: we know the stream layout
 *   3. Pool codec context: reuse AVCodecContext across decodes (biggest win)
 *
 * Modes:
 *   mode=0: original (full avformat probing, fresh context each time)
 *   mode=1: format hint only (skip probing, fresh context)
 *   mode=2: pool only (full probing, reuse context)
 *   mode=3: hint + pool (fastest)
 */
torch::Tensor decode_hevc_file_fast(
    const std::string& path,
    int mode = 3,
    bool skip_loop_filter = false,
    bool skip_idct = false,
    bool fast_decode = false);

// Forward declarations — defined later in this file
torch::Tensor decode_h265_frame_from_file(const std::string& path, int num_threads,
                                          bool skip_loop_filter,
                                          bool skip_idct,
                                          bool fast_decode);
static torch::Tensor decode_any_frame_from_file(
    const std::string& path, const std::string& ext, int num_threads,
    bool skip_loop_filter, bool skip_idct, bool fast_decode);

torch::Tensor decode_hevc_file_fast(
    const std::string& path,
    int mode,
    bool skip_loop_filter,
    bool skip_idct,
    bool fast_decode)
{
    // Mode 0: original path
    if (mode == 0) {
        return decode_h265_frame_from_file(path, 1, skip_loop_filter,
                                           skip_idct, fast_decode);
    }

    bool use_hint = (mode == 1 || mode == 3);
    bool use_pool = (mode == 2 || mode == 3);

    // --- Open file with avformat ---
    AVFormatContext* fmt_ctx = nullptr;
    int ret = avformat_open_input(&fmt_ctx, path.c_str(), nullptr, nullptr);
    TORCH_CHECK(ret >= 0, "Failed to open file: ", path);

    if (use_hint) {
        // Fast probing: minimize probe size and skip duration analysis.
        // Our files are small single-frame containers, so minimal probing suffices.
        fmt_ctx->probesize = 4096;
        fmt_ctx->max_analyze_duration = 0;
    }
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

    // --- Get or create codec context ---
    HevcDecoderPool::DecoderCtx dctx = {};
    AVCodecContext* codec_ctx = nullptr;
    AVPacket* pkt = nullptr;
    AVFrame* frame = nullptr;
    bool from_pool = false;

    if (use_pool) {
        dctx = HevcDecoderPool::instance().acquire(
            codecpar, skip_loop_filter, skip_idct, fast_decode);
        codec_ctx = dctx.codec_ctx;
        pkt = dctx.pkt;
        frame = dctx.frame;
        from_pool = true;
    } else {
        // Fresh context
        const AVCodec* codec = avcodec_find_decoder(codecpar->codec_id);
        TORCH_CHECK(codec != nullptr, "Codec not found");
        codec_ctx = avcodec_alloc_context3(codec);
        TORCH_CHECK(codec_ctx != nullptr, "Failed to alloc codec context");
        avcodec_parameters_to_context(codec_ctx, codecpar);
        codec_ctx->thread_count = 1;
        codec_ctx->thread_type = FF_THREAD_SLICE;
        if (skip_loop_filter) codec_ctx->skip_loop_filter = AVDISCARD_ALL;
        if (skip_idct) codec_ctx->skip_idct = AVDISCARD_ALL;
        if (fast_decode) codec_ctx->flags2 |= AV_CODEC_FLAG2_FAST;
        ret = avcodec_open2(codec_ctx, codec, nullptr);
        TORCH_CHECK(ret >= 0, "Failed to open codec");
        pkt = av_packet_alloc();
        frame = av_frame_alloc();
    }

    // --- Read and decode ---
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

    // --- Cleanup ---
    avformat_close_input(&fmt_ctx);

    if (from_pool) {
        HevcDecoderPool::instance().release(
            dctx, skip_loop_filter, skip_idct, fast_decode);
    } else {
        av_frame_free(&frame);
        av_packet_free(&pkt);
        avcodec_free_context(&codec_ctx);
    }

    TORCH_CHECK(decoded, "Failed to decode: ", path);
    return result;
}


/**
 * batch_decode_fast — Batch decode with raw parsing + pool.
 *
 * Like batch_decode_file_paths but uses decode_hevc_file_fast for H.265 files.
 */
std::vector<torch::Tensor> batch_decode_fast(
    const std::vector<std::string>& paths,
    int mode = 3,
    int max_parallel = 0,
    bool skip_loop_filter = false,
    bool skip_idct = false,
    bool fast_decode = false)
{
    const int64_t n = paths.size();
    if (n == 0) return {};
    if (max_parallel <= 0) max_parallel = n;

    std::vector<torch::Tensor> results(n);
    std::vector<std::string> errors(n);

    for (int64_t start = 0; start < n; start += max_parallel) {
        int64_t end = std::min(start + (int64_t)max_parallel, n);
        std::vector<std::thread> threads;
        for (int64_t i = start; i < end; i++) {
            threads.emplace_back([&, i]() {
                try {
                    std::string ext = ".h265";
                    size_t dot_pos = paths[i].rfind('.');
                    if (dot_pos != std::string::npos) {
                        ext = paths[i].substr(dot_pos);
                    }
                    if (ext == ".zst") {
                        // Zstd path unchanged
                        results[i] = decode_any_frame_from_file(
                            paths[i], ext, 1, skip_loop_filter,
                            skip_idct, fast_decode);
                    } else {
                        results[i] = decode_hevc_file_fast(
                            paths[i], mode, skip_loop_filter,
                            skip_idct, fast_decode);
                    }
                } catch (const std::exception& e) {
                    errors[i] = e.what();
                }
            });
        }
        for (auto& t : threads) t.join();
    }

    for (int64_t i = 0; i < n; i++) {
        TORCH_CHECK(errors[i].empty(), "Failed to decode ", paths[i], ": ", errors[i]);
    }
    return results;
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
torch::Tensor decode_h265_frame_from_file(const std::string& path, int num_threads = 0,
                                          bool skip_loop_filter = false,
                                          bool skip_idct = false,
                                          bool fast_decode = false) {
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

    // thread_count=0 → auto (libavcodec picks), 1 → single-threaded (for external parallelism)
    codec_ctx->thread_count = num_threads;
    codec_ctx->thread_type = FF_THREAD_SLICE;

    // Skip loop filter (deblocking) — saves ~15-23% decode time
    if (skip_loop_filter) {
        codec_ctx->skip_loop_filter = AVDISCARD_ALL;
    }

    // Skip IDCT — more aggressive, skips inverse DCT reconstruction
    if (skip_idct) {
        codec_ctx->skip_idct = AVDISCARD_ALL;
    }

    // AV_CODEC_FLAG2_FAST — allow non-spec-compliant speed tricks
    if (fast_decode) {
        codec_ctx->flags2 |= AV_CODEC_FLAG2_FAST;
    }

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
 * decode_any_frame_from_file - Generic frame decode that dispatches by extension.
 *
 * For .zst: reads file, Zstd decompresses, returns 1D uint8 tensor (flat).
 * For .h265/.h264/.mkv: calls decode_h265_frame_from_file, returns (H, W) uint8 tensor.
 *
 * The caller must handle the shape difference (1D for Zstd, 2D for video codecs).
 */
static torch::Tensor decode_any_frame_from_file(
    const std::string& path, const std::string& ext, int num_threads = 0,
    bool skip_loop_filter = false,
    bool skip_idct = false,
    bool fast_decode = false)
{
    if (ext == ".zst") {
        // Zstd path: read file + decompress
        FILE* f = fopen(path.c_str(), "rb");
        TORCH_CHECK(f != nullptr, "Failed to open Zstd file: ", path);
        fseek(f, 0, SEEK_END);
        size_t file_size = ftell(f);
        fseek(f, 0, SEEK_SET);
        std::vector<uint8_t> compressed(file_size);
        size_t nread = fread(compressed.data(), 1, file_size, f);
        fclose(f);
        TORCH_CHECK(nread == file_size, "Short read on ", path);

        // Get decompressed size from Zstd frame header
        unsigned long long content_size = ZSTD_getFrameContentSize(
            compressed.data(), file_size);
        TORCH_CHECK(content_size != ZSTD_CONTENTSIZE_UNKNOWN &&
                    content_size != ZSTD_CONTENTSIZE_ERROR,
                    "Cannot determine Zstd content size for: ", path);

        auto output = torch::empty({(int64_t)content_size}, torch::kUInt8);
        size_t decompressed = ZSTD_decompress(
            output.data_ptr<uint8_t>(), (size_t)content_size,
            compressed.data(), file_size);
        TORCH_CHECK(!ZSTD_isError(decompressed),
                    "Zstd decompress failed for ", path, ": ",
                    ZSTD_getErrorName(decompressed));
        return output;
    } else {
        // Video codec path (H.265, H.264, FFV1, etc.)
        return decode_h265_frame_from_file(path, num_threads, skip_loop_filter,
                                              skip_idct, fast_decode);
    }
}

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
    bool is_zstd = (ext == ".zst");
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
                decoded_frames[i] = decode_any_frame_from_file(path, ext);
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
        int64_t width;  // width for tiled frames, D for flat (Zstd) frames
    };
    std::vector<FrameInfo> frame_infos(num_frames);
    for (int64_t i = 0; i < num_frames; i++) {
        frame_infos[i].data = decoded_frames[i].data_ptr<uint8_t>();
        frame_infos[i].width = is_zstd ? D : decoded_frames[i].size(1);
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
            float* dst = out_ptr + i * D;

            if (is_zstd) {
                // Flat layout: row_in_frame * D + col
                const uint8_t* src = frame_data + row_in_frame * D;
                for (int64_t d = 0; d < D; d++) {
                    dst[d] = (static_cast<float>(src[d]) - fzp) * fscale;
                }
            } else {
                // Tiled layout: (ty*4+ly) * width + tx*4+lx
                int64_t width = frame_infos[fidx].width;
                int64_t ty = row_in_frame / tiles_per_row;
                int64_t tx = row_in_frame % tiles_per_row;
                for (int ly = 0; ly < 4; ly++) {
                    const uint8_t* src = frame_data + (ty * 4 + ly) * width + tx * 4;
                    for (int lx = 0; lx < 4; lx++) {
                        dst[ly * 4 + lx] = (static_cast<float>(src[lx]) - fzp) * fscale;
                    }
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
    const char* exts[] = {".h265", ".h264", ".mkv", ".zst"};
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
 * batch_decode_frames - Decode multiple frames in parallel, return decoded frames.
 *
 * Auto-detects codec from file extension (.h265, .h264, .mkv, .zst).
 * For video codecs: returns (H, W) uint8 tensors (tiled layout).
 * For Zstd: returns 1D uint8 tensors (flat layout, rows_per_frame * D bytes).
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
                decoded_frames[i] = decode_any_frame_from_file(path, ext);
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


/**
 * batch_decode_file_paths - Decode frames from explicit file paths in parallel.
 *
 * Auto-detects codec from file extension (.zst → Zstd, else → FFmpeg).
 * Uses std::thread for parallelism (no Python GIL, no at::parallel_for).
 * max_parallel controls how many concurrent decodes run at once.
 */
std::vector<torch::Tensor> batch_decode_file_paths(
    const std::vector<std::string>& paths,
    int num_threads_per_decode = 1,
    int max_parallel = 0,
    bool skip_loop_filter = false,
    bool skip_idct = false,
    bool fast_decode = false)
{
    const int64_t n = paths.size();
    if (n == 0) return {};
    if (max_parallel <= 0) max_parallel = n;  // default: all at once

    std::vector<torch::Tensor> results(n);
    std::vector<std::string> errors(n);

    // Process in chunks of max_parallel
    for (int64_t start = 0; start < n; start += max_parallel) {
        int64_t end = std::min(start + (int64_t)max_parallel, n);
        std::vector<std::thread> threads;
        for (int64_t i = start; i < end; i++) {
            threads.emplace_back([&, i]() {
                try {
                    // Detect extension from path
                    std::string ext = ".h265";  // default
                    size_t dot_pos = paths[i].rfind('.');
                    if (dot_pos != std::string::npos) {
                        ext = paths[i].substr(dot_pos);
                    }
                    results[i] = decode_any_frame_from_file(
                        paths[i], ext, num_threads_per_decode, skip_loop_filter,
                        skip_idct, fast_decode);
                } catch (const std::exception& e) {
                    errors[i] = e.what();
                }
            });
        }
        for (auto& t : threads) t.join();
    }

    for (int64_t i = 0; i < n; i++) {
        TORCH_CHECK(errors[i].empty(), "Failed to decode ", paths[i], ": ", errors[i]);
    }
    return results;
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


// gather_cold_embeddings: given batch indices, cached frames, and mappings,
// produce fp32 cold embeddings for each cold index. Fuses the Python gather loop.
//
// For each compressed table k:
//   1. Find cold indices in the batch (those where is_hot[idx] == false)
//   2. Map to cold_reordered via o2c_map
//   3. Compute frame_id and row_in_frame
//   4. Gather from the cached frame tensors
//
// Returns a list of K tensors, each (n_cold_in_batch, emb_dim) fp32,
// plus a list of K int64 tensors containing the original cold indices.
std::vector<torch::Tensor> gather_cold_embeddings(
    const std::vector<torch::Tensor>& lS_i_list,      // K tensors, each [N] int64
    const std::vector<torch::Tensor>& is_hot_list,     // K bool tensors
    const std::vector<torch::Tensor>& o2c_map_list,    // K int32 tensors
    const std::vector<std::vector<torch::Tensor>>& cached_frames_list, // K lists of frame tensors
    const std::vector<int64_t>& frame_offsets_list,    // K values: first frame ID for each table
    int64_t rows_per_frame,
    int64_t emb_dim
) {
    const int64_t K = lS_i_list.size();
    // Returns: [gathered_0, orig_cold_idx_0, gathered_1, orig_cold_idx_1, ...]
    std::vector<torch::Tensor> results(2 * K);

    at::parallel_for(0, K, 1, [&](int64_t k_begin, int64_t k_end) {
        for (int64_t k = k_begin; k < k_end; k++) {
            const auto& indices = lS_i_list[k];
            const auto& is_hot = is_hot_list[k];
            const auto& o2c = o2c_map_list[k];
            const auto& cached_frames = cached_frames_list[k];
            const int64_t frame_offset = frame_offsets_list[k];

            const int64_t N = indices.size(0);
            const int64_t* idx_ptr = indices.data_ptr<int64_t>();
            const bool* hot_ptr = is_hot.data_ptr<bool>();
            const int32_t* o2c_ptr = o2c.data_ptr<int32_t>();

            // First pass: count cold indices
            int64_t n_cold = 0;
            for (int64_t i = 0; i < N; i++) {
                if (!hot_ptr[idx_ptr[i]]) n_cold++;
            }

            if (n_cold == 0) {
                results[2*k] = torch::empty({0, emb_dim}, torch::kFloat32);
                results[2*k+1] = torch::empty({0}, torch::kInt64);
                continue;
            }

            auto gathered = torch::empty({n_cold, emb_dim}, torch::kFloat32);
            auto orig_indices = torch::empty({n_cold}, torch::kInt64);
            float* g_ptr = gathered.data_ptr<float>();
            int64_t* oi_ptr = orig_indices.data_ptr<int64_t>();

            int64_t j = 0;
            for (int64_t i = 0; i < N; i++) {
                int64_t idx = idx_ptr[i];
                if (!hot_ptr[idx]) {
                    int32_t cold_idx = o2c_ptr[idx];
                    int64_t fid = static_cast<int64_t>(cold_idx) / rows_per_frame;
                    int64_t row = static_cast<int64_t>(cold_idx) % rows_per_frame;

                    // Find frame in cached_frames list
                    int64_t frame_local = fid - frame_offset;
                    if (frame_local >= 0 && frame_local < (int64_t)cached_frames.size()) {
                        const auto& frame = cached_frames[frame_local];
                        const float* f_ptr = frame.data_ptr<float>();
                        // Copy row from frame
                        std::memcpy(g_ptr + j * emb_dim, f_ptr + row * emb_dim, emb_dim * sizeof(float));
                    }

                    oi_ptr[j] = idx;
                    j++;
                }
            }

            results[2*k] = gathered;
            results[2*k+1] = orig_indices;
        }
    });

    return results;
}


// scatter_cold_to_weights: combined scan + gather + writeback in one call.
// For each compressed table: find cold indices, gather from cached frames,
// and write directly into the weight tensors. No Python intermediate.
void scatter_cold_to_weights(
    const std::vector<torch::Tensor>& lS_i_list,
    const std::vector<torch::Tensor>& is_hot_list,
    const std::vector<torch::Tensor>& o2c_map_list,
    const std::vector<std::vector<torch::Tensor>>& cached_frames_list,
    std::vector<torch::Tensor>& weight_list,  // mutable: writes directly into model weights
    int64_t rows_per_frame,
    int64_t emb_dim
) {
    const int64_t K = lS_i_list.size();

    at::parallel_for(0, K, 1, [&](int64_t k_begin, int64_t k_end) {
        for (int64_t k = k_begin; k < k_end; k++) {
            const auto& indices = lS_i_list[k];
            const auto& is_hot = is_hot_list[k];
            const auto& o2c = o2c_map_list[k];
            const auto& cached_frames = cached_frames_list[k];
            auto& weight = weight_list[k];

            const int64_t N = indices.size(0);
            const int64_t* idx_ptr = indices.data_ptr<int64_t>();
            const bool* hot_ptr = is_hot.data_ptr<bool>();
            const int32_t* o2c_ptr = o2c.data_ptr<int32_t>();
            float* w_ptr = weight.data_ptr<float>();

            for (int64_t i = 0; i < N; i++) {
                int64_t idx = idx_ptr[i];
                if (!hot_ptr[idx]) {
                    int32_t cold_idx = o2c_ptr[idx];
                    if (cold_idx < 0) continue;
                    int64_t fid = static_cast<int64_t>(cold_idx) / rows_per_frame;
                    int64_t row = static_cast<int64_t>(cold_idx) % rows_per_frame;

                    if (fid < (int64_t)cached_frames.size() && cached_frames[fid].numel() > 0) {
                        const float* f_ptr = cached_frames[fid].data_ptr<float>();
                        // Direct memcpy into weight tensor
                        std::memcpy(w_ptr + idx * emb_dim, f_ptr + row * emb_dim, emb_dim * sizeof(float));
                    }
                }
            }
        }
    });
}


// ============================================================
// Zstd compression/decompression (lossless, ~35x faster decode than H.265)
// ============================================================

/**
 * zstd_compress_frame - Compress a contiguous uint8 tensor with Zstd.
 *
 * Returns a 1D uint8 tensor containing the compressed data.
 * level: Zstd compression level (1=fastest, 19=best, default=3).
 */
torch::Tensor zstd_compress_frame(const torch::Tensor& data, int level = 3) {
    TORCH_CHECK(data.dtype() == torch::kUInt8, "zstd_compress_frame: input must be uint8");
    auto contiguous = data.contiguous();
    const size_t src_size = contiguous.numel();
    const void* src = contiguous.data_ptr<uint8_t>();

    size_t bound = ZSTD_compressBound(src_size);
    auto output = torch::empty({(int64_t)bound}, torch::kUInt8);
    void* dst = output.data_ptr<uint8_t>();

    size_t compressed_size = ZSTD_compress(dst, bound, src, src_size, level);
    TORCH_CHECK(!ZSTD_isError(compressed_size),
                "Zstd compress failed: ", ZSTD_getErrorName(compressed_size));

    // Trim to actual size
    return output.slice(0, 0, (int64_t)compressed_size).clone();
}

/**
 * zstd_decompress_frame - Decompress Zstd data back to uint8 tensor.
 *
 * original_size: the expected decompressed size in bytes.
 * Returns a 1D uint8 tensor of exactly original_size bytes.
 */
torch::Tensor zstd_decompress_frame(const torch::Tensor& compressed, int64_t original_size) {
    TORCH_CHECK(compressed.dtype() == torch::kUInt8, "zstd_decompress_frame: input must be uint8");
    auto contiguous = compressed.contiguous();
    const size_t comp_size = contiguous.numel();
    const void* src = contiguous.data_ptr<uint8_t>();

    auto output = torch::empty({original_size}, torch::kUInt8);
    void* dst = output.data_ptr<uint8_t>();

    size_t decompressed_size = ZSTD_decompress(dst, (size_t)original_size, src, comp_size);
    TORCH_CHECK(!ZSTD_isError(decompressed_size),
                "Zstd decompress failed: ", ZSTD_getErrorName(decompressed_size));
    TORCH_CHECK((int64_t)decompressed_size == original_size,
                "Zstd decompress size mismatch: got ", decompressed_size,
                " expected ", original_size);

    return output;
}

/**
 * batch_zstd_compress - Compress multiple frames in parallel, write to .zst files.
 *
 * frames: vector of uint8 tensors (each is a flat frame).
 * output_dir: directory to write frame_XXXXX.zst files.
 * level: Zstd compression level.
 * Returns total compressed bytes written.
 */
int64_t batch_zstd_compress(
    const std::vector<torch::Tensor>& frames,
    const std::string& output_dir,
    int level = 3)
{
    const int64_t n = frames.size();
    if (n == 0) return 0;

    std::vector<int64_t> compressed_sizes(n, 0);
    std::vector<std::string> errors(n);
    std::vector<std::thread> threads;

    for (int64_t i = 0; i < n; i++) {
        threads.emplace_back([&, i]() {
            try {
                const auto& frame = frames[i].contiguous();
                TORCH_CHECK(frame.dtype() == torch::kUInt8,
                            "batch_zstd_compress: frame ", i, " must be uint8");
                const size_t src_size = frame.numel();
                const void* src = frame.data_ptr<uint8_t>();

                size_t bound = ZSTD_compressBound(src_size);
                std::vector<uint8_t> buf(bound);

                size_t comp_size = ZSTD_compress(buf.data(), bound, src, src_size, level);
                if (ZSTD_isError(comp_size)) {
                    errors[i] = std::string("Zstd compress failed: ") + ZSTD_getErrorName(comp_size);
                    return;
                }

                char fname[64];
                snprintf(fname, sizeof(fname), "frame_%05ld.zst", (long)i);
                std::string path = output_dir + "/" + fname;

                FILE* f = fopen(path.c_str(), "wb");
                if (!f) {
                    errors[i] = "Failed to open " + path;
                    return;
                }
                fwrite(buf.data(), 1, comp_size, f);
                fclose(f);
                compressed_sizes[i] = (int64_t)comp_size;
            } catch (const std::exception& e) {
                errors[i] = e.what();
            }
        });
    }
    for (auto& t : threads) t.join();

    int64_t total = 0;
    for (int64_t i = 0; i < n; i++) {
        TORCH_CHECK(errors[i].empty(), "batch_zstd_compress frame ", i, ": ", errors[i]);
        total += compressed_sizes[i];
    }
    return total;
}

/**
 * batch_zstd_decompress_files - Decompress multiple .zst files in parallel.
 *
 * paths: file paths to .zst files.
 * original_size: decompressed size per frame (all frames same size).
 * max_parallel: max concurrent decompressions (0 = all at once).
 * Returns vector of 1D uint8 tensors.
 */
std::vector<torch::Tensor> batch_zstd_decompress_files(
    const std::vector<std::string>& paths,
    int64_t original_size,
    int max_parallel = 0)
{
    const int64_t n = paths.size();
    if (n == 0) return {};
    if (max_parallel <= 0) max_parallel = (int)n;

    std::vector<torch::Tensor> results(n);
    std::vector<std::string> errors(n);

    for (int64_t start = 0; start < n; start += max_parallel) {
        int64_t end = std::min(start + (int64_t)max_parallel, n);
        std::vector<std::thread> threads;

        for (int64_t i = start; i < end; i++) {
            threads.emplace_back([&, i]() {
                try {
                    // Read compressed file
                    FILE* f = fopen(paths[i].c_str(), "rb");
                    if (!f) {
                        errors[i] = "Failed to open " + paths[i];
                        return;
                    }
                    fseek(f, 0, SEEK_END);
                    size_t file_size = ftell(f);
                    fseek(f, 0, SEEK_SET);
                    std::vector<uint8_t> compressed(file_size);
                    size_t read = fread(compressed.data(), 1, file_size, f);
                    fclose(f);
                    if (read != file_size) {
                        errors[i] = "Short read on " + paths[i];
                        return;
                    }

                    // Decompress
                    auto output = torch::empty({original_size}, torch::kUInt8);
                    void* dst = output.data_ptr<uint8_t>();
                    size_t decompressed = ZSTD_decompress(
                        dst, (size_t)original_size,
                        compressed.data(), file_size);
                    if (ZSTD_isError(decompressed)) {
                        errors[i] = std::string("Zstd decompress failed: ") +
                                    ZSTD_getErrorName(decompressed);
                        return;
                    }
                    if ((int64_t)decompressed != original_size) {
                        errors[i] = "Size mismatch: got " + std::to_string(decompressed) +
                                    " expected " + std::to_string(original_size);
                        return;
                    }
                    results[i] = output;
                } catch (const std::exception& e) {
                    errors[i] = e.what();
                }
            });
        }
        for (auto& t : threads) t.join();
    }

    for (int64_t i = 0; i < n; i++) {
        TORCH_CHECK(errors[i].empty(), "batch_zstd_decompress ", paths[i], ": ", errors[i]);
    }
    return results;
}


// ============================================================
// PIPELINED INFERENCE: background decode + LRU frame cache in C++
// ============================================================

// Per-table decode info for the pipeline
// PipelineTableInfo, PipelineState, g_pipeline defined earlier (before fast_forward_seq_pipelined)

/**
 * pipeline_init — Register per-table frame directories and cold params.
 * Called once after register_tables().
 *
 * row_cache_capacity: max rows to cache per table (0 = auto-size based on first batch)
 */
void pipeline_init(
    const std::vector<int64_t>& compressed_table_indices,
    const std::vector<std::string>& frame_dirs,
    const std::vector<int64_t>& rows_per_frame,
    const std::vector<double>& cold_scales,
    const std::vector<double>& cold_zps,
    const std::vector<int64_t>& n_colds,
    const std::vector<torch::Tensor>& is_hot_list,
    const std::vector<torch::Tensor>& o2c_map_list,
    int64_t budget)
{
    TORCH_CHECK(g_registered, "Call register_tables first");
    int64_t K = compressed_table_indices.size();

    g_pipeline.compressed_tables = compressed_table_indices;
    g_pipeline.table_info.resize(K);
    g_pipeline.budget = budget;

    for (int64_t k = 0; k < K; k++) {
        int64_t t = compressed_table_indices[k];
        auto& info = g_pipeline.table_info[k];
        info.frame_dir = frame_dirs[k];
        info.rpf = rows_per_frame[k];
        info.cold_scale = static_cast<float>(cold_scales[k]);
        info.cold_zp = static_cast<float>(cold_zps[k]);
        info.n_cold = n_colds[k];
        info.D = g_tables[t].D;
        info.is_hot = is_hot_list[k];
        info.o2c_map = o2c_map_list[k];

        // Auto-detect frame extension
        info.frame_ext = ".h265";
        for (const char* ext : {".h265", ".mkv", ".zst"}) {
            std::string test = info.frame_dir + "/frame_00000" + ext;
            FILE* f = fopen(test.c_str(), "rb");
            if (f) { fclose(f); info.frame_ext = ext; break; }
        }

        // Setup dynamic frame mode for this table (used by both
        // pipeline_forward and fast_forward_seq_pipelined)
        auto& tab = g_tables[t];
        tab.cold_dynamic = true;
        tab.has_cold_frames = true;
        tab.rows_per_frame = info.rpf;
        tab.cold_scale = info.cold_scale;
        tab.cold_zp = info.cold_zp;

        // Register cold_mapping so COLD_FROM_BITMAP can route cold indices
        tab.cold_mapping = info.o2c_map;
        tab.cold_mapping_ptr = info.o2c_map.data_ptr<int32_t>();
        tab.has_cold_mapping = true;

        // Pre-size per-frame pointer arrays
        int64_t max_frames = (info.n_cold + info.rpf - 1) / info.rpf + 1;
        tab.dyn_frame_ptrs.assign(max_frames, nullptr);
        tab.dyn_frame_tensors.resize(max_frames);

        // Enable tiled storage (skip untiling) for PURE H.265 tables with D=16
        // Disabled for mixed-codec tables (Zstd hot + H.265 cold) since Zstd is flat
        if (tab.D == 16 && info.frame_ext != ".zst" && info.frame_ext == ".h265") {
            tab.dyn_tiled = true;
            // Width from resolution: rpf = (W/4) * (H/4) for 4x4 tiles
            // For 1080p: rpf = 480 * 270 = 129600, W=1920
            // Derive width from rpf and known height
            // Common: 1080p→1920, 480p→640, 4K→3840
            if (info.rpf == 129600) tab.dyn_width = 1920;
            else if (info.rpf == 19200) tab.dyn_width = 640;
            else if (info.rpf == 518400) tab.dyn_width = 3840;
            else tab.dyn_tiled = false;  // unknown resolution
        }
    }

    // Pre-load ALL compressed frame files into memory (only ~1.6MB for H.265)
    // This eliminates file I/O during decode, reducing multi-table decode from 17ms to ~5ms
    g_pipeline.inmem_frames.clear();
    int64_t total_bytes = 0;
    for (int64_t k = 0; k < K; k++) {
        auto& info = g_pipeline.table_info[k];
        int64_t max_frames = (info.n_cold + info.rpf - 1) / info.rpf + 1;
        for (int64_t fid = 0; fid < max_frames; fid++) {
            char fid_str[16];
            snprintf(fid_str, sizeof(fid_str), "%05ld", (long)fid);
            std::string path;
            FILE* f = nullptr;
            for (const char* ext : {".zst", ".h265", ".mkv"}) {
                path = info.frame_dir + "/frame_" + fid_str + ext;
                f = fopen(path.c_str(), "rb");
                if (f) break;
            }
            if (!f) continue;
            fseek(f, 0, SEEK_END);
            size_t sz = ftell(f);
            fseek(f, 0, SEEK_SET);
            auto buf = torch::empty({(int64_t)sz}, torch::kUInt8);
            fread(buf.data_ptr<uint8_t>(), 1, sz, f);
            fclose(f);
            g_pipeline.inmem_frames[k * 100000 + fid] = buf;
            total_bytes += sz;
        }
    }
    fprintf(stderr, "[C++] Pre-loaded %zu compressed frames into memory (%.1f MB)\n",
            g_pipeline.inmem_frames.size(), total_bytes / (1024.0 * 1024.0));

    g_pipeline.initialized = true;
}

/**
 * pipeline_add_frame — Add a decoded frame to the dynamic cache (O(1)).
 * frame_data must be (rpf, D) uint8 (already untiled).
 */
static void pipeline_add_frame(int64_t table_idx, int64_t frame_id,
                                torch::Tensor frame_data) {
    auto& tab = g_tables[table_idx];
    TORCH_CHECK(tab.cold_dynamic, "Table not in dynamic mode");

    if (frame_id >= (int64_t)tab.dyn_frame_ptrs.size()) {
        // Grow arrays
        int64_t new_size = frame_id + 16;
        tab.dyn_frame_ptrs.resize(new_size, nullptr);
        tab.dyn_frame_tensors.resize(new_size);
    }

    // Ensure contiguous uint8
    auto data = frame_data.contiguous().to(torch::kUInt8);

    if (tab.dyn_tiled) {
        // Store tiled (H, W) directly — skip untiling (saves ~6ms per batch)
        // cold_frame_accum will compute tile coordinates on the fly
    } else if (data.dim() == 2 && data.size(0) == tab.rows_per_frame) {
        // (rpf, D) — good
    } else if (data.dim() == 1) {
        data = data.view({tab.rows_per_frame, tab.D});
    } else {
        // Tiled frame (H, W) — untile
        data = untile_frame_to_rows(data, tab.rows_per_frame);
    }

    tab.dyn_frame_tensors[frame_id] = data;       // keep alive
    tab.dyn_frame_ptrs[frame_id] = data.data_ptr<uint8_t>();
}

/**
 * pipeline_remove_frame — Remove a frame from the dynamic cache (O(1)).
 */
static void pipeline_remove_frame(int64_t table_idx, int64_t frame_id) {
    auto& tab = g_tables[table_idx];
    if (frame_id < (int64_t)tab.dyn_frame_ptrs.size()) {
        tab.dyn_frame_ptrs[frame_id] = nullptr;
        tab.dyn_frame_tensors[frame_id] = torch::Tensor();
    }
}

/**
 * pipeline_scan_needed — Find which frames are needed for a batch.
 * Returns vector of (table_index_in_pipeline, set<frame_id>).
 */
static std::vector<std::unordered_set<int64_t>> pipeline_scan(
    const torch::Tensor& lS_i  // (T, N) int64
) {
    int64_t K = g_pipeline.compressed_tables.size();
    std::vector<std::unordered_set<int64_t>> needed(K);

    for (int64_t k = 0; k < K; k++) {
        int64_t t = g_pipeline.compressed_tables[k];
        auto& info = g_pipeline.table_info[k];
        const int64_t* idx_ptr = lS_i[t].data_ptr<int64_t>();
        int64_t N = lS_i.size(1);
        const bool* hot_ptr = info.is_hot.data_ptr<bool>();
        const int32_t* o2c_ptr = info.o2c_map.data_ptr<int32_t>();

        for (int64_t i = 0; i < N; i++) {
            int64_t orig = idx_ptr[i];
            if (!hot_ptr[orig]) {
                int32_t cold_idx = o2c_ptr[orig];
                if (cold_idx >= 0) {
                    int64_t fid = cold_idx / info.rpf;
                    needed[k].insert(fid);
                }
            }
        }
    }
    return needed;
}

/**
 * pipeline_prescan_and_decode — Scan batch for cold rows, decode missing frames,
 * extract needed rows into per-table row cache.
 *
 * Flow:
 *   1. Scan batch indices → collect (table, cold_rank, frame_id) for uncached rows
 *   2. Group by (table, frame_id) → unique frames to decode
 *   3. Decode missing frames in parallel (one thread per frame)
 *   4. Extract needed rows from decoded frames → add to row cache
 *   5. Discard decoded frames (only keep the extracted rows)
 */
static void pipeline_prescan_and_decode(const torch::Tensor& lS_i) {
    int64_t K = g_pipeline.compressed_tables.size();

    // Per-table: frame_id → set of (cold_rank, row_in_frame) needing cache
    struct RowNeed {
        int64_t cold_rank;
        int64_t row_in_frame;
    };
    // frames_to_decode[k] = {frame_id → vector of RowNeed}
    std::vector<std::unordered_map<int64_t, std::vector<RowNeed>>> frames_to_decode(K);

    // Step 1: Scan batch for frames with uncached data
    for (int64_t k = 0; k < K; k++) {
        int64_t t = g_pipeline.compressed_tables[k];
        auto& tab = g_tables[t];
        auto& info = g_pipeline.table_info[k];
        const int64_t* idx_ptr = lS_i[t].data_ptr<int64_t>();
        int64_t N = lS_i.size(1);
        const bool* hot_ptr = info.is_hot.data_ptr<bool>();
        const int32_t* o2c_ptr = info.o2c_map.data_ptr<int32_t>();

        for (int64_t i = 0; i < N; i++) {
            int64_t orig = idx_ptr[i];
            if (hot_ptr[orig]) continue;
            int32_t cold_rank = o2c_ptr[orig];
            if (cold_rank < 0) continue;

            int64_t fid = cold_rank / info.rpf;

            // Check if frame is in dynamic cache
            if (tab.cold_dynamic &&
                fid < (int64_t)tab.dyn_frame_ptrs.size() &&
                tab.dyn_frame_ptrs[fid] != nullptr) continue;

            int64_t row_in_frame = cold_rank % info.rpf;
            frames_to_decode[k][fid].push_back({cold_rank, row_in_frame});
        }
    }

    // Step 2: Collect unique frames to decode
    struct DecodeJob {
        int64_t k;      // pipeline table index
        int64_t fid;    // frame id
    };
    std::vector<DecodeJob> jobs;
    for (int64_t k = 0; k < K; k++) {
        for (auto& [fid, rows] : frames_to_decode[k]) {
            jobs.push_back({k, fid});
        }
    }

    if (jobs.empty()) return;

    // Step 3: Batch decode with pool (fastest path)
    // Try multiple extensions per frame (supports mixed Zstd + H.265 per table)
    std::vector<std::string> paths(jobs.size());
    for (size_t i = 0; i < jobs.size(); i++) {
        auto& info = g_pipeline.table_info[jobs[i].k];
        std::string base = info.frame_dir + "/frame_";
        char fid_str[16];
        snprintf(fid_str, sizeof(fid_str), "%05ld", (long)jobs[i].fid);
        // Try preferred extension first, then alternatives
        paths[i] = base + fid_str + info.frame_ext;
        if (access(paths[i].c_str(), F_OK) != 0) {
            for (const char* ext : {".zst", ".h265", ".mkv"}) {
                std::string alt = base + fid_str + ext;
                if (access(alt.c_str(), F_OK) == 0) {
                    paths[i] = alt;
                    break;
                }
            }
        }
    }
    auto decoded = batch_decode_fast(paths, 2, paths.size(), true, false, false);

    // Step 4: Add decoded frames to dynamic pointer cache (O(1) per frame)
    for (size_t i = 0; i < jobs.size(); i++) {
        if (!decoded[i].defined()) continue;
        int64_t t = g_pipeline.compressed_tables[jobs[i].k];
        pipeline_add_frame(t, jobs[i].fid, decoded[i]);
    }
}

/**
 * pipeline_forward — Pipelined inference: run fast_forward on current batch
 * while background-decoding frames for the next batch.
 *
 * On each call:
 *   1. Wait for background decode from previous call (if any)
 *   2. Scan CURRENT batch for missing frames → decode synchronously (ensures correctness)
 *   3. Launch background decode for NEXT batch's missing frames
 *   4. Run fast_forward on current batch
 *
 * First batch is slower (no prior prefetch → synchronous decode of all needed frames).
 * Subsequent batches benefit from prefetch (background decode already finished).
 *
 * Args:
 *   lS_i: (T, N) int64 — current batch indices
 *   lS_o: (T, B) int64 — current batch offsets
 *   next_lS_i: (T, N) int64 — NEXT batch indices (for prefetch), or empty
 *
 * Returns: same as fast_forward
 */
std::vector<torch::Tensor> pipeline_forward(
    const torch::Tensor& lS_i,
    const torch::Tensor& lS_o,
    const torch::Tensor& next_lS_i)  // empty tensor if no next batch
{
    TORCH_CHECK(g_pipeline.initialized, "Call pipeline_init first");

    // 1. Wait for background decode from previous call
    if (g_pipeline.bg_future.valid()) {
        g_pipeline.bg_future.get();
    }

    // 2. Scan CURRENT batch — decode frames for uncached rows, extract rows
    pipeline_prescan_and_decode(lS_i);  // no-op if all rows cached

    // 3. Launch background prescan+decode for NEXT batch
    if (next_lS_i.numel() > 0) {
        // Copy tensor since lambda outlives this scope
        torch::Tensor next_copy = next_lS_i.clone();
        g_pipeline.bg_future = std::async(std::launch::async, [next_copy]() {
            pipeline_prescan_and_decode(next_copy);
        });
    }

    // 4. Run fast_forward — cold_frame_accum checks row cache (all hits)
    return fast_forward(lS_i, lS_o);
}

/**
 * pipeline_warmup — Synchronously decode rows needed by first batch.
 */
void pipeline_warmup(const torch::Tensor& lS_i) {
    TORCH_CHECK(g_pipeline.initialized, "Call pipeline_init first");
    pipeline_prescan_and_decode(lS_i);
}

/**
 * pipeline_clear_frames — Clear all dynamic frame pointers for all tables.
 * Used by no-cache pipeline to discard decoded frames after each batch.
 */
static void pipeline_clear_frames() {
    for (int64_t k = 0; k < (int64_t)g_pipeline.compressed_tables.size(); k++) {
        int64_t t = g_pipeline.compressed_tables[k];
        auto& tab = g_tables[t];
        if (tab.cold_dynamic) {
            for (size_t i = 0; i < tab.dyn_frame_ptrs.size(); i++) {
                tab.dyn_frame_ptrs[i] = nullptr;
                tab.dyn_frame_tensors[i] = torch::Tensor();
            }
        }
    }
}

/**
 * pipeline_forward_nocache — Decode-every-batch pipeline with NO persistent cache.
 *
 * Flow:
 *   1. Wait for background decode of THIS batch's frames (started during prev batch)
 *   2. Run fast_forward (cold lookups use decoded dynamic frame pointers)
 *   3. Clear all decoded frames (reclaim memory)
 *   4. Start background decode of NEXT batch's frames
 *
 * First batch: synchronous decode (no prior prefetch).
 * Subsequent batches: decode overlaps with previous batch's inference.
 *
 * Runtime memory: only mapping (12MB) + hot (22MB) + compressed frames (1.6MB) = ~36MB
 * No decoded frame cache between batches.
 */
std::vector<torch::Tensor> pipeline_forward_nocache(
    const torch::Tensor& lS_i,
    const torch::Tensor& lS_o,
    const torch::Tensor& next_lS_i)
{
    TORCH_CHECK(g_pipeline.initialized, "Call pipeline_init first");

    // 1. Wait for background decode from previous call
    if (g_pipeline.bg_future.valid()) {
        g_pipeline.bg_future.get();
    }

    // 2. If no frames decoded yet (first batch), decode synchronously
    {
        // Check if ANY frame is cached for first compressed table
        int64_t t0 = g_pipeline.compressed_tables[0];
        bool any_cached = false;
        for (size_t i = 0; i < g_tables[t0].dyn_frame_ptrs.size() && !any_cached; i++)
            if (g_tables[t0].dyn_frame_ptrs[i]) any_cached = true;
        if (!any_cached) {
            pipeline_prescan_and_decode(lS_i);
        }
    }

    // 3. Run fast_forward using decoded frames
    auto result = fast_forward(lS_i, lS_o);

    // 4. Clear all decoded frames (release memory)
    pipeline_clear_frames();

    // 5. Start background decode for NEXT batch
    if (next_lS_i.numel() > 0) {
        torch::Tensor next_copy = next_lS_i.clone();
        g_pipeline.bg_future = std::async(std::launch::async, [next_copy]() {
            pipeline_prescan_and_decode(next_copy);
        });
    }

    return result;
}


/**
 * pipeline_forward_full — All-in-one per-batch: decompress hot + decode cold + forward.
 *
 * Eliminates Python overhead between decode and forward steps.
 * Everything runs in C++ with no Python calls in between.
 *
 * Args:
 *   lS_i, lS_o: current batch indices/offsets
 *   next_lS_i: next batch indices for background cold prefetch
 *   hot_zstd_paths: paths to Zstd-compressed hot tensors (one per compressed table)
 *   hot_table_indices: which global table index each hot path corresponds to
 */
std::vector<torch::Tensor> pipeline_forward_full(
    const torch::Tensor& lS_i,
    const torch::Tensor& lS_o,
    const torch::Tensor& next_lS_i,
    const std::vector<std::string>& hot_zstd_paths,
    const std::vector<int64_t>& hot_table_indices)
{
    TORCH_CHECK(g_pipeline.initialized, "Call pipeline_init first");

    // 1. Wait for background cold decode from previous batch
    if (g_pipeline.bg_future.valid()) {
        g_pipeline.bg_future.get();
    }

    // 2. Decompress hot Zstd tensors (C++ parallel threads)
    if (!hot_zstd_paths.empty()) {
        // batch_decode_file_paths auto-detects .zst, uses parallel std::threads
        auto hot_decoded = batch_decode_file_paths(
            hot_zstd_paths, 1, hot_zstd_paths.size(), false, false, false);

        // Update hot weight pointers in g_tables
        for (size_t i = 0; i < hot_table_indices.size() && i < hot_decoded.size(); i++) {
            int64_t t = hot_table_indices[i];
            if (t >= 0 && t < (int64_t)g_tables.size()) {
                auto& tab = g_tables[t];
                int64_t D = tab.D;
                auto reshaped = hot_decoded[i].view({-1, D});
                tab.weight = reshaped;
            }
        }
    }

    // 3. Decode cold frames for current batch (if not already done by bg thread)
    pipeline_prescan_and_decode(lS_i);

    // 4. Run fast_forward
    auto result = fast_forward(lS_i, lS_o);

    // 5. Clear cold frames (free memory)
    pipeline_clear_frames();

    // 6. Start background cold decode for next batch
    if (next_lS_i.numel() > 0) {
        torch::Tensor next_copy = next_lS_i.clone();
        g_pipeline.bg_future = std::async(std::launch::async, [next_copy]() {
            pipeline_prescan_and_decode(next_copy);
        });
    }

    return result;
}


/**
 * embedding_bag_backward_sum — Fast C++ backward for EmbeddingBag (sum mode).
 *
 * Given grad_output (T, B, D) from the loss backward, compute per-row gradients
 * and apply them to the master weight tensors via index_add_.
 *
 * For each table t:
 *   For each index i in bag b: grad_weight[indices[i]] += lr * grad_output[t][b]
 *
 * This replaces the slow Python loop over bags and tables.
 */
void embedding_bag_backward_sum(
    const torch::Tensor& grad_output,  // (T, B, D)
    const std::vector<torch::Tensor>& lS_i_list,  // T tensors, each (N,) int64
    const std::vector<torch::Tensor>& lS_o_list,  // T tensors, each (B,) int64
    std::vector<torch::Tensor>& master_weights,   // T tensors, each (num_emb, D) fp32
    double lr)
{
    const int64_t T = lS_i_list.size();
    const int64_t B = grad_output.size(1);
    const int64_t D = grad_output.size(2);

    at::parallel_for(0, T, 1, [&](int64_t t_begin, int64_t t_end) {
        for (int64_t t = t_begin; t < t_end; t++) {
            const auto& indices = lS_i_list[t];
            const auto& offsets = lS_o_list[t];
            const int64_t N = indices.size(0);
            const int64_t* idx_ptr = indices.data_ptr<int64_t>();
            const int64_t* off_ptr = offsets.data_ptr<int64_t>();
            const float* grad_ptr = grad_output[t].data_ptr<float>();
            float* w_ptr = master_weights[t].data_ptr<float>();
            float neg_lr = static_cast<float>(-lr);

            for (int64_t b = 0; b < B; b++) {
                int64_t start = off_ptr[b];
                int64_t end = (b + 1 < B) ? off_ptr[b + 1] : N;
                const float* g = grad_ptr + b * D;
                for (int64_t i = start; i < end; i++) {
                    int64_t row = idx_ptr[i];
                    float* w = w_ptr + row * D;
                    for (int64_t d = 0; d < D; d++) {
                        w[d] += neg_lr * g[d];
                    }
                }
            }
        }
    });
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
    m.def("embedding_bag_backward_sum", &embedding_bag_backward_sum,
          "Fast C++ EmbeddingBag backward (sum mode): scatter grad to master weights",
          py::arg("grad_output"), py::arg("lS_i_list"), py::arg("lS_o_list"),
          py::arg("master_weights"), py::arg("lr"),
          py::call_guard<py::gil_scoped_release>());
    m.def("update_table_weight", [](int64_t table_idx, const torch::Tensor& new_weight) {
              TORCH_CHECK(g_registered && table_idx < (int64_t)g_tables.size(),
                         "Invalid table index");
              g_tables[table_idx].weight = new_weight;
          }, "Swap hot weight pointer for a registered table (for per-batch hot decompress)",
          py::arg("table_idx"), py::arg("new_weight"));
    m.def("register_tables", &register_tables,
          "Register all table metadata in C++ for fast_forward",
          py::arg("table_kinds"), py::arg("weights"), py::arg("mappings"),
          py::arg("scales"), py::arg("zero_points"), py::arg("use_hash_table") = false,
          py::arg("use_bitmap") = false);
    m.def("fast_forward", &fast_forward,
          "Process all registered tables with a single call (zero Python loop overhead)");
    m.def("fast_forward_seq", &fast_forward_seq,
          "Sequential tables, parallel batches within each table (matches baseline parallelism)",
          py::call_guard<py::gil_scoped_release>());
    m.def("fast_forward_seq_pipelined", &fast_forward_seq_pipelined,
          "Sequential tables with per-table frame decode pipeline (low memory)",
          py::call_guard<py::gil_scoped_release>());
    m.def("register_cold_frames_for_table", &register_cold_frames_for_table,
          "Register pre-decoded cold frames for a table (enables full C++ cold lookup)",
          py::arg("table_idx"), py::arg("frame_ids"), py::arg("frame_data"),
          py::arg("cold_scale"), py::arg("cold_zp"), py::arg("rows_per_frame"),
          py::arg("cold_mapping") = torch::Tensor());
    m.def("register_cold_flat", &register_cold_flat,
          "Register flat natural-order cold buffer (no cold_mapping needed, uses bitmap cold_rank)",
          py::arg("table_idx"), py::arg("cold_flat_data"),
          py::arg("cold_scale"), py::arg("cold_zp"), py::arg("n_cold_rows"));
    m.def("register_cold_sparse_flat", &register_cold_sparse_flat,
          "Register sparse flat cold buffer (only cached rows, validity bitmap+rank for O(1) lookup)",
          py::arg("table_idx"), py::arg("cold_data"), py::arg("valid_cold_ranks"),
          py::arg("cold_scale"), py::arg("cold_zp"), py::arg("n_cold_total"));
    m.def("register_cold_dct", &register_cold_dct,
          "Register DCT-domain cold data for compressed-domain embedding lookup",
          py::arg("table_idx"), py::arg("dc_values"), py::arg("ac_data"),
          py::arg("ac_positions"), py::arg("ac_block_offsets"),
          py::arg("step_size"), py::arg("quant_scale"), py::arg("quant_zp"),
          py::arg("n_cold_rows"), py::arg("block_size") = 8);
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
          "Decode single H.265 frame from file, returns (H,W) uint8 tensor",
          py::arg("path"), py::arg("num_threads") = 0,
          py::arg("skip_loop_filter") = false,
          py::arg("skip_idct") = false,
          py::arg("fast_decode") = false,
          py::call_guard<py::gil_scoped_release>());
    m.def("decode_h265_frame_from_bytes", &decode_h265_frame_from_bytes,
          "Decode single H.265 frame from in-memory bytes, returns (H,W) uint8 tensor");
    m.def("decode_h265_gather_dequant", &decode_h265_gather_dequant,
          "Fused: decode H.265 from file + gather specific rows + dequant to fp32");
    m.def("batch_decode_gather_dequant", &batch_decode_gather_dequant,
          "Batch: decode multiple H.265 frames in parallel + vectorized gather + dequant");
    m.def("batch_decode_frames", &batch_decode_frames,
          "Decode multiple H.265 frames in parallel, return list of tiled (H,W) tensors");
    m.def("batch_decode_file_paths", &batch_decode_file_paths,
          "Decode frames from explicit file paths with controlled parallelism",
          py::arg("paths"), py::arg("num_threads_per_decode") = 1,
          py::arg("max_parallel") = 0,
          py::arg("skip_loop_filter") = false,
          py::arg("skip_idct") = false,
          py::arg("fast_decode") = false,
          py::call_guard<py::gil_scoped_release>());
    m.def("decode_hevc_file_fast", &decode_hevc_file_fast,
          "Fast single-frame decode: mode 0=original, 2=raw, 3=raw+pool",
          py::arg("path"), py::arg("mode") = 3,
          py::arg("skip_loop_filter") = false,
          py::arg("skip_idct") = false,
          py::arg("fast_decode") = false,
          py::call_guard<py::gil_scoped_release>());
    m.def("batch_decode_fast", &batch_decode_fast,
          "Batch decode with raw parsing + context pool",
          py::arg("paths"), py::arg("mode") = 3,
          py::arg("max_parallel") = 0,
          py::arg("skip_loop_filter") = false,
          py::arg("skip_idct") = false,
          py::arg("fast_decode") = false,
          py::call_guard<py::gil_scoped_release>());
    // Pipeline functions
    m.def("pipeline_init", &pipeline_init,
          "Init pipelined decode with per-table frame dirs",
          py::arg("compressed_table_indices"),
          py::arg("frame_dirs"), py::arg("rows_per_frame"),
          py::arg("cold_scales"), py::arg("cold_zps"),
          py::arg("n_colds"),
          py::arg("is_hot_list"), py::arg("o2c_map_list"),
          py::arg("budget") = 9999);
    m.def("pipeline_warmup", &pipeline_warmup,
          "Decode all frames needed by first batch",
          py::arg("lS_i"),
          py::call_guard<py::gil_scoped_release>());
    m.def("pipeline_forward", &pipeline_forward,
          "Pipelined forward: fast_forward + background decode of next batch",
          py::arg("lS_i"), py::arg("lS_o"), py::arg("next_lS_i"),
          py::call_guard<py::gil_scoped_release>());
    m.def("pipeline_forward_full", &pipeline_forward_full,
          "All-in-one: decompress hot Zstd + decode cold H.265 + fast_forward (min Python overhead)",
          py::arg("lS_i"), py::arg("lS_o"), py::arg("next_lS_i"),
          py::arg("hot_zstd_paths"), py::arg("hot_table_indices"),
          py::call_guard<py::gil_scoped_release>());
    m.def("pipeline_forward_nocache", &pipeline_forward_nocache,
          "No-cache pipeline: decode every batch, discard after use (min memory)",
          py::arg("lS_i"), py::arg("lS_o"), py::arg("next_lS_i"),
          py::call_guard<py::gil_scoped_release>());
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
    // Fused gather from cached cold frames
    m.def("gather_cold_embeddings", &gather_cold_embeddings,
          "Gather cold embeddings from cached frames for all tables (fused C++)",
          py::arg("lS_i_list"), py::arg("is_hot_list"),
          py::arg("o2c_map_list"), py::arg("cached_frames_list"),
          py::arg("frame_offsets_list"), py::arg("rows_per_frame"),
          py::arg("emb_dim"));
    // Fused scan + gather + writeback (no Python intermediate)
    m.def("scatter_cold_to_weights", &scatter_cold_to_weights,
          "Scan, gather, and write cold embeddings directly into weight tensors",
          py::arg("lS_i_list"), py::arg("is_hot_list"),
          py::arg("o2c_map_list"), py::arg("cached_frames_list"),
          py::arg("weight_list"), py::arg("rows_per_frame"),
          py::arg("emb_dim"));
    // Zstd compression/decompression
    m.def("zstd_compress_frame", &zstd_compress_frame,
          "Compress uint8 tensor with Zstd (lossless)",
          py::arg("data"), py::arg("level") = 3);
    m.def("zstd_decompress_frame", &zstd_decompress_frame,
          "Decompress Zstd data back to uint8 tensor",
          py::arg("compressed"), py::arg("original_size"));
    m.def("batch_zstd_compress", &batch_zstd_compress,
          "Compress multiple frames in parallel, write .zst files",
          py::arg("frames"), py::arg("output_dir"), py::arg("level") = 3,
          py::call_guard<py::gil_scoped_release>());
    m.def("batch_zstd_decompress_files", &batch_zstd_decompress_files,
          "Decompress multiple .zst files in parallel",
          py::arg("paths"), py::arg("original_size"),
          py::arg("max_parallel") = 0,
          py::call_guard<py::gil_scoped_release>());
}
