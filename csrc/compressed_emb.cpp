// compressed_emb.cpp — C++ PyTorch extension for CompressedEmbeddingBag forward pass
// Replaces Python hot/cold routing + scatter_add with efficient C++ implementation.

#include <torch/extension.h>
#include <vector>

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

// Version that returns both output and cold_mask
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

    return {output, cold_mask};
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


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compressed_emb_bag_forward", &compressed_emb_bag_forward,
          "Compressed EmbeddingBag forward (hot path in C++, returns cold_mask)");
    m.def("cold_fixup", &cold_fixup,
          "Add cold embeddings into output in-place");
}
