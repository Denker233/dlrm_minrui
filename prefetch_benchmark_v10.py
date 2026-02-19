#!/usr/bin/env python3
"""
Prefetch Benchmark V10: Entropy Coding Optimizations

Builds on V9 (per-batch uint8 dequant + prefetch) to explore further optimizations:
  1. Baseline — streaming, no compression
  2. V9 reference — uint8, 80% hot, per-batch prefetch
  3. Lazy materialization — skip re-injection for already-materialized cold rows
  4. 4-bit quantization — pack 2 values/byte, ~257MB cold storage
  5. Block zlib compression — actual entropy coding with block cache
  6. Hot threshold 95% — fewer cold injections
  7. Torch-native dequant — all ops in torch, no numpy conversion
  8. Combined — lazy + torch-native + hot 95%
"""

import os, sys, time, json, threading, gc, zlib, subprocess, struct
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dlrm_data_pytorch as dp

MODEL_PATH = "./models/dlrm_kaggle_1epoch.pt"
PROCESSED_DATA = "./input/kaggleAdDisplayChallenge_processed.npz"
DATA_FILE = "./input/train.txt"
RESULTS_DIR = os.path.expanduser("~/experiment-control")

ARCH_SPARSE_FEATURE_SIZE = 16
ARCH_MLP_BOT = "13-512-256-64-16"
ARCH_MLP_TOP = "512-256-1"
TEST_BATCH_SIZE = 2048
EMB_DIM = 16
HOT_THRESHOLD_80 = 0.80
HOT_THRESHOLD_95 = 0.95
PROFILE_BATCHES = 200
LARGE_TABLE_THRESHOLD = 50000
ZLIB_BLOCK_SIZE = 4096  # rows per block

os.makedirs(RESULTS_DIR, exist_ok=True)


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def drop_caches():
    try:
        subprocess.run(['sync'], check=True, timeout=30)
        subprocess.run(['sudo', 'sh', '-c', 'echo 3 > /proc/sys/vm/drop_caches'],
                       check=True, timeout=30)
    except Exception:
        pass


def create_args():
    class Args: pass
    a = Args()
    a.arch_sparse_feature_size = ARCH_SPARSE_FEATURE_SIZE
    a.arch_mlp_bot = ARCH_MLP_BOT; a.arch_mlp_top = ARCH_MLP_TOP
    a.arch_interaction_op = "dot"; a.arch_interaction_itself = False
    a.data_generation = "dataset"; a.data_set = "kaggle"
    a.raw_data_file = DATA_FILE; a.processed_data_file = PROCESSED_DATA
    a.loss_function = "bce"; a.max_ind_range = -1
    a.test_mini_batch_size = TEST_BATCH_SIZE; a.test_num_workers = 0; a.num_workers = 0
    a.mlperf_logging = False; a.memory_map = False; a.data_randomize = "total"
    a.data_trace_enable_padding = False; a.data_sub_sample_rate = 0.0
    a.num_indices_per_lookup = 10; a.num_indices_per_lookup_fixed = False
    a.mini_batch_size = 128; a.round_targets = True
    a.mlperf_bin_loader = False; a.mlperf_bin_shuffle = False
    a.dataset_multiprocessing = False
    return a


def load_model_and_data():
    from dlrm_s_pytorch import DLRM_Net
    args = create_args()
    train_data, train_ld, test_data, test_ld = dp.make_criteo_data_and_loaders(args)
    ln_emb = np.array(train_data.counts)
    m_spa = args.arch_sparse_feature_size
    ln_bot = np.fromstring(args.arch_mlp_bot, dtype=int, sep="-")
    ln_bot[0] = train_data.m_den
    num_fea = ln_emb.size + 1
    m_den_out = ln_bot[ln_bot.size - 1]
    num_int = (num_fea * (num_fea - 1)) // 2 + m_den_out
    ln_top = np.fromstring(str(num_int) + "-" + args.arch_mlp_top, dtype=int, sep="-")
    dlrm = DLRM_Net(m_spa, ln_emb, ln_bot, ln_top,
                     arch_interaction_op="dot", arch_interaction_itself=False,
                     sigmoid_bot=-1, sigmoid_top=ln_top.size - 2, loss_function="bce")
    ld = torch.load(MODEL_PATH, map_location='cpu')
    dlrm.load_state_dict(ld["state_dict"])
    dlrm.eval()
    return dlrm, test_ld, train_ld, ln_emb


def quantize(w):
    mn, mx = w.min().item(), w.max().item()
    s = (mx - mn) / 255.0
    if s == 0: s = 1.0
    zp = round(-mn / s)
    q = ((w / s).round() + zp).clamp(0, 255).to(torch.uint8)
    return q, s, zp


def dequantize(q, s, zp):
    return (q.float() - zp) * s


def restore_weights(dlrm, state_dict, emb_keys):
    with torch.no_grad():
        for k in emb_keys:
            t = int(k.split('.')[1])
            dlrm.emb_l[t].weight.data = state_dict[k].clone()


def latency_stats(blats):
    a = np.array(blats)
    return {
        'count': len(a), 'mean_ms': float(np.mean(a) * 1000),
        'p50_ms': float(np.percentile(a, 50) * 1000),
        'p95_ms': float(np.percentile(a, 95) * 1000),
        'p99_ms': float(np.percentile(a, 99) * 1000),
    }


# ==============================================================
# COLD STORES
# ==============================================================

class EntropyColdStore:
    """V9 uint8 cold store with vectorized dequantization."""
    def __init__(self, large_tables):
        self.cold_uint8 = {}       # {t: np.array (num_cold, 16) uint8}
        self.cold_seq_lookup = {}  # {t: np.array (num_emb,) int32, -1 if not cold}
        self.quant_params = {}     # {t: (scale, zp)}
        self.large_tables = large_tables

    def add_table(self, t, num_emb, cold_indices, cold_weights):
        q, s, zp = quantize(cold_weights)
        self.cold_uint8[t] = q.numpy()
        self.quant_params[t] = (s, zp)
        lookup = np.full(num_emb, -1, dtype=np.int32)
        for seq, orig in enumerate(cold_indices):
            lookup[orig] = seq
        self.cold_seq_lookup[t] = lookup

    def dequantize_for_batch(self, lS_i):
        result = {}
        for t in self.large_tables:
            if t not in self.cold_uint8:
                continue
            indices = lS_i[t].numpy().flatten()
            unique = np.unique(indices)
            seq = self.cold_seq_lookup[t][unique]
            mask = seq >= 0
            if not mask.any():
                continue
            cold_orig = unique[mask]
            cold_seq = seq[mask]
            q_rows = self.cold_uint8[t][cold_seq]
            s, zp = self.quant_params[t]
            fp_rows = (q_rows.astype(np.float32) - zp) * s
            result[t] = (torch.from_numpy(cold_orig.astype(np.int64)),
                         torch.from_numpy(fp_rows))
        return result

    @staticmethod
    def inject(dlrm, result):
        with torch.no_grad():
            for t, (idx_tensor, val_tensor) in result.items():
                dlrm.emb_l[t].weight.data[idx_tensor] = val_tensor

    @property
    def memory_bytes(self):
        return sum(a.nbytes for a in self.cold_uint8.values())


class LazyColdStore:
    """uint8 cold store with lazy materialization — skip re-injection for already-injected rows."""
    def __init__(self, large_tables):
        self.cold_uint8 = {}
        self.cold_seq_lookup = {}
        self.quant_params = {}
        self.large_tables = large_tables
        self.materialized = {}     # {t: np.array (num_emb,) bool}
        self.total_injections = 0
        self.total_skipped = 0
        self.injection_counts = []  # per-batch injection count

    def add_table(self, t, num_emb, cold_indices, cold_weights):
        q, s, zp = quantize(cold_weights)
        self.cold_uint8[t] = q.numpy()
        self.quant_params[t] = (s, zp)
        lookup = np.full(num_emb, -1, dtype=np.int32)
        for seq, orig in enumerate(cold_indices):
            lookup[orig] = seq
        self.cold_seq_lookup[t] = lookup
        self.materialized[t] = np.zeros(num_emb, dtype=bool)

    def dequantize_for_batch(self, lS_i):
        result = {}
        batch_injections = 0
        batch_skipped = 0
        for t in self.large_tables:
            if t not in self.cold_uint8:
                continue
            indices = lS_i[t].numpy().flatten()
            unique = np.unique(indices)
            seq = self.cold_seq_lookup[t][unique]
            mask = seq >= 0
            if not mask.any():
                continue
            cold_orig = unique[mask]
            # Filter out already-materialized rows
            not_materialized = ~self.materialized[t][cold_orig]
            batch_skipped += int(np.sum(~not_materialized))
            if not not_materialized.any():
                continue
            new_cold = cold_orig[not_materialized]
            new_seq = self.cold_seq_lookup[t][new_cold]
            q_rows = self.cold_uint8[t][new_seq]
            s, zp = self.quant_params[t]
            fp_rows = (q_rows.astype(np.float32) - zp) * s
            result[t] = (torch.from_numpy(new_cold.astype(np.int64)),
                         torch.from_numpy(fp_rows))
            # Mark as materialized
            self.materialized[t][new_cold] = True
            batch_injections += len(new_cold)

        self.total_injections += batch_injections
        self.total_skipped += batch_skipped
        self.injection_counts.append(batch_injections)
        return result

    @staticmethod
    def inject(dlrm, result):
        EntropyColdStore.inject(dlrm, result)

    @property
    def memory_bytes(self):
        cold = sum(a.nbytes for a in self.cold_uint8.values())
        mat = sum(a.nbytes for a in self.materialized.values())
        return cold + mat


class FourBitColdStore:
    """4-bit quantization: pack 2 values per byte, ~half the memory of uint8."""
    def __init__(self, large_tables):
        self.cold_packed = {}      # {t: np.array (num_cold, 8) uint8}
        self.cold_seq_lookup = {}
        self.quant_params = {}     # {t: (scale, zp)}
        self.large_tables = large_tables

    @staticmethod
    def quantize_4bit(w):
        """Quantize to 0-15 range."""
        mn, mx = w.min().item(), w.max().item()
        s = (mx - mn) / 15.0
        if s == 0: s = 1.0
        zp = round(-mn / s)
        q = ((w / s).round() + zp).clamp(0, 15).to(torch.uint8)
        return q, s, zp

    @staticmethod
    def pack_4bit(q_np):
        """Pack pairs of 4-bit values into bytes. q_np shape: (N, 16) -> (N, 8)."""
        high = q_np[:, 0::2]  # even columns
        low = q_np[:, 1::2]   # odd columns
        packed = (high << 4) | low
        return packed.astype(np.uint8)

    @staticmethod
    def unpack_4bit(packed):
        """Unpack bytes to pairs of 4-bit values. packed shape: (N, 8) -> (N, 16)."""
        high = (packed >> 4) & 0x0F
        low = packed & 0x0F
        # Interleave: high goes to even columns, low to odd
        n = packed.shape[0]
        result = np.empty((n, 16), dtype=np.uint8)
        result[:, 0::2] = high
        result[:, 1::2] = low
        return result

    def add_table(self, t, num_emb, cold_indices, cold_weights):
        q, s, zp = self.quantize_4bit(cold_weights)
        packed = self.pack_4bit(q.numpy())
        self.cold_packed[t] = packed
        self.quant_params[t] = (s, zp)
        lookup = np.full(num_emb, -1, dtype=np.int32)
        for seq, orig in enumerate(cold_indices):
            lookup[orig] = seq
        self.cold_seq_lookup[t] = lookup

    def dequantize_for_batch(self, lS_i):
        result = {}
        for t in self.large_tables:
            if t not in self.cold_packed:
                continue
            indices = lS_i[t].numpy().flatten()
            unique = np.unique(indices)
            seq = self.cold_seq_lookup[t][unique]
            mask = seq >= 0
            if not mask.any():
                continue
            cold_orig = unique[mask]
            cold_seq = seq[mask]
            packed_rows = self.cold_packed[t][cold_seq]
            q_rows = self.unpack_4bit(packed_rows)
            s, zp = self.quant_params[t]
            fp_rows = (q_rows.astype(np.float32) - zp) * s
            result[t] = (torch.from_numpy(cold_orig.astype(np.int64)),
                         torch.from_numpy(fp_rows))
        return result

    @staticmethod
    def inject(dlrm, result):
        EntropyColdStore.inject(dlrm, result)

    @property
    def memory_bytes(self):
        return sum(a.nbytes for a in self.cold_packed.values())


class BlockZlibColdStore:
    """Block-based zlib compression with block cache for actual entropy coding."""
    def __init__(self, large_tables, block_size=ZLIB_BLOCK_SIZE):
        self.blocks = {}           # {t: [(compressed_bytes, num_rows), ...]}
        self.cold_seq_lookup = {}
        self.quant_params = {}
        self.large_tables = large_tables
        self.block_size = block_size
        self.block_cache = {}      # {(t, block_id): np.array}
        self.raw_bytes = 0
        self.compressed_bytes = 0
        self.decompress_count = 0
        self.decompress_time = 0.0

    def add_table(self, t, num_emb, cold_indices, cold_weights):
        q, s, zp = quantize(cold_weights)
        q_np = q.numpy()
        self.quant_params[t] = (s, zp)

        lookup = np.full(num_emb, -1, dtype=np.int32)
        for seq, orig in enumerate(cold_indices):
            lookup[orig] = seq
        self.cold_seq_lookup[t] = lookup

        # Compress into blocks
        n_rows = q_np.shape[0]
        table_blocks = []
        raw_total = 0
        comp_total = 0
        for start in range(0, n_rows, self.block_size):
            end = min(start + self.block_size, n_rows)
            block_data = q_np[start:end]
            raw = block_data.tobytes()
            compressed = zlib.compress(raw, 1)
            table_blocks.append((compressed, end - start))
            raw_total += len(raw)
            comp_total += len(compressed)
        self.blocks[t] = table_blocks
        self.raw_bytes += raw_total
        self.compressed_bytes += comp_total

    def _get_block(self, t, block_id):
        key = (t, block_id)
        if key in self.block_cache:
            return self.block_cache[key]
        t0 = time.time()
        compressed, num_rows = self.blocks[t][block_id]
        raw = zlib.decompress(compressed)
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(num_rows, EMB_DIM)
        self.block_cache[key] = arr
        self.decompress_count += 1
        self.decompress_time += time.time() - t0
        return arr

    def dequantize_for_batch(self, lS_i):
        result = {}
        for t in self.large_tables:
            if t not in self.blocks:
                continue
            indices = lS_i[t].numpy().flatten()
            unique = np.unique(indices)
            seq = self.cold_seq_lookup[t][unique]
            mask = seq >= 0
            if not mask.any():
                continue
            cold_orig = unique[mask]
            cold_seq = seq[mask]

            # Find which blocks we need
            block_ids = cold_seq // self.block_size
            unique_blocks = np.unique(block_ids)

            # Decompress needed blocks and extract rows
            q_rows = np.empty((len(cold_seq), EMB_DIM), dtype=np.uint8)
            for bid in unique_blocks:
                block_data = self._get_block(t, bid)
                block_mask = block_ids == bid
                local_offsets = cold_seq[block_mask] - bid * self.block_size
                q_rows[block_mask] = block_data[local_offsets]

            s, zp = self.quant_params[t]
            fp_rows = (q_rows.astype(np.float32) - zp) * s
            result[t] = (torch.from_numpy(cold_orig.astype(np.int64)),
                         torch.from_numpy(fp_rows))
        return result

    @staticmethod
    def inject(dlrm, result):
        EntropyColdStore.inject(dlrm, result)

    @property
    def memory_bytes(self):
        return self.compressed_bytes

    @property
    def compression_ratio(self):
        if self.compressed_bytes == 0:
            return 0.0
        return self.raw_bytes / self.compressed_bytes


class TorchNativeColdStore:
    """All dequant ops in torch — no numpy->torch conversion overhead."""
    def __init__(self, large_tables):
        self.cold_uint8_t = {}       # {t: torch.Tensor uint8 (num_cold, 16)}
        self.cold_seq_lookup_t = {}  # {t: torch.Tensor int32 (num_emb,)}
        self.quant_params = {}
        self.large_tables = large_tables

    def add_table(self, t, num_emb, cold_indices, cold_weights):
        q, s, zp = quantize(cold_weights)
        self.cold_uint8_t[t] = q  # keep as torch tensor
        self.quant_params[t] = (s, zp)
        lookup = torch.full((num_emb,), -1, dtype=torch.int32)
        for seq, orig in enumerate(cold_indices):
            lookup[orig] = seq
        self.cold_seq_lookup_t[t] = lookup

    def dequantize_for_batch(self, lS_i):
        result = {}
        for t in self.large_tables:
            if t not in self.cold_uint8_t:
                continue
            indices_t = lS_i[t].flatten()
            unique_t = torch.unique(indices_t)
            seq = self.cold_seq_lookup_t[t][unique_t.long()]
            mask = seq >= 0
            if not mask.any():
                continue
            cold_orig = unique_t[mask]
            cold_seq = seq[mask].long()
            q_rows = self.cold_uint8_t[t][cold_seq]
            s, zp = self.quant_params[t]
            fp_rows = (q_rows.float() - zp) * s
            result[t] = (cold_orig.long(), fp_rows)
        return result

    @staticmethod
    def inject(dlrm, result):
        EntropyColdStore.inject(dlrm, result)

    @property
    def memory_bytes(self):
        return sum(t.numel() for t in self.cold_uint8_t.values())


class CombinedColdStore:
    """Lazy materialization + torch-native dequant. Combines configs 3+7."""
    def __init__(self, large_tables):
        self.cold_uint8_t = {}
        self.cold_seq_lookup_t = {}
        self.quant_params = {}
        self.large_tables = large_tables
        self.materialized = {}     # {t: torch.Tensor bool (num_emb,)}
        self.total_injections = 0
        self.total_skipped = 0
        self.injection_counts = []

    def add_table(self, t, num_emb, cold_indices, cold_weights):
        q, s, zp = quantize(cold_weights)
        self.cold_uint8_t[t] = q
        self.quant_params[t] = (s, zp)
        lookup = torch.full((num_emb,), -1, dtype=torch.int32)
        for seq, orig in enumerate(cold_indices):
            lookup[orig] = seq
        self.cold_seq_lookup_t[t] = lookup
        self.materialized[t] = torch.zeros(num_emb, dtype=torch.bool)

    def dequantize_for_batch(self, lS_i):
        result = {}
        batch_injections = 0
        batch_skipped = 0
        for t in self.large_tables:
            if t not in self.cold_uint8_t:
                continue
            indices_t = lS_i[t].flatten()
            unique_t = torch.unique(indices_t)
            seq = self.cold_seq_lookup_t[t][unique_t.long()]
            mask = seq >= 0
            if not mask.any():
                continue
            cold_orig = unique_t[mask]
            # Filter out already-materialized
            not_mat = ~self.materialized[t][cold_orig.long()]
            batch_skipped += int((~not_mat).sum().item())
            if not not_mat.any():
                continue
            new_cold = cold_orig[not_mat]
            new_seq = self.cold_seq_lookup_t[t][new_cold.long()].long()
            q_rows = self.cold_uint8_t[t][new_seq]
            s, zp = self.quant_params[t]
            fp_rows = (q_rows.float() - zp) * s
            result[t] = (new_cold.long(), fp_rows)
            self.materialized[t][new_cold.long()] = True
            batch_injections += len(new_cold)

        self.total_injections += batch_injections
        self.total_skipped += batch_skipped
        self.injection_counts.append(batch_injections)
        return result

    @staticmethod
    def inject(dlrm, result):
        EntropyColdStore.inject(dlrm, result)

    @property
    def memory_bytes(self):
        cold = sum(t.numel() for t in self.cold_uint8_t.values())
        mat = sum(t.numel() for t in self.materialized.values())
        return cold + mat


# ==============================================================
# PREFETCHER
# ==============================================================

class PrefetchDequantizer:
    def __init__(self, store):
        self.store = store
        self._thread = None
        self._result = {}

    def prefetch_async(self, lS_i):
        if self._thread:
            self._thread.join()
        self._result = {}
        def work():
            self._result = self.store.dequantize_for_batch(lS_i)
        self._thread = threading.Thread(target=work)
        self._thread.start()

    def wait(self):
        if self._thread:
            self._thread.join()
            self._thread = None
        return self._result

    def fetch_sync(self, lS_i):
        return self.store.dequantize_for_batch(lS_i)


# ==============================================================
# SETUP FUNCTIONS
# ==============================================================

def setup_cold_store(dlrm, state_dict, emb_keys, ln_emb, num_tables,
                     large_tables, hot_indices, store):
    """Generic setup: small tables get uint8 round-trip, large tables split hot/cold."""
    for t in range(num_tables):
        w = state_dict[emb_keys[t]]
        if w.shape[0] < LARGE_TABLE_THRESHOLD:
            q, s, zp = quantize(w)
            with torch.no_grad():
                dlrm.emb_l[t].weight.data = dequantize(q, s, zp)

    for t in large_tables:
        w = state_dict[emb_keys[t]]
        hi = set(hot_indices[t].tolist())
        hot_idx = sorted(hi)
        if hot_idx:
            hw = w[hot_idx]; qh, sh, zh = quantize(hw)
            with torch.no_grad():
                dlrm.emb_l[t].weight.data[torch.tensor(hot_idx, dtype=torch.long)] = dequantize(qh, sh, zh)
        cold_idx = sorted(set(range(ln_emb[t])) - hi)
        cw = w[cold_idx]
        store.add_table(t, ln_emb[t], cold_idx, cw)
        log(f"    Table {t}: {len(hot_idx):,} hot, {len(cold_idx):,} cold")
    return store


# ==============================================================
# INFERENCE FUNCTIONS
# ==============================================================

def run_baseline_inference(dlrm, test_ld):
    scores, targets = [], []
    accu, samp = 0, 0
    blats = []
    nb = 0
    t0 = time.time()
    with torch.no_grad():
        for X, lS_o, lS_i, T in test_ld:
            bt0 = time.time()
            Z = dlrm(X, lS_o, lS_i)
            blats.append(time.time() - bt0)
            S = Z.detach().cpu().numpy().flatten()
            Tn = T.detach().cpu().numpy().flatten()
            accu += np.sum((np.round(S, 0) == Tn).astype(np.uint8))
            samp += Tn.shape[0]
            scores.extend(S.tolist()); targets.extend(Tn.tolist())
            nb += 1
            if nb % 500 == 0:
                log(f"    Batch {nb}, lat={blats[-1] * 1000:.1f}ms")
    total = time.time() - t0
    return accu / samp, roc_auc_score(targets, scores), total, blats, nb


def run_prefetch_inference(dlrm, test_ld, store):
    """Generic prefetch inference loop — works with any store that has dequantize_for_batch/inject."""
    prefetcher = PrefetchDequantizer(store)
    scores, targets = [], []
    accu, samp = 0, 0
    blats = []
    nb = 0

    t0 = time.time()
    dataloader_iter = iter(test_ld)

    try:
        current_batch = next(dataloader_iter)
    except StopIteration:
        return 0, 0, 0, [], 0

    _, _, lS_i_0, _ = current_batch
    current_result = prefetcher.fetch_sync(lS_i_0)

    try:
        lookahead_batch = next(dataloader_iter)
        _, _, la_lS_i, _ = lookahead_batch
        prefetcher.prefetch_async(la_lS_i)
        has_lookahead = True
    except StopIteration:
        has_lookahead = False

    while True:
        X, lS_o, lS_i, T = current_batch
        bt0 = time.time()

        store.inject(dlrm, current_result)

        with torch.no_grad():
            Z = dlrm(X, lS_o, lS_i)
        blats.append(time.time() - bt0)

        S = Z.detach().cpu().numpy().flatten()
        Tn = T.detach().cpu().numpy().flatten()
        accu += np.sum((np.round(S, 0) == Tn).astype(np.uint8))
        samp += Tn.shape[0]
        scores.extend(S.tolist()); targets.extend(Tn.tolist())
        nb += 1

        if nb % 500 == 0:
            log(f"    Batch {nb}, lat={blats[-1] * 1000:.1f}ms")

        if not has_lookahead:
            break

        next_result = prefetcher.wait()
        current_batch = lookahead_batch
        current_result = next_result

        try:
            lookahead_batch = next(dataloader_iter)
            _, _, la_lS_i, _ = lookahead_batch
            prefetcher.prefetch_async(la_lS_i)
            has_lookahead = True
        except StopIteration:
            has_lookahead = False

    total = time.time() - t0
    return accu / samp, roc_auc_score(targets, scores), total, blats, nb


# ==============================================================
# PROFILING
# ==============================================================

def profile_access(train_ld, num_tables, ln_emb, threshold):
    """Profile access patterns and compute hot indices at given threshold."""
    access_raw = [None] * num_tables
    for i, batch in enumerate(train_ld):
        if i >= PROFILE_BATCHES: break
        _, _, lS_i, _ = batch
        for t in range(num_tables):
            idx = lS_i[t].numpy().flatten()
            if access_raw[t] is None:
                access_raw[t] = idx.copy()
            else:
                access_raw[t] = np.concatenate([access_raw[t], idx])

    hot_indices = {}
    for t in range(num_tables):
        if access_raw[t] is None or len(access_raw[t]) == 0:
            hot_indices[t] = np.array([], dtype=np.int64)
            continue
        unique, counts = np.unique(access_raw[t], return_counts=True)
        si = np.argsort(-counts)
        cum = np.cumsum(counts[si])
        cutoff = np.searchsorted(cum, cum[-1] * threshold) + 1
        hot_indices[t] = unique[si[:cutoff]]
    return hot_indices


def compute_memory(dlrm, store, hot_indices, large_tables, state_dict,
                   emb_keys, num_tables, ln_emb):
    cold_mb = store.memory_bytes / 1024 / 1024
    hot_mb = sum(len(hot_indices[t]) * EMB_DIM * 4 for t in large_tables) / 1024 / 1024
    small_mb = sum(state_dict[emb_keys[t]].numel() * 4
                   for t in range(num_tables) if ln_emb[t] < LARGE_TABLE_THRESHOLD) / 1024 / 1024
    mlp_mb = sum(p.numel() * 4 for n, p in dlrm.named_parameters() if 'emb_l' not in n) / 1024 / 1024
    total_mb = hot_mb + small_mb + mlp_mb + cold_mb
    return {
        'hot_fp32_mb': hot_mb, 'small_fp32_mb': small_mb,
        'mlp_mb': mlp_mb, 'cold_store_mb': cold_mb, 'total_mb': total_mb,
    }


# ==============================================================
# RUN ONE CONFIG
# ==============================================================

def run_config(name, config_num, dlrm, test_ld, state_dict, emb_keys,
               ln_emb, num_tables, large_tables, hot_indices, baseline_auc,
               store_class, store_kwargs=None):
    """Run a single config: setup + inference + report."""
    log(f"\n{'=' * 60}")
    log(f"CONFIG {config_num}: {name}")
    log(f"{'=' * 60}")

    restore_weights(dlrm, state_dict, emb_keys)
    gc.collect()

    if store_kwargs is None:
        store_kwargs = {}
    store = store_class(large_tables, **store_kwargs)

    setup_t0 = time.time()
    setup_cold_store(dlrm, state_dict, emb_keys, ln_emb, num_tables,
                     large_tables, hot_indices, store)
    setup_time = time.time() - setup_t0

    mem = compute_memory(dlrm, store, hot_indices, large_tables,
                         state_dict, emb_keys, num_tables, ln_emb)
    log(f"  Setup: {setup_time:.2f}s")
    log(f"  Cold storage: {mem['cold_store_mb']:.1f}MB")

    drop_caches(); time.sleep(1); gc.collect()

    acc, auc, inf_time, blats, nb = run_prefetch_inference(dlrm, test_ld, store)
    auc_loss = (baseline_auc - auc) * 100

    log(f"  Acc={acc * 100:.4f}%, AUC={auc:.6f}")
    log(f"  Inference={inf_time:.2f}s, Setup={setup_time:.2f}s, Total={setup_time + inf_time:.2f}s")
    log(f"  Memory: hot={mem['hot_fp32_mb']:.1f}MB, small={mem['small_fp32_mb']:.1f}MB, "
        f"mlp={mem['mlp_mb']:.1f}MB, cold={mem['cold_store_mb']:.1f}MB, total={mem['total_mb']:.1f}MB")
    log(f"  AUC loss vs baseline: {auc_loss:.4f}pp")
    log(f"  Batch: mean={np.mean(blats) * 1000:.1f}ms, "
        f"p50={np.percentile(blats, 50) * 1000:.1f}ms, "
        f"p99={np.percentile(blats, 99) * 1000:.1f}ms")

    result = {
        'name': name, 'accuracy': acc, 'auc': auc,
        'auc_loss_pp': auc_loss,
        'inference_time': inf_time, 'setup_time': setup_time,
        'total_time': setup_time + inf_time,
        'memory_breakdown': mem, 'memory_mb': mem['total_mb'],
        'batch_latency': latency_stats(blats),
    }

    # Extra metrics for specific stores
    if hasattr(store, 'total_injections'):
        log(f"  Lazy: {store.total_injections:,} injections, {store.total_skipped:,} skipped")
        counts = store.injection_counts
        if len(counts) > 20:
            warmup = counts[:20]
            tail = counts[-20:]
            log(f"    Warmup (first 20): {[int(c) for c in warmup]}")
            log(f"    Tail   (last 20):  {[int(c) for c in tail]}")
        result['lazy_total_injections'] = store.total_injections
        result['lazy_total_skipped'] = store.total_skipped
        result['lazy_injection_counts_first50'] = counts[:50]
        result['lazy_injection_counts_last20'] = counts[-20:] if len(counts) > 20 else counts

    if hasattr(store, 'compression_ratio'):
        log(f"  Zlib: ratio={store.compression_ratio:.2f}x, "
            f"decompress_count={store.decompress_count}, "
            f"decompress_time={store.decompress_time * 1000:.1f}ms")
        result['zlib_compression_ratio'] = store.compression_ratio
        result['zlib_decompress_count'] = store.decompress_count
        result['zlib_decompress_time_ms'] = store.decompress_time * 1000

    return result


# ==============================================================
# MAIN
# ==============================================================

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    log("=" * 70)
    log("PREFETCH BENCHMARK V10: Entropy Coding Optimizations")
    log("=" * 70)

    log("Loading model and data...")
    dlrm, test_ld, train_ld, ln_emb = load_model_and_data()
    state_dict = torch.load(MODEL_PATH, map_location='cpu')['state_dict']
    emb_keys = sorted([k for k in state_dict if 'emb_l' in k and 'weight' in k],
                      key=lambda x: int(x.split('.')[1]))
    num_tables = len(emb_keys)
    large_tables = [t for t in range(num_tables) if ln_emb[t] >= LARGE_TABLE_THRESHOLD]
    log(f"Tables: {num_tables}, Large: {large_tables}")

    # Profile at both 80% and 95% thresholds
    log("Profiling access patterns (80% threshold)...")
    hot_80 = profile_access(train_ld, num_tables, ln_emb, HOT_THRESHOLD_80)
    log("Profiling access patterns (95% threshold)...")
    hot_95 = profile_access(train_ld, num_tables, ln_emb, HOT_THRESHOLD_95)

    for t in large_tables:
        log(f"  Table {t}: {ln_emb[t]:,} total, "
            f"{len(hot_80[t]):,} hot@80%, {len(hot_95[t]):,} hot@95%")

    results = {}

    # ================================================================
    # CONFIG 1: BASELINE
    # ================================================================
    log("\n" + "=" * 60)
    log("CONFIG 1: BASELINE — streaming, no compression")
    log("=" * 60)
    restore_weights(dlrm, state_dict, emb_keys)
    drop_caches(); time.sleep(1); gc.collect()

    b_acc, b_auc, b_time, b_blats, b_nb = run_baseline_inference(dlrm, test_ld)
    b_mem = sum(p.numel() * 4 for p in dlrm.parameters()) / 1024 / 1024
    log(f"  Acc={b_acc * 100:.4f}%, AUC={b_auc:.6f}")
    log(f"  Inference={b_time:.2f}s, Memory={b_mem:.1f}MB")
    log(f"  Batch: mean={np.mean(b_blats) * 1000:.1f}ms, "
        f"p50={np.percentile(b_blats, 50) * 1000:.1f}ms, "
        f"p99={np.percentile(b_blats, 99) * 1000:.1f}ms")

    baseline_auc = b_auc
    results['1_baseline'] = {
        'name': 'Baseline (streaming)', 'accuracy': b_acc, 'auc': b_auc,
        'auc_loss_pp': 0.0,
        'inference_time': b_time, 'setup_time': 0, 'total_time': b_time,
        'memory_mb': b_mem, 'batch_latency': latency_stats(b_blats),
    }

    # ================================================================
    # CONFIG 2: V9 reference (uint8, 80% hot)
    # ================================================================
    results['2_v9_reference'] = run_config(
        "V9 reference — uint8, 80% hot, prefetch",
        2, dlrm, test_ld, state_dict, emb_keys,
        ln_emb, num_tables, large_tables, hot_80, baseline_auc,
        EntropyColdStore)

    # ================================================================
    # CONFIG 3: Lazy materialization
    # ================================================================
    results['3_lazy'] = run_config(
        "Lazy materialization — skip re-injection",
        3, dlrm, test_ld, state_dict, emb_keys,
        ln_emb, num_tables, large_tables, hot_80, baseline_auc,
        LazyColdStore)

    # ================================================================
    # CONFIG 4: 4-bit quantization
    # ================================================================
    results['4_four_bit'] = run_config(
        "4-bit quantization — 2 vals/byte",
        4, dlrm, test_ld, state_dict, emb_keys,
        ln_emb, num_tables, large_tables, hot_80, baseline_auc,
        FourBitColdStore)

    # ================================================================
    # CONFIG 5: Block zlib compression
    # ================================================================
    results['5_block_zlib'] = run_config(
        "Block zlib — actual entropy coding",
        5, dlrm, test_ld, state_dict, emb_keys,
        ln_emb, num_tables, large_tables, hot_80, baseline_auc,
        BlockZlibColdStore, store_kwargs={'block_size': ZLIB_BLOCK_SIZE})

    # ================================================================
    # CONFIG 6: Hot threshold 95%
    # ================================================================
    results['6_hot_95'] = run_config(
        "Hot 95% — fewer cold injections",
        6, dlrm, test_ld, state_dict, emb_keys,
        ln_emb, num_tables, large_tables, hot_95, baseline_auc,
        EntropyColdStore)

    # ================================================================
    # CONFIG 7: Torch-native dequant
    # ================================================================
    results['7_torch_native'] = run_config(
        "Torch-native dequant — no numpy conversion",
        7, dlrm, test_ld, state_dict, emb_keys,
        ln_emb, num_tables, large_tables, hot_80, baseline_auc,
        TorchNativeColdStore)

    # ================================================================
    # CONFIG 8: Combined (lazy + torch-native + hot 95%)
    # ================================================================
    results['8_combined'] = run_config(
        "Combined — lazy + torch-native + hot 95%",
        8, dlrm, test_ld, state_dict, emb_keys,
        ln_emb, num_tables, large_tables, hot_95, baseline_auc,
        CombinedColdStore)

    # ================================================================
    # SUMMARY
    # ================================================================
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)

    fmt = "  %-50s  %8s  %10s  %10s  %10s  %10s  %10s"
    log(fmt % ("Config", "AUC", "AUC loss", "Infer(s)", "Setup(s)", "Total(s)", "Mem(MB)"))
    log("  " + "-" * 113)
    for key in sorted(results.keys()):
        r = results[key]
        al = r.get('auc_loss_pp', (baseline_auc - r['auc']) * 100)
        mem = r.get('memory_mb', 0)
        log(fmt % (r['name'], f"{r['auc']:.6f}", f"{al:.4f}pp",
                   f"{r['inference_time']:.2f}", f"{r.get('setup_time', 0):.2f}",
                   f"{r['total_time']:.2f}", f"{mem:.1f}"))

    # Save results
    log_path = os.path.join(RESULTS_DIR, 'prefetch_benchmark_v10.log')
    json_path = os.path.join(RESULTS_DIR, 'prefetch_v10_optimizations.json')

    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    log(f"\n  Saved JSON: {json_path}")

    # Also write a log summary
    summary_lines = []
    summary_lines.append("PREFETCH BENCHMARK V10: Entropy Coding Optimizations")
    summary_lines.append("=" * 70)
    for key in sorted(results.keys()):
        r = results[key]
        al = r.get('auc_loss_pp', 0)
        summary_lines.append(f"\n{r['name']}:")
        summary_lines.append(f"  AUC={r['auc']:.6f}, AUC loss={al:.4f}pp")
        summary_lines.append(f"  Inference={r['inference_time']:.2f}s, Setup={r.get('setup_time', 0):.2f}s, Total={r['total_time']:.2f}s")
        summary_lines.append(f"  Memory={r.get('memory_mb', 0):.1f}MB")
        bl = r.get('batch_latency', {})
        if bl:
            summary_lines.append(f"  Batch latency: mean={bl.get('mean_ms', 0):.1f}ms, "
                                 f"p50={bl.get('p50_ms', 0):.1f}ms, "
                                 f"p95={bl.get('p95_ms', 0):.1f}ms, "
                                 f"p99={bl.get('p99_ms', 0):.1f}ms")
        if 'lazy_total_injections' in r:
            summary_lines.append(f"  Lazy: {r['lazy_total_injections']:,} injections, {r['lazy_total_skipped']:,} skipped")
        if 'zlib_compression_ratio' in r:
            summary_lines.append(f"  Zlib: ratio={r['zlib_compression_ratio']:.2f}x, "
                                 f"decompress={r['zlib_decompress_count']}, "
                                 f"decompress_time={r['zlib_decompress_time_ms']:.1f}ms")

    with open(log_path, 'w') as f:
        f.write('\n'.join(summary_lines) + '\n')
    log(f"  Saved log: {log_path}")

    log("\n" + "=" * 70)
    log("DONE!")
    log("=" * 70)


if __name__ == '__main__':
    main()
