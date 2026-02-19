#!/usr/bin/env bash
set -euo pipefail
# DLRM inference + MBM bandwidth monitoring via pqos
# REDUCED configurations: No 8 threads, no 512 batch, no 16 dims, no 16 lookups

# -------------------------------
# User Configurations
# -------------------------------
DLRM_BIN="${DLRM_BIN:-dlrm_s_pytorch.py}"
RAW_FILE="${RAW_FILE:-./input/train.txt}"
PROC_FILE="${PROC_FILE:-./input/kaggleAdDisplayChallenge_processed.npz}"
MLP_BOT_PREFIX="${MLP_BOT_PREFIX:-13-512-256-64}"
MLP_TOP="${MLP_TOP:-512-256-1}"
PRINT_FREQ="${PRINT_FREQ:-8192}"
TEST_BS="${TEST_BS:-16384}"
TEST_WORKERS="${TEST_WORKERS:-16}"

# REDUCED Configurations: Largest values only
THREADS_LIST="${THREADS:-32 16}"
BATCH_SIZES_LIST="${BATCH_SIZES:-16384 2048 1024}"
SPARSE_DIMS_LIST="${SPARSE_DIMS:-64 32}"
LOOKUPS_LIST="${LOOKUPS:-64 32}"

# Use ALL 32 cores (but will limit threads via OMP_NUM_THREADS)
CPU_POOL="${CPU_POOL:-0-31}"
MEMNODE=0  # Single NUMA node
OUTDIR="${OUTDIR:-./inference_bw_mbm_logs}"
PYTHON_PATH="${PYTHON_PATH:-$(which python3)}"
mkdir -p "$OUTDIR"

# -------------------------------
# Cleanup Function
# -------------------------------
cleanup_processes() {
  echo ""
  echo "[CLEANUP] Stopping all monitoring and DLRM processes..."
  
  # Kill pqos
  sudo pkill -9 pqos 2>/dev/null || true
  
  # Kill perf
  sudo pkill -9 perf 2>/dev/null || true
  
  # Kill any DLRM processes
  pkill -9 -f "dlrm_s_pytorch.py" 2>/dev/null || true
  pkill -9 -f "python.*dlrm" 2>/dev/null || true
  
  # Remove lock files
  rm -f /run/lock/libpqos 2>/dev/null || true
  
  echo "[CLEANUP] Done"
}

# Set trap to cleanup on exit or interrupt
trap cleanup_processes EXIT INT TERM

# -------------------------------
# Helper Functions
# -------------------------------
expand_pool_to_array() {
  local pool="$1"; local -a out=()
  IFS=',' read -ra toks <<<"$pool"
  for t in "${toks[@]}"; do
    if [[ "$t" =~ ^[0-9]+-[0-9]+$ ]]; then
      IFS='-' read -r a b <<<"$t"
      for ((i=a;i<=b;i++)); do out+=("$i"); done
    else out+=("$i"); fi
  done
  echo "${out[@]}"
}
join_by_comma() { local IFS=,; echo "$*"; }

# -------------------------------
# MBM Monitor Function
# -------------------------------
run_mbm_monitor() {
  local outbase="$1"; shift
  [[ "${1:-}" == "--" ]] && shift
  
  local mbm_csv="${outbase}.mbm.csv"
  local cache_csv="${outbase}.cache.csv"
  local mbm_summary="${outbase}.mbm_summary.txt"
  local inference_log="${outbase}.inference.log"
  local pqos_pid_file="${outbase}.pqos.pid"
  local perf_pid_file="${outbase}.perf.pid"

  echo "[MBM] Starting monitoring..." | tee "$mbm_summary"
  echo "[MBM] Monitoring cores: $CPU_POOL" | tee -a "$mbm_summary"
  
  # Kill any existing DLRM processes first
  echo "[MBM] Cleaning up any existing DLRM processes..." | tee -a "$mbm_summary"
  pkill -9 -f "dlrm_s_pytorch.py" 2>/dev/null || true
  pkill -9 -f "python.*dlrm" 2>/dev/null || true
  sleep 1
  
  # Unmount resctrl if mounted (to avoid conflicts)
  if mount | grep -q resctrl; then
    echo "[MBM] Unmounting resctrl..." | tee -a "$mbm_summary"
    sudo umount /sys/fs/resctrl 2>/dev/null || true
  fi
  
  # Ensure lock file exists with proper permissions
  sudo touch /run/lock/libpqos
  sudo chmod 666 /run/lock/libpqos
  
  # Run the inference command in background
  echo "[MBM] Starting inference..." | tee -a "$mbm_summary"
  "$@" > "$inference_log" 2>&1 &
  local inference_pid=$!
  echo "[MBM] Inference started with PID: $inference_pid" | tee -a "$mbm_summary"
  
  # Wait a moment for inference to start
  sleep 3
  
  # Start pqos monitoring in background (MBM bandwidth)
  echo "[MBM] Starting pqos monitoring..." | tee -a "$mbm_summary"
  sudo pqos -I -m "mbt:[$CPU_POOL]" -u csv -o "$mbm_csv" &
  local pqos_pid=$!
  echo "$pqos_pid" > "$pqos_pid_file"
  echo "[MBM] pqos started with PID: $pqos_pid" | tee -a "$mbm_summary"
  
  # Start perf monitoring for cache stats (L1 and LLC)
  echo "[MBM] Starting perf cache monitoring..." | tee -a "$mbm_summary"
  
  # Run perf stat in interval mode for cache events
  sudo perf stat -C "$CPU_POOL" -I 1000 -x, \
    -e L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses \
    -o "$cache_csv" sleep 999999 &
  local perf_pid=$!
  echo "$perf_pid" > "$perf_pid_file"
  echo "[MBM] perf started with PID: $perf_pid" | tee -a "$mbm_summary"
  
  # Wait for inference to complete
  echo "[MBM] Waiting for inference to complete (PID: $inference_pid)..." | tee -a "$mbm_summary"
  
  # Check if process is still running periodically
  local wait_count=0
  while kill -0 "$inference_pid" 2>/dev/null; do
    sleep 5
    wait_count=$((wait_count + 1))
    if [ $((wait_count % 12)) -eq 0 ]; then
      echo "[MBM] Still running inference... (${wait_count}*5 seconds elapsed)" | tee -a "$mbm_summary"
    fi
  done
  
  # Get exit code
  wait "$inference_pid" 2>/dev/null
  local inference_exit=$?
  echo "[MBM] Inference completed with exit code: $inference_exit" | tee -a "$mbm_summary"
  
  # Double-check: kill any remaining DLRM processes
  pkill -9 -f "dlrm_s_pytorch.py" 2>/dev/null || true
  sleep 1
  
  # Stop monitoring tools
  echo "[MBM] Stopping monitoring tools..." | tee -a "$mbm_summary"
  
  # Stop pqos
  sudo kill "$pqos_pid" 2>/dev/null || true
  sleep 1
  sudo pkill -9 pqos 2>/dev/null || true
  rm -f "$pqos_pid_file"
  
  # Stop perf
  sudo kill "$perf_pid" 2>/dev/null || true
  sleep 1
  sudo pkill -9 perf 2>/dev/null || true
  rm -f "$perf_pid_file"
  
  # Analyze MBM results
  if [[ -f "$mbm_csv" ]]; then
    echo "" | tee -a "$mbm_summary"
    echo "=== Memory Bandwidth Analysis ===" | tee -a "$mbm_summary"
    
    awk -F',' '
      NR==1 { next }  # Skip header
      {
        time = $1
        cores = $2
        bw = $5  # MBT column
        
        if (bw > 0) {
          sum += bw
          count++
          if (bw > peak) {
            peak = bw
            peak_time = time
            peak_cores = cores
          }
        }
      }
      END {
        if (count > 0) {
          avg = sum / count
          printf "Peak Bandwidth: %.2f MB/s (%.3f GB/s) at %s\n", peak, peak/1024, peak_time
          printf "Average Bandwidth: %.2f MB/s (%.3f GB/s)\n", avg, avg/1024
          printf "Total Samples: %d\n", count
          printf "\nPEAK_MBps=%.2f\n", peak
          printf "PEAK_GBps=%.3f\n", peak/1024
          printf "AVG_MBps=%.2f\n", avg
          printf "AVG_GBps=%.3f\n", avg/1024
        } else {
          print "No bandwidth data collected"
        }
      }
    ' "$mbm_csv" | tee -a "$mbm_summary"
  fi
  
  # Analyze cache results
  if [[ -f "$cache_csv" ]]; then
    echo "" | tee -a "$mbm_summary"
    echo "=== Cache Miss Rate Analysis ===" | tee -a "$mbm_summary"
    
    # Parse perf output (format: timestamp,counter_value,unit,event_name,...)
    awk -F',' '
      /L1-dcache-loads/ && $2 != "<not counted>" && $2 != "" {
        l1_loads += $2
      }
      /L1-dcache-load-misses/ && $2 != "<not counted>" && $2 != "" {
        l1_misses += $2
      }
      /LLC-loads/ && $2 != "<not counted>" && $2 != "" {
        llc_loads += $2
      }
      /LLC-load-misses/ && $2 != "<not counted>" && $2 != "" {
        llc_misses += $2
      }
      END {
        if (l1_loads > 0) {
          l1_miss_rate = (l1_misses / l1_loads) * 100
          printf "L1 Cache:\n"
          printf "  Total Loads: %'\''d\n", l1_loads
          printf "  Total Misses: %'\''d\n", l1_misses
          printf "  Miss Rate: %.2f%%\n", l1_miss_rate
          printf "\n"
        }
        if (llc_loads > 0) {
          llc_miss_rate = (llc_misses / llc_loads) * 100
          printf "LLC (Last Level Cache):\n"
          printf "  Total Loads: %'\''d\n", llc_loads
          printf "  Total Misses: %'\''d\n", llc_misses
          printf "  Miss Rate: %.2f%%\n", llc_miss_rate
          printf "\n"
        }
        if (l1_loads > 0 || llc_loads > 0) {
          printf "L1_MISS_RATE=%.2f\n", l1_miss_rate
          printf "LLC_MISS_RATE=%.2f\n", llc_miss_rate
        } else {
          print "No cache data collected"
        }
      }
    ' "$cache_csv" | tee -a "$mbm_summary"
  fi
  
  echo "" | tee -a "$mbm_summary"
  echo "MBM_CSV=$mbm_csv" | tee -a "$mbm_summary"
  echo "CACHE_CSV=$cache_csv" | tee -a "$mbm_summary"
  echo "INFERENCE_LOG=$inference_log" | tee -a "$mbm_summary"
  
  return $inference_exit
}

# -------------------------------
# Main Experiment Loop
# -------------------------------
ALL_CORES=( $(expand_pool_to_array "$CPU_POOL") )
CPULIST_CSV="$(join_by_comma "${ALL_CORES[@]}")"

echo "========================================"
echo "DLRM Inference Bandwidth Profiling (REDUCED)"
echo "Available cores: $CPULIST_CSV"
echo "Thread counts: $THREADS_LIST"
echo "Batch sizes: $BATCH_SIZES_LIST"
echo "Sparse dims: $SPARSE_DIMS_LIST"
echo "Lookups: $LOOKUPS_LIST"
echo "Output directory: $OUTDIR"
echo "Total experiments: 24 (2T × 2D × 3B × 2L)"
echo "========================================"
echo ""

# Count total and skipped experiments
total_experiments=0
skipped_experiments=0
completed_experiments=0

# Run experiments one at a time (largest configs first)
for T in $THREADS_LIST; do
  for D in $SPARSE_DIMS_LIST; do
    MLP_BOT="${MLP_BOT_PREFIX}-${D}"
    for B in $BATCH_SIZES_LIST; do
      for L in $LOOKUPS_LIST; do
        total_experiments=$((total_experiments + 1))
        
        # Check if this experiment has already been run
        existing_logs=("$OUTDIR"/inference_T${T}_D${D}_B${B}_L${L}_*.mbm_summary.txt)
        if [[ -f "${existing_logs[0]}" ]]; then
          skipped_experiments=$((skipped_experiments + 1))
          echo "========================================="
          echo ">>> SKIP ($skipped_experiments skipped): T=$T D=$D B=$B L=$L"
          echo ">>> Existing: $(basename "${existing_logs[0]}")"
          echo "========================================="
          echo ""
          continue
        fi
        
        # Clear caches before each run
        sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
        
        stamp=$(date +%Y%m%d_%H%M%S)
        logfile="$OUTDIR/inference_T${T}_D${D}_B${B}_L${L}_${stamp}.log"

        echo "========================================="
        echo ">>> RUN ($((completed_experiments + 1))/$((24 - skipped_experiments))): T=$T D=$D B=$B L=$L"
        echo ">>> CPUs: $CPULIST_CSV, Threads: $T"
        echo ">>> Base log: $logfile"
        echo "========================================="

        # Set thread count via environment variable
        export OMP_NUM_THREADS=$T

        cmd=( numactl --physcpubind="$CPULIST_CSV" --membind="$MEMNODE"
              "$PYTHON_PATH" "$DLRM_BIN"
              --arch-sparse-feature-size="$D"
              --arch-mlp-bot="$MLP_BOT"
              --arch-mlp-top="$MLP_TOP"
              --data-generation=dataset
              --data-set=kaggle
              --raw-data-file="$RAW_FILE"
              --processed-data-file="$PROC_FILE"
              --dataset-multiprocessing
              --loss-function=bce
              --round-targets=True
              --mini-batch-size="$B"
              --print-freq="$PRINT_FREQ"
              --print-time
              --test-mini-batch-size="$B"
              --test-num-workers="$TEST_WORKERS"
              --num-indices-per-lookup="$L"
              --inference-only
              --nepochs=0 )

        MBM_BASE="${logfile%.log}"
        
        # Run one experiment at a time and wait for completion
        if run_mbm_monitor "$MBM_BASE" "${cmd[@]}"; then
          completed_experiments=$((completed_experiments + 1))
          echo "[SUCCESS] Completed T=$T D=$D B=$B L=$L" >&2
        else
          echo "[WARN] Failed T=$T D=$D B=$B L=$L — see ${MBM_BASE}.inference.log" >&2
        fi

        echo "[SAVED] Logs at: ${MBM_BASE}.*"
        echo ""
        
        # Small delay between experiments
        sleep 2
      done
    done
  done
done

echo "========================================"
echo "All experiments completed!"
echo "Total experiments: $total_experiments"
echo "Skipped (already done): $skipped_experiments"
echo "Newly completed: $completed_experiments"
echo "Results in: $OUTDIR"
echo "========================================"
