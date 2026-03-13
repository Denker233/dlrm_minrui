#!/usr/bin/env bash
set -euo pipefail
# DLRM Merge Table Performance Testing - TRAINING VERSION
# Optimized for dual-NUMA machine with 80 physical cores per node

# -------------------------------
# User Configurations
# -------------------------------
DLRM_BIN="${DLRM_BIN:-dlrm_merge_pytorch.py}"
RAW_FILE="${RAW_FILE:-./input/train.txt}"
PROC_FILE="${PROC_FILE:-./input/kaggleAdDisplayChallenge_processed.npz}"
MLP_BOT_PREFIX="${MLP_BOT_PREFIX:-13-512-256-64}"
MLP_TOP="${MLP_TOP:-512-256-1}"
PRINT_FREQ="${PRINT_FREQ:-100}"
NEPOCHS="${NEPOCHS:-1}"  # Number of training epochs
LEARNING_RATE="${LEARNING_RATE:-0.01}"
NUM_BATCHES="${NUM_BATCHES:-0}"  # Limit number of batches per epoch
TEST_FREQ="${TEST_FREQ:-0}"  # How often to test (0 = no testing during training)

# Fixed configuration for merge testing
BATCH_SIZE="${BATCH_SIZE:-10240}"
SPARSE_DIM="${SPARSE_DIM:-64}"
LOOKUPS="${LOOKUPS:-64}"

# Test different merge thresholds
MERGE_THRESHOLDS="${MERGE_THRESHOLDS:-0 50 100 500 1000 10000}"
# 0 = no merging (baseline)
# 1000 = merge tiny tables only (C6,C9,C14,C17,C20,C22,C23,C25)
# 10000 = merge small tables (adds C2,C5,C8,C11,C13,C18,C19)
# 100000 = merge medium tables (adds C1,C7,C10,C15)

# Thread configurations to test
THREADS_LIST="${THREADS:-20}"

# CPU Pool Configuration
# Physical cores: 0-79 (40 per NUMA node)
#   NUMA node0: 0,2,4,6,...,78 (even numbers, 40 physical cores)
#   NUMA node1: 1,3,5,7,...,79 (odd numbers, 40 physical cores)
# Hyperthreads: 80-159 (siblings of 0-79)
#   NUMA node0 HT: 80,82,84,...,158 (even numbers, 40 hyperthreads)
#   NUMA node1 HT: 81,83,85,...,159 (odd numbers, 40 hyperthreads)

CPU_POOL=""
MEMNODE=0  # Always use NUMA node 0

OUTDIR="${OUTDIR:-./9_merge_training_logs}"
PYTHON_PATH="${PYTHON_PATH:-$(which python3)}"
mkdir -p "$OUTDIR"

# -------------------------------
# CPU Pool Selection Function
# -------------------------------
select_cpu_pool() {
    local threads=$1
    local pool=""
    
    if [ "$threads" -le 20 ]; then
        # Use first 20 physical cores from node0: 0,2,4,...,38
        local cores=""
        for i in $(seq 0 2 38); do
            cores="${cores}${cores:+,}$i"
        done
        pool="$cores"
    elif [ "$threads" -le 40 ]; then
        # Use all 40 physical cores from node0: 0,2,4,...,78
        local cores=""
        for i in $(seq 0 2 78); do
            cores="${cores}${cores:+,}$i"
        done
        pool="$cores"
    elif [ "$threads" -le 80 ]; then
        # Use 40 physical + 40 hyperthreads from node0
        # Physical: 0,2,4,...,78 (40 cores)
        # Hyperthreads: 80,82,84,...,158 (40 hyperthreads)
        local cores=""
        # All 40 physical cores from node0
        for i in $(seq 0 2 78); do
            cores="${cores}${cores:+,}$i"
        done
        # All 40 hyperthreads from node0
        for i in $(seq 80 2 158); do
            cores="${cores}${cores:+,}$i"
        done
        pool="$cores"
    else
        # Default to all node0 resources
        local cores=""
        for i in $(seq 0 2 78); do
            cores="${cores}${cores:+,}$i"
        done
        for i in $(seq 80 2 158); do
            cores="${cores}${cores:+,}$i"
        done
        pool="$cores"
    fi
    
    echo "$pool"
}

# -------------------------------
# Cleanup Function
# -------------------------------
cleanup_processes() {
  echo ""
  echo "[CLEANUP] Stopping all monitoring and DLRM processes..."
  
  sudo pkill -9 pqos 2>/dev/null || true
  sudo pkill -9 perf 2>/dev/null || true
  pkill -9 -f "dlrm_merge_pytorch.py" 2>/dev/null || true
  pkill -9 -f "python.*dlrm" 2>/dev/null || true
  rm -f /run/lock/libpqos 2>/dev/null || true
  
  echo "[CLEANUP] Done"
}

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
# MBM Monitor Function for Training
# -------------------------------
run_mbm_monitor() {
  local outbase="$1"; shift
  [[ "${1:-}" == "--" ]] && shift
  
  local mbm_csv="${outbase}.mbm.csv"
  local cache_csv="${outbase}.cache.csv"
  local mbm_summary="${outbase}.mbm_summary.txt"
  local training_log="${outbase}.training.log"
  local pqos_pid_file="${outbase}.pqos.pid"
  local perf_pid_file="${outbase}.perf.pid"

  echo "[MBM] Starting monitoring..." | tee "$mbm_summary"
  echo "[MBM] Monitoring cores: $CPU_POOL" | tee -a "$mbm_summary"
  
  # Kill any existing DLRM processes first
  echo "[MBM] Cleaning up any existing DLRM processes..." | tee -a "$mbm_summary"
  pkill -9 -f "dlrm_merge_pytorch.py" 2>/dev/null || true
  pkill -9 -f "python.*dlrm" 2>/dev/null || true
  sleep 1
  
  # Unmount resctrl if mounted
  if mount | grep -q resctrl; then
    echo "[MBM] Unmounting resctrl..." | tee -a "$mbm_summary"
    sudo umount /sys/fs/resctrl 2>/dev/null || true
  fi
  
  # Ensure lock file exists with proper permissions
  sudo touch /run/lock/libpqos
  sudo chmod 666 /run/lock/libpqos
  
  # Run the training command in background
  echo "[MBM] Starting training..." | tee -a "$mbm_summary"
  "$@" > "$training_log" 2>&1 &
  local training_pid=$!
  echo "[MBM] Training started with PID: $training_pid" | tee -a "$mbm_summary"
  
  sleep 3
  
  # Start pqos monitoring in background
  echo "[MBM] Starting pqos monitoring..." | tee -a "$mbm_summary"
  sudo pqos -I -m "mbt:[$CPU_POOL]" -u csv -o "$mbm_csv" &
  local pqos_pid=$!
  echo "$pqos_pid" > "$pqos_pid_file"
  echo "[MBM] pqos started with PID: $pqos_pid" | tee -a "$mbm_summary"
  
  # Start perf monitoring for cache stats
  echo "[MBM] Starting perf cache monitoring..." | tee -a "$mbm_summary"
  sudo perf stat -C "$CPU_POOL" -I 1000 -x, \
    -e L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses \
    -o "$cache_csv" sleep 999999 &
  local perf_pid=$!
  echo "$perf_pid" > "$perf_pid_file"
  echo "[MBM] perf started with PID: $perf_pid" | tee -a "$mbm_summary"
  
  # Wait for training to complete
  echo "[MBM] Waiting for training to complete (PID: $training_pid)..." | tee -a "$mbm_summary"
  
  local wait_count=0
  while kill -0 "$training_pid" 2>/dev/null; do
    sleep 5
    wait_count=$((wait_count + 1))
    if [ $((wait_count % 12)) -eq 0 ]; then
      echo "[MBM] Still running training... (${wait_count}*5 seconds elapsed)" | tee -a "$mbm_summary"
    fi
  done
  
  wait "$training_pid" 2>/dev/null
  local training_exit=$?
  echo "[MBM] Training completed with exit code: $training_exit" | tee -a "$mbm_summary"
  
  # Double-check: kill any remaining processes
  pkill -9 -f "dlrm_merge_pytorch.py" 2>/dev/null || true
  sleep 1
  
  # Stop monitoring tools
  echo "[MBM] Stopping monitoring tools..." | tee -a "$mbm_summary"
  
  sudo kill "$pqos_pid" 2>/dev/null || true
  sleep 1
  sudo pkill -9 pqos 2>/dev/null || true
  rm -f "$pqos_pid_file"
  
  sudo kill "$perf_pid" 2>/dev/null || true
  sleep 1
  sudo pkill -9 perf 2>/dev/null || true
  rm -f "$perf_pid_file"
  
  # Analyze MBM results
  if [[ -f "$mbm_csv" ]]; then
    echo "" | tee -a "$mbm_summary"
    echo "=== Memory Bandwidth Analysis ===" | tee -a "$mbm_summary"
    
    awk -F',' '
      NR==1 { next }
      {
        time = $1
        cores = $2
        bw = $5
        
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
  
  # Extract training performance metrics from log
  if [[ -f "$training_log" ]]; then
    echo "" | tee -a "$mbm_summary"
    echo "=== Training Performance ===" | tee -a "$mbm_summary"
    
    # Extract throughput (it/s or samples/sec)
    echo "Throughput:" | tee -a "$mbm_summary"
    grep -i "it/s\|samples/sec\|throughput" "$training_log" | tail -10 | tee -a "$mbm_summary" || echo "  No throughput data" | tee -a "$mbm_summary"
    
    # Extract final training loss
    echo "" | tee -a "$mbm_summary"
    echo "Training Loss:" | tee -a "$mbm_summary"
    grep -i "loss" "$training_log" | tail -5 | tee -a "$mbm_summary" || echo "  No loss data" | tee -a "$mbm_summary"
    
    # Extract timing information
    echo "" | tee -a "$mbm_summary"
    echo "Timing:" | tee -a "$mbm_summary"
    grep -i "time taken\|elapsed\|total.*time\|epoch.*time" "$training_log" | tail -5 | tee -a "$mbm_summary" || echo "  No timing data" | tee -a "$mbm_summary"
    
    # Extract embedding lookup timing if available
    echo "" | tee -a "$mbm_summary"
    echo "Embedding Lookup Time:" | tee -a "$mbm_summary"
    grep -i "look up emb\|embedding.*time" "$training_log" | tail -5 | tee -a "$mbm_summary" || echo "  No embedding timing data" | tee -a "$mbm_summary"
  fi
  
  echo "" | tee -a "$mbm_summary"
  echo "MBM_CSV=$mbm_csv" | tee -a "$mbm_summary"
  echo "CACHE_CSV=$cache_csv" | tee -a "$mbm_summary"
  echo "TRAINING_LOG=$training_log" | tee -a "$mbm_summary"
  
  return $training_exit
}

# -------------------------------
# Main Experiment Loop
# -------------------------------
MLP_BOT="${MLP_BOT_PREFIX}-${SPARSE_DIM}"

echo "========================================"
echo "DLRM Merge Table Performance Testing - TRAINING"
echo "NUMA Configuration: 2 nodes"
echo "Physical cores: 0-79 (40 per node)"
echo "  Node0: 0,2,4,...,78 (even, 40 physical)"
echo "  Node1: 1,3,5,...,79 (odd, 40 physical)"
echo "Hyperthreads: 80-159 (siblings of 0-79)"
echo ""
echo "Script: $DLRM_BIN"
echo "Thread counts: $THREADS_LIST"
echo "Batch size: $BATCH_SIZE"
echo "Sparse dim: $SPARSE_DIM"
echo "Lookups: $LOOKUPS"
echo "Merge thresholds: $MERGE_THRESHOLDS"
echo "Training epochs: $NEPOCHS"
echo "Batches per epoch: $NUM_BATCHES"
echo "Learning rate: $LEARNING_RATE"
echo "Output directory: $OUTDIR"
echo ""
echo "Thread allocation strategy:"
echo "  20: First 20 physical cores from node0"
echo "  40: All 40 physical cores from node0"
echo "  80: 40 physical + 40 HT from node0"
echo ""
total_experiments=$(($(echo $MERGE_THRESHOLDS | wc -w) * $(echo $THREADS_LIST | wc -w)))
echo "Total experiments: $total_experiments"
echo "========================================"
echo ""

experiment_count=0
skipped_experiments=0
completed_experiments=0

# Run experiments for each thread count and merge threshold
for T in $THREADS_LIST; do
  # Select CPU pool based on thread count
  CPU_POOL=$(select_cpu_pool $T)
  CPULIST_CSV="$CPU_POOL"
  
  # Count actual cores in pool
  num_cores=$(echo "$CPU_POOL" | tr ',' '\n' | wc -l)
  
  echo "========================================"
  echo "Thread Configuration: $T threads"
  echo "CPU Pool: Using $num_cores CPUs"
  echo "NUMA Node: $MEMNODE"
  echo "========================================"
  echo ""
  
  for THRESHOLD in $MERGE_THRESHOLDS; do
    experiment_count=$((experiment_count + 1))
    
    # Check if already run
    existing_logs=("$OUTDIR"/training_T${T}_B${BATCH_SIZE}_D${SPARSE_DIM}_L${LOOKUPS}_THRESH${THRESHOLD}_*.mbm_summary.txt)
    if [[ -f "${existing_logs[0]}" ]]; then
      skipped_experiments=$((skipped_experiments + 1))
      echo "========================================="
      echo ">>> SKIP ($skipped_experiments skipped): T=$T THRESH=$THRESHOLD"
      echo ">>> Existing: $(basename "${existing_logs[0]}")"
      echo "========================================="
      echo ""
      continue
    fi
    
    # Clear caches before each run
    sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
    
    stamp=$(date +%Y%m%d_%H%M%S)
    logfile="$OUTDIR/training_T${T}_B${BATCH_SIZE}_D${SPARSE_DIM}_L${LOOKUPS}_THRESH${THRESHOLD}_${stamp}.log"

    echo "========================================="
    echo ">>> RUN ($((completed_experiments + 1))/$((total_experiments - skipped_experiments))): T=$T Threshold=$THRESHOLD"
    if [ "$THRESHOLD" -eq 0 ]; then
      echo ">>> Mode: BASELINE (No merging)"
    elif [ "$THRESHOLD" -eq 100 ]; then
      echo ">>> Mode: Merge tiny tables (<100 rows)"
      echo ">>>       Merges: C6,C9,C14,C17,C20,C22,C23,C25"
    elif [ "$THRESHOLD" -eq 1000 ]; then
      echo ">>> Mode: Merge tiny tables (<1000 rows)"
      echo ">>>       Merges: C6,C9,C14,C17,C20,C22,C23,C25"
    elif [ "$THRESHOLD" -eq 10000 ]; then
      echo ">>> Mode: Merge small tables (<10000 rows)"
      echo ">>>       Adds: C2,C5,C8,C11,C13,C18,C19"
    elif [ "$THRESHOLD" -eq 100000 ]; then
      echo ">>> Mode: Merge medium tables (<100000 rows)"
      echo ">>>       Adds: C1,C7,C10,C15"
    fi
    echo ">>> CPUs: $num_cores cores, OMP Threads: $T"
    echo ">>> Training: $NEPOCHS epochs, $NUM_BATCHES batches/epoch, LR=$LEARNING_RATE"
    echo ">>> Base log: $logfile"
    echo "========================================="

    # Set thread count
    export OMP_NUM_THREADS=$T

    cmd=( numactl --physcpubind="$CPULIST_CSV" --membind="$MEMNODE"
          "$PYTHON_PATH" "$DLRM_BIN"
          --arch-sparse-feature-size="$SPARSE_DIM"
          --arch-mlp-bot="$MLP_BOT"
          --arch-mlp-top="$MLP_TOP"
          --data-generation=dataset
          --data-set=kaggle
          --raw-data-file="$RAW_FILE"
          --processed-data-file="$PROC_FILE"
          --dataset-multiprocessing
          --loss-function=bce
          --round-targets=True
          --learning-rate="$LEARNING_RATE"
          --mini-batch-size="$BATCH_SIZE"
          --print-freq="$PRINT_FREQ"
          --print-time
          --num-batches="$NUM_BATCHES"
          --num-indices-per-lookup="$LOOKUPS"
          --merge-emb-threshold="$THRESHOLD"
          --test-freq="$TEST_FREQ" )

    MBM_BASE="${logfile%.log}"
    
    # Run experiment
    if run_mbm_monitor "$MBM_BASE" "${cmd[@]}"; then
      completed_experiments=$((completed_experiments + 1))
      echo "[SUCCESS] Completed T=$T THRESH=$THRESHOLD" >&2
    else
      echo "[WARN] Failed T=$T THRESH=$THRESHOLD — see ${MBM_BASE}.training.log" >&2
    fi

    echo "[SAVED] Logs at: ${MBM_BASE}.*"
    echo ""
    
    sleep 2
  done
done

# Generate summary comparison
echo "========================================"
echo "Generating performance comparison..."
echo "========================================"

SUMMARY_FILE="$OUTDIR/merge_training_summary.txt"
echo "DLRM Merged Embedding Table Training Performance Comparison" > "$SUMMARY_FILE"
echo "Generated: $(date)" >> "$SUMMARY_FILE"
echo "Machine: 80-core dual-NUMA" >> "$SUMMARY_FILE"
echo "Training: $NEPOCHS epochs, $NUM_BATCHES batches/epoch" >> "$SUMMARY_FILE"
echo "========================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

for T in $THREADS_LIST; do
  echo "======== THREADS: $T ========" >> "$SUMMARY_FILE"
  for THRESHOLD in $MERGE_THRESHOLDS; do
    echo "" >> "$SUMMARY_FILE"
    echo "Merge Threshold: $THRESHOLD" >> "$SUMMARY_FILE"
    echo "----------------------------------------" >> "$SUMMARY_FILE"
    
    summary_files=("$OUTDIR"/training_T${T}_B${BATCH_SIZE}_D${SPARSE_DIM}_L${LOOKUPS}_THRESH${THRESHOLD}_*.mbm_summary.txt)
    if [[ -f "${summary_files[0]}" ]]; then
      grep "PEAK_GBps\|AVG_GBps\|L1_MISS_RATE\|LLC_MISS_RATE\|it/s\|loss" "${summary_files[0]}" >> "$SUMMARY_FILE" 2>/dev/null || echo "No metrics found" >> "$SUMMARY_FILE"
    else
      echo "No results found" >> "$SUMMARY_FILE"
    fi
  done
  echo "" >> "$SUMMARY_FILE"
  echo "========================================" >> "$SUMMARY_FILE"
  echo "" >> "$SUMMARY_FILE"
done

echo "========================================"
echo "All experiments completed!"
echo "Total experiments: $total_experiments"
echo "Skipped (already done): $skipped_experiments"
echo "Newly completed: $completed_experiments"
echo "Results in: $OUTDIR"
echo "Summary: $SUMMARY_FILE"
echo "========================================"

cat "$SUMMARY_FILE"
