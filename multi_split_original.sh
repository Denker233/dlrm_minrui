#!/usr/bin/env bash
set -euo pipefail
# DLRM Merge/Split Table Performance Testing - Multi-Batch Multi-Run
# Runs experiments across different batch sizes, multiple times

# -------------------------------
# User Configurations
# -------------------------------
DLRM_BIN="${DLRM_BIN:-dlrm_split_merge_pytorch.py}"
RAW_FILE="${RAW_FILE:-./input/train.txt}"
PROC_FILE="${PROC_FILE:-./input/kaggleAdDisplayChallenge_processed.npz}"
MLP_BOT_PREFIX="${MLP_BOT_PREFIX:-13-512-256-64}"
MLP_TOP="${MLP_TOP:-512-256-1}"
PRINT_FREQ="${PRINT_FREQ:-8192}"
TEST_WORKERS="${TEST_WORKERS:-16}"

# Experiment configurations
SPARSE_DIM="${SPARSE_DIM:-64}"
LOOKUPS="${LOOKUPS:-64}"
MERGE_THRESHOLDS="${MERGE_THRESHOLDS:-0}"
SPLIT_THRESHOLDS="${SPLIT_THRESHOLDS:-0 100000 500000 1000000 5000000 10000000}"
NUM_SPLITS_LIST="${NUM_SPLITS_LIST:-4}"  # ADD THIS LINE
MIN_TABLE_SIZE="${MIN_TABLE_SIZE:-50}"  # ADD THIS LINE
THREADS_LIST="${THREADS:-40 20}"

# Multi-batch multi-run configuration
BATCH_SIZES="${BATCH_SIZES:-16380 8190 4096 1024 512}"
NUM_RUNS="${NUM_RUNS:-3}"

# CPU and NUMA configuration
MEMNODE=0
BASE_OUTDIR="${BASE_OUTDIR:-./brandnew_full_data_prime_merge_split_multibatch_runs}"
PYTHON_PATH="${PYTHON_PATH:-$(which python3)}"

mkdir -p "$BASE_OUTDIR"

# -------------------------------
# CPU Pool Selection Function
# -------------------------------
select_cpu_pool() {
    local threads=$1
    local pool=""
    
    if [ "$threads" -le 20 ]; then
        local cores=""
        for i in $(seq 0 2 38); do
            cores="${cores}${cores:+,}$i"
        done
        pool="$cores"
    elif [ "$threads" -le 40 ]; then
        local cores=""
        for i in $(seq 0 2 78); do
            cores="${cores}${cores:+,}$i"
        done
        pool="$cores"
    elif [ "$threads" -le 60 ]; then
        local cores=""
        for i in $(seq 0 2 78); do
            cores="${cores}${cores:+,}$i"
        done
        for i in $(seq 80 2 98); do
            cores="${cores}${cores:+,}$i"
        done
        pool="$cores"
    elif [ "$threads" -le 80 ]; then
        local cores=""
        for i in $(seq 0 2 78); do
            cores="${cores}${cores:+,}$i"
        done
        for i in $(seq 80 2 158); do
            cores="${cores}${cores:+,}$i"
        done
        pool="$cores"
    else
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
  pkill -9 -f "dlrm_split_merge_pytorch.py" 2>/dev/null || true
  pkill -9 -f "python.*dlrm" 2>/dev/null || true
  rm -f /run/lock/libpqos 2>/dev/null || true
  
  echo "[CLEANUP] Done"
}

trap cleanup_processes EXIT INT TERM

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
  
  pkill -9 -f "dlrm_split_merge_pytorch.py" 2>/dev/null || true
  pkill -9 -f "python.*dlrm" 2>/dev/null || true
  sleep 1
  
  if mount | grep -q resctrl; then
    echo "[MBM] Unmounting resctrl..." | tee -a "$mbm_summary"
    sudo umount /sys/fs/resctrl 2>/dev/null || true
  fi
  
  sudo touch /run/lock/libpqos
  sudo chmod 666 /run/lock/libpqos
  
  echo "[MBM] Starting inference..." | tee -a "$mbm_summary"
  "$@" > "$inference_log" 2>&1 &
  local inference_pid=$!
  echo "[MBM] Inference started with PID: $inference_pid" | tee -a "$mbm_summary"
  
  sleep 3
  
  echo "[MBM] Starting pqos monitoring..." | tee -a "$mbm_summary"
  sudo pqos -I -m "mbt:[$CPU_POOL]" -u csv -o "$mbm_csv" &
  local pqos_pid=$!
  echo "$pqos_pid" > "$pqos_pid_file"
  echo "[MBM] pqos started with PID: $pqos_pid" | tee -a "$mbm_summary"
  
  echo "[MBM] Starting perf cache monitoring..." | tee -a "$mbm_summary"
  sudo perf stat -C "$CPU_POOL" -I 1000 -x, \
    -e L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses \
    -o "$cache_csv" sleep 999999 &
  local perf_pid=$!
  echo "$perf_pid" > "$perf_pid_file"
  echo "[MBM] perf started with PID: $perf_pid" | tee -a "$mbm_summary"
  
  echo "[MBM] Waiting for inference to complete (PID: $inference_pid)..." | tee -a "$mbm_summary"
  
  local wait_count=0
  while kill -0 "$inference_pid" 2>/dev/null; do
    sleep 5
    wait_count=$((wait_count + 1))
    if [ $((wait_count % 12)) -eq 0 ]; then
      echo "[MBM] Still running inference... (${wait_count}*5 seconds elapsed)" | tee -a "$mbm_summary"
    fi
  done
  
  wait "$inference_pid" 2>/dev/null
  local inference_exit=$?
  echo "[MBM] Inference completed with exit code: $inference_exit" | tee -a "$mbm_summary"
  
  pkill -9 -f "dlrm_split_merge_pytorch.py" 2>/dev/null || true
  sleep 1
  
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
  
  # Extract performance metrics from inference log
  if [[ -f "$inference_log" ]]; then
    echo "" | tee -a "$mbm_summary"
    echo "=== Inference Performance ===" | tee -a "$mbm_summary"
    
    grep -E "The MLP time|The embedding time|The interaction time|The total time" "$inference_log" | tee -a "$mbm_summary" || true
  fi
  
  echo "" | tee -a "$mbm_summary"
  echo "MBM_CSV=$mbm_csv" | tee -a "$mbm_summary"
  echo "CACHE_CSV=$cache_csv" | tee -a "$mbm_summary"
  echo "INFERENCE_LOG=$inference_log" | tee -a "$mbm_summary"
  
  return $inference_exit
}

# -------------------------------
# Main Multi-Run Loop
# -------------------------------
echo "========================================"
echo "DLRM Multi-Batch Multi-Run Experiments"
echo "========================================"
echo "Batch sizes: $BATCH_SIZES"
echo "Number of runs: $NUM_RUNS"
echo "Threads: $THREADS_LIST"
echo "Merge thresholds: $MERGE_THRESHOLDS"
echo "Split thresholds: $SPLIT_THRESHOLDS"
echo "Num splits: $NUM_SPLITS_LIST"  # ADD THIS LINE
echo "Min table size: $MIN_TABLE_SIZE"  # ADD THIS LINE
echo "Base output directory: $BASE_OUTDIR"
echo ""

# UPDATED calculation to include num_splits dimension
experiments_per_batch=$(($(echo $MERGE_THRESHOLDS | wc -w) * $(echo $SPLIT_THRESHOLDS | wc -w) * $(echo $NUM_SPLITS_LIST | wc -w) * $(echo $THREADS_LIST | wc -w)))
total_experiments=$((experiments_per_batch * $(echo $BATCH_SIZES | wc -w) * NUM_RUNS))

echo "Experiments per batch size: $experiments_per_batch"
echo "Total experiments across all runs: $total_experiments"
echo "========================================"
echo ""

overall_start_time=$(date +%s)

# Loop over runs
for RUN_NUM in $(seq 1 $NUM_RUNS); do
  echo ""
  echo "###############################################"
  echo "###  STARTING RUN $RUN_NUM of $NUM_RUNS"
  echo "###############################################"
  echo ""
  
  # Loop over batch sizes
  for BATCH_SIZE in $BATCH_SIZES; do
    echo ""
    echo "================================================"
    echo "RUN $RUN_NUM - BATCH SIZE: $BATCH_SIZE"
    echo "================================================"
    
    # Create output directory for this run and batch size
    OUTDIR="$BASE_OUTDIR/run${RUN_NUM}_batch${BATCH_SIZE}"
    mkdir -p "$OUTDIR"
    
    MLP_BOT="${MLP_BOT_PREFIX}-${SPARSE_DIM}"
    
    experiment_count=0
    skipped_experiments=0
    completed_experiments=0
    
    # Run experiments for each configuration
    for T in $THREADS_LIST; do
      CPU_POOL=$(select_cpu_pool $T)
      CPULIST_CSV="$CPU_POOL"
      num_cores=$(echo "$CPU_POOL" | tr ',' '\n' | wc -l)
      
      echo ""
      echo "---------- Thread Config: $T (Run $RUN_NUM, Batch $BATCH_SIZE) ----------"
      
      for MERGE_THRESH in $MERGE_THRESHOLDS; do
        for SPLIT_THRESH in $SPLIT_THRESHOLDS; do
          for NUM_SPLITS in $NUM_SPLITS_LIST; do  # ADD THIS LOOP
            experiment_count=$((experiment_count + 1))
            
            # UPDATED: Check if already run (now includes NUM_SPLITS in filename)
            existing_logs=("$OUTDIR"/reorg_T${T}_B${BATCH_SIZE}_D${SPARSE_DIM}_L${LOOKUPS}_M${MERGE_THRESH}_S${SPLIT_THRESH}_N${NUM_SPLITS}_*.mbm_summary.txt)
            if [[ -f "${existing_logs[0]}" ]]; then
              skipped_experiments=$((skipped_experiments + 1))
              echo ">>> SKIP: T=$T M=$MERGE_THRESH S=$SPLIT_THRESH N=$NUM_SPLITS (already exists)"
              continue
            fi
            
            # Clear caches
            sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
            
            stamp=$(date +%Y%m%d_%H%M%S)
            # UPDATED: logfile now includes NUM_SPLITS
            logfile="$OUTDIR/reorg_T${T}_B${BATCH_SIZE}_D${SPARSE_DIM}_L${LOOKUPS}_M${MERGE_THRESH}_S${SPLIT_THRESH}_N${NUM_SPLITS}_${stamp}.log"

            echo ">>> RUN $RUN_NUM ($((completed_experiments + 1))/$experiments_per_batch): T=$T M=$MERGE_THRESH S=$SPLIT_THRESH N=$NUM_SPLITS B=$BATCH_SIZE"
            
            # UPDATED: Describe mode (now includes num_splits info)
            if [ "$MERGE_THRESH" -eq 0 ] && [ "$SPLIT_THRESH" -eq 0 ]; then
              mode="BASELINE"
            elif [ "$MERGE_THRESH" -gt 0 ] && [ "$SPLIT_THRESH" -eq 0 ]; then
              mode="MERGE($MERGE_THRESH)"
            elif [ "$MERGE_THRESH" -eq 0 ] && [ "$SPLIT_THRESH" -gt 0 ]; then
              mode="SPLIT($SPLIT_THRESH,n=$NUM_SPLITS)"
            else
              mode="MERGE+SPLIT($MERGE_THRESH,$SPLIT_THRESH,n=$NUM_SPLITS)"
            fi
            echo ">>> Mode: $mode | Cores: $num_cores"

            export OMP_NUM_THREADS=$T

            # UPDATED: Command now includes --num-splits and --min-table-size
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
                  --mini-batch-size="$BATCH_SIZE"
                  --print-freq="$PRINT_FREQ"
                  --print-time
                  --test-mini-batch-size="$BATCH_SIZE"
                  --test-num-workers="$TEST_WORKERS"
                  --num-indices-per-lookup="$LOOKUPS"
                  --merge-emb-threshold="$MERGE_THRESH"
                  --split-emb-threshold="$SPLIT_THRESH"
                  --num-splits="$NUM_SPLITS"
                  --min-table-size="$MIN_TABLE_SIZE"
                  --inference-only
                  --nepochs=0 )

            MBM_BASE="${logfile%.log}"
            
            if run_mbm_monitor "$MBM_BASE" "${cmd[@]}"; then
              completed_experiments=$((completed_experiments + 1))
              echo "[SUCCESS] Run $RUN_NUM - Batch $BATCH_SIZE - T=$T M=$MERGE_THRESH S=$SPLIT_THRESH N=$NUM_SPLITS"
            else
              echo "[WARN] Failed - see ${MBM_BASE}.inference.log"
            fi

            echo "[SAVED] ${MBM_BASE}.*"
            sleep 2
          done  # ADD THIS (closes NUM_SPLITS loop)
        done
      done
    done
    
    # UPDATED: Generate summary for this batch size in this run (now includes NUM_SPLITS)
    echo ""
    echo "Generating summary for Run $RUN_NUM, Batch $BATCH_SIZE..."
    
    SUMMARY_FILE="$OUTDIR/run${RUN_NUM}_batch${BATCH_SIZE}_summary.txt"
    echo "Run $RUN_NUM - Batch Size $BATCH_SIZE Performance Summary" > "$SUMMARY_FILE"
    echo "Generated: $(date)" >> "$SUMMARY_FILE"
    echo "Completed: $completed_experiments, Skipped: $skipped_experiments" >> "$SUMMARY_FILE"
    echo "========================================" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"

    for T in $THREADS_LIST; do
      echo "======== THREADS: $T ========" >> "$SUMMARY_FILE"
      for MERGE_THRESH in $MERGE_THRESHOLDS; do
        for SPLIT_THRESH in $SPLIT_THRESHOLDS; do
          for NUM_SPLITS in $NUM_SPLITS_LIST; do  # ADD THIS LOOP
            echo "" >> "$SUMMARY_FILE"
            echo "Config: M=$MERGE_THRESH, S=$SPLIT_THRESH, N=$NUM_SPLITS" >> "$SUMMARY_FILE"  # UPDATED
            echo "----------------------------------------" >> "$SUMMARY_FILE"
            
            # UPDATED: summary_files pattern now includes NUM_SPLITS
            summary_files=("$OUTDIR"/reorg_T${T}_B${BATCH_SIZE}_D${SPARSE_DIM}_L${LOOKUPS}_M${MERGE_THRESH}_S${SPLIT_THRESH}_N${NUM_SPLITS}_*.mbm_summary.txt)
            if [[ -f "${summary_files[0]}" ]]; then
              grep "PEAK_GBps\|AVG_GBps\|L1_MISS_RATE\|LLC_MISS_RATE\|The.*time" "${summary_files[0]}" >> "$SUMMARY_FILE" 2>/dev/null || echo "No metrics" >> "$SUMMARY_FILE"
            else
              echo "No results" >> "$SUMMARY_FILE"
            fi
          done  # ADD THIS (closes NUM_SPLITS loop)
        done
      done
      echo "" >> "$SUMMARY_FILE"
    done
    
    echo "Summary saved: $SUMMARY_FILE"
  done
done

# Generate cross-run aggregate summary
echo ""
echo "========================================"
echo "Generating aggregate summary across all runs..."
echo "========================================"

AGGREGATE_SUMMARY="$BASE_OUTDIR/aggregate_summary.txt"
echo "DLRM Multi-Batch Multi-Run Aggregate Summary" > "$AGGREGATE_SUMMARY"
echo "Generated: $(date)" >> "$AGGREGATE_SUMMARY"
echo "Runs: $NUM_RUNS" >> "$AGGREGATE_SUMMARY"
echo "Batch Sizes: $BATCH_SIZES" >> "$AGGREGATE_SUMMARY"
echo "Num Splits: $NUM_SPLITS_LIST" >> "$AGGREGATE_SUMMARY"  # ADD THIS LINE
echo "========================================" >> "$AGGREGATE_SUMMARY"
echo "" >> "$AGGREGATE_SUMMARY"

for BATCH_SIZE in $BATCH_SIZES; do
  echo "" >> "$AGGREGATE_SUMMARY"
  echo "############### BATCH SIZE: $BATCH_SIZE ###############" >> "$AGGREGATE_SUMMARY"
  echo "" >> "$AGGREGATE_SUMMARY"
  
  for RUN_NUM in $(seq 1 $NUM_RUNS); do
    echo "--- Run $RUN_NUM ---" >> "$AGGREGATE_SUMMARY"
    OUTDIR="$BASE_OUTDIR/run${RUN_NUM}_batch${BATCH_SIZE}"
    
    if [[ -f "$OUTDIR/run${RUN_NUM}_batch${BATCH_SIZE}_summary.txt" ]]; then
      tail -n 30 "$OUTDIR/run${RUN_NUM}_batch${BATCH_SIZE}_summary.txt" >> "$AGGREGATE_SUMMARY"
    else
      echo "No summary found for Run $RUN_NUM, Batch $BATCH_SIZE" >> "$AGGREGATE_SUMMARY"
    fi
    echo "" >> "$AGGREGATE_SUMMARY"
  done
done

overall_end_time=$(date +%s)
overall_elapsed=$((overall_end_time - overall_start_time))
overall_hours=$((overall_elapsed / 3600))
overall_mins=$(((overall_elapsed % 3600) / 60))

echo "" >> "$AGGREGATE_SUMMARY"
echo "========================================" >> "$AGGREGATE_SUMMARY"
echo "Total experiment time: ${overall_hours}h ${overall_mins}m" >> "$AGGREGATE_SUMMARY"
echo "========================================" >> "$AGGREGATE_SUMMARY"

echo ""
echo "========================================"
echo "ALL EXPERIMENTS COMPLETED!"
echo "========================================"
echo "Total runs: $NUM_RUNS"
echo "Batch sizes tested: $BATCH_SIZES"
echo "Num splits tested: $NUM_SPLITS_LIST"  # ADD THIS LINE
echo "Total experiments: $total_experiments"
echo "Total time: ${overall_hours}h ${overall_mins}m"
echo ""
echo "Results stored in: $BASE_OUTDIR"
echo "  - run1_batch16380/"
echo "  - run1_batch1024/"
echo "  - run1_batch512/"
echo "  - run2_batch16380/"
echo "  - ..."
echo ""
echo "Aggregate summary: $AGGREGATE_SUMMARY"
echo "========================================"

cat "$AGGREGATE_SUMMARY"
