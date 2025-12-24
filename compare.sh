#!/usr/bin/env bash
set -euo pipefail

# Compare three approaches on FULL TEST DATASET
# Order: Baseline -> Preprocessed -> Runtime (slowest last)
# Single run with optimal settings

DLRM_BIN="${DLRM_BIN:-dlrm_hot.py}"
RAW_FILE="${RAW_FILE:-./input/train.txt}"
PROC_FILE="${PROC_FILE:-./input/kaggleAdDisplayChallenge_processed.npz}"
PROFILE_FILE="${PROFILE_FILE:-./profiles/kaggle_profile_P80_FULL.pkl}"
PREPROCESSED_DIR="${PREPROCESSED_DIR:-./preprocessed_hotcold_identical}"


BATCH_SIZES="${BATCH_SIZES:-16384 8192}"  
THREADS_LIST="${THREADS_LIST:-40 60 80}"
NUM_RUNS=3

MEMNODE=0
OUTPUT_DIR="${OUTPUT_DIR:-./comparison_results}"
PYTHON_PATH="${PYTHON_PATH:-$(which python3)}"

mkdir -p "$OUTPUT_DIR"

# Fixed DLRM parameters
SPARSE_DIM=64
MLP_BOT="13-512-256-64"
MLP_TOP="512-256-1"
PRINT_FREQ=10000
TEST_WORKERS=8  # Optimal for 40-80 threads
LOOKUPS=64

select_cpu_pool() {
    local threads=$1
    if [ "$threads" -le 20 ]; then
        echo $(seq 0 2 38 | tr '\n' ',' | sed 's/,$//')
    elif [ "$threads" -le 40 ]; then
        echo $(seq 0 2 78 | tr '\n' ',' | sed 's/,$//')
    elif [ "$threads" -le 80 ]; then
        echo $(seq 0 2 78 | tr '\n' ',' | sed 's/,$//')","$(seq 80 2 158 | tr '\n' ',' | sed 's/,$//')
    else
        echo $(seq 0 2 158 | tr '\n' ',' | sed 's/,$//')
    fi
}

cleanup_processes() {
    sudo pkill -9 pqos 2>/dev/null || true
    sudo pkill -9 perf 2>/dev/null || true
    pkill -9 -f "dlrm_hot.py" 2>/dev/null || true
    pkill -9 -f "python.*dlrm" 2>/dev/null || true
    rm -f /run/lock/libpqos 2>/dev/null || true
}

trap cleanup_processes EXIT INT TERM

run_with_monitoring() {
    local label="$1"
    local outbase="$2"
    shift 2
    
    local mbm_csv="${outbase}.mbm.csv"
    local cache_csv="${outbase}.cache.csv"
    local summary="${outbase}.summary.txt"
    local inference_log="${outbase}.inference.log"
    
    echo "[${label}] Starting monitoring..." | tee "$summary"
    
    cleanup_processes
    sleep 1
    
    if mount | grep -q resctrl; then
        sudo umount /sys/fs/resctrl 2>/dev/null || true
    fi
    
    sudo touch /run/lock/libpqos
    sudo chmod 666 /run/lock/libpqos
    
    # Start inference
    "$@" > "$inference_log" 2>&1 &
    local inference_pid=$!
    echo "[${label}] Inference PID: $inference_pid" | tee -a "$summary"
    sleep 3
    
    # Start memory bandwidth monitoring
    sudo pqos -I -m "mbt:[$CPU_POOL]" -u csv -o "$mbm_csv" &
    local pqos_pid=$!
    
    # Start cache monitoring
    sudo perf stat -C "$CPU_POOL" -I 1000 -x, \
        -e L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses \
        -o "$cache_csv" sleep 999999 &
    local perf_pid=$!
    
    # Wait for completion
    wait "$inference_pid" 2>/dev/null
    local exit_code=$?
    echo "[${label}] Completed with exit code: $exit_code" | tee -a "$summary"
    
    # Stop monitoring
    sudo kill "$pqos_pid" "$perf_pid" 2>/dev/null || true
    sleep 1
    cleanup_processes
    
    # Analyze results
    echo "" | tee -a "$summary"
    echo "=== Memory Bandwidth ===" | tee -a "$summary"
    if [[ -f "$mbm_csv" ]]; then
        awk -F',' 'NR>1 && $5>0 {sum+=$5; count++; if($5>peak) peak=$5} 
                   END {if(count>0) {
                     printf "Peak: %.2f GB/s\nAvg: %.2f GB/s\n", peak/1024, sum/count/1024;
                     printf "PEAK_BW=%.3f\nAVG_BW=%.3f\n", peak/1024, sum/count/1024
                   }}' "$mbm_csv" | tee -a "$summary"
    fi
    
    echo "" | tee -a "$summary"
    echo "=== Cache Statistics ===" | tee -a "$summary"
    if [[ -f "$cache_csv" ]]; then
        awk -F',' '
          /L1-dcache-loads/ && $2!="<not counted>" && $2!="" {l1_loads+=$2}
          /L1-dcache-load-misses/ && $2!="<not counted>" && $2!="" {l1_misses+=$2}
          /LLC-loads/ && $2!="<not counted>" && $2!="" {llc_loads+=$2}
          /LLC-load-misses/ && $2!="<not counted>" && $2!="" {llc_misses+=$2}
          END {
            if(l1_loads>0) {
              printf "L1 Loads: %d\nL1 Misses: %d\nL1 Miss Rate: %.2f%%\n", 
                     l1_loads, l1_misses, (l1_misses/l1_loads)*100
              printf "L1_MISS_RATE=%.2f\n", (l1_misses/l1_loads)*100
            }
            if(llc_loads>0) {
              printf "LLC Loads: %d\nLLC Misses: %d\nLLC Miss Rate: %.2f%%\n",
                     llc_loads, llc_misses, (llc_misses/llc_loads)*100
              printf "LLC_MISS_RATE=%.2f\n", (llc_misses/llc_loads)*100
            }
          }' "$cache_csv" | tee -a "$summary"
    fi
    
    echo "" | tee -a "$summary"
    echo "=== Performance ===" | tee -a "$summary"
    if [[ -f "$inference_log" ]]; then
        grep -E "embedding time|total time|accuracy" "$inference_log" | tee -a "$summary" || true
        
        emb_time=$(grep "The embedding time is" "$inference_log" | awk '{print $NF}' || echo "0")
        total_time=$(grep "The total time is" "$inference_log" | awk '{print $NF}' || echo "0")
        accuracy=$(grep "accuracy" "$inference_log" | tail -1 | awk '{print $2}' | tr -d ',%' || echo "0")
        
        remap_time=$(grep "Total remap time:" "$inference_log" | awk '{print $4}' | tr -d 's' || echo "0")
        lookup_time=$(grep "Total lookup time" "$inference_log" | awk '{print $5}' | tr -d 's' || echo "0")
        
        echo "EMB_TIME=$emb_time" | tee -a "$summary"
        echo "TOTAL_TIME=$total_time" | tee -a "$summary"
        echo "ACCURACY=$accuracy" | tee -a "$summary"
        [[ -n "$remap_time" && "$remap_time" != "0" ]] && echo "REMAP_TIME=$remap_time" | tee -a "$summary"
        [[ -n "$lookup_time" && "$lookup_time" != "0" ]] && echo "LOOKUP_TIME=$lookup_time" | tee -a "$summary"
    fi
    
    return $exit_code
}

echo "========================================"
echo "DLRM FULL DATASET COMPARISON"
echo "========================================"
echo "Batch sizes: $BATCH_SIZES"
echo "Threads: $THREADS_LIST"
echo "Test workers: $TEST_WORKERS (optimized)"
echo "Runs: $NUM_RUNS (single run)"
echo "Test set: FULL (4,584,062 samples)"
echo "Order: Baseline -> Preprocessed -> Runtime"
echo "========================================"
echo ""

# Calculate iterations and time
echo "BATCH ITERATIONS:"
for bs in $BATCH_SIZES; do
    batches=$(( (4584062 + bs - 1) / bs ))
    echo "  Batch $bs: $batches batches"
done
echo ""

total_configs=$(($(echo $BATCH_SIZES | wc -w) * $(echo $THREADS_LIST | wc -w)))
total_experiments=$((total_configs * 3))  # 3 approaches × 1 run

echo "Total configurations: $total_configs"
echo "Total experiments: $total_experiments"
echo ""
echo "ESTIMATED TIME:"
echo "  Batch 16384 (280 batches): ~4 min per experiment"
echo "  Batch 8192 (560 batches): ~8 min per experiment"
echo ""
echo "Per configuration (3 approaches):"
echo "  16384: ~12 min"
echo "  8192: ~24 min"
echo ""
echo "Total estimated: ~2 hours"
echo "  - 16384 configs: ~36 min (3 threads × 12 min)"
echo "  - 8192 configs: ~72 min (3 threads × 24 min)"
echo ""
read -p "Press ENTER to continue or Ctrl+C to cancel..."

# Check prerequisites
if [[ ! -f "$PROFILE_FILE" ]]; then
    echo "ERROR: Profile not found: $PROFILE_FILE"
    exit 1
fi

if [[ ! -d "$PREPROCESSED_DIR" ]]; then
    echo "ERROR: Preprocessed directory not found: $PREPROCESSED_DIR"
    exit 1
fi

start_time=$(date +%s)
experiment_count=0

# Main loop: batch sizes -> threads -> {baseline, preprocessed, runtime}
for BATCH_SIZE in $BATCH_SIZES; do
    for THREADS in $THREADS_LIST; do
        echo ""
        echo "================================================"
        echo "Batch=$BATCH_SIZE, Threads=$THREADS"
        echo "================================================"
        
        RUN_DIR="$OUTPUT_DIR/B${BATCH_SIZE}_T${THREADS}"
        mkdir -p "$RUN_DIR"
        
        CPU_POOL=$(select_cpu_pool $THREADS)
        export OMP_NUM_THREADS=$THREADS
        
        # ========== BASELINE ==========
        # Replace the experiment sections with skip checks:

        # ========== BASELINE ==========
        experiment_count=$((experiment_count + 1))

        if [[ -f "$RUN_DIR/baseline.summary.txt" ]]; then
            echo "[$experiment_count/$total_experiments] >>> SKIP: BASELINE (already exists)"
        else
            echo ""
            echo "[$experiment_count/$total_experiments] >>> BASELINE"
            sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
            sleep 2
            
            cmd=(numactl --physcpubind="$CPU_POOL" --membind="$MEMNODE"
                "$PYTHON_PATH" "$DLRM_BIN"
                --arch-sparse-feature-size=$SPARSE_DIM
                --arch-mlp-bot="$MLP_BOT"
                --arch-mlp-top="$MLP_TOP"
                --data-generation=dataset
                --data-set=kaggle
                --raw-data-file="$RAW_FILE"
                --processed-data-file="$PROC_FILE"
                --dataset-multiprocessing
                --loss-function=bce
                --round-targets=True
                --mini-batch-size=$BATCH_SIZE
                --print-freq=$PRINT_FREQ
                --print-time
                --test-mini-batch-size=$BATCH_SIZE
                --test-num-workers=$TEST_WORKERS
                --inference-only
                --nepochs=0)
            
            run_with_monitoring "BASELINE" "$RUN_DIR/baseline" "${cmd[@]}" || true
            sleep 2
        fi

        # ========== PREPROCESSED ==========
        experiment_count=$((experiment_count + 1))

        if [[ -f "$RUN_DIR/preprocessed.summary.txt" ]]; then
            echo "[$experiment_count/$total_experiments] >>> SKIP: PREPROCESSED (already exists)"
        else
            echo ""
            echo "[$experiment_count/$total_experiments] >>> PREPROCESSED"
            sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
            sleep 2
            
            cmd=(numactl --physcpubind="$CPU_POOL" --membind="$MEMNODE"
                "$PYTHON_PATH" "$DLRM_BIN"
                --arch-sparse-feature-size=$SPARSE_DIM
                --arch-mlp-bot="$MLP_BOT"
                --arch-mlp-top="$MLP_TOP"
                --use-hotcold-preprocessed
                --hotcold-preprocessed-dir="$PREPROCESSED_DIR"
                --loss-function=bce
                --round-targets=True
                --mini-batch-size=$BATCH_SIZE
                --print-freq=$PRINT_FREQ
                --print-time
                --inference-only
                --nepochs=0)
            
            run_with_monitoring "PREPROCESSED" "$RUN_DIR/preprocessed" "${cmd[@]}" || true
            sleep 2
        fi

        # ========== RUNTIME REMAPPING ==========
        experiment_count=$((experiment_count + 1))

        if [[ -f "$RUN_DIR/runtime.summary.txt" ]]; then
            echo "[$experiment_count/$total_experiments] >>> SKIP: RUNTIME (already exists)"
        else
            echo ""
            echo "[$experiment_count/$total_experiments] >>> RUNTIME REMAPPING (slowest)"
            sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
            sleep 2
            
            cmd=(numactl --physcpubind="$CPU_POOL" --membind="$MEMNODE"
                "$PYTHON_PATH" "$DLRM_BIN"
                --arch-sparse-feature-size=$SPARSE_DIM
                --arch-mlp-bot="$MLP_BOT"
                --arch-mlp-top="$MLP_TOP"
                --data-generation=dataset
                --data-set=kaggle
                --raw-data-file="$RAW_FILE"
                --processed-data-file="$PROC_FILE"
                --dataset-multiprocessing
                --loss-function=bce
                --round-targets=True
                --mini-batch-size=$BATCH_SIZE
                --print-freq=$PRINT_FREQ
                --print-time
                --test-mini-batch-size=$BATCH_SIZE
                --test-num-workers=$TEST_WORKERS
                --thread-count=$THREADS                  
                --load-access-profile="$PROFILE_FILE"
                --hotcold-percentile=80
                --hotcold-emb-threshold=1000000
                --inference-only
                --nepochs=0)
            
            run_with_monitoring "RUNTIME" "$RUN_DIR/runtime" "${cmd[@]}" || true
            sleep 2
        fi
    done
done

# ========== GENERATE COMPARISON SUMMARY ==========
SUMMARY="$OUTPUT_DIR/comparison_summary.txt"

echo "DLRM Full Dataset Comparison Summary" > "$SUMMARY"
echo "Generated: $(date)" >> "$SUMMARY"
echo "Test samples: 4,584,062" >> "$SUMMARY"
echo "Batch sizes: $BATCH_SIZES" >> "$SUMMARY"
echo "Threads: $THREADS_LIST" >> "$SUMMARY"
echo "Test workers: $TEST_WORKERS" >> "$SUMMARY"
echo "========================================" >> "$SUMMARY"
echo "" >> "$SUMMARY"

for BATCH_SIZE in $BATCH_SIZES; do
    for THREADS in $THREADS_LIST; do
        echo "" >> "$SUMMARY"
        echo "### Batch=$BATCH_SIZE, Threads=$THREADS ###" >> "$SUMMARY"
        echo "" >> "$SUMMARY"
        
        printf "%-20s %12s %12s %12s\n" "Metric" "Baseline" "Preprocessed" "Runtime" >> "$SUMMARY"
        printf "%-20s %12s %12s %12s\n" "--------------------" "------------" "------------" "------------" >> "$SUMMARY"
        
        # Extract metrics
        for approach in baseline preprocessed runtime; do
            summary_file="$OUTPUT_DIR/B${BATCH_SIZE}_T${THREADS}/${approach}.summary.txt"
            if [[ -f "$summary_file" ]]; then
                declare "${approach}_emb"=$(grep "^EMB_TIME=" "$summary_file" 2>/dev/null | cut -d= -f2 || echo "N/A")
                declare "${approach}_total"=$(grep "^TOTAL_TIME=" "$summary_file" 2>/dev/null | cut -d= -f2 || echo "N/A")
                declare "${approach}_acc"=$(grep "^ACCURACY=" "$summary_file" 2>/dev/null | cut -d= -f2 || echo "N/A")
                declare "${approach}_peak_bw"=$(grep "^PEAK_BW=" "$summary_file" 2>/dev/null | cut -d= -f2 || echo "N/A")
                declare "${approach}_avg_bw"=$(grep "^AVG_BW=" "$summary_file" 2>/dev/null | cut -d= -f2 || echo "N/A")
                declare "${approach}_l1_miss"=$(grep "^L1_MISS_RATE=" "$summary_file" 2>/dev/null | cut -d= -f2 || echo "N/A")
                declare "${approach}_llc_miss"=$(grep "^LLC_MISS_RATE=" "$summary_file" 2>/dev/null | cut -d= -f2 || echo "N/A")
                declare "${approach}_remap"=$(grep "^REMAP_TIME=" "$summary_file" 2>/dev/null | cut -d= -f2 || echo "N/A")
            fi
        done
        
        printf "%-20s %12s %12s %12s\n" "EMB_TIME (s)" "$baseline_emb" "$preprocessed_emb" "$runtime_emb" >> "$SUMMARY"
        printf "%-20s %12s %12s %12s\n" "TOTAL_TIME (s)" "$baseline_total" "$preprocessed_total" "$runtime_total" >> "$SUMMARY"
        printf "%-20s %12s %12s %12s\n" "ACCURACY (%)" "$baseline_acc" "$preprocessed_acc" "$runtime_acc" >> "$SUMMARY"
        echo "" >> "$SUMMARY"
        printf "%-20s %12s %12s %12s\n" "PEAK_BW (GB/s)" "$baseline_peak_bw" "$preprocessed_peak_bw" "$runtime_peak_bw" >> "$SUMMARY"
        printf "%-20s %12s %12s %12s\n" "AVG_BW (GB/s)" "$baseline_avg_bw" "$preprocessed_avg_bw" "$runtime_avg_bw" >> "$SUMMARY"
        echo "" >> "$SUMMARY"
        printf "%-20s %12s %12s %12s\n" "L1_MISS_RATE (%)" "$baseline_l1_miss" "$preprocessed_l1_miss" "$runtime_l1_miss" >> "$SUMMARY"
        printf "%-20s %12s %12s %12s\n" "LLC_MISS_RATE (%)" "$baseline_llc_miss" "$preprocessed_llc_miss" "$runtime_llc_miss" >> "$SUMMARY"
        
        if [[ "$runtime_remap" != "N/A" && "$runtime_remap" != "0" ]]; then
            echo "" >> "$SUMMARY"
            echo "Runtime remap overhead: ${runtime_remap}s" >> "$SUMMARY"
        fi
        
        # Speedup calculation
        if [[ "$baseline_total" != "N/A" && "$preprocessed_total" != "N/A" ]]; then
            speedup=$(echo "scale=2; $baseline_total / $preprocessed_total" | bc)
            echo "Preprocessed speedup: ${speedup}x" >> "$SUMMARY"
        fi
    done
done

end_time=$(date +%s)
elapsed=$((end_time - start_time))
hours=$((elapsed / 3600))
mins=$(((elapsed % 3600) / 60))

echo "" >> "$SUMMARY"
echo "========================================" >> "$SUMMARY"
echo "Total experiment time: ${hours}h ${mins}m" >> "$SUMMARY"
echo "========================================" >> "$SUMMARY"

echo ""
echo "========================================"
echo "COMPARISON COMPLETE!"
echo "========================================"
echo "Results: $OUTPUT_DIR"
echo "Summary: $SUMMARY"
echo "Total time: ${hours}h ${mins}m"
echo ""
cat "$SUMMARY"
