#!/usr/bin/env bash
set -euo pipefail

# Profile embedding access patterns for hot/cold splitting

DLRM_BIN="${DLRM_BIN:-dlrm_hot.py}"
RAW_FILE="${RAW_FILE:-./input/train.txt}"
PROC_FILE="${PROC_FILE:-./input/kaggleAdDisplayChallenge_processed.npz}"
MLP_BOT_PREFIX="${MLP_BOT_PREFIX:-13-512-256-64}"
MLP_TOP="${MLP_TOP:-512-256-1}"
PRINT_FREQ="${PRINT_FREQ:-8192}"
TEST_WORKERS="${TEST_WORKERS:-16}"

SPARSE_DIM="${SPARSE_DIM:-16}"
LOOKUPS="${LOOKUPS:-64}"

# Profiling configuration
PROFILE_BATCHES="${PROFILE_BATCHES:-2000}"
HOTCOLD_PERCENTILES="${HOTCOLD_PERCENTILES:-80}"  # Will create separate profiles
HOTCOLD_THRESHOLD="${HOTCOLD_THRESHOLD:-1000000}"

# Test one representative configuration for profiling
PROFILE_BATCH_SIZE="${PROFILE_BATCH_SIZE:-16384}"
PROFILE_THREADS="${PROFILE_THREADS:-40}"

MEMNODE=0
PROFILE_DIR="${PROFILE_DIR:-./profiles}"
PYTHON_PATH="${PYTHON_PATH:-$(which python3)}"

mkdir -p "$PROFILE_DIR"

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

echo "========================================"
echo "PHASE 1: PROFILING EMBEDDING ACCESS PATTERNS"
echo "========================================"
echo "DLRM Binary: $DLRM_BIN"
echo "Profile batch size: $PROFILE_BATCH_SIZE"
echo "Profile threads: $PROFILE_THREADS"
echo "Profile batches: $PROFILE_BATCHES"
echo "Percentiles: $HOTCOLD_PERCENTILES"
echo "Size threshold: $HOTCOLD_THRESHOLD"
echo "Output directory: $PROFILE_DIR"
echo "========================================"
echo ""

CPU_POOL=$(select_cpu_pool $PROFILE_THREADS)
MLP_BOT="${MLP_BOT_PREFIX}-${SPARSE_DIM}"
export OMP_NUM_THREADS=$PROFILE_THREADS

echo "CPU Pool: $CPU_POOL"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo ""

# Check if profiles already exist
existing_profiles=0
for P in $HOTCOLD_PERCENTILES; do
    PROFILE_FILE="$PROFILE_DIR/kaggle_profile_P${P}.pkl"
    ANALYZED_FILE="$PROFILE_DIR/kaggle_profile_P${P}_analyzed.pkl"
    if [[ -f "$PROFILE_FILE" ]] && [[ -f "$ANALYZED_FILE" ]]; then
        echo "[EXISTS] Profile P=${P}%: $PROFILE_FILE"
        existing_profiles=$((existing_profiles + 1))
    fi
done

if [[ $existing_profiles -eq $(echo $HOTCOLD_PERCENTILES | wc -w) ]]; then
    echo ""
    echo "All profiles already exist! Skipping profiling."
    echo "Delete files in $PROFILE_DIR to re-profile."
    echo ""
    exit 0
fi

echo ""
echo "Starting profiling run..."
echo ""

start_time=$(date +%s)

# Run profiling for each percentile threshold
for P in $HOTCOLD_PERCENTILES; do
    PROFILE_FILE="$PROFILE_DIR/kaggle_profile_P${P}.pkl"
    ANALYZED_FILE="$PROFILE_DIR/kaggle_profile_P${P}_analyzed.pkl"
    
    if [[ -f "$PROFILE_FILE" ]] && [[ -f "$ANALYZED_FILE" ]]; then
        echo ">>> SKIP: Profile P=${P}% already exists"
        echo ""
        continue
    fi
    
    echo "========================================"
    echo ">>> PROFILING: Percentile=${P}%"
    echo "========================================"
    echo "Output: $PROFILE_FILE"
    echo ""
    
    # Clear caches
    sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
    cleanup_processes
    sleep 2
    
    # Run profiling
    stamp=$(date +%Y%m%d_%H%M%S)
    logfile="$PROFILE_DIR/profile_P${P}_${stamp}.log"
    
    numactl --physcpubind="$CPU_POOL" --membind="$MEMNODE" \
        "$PYTHON_PATH" "$DLRM_BIN" \
        --arch-sparse-feature-size=$SPARSE_DIM \
        --arch-mlp-bot="$MLP_BOT" \
        --arch-mlp-top="$MLP_TOP" \
        --data-generation=dataset \
        --data-set=kaggle \
        --raw-data-file="$RAW_FILE" \
        --processed-data-file="$PROC_FILE" \
        --dataset-multiprocessing \
        --loss-function=bce \
        --round-targets=True \
        --mini-batch-size=$PROFILE_BATCH_SIZE \
        --print-freq=$PRINT_FREQ \
        --print-time \
        --test-mini-batch-size=$PROFILE_BATCH_SIZE \
        --test-num-workers=$TEST_WORKERS \
        --num-indices-per-lookup=$LOOKUPS \
        --profile-embedding-access \
        --profile-batches=$PROFILE_BATCHES \
        --save-access-profile="$PROFILE_FILE" \
        --hotcold-percentile=$P \
        --hotcold-emb-threshold=$HOTCOLD_THRESHOLD \
        --inference-only \
        --nepochs=0 \
        > "$logfile" 2>&1
    
    exit_code=$?
    
    if [[ $exit_code -eq 0 ]] && [[ -f "$PROFILE_FILE" ]] && [[ -f "$ANALYZED_FILE" ]]; then
        echo "[SUCCESS] Profiling completed"
        echo "  Raw profile: $PROFILE_FILE"
        echo "  Analyzed profile: $ANALYZED_FILE"
        
        # Print summary from log
        echo ""
        echo "=== PROFILE SUMMARY ==="
        grep -E "\[PROFILE\].*C[0-9]+" "$logfile" || true
        echo "======================="
    else
        echo "[ERROR] Profiling failed! Exit code: $exit_code"
        echo "Check log: $logfile"
    fi
    
    echo ""
    sleep 2
done

end_time=$(date +%s)
elapsed=$((end_time - start_time))
mins=$((elapsed / 60))
secs=$((elapsed % 60))

echo ""
echo "========================================"
echo "PROFILING COMPLETE!"
echo "========================================"
echo "Time elapsed: ${mins}m ${secs}s"
echo "Profiles saved in: $PROFILE_DIR"
echo ""

# List created profiles
echo "Created profiles:"
for P in $HOTCOLD_PERCENTILES; do
    PROFILE_FILE="$PROFILE_DIR/kaggle_profile_P${P}.pkl"
    ANALYZED_FILE="$PROFILE_DIR/kaggle_profile_P${P}_analyzed.pkl"
    if [[ -f "$PROFILE_FILE" ]] && [[ -f "$ANALYZED_FILE" ]]; then
        size_raw=$(du -h "$PROFILE_FILE" | cut -f1)
        size_analyzed=$(du -h "$ANALYZED_FILE" | cut -f1)
        echo "  P=${P}%:"
        echo "    Raw: $PROFILE_FILE ($size_raw)"
        echo "    Analyzed: $ANALYZED_FILE ($size_analyzed)"
    fi
done

echo ""
echo "Next step: Run hot/cold split testing with:"
echo "  ./test_hotcold.sh"
echo "========================================"