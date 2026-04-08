#!/bin/bash

# comparison_script.sh - Compare CAFE/Sketch version vs Original DLRM
# WITH CACHE DROPPING for fair comparison

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}DLRM Performance Comparison${NC}"
echo -e "${BLUE}Sketch (CAFE) vs Original${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Check if running with sudo privileges (needed for cache dropping)
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}Error: This script must be run with sudo for cache dropping${NC}"
    echo "Usage: sudo -E ./comparison_script.sh"
    exit 1
fi

# Get the actual user's home directory (not root's)
ACTUAL_USER=${SUDO_USER:-$USER}
ACTUAL_HOME=$(eval echo ~$ACTUAL_USER)

echo -e "${GREEN}Running as: $ACTUAL_USER${NC}"
echo -e "${GREEN}Home directory: $ACTUAL_HOME${NC}\n"

# Directories
CAFE_DIR="$ACTUAL_HOME/expr/dlrm_minrui/CAFE_old"
ORIGINAL_DIR="$ACTUAL_HOME/expr/dlrm_minrui"
DATA_DIR="$ACTUAL_HOME/expr/dlrm_minrui/criteo_24days"

# Common parameters
ARCH_SPARSE="16"
ARCH_MLP_BOT="13-512-256-64-16"
ARCH_MLP_TOP="512-256-1"
TEST_BATCH_SIZE="2048"
NUM_WORKERS="0"

# Sketch-specific parameters
COMPRESS_RATE="0.007"
HASH_RATE="0.2"
SKETCH_MODEL="cafe_experiments_20260120_045901/exp1_150x_conservative/model.pt"

# Output files
SKETCH_OUTPUT="sketch_timing_results.txt"
ORIGINAL_OUTPUT="original_timing_results.txt"
COMPARISON_OUTPUT="comparison_summary.txt"

# Function to drop cache
drop_cache() {
    echo -e "${YELLOW}Dropping filesystem cache...${NC}"
    sync
    echo 3 > /proc/sys/vm/drop_caches
    sleep 2  # Wait for cache to be fully dropped
    echo -e "${GREEN}Cache dropped successfully${NC}"
}

# Function to run command as user
run_as_user() {
    su - $ACTUAL_USER -c "cd $1 && shift && $@"
}

# Function to extract timing from output
extract_timing() {
    local output_file=$1
    local mlp_time=$(grep "The MLP time is" "$output_file" | awk '{print $5}')
    local emb_time=$(grep "The embedding time is" "$output_file" | awk '{print $5}')
    local interact_time=$(grep "The interaction time is" "$output_file" | awk '{print $5}')
    local total_time=$(grep "The total time is" "$output_file" | awk '{print $5}')
    local accuracy=$(grep "accuracy.*%, auc.*%" "$output_file" | tail -1 | awk '{print $2}' | tr -d '%,')
    local auc=$(grep "accuracy.*%, auc.*%" "$output_file" | tail -1 | awk '{print $5}' | tr -d '%,')
    
    echo "$mlp_time $emb_time $interact_time $total_time $accuracy $auc"
}

#############################################
# 1. Run Sketch (CAFE) Version
#############################################
echo -e "${BLUE}========================================${NC}"
echo -e "${YELLOW}[1/2] Running Sketch (CAFE) version...${NC}"
echo -e "${BLUE}========================================${NC}"

# Drop cache before sketch version
drop_cache

# Run as user to preserve conda environment
su - $ACTUAL_USER << EOSU
cd "$CAFE_DIR"
time python dlrm_s_pytorch.py \
    --arch-sparse-feature-size="$ARCH_SPARSE" \
    --arch-mlp-bot="$ARCH_MLP_BOT" \
    --arch-mlp-top="$ARCH_MLP_TOP" \
    --data-generation=dataset \
    --data-set=kaggle \
    --test-mini-batch-size="$TEST_BATCH_SIZE" \
    --num-workers="$NUM_WORKERS" \
    --test-num-workers="$NUM_WORKERS" \
    --print-time \
    --sketch-flag \
    --compress-rate="$COMPRESS_RATE" \
    --hash-rate="$HASH_RATE" \
    --cat-path="$DATA_DIR/sparse" \
    --dense-path="$DATA_DIR/dense" \
    --label-path="$DATA_DIR/label" \
    --count-path="$DATA_DIR/processed_count.bin" \
    --load-model="$SKETCH_MODEL" \
    --inference-only 2>&1 | tee "$SKETCH_OUTPUT"
EOSU

echo -e "${GREEN}Sketch version completed!${NC}\n"

#############################################
# 2. Run Original Version
#############################################
echo -e "${BLUE}========================================${NC}"
echo -e "${YELLOW}[2/2] Running Original DLRM version...${NC}"
echo -e "${BLUE}========================================${NC}"

# Drop cache before original version
drop_cache

# Check if original baseline model exists
if [ -f "$ORIGINAL_DIR/baseline_model.pt" ]; then
    MODEL_FLAG="--load-model=baseline_model.pt"
else
    MODEL_FLAG=""
    echo -e "${YELLOW}Warning: No baseline model found. Running without pre-trained model.${NC}"
fi

# Run as user to preserve conda environment
su - $ACTUAL_USER << EOSU
cd "$ORIGINAL_DIR"
time python dlrm_s_pytorch.py \
    --arch-sparse-feature-size="$ARCH_SPARSE" \
    --arch-mlp-bot="$ARCH_MLP_BOT" \
    --arch-mlp-top="$ARCH_MLP_TOP" \
    --data-generation=dataset \
    --data-set=kaggle \
    --test-mini-batch-size="$TEST_BATCH_SIZE" \
    --num-workers="$NUM_WORKERS" \
    --test-num-workers="$NUM_WORKERS" \
    --print-time \
    --processed-data-file="$DATA_DIR/../criteo/kaggle_processed_sparse.bin" \
    $MODEL_FLAG \
    --inference-only 2>&1 | tee "$ORIGINAL_OUTPUT"
EOSU

echo -e "${GREEN}Original version completed!${NC}\n"

#############################################
# 3. Generate Comparison Report
#############################################
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Generating Comparison Report${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Extract timings
read sketch_mlp sketch_emb sketch_interact sketch_total sketch_acc sketch_auc <<< $(extract_timing "$CAFE_DIR/$SKETCH_OUTPUT")
read orig_mlp orig_emb orig_interact orig_total orig_acc orig_auc <<< $(extract_timing "$ORIGINAL_DIR/$ORIGINAL_OUTPUT")

# Create comparison report
{
    echo "========================================="
    echo "DLRM Performance Comparison"
    echo "Sketch (CAFE) vs Original"
    echo "Date: $(date)"
    echo "========================================="
    echo ""
    echo "Configuration:"
    echo "  - Sparse Feature Size: $ARCH_SPARSE"
    echo "  - Bottom MLP: $ARCH_MLP_BOT"
    echo "  - Top MLP: $ARCH_MLP_TOP"
    echo "  - Test Batch Size: $TEST_BATCH_SIZE"
    echo "  - Compression Rate: $COMPRESS_RATE"
    echo "  - Hash Rate: $HASH_RATE"
    echo "  - Cache: DROPPED before each run"
    echo ""
    echo "========================================="
    echo "TIMING BREAKDOWN (seconds)"
    echo "========================================="
    printf "%-20s %15s %15s %15s\n" "Component" "Sketch" "Original" "Speedup"
    echo "-----------------------------------------"
    
    # Calculate speedups
    if [ ! -z "$sketch_mlp" ] && [ ! -z "$orig_mlp" ]; then
        mlp_speedup=$(echo "scale=2; $orig_mlp / $sketch_mlp" | bc)
        emb_speedup=$(echo "scale=2; $orig_emb / $sketch_emb" | bc)
        interact_speedup=$(echo "scale=2; $orig_interact / $sketch_interact" | bc)
        total_speedup=$(echo "scale=2; $orig_total / $sketch_total" | bc)
        
        printf "%-20s %15.2f %15.2f %15.2fx\n" "MLP Time" "$sketch_mlp" "$orig_mlp" "$mlp_speedup"
        printf "%-20s %15.2f %15.2f %15.2fx\n" "Embedding Time" "$sketch_emb" "$orig_emb" "$emb_speedup"
        printf "%-20s %15.2f %15.2f %15.2fx\n" "Interaction Time" "$sketch_interact" "$orig_interact" "$interact_speedup"
        echo "-----------------------------------------"
        printf "%-20s %15.2f %15.2f %15.2fx\n" "TOTAL Time" "$sketch_total" "$orig_total" "$total_speedup"
        
        # Calculate percentage breakdown
        echo ""
        echo "Percentage Breakdown (Sketch):"
        sketch_sum=$(echo "scale=2; $sketch_mlp + $sketch_emb + $sketch_interact" | bc)
        sketch_mlp_pct=$(echo "scale=2; 100 * $sketch_mlp / $sketch_sum" | bc)
        sketch_emb_pct=$(echo "scale=2; 100 * $sketch_emb / $sketch_sum" | bc)
        sketch_interact_pct=$(echo "scale=2; 100 * $sketch_interact / $sketch_sum" | bc)
        printf "  MLP: %.1f%%, Embedding: %.1f%%, Interaction: %.1f%%\n" "$sketch_mlp_pct" "$sketch_emb_pct" "$sketch_interact_pct"
        
        echo ""
        echo "Percentage Breakdown (Original):"
        orig_sum=$(echo "scale=2; $orig_mlp + $orig_emb + $orig_interact" | bc)
        orig_mlp_pct=$(echo "scale=2; 100 * $orig_mlp / $orig_sum" | bc)
        orig_emb_pct=$(echo "scale=2; 100 * $orig_emb / $orig_sum" | bc)
        orig_interact_pct=$(echo "scale=2; 100 * $orig_interact / $orig_sum" | bc)
        printf "  MLP: %.1f%%, Embedding: %.1f%%, Interaction: %.1f%%\n" "$orig_mlp_pct" "$orig_emb_pct" "$orig_interact_pct"
    else
        echo "Error: Could not extract timing information"
    fi
    
    echo ""
    echo "========================================="
    echo "ACCURACY COMPARISON"
    echo "========================================="
    printf "%-20s %15s %15s %15s\n" "Metric" "Sketch" "Original" "Difference"
    echo "-----------------------------------------"
    
    if [ ! -z "$sketch_acc" ] && [ ! -z "$orig_acc" ]; then
        acc_diff=$(echo "scale=3; $sketch_acc - $orig_acc" | bc)
        auc_diff=$(echo "scale=3; $sketch_auc - $orig_auc" | bc)
        
        printf "%-20s %14.2f%% %14.2f%% %14.3f%%\n" "Accuracy" "$sketch_acc" "$orig_acc" "$acc_diff"
        printf "%-20s %14.2f%% %14.2f%% %14.3f%%\n" "AUC" "$sketch_auc" "$orig_auc" "$auc_diff"
    else
        echo "Error: Could not extract accuracy information"
    fi
    
    echo ""
    echo "========================================="
    echo "COMPRESSION ANALYSIS"
    echo "========================================="
    
    echo "Compression Rate: ${COMPRESS_RATE} (0.7%)"
    echo "Hash Rate: ${HASH_RATE} (20%)"
    echo "Effective Compression: ~143x (estimated)"
    
    echo ""
    echo "========================================="
    echo "SUMMARY"
    echo "========================================="
    
    if [ ! -z "$total_speedup" ]; then
        if (( $(echo "$total_speedup > 1" | bc -l) )); then
            echo "Result: Original is ${total_speedup}x FASTER than Sketch"
        else
            inverse_speedup=$(echo "scale=2; 1 / $total_speedup" | bc)
            echo "Result: Sketch is ${inverse_speedup}x FASTER than Original"
        fi
    fi
    
    if [ ! -z "$acc_diff" ]; then
        echo "Accuracy Loss: ${acc_diff}%"
    fi
    
    echo ""
    echo "Output files:"
    echo "  - Sketch results: $CAFE_DIR/$SKETCH_OUTPUT"
    echo "  - Original results: $ORIGINAL_DIR/$ORIGINAL_OUTPUT"
    echo "  - This summary: $COMPARISON_OUTPUT"
    
} | tee "$COMPARISON_OUTPUT"

echo -e "\n${GREEN}Comparison complete! Summary saved to $COMPARISON_OUTPUT${NC}"

# Display the comparison
cat "$COMPARISON_OUTPUT"
