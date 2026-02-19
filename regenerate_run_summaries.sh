#!/usr/bin/env bash
set -euo pipefail

# Regenerate run summary files after bandwidth reprocessing
# Usage: ./regenerate_run_summaries.sh <base_directory>

if [ $# -lt 1 ]; then
  echo "Usage: $0 <base_directory>"
  echo "Example: $0 ./merge_split_multibatch_runs"
  exit 1
fi

BASE_DIR="$1"

if [ ! -d "$BASE_DIR" ]; then
  echo "Error: Directory '$BASE_DIR' not found"
  exit 1
fi

echo "========================================"
echo "Regenerating Run Summary Files"
echo "========================================"
echo "Base directory: $BASE_DIR"
echo ""

# Find all run directories
mapfile -t run_dirs < <(find "$BASE_DIR" -maxdepth 1 -type d -name "run*_batch*" | sort)

if [ ${#run_dirs[@]} -eq 0 ]; then
  echo "No run directories found matching pattern 'run*_batch*'"
  exit 1
fi

echo "Found ${#run_dirs[@]} run directories"
echo ""

for run_dir in "${run_dirs[@]}"; do
  dir_name=$(basename "$run_dir")
  
  if [[ "$dir_name" =~ run([0-9]+)_batch([0-9]+) ]]; then
    run_num=${BASH_REMATCH[1]}
    batch_size=${BASH_REMATCH[2]}
    
    summary_file="$run_dir/run${run_num}_batch${batch_size}_summary.txt"
    
    echo "Creating/Updating: $summary_file"
    
    # Always backup existing summary if it exists
    if [ -f "$summary_file" ]; then
      backup_file="${summary_file}.backup.$(date +%Y%m%d_%H%M%S)"
      cp "$summary_file" "$backup_file"
      echo "  - Backed up existing file to $(basename $backup_file)"
    fi
    
    # Always create new summary
    cat > "$summary_file" <<EOF
Run $run_num - Batch Size $batch_size Performance Summary (REPROCESSED)
Generated: $(date)
========================================

EOF
    
    # Find all result files in this directory
    mapfile -t result_files < <(find "$run_dir" -name "reorg_T*_B${batch_size}_*.mbm_summary.txt" ! -name "*.backup*" | sort)
    
    if [ ${#result_files[@]} -eq 0 ]; then
      echo "  WARNING: No experiment files found in $run_dir"
      echo "No experiment data found" >> "$summary_file"
      continue
    fi
    
    echo "  - Found ${#result_files[@]} experiment files"
    
    # Group by configuration
    declare -A configs
    
    for result_file in "${result_files[@]}"; do
      filename=$(basename "$result_file")
      
      # Extract configuration from filename - NOW INCLUDING NUM_SPLITS
      if [[ "$filename" =~ T([0-9]+)_B([0-9]+).*M([0-9]+)_S([0-9]+)_N([0-9]+) ]]; then
        threads=${BASH_REMATCH[1]}
        merge=${BASH_REMATCH[3]}
        split=${BASH_REMATCH[4]}
        num_splits=${BASH_REMATCH[5]}
        
        config_key="T${threads}_M${merge}_S${split}_N${num_splits}"
        configs[$config_key]="$result_file"
      # Fallback for older files without N parameter
      elif [[ "$filename" =~ T([0-9]+)_B([0-9]+).*M([0-9]+)_S([0-9]+) ]]; then
        threads=${BASH_REMATCH[1]}
        merge=${BASH_REMATCH[3]}
        split=${BASH_REMATCH[4]}
        
        config_key="T${threads}_M${merge}_S${split}_N0"
        configs[$config_key]="$result_file"
      fi
    done
    
    # Sort and output configurations
    for config_key in $(printf '%s\n' "${!configs[@]}" | sort); do
      result_file="${configs[$config_key]}"
      base_name="${result_file%.mbm_summary.txt}"
      inference_log="${base_name}.inference.log"
      
      # Parse config key - NOW INCLUDING NUM_SPLITS
      if [[ "$config_key" =~ T([0-9]+)_M([0-9]+)_S([0-9]+)_N([0-9]+) ]]; then
        threads=${BASH_REMATCH[1]}
        merge=${BASH_REMATCH[2]}
        split=${BASH_REMATCH[3]}
        num_splits=${BASH_REMATCH[4]}
        
        # Format output with num_splits
        if [ "$num_splits" -eq 0 ]; then
          config_header="Configuration: Threads=$threads, Merge=$merge, Split=$split"
        else
          config_header="Configuration: Threads=$threads, Merge=$merge, Split=$split, NumSplits=$num_splits"
        fi
        
        cat >> "$summary_file" <<EOF

$config_header
----------------------------------------
EOF
        
        # Track if we found any metrics
        metrics_found=0
        
        # Extract bandwidth metrics from mbm_summary
        if grep -q "PEAK_GBps=" "$result_file" 2>/dev/null; then
          grep "PEAK_GBps=" "$result_file" >> "$summary_file" 2>/dev/null
          grep "AVG_GBps=" "$result_file" >> "$summary_file" 2>/dev/null
          metrics_found=1
        fi
        
        # Extract cache metrics from mbm_summary
        if grep -q "L1_MISS_RATE=" "$result_file" 2>/dev/null; then
          grep "L1_MISS_RATE=" "$result_file" >> "$summary_file" 2>/dev/null
          grep "LLC_MISS_RATE=" "$result_file" >> "$summary_file" 2>/dev/null
          metrics_found=1
        fi
        
        # Extract timing metrics - try both mbm_summary and inference log
        timing_found=0
        
        # First try mbm_summary
        if grep -q "The MLP time is" "$result_file" 2>/dev/null; then
          grep "The MLP time is" "$result_file" >> "$summary_file" 2>/dev/null
          grep "The embedding time is" "$result_file" >> "$summary_file" 2>/dev/null
          grep "The interaction time is" "$result_file" >> "$summary_file" 2>/dev/null
          grep "The total time is" "$result_file" >> "$summary_file" 2>/dev/null
          timing_found=1
          metrics_found=1
        # If not in mbm_summary, try inference log
        elif [ -f "$inference_log" ] && grep -q "The MLP time is" "$inference_log" 2>/dev/null; then
          grep "The MLP time is" "$inference_log" >> "$summary_file" 2>/dev/null
          grep "The embedding time is" "$inference_log" >> "$summary_file" 2>/dev/null
          grep "The interaction time is" "$inference_log" >> "$summary_file" 2>/dev/null
          grep "The total time is" "$inference_log" >> "$summary_file" 2>/dev/null
          timing_found=1
          metrics_found=1
        fi
        
        # Debug info if timing not found
        if [ $timing_found -eq 0 ]; then
          echo "  - WARNING: Timing data not found for Threads=$threads, Merge=$merge, Split=$split, NumSplits=$num_splits"
          echo "    Checked: $result_file"
          echo "    Checked: $inference_log"
        fi
        
        # If no metrics found, note it
        if [ $metrics_found -eq 0 ]; then
          echo "No metrics found for this configuration" >> "$summary_file"
        fi
      fi
    done
    
    echo "  ✓ Summary created/updated successfully"
    
    # Clear the associative array for next iteration
    unset configs
    declare -A configs
  fi
done

echo ""
echo "========================================"
echo "Completed!"
echo "========================================"
echo "Summary files updated with:"
echo "  - Memory bandwidth (PEAK_GBps, AVG_GBps)"
echo "  - Cache miss rates (L1_MISS_RATE, LLC_MISS_RATE)"  
echo "  - Timing breakdown (MLP, embedding, interaction, total)"
echo "  - Number of splits (NumSplits) for split configurations"
echo ""
echo "Total summaries processed: ${#run_dirs[@]}"
echo "========================================"
