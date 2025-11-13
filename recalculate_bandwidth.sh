#!/usr/bin/env bash
set -euo pipefail

# Script to recalculate bandwidth metrics from .mbm.csv files
# and update the .mbm_summary.txt files

RESULTS_DIR="${1:-./training_bw_stress_logs}"
BACKUP_DIR="${RESULTS_DIR}/backup_summaries_$(date +%Y%m%d_%H%M%S)"

echo "=========================================="
echo "Recalculating Bandwidth Metrics"
echo "=========================================="
echo "Results directory: $RESULTS_DIR"
echo ""

# Check if directory exists
if [ ! -d "$RESULTS_DIR" ]; then
    echo "ERROR: Directory not found: $RESULTS_DIR"
    echo "Usage: $0 <results_directory>"
    exit 1
fi

# Find all .mbm.csv files
mbm_files=("$RESULTS_DIR"/*.mbm.csv)
if [ ! -f "${mbm_files[0]}" ]; then
    echo "ERROR: No .mbm.csv files found in $RESULTS_DIR"
    exit 1
fi

num_files=${#mbm_files[@]}
echo "Found $num_files .mbm.csv files to process"
echo ""

# Create backup directory
mkdir -p "$BACKUP_DIR"
echo "Backing up existing summary files to: $BACKUP_DIR"
cp "$RESULTS_DIR"/*.mbm_summary.txt "$BACKUP_DIR"/ 2>/dev/null || echo "No existing summary files to backup"
echo ""

# Process each .mbm.csv file
count=0
success=0
failed=0

for mbm_csv in "${mbm_files[@]}"; do
    count=$((count + 1))
    basename=$(basename "$mbm_csv")
    base="${mbm_csv%.mbm.csv}"
    summary_file="${base}.mbm_summary.txt"
    cache_csv="${base}.cache.csv"
    training_log="${base}.training.log"
    
    echo "[$count/$num_files] Processing: $basename"
    
    # Check if file has data
    line_count=$(wc -l < "$mbm_csv")
    if [ "$line_count" -le 1 ]; then
        echo "  ⚠ SKIP: No data in CSV (only $line_count lines)"
        failed=$((failed + 1))
        continue
    fi
    
    # Show first few lines for debugging
    echo "  First 3 lines of CSV:"
    head -3 "$mbm_csv" | sed 's/^/    /'
    
    # Analyze MBM data
    echo "  Analyzing bandwidth data..."
    
    # Use the last column which should be MBT[MB/s]
    awk -F',' '
    BEGIN {
        peak = 0
        peak_time = ""
        sum = 0
        count = 0
    }
    NR == 1 { next }
    {
        time = $1
        bw = $NF + 0
        
        if (bw > 0) {
            sum += bw
            count++
            if (bw > peak) {
                peak = bw
                peak_time = time
            }
        }
    }
    END {
        if (count > 0) {
            avg = sum / count
            printf "%.2f\n%.6f\n%.2f\n%.6f\n%d\n%s\n", peak, peak/1024, avg, avg/1024, count, peak_time
        } else {
            printf "0\n0\n0\n0\n0\n\n"
            exit 1
        }
    }
    ' "$mbm_csv" 2>/dev/null > /tmp/bw_metrics_$$.txt
    
    # Check if analysis succeeded
    if [ $? -ne 0 ]; then
        echo "  ✗ FAILED to analyze bandwidth data"
        failed=$((failed + 1))
        continue
    fi
    
    # Read the metrics line by line
    {
        read PEAK_MBps
        read PEAK_GBps
        read AVG_MBps
        read AVG_GBps
        read TOTAL_SAMPLES
        read PEAK_TIME
    } < /tmp/bw_metrics_$$.txt
    
    # Validate we got numeric data
    if ! [[ "$PEAK_MBps" =~ ^[0-9.]+$ ]]; then
        echo "  ✗ Invalid bandwidth data: PEAK_MBps='$PEAK_MBps'"
        failed=$((failed + 1))
        continue
    fi
    
    echo "  Peak: ${PEAK_MBps} MB/s (${PEAK_GBps} GB/s), Avg: ${AVG_MBps} MB/s"
    
    # Analyze cache data if available
    echo "  Analyzing cache data..."
    if [ -f "$cache_csv" ]; then
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
                print "L1_MISS_RATE=" l1_miss_rate
            } else {
                print "L1_MISS_RATE=0"
            }
            if (llc_loads > 0) {
                llc_miss_rate = (llc_misses / llc_loads) * 100
                print "LLC_MISS_RATE=" llc_miss_rate
            } else {
                print "LLC_MISS_RATE=0"
            }
            print "L1_LOADS=" l1_loads
            print "L1_MISSES=" l1_misses
            print "LLC_LOADS=" llc_loads
            print "LLC_MISSES=" llc_misses
        }
        ' "$cache_csv" > /tmp/cache_metrics_$$.txt
        
        source /tmp/cache_metrics_$$.txt
        echo "  L1 miss rate: ${L1_MISS_RATE}%"
        echo "  LLC miss rate: ${LLC_MISS_RATE}%"
    else
        echo "  ⚠ Cache CSV not found"
        L1_MISS_RATE=0
        LLC_MISS_RATE=0
        L1_LOADS=0
        L1_MISSES=0
        LLC_LOADS=0
        LLC_MISSES=0
    fi
    
    # Extract training metrics if available
    if [ -f "$training_log" ]; then
        # The format is "The MLP time is 12.331861972808838"
        MLP_TIME=$(grep "The MLP time is" "$training_log" | awk '{print $NF}' | tail -1 || echo "0")
        EMB_TIME=$(grep "The embedding time is" "$training_log" | awk '{print $NF}' | tail -1 || echo "0")
        INTERACT_TIME=$(grep "The interaction time is" "$training_log" | awk '{print $NF}' | tail -1 || echo "0")
        TOTAL_TIME=$(grep "The total time is" "$training_log" | awk '{print $NF}' | tail -1 || echo "0")
        
        echo "  Training times: MLP=${MLP_TIME}s, Emb=${EMB_TIME}s, Interact=${INTERACT_TIME}s, Total=${TOTAL_TIME}s"
    else
        MLP_TIME=0
        EMB_TIME=0
        INTERACT_TIME=0
        TOTAL_TIME=0
    fi
    
    # Generate new summary file
    echo "  Generating new summary file..."
    
    cat > "$summary_file" << EOF
========================================
DLRM Bandwidth Monitoring Summary
========================================
Generated: $(date)
Base: $(basename "$base")

=== Memory Bandwidth Analysis ===
Peak Bandwidth: ${PEAK_MBps} MB/s (${PEAK_GBps} GB/s)
Average Bandwidth: ${AVG_MBps} MB/s (${AVG_GBps} GB/s)
Total Samples: ${TOTAL_SAMPLES}
Peak Time: ${PEAK_TIME}

PEAK_MBps=${PEAK_MBps}
PEAK_GBps=${PEAK_GBps}
AVG_MBps=${AVG_MBps}
AVG_GBps=${AVG_GBps}

=== Cache Miss Rate Analysis ===
L1 Cache:
  Total Loads: ${L1_LOADS}
  Total Misses: ${L1_MISSES}
  Miss Rate: ${L1_MISS_RATE}%

LLC (Last Level Cache):
  Total Loads: ${LLC_LOADS}
  Total Misses: ${LLC_MISSES}
  Miss Rate: ${LLC_MISS_RATE}%

L1_MISS_RATE=${L1_MISS_RATE}
LLC_MISS_RATE=${LLC_MISS_RATE}

=== Training Performance Metrics ===
MLP Time: ${MLP_TIME} seconds
Embedding Time: ${EMB_TIME} seconds
Interaction Time: ${INTERACT_TIME} seconds
Total Time: ${TOTAL_TIME} seconds

=== File Paths ===
MBM_CSV=$mbm_csv
CACHE_CSV=$cache_csv
TRAINING_LOG=$training_log

========================================
EOF
    
    echo "  ✓ Updated: $(basename "$summary_file")"
    success=$((success + 1))
    echo ""
    
    # Cleanup temp files
    rm -f /tmp/bw_metrics_$$.txt /tmp/cache_metrics_$$.txt
done

echo "=========================================="
echo "Processing Complete"
echo "=========================================="
echo "Total files: $num_files"
echo "Successfully updated: $success"
echo "Failed: $failed"
echo ""
echo "Backup location: $BACKUP_DIR"
echo ""

if [ $success -gt 0 ]; then
    echo "✓ Summary files have been updated!"
    echo ""
    echo "You can now run analysis:"
    echo "  bash analyze_results.sh $RESULTS_DIR"
else
    echo "✗ No files were successfully updated"
    echo "  Check the error messages above"
fi

echo "=========================================="
