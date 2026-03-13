#!/bin/bash

# Create summary file
SUMMARY_FILE="perf_cache_summary_para_128.log"
echo "Script,Threads,L1_loads,L1_misses,L1_miss_rate,LLC_loads,LLC_misses,LLC_miss_rate,Cache_refs,Cache_misses,Cache_miss_rate" > $SUMMARY_FILE

# Events to collect
EVENTS="L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses,cache-references,cache-misses"

# Loop through each script
for script in para_run_dlrm_b128_t*.sh; do
    echo "Running $script with perf..."
    tag=$(basename "$script" .sh)
    threads=$(echo "$tag" | grep -oP '(?<=_t)\d+')

    TEMPFILE=$(mktemp)

    # Run perf stat
    perf stat -e $EVENTS -o $TEMPFILE ./$script

    # Extract values
    L1_loads=$(grep "L1-dcache-loads" $TEMPFILE | awk '{gsub(",", "", $1); print $1}')
    L1_misses=$(grep "L1-dcache-load-misses" $TEMPFILE | awk '{gsub(",", "", $1); print $1}')
    LLC_loads=$(grep "LLC-loads" $TEMPFILE | awk '{gsub(",", "", $1); print $1}')
    LLC_misses=$(grep "LLC-load-misses" $TEMPFILE | awk '{gsub(",", "", $1); print $1}')
    cache_refs=$(grep "cache-references" $TEMPFILE | awk '{gsub(",", "", $1); print $1}')
    cache_misses=$(grep "cache-misses" $TEMPFILE | awk '{gsub(",", "", $1); print $1}')

    # Compute rates
    L1_rate=$(awk "BEGIN {if ($L1_loads == 0) print 0; else printf \"%.6f\", $L1_misses / $L1_loads}")
    LLC_rate=$(awk "BEGIN {if ($LLC_loads == 0) print 0; else printf \"%.6f\", $LLC_misses / $LLC_loads}")
    total_rate=$(awk "BEGIN {if ($cache_refs == 0) print 0; else printf \"%.6f\", $cache_misses / $cache_refs}")

    # Append parsed summary row
    echo "$tag,$threads,$L1_loads,$L1_misses,$L1_rate,$LLC_loads,$LLC_misses,$LLC_rate,$cache_refs,$cache_misses,$total_rate" >> $SUMMARY_FILE

    # Append raw perf output below the summary
    echo -e "\n==== Raw perf output for $tag ====\n" >> $SUMMARY_FILE
    cat $TEMPFILE >> $SUMMARY_FILE

    # Clean up
    rm $TEMPFILE
done

echo "✅ Perf stats and parsed summary saved in: $SUMMARY_FILE"
