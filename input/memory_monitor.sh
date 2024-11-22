#!/bin/bash

# PID=3194205  # Replace <your_pid_here> with the actual PID
# MAX_RSS=0
# echo "PID is: $PID  runing 11GB raw data" >> memory_consumption.log
# while ps -p $PID > /dev/null; do
#     RSS=$(ps -o rss= -p $PID | awk '{print $1}')
#     if [ "$RSS" -gt "$MAX_RSS" ]; then
#         MAX_RSS=$RSS
#     fi
#     sleep 1  # Adjust the interval as needed
# done

# echo "Maximum memory usage: $MAX_RSS KB runing 11GB raw data" >> memory_consumption.log


#!/bin/bash

# max_mem_total=0

# # Function to calculate CPU usage
# get_cpu_usage() {
#     # CPUs associated with NUMA 1 (example: 10-19, 30-39)
#     numa1_cpus=(10 11 12 13 14 15 16 17 18 19 30 31 32 33 34 35 36 37 38 39)

#     total=0
#     idle_total=0

#     # Loop through each CPU in NUMA 1
#     for cpu in "${numa1_cpus[@]}"; do
#         # Read the CPU stats for the current CPU
#         cpu_line=$(grep -m 1 "cpu$cpu " /proc/stat)

#         # Extract values for user, nice, system, idle, iowait, irq, softirq
#         read -r _ user nice system idle iowait irq softirq <<<$(echo "$cpu_line" | awk '{print $2, $3, $4, $5, $6, $7, $8}')

#         # Accumulate total and idle time
#         total=$((total + user + nice + system + idle + iowait + irq + softirq))
#         idle_total=$((idle_total + idle + iowait))
#     done

#     # Return aggregated total and idle values
#     echo "$total $idle_total"
# }

# # Initialize previous CPU stats
# read -r prev_total prev_idle < <(get_cpu_usage)
# sleep 10



# while true; do
#     # Extract memory usage for NUMA nodes 0 and 1
#     # mem_node0=$(grep "Node 0 MemUsed:" /sys/devices/system/node/node0/meminfo | cut -d':' -f2 | tr -d ' kB' | awk '{printf "%.2f", $1/1024/1024}')
#     mem_node1=$(grep "Node 1 MemUsed:" /sys/devices/system/node/node1/meminfo | cut -d':' -f2 | tr -d ' kB' | awk '{printf "%.2f", $1/1024/1024}')

#     # total_mem=$(echo "$mem_node0 + $mem_node1" | bc)
#     total_mem=$(echo "$mem_node1" | bc)
#     # echo "New Maximum Used Memory Usage: $max_mem_total GB" >> numa_memory_consumption.log
#     # Update maximum memory usage if needed
#     if (( $(echo "$total_mem > $max_mem_total" | bc -l) )); then
#         max_mem_total=$total_mem
#         # echo "New Maximum Used Memory Usage: $max_mem_total GB" >> numa_memory_consumption.log
#     fi

#     # Calculate CPU usage
#     read -r current_total current_idle < <(get_cpu_usage)
#     # total_diff=$((current_total - prev_total))
#     # idle_diff=$((current_idle - prev_idle))
#     # echo "Debug: current_total=$current_total, current_idle=$current_idle" >> numa_memory_consumption.log

#     cpu_usage=$(awk -v idle="$current_idle" -v total="$current_total -v " 'BEGIN { if (total > 0) print (total - idle)/total; else print 0 }')



#     # Log memory and CPU usage
#     echo "$(date): Memory Usage: $total_mem GB, CPU Usage: $cpu_usage%" >> ./input/numa_memory_consumption.log

#     # Update previous CPU stats
#     # prev_total=$current_total
#     # prev_idle=$current_idle

#     sleep 60  # Check every 60 seconds
# done


#!/bin/bash

# Check if PID is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <PID>"
    exit 1
fi

PID=$1
LOG_FILE="numa_memory_consumption_10GB.log"

# Check if the process exists
if ! ps -p $PID > /dev/null; then
    echo "Error: Process with PID $PID not found."
    exit 1
fi

echo "Monitoring CPU, physical memory (RSS), and virtual memory (VIRT) usage for PID $PID."
echo "Output will be logged to $LOG_FILE."
echo "Timestamp, %CPU, RSS (GB), VIRT (GB), CMD" > $LOG_FILE

# Monitor the process until it exits
while ps -p $PID > /dev/null; do
    # Get CPU usage, RSS (Resident Set Size), and VIRT (Virtual Memory Size) in kilobytes
    usage=$(ps -p $PID -o %cpu,rss,vsz,cmd --no-headers)
    mem_node1=$(grep "Node 1 MemUsed:" /sys/devices/system/node/node1/meminfo | cut -d':' -f2 | tr -d ' kB' | awk '{printf "%.2f", $1/1024/1024}')
    total_mem=$(echo "$mem_node1" | bc)
    # Extract fields
    cpu=$(echo $usage | awk '{print $1}')
    rss_kb=$(echo $usage | awk '{print $2}')
    virt_kb=$(echo $usage | awk '{print $3}')
    cmd=$(echo $usage | awk '{$1=""; $2=""; $3=""; $4=""; print $0}' | sed 's/^ *//')

    # Convert RSS and VIRT from KB to GB
    rss_gb=$(awk "BEGIN {print $rss_kb / 1024 / 1024}")
    virt_gb=$(awk "BEGIN {print $virt_kb / 1024 / 1024}")

    # Get the current timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    # Log the usage
    echo "$timestamp, $cpu, $rss_gb, $virt_gb,Total NUMA 1 Memory Usage: $total_mem GB, $cmd" >> $LOG_FILE

    # Wait for 1 minute before the next check
    sleep 60
done

echo "Process $PID has finished. Stopped monitoring."
