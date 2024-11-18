#!/bin/bash

PID=3194205  # Replace <your_pid_here> with the actual PID
MAX_RSS=0
echo "PID is: $PID  runing 11GB raw data" >> memory_consumption.log
while ps -p $PID > /dev/null; do
    RSS=$(ps -o rss= -p $PID | awk '{print $1}')
    if [ "$RSS" -gt "$MAX_RSS" ]; then
        MAX_RSS=$RSS
    fi
    sleep 1  # Adjust the interval as needed
done

echo "Maximum memory usage: $MAX_RSS KB runing 11GB raw data" >> memory_consumption.log
