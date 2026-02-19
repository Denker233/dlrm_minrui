#!/bin/bash

echo "Checking for recent DQRM comparison errors..."
echo ""

# Check if training log exists
if [ -f "./models/dqrm_comparison/training_log.txt" ]; then
    echo "Found training log! Last 50 lines:"
    echo "=================================="
    tail -50 ./models/dqrm_comparison/training_log.txt
    echo ""
else
    echo "No training log found at ./models/dqrm_comparison/training_log.txt"
    echo ""
fi

# Check for any recent error messages in the directory
if [ -d "./models/dqrm_comparison" ]; then
    echo "Files in dqrm_comparison directory:"
    ls -lh ./models/dqrm_comparison/
    echo ""
fi

# Check for any Python error traces
echo "Checking for Python errors in current directory:"
grep -r "Traceback\|Error\|Exception" ./models/dqrm_comparison/ 2>/dev/null | head -20

# Check if the scripts exist
echo ""
echo "Checking required scripts:"
[ -f "dlrm_s_pytorch.py" ] && echo "✓ dlrm_s_pytorch.py exists" || echo "✗ dlrm_s_pytorch.py MISSING"
[ -f "test_quantization_accuracy.py" ] && echo "✓ test_quantization_accuracy.py exists" || echo "✗ test_quantization_accuracy.py MISSING"
[ -f "compress_with_details.py" ] && echo "✓ compress_with_details.py exists" || echo "✗ compress_with_details.py MISSING"
[ -f "decompress_model.py" ] && echo "✓ decompress_model.py exists" || echo "✗ decompress_model.py MISSING"

