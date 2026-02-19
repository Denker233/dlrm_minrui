#!/usr/bin/env python3
import sys

with open('dlrm_data_pytorch.py', 'r') as f:
    lines = f.readlines()

# Find and fix line 329
for i, line in enumerate(lines):
    if 'X_cat = torch.tensor(transposed_data[1], dtype=torch.long)' in line:
        # Replace with a more robust version
        indent = len(line) - len(line.lstrip())
        lines[i] = ' ' * indent + 'X_cat = torch.tensor(np.array(transposed_data[1], dtype=np.int64), dtype=torch.long)\n'
        print(f"✓ Patched line {i+1}")
        break

with open('dlrm_data_pytorch.py', 'w') as f:
    f.writelines(lines)

print("✓ File patched successfully")
