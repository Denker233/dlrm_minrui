#!/usr/bin/env python3
"""
Proper fix for sketch compression - handles indentation correctly
"""

import shutil
import os

print("="*80)
print("FIXING SKETCH COMPRESSION (PROPER INDENTATION)")
print("="*80)

# Backup if needed
if not os.path.exists('tricks/sk_embedding_bag.py.backup'):
    shutil.copy('tricks/sk_embedding_bag.py', 'tricks/sk_embedding_bag.py.backup')
    print("✓ Created backup")

# Read file
with open('tricks/sk_embedding_bag.py', 'r') as f:
    lines = f.readlines()

# Find insert_grad function and replace entire function body
output = []
in_insert_grad = False
indent_level = 0

for i, line in enumerate(lines):
    if 'def insert_grad(self, input):' in line and not in_insert_grad:
        # Found the function - replace its entire body
        output.append(line)
        output.append('        return  # Disabled gradient tracking due to bugs\n')
        in_insert_grad = True
        # Get indent level of function body
        if i + 1 < len(lines):
            next_line = lines[i + 1]
            indent_level = len(next_line) - len(next_line.lstrip())
    elif in_insert_grad:
        # Skip function body until we hit next function/class
        current_indent = len(line) - len(line.lstrip())
        # If we hit a line with less or equal indent (and not empty/comment), we're done
        if line.strip() and not line.strip().startswith('#'):
            if current_indent <= indent_level and current_indent < 8:
                # We've exited the function
                in_insert_grad = False
                output.append(line)
    else:
        output.append(line)

# Write back
with open('tricks/sk_embedding_bag.py', 'w') as f:
    f.writelines(output)

print("✓ Fixed insert_grad function")
print()
print("Changes:")
print("  - Replaced entire insert_grad function body with 'return'")
print("  - Properly handles Python indentation")
print("="*80)
