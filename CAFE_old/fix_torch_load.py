#!/usr/bin/env python3
import re

print("Fixing PyTorch torch.load weights_only issue...")

with open('dlrm_s_pytorch.py', 'r') as f:
    content = f.read()

# Find all torch.load calls and add weights_only=False
# Pattern: torch.load(..., map_location=...)
old_pattern = r'torch\.load\(\s*args\.load_model,\s*map_location='
new_pattern = r'torch.load(args.load_model, weights_only=False, map_location='

if 'weights_only=False' in content:
    print("✓ Already patched")
else:
    content = re.sub(old_pattern, new_pattern, content)
    
    with open('dlrm_s_pytorch.py', 'w') as f:
        f.write(content)
    
    print("✓ Added weights_only=False to torch.load")

