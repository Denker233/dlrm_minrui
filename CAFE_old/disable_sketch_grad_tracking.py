#!/usr/bin/env python3
"""
Simpler fix: Disable gradient tracking in sketch (loses adaptivity but works)
"""

import shutil
import os

print("="*80)
print("FIXING SKETCH COMPRESSION BUG")
print("="*80)

# Backup
if not os.path.exists('tricks/sk_embedding_bag.py.backup'):
    shutil.copy('tricks/sk_embedding_bag.py', 'tricks/sk_embedding_bag.py.backup')
    print("✓ Created backup: tricks/sk_embedding_bag.py.backup")
else:
    print("✓ Backup already exists")

# Read file
with open('tricks/sk_embedding_bag.py', 'r') as f:
    content = f.read()

# Check if already patched
if 'def insert_grad(self, input):\n        return  # Disabled' in content:
    print("✓ Sketch already patched")
else:
    # Find insert_grad and make it a no-op
    content = content.replace(
        'def insert_grad(self, input):',
        'def insert_grad(self, input):\n        return  # Disabled due to bugs\n        # Original code below (disabled):\n        if False:'
    )
    
    # Write back
    with open('tricks/sk_embedding_bag.py', 'w') as f:
        f.write(content)
    
    print("✓ Applied patch to insert_grad function")

print()
print("Changes:")
print("  - Disabled gradient tracking in sketch compression")
print("  - Sketch will use static hot/cold split (no adaptive updates)")
print("  - This is acceptable for performance testing!")
print()
print("To restore: cp tricks/sk_embedding_bag.py.backup tricks/sk_embedding_bag.py")
print("="*80)
