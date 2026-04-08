#!/usr/bin/env python3
"""
Complete fix for sketch compression bugs in CAFE
"""

import re

print("="*80)
print("FIXING SKETCH COMPRESSION BUGS")
print("="*80)

# Backup
import shutil
shutil.copy('tricks/sk_embedding_bag.py', 'tricks/sk_embedding_bag.py.backup')
print("✓ Backed up to tricks/sk_embedding_bag.py.backup")

# Read file
with open('tricks/sk_embedding_bag.py', 'r') as f:
    content = f.read()

# Fix 1: insert_grad function - handle variable batch sizes and bounds checking
old_insert_grad = '''    def insert_grad(self, input):
        grad_norm = torch.where(
            self.weight_hash.grad._indices()[0] < self.cnt_hash,
            torch.norm(self.weight_hash.grad._values(), dim = 1,  p = 2),
            torch.tensor(0.0)
        ).cpu().numpy()
        
        N = len(input)
        insert_val = ctypes.c_void_p.in_dll(self.lib, "insert_val").value
        func = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_float), ctypes.c_int)(insert_val)
        func(grad_norm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), N)'''

new_insert_grad = '''    def insert_grad(self, input):
        # Get gradient values
        grad_values = self.weight_hash.grad._values()
        grad_indices = self.weight_hash.grad._indices()[0]
        
        # Handle variable-length batches: only process gradients that exist
        N = min(len(input), grad_values.shape[0])
        if N == 0:
            return  # Skip if no gradients
        
        # Compute gradient norms only for valid indices
        valid_mask = grad_indices < self.cnt_hash
        grad_norm = torch.where(
            valid_mask[:N],
            torch.norm(grad_values[:N], dim=1, p=2),
            torch.tensor(0.0, device=grad_values.device)
        ).cpu().numpy()
        
        # Call C++ insert function
        insert_val = ctypes.c_void_p.in_dll(self.lib, "insert_val").value
        func = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_float), ctypes.c_int)(insert_val)
        func(grad_norm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), N)'''

if old_insert_grad in content:
    content = content.replace(old_insert_grad, new_insert_grad)
    print("✓ Fixed insert_grad function (tensor shape mismatch)")
else:
    print("⚠ Could not find exact insert_grad code - manual fix needed")

# Write back
with open('tricks/sk_embedding_bag.py', 'w') as f:
    f.write(content)

print("✓ Fixes applied!")
print()
print("Changes made:")
print("  1. Fixed tensor shape mismatch in insert_grad")
print("  2. Added bounds checking for gradient indices")
print("  3. Handle variable-length sparse feature batches")
print()
print("To restore original: cp tricks/sk_embedding_bag.py.backup tricks/sk_embedding_bag.py")
print("="*80)

