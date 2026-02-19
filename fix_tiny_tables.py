#!/usr/bin/env python3

with open('dlrm_s_pytorch.py', 'r') as f:
    content = f.read()

# Fix: Add minimum height requirement
old_code = '''        # Video codecs have dimension limits
        MAX_DIM = 16384
        MIN_DIM = 64  # libx265 requires minimum width
        original_width, original_height = width, height
        
        if height > MAX_DIM or width < MIN_DIM:'''

new_code = '''        # Video codecs have dimension limits
        MAX_DIM = 16384
        MIN_WIDTH = 64   # libx265 minimum width
        MIN_HEIGHT = 64  # libx265 minimum height
        original_width, original_height = width, height
        
        if height > MAX_DIM or width < MIN_WIDTH or height < MIN_HEIGHT:'''

content = content.replace(old_code, new_code)

# Fix reshape logic for all cases
old_reshape = '''            # Reshape to fit within limits
            total_pixels = width * height
            
            if height > MAX_DIM:
                # Table is too tall
                height = MAX_DIM
                width = (total_pixels + height - 1) // height
            elif width < MIN_DIM:
                # Table is too narrow
                width = MIN_DIM
                height = (total_pixels + width - 1) // width'''

new_reshape = '''            # Reshape to fit within limits
            total_pixels = width * height
            
            # Calculate target dimensions
            if height > MAX_DIM:
                # Table is too tall
                target_height = MAX_DIM
                target_width = (total_pixels + target_height - 1) // target_height
            elif width < MIN_WIDTH:
                # Table is too narrow
                target_width = MIN_WIDTH
                target_height = (total_pixels + target_width - 1) // target_width
            elif height < MIN_HEIGHT:
                # Table is too short
                target_height = MIN_HEIGHT
                target_width = (total_pixels + target_height - 1) // target_height
            
            # Ensure both dimensions meet minimums
            if target_width < MIN_WIDTH:
                target_width = MIN_WIDTH
                target_height = (total_pixels + target_width - 1) // target_width
            if target_height < MIN_HEIGHT:
                target_height = MIN_HEIGHT
                target_width = (total_pixels + target_height - 1) // target_width
            
            width, height = target_width, target_height'''

content = content.replace(old_reshape, new_reshape)

with open('dlrm_s_pytorch.py', 'w') as f:
    f.write(content)

print("✓ Fixed tiny table handling")
print("  - Added MIN_HEIGHT = 64")
print("  - Improved reshape logic to ensure both dimensions meet minimums")
