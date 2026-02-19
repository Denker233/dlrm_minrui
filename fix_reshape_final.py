#!/usr/bin/env python3

with open('dlrm_s_pytorch.py', 'r') as f:
    content = f.read()

# Replace the entire reshape logic with corrected version
old_section = '''            # Reshape to fit within limits
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

new_section = '''            # Reshape to fit within limits
            total_pixels = width * height
            
            # For very small tables, ensure we meet minimum dimensions
            min_required_pixels = MIN_WIDTH * MIN_HEIGHT
            
            if total_pixels < min_required_pixels:
                # Table is too small - pad to minimum size
                width = MIN_WIDTH
                height = MIN_HEIGHT
            elif height > MAX_DIM:
                # Table is too tall
                width = (total_pixels + MAX_DIM - 1) // MAX_DIM
                height = MAX_DIM
                if width < MIN_WIDTH:
                    width = MIN_WIDTH
                    height = (total_pixels + width - 1) // width
            elif width < MIN_WIDTH:
                # Table is too narrow
                width = MIN_WIDTH
                height = (total_pixels + width - 1) // width
                if height < MIN_HEIGHT:
                    height = MIN_HEIGHT
            elif height < MIN_HEIGHT:
                # Table is too short
                height = MIN_HEIGHT
                width = (total_pixels + height - 1) // height
                if width < MIN_WIDTH:
                    width = MIN_WIDTH
                    height = (total_pixels + width - 1) // width
                    if height < MIN_HEIGHT:
                        # Still too small, use minimums
                        width = MIN_WIDTH
                        height = MIN_HEIGHT'''

content = content.replace(old_section, new_section)

with open('dlrm_s_pytorch.py', 'w') as f:
    f.write(content)

print("✓ Fixed reshape logic for tiny tables")
print("  - Tables smaller than 64x64 will be padded to 64x64")
