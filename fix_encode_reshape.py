#!/usr/bin/env python3

with open('dlrm_s_pytorch.py', 'r') as f:
    content = f.read()

# Find and replace the _encode_qsv function
old_function = '''    def _encode_qsv(self, input_file, output_file, width, height):
        """Encode using QSV"""
        cmd = ['''

new_function = '''    def _encode_qsv(self, input_file, output_file, width, height):
        """Encode using QSV with dimension reshaping for large tables"""
        import numpy as np
        
        # Video codecs have max dimension limits (typically 16384)
        MAX_DIM = 16384
        original_width, original_height = width, height
        
        if height > MAX_DIM:
            # Reshape to fit within limits
            total_pixels = width * height
            height = MAX_DIM
            width = (total_pixels + height - 1) // height  # Ceiling division
            
            # Pad the data if needed
            padded_pixels = width * height
            if padded_pixels > total_pixels:
                data = np.fromfile(input_file, dtype=np.uint8)
                padded_data = np.zeros(padded_pixels, dtype=np.uint8)
                padded_data[:len(data)] = data
                padded_data.tofile(input_file)
            
            print(f"      Reshaped from {original_width}x{original_height} to {width}x{height}")
        
        cmd = ['''

if old_function in content:
    content = content.replace(old_function, new_function)
    print("✓ Added reshaping logic to _encode_qsv")
else:
    print("✗ Exact pattern not found")
    print("Trying alternative approach...")

# Save
with open('dlrm_s_pytorch.py', 'w') as f:
    f.write(content)

print("✓ File updated")
