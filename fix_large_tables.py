with open('dlrm_s_pytorch.py', 'r') as f:
    content = f.read()

# Find the _encode_qsv function and add reshaping logic
old_encode = '''def _encode_qsv(self, raw_file, video_file, width, height):
        """Encode raw pixel data to video using QSV"""
        import subprocess'''

new_encode = '''def _encode_qsv(self, raw_file, video_file, width, height):
        """Encode raw pixel data to video using QSV"""
        import subprocess
        
        # Video codecs have max dimension limits (typically 16384 or 65536)
        # Reshape if height > 16384
        MAX_DIM = 16384
        original_width = width
        original_height = height
        
        if height > MAX_DIM:
            # Reshape to more square dimensions
            total_pixels = width * height
            # Find dimensions that fit within MAX_DIM
            new_height = MAX_DIM
            new_width = (total_pixels + new_height - 1) // new_height  # Ceiling division
            
            # Pad if necessary to make it divisible
            padded_pixels = new_width * new_height
            if padded_pixels > total_pixels:
                # Need to pad the raw file
                import numpy as np
                data = np.fromfile(raw_file, dtype=np.uint8)
                padded_data = np.zeros(padded_pixels, dtype=np.uint8)
                padded_data[:len(data)] = data
                padded_data.tofile(raw_file)
            
            width = new_width
            height = new_height
            print(f"      Reshaped from {original_width}x{original_height} to {width}x{height}")'''

if old_encode in content:
    content = content.replace(old_encode, new_encode)
    print("✓ Added reshaping logic to _encode_qsv")
else:
    print("✗ Pattern not found")

# Also need to fix _decode_qsv to handle reshaped data
old_decode_start = '''def _decode_qsv(self, input_file, output_file):
        """Decode using QSV"""'''

new_decode_start = '''def _decode_qsv(self, input_file, output_file, expected_size=None):
        """Decode using QSV"""'''

content = content.replace(old_decode_start, new_decode_start)

# Update decompress_table to pass expected size
old_decompress_call = '''self._decode_qsv(video_file, raw_file)'''
new_decompress_call = '''expected_size = shape[0] * shape[1]
        self._decode_qsv(video_file, raw_file, expected_size)'''

content = content.replace(old_decompress_call, new_decompress_call)

# Add trimming logic at end of _decode_qsv
decode_pattern = '''result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"QSV decoding failed: {result.stderr}")'''

decode_replacement = '''result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"QSV decoding failed: {result.stderr}")
        
        # Trim to expected size if provided (handles reshaped tables)
        if expected_size is not None:
            import numpy as np
            data = np.fromfile(output_file, dtype=np.uint8)
            if len(data) > expected_size:
                data = data[:expected_size]
                data.tofile(output_file)'''

content = content.replace(decode_pattern, decode_replacement)

with open('dlrm_s_pytorch.py', 'w') as f:
    f.write(content)

print("✓ Fixed handling of large tables in dlrm_s_pytorch.py")
