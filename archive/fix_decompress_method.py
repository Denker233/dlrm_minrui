#!/usr/bin/env python3

with open('dlrm_s_pytorch.py', 'r') as f:
    content = f.read()

# Find and replace the broken decompress_table method
old_method = '''    def decompress_table(self, compressed_data, metadata):
        """Decompress an embedding table"""
        num_emb, emb_dim = metadata['shape']
        
        # Decode from QSV
        with tempfile.TemporaryDirectory() as tmpdir:
            video_file = os.path.join(tmpdir, 'compressed.mp4')
            raw_file = os.path.join(tmpdir, 'decoded.raw')
            
            # Write compressed
            with open(video_file, 'wb') as f:
                f.write(compressed_data)
            
            # Decode
            expected_size = shape[0] * shape[1]
        self._decode_qsv(video_file, raw_file, expected_size)
            
            # Read raw
            import numpy as np
            pixels_uint8 = np.fromfile(raw_file, dtype=np.uint8)
            pixels_uint8 = torch.from_numpy(pixels_uint8).reshape(num_emb, emb_dim)
        
        # Dequantize
        dequantize_fn = self.quantizers[metadata['quantization']][1]
        weights = dequantize_fn(pixels_uint8, metadata['quant_params'])
        
        return weights'''

new_method = '''    def decompress_table(self, compressed_data, metadata):
        """Decompress an embedding table"""
        import numpy as np
        
        num_emb, emb_dim = metadata['shape']
        
        # Decode from QSV
        with tempfile.TemporaryDirectory() as tmpdir:
            video_file = os.path.join(tmpdir, 'compressed.mp4')
            raw_file = os.path.join(tmpdir, 'decoded.raw')
            
            # Write compressed
            with open(video_file, 'wb') as f:
                f.write(compressed_data)
            
            # Decode
            expected_size = num_emb * emb_dim
            self._decode_qsv(video_file, raw_file, expected_size)
            
            # Read raw
            pixels_uint8 = np.fromfile(raw_file, dtype=np.uint8)
            pixels_uint8 = torch.from_numpy(pixels_uint8[:expected_size]).reshape(num_emb, emb_dim)
        
        # Dequantize
        dequantize_fn = self.quantizers[metadata['quantization']][1]
        weights = dequantize_fn(pixels_uint8, metadata['quant_params'])
        
        return weights'''

if old_method in content:
    content = content.replace(old_method, new_method)
    print("✓ Fixed decompress_table method")
else:
    print("✗ Exact pattern not found, trying line-by-line fix")
    # Alternative: just fix the specific bad lines
    content = content.replace(
        "            expected_size = shape[0] * shape[1]\n        self._decode_qsv(video_file, raw_file, expected_size)",
        "            expected_size = num_emb * emb_dim\n            self._decode_qsv(video_file, raw_file, expected_size)"
    )
    # Remove duplicate numpy import if it's wrongly placed
    lines = content.split('\n')
    fixed_lines = []
    skip_next_numpy = False
    for i, line in enumerate(lines):
        if '        import numpy as np' in line and i > 100:  # In the class section
            if any('import numpy as np' in l for l in fixed_lines[-20:]):
                skip_next_numpy = True
                continue
        fixed_lines.append(line)
    content = '\n'.join(fixed_lines)
    print("✓ Applied alternative fixes")

with open('dlrm_s_pytorch.py', 'w') as f:
    f.write(content)

print("✓ Fixed dlrm_s_pytorch.py")
