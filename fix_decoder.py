with open('dlrm_s_pytorch.py', 'r') as f:
    lines = f.readlines()

# Find the _decode_qsv function and add codec mapping
new_lines = []
for i, line in enumerate(lines):
    new_lines.append(line)
    
    # Add codec mapping right after the _decode_qsv function definition
    if 'def _decode_qsv(self, video_file, raw_file, width, height):' in line:
        indent = ' ' * (len(line) - len(line.lstrip()) + 4)
        # Insert codec mapping after the docstring (if exists)
        # We'll add it after finding where to insert
        insert_pos = i + 1
        
        # Check if next line is a docstring
        if i + 1 < len(lines) and '"""' in lines[i + 1]:
            # Skip to end of docstring
            for j in range(i + 2, len(lines)):
                if '"""' in lines[j]:
                    insert_pos = j + 1
                    break
        
        # Now we need to insert the mapping
        # Let's add it as the first line of actual code
        mapping_code = f'{indent}# Map encoder name to decoder name\n'
        mapping_code += f'{indent}decoder_map = {{\n'
        mapping_code += f'{indent}    "hevc_qsv": "hevc",\n'
        mapping_code += f'{indent}    "h264_qsv": "h264",\n'
        mapping_code += f'{indent}    "libx265": "hevc",\n'
        mapping_code += f'{indent}    "libx264": "h264",\n'
        mapping_code += f'{indent}}}\n'
        mapping_code += f'{indent}decoder = decoder_map.get(self.codec, self.codec)\n'
        mapping_code += f'{indent}\n'

# Actually, let's do a simpler approach - just replace the codec usage in the decode function
with open('dlrm_s_pytorch.py', 'r') as f:
    content = f.read()

# Find and replace in the _decode_qsv function
# Replace: "-c:v", self.codec,
# With: "-c:v", decoder_name,

# Add codec mapping at the start of _decode_qsv
old_pattern = 'def _decode_qsv(self, video_file, raw_file, width, height):\n        """Decode compressed video back to raw pixels"""'
new_pattern = '''def _decode_qsv(self, video_file, raw_file, width, height):
        """Decode compressed video back to raw pixels"""
        # Map encoder names to decoder names
        decoder_map = {
            "hevc_qsv": "hevc",
            "h264_qsv": "h264", 
            "libx265": "hevc",
            "libx264": "h264",
        }
        decoder = decoder_map.get(self.codec, "hevc")'''

if old_pattern in content:
    content = content.replace(old_pattern, new_pattern)
    print("✓ Added decoder mapping")
else:
    print("Pattern not found, trying alternative...")
    # Try simpler pattern
    import re
    pattern = r'(def _decode_qsv\(self, video_file, raw_file, width, height\):.*?""".*?""")'
    replacement = r'\1\n        decoder_map = {"hevc_qsv": "hevc", "h264_qsv": "h264", "libx265": "hevc", "libx264": "h264"}\n        decoder = decoder_map.get(self.codec, "hevc")'
    content = re.sub(pattern, replacement, content, flags=re.DOTALL)

# Also replace self.codec with decoder in the ffmpeg command
content = content.replace('"-c:v", self.codec,', '"-c:v", decoder,')

with open('dlrm_s_pytorch.py', 'w') as f:
    f.write(content)

print("✓ Fixed decoder mapping in dlrm_s_pytorch.py")
