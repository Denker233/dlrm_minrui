with open('dlrm_s_pytorch.py', 'r') as f:
    content = f.read()

# Find the _decode_qsv function and replace the codec line
old_line = "'-c:v', self.codec.replace('_qsv', ''),"
new_line = """'-c:v', {'hevc_qsv': 'hevc', 'h264_qsv': 'h264', 'libx265': 'hevc', 'libx264': 'h264'}.get(self.codec, 'hevc'),"""

if old_line in content:
    content = content.replace(old_line, new_line)
    print("✓ Fixed decoder codec mapping")
else:
    print("✗ Could not find the exact line")
    # Try variations
    if "self.codec.replace('_qsv', '')" in content:
        print("Found the pattern, trying to replace...")
        import re
        # Replace any occurrence of self.codec.replace('_qsv', '') in decode function
        pattern = r"self\.codec\.replace\('_qsv', ''\)"
        replacement = "{'hevc_qsv': 'hevc', 'h264_qsv': 'h264', 'libx265': 'hevc', 'libx264': 'h264'}.get(self.codec, 'hevc')"
        content = re.sub(pattern, replacement, content)
        print("✓ Replaced with regex")

with open('dlrm_s_pytorch.py', 'w') as f:
    f.write(content)

print("✓ Decoder fixed in dlrm_s_pytorch.py")
