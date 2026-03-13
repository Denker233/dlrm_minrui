with open('compress_with_details.py', 'r') as f:
    content = f.read()

# Replace the codec choices line
old_line = "choices=['h264_qsv', 'hevc_qsv']"
new_line = "choices=['h264_qsv', 'hevc_qsv', 'libx264', 'libx265']"

if old_line in content:
    content = content.replace(old_line, new_line)
    print("✓ Updated codec choices")
else:
    print("✗ Could not find line to replace")
    print("Searching for variations...")
    import re
    matches = re.findall(r"choices=\[.*?qsv.*?\]", content)
    for match in matches:
        print(f"  Found: {match}")

# Also update default if it exists
content = content.replace("default='hevc_qsv'", "default='libx265'")

with open('compress_with_details.py', 'w') as f:
    f.write(content)

print("✓ File updated")
