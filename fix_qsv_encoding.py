#!/usr/bin/env python3
"""
Fix QSV encoding to properly initialize hardware and handle small dimensions
"""

with open('dlrm_s_pytorch.py', 'r') as f:
    lines = f.readlines()

# Find the _encode_qsv function and replace it
in_function = False
function_indent = None
new_lines = []
skip_until_next_def = False

for i, line in enumerate(lines):
    if 'def _encode_qsv(self, raw_file, video_file, width, height):' in line:
        in_function = True
        function_indent = len(line) - len(line.lstrip())
        
        # Insert the new fixed function
        indent = ' ' * function_indent
        new_lines.append(line)
        new_lines.append(f'{indent}    """Encode raw pixel data using QSV with proper device init"""\n')
        new_lines.append(f'{indent}    import subprocess\n')
        new_lines.append(f'{indent}    \n')
        new_lines.append(f'{indent}    # QSV requires minimum width (usually 64), pad if needed\n')
        new_lines.append(f'{indent}    min_width = 64\n')
        new_lines.append(f'{indent}    if width < min_width:\n')
        new_lines.append(f'{indent}        pad_width = min_width\n')
        new_lines.append(f'{indent}    else:\n')
        new_lines.append(f'{indent}        pad_width = width\n')
        new_lines.append(f'{indent}    \n')
        new_lines.append(f'{indent}    cmd = [\n')
        new_lines.append(f'{indent}        "ffmpeg", "-y",\n')
        new_lines.append(f'{indent}        "-init_hw_device", "qsv=hw",  # Initialize QSV device\n')
        new_lines.append(f'{indent}        "-filter_hw_device", "hw",\n')
        new_lines.append(f'{indent}        "-f", "rawvideo",\n')
        new_lines.append(f'{indent}        "-pix_fmt", "gray",\n')
        new_lines.append(f'{indent}        "-s", f"{{width}}x{{height}}",\n')
        new_lines.append(f'{indent}        "-i", raw_file,\n')
        new_lines.append(f'{indent}        "-vf", f"pad={{pad_width}}:{{height}}:0:0:gray,hwupload=extra_hw_frames=64",\n')
        new_lines.append(f'{indent}        "-c:v", self.codec,\n')
        new_lines.append(f'{indent}        "-global_quality", str(self.quality),\n')
        new_lines.append(f'{indent}        "-look_ahead", "0",  # Disable look-ahead for speed\n')
        new_lines.append(f'{indent}        video_file\n')
        new_lines.append(f'{indent}    ]\n')
        new_lines.append(f'{indent}    \n')
        new_lines.append(f'{indent}    result = subprocess.run(cmd, capture_output=True, text=True)\n')
        new_lines.append(f'{indent}    if result.returncode != 0:\n')
        new_lines.append(f'{indent}        raise RuntimeError(f"QSV encoding failed: {{result.stderr}}")\n')
        
        skip_until_next_def = True
        continue
    
    if skip_until_next_def:
        # Skip old function body until we hit the next function or class
        if line.strip() and not line.startswith(' ' * (function_indent + 4)):
            if 'def ' in line or 'class ' in line or (function_indent == 0 and line[0] not in ' \t'):
                skip_until_next_def = False
                new_lines.append(line)
        continue
    
    new_lines.append(line)

with open('dlrm_s_pytorch.py', 'w') as f:
    f.writelines(new_lines)

print("✓ Fixed QSV encoding in dlrm_s_pytorch.py")
print("  - Added proper QSV device initialization")
print("  - Added padding for narrow dimensions (min width = 64)")
print("  - Added hardware upload filter")
