#!/usr/bin/env python3

with open('dlrm_s_pytorch.py', 'r') as f:
    content = f.read()

# Fix 1: Update encoding command to always use minimum width of 64
old_encode_cmd = '''        cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-pix_fmt', 'gray',
            '-s', f'{width}x{height}',
            '-r', '1',
            '-i', input_file,
            '-c:v', self.codec,
            '-global_quality', str(self.quality),
            '-frames:v', '1',
            output_file
        ]'''

new_encode_cmd = '''        cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-pix_fmt', 'gray',
            '-s', f'{width}x{height}',
            '-r', '1',
            '-i', input_file,
            '-c:v', self.codec,
            '-crf', str(self.quality),
            '-preset', 'ultrafast',
            '-x265-params', 'log-level=error:allow-non-conformance=1',
            '-frames:v', '1',
            output_file
        ]'''

content = content.replace(old_encode_cmd, new_encode_cmd)

# Fix 2: Always reshape narrow tables (width < 64)
old_reshape = '''        # Video codecs have max dimension limits (typically 16384)
        MAX_DIM = 16384
        original_width, original_height = width, height
        
        if height > MAX_DIM:'''

new_reshape = '''        # Video codecs have dimension limits
        MAX_DIM = 16384
        MIN_DIM = 64  # libx265 requires minimum width
        original_width, original_height = width, height
        
        if height > MAX_DIM or width < MIN_DIM:'''

content = content.replace(old_reshape, new_reshape)

# Fix 3: Better reshape logic
old_reshape_logic = '''            # Reshape to fit within limits
            total_pixels = width * height
            height = MAX_DIM
            width = (total_pixels + height - 1) // height  # Ceiling division'''

new_reshape_logic = '''            # Reshape to fit within limits
            total_pixels = width * height
            
            if height > MAX_DIM:
                # Table is too tall
                height = MAX_DIM
                width = (total_pixels + height - 1) // height
            elif width < MIN_DIM:
                # Table is too narrow
                width = MIN_DIM
                height = (total_pixels + width - 1) // width'''

content = content.replace(old_reshape_logic, new_reshape_logic)

with open('dlrm_s_pytorch.py', 'w') as f:
    f.write(content)

print("✓ Fixed encoding issues:")
print("  - Added allow-non-conformance for libx265")
print("  - Changed to -crf instead of -global-quality")
print("  - Added minimum width requirement (64 pixels)")
print("  - Improved reshape logic")
