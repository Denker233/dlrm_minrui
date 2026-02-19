#!/usr/bin/env python3

with open('dlrm_s_pytorch.py', 'r') as f:
    content = f.read()

# Add tiling helper functions after the import statements
tiling_functions = '''
def tile_embeddings(data, emb_dim, num_emb, tile_size=4):
    """
    Arrange embeddings as tiles in a 2D grid.
    Each embedding becomes a tile_size×tile_size block.
    """
    import numpy as np
    
    # Reshape embeddings to tile_size×tile_size
    # Pad embedding dimension to tile_size² if needed
    tiles_per_emb = (emb_dim + tile_size**2 - 1) // tile_size**2
    padded_emb_dim = tiles_per_emb * tile_size**2
    
    # Pad data if needed
    if emb_dim < padded_emb_dim:
        data_2d = data.reshape(num_emb, emb_dim)
        padded_data = np.zeros((num_emb, padded_emb_dim), dtype=data.dtype)
        padded_data[:, :emb_dim] = data_2d
        data_2d = padded_data
    else:
        data_2d = data.reshape(num_emb, emb_dim)
    
    # Reshape each embedding into tile(s)
    tiles = data_2d.reshape(num_emb, tiles_per_emb, tile_size, tile_size)
    
    # Arrange tiles in a grid
    grid_size = int(np.ceil(np.sqrt(num_emb * tiles_per_emb)))
    total_tiles = grid_size * grid_size
    
    # Pad to fill grid
    if num_emb * tiles_per_emb < total_tiles:
        padding = np.zeros((total_tiles - num_emb * tiles_per_emb, tile_size, tile_size), dtype=data.dtype)
        all_tiles = np.concatenate([tiles.reshape(-1, tile_size, tile_size), padding], axis=0)
    else:
        all_tiles = tiles.reshape(-1, tile_size, tile_size)
    
    # Arrange in grid
    tiles_grid = all_tiles.reshape(grid_size, grid_size, tile_size, tile_size)
    
    # Merge tiles into final image
    rows = []
    for i in range(grid_size):
        row_tiles = [tiles_grid[i, j] for j in range(grid_size)]
        row = np.concatenate(row_tiles, axis=1)
        rows.append(row)
    image = np.concatenate(rows, axis=0)
    
    return image, grid_size, tiles_per_emb

def untile_embeddings(image, emb_dim, num_emb, grid_size, tiles_per_emb, tile_size=4):
    """
    Extract embeddings from tiled image.
    """
    import numpy as np
    
    # Split image back into tiles
    tiles_grid = image.reshape(grid_size, tile_size, grid_size, tile_size)
    tiles_grid = tiles_grid.transpose(0, 2, 1, 3)  # [grid_size, grid_size, tile_size, tile_size]
    
    # Flatten to list of tiles
    all_tiles = tiles_grid.reshape(-1, tile_size, tile_size)
    
    # Take only the tiles we need
    needed_tiles = num_emb * tiles_per_emb
    tiles = all_tiles[:needed_tiles]
    
    # Reshape back to embeddings
    embeddings = tiles.reshape(num_emb, tiles_per_emb * tile_size * tile_size)
    
    # Remove padding from embedding dimension
    embeddings = embeddings[:, :emb_dim]
    
    return embeddings.reshape(-1)

'''

# Insert tiling functions before the QSVEmbeddingCompressor class
class_start = content.find('class QSVEmbeddingCompressor:')
content = content[:class_start] + tiling_functions + content[class_start:]

# Now replace the _encode_qsv method to use tiling for large tables
old_encode = '''    def _encode_qsv(self, input_file, output_file, width, height):
        """Encode using QSV with dimension reshaping for large tables"""
        import numpy as np
        
        # Video codecs have dimension limits
        MAX_DIM = 16384
        MIN_WIDTH = 64   # libx265 minimum width
        MIN_HEIGHT = 64  # libx265 minimum height
        original_width, original_height = width, height'''

new_encode = '''    def _encode_qsv(self, input_file, output_file, width, height):
        """Encode using QSV with tiling for large tables"""
        import numpy as np
        
        # Video codecs have dimension limits
        MAX_DIM = 16384
        MIN_WIDTH = 64
        MIN_HEIGHT = 64
        TILE_SIZE = 4  # Each embedding → 4×4 tile
        original_width, original_height = width, height
        
        # For large tables, use tiling to preserve structure
        TILING_THRESHOLD = 50000  # Use tiling for tables with >50K embeddings
        
        if original_height > TILING_THRESHOLD:
            # Read data and tile it
            data = np.fromfile(input_file, dtype=np.uint8)
            image, grid_size, tiles_per_emb = tile_embeddings(
                data, original_width, original_height, TILE_SIZE
            )
            
            # Write tiled image
            image.tofile(input_file)
            width, height = image.shape[1], image.shape[0]
            
            # Store tiling info for decompression
            self._tiling_metadata = {
                'tiled': True,
                'grid_size': grid_size,
                'tiles_per_emb': tiles_per_emb,
                'tile_size': TILE_SIZE
            }
            
            print(f"      Tiled into {grid_size}×{grid_size} grid of {TILE_SIZE}×{TILE_SIZE} tiles → {width}×{height}")
        else:
            self._tiling_metadata = {'tiled': False}'''

content = content.replace(old_encode, new_encode)

# Update compress_table to save tiling metadata
old_compress_metadata = '''        # Metadata
        metadata = {
            'shape': (num_emb, emb_dim),
            'quantization': self.quantization,
            'quant_params': quant_metadata,
            'bits': self.bits,
            'codec': self.codec,
            'quality': self.quality
        }'''

new_compress_metadata = '''        # Metadata
        metadata = {
            'shape': (num_emb, emb_dim),
            'quantization': self.quantization,
            'quant_params': quant_metadata,
            'bits': self.bits,
            'codec': self.codec,
            'quality': self.quality,
            'tiling': getattr(self, '_tiling_metadata', {'tiled': False})
        }'''

content = content.replace(old_compress_metadata, new_compress_metadata)

# Update decompress_table to handle tiling
old_decompress_read = '''            # Read raw
            pixels_uint8 = np.fromfile(raw_file, dtype=np.uint8)
            pixels_uint8 = torch.from_numpy(pixels_uint8[:expected_size]).reshape(num_emb, emb_dim)'''

new_decompress_read = '''            # Read raw
            pixels_uint8 = np.fromfile(raw_file, dtype=np.uint8)
            
            # Handle tiling if used
            if metadata.get('tiling', {}).get('tiled', False):
                tiling = metadata['tiling']
                # Read tiled image dimensions from decoded data
                grid_size = tiling['grid_size']
                tile_size = tiling['tile_size']
                image_size = grid_size * tile_size
                
                # Untile
                pixels_uint8 = untile_embeddings(
                    pixels_uint8[:image_size*image_size].reshape(image_size, image_size),
                    emb_dim, num_emb,
                    grid_size, tiling['tiles_per_emb'], tile_size
                )
            else:
                pixels_uint8 = pixels_uint8[:expected_size]
            
            pixels_uint8 = torch.from_numpy(pixels_uint8).reshape(num_emb, emb_dim)'''

content = content.replace(old_decompress_read, new_decompress_read)

with open('dlrm_s_pytorch.py', 'w') as f:
    f.write(content)

print("✓ Implemented tile-based embedding compression")
print("  - Tables >50K rows use 4×4 tiling")
print("  - Preserves embedding structure in spatial layout")
print("  - Should improve compression quality for large tables")
