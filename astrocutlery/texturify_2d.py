"""
ANDRIX ® 2025-2026 🤙

Generates 2D textures from text data containing space-separated numeric columns.
Converts from raw particle/field data (in text format) to Unity-ready EXR images.

All values must be separated by spaces with newline terminators.
Outputted textures can be sampled in Unity VFX graph to retrieve information.

Requirements:
- OpenEXR for EXR encoding
- numpy for efficient array operations
"""

import math
import os
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import OpenEXR


def normalize_value(value: float, v_min: float, v_max: float) -> float:
    """Remap a value from [v_min, v_max] to [0, 1]."""
    if v_max == v_min:
        return 0.0
    return max(0.0, min(1.0, (value - v_min) / (v_max - v_min)))


def compute_texture_size(line_count: int) -> int:
    """Compute smallest square texture size needed to store line_count pixels."""
    return math.ceil(math.sqrt(line_count))


def prepend_zeros(value: int, target_length: int) -> str:
    """Pad integer with leading zeros."""
    return str(value).zfill(target_length)


def read_text_data(file_path: str) -> list:
    """Read space-separated numeric data from text file, return list of float lists."""
    with open(file_path, 'r') as f:
        lines = f.read().strip().split('\n')
    
    data = []
    for line in lines:
        if line.strip():
            values = [float(x) for x in line.split()]
            data.append(values)
    return data


def create_rgba_arrays(data: list, mins: list, maxs: list) -> Tuple[list, int]:
    """
    Create RGBA color arrays from parsed data.
    Groups first 3 values (x,y,z) into first texture.
    Groups remaining values into additional textures (3 per texture).
    
    Returns:
    - color_arrays: list of numpy arrays (each size x size x 4)
    - num_textures: number of textures created
    """
    num_lines = len(data)
    num_dimensions = len(mins)
    
    # Calculate number of textures (1 for x,y,z + ceil((remaining)/3))
    remaining = max(0, num_dimensions - 3)
    num_textures = 1 + math.ceil(remaining / 3.0)
    
    # Texture size
    size = compute_texture_size(num_lines)
    
    # Initialize color arrays (flattened for efficiency)
    color_arrays = [np.zeros(size * size * 4, dtype=np.float32) for _ in range(num_textures)]
    
    # Process each data line
    for pixel_idx, values in enumerate(data):
        # Extract and normalize x, y, z
        x = normalize_value(values[0], mins[0], maxs[0])
        y = normalize_value(values[1], mins[1], maxs[1])
        z = normalize_value(values[2], mins[2], maxs[2])
        
        # First texture always gets x, y, z
        color_arrays[0][pixel_idx * 4 + 0] = x
        color_arrays[0][pixel_idx * 4 + 1] = y
        color_arrays[0][pixel_idx * 4 + 2] = z
        color_arrays[0][pixel_idx * 4 + 3] = 1.0
        
        # Additional textures get remaining values (3 per texture)
        for tex_idx in range(1, num_textures):
            base_dim = 3 + (tex_idx - 1) * 3
            
            r = normalize_value(values[base_dim], mins[base_dim], maxs[base_dim]) if base_dim < num_dimensions else 0.0
            g = normalize_value(values[base_dim + 1], mins[base_dim + 1], maxs[base_dim + 1]) if base_dim + 1 < num_dimensions else 0.0
            b = normalize_value(values[base_dim + 2], mins[base_dim + 2], maxs[base_dim + 2]) if base_dim + 2 < num_dimensions else 0.0
            
            color_arrays[tex_idx][pixel_idx * 4 + 0] = r
            color_arrays[tex_idx][pixel_idx * 4 + 1] = g
            color_arrays[tex_idx][pixel_idx * 4 + 2] = b
            color_arrays[tex_idx][pixel_idx * 4 + 3] = 1.0
    
    # Reshape to 2D texture format (size x size x 4)
    color_arrays = [arr.reshape((size, size, 4)) for arr in color_arrays]
    
    return color_arrays, num_textures, size


def write_exr_rgba(file_path: str, data: np.ndarray) -> None:
    """
    Write OpenEXR file from RGBA data using OpenEXR library.
    Uses proven approach from mathviz-python-script.txt.
    data shape: (height, width, 4) for RGBA
    """
    channels = {"RGBA": data}
    header = {
        "compression": OpenEXR.NO_COMPRESSION,
        "type": OpenEXR.scanlineimage
    }
    with OpenEXR.File(header, channels) as outfile:
        outfile.write(file_path)


def create_texture_2d(
    source_author_folder: str,
    source_frames_folder: str,
    source_file_name: str,
    array_of_mins: list,
    array_of_maxs: list,
    output_base_dir: str,
    forced_size: Optional[int] = None
) -> None:
    """
    Main function to convert text data to 2D textures.
    
    Args:
        source_author_folder: Author identifier (e.g., 'maximelombart')
        source_frames_folder: Frame set identifier (e.g., '1-frame')
        source_file_name: Base name of text file (without extension)
        array_of_mins: Minimum values for each dimension
        array_of_maxs: Maximum values for each dimension
        output_base_dir: Root directory for output files
        forced_size: Optional texture size to match across sequence
    """
    import time
    start_time = time.time()
    
    num_dimensions = len(array_of_mins)
    remaining = max(0, num_dimensions - 3)
    num_textures = 1 + math.ceil(remaining / 3.0)
    
    print(f"[texturify_2d] Data has {num_dimensions} columns. {num_textures} texture(s) with name '{source_file_name}' will be generated.")
    
    # Construct file path
    input_path = Path(output_base_dir) / source_author_folder / source_frames_folder / f"{source_file_name}.txt"
    
    # Read data
    data = read_text_data(str(input_path))
    num_lines = len(data)
    
    # Create color arrays
    color_arrays, num_textures, natural_size = create_rgba_arrays(data, array_of_mins, array_of_maxs)
    
    # Apply forced size if specified
    size = forced_size if forced_size else natural_size
    if forced_size and forced_size < natural_size:
        print(f"[texturify_2d] Warning: forcedSize {forced_size} is lower than natural size {natural_size}")
    
    plural = "s" if num_textures > 1 else ""
    print(f"[texturify_2d] Generating {num_textures} texture{plural} of size {size}x{size} from {num_lines} lines of data...")
    
    # Output directory
    output_dir = Path(output_base_dir) / source_author_folder / source_frames_folder
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save textures
    for tex_idx in range(num_textures):
        # Dimension folder name (always create folder, even for single texture)
        dim_start = 1 + tex_idx * 3
        dim_end_r = dim_start
        dim_end_g = dim_start + 1
        dim_end_b = dim_start + 2
        
        folder_name = f"dimensions-{prepend_zeros(dim_end_r, 2)}-{prepend_zeros(dim_end_g, 2)}-{prepend_zeros(dim_end_b, 2)}"
        tex_dir = output_dir / folder_name
        tex_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare texture data (pad if needed)
        tex_data = color_arrays[tex_idx]
        if forced_size and forced_size > natural_size:
            padded = np.zeros((forced_size, forced_size, 4), dtype=np.float32)
            padded[:natural_size, :natural_size, :] = tex_data
            tex_data = padded
        
        # Save EXR file
        output_file = tex_dir / f"{source_file_name}-tex-{size}-{tex_idx + 1}.exr"
        write_exr_rgba(str(output_file), tex_data)
    
    elapsed = time.time() - start_time
    print(f"[texturify_2d] Created and saved texture{plural} in {elapsed:.2f} seconds 👌")


# Convenience wrapper for batch processing
def create_textures_batch(configs: list, output_base_dir: str) -> None:
    """
    Process multiple texture generation configs sequentially.
    
    Args:
        configs: List of dicts with keys:
                 - author, frames, name, mins, maxs, (optional) size
        output_base_dir: Root output directory
    """
    import time
    batch_start = time.time()
    
    for config in configs:
        create_texture_2d(
            config['author'],
            config['frames'],
            config['name'],
            config['mins'],
            config['maxs'],
            output_base_dir,
            config.get('size')
        )
    
    elapsed = time.time() - batch_start
    total = len(configs)
    print(f"[texturify_2d] Batch processing complete: {total} texture set(s) in {elapsed:.2f} seconds 👌")
