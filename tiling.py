import torch
import math
import torch.nn.functional as F

# Helper function to generate a feather mask
def generate_feather_mask(shape, radius, device):
    """
    Generates a weight mask with a linear fade-out at the edges.
    shape: (H, W, C) or (H, W)
    radius: pixels to fade
    """
    h, w = shape[:2]
    mask = torch.ones((h, w), dtype=torch.float32, device=device)
    
    if radius <= 0:
        return mask

    # Clamp radius to half the size to prevent negative indices
    # This automatically handles "max amount" blending when a large radius is requested
    radius = min(radius, h // 2, w // 2)

    # Create ramps
    ramp = torch.linspace(0, 1, radius, device=device)
    
    # Top edge
    mask[:radius, :] *= ramp.unsqueeze(1)
    # Bottom edge
    mask[-radius:, :] *= ramp.flip(0).unsqueeze(1)
    # Left edge
    mask[:, :radius] *= ramp.unsqueeze(0)
    # Right edge
    mask[:, -radius:] *= ramp.flip(0).unsqueeze(0)
    
    return mask

class CreateTiles:
    """
    Splits an image into a grid of tiles with configurable overlap.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The source image to be tiled."}),
                "rows": ("INT", {"default": 2, "min": 1, "max": 64, "step": 1, "tooltip": "Number of rows in the grid."}),
                "cols": ("INT", {"default": 2, "min": 1, "max": 64, "step": 1, "tooltip": "Number of columns in the grid."}),
                "overlap": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Overlap factor (0.0-1.0). 0.0 means distinct grid cells. 1.0 means the tile extends into adjacent cells by 50% of the grid size."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STITCH_INFO")
    RETURN_NAMES = ("tiles", "stitch_info")
    FUNCTION = "execute"
    CATEGORY = "SuperNodes"
    
    def execute(self, image, rows, cols, overlap):
        # image shape: [B, H, W, C]
        batch_size, h, w, c = image.shape
        
        # 1. Calculate the base grid size (stride)
        base_h = h // rows
        base_w = w // cols
        
        # 2. Calculate the actual pixel size of the tile.
        #    User logic: At overlap 1.0, we want "1/4 corner + half of adjacent".
        #    In a 2x2, Base is 50%. Adjacent is 50%. Half adjacent is 25%.
        #    Total target = 75%.
        #    Math: Base + (Base * 0.5 * overlap)
        tile_h = base_h + int(base_h * 0.5 * overlap)
        tile_w = base_w + int(base_w * 0.5 * overlap)

        # Sanity check: Tile cannot be larger than the image itself
        tile_h = min(tile_h, h)
        tile_w = min(tile_w, w)

        all_tiles = []
        tile_coords = []
        
        # We record the effective overlap in pixels for the stitch node to use later if needed,
        # though the stitch node primarily relies on absolute coordinates.
        overlap_h_px = tile_h - base_h
        overlap_w_px = tile_w - base_w
        stored_overlap_px = max(overlap_h_px, overlap_w_px)

        for b in range(batch_size):
            img = image[b] 
            
            for r in range(rows):
                for c_idx in range(cols):
                    # 3. Calculate Coordinates
                    # To ensure consistent tile sizes for batch processing (important for VAEs),
                    # we calculate the ideal center of the grid cell, then expand outwards.
                    # If we hit an edge, we slide the window back in rather than shrinking it.
                    
                    center_y = r * base_h + (base_h // 2)
                    center_x = c_idx * base_w + (base_w // 2)
                    
                    # Determine top-left corner based on center and calculated tile size
                    y_start = center_y - (tile_h // 2)
                    x_start = center_x - (tile_w // 2)
                    
                    # 4. Slide-to-fit (Keep tile within bounds, but preserve size)
                    # Constraint: 0 <= y <= H - tile_h
                    y_start = max(0, min(y_start, h - tile_h))
                    x_start = max(0, min(x_start, w - tile_w))
                    
                    y_end = y_start + tile_h
                    x_end = x_start + tile_w
                    
                    # Crop
                    crop = img[y_start:y_end, x_start:x_end, :]
                    all_tiles.append(crop)
                    
                    tile_coords.append({
                        "b_index": b,
                        "row_idx": r,
                        "col_idx": c_idx,
                        "y": y_start,
                        "x": x_start,
                        "h": tile_h,
                        "w": tile_w
                    })

        # Stack into [Batch * Rows * Cols, TileH, TileW, C]
        output_tiles = torch.stack(all_tiles)
        
        stitch_info = {
            "original_height": h,
            "original_width": w,
            "original_batch_size": batch_size,
            "rows": rows,
            "cols": cols,
            "overlap": stored_overlap_px, 
            "tiles": tile_coords
        }

        return (output_tiles, stitch_info)

class StitchTiles:
    """
    Reconstructs an image from tiles using the metadata provided by CreateTiles.
    Includes automatic feathering to blend seams.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tiles": ("IMAGE", {"tooltip": "The batch of tiles to be stitched back together."}),
                "stitch_info": ("STITCH_INFO", {"tooltip": "Metadata generated by the CreateTiles node."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "SuperNodes"

    def execute(self, tiles, stitch_info):
        if tiles.shape[0] != len(stitch_info["tiles"]):
            raise ValueError(f"Mismatch: Info expects {len(stitch_info['tiles'])} tiles, but got {tiles.shape[0]}.")
            
        device = tiles.device
        
        # Detect resizing (e.g. if tiles were upscaled)
        current_tile_h, current_tile_w = tiles.shape[1], tiles.shape[2]
        orig_tile_h = stitch_info["tiles"][0]["h"]
        orig_tile_w = stitch_info["tiles"][0]["w"]
        
        scale_h = current_tile_h / orig_tile_h
        scale_w = current_tile_w / orig_tile_w
        
        final_h = round(stitch_info["original_height"] * scale_h)
        final_w = round(stitch_info["original_width"] * scale_w)
        original_batch_size = stitch_info["original_batch_size"]
        channels = tiles.shape[3]

        out_image = torch.zeros((original_batch_size, final_h, final_w, channels), device=device)
        out_weights = torch.zeros((original_batch_size, final_h, final_w, 1), device=device)
        
        # Calculate blending radius
        # We use a radius proportional to the overlap, but robust enough to cover the seams.
        # Since tiles can slightly vary in position due to scale, we use the max dimension for smooth falloff.
        effective_radius = max(current_tile_h, current_tile_w) // 4 
        # Note: generate_feather_mask handles large radii gracefully by clamping to half-size internally.
        
        weight_mask = generate_feather_mask((current_tile_h, current_tile_w), effective_radius, device)
        weight_mask = weight_mask.unsqueeze(-1)

        for i, tile_meta in enumerate(stitch_info["tiles"]):
            b_idx = tile_meta["b_index"]
            tile_img = tiles[i]
            
            # Map original coordinates to new scaled coordinates
            y_start = round(tile_meta["y"] * scale_h)
            x_start = round(tile_meta["x"] * scale_w)
            
            y_end = y_start + current_tile_h
            x_end = x_start + current_tile_w
            
            # Bounds check for rounding errors
            y_end = min(y_end, final_h)
            x_end = min(x_end, final_w)
            
            h_actual = y_end - y_start
            w_actual = x_end - x_start
            
            if h_actual <= 0 or w_actual <= 0:
                continue

            tile_crop = tile_img[:h_actual, :w_actual, :]
            mask_crop = weight_mask[:h_actual, :w_actual, :]
            
            out_image[b_idx, y_start:y_end, x_start:x_end, :] += tile_crop * mask_crop
            out_weights[b_idx, y_start:y_end, x_start:x_end, :] += mask_crop

        out_weights[out_weights == 0] = 1.0
        final_image = out_image / out_weights
        
        return (final_image,)

class SeedVRCalculateTiles:
    """
    Calculates the optimal number of rows, cols, and overlap to tile an image
    based on a target pixel limit (target_tile_size^2) and anticipated upscale factor.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The source image to measure."}),
                "target_tile_size": ("INT", {"default": 1024, "min": 256, "max": 8192, "step": 64, "tooltip": "Target resolution side length (e.g. 1024 = 1MP limit)."}),
                "upscale_by": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 1024.0, "step": 0.1, "tooltip": "The scale factor you intend to use. Tiling is calculated relative to the upscaled dimensions."}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "FLOAT")
    RETURN_NAMES = ("rows", "cols", "suggested_overlap")
    FUNCTION = "calculate"
    CATEGORY = "SuperNodes"

    def calculate(self, image, target_tile_size, upscale_by):
        batch_size, h_orig, w_orig, c = image.shape
        
        # Calculate the dimensions of the final upscaled image
        h_final = int(h_orig * upscale_by)
        w_final = int(w_orig * upscale_by)
        
        # Target area (e.g. 1024x1024 = 1,048,576 pixels)
        target_area = target_tile_size * target_tile_size
        
        # Start with 1x1
        rows = 1
        cols = 1
        overlap = 0.0
        
        # Iteratively split until the tile size is within acceptable bounds of the target
        # We allow a small tolerance (e.g., 10%) to prevent unnecessary splitting for edge cases
        tolerance_multiplier = 1.1 
        
        while True:
            # 1. Determine Overlap for current grid size
            # Formula: 0.05 * max(rows, cols), maxing out at 1.0. 
            # 1x1 is a special case with 0 overlap.
            if rows == 1 and cols == 1:
                overlap = 0.0
            else:
                overlap = min(1.0, 0.05 * max(rows, cols))
            
            # 2. Calculate resulting tile dimensions with this overlap
            # Base dimensions (no overlap)
            base_h = h_final / rows
            base_w = w_final / cols
            
            # Tile dimensions (Base + Overlap portion)
            # CreateTiles logic: tile_size = base + (base * 0.5 * overlap)
            tile_h = base_h * (1 + 0.5 * overlap)
            tile_w = base_w * (1 + 0.5 * overlap)
            
            tile_area = tile_h * tile_w
            
            # 3. Check if we fit within target
            if tile_area <= (target_area * tolerance_multiplier):
                break
                
            # 4. If not, split.
            # Strategy: Maintain squareness of the *tiles*.
            # Calculate aspect ratio of the potential new grid cells to see which split 
            # brings us closer to 1:1 tile aspect ratio.
            
            # Current tile aspect ratio (Width / Height)
            current_tile_ar = tile_w / tile_h
            
            # If AR > 1 (Wide tile), splitting columns makes it more square (reduces width).
            # If AR < 1 (Tall tile), splitting rows makes it more square (reduces height).
            if current_tile_ar > 1.0:
                cols += 1
            else:
                rows += 1
                
        return (rows, cols, overlap)
