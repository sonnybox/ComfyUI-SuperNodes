from comfy_api.latest import io
import torch


class CreateTiles(io.ComfyNode):
    """
    Splits an image into a grid of tiles with configurable overlap.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SuperCreateTiles",
            display_name="🐧 Create Tiles",
            category="SuperNodes/Tiling",
            inputs=[
                io.Image.Input(
                    "image", tooltip="The source image to be tiled."
                ),
                io.Int.Input(
                    "rows",
                    default=2,
                    min=1,
                    max=64,
                    step=1,
                    tooltip="Number of rows in the grid.",
                ),
                io.Int.Input(
                    "cols",
                    default=2,
                    min=1,
                    max=64,
                    step=1,
                    tooltip="Number of columns in the grid.",
                ),
                io.Float.Input(
                    "overlap",
                    default=0.25,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    tooltip="Overlap factor (0.0-1.0). 0.0 means distinct grid cells. 1.0 means the tile extends into adjacent cells by 50% of the grid size.",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="tiles"),
                io.Custom("STITCH_INFO").Output(display_name="stitch_info"),
            ],
        )

    @classmethod
    def execute(cls, image, rows, cols, overlap) -> io.NodeOutput:
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

                    tile_coords.append(
                        {
                            "b_index": b,
                            "row_idx": r,
                            "col_idx": c_idx,
                            "y": y_start,
                            "x": x_start,
                            "h": tile_h,
                            "w": tile_w,
                        }
                    )

        # Stack into [Batch * Rows * Cols, TileH, TileW, C]
        output_tiles = torch.stack(all_tiles)

        stitch_info = {
            "original_height": h,
            "original_width": w,
            "original_batch_size": batch_size,
            "rows": rows,
            "cols": cols,
            "overlap": stored_overlap_px,
            "tiles": tile_coords,
        }

        return io.NodeOutput(output_tiles, stitch_info)


V3_NODES = [CreateTiles]
