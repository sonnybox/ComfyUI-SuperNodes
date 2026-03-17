from comfy_api.latest import io


class SeedVRCalculateTiles(io.ComfyNode):
    """
    Calculates the optimal number of rows, cols, and overlap to tile an image
    based on a target pixel limit (target_tile_size^2) and anticipated upscale factor.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SuperSVRCalcTiles",
            display_name="🐧 SeedVR Calculate Tiles",
            category="SuperNodes/Tiling",
            inputs=[
                io.Image.Input("image", tooltip="The source image to measure."),
                io.Int.Input(
                    "target_tile_size",
                    default=1024,
                    min=256,
                    max=8192,
                    step=64,
                    tooltip="Target resolution side length (e.g. 1024 = 1MP limit).",
                ),
                io.Float.Input(
                    "upscale_by",
                    default=1.0,
                    min=1.0,
                    max=1024.0,
                    step=0.1,
                    tooltip="The scale factor you intend to use. Tiling is calculated relative to the upscaled dimensions.",
                ),
            ],
            outputs=[
                io.Int.Output(display_name="rows"),
                io.Int.Output(display_name="cols"),
                io.Float.Output(display_name="suggested_overlap"),
            ],
        )

    @classmethod
    def execute(cls, image, target_tile_size, upscale_by) -> io.NodeOutput:
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
            # Strategy: Maintain squareness of the tiles.
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

        return io.NodeOutput(rows, cols, overlap)


NODE = [SeedVRCalculateTiles]
