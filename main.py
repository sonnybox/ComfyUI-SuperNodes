import torch
import numpy as np
from PIL import Image, ImageOps
import torch.nn.functional as F

CATEGORY = "Super Nodes"

class CalculateUpscaleTiles:
    """
    A ComfyUI node to calculate the optimal number of rows and columns to divide an image into,
    based on a target tile size and a slicing algorithm that aims for square-like tiles.
    """
    @classmethod
    def INPUT_TYPES(cls):
        """
        Defines the input types for the node.
        """
        return {
            "required": {
                "image": ("IMAGE",),
                "tile_size": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 8192,
                    "step": 8,
                    "tooltip": "The target value after upscaling by multiplier."
                }),
                "multiplier": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "The upscaling factor."
                }),
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("rows", "columns")
    FUNCTION = "calculate"
    CATEGORY = CATEGORY + "/upscale"

    def calculate(self, image, tile_size, multiplier):
        """
        Calculates the optimal row and column count based on the user's iterative algorithm.
        The goal is to find a configuration where the tile's average side length approaches
        (tile_size / multiplier).
        """
        # Get image dimensions from the input tensor.
        # The tensor shape is (batch, height, width, channels).
        image_height = image.shape[1]
        image_width = image.shape[2]

        # The effective target for the average side length of a tile.
        # A higher multiplier results in a smaller target side length, thus more tiles.
        if multiplier == 0:
             multiplier = 1.0 # Avoid division by zero
        effective_target = tile_size / multiplier

        rows = 1
        cols = 1

        # Calculate the initial average side length and its difference from the target.
        current_avg_side = ((image_width / cols) + (image_height / rows)) / 2.0
        prev_diff = abs(current_avg_side - effective_target)

        # Store the state of the previous iteration, which will be our result if the next step is worse.
        prev_rows = 1
        prev_cols = 1

        # Set a reasonable iteration limit to prevent any potential infinite loops.
        iteration_limit = 250

        for i in range(iteration_limit):
            # Store the current state before calculating the next step.
            prev_rows = rows
            prev_cols = cols

            # Calculate current tile dimensions for decision making.
            tile_width = image_width / float(cols)
            tile_height = image_height / float(rows)

            # Decision logic: always add a slice to the longest dimension of the tiles.
            if tile_width > tile_height:
                cols += 1
            elif tile_height > tile_width:
                rows += 1
            else:
                # If tiles are perfectly square, slice the dimension with fewer divisions to break the tie.
                if rows < cols:
                    rows += 1
                else:
                    cols += 1

            # Calculate the new average side length and its difference from the target.
            current_avg_side = ((image_width / float(cols)) + (image_height / float(rows))) / 2.0
            current_diff = abs(current_avg_side - effective_target)

            # Stopping condition: if the difference from the target starts to increase,
            # it means the previous step was the optimal one.
            if current_diff > prev_diff:
                # The loop has gone one step too far. The previous state is the answer.
                return (prev_rows, prev_cols)

            # If we haven't found the minimum yet, update the previous difference for the next check.
            prev_diff = current_diff

        # If the loop finishes (hits the limit), return the last calculated best state.
        return (prev_rows, prev_cols)


class CreateTiles:
    """
    A node that takes an image and slices it into a specified number of rows and columns,
    with an optional overlap which adds a black border for easy reconstruction.
    It also outputs the data needed to reconstruct the image, using ratios
    to support resizing of tiles before untiling.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "rows": ("INT", {"default": 2, "min": 1, "max": 1024, "step": 1}),
                "columns": ("INT", {"default": 2, "min": 1, "max": 1024, "step": 1}),
                "overlap": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 0.5, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE", "TILING_DATA")
    RETURN_NAMES = ("tiled_image", "tiling_data")
    FUNCTION = "create_tiles"
    CATEGORY = CATEGORY + "/upscale"

    def tensor_to_pil(self, tensor_image):
        return [Image.fromarray(np.clip(255. * i.cpu().numpy(), 0, 255).astype(np.uint8)) for i in tensor_image]

    def pil_to_tensor(self, pil_images):
        return torch.stack([torch.from_numpy(np.array(i).astype(np.float32) / 255.0) for i in pil_images])

    def create_tiles(self, image: torch.Tensor, rows: int, columns: int, overlap: float):
        source_image_pil = self.tensor_to_pil(image)[0]
        original_width, original_height = source_image_pil.size
        base_width = (original_width // columns) * columns
        base_height = (original_height // rows) * rows

        if rows == 1 and columns == 1:
            overlap = 0.0

        if base_width != original_width or base_height != original_height:
            resized_image_pil = source_image_pil.resize((base_width, base_height), resample=Image.Resampling.LANCZOS)
        else:
            resized_image_pil = source_image_pil

        tile_width = base_width // columns
        tile_height = base_height // rows

        overlap_w_px = (int(tile_width * overlap) // 2) * 2
        overlap_h_px = (int(tile_height * overlap) // 2) * 2

        if overlap > 0:
            border_w = overlap_w_px // 2
            border_h = overlap_h_px // 2
            padded_image_pil = ImageOps.expand(resized_image_pil, border=(border_w, border_h), fill='black')
        else:
            padded_image_pil = resized_image_pil

        tiles_pil = []
        for r in range(rows):
            for c in range(columns):
                left = c * tile_width
                upper = r * tile_height
                right = left + tile_width + overlap_w_px
                lower = upper + tile_height + overlap_h_px
                box = (left, upper, right, lower)
                tile = padded_image_pil.crop(box)
                tiles_pil.append(tile)

        output_tensors = self.pil_to_tensor(tiles_pil)

        # Create the data dictionary using ratios
        # This allows the untile node to work even if the tiles are resized.
        overlap_percent_w = overlap_w_px / tile_width if tile_width > 0 else 0
        overlap_percent_h = overlap_h_px / tile_height if tile_height > 0 else 0

        tiling_data = {
            "rows": rows,
            "columns": columns,
            "original_width": base_width,
            "original_height": base_height,
            "overlap_percent_w": overlap_percent_w,
            "overlap_percent_h": overlap_percent_h,
        }

        return output_tensors, tiling_data


class StitchTiles:
    """
    A node that takes a batch of tiled images and reconstruction data,
    then reassembles them into a single image. It correctly handles
    tiles that have been resized after creation (e.g., by an upscaler),
    can feather the edges to create a seamless blend, and allows for
    seam adjustments via x/y offsets.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_batch": ("IMAGE",),
                "tiling_data": ("TILING_DATA",),
                "feather_edges": ("BOOLEAN", {"default": True}),
                "offset_x": ("INT", {"default": 0, "min": -2048, "max": 2048, "step": 1}),
                "offset_y": ("INT", {"default": 0, "min": -2048, "max": 2048, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "untile_images"
    CATEGORY = CATEGORY + "/upscale"

    def untile_images(self, image_batch: torch.Tensor, tiling_data: dict, feather_edges: bool, offset_x: int,
                      offset_y: int):
        # --- 1. Unpack data and get dimensions ---
        rows = tiling_data["rows"]
        cols = tiling_data["columns"]
        overlap_percent_w = tiling_data.get("overlap_percent_w", 0.0)
        overlap_percent_h = tiling_data.get("overlap_percent_h", 0.0)
        has_overlap = overlap_percent_w > 0 or overlap_percent_h > 0

        num_tiles = image_batch.shape[0]
        if num_tiles != rows * cols:
            raise ValueError(
                f"Number of images in batch ({num_tiles}) does not match rows*cols ({rows * cols}) from tiling_data.")

        incoming_h, incoming_w = image_batch.shape[1:3]
        print(f"StitchTiles: Received {num_tiles} tiles of size {incoming_w}x{incoming_h}.")

        # --- 2. Calculate core, border, and step dimensions of the upscaled tiles ---
        core_upscaled_w = round(incoming_w / (1 + overlap_percent_w))
        core_upscaled_h = round(incoming_h / (1 + overlap_percent_h))
        border_w = (incoming_w - core_upscaled_w) // 2
        border_h = (incoming_h - core_upscaled_h) // 2

        # The step size is the distance between the top-left corner of adjacent tiles.
        # The offset adjusts this distance.
        step_w = core_upscaled_w - offset_x
        step_h = core_upscaled_h - offset_y

        # --- 3. Route to either feathered or non-feathered method ---
        if not feather_edges or not has_overlap:
            # --- METHOD A: Hard-Edge Reconstruction ---

            # Calculate final canvas size and paste offsets to handle potential negative coordinates.
            x_coords = [c * step_w for c in range(cols)]
            y_coords = [r * step_h for r in range(rows)]

            min_x = min(x_coords)
            max_x = max(x_coords)
            min_y = min(y_coords)
            max_y = max(y_coords)

            canvas_w = (max_x - min_x) + core_upscaled_w
            canvas_h = (max_y - min_y) + core_upscaled_h
            paste_offset_x = -min_x
            paste_offset_y = -min_y

            final_image = Image.new('RGB', (canvas_w, canvas_h))
            tiles_pil = [Image.fromarray(np.clip(255. * i.cpu().numpy(), 0, 255).astype(np.uint8)) for i in image_batch]
            crop_box = (border_w, border_h, incoming_w - border_w, incoming_h - border_h)

            tile_index = 0
            for r in range(rows):
                for c in range(cols):
                    core_tile = tiles_pil[tile_index].crop(crop_box)
                    paste_position = (c * step_w + paste_offset_x, r * step_h + paste_offset_y)
                    final_image.paste(core_tile, paste_position)
                    tile_index += 1

            output_tensor = torch.from_numpy(np.array(final_image).astype(np.float32) / 255.0).unsqueeze(0)

        else:
            # --- METHOD B: Feathered Blending Reconstruction ---

            # Create a 2D feathering mask (a window function)
            feather_mask = np.ones((incoming_h, incoming_w, 1), dtype=np.float32)
            if border_w > 0:
                ramp_w = np.linspace(0.0, 1.0, border_w)
                feather_mask[:, :border_w, :] *= ramp_w.reshape(1, -1, 1)
                feather_mask[:, -border_w:, :] *= np.flip(ramp_w).reshape(1, -1, 1)
            if border_h > 0:
                ramp_h = np.linspace(0.0, 1.0, border_h)
                feather_mask[:border_h, :, :] *= ramp_h.reshape(-1, 1, 1)
                feather_mask[-border_h:, :, :] *= np.flip(ramp_h).reshape(-1, 1, 1)

            # Calculate canvas size and offsets to handle any negative coordinates from offsets
            x_coords = [c * step_w for c in range(cols)]
            y_coords = [r * step_h for r in range(rows)]
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)

            canvas_w = (max_x - min_x) + incoming_w
            canvas_h = (max_y - min_y) + incoming_h
            paste_offset_x = -min_x
            paste_offset_y = -min_y

            final_image_np = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
            weight_map = np.zeros((canvas_h, canvas_w, 1), dtype=np.float32)
            tiles_np = image_batch.cpu().numpy()

            # Iterate, apply mask, and accumulate values and weights
            tile_index = 0
            for r in range(rows):
                for c in range(cols):
                    y_start = r * step_h + paste_offset_y
                    x_start = c * step_w + paste_offset_x
                    final_image_np[y_start:y_start + incoming_h, x_start:x_start + incoming_w] += tiles_np[
                                                                                                      tile_index] * feather_mask
                    weight_map[y_start:y_start + incoming_h, x_start:x_start + incoming_w] += feather_mask
                    tile_index += 1

            # Normalize the image by dividing by the accumulated weights
            weight_map[weight_map == 0] = 1.0
            final_image_np /= weight_map

            # Crop the final image to the target size
            target_w = (max_x - min_x) + core_upscaled_w
            target_h = (max_y - min_y) + core_upscaled_h

            crop_start_x = paste_offset_x + border_w
            crop_start_y = paste_offset_y + border_h

            final_image_np = final_image_np[crop_start_y:crop_start_y + target_h, crop_start_x:crop_start_x + target_w]

            # Clip and convert to tensor
            final_image_np = np.clip(final_image_np, 0.0, 1.0)
            output_tensor = torch.from_numpy(final_image_np).unsqueeze(0)

        return (output_tensor,)