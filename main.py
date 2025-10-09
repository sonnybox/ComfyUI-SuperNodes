import torch
import numpy as np
from PIL import Image, ImageOps


class SuperCreateTiles:
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
    CATEGORY = "SuperComfyNodes"

    def tensor_to_pil(self, tensor_image):
        return [Image.fromarray(np.clip(255. * i.cpu().numpy(), 0, 255).astype(np.uint8)) for i in tensor_image]

    def pil_to_tensor(self, pil_images):
        return torch.stack([torch.from_numpy(np.array(i).astype(np.float32) / 255.0) for i in pil_images])

    def create_tiles(self, image: torch.Tensor, rows: int, columns: int, overlap: float):
        source_image_pil = self.tensor_to_pil(image)[0]
        original_width, original_height = source_image_pil.size
        base_width = (original_width // columns) * columns
        base_height = (original_height // rows) * rows

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


class SuperUntile:
    """
    A node that takes a batch of tiled images and reconstruction data,
    then reassembles them into a single image. It correctly handles
    tiles that have been resized after creation (e.g., by an upscaler)
    and can feather the edges to create a seamless blend.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_batch": ("IMAGE",),
                "tiling_data": ("TILING_DATA",),
                "feather_edges": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "untile_images"
    CATEGORY = "SuperComfyNodes"

    def untile_images(self, image_batch: torch.Tensor, tiling_data: dict, feather_edges: bool):
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
        print(f"Super Untile: Received {num_tiles} tiles of size {incoming_w}x{incoming_h}.")

        # --- 2. Calculate core and border dimensions of the upscaled tiles ---
        core_upscaled_w = round(incoming_w / (1 + overlap_percent_w))
        core_upscaled_h = round(incoming_h / (1 + overlap_percent_h))
        border_w = (incoming_w - core_upscaled_w) // 2
        border_h = (incoming_h - core_upscaled_h) // 2

        # --- 3. Route to either feathered or non-feathered method ---
        if not feather_edges or not has_overlap:
            # --- METHOD A: Original Hard-Edge Reconstruction ---
            final_w = core_upscaled_w * cols
            final_h = core_upscaled_h * rows

            final_image = Image.new('RGB', (final_w, final_h))
            tiles_pil = [Image.fromarray(np.clip(255. * i.cpu().numpy(), 0, 255).astype(np.uint8)) for i in image_batch]

            crop_box = (border_w, border_h, incoming_w - border_w, incoming_h - border_h)

            tile_index = 0
            for r in range(rows):
                for c in range(cols):
                    core_tile = tiles_pil[tile_index].crop(crop_box)
                    paste_position = (c * core_upscaled_w, r * core_upscaled_h)
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

            # Prepare large canvases for blending
            final_w = (cols - 1) * core_upscaled_w + incoming_w
            final_h = (rows - 1) * core_upscaled_h + incoming_h
            final_image_np = np.zeros((final_h, final_w, 3), dtype=np.float32)
            weight_map = np.zeros((final_h, final_w, 1), dtype=np.float32)

            tiles_np = image_batch.cpu().numpy()

            # Iterate, apply mask, and accumulate values and weights
            tile_index = 0
            for r in range(rows):
                for c in range(cols):
                    y_start = r * core_upscaled_h
                    x_start = c * core_upscaled_w

                    final_image_np[y_start:y_start + incoming_h, x_start:x_start + incoming_w] += tiles_np[
                                                                                                      tile_index] * feather_mask
                    weight_map[y_start:y_start + incoming_h, x_start:x_start + incoming_w] += feather_mask
                    tile_index += 1

            # Normalize the image by dividing by the accumulated weights
            weight_map[weight_map == 0] = 1.0  # Avoid division by zero
            final_image_np /= weight_map

            # Crop the final image to the target size, removing the outer feathered edges
            target_w = cols * core_upscaled_w
            target_h = rows * core_upscaled_h
            final_image_np = final_image_np[border_h:border_h + target_h, border_w:border_w + target_w]

            # Clip and convert to tensor
            final_image_np = np.clip(final_image_np, 0.0, 1.0)
            output_tensor = torch.from_numpy(final_image_np).unsqueeze(0)

        return (output_tensor,)