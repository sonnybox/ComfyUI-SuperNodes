import numpy as np
from PIL import Image
import torch


class ImageMaskCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (
                    "IMAGE",
                    {"tooltip": "The source image to be cropped."},
                ),
                "mask": (
                    "MASK",
                    {
                        "tooltip": "The binary mask defining the region of interest to crop."
                    },
                ),
                "padding": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4096,
                        "step": 1,
                        "tooltip": "Amount of padding (in pixels) to add around the mask bounding box.",
                    },
                ),
                "multiple_of": (
                    "INT",
                    {
                        "default": 16,
                        "min": 1,
                        "max": 512,
                        "step": 1,
                        "tooltip": "Ensure the crop dimensions are a multiple of this value (critical for UNet based processing).",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "CROP_INFO")
    RETURN_NAMES = ("cropped_image", "cropped_mask", "uncrop_info")
    OUTPUT_TOOLTIPS = (
        "The cropped image region.",
        "The cropped mask region.",
        "Metadata containing coordinates and original size, required for restoration.",
    )
    FUNCTION = "crop"
    CATEGORY = "SuperNodes"
    DESCRIPTION = "Crops an image based on a mask's bounding box, with optional padding and dimension constraints."

    def crop(self, image, mask, padding, multiple_of):
        # Handle empty mask
        if mask.max() == 0:
            empty_info = {
                "x": 0,
                "y": 0,
                "w": image.shape[2],
                "h": image.shape[1],
                "original_size": (image.shape[1], image.shape[2]),
                "mask_patch": None,
            }
            return (image, mask, empty_info)

        # 1. Binarize Mask (Round to nearest 0 or 1 based on 0.5 threshold)
        mask_binary = (mask > 0.5).float()

        # 2. Calculate Bounding Box
        mask_flat = (
            torch.max(mask_binary, dim=0).values
            if mask.dim() > 2
            else mask_binary
        )
        non_zero = torch.nonzero(mask_flat)

        if non_zero.numel() == 0:
            min_y, min_x = 0, 0
            max_y, max_x = image.shape[1], image.shape[2]
        else:
            min_y = torch.min(non_zero[:, 0]).item()
            max_y = torch.max(non_zero[:, 0]).item() + 1
            min_x = torch.min(non_zero[:, 1]).item()
            max_x = torch.max(non_zero[:, 1]).item() + 1

        # 3. Apply Padding
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(image.shape[2], max_x + padding)
        max_y = min(image.shape[1], max_y + padding)

        # 4. Apply 'multiple_of' constraint
        width = max_x - min_x
        height = max_y - min_y

        if width % multiple_of != 0:
            target_width = ((width // multiple_of) + 1) * multiple_of
            diff = target_width - width
            pad_l = diff // 2
            pad_r = diff - pad_l

            if min_x - pad_l < 0:
                min_x = 0
                max_x = min(image.shape[2], min_x + target_width)
            elif max_x + pad_r > image.shape[2]:
                max_x = image.shape[2]
                min_x = max(0, max_x - target_width)
            else:
                min_x -= pad_l
                max_x += pad_r

        if height % multiple_of != 0:
            target_height = ((height // multiple_of) + 1) * multiple_of
            diff = target_height - height
            pad_t = diff // 2
            pad_b = diff - pad_t

            if min_y - pad_t < 0:
                min_y = 0
                max_y = min(image.shape[1], min_y + target_height)
            elif max_y + pad_b > image.shape[1]:
                max_y = image.shape[1]
                min_y = max(0, max_y - target_height)
            else:
                min_y -= pad_t
                max_y += pad_b

        # Final Crop Coords
        crop_x, crop_y = min_x, min_y
        crop_w = max_x - min_x
        crop_h = max_y - min_y

        # 5. Crop Image and Mask
        cropped_image = image[
            :, crop_y : crop_y + crop_h, crop_x : crop_x + crop_w, :
        ]
        cropped_mask = mask[
            :, crop_y : crop_y + crop_h, crop_x : crop_x + crop_w
        ]

        # 6. Prepare Uncrop Info
        uncrop_info = {
            "x": crop_x,
            "y": crop_y,
            "w": crop_w,
            "h": crop_h,
            "original_size": (image.shape[1], image.shape[2]),  # H, W
            "mask_patch": cropped_mask,  # Store original mask crop for restoration
        }

        return (cropped_image, cropped_mask, uncrop_info)


class RestoreMaskCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": (
                    "IMAGE",
                    {"tooltip": "The original source image before cropping."},
                ),
                "cropped_image": (
                    "IMAGE",
                    {
                        "tooltip": "The processed cropped image to be pasted back."
                    },
                ),
                "crop_info": (
                    "CROP_INFO",
                    {
                        "tooltip": "The crop metadata generated by the ImageMaskCrop node."
                    },
                ),
                "strategy": (
                    ["scale_cropped", "scale_original"],
                    {
                        "tooltip": "How to handle size mismatches. 'scale_cropped' resizes the input crop to the original hole. 'scale_original' resizes the background to match the new crop scale."
                    },
                ),
                "scale_method": (
                    ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"],
                    {
                        "tooltip": "The interpolation method used for resizing images."
                    },
                ),
            },
            "optional": {
                "override_mask": (
                    "MASK",
                    {
                        "tooltip": "An optional mask to use for blending, overriding the original crop mask."
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_TOOLTIPS = (
        "The composite image with the crop restored to its original location.",
    )
    FUNCTION = "restore"
    CATEGORY = "SuperNodes"
    DESCRIPTION = "Restores a cropped image back into the original image context, handling scaling and blending."

    def restore(
        self,
        original_image,
        cropped_image,
        crop_info,
        strategy,
        scale_method,
        override_mask=None,
    ):
        def resize_tensor(tensor, width, height, method, is_mask=False):
            """
            Optimized resizing using torch.interpolate where possible,
            falling back to PIL for unsupported methods (Lanczos).
            """
            # Use PIL for Lanczos
            if method == "lanczos":
                return pil_resize_fallback(
                    tensor, width, height, method, is_mask
                )

            # Map method names to torch modes
            # "nearest-exact" -> "nearest"
            mode = method if method != "nearest-exact" else "nearest"

            # Prepare tensors for Torch Interpolation
            # Images: [B, H, W, C] -> [B, C, H, W]
            # Masks:  [B, H, W]    -> [B, 1, H, W]
            if is_mask:
                if tensor.dim() == 3:
                    t = tensor.unsqueeze(1)
                else:
                    t = tensor
            else:
                t = tensor.permute(0, 3, 1, 2)

            # Perform Interpolation
            # align_corners=False is standard for bilinear/bicubic in Comfy utils
            if mode in ["bilinear", "bicubic"]:
                t = torch.nn.functional.interpolate(
                    t, size=(height, width), mode=mode, align_corners=False
                )
            else:
                # area / nearest
                t = torch.nn.functional.interpolate(
                    t, size=(height, width), mode=mode
                )

            # Restore Dimensions
            if is_mask:
                # [B, 1, H, W] -> [B, H, W]
                return t.squeeze(1)
            else:
                # [B, C, H, W] -> [B, H, W, C]
                return t.permute(0, 2, 3, 1)

        def pil_resize_fallback(
            tensor_data, width, height, method, is_mask=False
        ):
            pil_method = Image.Resampling.LANCZOS
            if method == "nearest-exact":
                pil_method = Image.Resampling.NEAREST
            elif method == "area":
                pil_method = Image.Resampling.BOX
            elif method == "bicubic":
                pil_method = Image.Resampling.BICUBIC
            elif method == "bilinear":
                pil_method = Image.Resampling.BILINEAR

            results = []
            if is_mask:
                # Handle Mask [B, H, W]
                for m in tensor_data:
                    i = Image.fromarray(
                        (m.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8),
                        mode="L",
                    )
                    i = i.resize((width, height), resample=pil_method)
                    results.append(
                        torch.from_numpy(np.array(i).astype(np.float32) / 255.0)
                    )
            else:
                # Handle Image [B, H, W, C]
                for img in tensor_data:
                    i = Image.fromarray(
                        (img.cpu().numpy() * 255.0)
                        .clip(0, 255)
                        .astype(np.uint8)
                    )
                    i = i.resize((width, height), resample=pil_method)
                    results.append(
                        torch.from_numpy(np.array(i).astype(np.float32) / 255.0)
                    )

            return torch.stack(results)

        # Unpack info
        orig_x, orig_y = crop_info["x"], crop_info["y"]
        orig_w, orig_h = crop_info["w"], crop_info["h"]

        # Determine the blending mask
        if override_mask is not None:
            composite_mask = override_mask
        else:
            composite_mask = crop_info.get(
                "mask_patch",
                torch.ones((1, cropped_image.shape[1], cropped_image.shape[2])),
            )

        # Ensure mask batch size matches image
        if composite_mask.shape[0] < cropped_image.shape[0]:
            composite_mask = composite_mask.repeat(cropped_image.shape[0], 1, 1)

        # --- Logic Branching ---
        if strategy == "scale_cropped":
            # Resize the cropped_image to fit the original hole
            if (cropped_image.shape[2] != orig_w) or (
                cropped_image.shape[1] != orig_h
            ):
                cropped_image = resize_tensor(
                    cropped_image, orig_w, orig_h, scale_method, is_mask=False
                )
                composite_mask = resize_tensor(
                    composite_mask, orig_w, orig_h, scale_method, is_mask=True
                )

            out_image = original_image.clone()
            paste_x, paste_y = orig_x, orig_y

        elif strategy == "scale_original":
            # Calculate scale factor based on the difference between the NEW crop width and the OLD crop width
            # This causes the original image to scale proportionally to how the crop was scaled
            scale_factor = cropped_image.shape[2] / orig_w

            # New dimensions for the full original image
            new_orig_w = int(crop_info["original_size"][1] * scale_factor)
            new_orig_h = int(crop_info["original_size"][0] * scale_factor)

            # 1. Resize the background (original) image
            out_image = resize_tensor(
                original_image,
                new_orig_w,
                new_orig_h,
                scale_method,
                is_mask=False,
            )

            # 2. Resize the blending mask to match the NEW crop size
            if (composite_mask.shape[2] != cropped_image.shape[2]) or (
                composite_mask.shape[1] != cropped_image.shape[1]
            ):
                composite_mask = resize_tensor(
                    composite_mask,
                    cropped_image.shape[2],
                    cropped_image.shape[1],
                    scale_method,
                    is_mask=True,
                )

            # 3. Calculate new paste coordinates
            # Use round() instead of int() to minimize misalignment
            paste_x = int(round(orig_x * scale_factor))
            paste_y = int(round(orig_y * scale_factor))

        # --- Compositing ---
        # Get dimensions
        h_ins, w_ins = cropped_image.shape[1], cropped_image.shape[2]
        max_h, max_w = out_image.shape[1], out_image.shape[2]

        # Calculate valid intersection (clipping)
        # Coordinates in out_image
        y1 = max(0, paste_y)
        x1 = max(0, paste_x)
        y2 = min(max_h, paste_y + h_ins)
        x2 = min(max_w, paste_x + w_ins)

        # If the paste area is entirely out of bounds, return early
        if y2 <= y1 or x2 <= x1:
            return (out_image,)

        # Coordinates in cropped_image (source)
        # We need to offset if we clipped top/left
        sy1 = y1 - paste_y
        sx1 = x1 - paste_x
        # Width/Height of the valid region
        region_h = y2 - y1
        region_w = x2 - x1
        sy2 = sy1 + region_h
        sx2 = sx1 + region_w

        # Extract regions
        dest_region = out_image[:, y1:y2, x1:x2, :]
        src_region = cropped_image[:, sy1:sy2, sx1:sx2, :]
        mask_region = composite_mask[:, sy1:sy2, sx1:sx2]

        # Expand mask for broadcasting: [B, H, W] -> [B, H, W, 1]
        mask_region = mask_region.unsqueeze(-1)

        # Blend
        blended = src_region * mask_region + dest_region * (1.0 - mask_region)

        # Apply back to output
        out_image[:, y1:y2, x1:x2, :] = blended

        return (out_image,)
