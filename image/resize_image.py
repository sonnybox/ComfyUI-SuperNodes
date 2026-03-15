from typing import Any, Dict, Optional, Tuple

import comfy.utils
import torch


class SuperResizeImage:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "width": (
                    "INT",
                    {"default": 512, "min": 1, "max": 65536, "step": 1},
                ),
                "height": (
                    "INT",
                    {"default": 512, "min": 1, "max": 65536, "step": 1},
                ),
                "crop": (
                    [
                        "center",
                        "disabled",
                        "left",
                        "right",
                        "top",
                        "bottom",
                        "top-left",
                        "top-right",
                        "bottom-left",
                        "bottom-right",
                    ],
                ),
                "cover": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "False: crops a literal chunk of pixels from the source. True: Resizes the image to fit the dimensions while preserving aspect ratio, then crops.",
                    },
                ),
                "upscale_method": (
                    ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"],
                ),
                "multiple_of": (
                    "INT",
                    {"default": 16, "min": 1, "max": 512, "step": 1},
                ),
            },
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "resize"
    CATEGORY = "SuperNodes/Image"

    def resize(
        self,
        width: int,
        height: int,
        crop: str,
        upscale_method: str,
        multiple_of: int,
        cover: bool,
        image: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if image is None and mask is None:
            raise ValueError(
                "SuperResizeImage: You must provide either an image or a mask."
            )

        def make_multiple_round(value: float, m: int) -> int:
            return max(m, round(value / m) * m)

        def make_multiple_floor(value: float, m: int) -> int:
            return max(m, (int(value) // m) * m)

        B, H_orig, W_orig = 0, 0, 0

        # 1. Determine original dims
        if image is not None:
            B, H_orig, W_orig, _ = image.shape
        elif mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)  # -> (1,H,W)
            elif mask.dim() == 4 and mask.shape[1] == 1:
                # If something upstream handed (B,1,H,W), normalize to (B,H,W)
                mask = mask.squeeze(1)
            B, H_orig, W_orig = mask.shape[0], mask.shape[1], mask.shape[2]

        # 2. Target dims (for cover/stretch modes)
        target_w = make_multiple_round(width, multiple_of)
        target_h = make_multiple_round(height, multiple_of)

        # 3. Resizing helper
        def process_tensor(
            tensor_in: torch.Tensor,
            w: int,
            h: int,
            method: str,
            is_mask: bool = False,
        ) -> torch.Tensor:
            if is_mask:
                # Force nearest for masks to preserve edges (and avoid “mask drift”)
                method = "nearest-exact"
                samples = tensor_in.unsqueeze(1)  # (B,1,H,W)
            else:
                samples = tensor_in.movedim(-1, 1)  # (B,C,H,W)

            out = comfy.utils.common_upscale(samples, w, h, method, "disabled")

            if is_mask:
                out = out.squeeze(1)  # (B,H,W)
                return out.clamp(0.0, 1.0)
            else:
                return out.movedim(1, -1)  # (B,H,W,C)

        # Stretch
        if crop == "disabled":
            out_image = (
                process_tensor(image, target_w, target_h, upscale_method, False)
                if image is not None
                else torch.zeros(
                    (B, target_h, target_w, 3),
                    device=mask.device if mask is not None else "cpu",
                )
            )
            out_mask = (
                process_tensor(mask, target_w, target_h, upscale_method, True)
                if mask is not None
                else torch.zeros(
                    (B, target_h, target_w),
                    device=image.device if image is not None else "cpu",
                )
            )
            return (out_image, out_mask)

        # Cropping (cover or viewport)
        if cover:
            # Aspect-cover resize to at least target size, then crop to target
            scale = max(target_w / W_orig, target_h / H_orig)
            new_w = max(1, int(round(W_orig * scale)))
            new_h = max(1, int(round(H_orig * scale)))

            working_image = (
                process_tensor(image, new_w, new_h, upscale_method, False)
                if image is not None
                else None
            )
            working_mask = (
                process_tensor(mask, new_w, new_h, upscale_method, True)
                if mask is not None
                else None
            )

            H_curr, W_curr = new_h, new_w
            final_w, final_h = target_w, target_h
        else:
            # Viewport crop: literal window, anchored by crop position.
            # Don't round up to multiple here (can exceed source)
            final_w = int(width)
            final_h = int(height)

            # If viewport larger than source, clamp it
            final_w = min(final_w, W_orig)
            final_h = min(final_h, H_orig)

            # Floor to multiple so we never exceed source again
            final_w = min(W_orig, make_multiple_floor(final_w, multiple_of))
            final_h = min(H_orig, make_multiple_floor(final_h, multiple_of))

            working_image = image
            working_mask = mask
            H_curr, W_curr = H_orig, W_orig

        # Anchor / crop coordinates (clamped)
        max_y1 = max(0, H_curr - final_h)
        max_x1 = max(0, W_curr - final_w)

        if "top" in crop:
            y1 = 0
        elif "bottom" in crop:
            y1 = max_y1
        else:
            y1 = max_y1 // 2

        if "left" in crop:
            x1 = 0
        elif "right" in crop:
            x1 = max_x1
        else:
            x1 = max_x1 // 2

        # Clamp just in case
        y1 = max(0, min(y1, max_y1))
        x1 = max(0, min(x1, max_x1))

        y2 = y1 + final_h
        x2 = x1 + final_w

        # Apply crop
        if working_image is not None:
            out_image = working_image[:, y1:y2, x1:x2, :]
        else:
            device = working_mask.device if working_mask is not None else "cpu"
            out_image = torch.zeros((B, final_h, final_w, 3), device=device)

        if working_mask is not None:
            out_mask = working_mask[:, y1:y2, x1:x2]
        else:
            device = (
                working_image.device if working_image is not None else "cpu"
            )
            out_mask = torch.zeros((B, final_h, final_w), device=device)

        return (out_image, out_mask)


NODE_CLASS_MAPPINGS = {"SuperResizeImage": SuperResizeImage}

NODE_DISPLAY_NAME_MAPPINGS = {"SuperResizeImage": "🐧 Resize Image & Mask"}
