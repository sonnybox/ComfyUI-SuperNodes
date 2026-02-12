import comfy.model_management  # type: ignore
import comfy.utils  # type: ignore
import torch


class SetReserveVRAM:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "any": ("*",),  # Wildcard input: accepts any type
                "reserved_gb": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1024.0, "step": 0.1},
                ),
            }
        }

    RETURN_TYPES = ("*",)
    FUNCTION = "set_vram"
    CATEGORY = "SuperNodes"
    DESCRIPTION = "Set --reserve-vram dynamically anywhere in a workflow."

    def set_vram(self, any, reserved_gb):
        comfy.model_management.EXTRA_RESERVED_VRAM = (
            reserved_gb * 1024 * 1024 * 1024
        )
        return (any,)


class GetCommonAspectRatio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": 1, "max": 65536}),
                "height": ("INT", {"default": 1024, "min": 1, "max": 65536}),
                # We use the ratio string directly as the key.
                # ComfyUI will display this text next to the toggle.
                "1:1": ("BOOLEAN", {"default": True}),
                "4:3": ("BOOLEAN", {"default": True}),
                "3:2": ("BOOLEAN", {"default": True}),
                "5:4": ("BOOLEAN", {"default": True}),
                "16:9": ("BOOLEAN", {"default": True}),
                "16:10": ("BOOLEAN", {"default": True}),
                "21:9": ("BOOLEAN", {"default": True}),
                "2:1": ("BOOLEAN", {"default": True}),
                "1.85:1": ("BOOLEAN", {"default": True}),
                "2.39:1": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("aspect_w", "aspect_h")
    FUNCTION = "get_ratio"
    CATEGORY = "SuperNodes"

    def get_ratio(self, width, height, **kwargs):
        # 1. Map the string keys to their mathematical ratios
        ratios = {
            "1:1": (1, 1),
            "4:3": (4, 3),
            "3:2": (3, 2),
            "5:4": (5, 4),
            "16:9": (16, 9),
            "16:10": (16, 10),
            "21:9": (21, 9),
            "2:1": (2, 1),
            "1.85:1": (37, 20),
            "2.39:1": (239, 100),
        }

        # 2. Filter enabled ratios
        enabled_ratios = {}
        for key, value in ratios.items():
            # kwargs[key] will be True/False based on the toggle
            if kwargs.get(key, True):
                enabled_ratios[key] = value

        # Fallback to 16:9 if everything is disabled
        if not enabled_ratios:
            enabled_ratios = {"16:9": (16, 9)}

        # 3. Calculate Input Aspect Ratio
        is_portrait = height > width
        if is_portrait:
            input_float = height / width
        else:
            input_float = width / height

        # 4. Find the Closest Match
        best_match_name = None
        min_diff = float("inf")

        for name, (rw, rh) in enabled_ratios.items():
            target_float = max(rw, rh) / min(rw, rh)
            diff = abs(input_float - target_float)

            if diff < min_diff:
                min_diff = diff
                best_match_name = name

        # 5. Retrieve the winner
        target_w, target_h = enabled_ratios[best_match_name]

        # 6. Correct for Orientation
        if is_portrait:
            final_w = min(target_w, target_h)
            final_h = max(target_w, target_h)
        else:
            final_w = max(target_w, target_h)
            final_h = min(target_w, target_h)

        return (final_w, final_h)


class ImageSizeCalculator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aspect_w": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 1024,
                        "step": 1,
                        "tooltip": "The width ratio of the desired aspect ratio (e.g., 16 for 16:9).",
                    },
                ),
                "aspect_h": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 1024,
                        "step": 1,
                        "tooltip": "The height ratio of the desired aspect ratio (e.g., 9 for 16:9).",
                    },
                ),
                "mode": (
                    ["max", "min"],
                    {
                        "tooltip": "Determines if the 'size' input applies to the largest (max) or smallest (min) dimension."
                    },
                ),
                "size": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 1,
                        "max": 32768,
                        "step": 1,
                        "tooltip": "The target length for the dimension specified by dimension mode.",
                    },
                ),
                "multiple_of": (
                    "INT",
                    {
                        "default": 16,
                        "min": 1,
                        "max": 1024,
                        "step": 1,
                        "tooltip": "The final dimensions will be rounded to the nearest multiple of this value.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    OUTPUT_TOOLTIPS = ("The calculated width.", "The calculated height.")
    FUNCTION = "calculate"

    CATEGORY = "SuperNodes/Utils"
    DESCRIPTION = "Calculates width and height based on an aspect ratio and a target dimension size, rounding to a specific multiple."

    def calculate(self, aspect_w, aspect_h, mode, size, multiple_of):
        # Calculate aspect ratio
        ratio = aspect_w / aspect_h

        target_w = 0.0
        target_h = 0.0

        if mode == "max":
            if aspect_w >= aspect_h:
                # Width is the longest side
                target_w = size
                target_h = size / ratio
            else:
                # Height is the longest side
                target_h = size
                target_w = size * ratio
        else:  # mode == "min"
            if aspect_w <= aspect_h:
                # Width is the shortest side
                target_w = size
                target_h = size / ratio
            else:
                # Height is the shortest side
                target_h = size
                target_w = size * ratio

        # Round to nearest multiple
        final_w = int(round(target_w / multiple_of)) * multiple_of
        final_h = int(round(target_h / multiple_of)) * multiple_of

        # Ensure we don't return 0 if the size is very small relative to multiple_of
        final_w = max(multiple_of, final_w)
        final_h = max(multiple_of, final_h)

        return (final_w, final_h)


class SuperResizeImage:
    @classmethod
    def INPUT_TYPES(cls):
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
    CATEGORY = "SuperNodes"

    def resize(
        self,
        width,
        height,
        crop,
        upscale_method,
        multiple_of,
        cover,
        image=None,
        mask=None,
    ):
        if image is None and mask is None:
            raise ValueError(
                "SuperResizeImage: You must provide either an image or a mask."
            )

        def make_multiple_round(value, m):
            return max(m, round(value / m) * m)

        def make_multiple_floor(value, m):
            return max(m, (int(value) // m) * m)

        # 1) Determine original dims
        if image is not None:
            B, H_orig, W_orig, C = image.shape
        else:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)  # -> (1,H,W)
            elif mask.dim() == 4 and mask.shape[1] == 1:
                # If something upstream handed (B,1,H,W), normalize to (B,H,W)
                mask = mask.squeeze(1)
            B, H_orig, W_orig = mask.shape[0], mask.shape[1], mask.shape[2]

        # 2) Target dims (for cover/stretch modes)
        target_w = make_multiple_round(width, multiple_of)
        target_h = make_multiple_round(height, multiple_of)

        # 3) Resizing helper
        def process_tensor(tensor_in, w, h, method, is_mask=False):
            if tensor_in is None:
                return None

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

        # --- MODE 1: Stretch (No Aspect Ratio) ---
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

        # --- MODE 2/3: Cropping modes ---
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
            # IMPORTANT: never round UP to multiple here (can exceed source)
            final_w = int(width)
            final_h = int(height)

            # If viewport larger than source, clamp it (don’t scale the source)
            final_w = min(final_w, W_orig)
            final_h = min(final_h, H_orig)

            # Floor to multiple so we never exceed source again
            final_w = min(W_orig, make_multiple_floor(final_w, multiple_of))
            final_h = min(H_orig, make_multiple_floor(final_h, multiple_of))

            working_image = image
            working_mask = mask
            H_curr, W_curr = H_orig, W_orig

        # --- Anchor / crop coordinates (clamped) ---
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

        # --- Apply crop ---
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


class FaceBBoxToMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "face_bboxes": ("BBOX",),
                "images": ("IMAGE",),
                "extend_up_percent": (
                    "INT",
                    {"default": 0, "min": -100, "max": 500, "step": 1},
                ),
                "extend_down_percent": (
                    "INT",
                    {"default": 0, "min": -100, "max": 500, "step": 1},
                ),
                "extend_left_percent": (
                    "INT",
                    {"default": 0, "min": -100, "max": 500, "step": 1},
                ),
                "extend_right_percent": (
                    "INT",
                    {"default": 0, "min": -100, "max": 500, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASK",)
    FUNCTION = "process"
    CATEGORY = "SuperNodes/Utils"
    DESCRIPTION = "Converts face bounding boxes into a mask batch, with percent-based directional extension."

    def process(
        self,
        face_bboxes,
        images,
        extend_up_percent,
        extend_down_percent,
        extend_left_percent,
        extend_right_percent,
    ):
        batch_size, height, width, _ = images.shape
        masks = torch.zeros((batch_size, height, width), dtype=torch.float32)

        for i in range(min(batch_size, len(face_bboxes))):
            bbox = face_bboxes[i]
            if bbox is None:
                continue

            x1, y1, x2, y2 = bbox
            bw = float(x2 - x1)
            bh = float(y2 - y1)

            if bw <= 1 or bh <= 1:
                continue

            up_px = int(round(bh * (extend_up_percent / 100.0)))
            down_px = int(round(bh * (extend_down_percent / 100.0)))
            left_px = int(round(bw * (extend_left_percent / 100.0)))
            right_px = int(round(bw * (extend_right_percent / 100.0)))

            x1 = int(round(x1)) - left_px
            x2 = int(round(x2)) + right_px
            y1 = int(round(y1)) - up_px
            y2 = int(round(y2)) + down_px

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)

            if x2 > x1 and y2 > y1:
                masks[i, y1:y2, x1:x2] = 1.0

        return (masks,)


class SuperStopExecution:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "message": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "Message.",
                    },
                ),
                "trigger": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "If True, execution halts. If False, nothing happens.",
                        "forceInput": True,
                    },
                ),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "halt_execution"
    OUTPUT_NODE = True
    CATEGORY = "SuperNodes/debug"

    def halt_execution(self, message, trigger):
        if trigger:
            alert = str(message)
            raise Exception(f"{alert}")

        return ()
