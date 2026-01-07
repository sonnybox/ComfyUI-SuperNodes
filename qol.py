import comfy.utils  # type: ignore


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
                        "tooltip": "The target length for the dimension specified by dimension_mode.",
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
                "image": ("IMAGE",),
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
                        "disabled",
                        "center",
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
                        "default": False,
                        "tooltip": "False (Viewport): Slices a literal chunk of pixels from the source. True (Cover): Resizes the image to fit the dimensions while preserving aspect ratio, then crops.",
                    },
                ),
                "upscale_method": (
                    ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"],
                ),
                "multiple_of": (
                    "INT",
                    {"default": 16, "min": 1, "max": 512, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "resize"
    CATEGORY = "SuperNodes"

    def resize(
        self, image, width, height, crop, upscale_method, multiple_of, cover
    ):
        def make_multiple(value, m):
            return max(m, round(value / m) * m)

        B, H_orig, W_orig, C = image.shape

        # Mode 1: Stretch (No Aspect Ratio)
        if crop == "disabled":
            target_w = make_multiple(width, multiple_of)
            target_h = make_multiple(height, multiple_of)
            samples = image.movedim(-1, 1)
            out = comfy.utils.common_upscale(
                samples, target_w, target_h, upscale_method, "disabled"
            )
            return (out.movedim(1, -1),)

        # Mode 2: Aspect-Fit then Crop (Cover)
        if cover:
            target_w = make_multiple(width, multiple_of)
            target_h = make_multiple(height, multiple_of)

            # Calculate scale to ensure target is covered
            scale = max(target_w / W_orig, target_h / H_orig)
            new_w = round(W_orig * scale)
            new_h = round(H_orig * scale)

            samples = image.movedim(-1, 1)
            resized = comfy.utils.common_upscale(
                samples, new_w, new_h, upscale_method, "disabled"
            )
            working_image = resized.movedim(1, -1)
            H_curr, W_curr = new_h, new_w
            final_w, final_h = target_w, target_h

        # Mode 3: Direct Viewport Crop
        else:
            final_w = width
            final_h = height

            # If viewport is larger than image, scale the requested window down to fit
            if final_w > W_orig or final_h > H_orig:
                scale = min(W_orig / final_w, H_orig / final_h)
                final_w = int(final_w * scale)
                final_h = int(final_h * scale)

            final_w = make_multiple(final_w, multiple_of)
            final_h = make_multiple(final_h, multiple_of)

            working_image = image
            H_curr, W_curr = H_orig, W_orig

        # Shared Anchor Coordinate Logic
        # Vertical
        if "top" in crop:
            y1 = 0
        elif "bottom" in crop:
            y1 = H_curr - final_h
        else:  # center
            y1 = (H_curr - final_h) // 2

        # Horizontal
        if "left" in crop:
            x1 = 0
        elif "right" in crop:
            x1 = W_curr - final_w
        else:  # center
            x1 = (W_curr - final_w) // 2

        y2 = y1 + final_h
        x2 = x1 + final_w

        return (working_image[:, y1:y2, x1:x2, :],)


NODE_CLASS_MAPPINGS = {"SuperResizeImage": SuperResizeImage}
NODE_DISPLAY_NAME_MAPPINGS = {"SuperResizeImage": "Resize Image (Super)"}
