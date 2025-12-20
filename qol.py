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

    CATEGORY = "SuperNodes"
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
