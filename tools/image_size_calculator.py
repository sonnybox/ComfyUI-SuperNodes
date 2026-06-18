import math

from comfy_api.latest import io


class ImageSizeCalculator(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ImageSizeCalculator",
            display_name="🐧 Image Size Calculator",
            category="SuperNodes/Tools",
            description="Calculates width and height based on an aspect ratio and a target dimension size, rounding to a specific multiple.",
            inputs=[
                io.Int.Input(
                    "aspect_w",
                    default=1,
                    min=1,
                    max=1024,
                    step=1,
                    tooltip="The width ratio of the desired aspect ratio (e.g., 16 for 16:9).",
                ),
                io.Int.Input(
                    "aspect_h",
                    default=1,
                    min=1,
                    max=1024,
                    step=1,
                    tooltip="The height ratio of the desired aspect ratio (e.g., 9 for 16:9).",
                ),
                io.Combo.Input(
                    "mode",
                    options=["max_size", "min_size", "megapixels"],
                    tooltip="Determines if the size applies to max/min dimension, or if target is total megapixels.",
                ),
                io.Int.Input(
                    "size",
                    default=1024,
                    min=1,
                    max=32768,
                    step=1,
                    tooltip="The target length for the dimension specified by dimension mode.",
                ),
                io.Float.Input(
                    "megapixels",
                    default=1.00,
                    min=0.01,
                    max=1024.0,
                    step=0.01,
                    tooltip="The target total megapixels when mode is 'megapixels'.",
                ),
                io.Int.Input(
                    "multiple_of",
                    default=16,
                    min=1,
                    max=1024,
                    step=1,
                    tooltip="The final dimensions will be rounded to the nearest multiple of this value.",
                ),
            ],
            outputs=[
                io.Int.Output(
                    display_name="width", tooltip="The calculated width."
                ),
                io.Int.Output(
                    display_name="height", tooltip="The calculated height."
                ),
                io.Int.Output(
                    display_name="aspect_w", tooltip="Passthrough width aspect."
                ),
                io.Int.Output(
                    display_name="aspect_h",
                    tooltip="Passthrough height aspect.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls, aspect_w, aspect_h, mode, size, megapixels, multiple_of
    ) -> io.NodeOutput:
        # Calculate aspect ratio
        ratio = aspect_w / aspect_h

        target_w = 0.0
        target_h = 0.0

        if mode == "max_size":
            if aspect_w >= aspect_h:
                # Width is the longest side
                target_w = size
                target_h = size / ratio
            else:
                # Height is the longest side
                target_h = size
                target_w = size * ratio
        elif mode == "min_size":
            if aspect_w <= aspect_h:
                # Width is the shortest side
                target_w = size
                target_h = size / ratio
            else:
                # Height is the shortest side
                target_h = size
                target_w = size * ratio
        elif mode == "megapixels":
            target_pixels = megapixels * 1024 * 1024
            target_h = math.sqrt(target_pixels / ratio)
            target_w = target_h * ratio

        # Round to nearest multiple
        final_w = int(round(target_w / multiple_of)) * multiple_of
        final_h = int(round(target_h / multiple_of)) * multiple_of

        # Ensure we don't return 0 if the size is very small relative to multiple_of
        final_w = max(multiple_of, final_w)
        final_h = max(multiple_of, final_h)

        return io.NodeOutput(final_w, final_h, aspect_w, aspect_h)


NODE = [ImageSizeCalculator]
