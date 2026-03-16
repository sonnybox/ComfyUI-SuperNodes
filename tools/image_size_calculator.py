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
                    options=["max", "min"],
                    tooltip="Determines if the 'size' input applies to the largest (max) or smallest (min) dimension.",
                ),
                io.Int.Input(
                    "size",
                    default=1024,
                    min=1,
                    max=32768,
                    step=1,
                    tooltip="The target length for the dimension specified by dimension mode.",
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
            ],
        )

    @classmethod
    def execute(
        cls, aspect_w, aspect_h, mode, size, multiple_of
    ) -> io.NodeOutput:
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

        return io.NodeOutput(final_w, final_h)


V3_NODES = [ImageSizeCalculator]
