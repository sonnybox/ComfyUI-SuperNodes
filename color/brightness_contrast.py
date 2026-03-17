from comfy_api.latest import io

from .utils import _apply_brightness_contrast_gamma


class SuperBrightnessContrast(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SuperBrightnessContrast",
            display_name="🐧 Adjust Brightness Contrast Gamma",
            category="SuperNodes/Color",
            inputs=[
                io.Image.Input("image"),
                io.Float.Input(
                    "brightness", default=1.0, min=0.0, max=4.0, step=0.01
                ),
                io.Float.Input(
                    "contrast", default=1.0, min=0.0, max=4.0, step=0.01
                ),
                io.Float.Input(
                    "gamma", default=1.0, min=0.05, max=4.0, step=0.01
                ),
            ],
            outputs=[
                io.Image.Output(display_name="IMAGE"),
            ],
        )

    @classmethod
    def execute(
        cls, image, brightness=1.0, contrast=1.0, gamma=1.0
    ) -> io.NodeOutput:
        return io.NodeOutput(
            _apply_brightness_contrast_gamma(image, brightness, contrast, gamma)
        )


NODE = [SuperBrightnessContrast]
