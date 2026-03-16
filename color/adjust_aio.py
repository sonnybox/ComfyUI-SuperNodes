from comfy_api.latest import io

from .utils import (
    _apply_brightness_contrast_gamma,
    _apply_saturation_hue,
    _apply_white_balance_cat,
)


class SuperColorAdjustAllInOne(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SuperColorAdjustAllInOne",
            display_name="🐧 Adjust Color AIO",
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
                io.Float.Input(
                    "saturation", default=1.0, min=0.0, max=4.0, step=0.01
                ),
                io.Float.Input(
                    "hue_degrees", default=0.0, min=-180.0, max=180.0, step=0.5
                ),
                io.Int.Input(
                    "temperature_k", default=6500, min=1650, max=25000, step=50
                ),
                io.Float.Input(
                    "tint", default=0.0, min=-1.0, max=1.0, step=0.01
                ),
            ],
            outputs=[
                io.Image.Output(display_name="IMAGE"),
            ],
        )

    @classmethod
    def execute(
        cls,
        image,
        brightness=1.0,
        contrast=1.0,
        gamma=1.0,
        saturation=1.0,
        hue_degrees=0.0,
        temperature_k=6500,
        tint=0.0,
    ) -> io.NodeOutput:
        out = _apply_brightness_contrast_gamma(
            image, brightness, contrast, gamma
        )
        out = _apply_saturation_hue(out, saturation, hue_degrees)
        out = _apply_white_balance_cat(out, float(temperature_k), float(tint))
        return io.NodeOutput(out)


V3_NODES = [SuperColorAdjustAllInOne]
