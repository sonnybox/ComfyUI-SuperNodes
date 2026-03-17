from comfy_api.latest import io

from .utils import _apply_saturation_hue


class SuperHueSaturation(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SuperHueSaturation",
            display_name="🐧 Adjust Hue Saturation",
            category="SuperNodes/Color",
            inputs=[
                io.Image.Input("image"),
                io.Float.Input(
                    "saturation", default=1.0, min=0.0, max=4.0, step=0.01
                ),
                io.Float.Input(
                    "hue_degrees", default=0.0, min=-180.0, max=180.0, step=0.5
                ),
            ],
            outputs=[
                io.Image.Output(display_name="IMAGE"),
            ],
        )

    @classmethod
    def execute(cls, image, saturation=1.0, hue_degrees=0.0) -> io.NodeOutput:
        return io.NodeOutput(
            _apply_saturation_hue(image, saturation, hue_degrees)
        )


NODE = [SuperHueSaturation]
