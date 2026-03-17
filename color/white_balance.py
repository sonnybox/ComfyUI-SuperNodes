from comfy_api.latest import io

from .utils import _apply_white_balance_cat


class SuperWhiteBalanceCAT(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SuperWhiteBalanceCAT",
            display_name="🐧 Adjust White Balance",
            category="SuperNodes/Color",
            inputs=[
                io.Image.Input("image"),
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
    def execute(cls, image, temperature_k=6500, tint=0.0) -> io.NodeOutput:
        out = _apply_white_balance_cat(image, float(temperature_k), float(tint))
        return io.NodeOutput(out)


NODE = [SuperWhiteBalanceCAT]
