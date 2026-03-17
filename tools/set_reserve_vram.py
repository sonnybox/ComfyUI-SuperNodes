import comfy.model_management
from comfy_api.latest import io


class SetReserveVRAM(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SetReserveVRAM",
            display_name="🐧 Set Reserve VRAM",
            category="SuperNodes/Tools",
            description="Sets --reserve-vram dynamically anywhere in a workflow.",
            inputs=[
                io.Custom("*").Input("any"),
                io.Float.Input(
                    "reserved_gb",
                    default=0.0,
                    min=0.0,
                    max=1024.0,
                    step=0.1,
                ),
            ],
            outputs=[
                io.Custom("*").Output(display_name="any"),
            ],
        )

    @classmethod
    def execute(cls, any, reserved_gb) -> io.NodeOutput:
        comfy.model_management.EXTRA_RESERVED_VRAM = (
            reserved_gb * 1024 * 1024 * 1024
        )
        return io.NodeOutput(any)


NODE = [SetReserveVRAM]
