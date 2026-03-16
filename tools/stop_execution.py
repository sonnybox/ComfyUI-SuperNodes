from comfy_api.latest import io


class SuperStopExecution(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="User Error",
            display_name="🐧 Show Error Message",
            category="SuperNodes/Tools",
            is_output_node=True,
            inputs=[
                io.String.Input(
                    "message",
                    multiline=False,
                    default="Message.",
                ),
                io.Boolean.Input(
                    "trigger",
                    default=True,
                    tooltip="If True, execution halts. If False, nothing happens.",
                    force_input=True,
                ),
            ],
            outputs=[],
        )

    @classmethod
    def execute(cls, message, trigger) -> io.NodeOutput:
        if trigger:
            alert = str(message)
            raise Exception(f"{alert}")

        return io.NodeOutput()


V3_NODES = [SuperStopExecution]
