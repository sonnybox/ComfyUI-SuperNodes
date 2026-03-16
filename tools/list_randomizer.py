import random

from comfy_api.latest import io


class SuperListRandomizer(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SuperListRandomizer",
            display_name="🐧 List Randomizer",
            category="SuperNodes/Tools",
            inputs=[
                io.String.Input("text", multiline=True, default=""),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=9007199254740991,
                    step=1,
                    control_after_generate=True,
                ),
            ],
            outputs=[
                io.String.Output(display_name="text"),
            ],
        )

    @classmethod
    def execute(cls, text, seed) -> io.NodeOutput:
        # We process the string literal by literal, even if blank
        items = text.split(",")

        # Strip pre/post whitespace for each item
        items = [item.strip() for item in items]

        if not items:
            return io.NodeOutput("")

        random.seed(seed)
        result = random.choice(items)

        return io.NodeOutput(result)


V3_NODES = [SuperListRandomizer]
