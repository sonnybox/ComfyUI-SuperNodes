from comfy_api.latest import io
import folder_paths


class SuperSelectLoraName(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SuperSelectLoraName",
            display_name="🐧 Select LoRA Name",
            category="SuperNodes/Tools",
            description="Outputs a LoRA filename for one or more LoRA loaders to share. Feeding several loaders from one of these lets them run different strengths off a single selection, so swapping the LoRA is a one place change.",
            inputs=[
                io.Combo.Input(
                    "lora_name",
                    options=folder_paths.get_filename_list("loras"),
                ),
            ],
            outputs=[
                io.Custom("*").Output(display_name="lora_name"),
            ],
        )

    @classmethod
    def execute(cls, lora_name) -> io.NodeOutput:
        return io.NodeOutput(lora_name)


NODE = [SuperSelectLoraName]
