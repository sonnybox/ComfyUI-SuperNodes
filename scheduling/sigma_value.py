from comfy_api.latest import io


class SigmaValue(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SigmaValue",
            display_name="🐧 Sigma Value",
            category="SuperNodes/Scheduling",
            description="Get a sigma's float value by index; -1 is the last.",
            search_aliases=[
                "sigma",
                "index",
                "float",
                "value",
                "sigma at index",
                "get sigma",
            ],
            inputs=[
                io.Custom("SIGMAS").Input(
                    "sigmas", tooltip="Input sigma schedule."
                ),
                io.Int.Input(
                    "index",
                    default=0,
                    min=-1,
                    max=1000,
                    step=1,
                ),
            ],
            outputs=[
                io.Float.Output(),
            ],
        )

    @classmethod
    def execute(cls, sigmas, index) -> io.NodeOutput:
        length = sigmas.shape[0]

        # -1 is the only negative index accepted; it maps to the final sigma
        target_idx = length - 1 if index == -1 else index

        if target_idx < 0 or target_idx >= length:
            raise IndexError(
                f"Sigma index out of range: {index} (resolved to target index: {target_idx}, length of sigmas: {length})"
            )

        return io.NodeOutput(float(sigmas[target_idx]))


NODE = [SigmaValue]
