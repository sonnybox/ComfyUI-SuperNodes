from comfy_api.latest import io


class SigmaReplace(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SigmaReplace",
            display_name="🐧 Sigma Replace",
            category="SuperNodes/Scheduling",
            description="Replaces a single sigma value at the specified index. Index -1 is last.",
            inputs=[
                io.Custom("SIGMAS").Input(
                    "sigmas", tooltip="Input sigma schedule."
                ),
                io.Int.Input(
                    "index",
                    default=0,
                    min=-1,
                    max=10_000,
                    step=1,
                    tooltip="Index to replace. 0 = first, -1 = last.",
                ),
                io.Float.Input(
                    "value",
                    default=1.0,
                    min=0.0,
                    max=10_000.0,
                    step=0.01,
                    tooltip="New sigma value to replace at the given index.",
                ),
            ],
            outputs=[
                io.Custom("SIGMAS").Output(
                    tooltip="Sigma schedule with one value replaced."
                ),
            ],
        )

    @classmethod
    def execute(cls, sigmas, index, value) -> io.NodeOutput:
        s = sigmas.clone()
        length = s.shape[0]

        # Normalize negative index
        if index < 0:
            index = length + index

        # Bounds check
        if index < 0 or index >= length:
            raise IndexError(f"Sigma index out of range: {index}")

        # Validate against previous sigma
        if index > 0:
            prev_sigma = s[index - 1]
            if value > prev_sigma:
                raise ValueError(
                    f"Invalid sigma schedule: sigma[{index}] = {value} "
                    f"is greater than preceding sigma[{index - 1}] = {prev_sigma}"
                )

        # Validate against next sigma
        if index < length - 1:
            next_sigma = s[index + 1]
            if value < next_sigma:
                raise ValueError(
                    f"Invalid sigma schedule: sigma[{index}] = {value} "
                    f"is less than following sigma[{index + 1}] = {next_sigma}"
                )

        s[index] = value
        return io.NodeOutput(s)


NODE = [SigmaReplace]
