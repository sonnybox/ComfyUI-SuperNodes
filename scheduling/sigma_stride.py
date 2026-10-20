from comfy_api.latest import io


class SigmaStride(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SigmaStride",
            display_name="🐧 Sigma Stride",
            category="SuperNodes/Scheduling",
            description="Keeps every nth sigma from the schedule.",
            inputs=[
                io.Custom("SIGMAS").Input(
                    "sigmas", tooltip="Input sigma schedule."
                ),
                io.Int.Input(
                    "nth",
                    default=2,
                    min=2,
                    max=8,
                    step=1,
                    tooltip="Keep every nth sigma.",
                ),
                io.Boolean.Input(
                    "start_after_first",
                    default=True,
                    tooltip="Keep the first sigma and count from it (1st, 1+nth, ...). When off, counting starts at the nth sigma.",
                ),
                io.Boolean.Input(
                    "keep_end",
                    default=True,
                    tooltip="Always keep the last sigma (usually 0) even if the stride skips it.",
                ),
            ],
            outputs=[
                io.Custom("SIGMAS").Output(
                    tooltip="Sigma schedule with only every nth sigma kept."
                ),
            ],
        )

    @classmethod
    def execute(cls, sigmas, nth, start_after_first, keep_end) -> io.NodeOutput:
        length = sigmas.shape[0]
        start = 0 if start_after_first else nth - 1
        indices = list(range(start, length, nth))

        if keep_end and (not indices or indices[-1] != length - 1):
            indices.append(length - 1)

        return io.NodeOutput(sigmas[indices])


NODE = [SigmaStride]
