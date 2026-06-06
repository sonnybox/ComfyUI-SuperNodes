from comfy_api.latest import io
import torch


class SigmaRemove(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SigmaRemove",
            display_name="🐧 Sigma Remove",
            category="SuperNodes/Scheduling",
            description="Removes the sigma value at the specified index from the schedule.",
            inputs=[
                io.Custom("SIGMAS").Input(
                    "sigmas", tooltip="Input sigma schedule."
                ),
                io.Int.Input(
                    "index",
                    default=0,
                    min=-10_000,
                    max=10_000,
                    step=1,
                    tooltip="Index of the sigma to remove. 0 = first, -1 = last, -2 = second to last, etc.",
                ),
            ],
            outputs=[
                io.Custom("SIGMAS").Output(
                    tooltip="Sigma schedule with the value at the index removed."
                ),
            ],
        )

    @classmethod
    def execute(cls, sigmas, index) -> io.NodeOutput:
        s = sigmas.clone()
        length = s.shape[0]

        # Normalize negative index
        if index < 0:
            index = length + index

        # Bounds check
        if index < 0 or index >= length:
            raise IndexError(
                f"Sigma index out of range: {index} (length of sigmas: {length})"
            )

        # Remove the element at the normalized index
        new_sigmas = torch.cat([s[:index], s[index + 1 :]])

        return io.NodeOutput(new_sigmas)


NODE = [SigmaRemove]
