from comfy_api.latest import io
import torch


class SigmasRescale(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SigmasRescale",
            display_name="🐧 Sigmas Rescale",
            category="SuperNodes/Scheduling",
            description="Rescales a sigma schedule to a new maximum and minimum range while preserving the exact curve of the original schedule.",
            inputs=[
                io.Custom("SIGMAS").Input(
                    "sigmas", tooltip="The input sigma schedule to be rescaled."
                ),
                io.Float.Input(
                    "max",
                    default=1.0,
                    min=0.0,
                    max=10000.0,
                    step=0.01,
                    tooltip="The new maximum value (start of the schedule).",
                ),
                io.Float.Input(
                    "min",
                    default=0.0,
                    min=0.0,
                    max=1000.0,
                    step=0.001,
                    tooltip="The new minimum value (end of the schedule).",
                ),
            ],
            outputs=[
                io.Custom("SIGMAS").Output(
                    tooltip="The rescaled sigma schedule."
                ),
            ],
        )

    @classmethod
    def execute(cls, sigmas, max, min) -> io.NodeOutput:
        # Avoid modifying the original tensor
        s = sigmas.clone()

        # Get the current range of the input sigmas
        # Sigmas usually go from High to Low, so index 0 is max, index -1 is min
        current_max = s[0]
        current_min = s[-1]

        # Handle edge case where max equals min to avoid division by zero
        if current_max == current_min:
            # If the schedule is flat, return a flat schedule at the new max
            return io.NodeOutput(torch.full_like(s, max))

        # Normalize the curve to 0.0 - 1.0
        # Formula: (value - min) / (max - min)
        normalized_curve = (s - current_min) / (current_max - current_min)

        # Scale to the new range
        # Formula: normalized * (new_max - new_min) + new_min
        new_sigmas = normalized_curve * (max - min) + min

        return io.NodeOutput(new_sigmas)


NODE = [SigmasRescale]
