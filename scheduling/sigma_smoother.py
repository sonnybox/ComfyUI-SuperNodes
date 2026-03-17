from comfy_api.latest import io
import torch


class SigmaSmoother(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SigmaSmoother",
            display_name="🐧 Sigma Smoother",
            category="SuperNodes/Scheduling",
            description="Inserts smoothed interpolation steps at the end of a sigma schedule before the final zero, useful for refining the final denoising steps.",
            inputs=[
                io.Custom("SIGMAS").Input(
                    "sigmas",
                    tooltip="The input sigmas tensor, typically ending in 0.0.",
                ),
                io.Int.Input(
                    "smooth_steps",
                    default=1,
                    min=1,
                    max=100,
                    tooltip="Number of additional smoothed steps to insert between the last non-zero sigma and 0.0.",
                ),
                io.Combo.Input(
                    "interpolation_type",
                    options=["linear", "decay"],
                    default="linear",
                    tooltip="Method to calculate the intermediate sigma values.",
                ),
            ],
            outputs=[
                io.Custom("SIGMAS").Output(display_name="SIGMAS"),
            ],
        )

    @classmethod
    def execute(cls, sigmas, smooth_steps, interpolation_type) -> io.NodeOutput:
        # Ensure we are working with a float tensor
        if sigmas.dtype != torch.float32 and sigmas.dtype != torch.float64:
            sigmas = sigmas.float()

        # Check if the last element is zero and remove it for calculation
        if sigmas[-1] == 0.0:
            active_sigmas = sigmas[:-1]
        else:
            active_sigmas = sigmas

        if len(active_sigmas) == 0:
            # Edge case: empty or only zero input
            return io.NodeOutput(
                sigmas,
            )

        last_sigma = active_sigmas[-1].item()
        new_steps = []

        if interpolation_type == "linear":
            # Linear interpolation from last_sigma to 0
            # Total intervals = smooth_steps + 1 (the final drop to 0 is the +1)
            step_size = last_sigma / (smooth_steps + 1)
            for i in range(1, smooth_steps + 1):
                new_val = last_sigma - (step_size * i)
                new_steps.append(new_val)

        elif interpolation_type == "decay":
            # Decay interpolation: previous / 2, then previous / 3, etc.
            # Example: 3 -> 1.5 -> 0.5 -> 0.125
            current_val = last_sigma
            for i in range(1, smooth_steps + 1):
                divisor = i + 1
                current_val = current_val / divisor
                new_steps.append(current_val)

        # Convert new steps to tensor ensuring matching device and type
        new_sigmas_tensor = torch.tensor(
            new_steps, dtype=sigmas.dtype, device=sigmas.device
        )

        # Reconstruct: Old sigmas (minus zero) + New Steps + Zero
        parts = [active_sigmas, new_sigmas_tensor]

        # Always append 0.0 at the end as per ComfyUI sigma standards
        zero_tensor = torch.tensor(
            [0.0], dtype=sigmas.dtype, device=sigmas.device
        )
        parts.append(zero_tensor)

        result_sigmas = torch.cat(parts)

        return io.NodeOutput(result_sigmas)


NODE = [SigmaSmoother]
