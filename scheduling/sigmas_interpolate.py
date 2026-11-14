from comfy_api.latest import io
import torch


class SigmasInterpolate(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SigmasInterpolate",
            display_name="🐧 Sigmas Interpolate",
            category="SuperNodes/Scheduling",
            description="Resamples a sigma schedule to a new step count while keeping the shape of the original curve.",
            inputs=[
                io.Custom("SIGMAS").Input(
                    "sigmas",
                    tooltip="The input sigma schedule to resample.",
                ),
                io.Int.Input(
                    "steps",
                    default=10,
                    min=1,
                    max=1000,
                    tooltip="Number of steps for the output schedule. The output contains steps + 1 sigma values.",
                ),
            ],
            outputs=[
                io.Custom("SIGMAS").Output(
                    tooltip="The resampled sigma schedule.",
                ),
            ],
        )

    @classmethod
    def execute(cls, sigmas, steps) -> io.NodeOutput:
        s = sigmas.clone().float()

        if len(s) < 2:
            return io.NodeOutput(s)

        # Evenly spaced positions along the original schedule, so the first
        # and last sigmas (including a final 0.0) are preserved exactly.
        positions = torch.linspace(0.0, 1.0, steps + 1, dtype=torch.float64) * (len(s) - 1)
        idx = positions.floor().long().clamp(max=len(s) - 2)
        frac = (positions - idx).to(s.dtype)
        new_sigmas = torch.lerp(s[idx], s[idx + 1], frac)

        return io.NodeOutput(new_sigmas.to(device=sigmas.device))


NODE = [SigmasInterpolate]
