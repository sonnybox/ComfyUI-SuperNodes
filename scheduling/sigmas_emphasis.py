from comfy_api.latest import io
import torch


class SigmasEmphasis(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SigmasEmphasis",
            display_name="🐧 Sigmas Emphasis",
            category="SuperNodes/Scheduling",
            description="Resamples a sigma schedule to a new step count using the original curve's geometry, with an emphasis control that clusters the new steps toward the beginning, middle, or end of the schedule.",
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
                io.Combo.Input(
                    "emphasis",
                    options=["beginning", "middle", "end"],
                    default="middle",
                    tooltip="Where the resampled steps are clustered along the schedule.",
                ),
                io.Float.Input(
                    "emphasis_factor",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="0.0 = even spacing along the curve, 1.0 = steps fully clustered toward the emphasis point.",
                ),
            ],
            outputs=[
                io.Custom("SIGMAS").Output(
                    tooltip="The resampled sigma schedule.",
                ),
            ],
        )

    @classmethod
    def execute(cls, sigmas, steps, emphasis, emphasis_factor) -> io.NodeOutput:
        s = sigmas.clone().float()

        if len(s) < 2:
            return io.NodeOutput(s)

        # Evenly spaced positions for the new schedule, then warped so that
        # sample density increases around the emphasis point. The warp keeps
        # 0 and 1 fixed, so the first and last sigmas (including a final 0.0)
        # are always preserved exactly.
        u = torch.linspace(0.0, 1.0, steps + 1, dtype=torch.float64)

        if emphasis == "beginning":
            warped = u**3
        elif emphasis == "end":
            warped = 1.0 - (1.0 - u) ** 3
        else:  # middle
            warped = 0.5 + 0.5 * (2.0 * u - 1.0) ** 3

        t = (1.0 - emphasis_factor) * u + emphasis_factor * warped

        # Sample the original curve at the warped positions via linear
        # interpolation between the original schedule's points.
        positions = t * (len(s) - 1)
        idx = positions.floor().long().clamp(max=len(s) - 2)
        frac = (positions - idx).to(s.dtype)
        new_sigmas = torch.lerp(s[idx], s[idx + 1], frac)

        return io.NodeOutput(new_sigmas.to(device=sigmas.device))


NODE = [SigmasEmphasis]
