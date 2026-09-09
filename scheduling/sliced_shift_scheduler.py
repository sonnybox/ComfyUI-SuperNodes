from comfy_api.latest import io
import torch


def _sigma_at(shift: float, t: float) -> float:
    # ComfyUI's time_snr_shift: schedule position t -> sigma
    if shift == 1.0:
        return t
    return shift * t / (1 + (shift - 1) * t)


def _t_at(shift: float, sigma: float) -> float:
    # Inverse of _sigma_at: sigma -> schedule position t
    if shift == 1.0:
        return sigma
    return sigma / (shift - sigma * (shift - 1))


class SlicedShiftScheduler(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SlicedShiftScheduler",
            display_name="🐧 Sliced Shift Scheduler",
            category="SuperNodes/Scheduling",
            description="A simple scheduler with user specified sigma ranges",
            inputs=[
                io.Float.Input(
                    "shift",
                    default=3.0,
                    min=1.0,
                    max=100.0,
                    step=0.01,
                ),
                io.Int.Input(
                    "steps",
                    default=20,
                    min=1,
                    max=10_000,
                    step=1,
                ),
                io.Float.Input(
                    "max",
                    default=1.0,
                    min=0.001,
                    max=1.0,
                    step=0.001,
                    tooltip="Sigma the slice starts at. 1.0 starts at the beginning of the schedule.",
                ),
                io.Float.Input(
                    "min",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.001,
                    tooltip="Sigma the slice ends at. 0.0 ends at the natural end of the schedule.",
                ),
            ],
            outputs=[
                io.Custom("SIGMAS").Output(
                    tooltip="The sliced sigma schedule."
                ),
            ],
        )

    @classmethod
    def execute(cls, shift, steps, max, min) -> io.NodeOutput:
        if max <= 0.0:
            raise ValueError(
                f"Invalid sigma slice: max must be greater than 0.0 (got {max})"
            )
        if max > 1.0:
            raise ValueError(
                f"Invalid sigma slice: max cannot exceed 1.0 (got {max})"
            )
        if min < 0.0:
            raise ValueError(
                f"Invalid sigma slice: min cannot be negative (got {min})"
            )
        if min >= max:
            raise ValueError(
                f"Invalid sigma slice: min ({min}) must be lower than max ({max}). "
                f"A slice with no span produces a flat schedule that samplers cannot use."
            )

        # Find where the requested sigmas sit on the shifted curve, then walk the
        # curve parameter in equal increments between them.
        t_max = _t_at(shift, max)
        t_min = _t_at(shift, min)
        span = t_max - t_min

        sigmas = [
            _sigma_at(shift, t_max - span * (x / steps))
            for x in range(steps + 1)
        ]

        # Pin the endpoints so float error cannot drift the terminal sigma off 0.0
        sigmas[0] = max
        sigmas[-1] = min

        return io.NodeOutput(torch.tensor(sigmas, dtype=torch.float32))


NODE = [SlicedShiftScheduler]
