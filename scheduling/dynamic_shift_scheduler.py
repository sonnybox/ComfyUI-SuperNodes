import math

from comfy_api.latest import io
import torch

TIME_SHIFT_TYPES = ["exponential", "linear"]


def _calculate_mu(
    seq_length: int,
    base_seq_length: int,
    max_seq_length: int,
    base_shift: float,
    max_shift: float,
) -> float:
    # diffusers calculate_shift(): the straight line through
    # (base_seq_length, base_shift) and (max_seq_length, max_shift), read at
    # seq_length.
    m = (max_shift - base_shift) / (max_seq_length - base_seq_length)
    b = base_shift - m * base_seq_length
    return seq_length * m + b


def _time_shift(
    mu: float, t: torch.Tensor, time_shift_type: str
) -> torch.Tensor:
    # diffusers _time_shift_exponential / _time_shift_linear with sigma fixed at 1.0.
    # The exponential branch is the same curve as ComfyUI's shift with shift = e**mu.
    if time_shift_type == "exponential":
        return math.exp(mu) / (math.exp(mu) + (1.0 / t - 1.0))
    return mu / (mu + (1.0 / t - 1.0))


def _stretch_to_terminal(
    sigmas: torch.Tensor, shift_terminal: float
) -> torch.Tensor:
    # diffusers stretch_shift_to_terminal(): rescales the schedule in (1 - sigma) space
    # so the last sigma lands on shift_terminal while the first stays pinned at 1.0.
    one_minus_z = 1.0 - sigmas
    scale_factor = one_minus_z[-1] / (1.0 - shift_terminal)
    return 1.0 - (one_minus_z / scale_factor)


class DynamicShiftScheduler(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DynamicShiftScheduler",
            display_name="🐧 Dynamic Shift Scheduler",
            category="SuperNodes/Scheduling",
            description="Builds a flow match sigma schedule from a diffusers scheduler config.",
            search_aliases=[
                "flow match",
                "FlowMatchEulerDiscreteScheduler",
                "dynamic shifting",
                "shift terminal",
                "mu",
                "diffusers",
                "huggingface",
            ],
            inputs=[
                io.Int.Input(
                    "steps",
                    default=20,
                    min=2,
                    max=10_000,
                    step=1,
                ),
                io.Int.Input(
                    "seq_length",
                    default=4096,
                    min=1,
                    max=1_000_000,
                    step=1,
                    tooltip="Token count of the latent being sampled.",
                ),
                io.Int.Input(
                    "base_seq_length",
                    default=256,
                    min=1,
                    max=1_000_000,
                    step=1,
                ),
                io.Int.Input(
                    "max_seq_length",
                    default=4096,
                    min=1,
                    max=1_000_000,
                    step=1,
                ),
                io.Float.Input(
                    "base_shift",
                    default=0.5,
                    min=-100.0,
                    max=100.0,
                    step=0.01,
                ),
                io.Float.Input(
                    "max_shift",
                    default=1.15,
                    min=-100.0,
                    max=100.0,
                    step=0.01,
                ),
                io.Float.Input(
                    "shift_terminal",
                    default=0.0,
                    min=0.0,
                    max=0.999,
                    step=0.001,
                    tooltip="0.0 disables it, matching a config with no shift_terminal.",
                ),
                io.Combo.Input(
                    "time_shift_type",
                    options=TIME_SHIFT_TYPES,
                ),
            ],
            outputs=[
                io.Custom("SIGMAS").Output(
                    tooltip="The flow match sigma schedule."
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        steps,
        seq_length,
        base_seq_length,
        max_seq_length,
        base_shift,
        max_shift,
        shift_terminal,
        time_shift_type,
    ) -> io.NodeOutput:
        if base_seq_length == max_seq_length:
            raise ValueError(
                f"Invalid config: base_seq_length and max_seq_length are both {base_seq_length}. "
                f"The shift line needs two distinct sequence lengths to be defined."
            )

        mu = _calculate_mu(
            seq_length,
            base_seq_length,
            max_seq_length,
            base_shift,
            max_shift,
        )

        # The linear form divides by mu directly rather than by e**mu, so it has no
        # useful branch at or below zero the way the exponential form does
        if time_shift_type == "linear" and mu <= 0.0:
            raise ValueError(
                f"Invalid config: linear time_shift_type produced mu = {mu:.4f} at a sequence "
                f"length of {seq_length}, but it requires a positive mu. Check base_shift "
                f"and max_shift, or switch to exponential."
            )

        # This grid is what ComfyUI's
        # simple scheduler walks, so a config with no shift_terminal comes out
        # the same as ModelSamplingFlux plus BasicScheduler on simple.
        sigmas = torch.linspace(1.0, 1.0 / steps, steps, dtype=torch.float64)
        sigmas = _time_shift(mu, sigmas, time_shift_type)

        # A config without shift_terminal leaves the schedule ending on its natural
        # final sigma.
        if shift_terminal > 0.0:
            sigmas = _stretch_to_terminal(sigmas, shift_terminal)

        sigmas = torch.cat([sigmas, torch.zeros(1, dtype=torch.float64)])

        return io.NodeOutput(sigmas.to(dtype=torch.float32))


NODE = [DynamicShiftScheduler]
