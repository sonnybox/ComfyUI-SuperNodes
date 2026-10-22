from comfy.k_diffusion import sampling as k_diffusion_sampling
import comfy.model_sampling
from comfy_api.latest import io
import torch
from tqdm.auto import trange

from .utils import ChainedKSAMPLER, StreamOptions, res_multistep_stream


@torch.no_grad()
def sample_res_multistep_flow(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    eta=0.0,
    s_noise=1.0,
    noise_sampler=None,
    seed_offset=0,
):
    """Core's res_multistep at eta 0, the dual sampler's flow-model ancestral above it.

    Drives the same per-stream coroutine the dual sampler steps each stream with, here over
    the whole latent on one schedule.
    """
    extra_args = {} if extra_args is None else extra_args
    model_sampling = model.inner_model.model_patcher.get_model_object(
        "model_sampling"
    )
    if eta > 0 and not isinstance(model_sampling, comfy.model_sampling.CONST):
        raise ValueError(
            "eta above 0 uses the flow-model noise split, which needs a rectified flow (CONST) model such as Flux, SD3, Wan or H3. Use eta 0 or core's res_multistep_ancestral."
        )
    seed = extra_args.get("seed", None)
    if noise_sampler is None:
        noise_sampler = k_diffusion_sampling.default_noise_sampler(
            x, seed=None if seed is None else seed ^ seed_offset
        )
    s_noise = s_noise * getattr(model_sampling, "noise_scale", 1.0)
    o = StreamOptions(eta, s_noise, noise_sampler, model_sampling)
    s_in = x.new_ones([x.shape[0]])

    stream = res_multistep_stream(x, sigmas, o)
    i, sigma, x = next(stream)
    for _ in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigma * s_in, **extra_args)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigma,
                    "sigma_hat": sigma,
                    "denoised": denoised,
                }
            )
        try:
            i, sigma, x = stream.send(denoised)
        except StopIteration as stop:
            return stop.value
    return x


class SamplerResMultistepFlow(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SamplerResMultistepFlow",
            display_name="🐧 SamplerResMultistepFlow",
            category="SuperNodes/Sampling",
            description="res_multistep with an ancestral mode that uses the flow-model noise split.",
            inputs=[
                io.Float.Input(
                    "eta",
                    default=0.0,
                    min=0.0,
                    max=100.0,
                    step=0.01,
                    round=False,
                    tooltip="Ancestral noise. 0 is res_multistep.",
                ),
                io.Float.Input(
                    "s_noise",
                    default=1.0,
                    min=0.0,
                    max=100.0,
                    step=0.01,
                    round=False,
                    tooltip="Scales the noise added each step.",
                ),
            ],
            outputs=[io.Sampler.Output()],
        )

    @classmethod
    def execute(cls, eta, s_noise) -> io.NodeOutput:
        return io.NodeOutput(
            ChainedKSAMPLER(
                sample_res_multistep_flow,
                extra_options={"eta": eta, "s_noise": s_noise},
            )
        )


NODE = [SamplerResMultistepFlow]
