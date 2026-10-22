import math

from comfy.k_diffusion import sampling as k_diffusion_sampling
import comfy.model_sampling
from comfy_api.latest import io
import torch
from tqdm.auto import trange

from .utils import ChainedKSAMPLER, StreamOptions, res_multistep_stream

TAPERS = ["none", "start", "end", "both"]
TAPER_SHAPES = {
    "linear": lambda x: x,
    "quadratic": lambda x: x * x,
    "sqrt": math.sqrt,
    "cosine": lambda x: 0.5 - 0.5 * math.cos(math.pi * x),
}
# sub-intervals per step for the rate integral
RATE_SAMPLES = 32


def half_log_snr(sigma):
    # comfy.k_diffusion.sampling.sigma_to_half_log_snr for a CONST model
    return math.log((1.0 - sigma) / sigma)


def sigma_at(lambda_):
    # sigmoid(-lambda), split so exp cannot overflow
    if lambda_ >= 0.0:
        e = math.exp(-lambda_)
        return e / (1.0 + e)
    return 1.0 / (1.0 + math.exp(lambda_))


class Profile:
    """Where in sigma the stochasticity acts, and how strongly.

    The window runs from start down to end. A taper ramps the weight up from 0 at the start
    edge, down to 0 at the end edge, or both. Everything is a function of sigma alone, so it
    lands the same at any step count and across split schedules.
    """

    def __init__(self, start_sigma, end_sigma, taper, taper_type):
        self.hi, self.lo = (
            max(start_sigma, end_sigma),
            min(start_sigma, end_sigma),
        )
        self.taper = taper
        self.shape = TAPER_SHAPES[taper_type]

    def weight(self, sigma):
        if self.hi == self.lo or not self.lo <= sigma <= self.hi:
            return 0.0
        if self.taper == "none":
            return 1.0
        # 0 at the start edge, 1 at the end edge
        u = (self.hi - sigma) / (self.hi - self.lo)
        if self.taper == "start":
            return self.shape(u)
        if self.taper == "end":
            return self.shape(1.0 - u)
        return self.shape(min(2.0 * u, 2.0 - 2.0 * u))

    def renewal(self, sigma, sigma_next, stochasticity):
        """The rate integrated over this step in half log-SNR."""
        lambda_s, lambda_t = half_log_snr(sigma), half_log_snr(sigma_next)
        dl = (lambda_t - lambda_s) / RATE_SAMPLES
        return (
            stochasticity
            * dl
            * sum(
                self.weight(sigma_at(lambda_s + (k + 0.5) * dl))
                for k in range(RATE_SAMPLES)
            )
        )


def step_eta(sigma, sigma_next, renewal):
    """The ancestral eta that renews this step's noise by the given amount.

    DPM++ 2M SDE at eta = stochasticity keeps exp(-stochasticity * h) of the noise over a
    step, h being the step in half log-SNR. The flow ancestral step keeps
    sigma_down (1 - sigma_next) / (sigma_next (1 - sigma_down)) of it, which matches when
    sigma_down sits at half log-SNR lambda(sigma_next) + renewal, renewal being that rate
    integrated over the step.
    """
    if renewal == 0.0:
        return 0.0
    sigma_down = sigma_at(half_log_snr(sigma_next) + renewal)
    return (sigma_down / sigma_next - 1.0) / (sigma_next / sigma - 1.0)


@torch.no_grad()
def sample_res_multistep_stochastic(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    stochasticity=0.0,
    s_noise=1.0,
    power=1.0,
    start_sigma=1.0,
    end_sigma=0.0,
    taper="none",
    taper_type="linear",
    noise_sampler=None,
    seed_offset=0,
):
    """SamplerResMultistepFlow with its eta solved per step from a stochasticity rate, plus a
    deterministic power that lowers the latent's noise without adding any.
    """
    extra_args = {} if extra_args is None else extra_args
    model_sampling = model.inner_model.model_patcher.get_model_object(
        "model_sampling"
    )
    if (stochasticity > 0 or power < 1) and not isinstance(
        model_sampling, comfy.model_sampling.CONST
    ):
        raise ValueError(
            "stochasticity above 0 and power below 1 use the flow-model noise split, which needs a rectified flow (CONST) model such as Flux, SD3, Wan or H3."
        )
    seed = extra_args.get("seed", None)
    if noise_sampler is None:
        noise_sampler = k_diffusion_sampling.default_noise_sampler(
            x, seed=None if seed is None else seed ^ seed_offset
        )
    profile = Profile(start_sigma, end_sigma, taper, taper_type)
    values = sigmas.tolist()
    renewals = []
    for i in range(len(values) - 1):
        sigma, sigma_next = values[i], values[i + 1]
        # pure noise is infinitely far in log-SNR, so any rate would renew all of it
        if stochasticity == 0.0 or sigma >= 1.0 or not 0.0 < sigma_next < sigma:
            renewals.append(0.0)
            continue
        renewals.append(profile.renewal(sigma, sigma_next, stochasticity))
    etas = [
        step_eta(values[i], values[i + 1], r) for i, r in enumerate(renewals)
    ]
    # Noise ratio the latent should carry at each sigma: power over the whole 1 -> 0 range,
    # each step taking its share by how far it moves, so it holds across step counts and splits
    ramps = [power ** (1.0 - sigma) for sigma in values]
    # The noise ratio s_noise alone would leave, tracked so power lands on ramp * that and
    # the renoise does not refill what power took out. Starts at 1 per call, so a split with
    # s_noise != 1 drifts slightly from the unsplit run.
    baseline = 1.0
    o = StreamOptions(
        0.0,
        s_noise * getattr(model_sampling, "noise_scale", 1.0),
        noise_sampler,
        model_sampling,
    )
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
        # the stream reads o.eta after the send, so this is step i's eta
        o.eta = etas[i]
        done = False
        try:
            i_next, sigma_next, x_next = stream.send(denoised)
        except StopIteration as stop:
            x_next, done = stop.value, True
        sigma_next_value = values[i + 1]
        # the share of the old noise this step keeps; the renoise tops up the rest
        kept = math.exp(-2.0 * renewals[i])
        refill = s_noise**2 * (1.0 - kept)
        baseline_next = math.sqrt(kept * baseline**2 + refill)
        if sigma_next_value > 0.0 and (ramps[i] != 1.0 or ramps[i + 1] != 1.0):
            # An ODE step carries the latent's noise ratio through unchanged and the renoise
            # moves it predictably, so scaling the noise estimate around the clean one puts
            # it onto this sigma's ramp. In place: the stream steps on from this same tensor.
            actual = math.sqrt(kept * (ramps[i] * baseline) ** 2 + refill)
            # at power 0 the noise is already gone, and so is the target
            k = ramps[i + 1] * baseline_next / actual if actual > 0.0 else 0.0
            x_next.lerp_((1.0 - sigma_next_value) * denoised, 1.0 - k)
        baseline = baseline_next
        if done:
            return x_next
        i, sigma, x = i_next, sigma_next, x_next
    return x


class SamplerResMultistepStochastic(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SamplerResMultistepStochastic",
            display_name="🐧 SamplerResMultistepStochastic",
            category="SuperNodes/Sampling",
            description="SamplerResMultistepFlow with eta replaced by a stochasticity rate that stays the same across step counts and split schedules, and a deterministic power that calms the noise without adding any.",
            inputs=[
                io.Float.Input(
                    "stochasticity",
                    default=0.5,
                    min=0.0,
                    max=100.0,
                    step=0.01,
                    round=False,
                    tooltip="Noise renewal rate. 0 is res_multistep, 1 renews noise as fast as DPM++ 2M SDE at eta 1.",
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
                io.Float.Input(
                    "power",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    round=False,
                    tooltip="Noise the latent keeps from sigma 1 to 0, without adding any; each step takes its share. 1 is off, lower calms and dims, 0 is gray.",
                ),
                io.Float.Input(
                    "start_sigma",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.001,
                    round=False,
                    tooltip="Sigma the stochasticity window starts at. Equal start and end disables stochasticity.",
                ),
                io.Float.Input(
                    "end_sigma",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.001,
                    round=False,
                    tooltip="Sigma the stochasticity window ends at.",
                ),
                io.Combo.Input(
                    "taper",
                    options=TAPERS,
                    default=TAPERS[0],
                    tooltip="Which window edges fade in: start ramps up from the start sigma, end fades out towards the end sigma.",
                ),
                io.Combo.Input(
                    "taper_type",
                    options=list(TAPER_SHAPES),
                    default="linear",
                    tooltip="Fade curve. quadratic ramps slow then fast, sqrt fast then slow, cosine eases both ends.",
                ),
            ],
            outputs=[io.Sampler.Output()],
        )

    @classmethod
    def execute(
        cls,
        stochasticity,
        s_noise,
        power,
        start_sigma,
        end_sigma,
        taper,
        taper_type,
    ) -> io.NodeOutput:
        return io.NodeOutput(
            ChainedKSAMPLER(
                sample_res_multistep_stochastic,
                extra_options={
                    "stochasticity": stochasticity,
                    "s_noise": s_noise,
                    "power": power,
                    "start_sigma": start_sigma,
                    "end_sigma": end_sigma,
                    "taper": taper,
                    "taper_type": taper_type,
                },
            )
        )


NODE = [SamplerResMultistepStochastic]
