"""Shared machinery for the dual-schedule sampler (MiniMax H3)."""

import copy
from functools import partial
import hashlib
import logging
import math

from comfy.k_diffusion import sampling as k_diffusion_sampling
import comfy.model_sampling
import comfy.samplers
from comfy_api.latest import io
import torch
from tqdm.auto import trange

# Video cannot run at sigma 0 because the model's noise mapping fixes 0, which would
# pin audio to 0 as well. Running a frozen video at a constant epsilon leaves it
# practically clean while keeping the audio mapping solvable.
FROZEN_VIDEO_SIGMA = 0.0001

# DPM++ SDE's midpoint position. Core exposes it as a widget; this node does not, so the
# general (1 - fac)/fac blend below collapses to denoised_2 - it is kept in that form to
# stay diffable against comfy.k_diffusion.sampling.sample_dpmpp_sde.
DPMPP_SDE_R = 0.5

# A socket of its own, so a stock SAMPLER cannot be wired into a dual sampling node.
DualSamplerType = io.Custom("DUAL_SAMPLER")


def audio_shift_for(sigma_v, sigma_a):
    """The to_shift that makes time_shift_sigma(sigma_v, 1.0, to_shift) == sigma_a.

    With from_shift 1.0 the model's inversion is the identity, so its mapping reduces to
    sigma_a = to*sigma_v / (1 + (to-1)*sigma_v), which inverts in closed form. The mapping
    fixes sigma = 1, which check_schedules rejects, so the guard here is only numerical.
    """
    if sigma_v >= 1.0 or sigma_a >= 1.0:
        return 1.0
    return sigma_a * (1.0 - sigma_v) / (sigma_v * (1.0 - sigma_a))


def is_frozen(sigmas):
    """An all-zero schedule means: leave this stream alone, it is already x0."""
    return bool((sigmas == 0).all())


def check_schedules(video_sigmas, audio_sigmas):
    """Reject the two schedule pairings that cannot be sampled. Nothing is rewritten."""
    if video_sigmas.shape[-1] != audio_sigmas.shape[-1]:
        raise ValueError(
            "video_sigmas and audio_sigmas must have the same number of steps, got {} and {}.".format(
                video_sigmas.shape[-1], audio_sigmas.shape[-1]
            )
        )
    # n sigmas are n - 1 steps, so two is the shortest real schedule ([1.0, 0.0] is one
    # step and samples fine). One sigma describes a noise level and no step to take from
    # it, which would otherwise surface as a bare StopIteration out of the driver.
    if video_sigmas.shape[-1] < 2:
        raise ValueError(
            "A sigma schedule needs at least two values, which is one step, got {}. Use [1.0, 0.0] for a single step.".format(
                video_sigmas.shape[-1]
            )
        )
    # The model maps between the two streams' noise levels with a transform that fixes 1.0,
    # so at 1.0 both streams are pinned together and the other one cannot be expressed.
    first_v, first_a = float(video_sigmas[0]), float(audio_sigmas[0])
    if first_v >= 1.0 and first_a < 1.0:
        raise ValueError(
            "Video sigma starts at 1.0 while the audio sigma is less than 1 ({:g}). Use 0.9999 for the first video sigma.".format(
                first_a
            )
        )
    if first_a >= 1.0 and first_v < 1.0:
        raise ValueError(
            "Audio sigma starts at 1.0 while the video sigma is less than 1 ({:g}). Use 0.9999 for the first audio sigma.".format(
                first_v
            )
        )


def ancestral_stream(x, sigmas, o):
    """Ancestral Euler over one stream, mirroring core's sample_euler_ancestral_RF.

    A coroutine: it yields (step, sigma, x) to ask for this stream's x0 estimate at that
    noise level and receives it back. The driver pairs both streams' requests into a single
    DiT forward, which is what lets the two run on independent schedules. eta 0 makes it
    plain Euler.
    """
    for i in range(len(sigmas) - 1):
        denoised = yield (i, sigmas[i], x)
        if sigmas[i + 1] == 0:
            x = denoised
            continue
        downstep_ratio = 1 + (sigmas[i + 1] / sigmas[i] - 1) * o.eta
        sigma_down = sigmas[i + 1] * downstep_ratio
        alpha_ip1 = 1 - sigmas[i + 1]
        alpha_down = 1 - sigma_down
        renoise_coeff = (
            sigmas[i + 1] ** 2 - sigma_down**2 * alpha_ip1**2 / alpha_down**2
        ) ** 0.5
        sigma_down_i_ratio = sigma_down / sigmas[i]
        x = sigma_down_i_ratio * x + (1 - sigma_down_i_ratio) * denoised
        if o.eta > 0:
            x = (alpha_ip1 / alpha_down) * x + o.noise(
                sigmas[i], sigmas[i + 1]
            ) * o.s_noise * renoise_coeff
    return x


def dpmpp_sde_stream(x, sigmas, o):
    """DPM-Solver++ (stochastic) over one stream, mirroring core's sample_dpmpp_sde.

    Two model evaluations per step: one at sigmas[i], then a midpoint one at sigma_s_1 that
    is yielded with a step index of None so it does not tick the progress bar. A terminal
    step (sigma_next 0) costs only the first.

    The logSNR of a flow model is log((1 - sigma) / sigma), which is -inf at sigma 1, so the
    first sigma is nudged off 1.0 the way core does. That happens on the local copy only -
    the initial noise_scaling has already run against the schedule as the user wrote it,
    which is the same split core makes between KSAMPLER.sample and the sampler function.
    """
    sigmas = k_diffusion_sampling.offset_first_sigma_for_snr(
        sigmas, o.model_sampling
    )
    sigma_fn = partial(
        k_diffusion_sampling.half_log_snr_to_sigma, model_sampling=o.model_sampling
    )
    lambda_fn = partial(
        k_diffusion_sampling.sigma_to_half_log_snr, model_sampling=o.model_sampling
    )

    for i in range(len(sigmas) - 1):
        denoised = yield (i, sigmas[i], x)
        if sigmas[i + 1] == 0:
            x = denoised
            continue

        lambda_s, lambda_t = lambda_fn(sigmas[i]), lambda_fn(sigmas[i + 1])
        h = lambda_t - lambda_s
        lambda_s_1 = lambda_s + DPMPP_SDE_R * h
        fac = 1 / (2 * DPMPP_SDE_R)

        sigma_s_1 = sigma_fn(lambda_s_1)

        # for a flow model exp(lambda) is (1 - sigma) / sigma, so these are the RF alphas
        alpha_s = sigmas[i] * lambda_s.exp()
        alpha_s_1 = sigma_s_1 * lambda_s_1.exp()
        alpha_t = sigmas[i + 1] * lambda_t.exp()

        # Step 1. get_ancestral_step takes exp(-lambda), the variance-preserving sigma, not
        # the flow sigma - the noise split is computed in that parameterisation.
        sd, su = k_diffusion_sampling.get_ancestral_step(
            lambda_s.neg().exp(), lambda_s_1.neg().exp(), o.eta
        )
        lambda_s_1_ = sd.log().neg()
        h_ = lambda_s_1_ - lambda_s
        x_2 = (alpha_s_1 / alpha_s) * (-h_).exp() * x - alpha_s_1 * (
            -h_
        ).expm1() * denoised
        if o.eta > 0 and o.s_noise > 0:
            x_2 = x_2 + alpha_s_1 * o.noise(sigmas[i], sigma_s_1) * o.s_noise * su
        denoised_2 = yield (None, sigma_s_1, x_2)

        # Step 2
        sd, su = k_diffusion_sampling.get_ancestral_step(
            lambda_s.neg().exp(), lambda_t.neg().exp(), o.eta
        )
        lambda_t_ = sd.log().neg()
        h_ = lambda_t_ - lambda_s
        denoised_d = (1 - fac) * denoised + fac * denoised_2
        x = (alpha_t / alpha_s) * (-h_).exp() * x - alpha_t * (
            -h_
        ).expm1() * denoised_d
        if o.eta > 0 and o.s_noise > 0:
            x = x + alpha_t * o.noise(sigmas[i], sigmas[i + 1]) * o.s_noise * su
    return x


def brownian_noise(x, sigmas, seed, cpu):
    """A torchsde Brownian path over one stream's own schedule.

    An SDE solver draws twice inside a step and both draws have to come off the same path,
    so the independent draw ancestral_stream uses will not do. Indexing the path by this
    stream's sigmas rather than the video's is what makes the audio stream's solver
    self-consistent when the two schedules differ.
    """
    positive = sigmas[sigmas > 0]
    if positive.numel() == 0 or float(positive.min()) == float(sigmas.max()):
        # A constant schedule - the frozen-video path - leaves the tree no interval to span.
        # Nothing rides on the draw: a constant schedule's ancestral split is 0 at every
        # step, so whatever comes back is multiplied by zero.
        return k_diffusion_sampling.default_noise_sampler(x, seed=seed)
    return k_diffusion_sampling.BrownianTreeNoiseSampler(
        x, positive.min(), sigmas.max(), seed=seed, cpu=cpu
    )


def stage_offset(video_sigmas, audio_sigmas):
    """A stable per-stage offset for the ancestral noise stream.

    DisableNoise reports seed 0 on every pass, so two stages chained over a latent of the
    same shape draw the identical ancestral noise and stamp the same pattern in twice. The
    schedule is what tells one stage from the next, so it is what the offset comes from.
    Two stages given the exact same schedule still collide - from in here they are the same
    stage - which is fine for splitting one schedule across passes, the case this covers.

    blake2b rather than hash(), which is salted per process and would not reproduce across
    restarts. Masked to 62 bits: default_noise_sampler adds 1 for a CPU generator, so the
    result needs headroom under the 64-bit seed limit.
    """
    payload = b"".join(
        s.detach().to("cpu", torch.float32).contiguous().numpy().tobytes()
        for s in (video_sigmas, audio_sigmas)
    )
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") & ((1 << 62) - 1)


class StreamOptions:
    """Per-stream knobs and noise source handed to the sampler coroutine."""

    def __init__(self, eta, s_noise, noise, model_sampling):
        self.eta, self.s_noise, self.noise = eta, s_noise, noise
        self.model_sampling = model_sampling


class DualSampler(comfy.samplers.Sampler):
    """Steps a packed [video, audio] latent on two independent sigma schedules.

    The video schedule arrives through the normal `sigmas` argument; the audio schedule and
    the audio stream's seed are attached by DualSamplerCustomAdvanced via
    with_audio_sigmas().

    stream_fn selects the algorithm both streams are stepped with. noise_device picks where
    the SDE solvers build their Brownian path, and doubles as the flag for needing one at
    all: None means the algorithm is happy with independent draws.
    """

    def __init__(
        self,
        eta_video=1.0,
        eta_audio=1.0,
        s_noise_video=1.0,
        s_noise_audio=1.0,
        stream_fn=ancestral_stream,
        noise_device=None,
    ):
        self.eta_video = eta_video
        self.eta_audio = eta_audio
        self.s_noise_video = s_noise_video
        self.s_noise_audio = s_noise_audio
        self.stream_fn = stream_fn
        self.noise_device = noise_device
        self.audio_sigmas = None
        self.audio_seed = None

    def with_audio_sigmas(self, audio_sigmas, audio_seed=None):
        sampler = copy.copy(self)
        sampler.audio_sigmas = audio_sigmas
        sampler.audio_seed = audio_seed
        return sampler

    def stream_options(
        self, x, split, video, seed, model_sampling, sigmas, offset=0
    ):
        """Noise source and knobs for one stream.

        Without a noise device the sampler is built on the *packed* shape and sliced, so
        with equal schedules the two streams draw exactly the tensor core's single sampler
        would have drawn. With one, each stream gets its own Brownian path over its own
        schedule instead, which is what an SDE solver needs.

        offset separates one chained stage's draws from the next's. It lands after the audio
        seed is picked, so both streams carry it and stay distinct from each other; the same
        value goes to both, which is what keeps the sliced path's core parity intact.
        """
        region = slice(None, split) if video else slice(split, None)
        if self.noise_device is None:
            draw = k_diffusion_sampling.default_noise_sampler(
                x, seed=None if seed is None else seed ^ offset
            )

            def noise(a, b):
                return draw(a, b)[..., region]
        else:
            # a path of its own per stream, so the audio solver is not steered by the video
            # seed; the sliced path above stays on one seed to keep its core parity
            if not video and self.audio_seed is not None:
                seed = self.audio_seed
            noise = brownian_noise(
                x[..., region],
                sigmas,
                None if seed is None else seed ^ offset,
                self.noise_device == "cpu",
            )
        noise_scale = getattr(model_sampling, "noise_scale", 1.0)
        return StreamOptions(
            eta=self.eta_video if video else self.eta_audio,
            s_noise=(self.s_noise_video if video else self.s_noise_audio)
            * noise_scale,
            noise=noise,
            model_sampling=model_sampling,
        )

    def sample(
        self,
        model_wrap,
        sigmas,
        extra_args,
        callback,
        noise,
        latent_image=None,
        denoise_mask=None,
        disable_pbar=False,
    ):
        if self.audio_sigmas is None:
            raise ValueError(
                "A dual sampler needs both schedules: connect it to DualSamplerCustomAdvanced, not SamplerCustomAdvanced."
            )

        inner_model = model_wrap.inner_model
        model_sampling = inner_model.model_sampling
        if not isinstance(model_sampling, comfy.model_sampling.CONST):
            raise ValueError(
                "The dual sampler only supports rectified flow (CONST) models such as MiniMax H3."
            )
        # The shifts on the guider's model are irrelevant here - this node overrides both per
        # step and drives the streams from the input sigmas. The multiplier is not: it is the
        # timestep scale the model is called on, and the H3 DiT reads sigma back as
        # timestep/1000. ModelSamplingAuraFlow patches at multiplier 1.0, which the stock path
        # rejects for other reasons but this one would otherwise run 1000x off in silence.
        multiplier = getattr(model_sampling, "multiplier", None)
        if multiplier != 1000:
            raise ValueError(
                "The guider's model is patched at timestep multiplier {} instead of 1000, so the DiT would read every sigma too small. Use ModelSamplingMiniMaxH3 or ModelSamplingSD3 on the guider and keep AuraFlow on a separate branch feeding BasicScheduler.".format(
                    multiplier
                )
            )

        latent_shapes = inner_model.latent_shapes
        if latent_shapes is None or len(latent_shapes) < 2:
            raise ValueError(
                "DualSamplerCustomAdvanced needs a packed audio-video latent (two streams), got one stream."
            )

        # pack_latents lays the streams out flat and in order: video first, then audio
        split = math.prod(latent_shapes[0][1:])

        video_sigmas = sigmas
        audio_sigmas = self.audio_sigmas.to(
            device=video_sigmas.device, dtype=video_sigmas.dtype
        )

        model_options = extra_args.get("model_options", {})
        # a per-sample deep clone, so per-step writes stay local to this run
        transformer_options = model_options.setdefault(
            "transformer_options", {}
        )
        transformer_options["sample_sigmas_audio"] = audio_sigmas

        seed = extra_args.get("seed", None)

        check_schedules(video_sigmas, audio_sigmas)
        total_steps = len(video_sigmas) - 1

        # A pass handed no noise is resuming, and carries DisableNoise's fixed seed 0 - the
        # same one every stage of a chain reports. Spread those apart by the schedule, taken
        # here while it is still the one the user wrote: the frozen-video path below rewrites
        # it. A pass that was handed noise already has a seed of its own, so it keeps it and
        # nothing that reproduces today changes.
        offset = 0
        if seed is not None and not bool(noise.any()):
            offset = stage_offset(video_sigmas, audio_sigmas)

        # an all-zero schedule freezes its stream: the latent is already x0, so it is never
        # stepped and the DiT is told t = 1 (clean) for it every pass
        frozen_v, frozen_a = is_frozen(video_sigmas), is_frozen(audio_sigmas)
        if frozen_v and frozen_a:
            raise ValueError(
                "both schedules are all zeros, so there is nothing to sample."
            )
        if frozen_v:
            # audio_shift_for cannot solve against a video sigma of 0 - the model's mapping
            # fixes 0, so the audio would be pinned there too. A constant epsilon leaves the
            # video untouched and keeps the mapping solvable.
            logging.info(
                "Received an all-zero video schedule, running it at a constant %g instead.",
                FROZEN_VIDEO_SIGMA,
            )
            video_sigmas = torch.full_like(video_sigmas, FROZEN_VIDEO_SIGMA)
            frozen_v = False

        # each stream is brought up to its own first sigma, so the two can start anywhere
        x = torch.empty_like(noise)
        x[..., :split] = model_sampling.noise_scaling(
            video_sigmas[0],
            noise[..., :split],
            latent_image[..., :split],
            self.max_denoise(model_wrap, video_sigmas),
        )
        x[..., split:] = model_sampling.noise_scaling(
            audio_sigmas[0],
            noise[..., split:],
            latent_image[..., split:],
            self.max_denoise(model_wrap, audio_sigmas),
        )
        s_in = x.new_ones([x.shape[0]])

        # Where a stream sits while it is not being stepped: a frozen one never moves, a
        # finished one waits at its last sigma for the other to catch up. Audio at 0 reads
        # as clean to the model, which is what a frozen audio stream wants; video at 0 does
        # not, since audio_shift_for cannot solve against it, so video holds at
        # FROZEN_VIDEO_SIGMA for the same reason the frozen video path does.
        held_v, held_a = x[..., :split], x[..., split:]
        held_sigma_v, held_sigma_a = FROZEN_VIDEO_SIGMA, 0.0

        # a frozen stream is never stepped, so it needs no noise source
        opts_v = (
            None
            if frozen_v
            else self.stream_options(
                x, split, True, seed, model_sampling, video_sigmas, offset
            )
        )
        opts_a = (
            None
            if frozen_a
            else self.stream_options(
                x, split, False, seed, model_sampling, audio_sigmas, offset
            )
        )

        gen_v = (
            None if frozen_v else self.stream_fn(held_v, video_sigmas, opts_v)
        )
        gen_a = (
            None if frozen_a else self.stream_fn(held_a, audio_sigmas, opts_a)
        )

        progress = trange(total_steps, disable=disable_pbar)
        req_v = next(gen_v) if gen_v is not None else None
        req_a = next(gen_a) if gen_a is not None else None
        done_v = held_v if frozen_v else None
        done_a = held_a if frozen_a else None
        while True:
            # a substep carries no step index; take the first real one on offer, so a
            # stream mid-step does not swallow the other's finished step
            step = next(
                (
                    req[0]
                    for req in (req_v, req_a)
                    if req is not None and req[0] is not None
                ),
                None,
            )
            sigma_v, xv = (
                (req_v[1], req_v[2])
                if req_v is not None
                else (held_sigma_v, held_v)
            )
            sigma_a, xa = (
                (req_a[1], req_a[2])
                if req_a is not None
                else (held_sigma_a, held_a)
            )
            sigma_v, sigma_a = float(sigma_v), float(sigma_a)
            x = torch.cat([xv, xa], dim=-1)

            x_in = x
            if denoise_mask is not None:
                x_in = self.apply_mask(
                    inner_model,
                    x,
                    noise,
                    latent_image,
                    denoise_mask,
                    sigma_v,
                    sigma_a,
                    split,
                    model_options,
                )

            # the model derives the audio timestep from the video sigma and the two flow
            # shifts; pinning the video shift to 1.0 makes that mapping solvable for the
            # audio shift that reproduces this evaluation's audio sigma exactly
            transformer_options["minimax_h3_sigma_shift_video"] = 1.0
            transformer_options["minimax_h3_sigma_shift_audio"] = (
                audio_shift_for(sigma_v, sigma_a)
            )
            denoised = model_wrap(
                x_in, s_in * sigma_v, model_options=model_options, seed=seed
            )

            # calculate_denoised applied the video sigma to the whole pack; recover the
            # velocity and re-solve the audio stream's x0 at its own sigma
            velocity = (x_in - denoised) / sigma_v
            denoised = torch.cat(
                [
                    denoised[..., :split],
                    x_in[..., split:] - velocity[..., split:] * sigma_a,
                ],
                dim=-1,
            )

            if denoise_mask is not None:
                denoised = denoised * denoise_mask + latent_image * (
                    1.0 - denoise_mask
                )

            if step is not None:
                if callback is not None:
                    # Sampler.sample gets ComfyUI's callback directly, not k-diffusion's dict form
                    callback(step, denoised, x, total_steps)
                progress.update(1)

            # a stream that runs out first is pinned where it landed and keeps conditioning
            # the other; only when both are done does the loop end
            if req_v is not None:
                try:
                    req_v = gen_v.send(denoised[..., :split])
                except StopIteration as stop:
                    req_v, done_v = None, stop.value
                    held_v = done_v
                    last_v = float(video_sigmas[-1])
                    held_sigma_v = max(last_v, FROZEN_VIDEO_SIGMA)
            if req_a is not None:
                try:
                    req_a = gen_a.send(denoised[..., split:])
                except StopIteration as stop:
                    req_a, done_a = None, stop.value
                    held_a = done_a
                    held_sigma_a = float(audio_sigmas[-1])
            if req_v is None and req_a is None:
                break
        progress.close()

        return torch.cat(
            [
                model_sampling.inverse_noise_scaling(video_sigmas[-1], done_v),
                model_sampling.inverse_noise_scaling(audio_sigmas[-1], done_a),
            ],
            dim=-1,
        )

    def apply_mask(
        self,
        inner_model,
        x,
        noise,
        latent_image,
        denoise_mask,
        sigma_v,
        sigma_a,
        split,
        model_options,
    ):
        """KSamplerX0Inpaint's masked blend, with each stream pinned at its own sigma."""
        if "denoise_mask_function" in model_options:
            denoise_mask = model_options["denoise_mask_function"](
                sigma_v,
                denoise_mask,
                extra_options={
                    "model": inner_model,
                    "sigmas": self.audio_sigmas,
                },
            )
        latent_mask = 1.0 - denoise_mask
        pinned = torch.cat(
            [
                inner_model.scale_latent_inpaint(
                    x=x[..., :split],
                    sigma=x.new_tensor([sigma_v]),
                    noise=noise[..., :split],
                    latent_image=latent_image[..., :split],
                ),
                inner_model.scale_latent_inpaint(
                    x=x[..., split:],
                    sigma=x.new_tensor([sigma_a]),
                    noise=noise[..., split:],
                    latent_image=latent_image[..., split:],
                ),
            ],
            dim=-1,
        )
        return x * denoise_mask + pinned * latent_mask
