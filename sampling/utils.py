"""Shared machinery for the dual-schedule sampler (MiniMax H3).

H3 denoises a video stream and an audio stream in one DiT forward, and the two were trained
on different flow shifts (12.0 video / 3.0 audio). Core keeps a single SIGMAS input by
carrying the audio latent onto the video schedule, which locks the audio schedule to a
re-shift of the video one. This sampler takes both schedules and steps each stream on its
own, without patching core:

- the carry is switched off by shadowing MiniMaxH3.audio_scale with 1.0 for the run, so the
  sampler holds the real audio latent at its own sigma
- the audio noise level reaches the DiT through the flow shifts it already reads from
  transformer_options: the video shift is pinned to 1.0 (identity) and the audio shift
  solved so the model's own mapping lands on the step's audio sigma
- the audio stream's x0 is re-derived from the returned velocity at the audio sigma, since
  calculate_denoised applies the single timestep sigma to the whole pack

Both schedules are used exactly as given. An all-zero schedule freezes its stream: the
latent is left alone and the DiT is told that stream is clean, which is how one stream is
refined while the other only conditions it.
"""

import copy
import math

from comfy.k_diffusion import sampling as k_diffusion_sampling
import comfy.model_sampling
import comfy.samplers
from comfy_api.latest import io
import torch
from tqdm.auto import trange

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
    if video_sigmas.shape[-1] == 0:
        return
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


class StreamOptions:
    """Per-stream knobs and noise source handed to the sampler coroutine."""

    def __init__(self, eta, s_noise, noise):
        self.eta, self.s_noise, self.noise = eta, s_noise, noise


class DualSampler(comfy.samplers.Sampler):
    """Steps a packed [video, audio] latent on two independent sigma schedules.

    The video schedule arrives through the normal `sigmas` argument; the audio schedule is
    attached by DualSamplerCustomAdvanced via with_audio_sigmas().
    """

    def __init__(
        self, eta_video=1.0, eta_audio=1.0, s_noise_video=1.0, s_noise_audio=1.0
    ):
        self.eta_video = eta_video
        self.eta_audio = eta_audio
        self.s_noise_video = s_noise_video
        self.s_noise_audio = s_noise_audio
        self.audio_sigmas = None

    def with_audio_sigmas(self, audio_sigmas):
        sampler = copy.copy(self)
        sampler.audio_sigmas = audio_sigmas
        return sampler

    def stream_options(self, x, split, video, seed, model_sampling):
        """Noise source for one stream.

        The sampler is built on the *packed* shape and sliced, so with equal schedules the
        two streams draw exactly the tensor core's single sampler would have drawn.
        """
        region = slice(None, split) if video else slice(split, None)
        draw = k_diffusion_sampling.default_noise_sampler(x, seed=seed)
        noise_scale = getattr(model_sampling, "noise_scale", 1.0)
        return StreamOptions(
            eta=self.eta_video if video else self.eta_audio,
            s_noise=(self.s_noise_video if video else self.s_noise_audio)
            * noise_scale,
            noise=lambda a, b: draw(a, b)[..., region],
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

        # an all-zero schedule freezes its stream: the latent is already x0, so it is never
        # stepped and the DiT is told t = 1 (clean) for it every pass
        frozen_v, frozen_a = is_frozen(video_sigmas), is_frozen(audio_sigmas)
        if frozen_v and frozen_a:
            raise ValueError(
                "both schedules are all zeros, so there is nothing to sample."
            )

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
        held_v, held_a = x[..., :split], x[..., split:]

        # a frozen stream is never stepped, so it needs no noise source
        opts_v = (
            None
            if frozen_v
            else self.stream_options(x, split, True, seed, model_sampling)
        )
        opts_a = (
            None
            if frozen_a
            else self.stream_options(x, split, False, seed, model_sampling)
        )

        gen_v = (
            None if frozen_v else ancestral_stream(held_v, video_sigmas, opts_v)
        )
        gen_a = (
            None if frozen_a else ancestral_stream(held_a, audio_sigmas, opts_a)
        )

        progress = trange(total_steps, disable=disable_pbar)
        req_v = next(gen_v) if gen_v is not None else None
        req_a = next(gen_a) if gen_a is not None else None
        done_v = held_v if frozen_v else None
        done_a = held_a if frozen_a else None
        while True:
            step = (req_v or req_a)[0]
            sigma_v, xv = (
                (req_v[1], req_v[2]) if req_v is not None else (0.0, held_v)
            )
            sigma_a, xa = (
                (req_a[1], req_a[2]) if req_a is not None else (0.0, held_a)
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

            if gen_v is not None:
                try:
                    req_v = gen_v.send(denoised[..., :split])
                except StopIteration as stop:
                    req_v, done_v = None, stop.value
            if gen_a is not None:
                try:
                    req_a = gen_a.send(denoised[..., split:])
                except StopIteration as stop:
                    req_a, done_a = None, stop.value
            if req_v is None and req_a is None:
                break
            if (
                gen_v is not None
                and gen_a is not None
                and (req_v is None) != (req_a is None)
            ):
                raise RuntimeError(
                    "dual sampler streams desynchronised; both schedules must have the same length"
                )
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
