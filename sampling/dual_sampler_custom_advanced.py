import logging

import comfy.model_management
import comfy.nested_tensor
import comfy.sample
import comfy.utils
import numpy
from comfy_api.latest import io
import latent_preview

from .utils import DualSampler, DualSamplerType, check_schedules

# Stamped onto the output so a later pass can tell whether it is resuming from the right
# noise level.
END_SIGMAS_KEY = "dual_end_sigmas"


def decimals(value):
    """How many decimals the value was written with, at float32 precision."""
    text = numpy.format_float_positional(numpy.float32(value), unique=True, trim="-")
    return len(text.partition(".")[2])


def check_resume(latent, video_sigmas, audio_sigmas, noise):
    previous = latent.get(END_SIGMAS_KEY)
    if previous is None:
        return
    for stream, was, now, added in (
        ("video", previous[0], float(video_sigmas[0]), noise[0]),
        ("audio", previous[1], float(audio_sigmas[0]), noise[1]),
    ):
        if was == 0.0 or bool(added.any()):
            # re-noised, so whatever noise level the latent was left at no longer matters
            continue
        # a schedule written to fewer decimals than the one it resumes still matches, so
        # compare both at the coarser of the two, never coarser than the 4 decimals the
        # sigma widgets expose
        places = max(4, min(decimals(was), decimals(now)))
        if round(was, places) == round(now, places):
            continue
        logging.warning(
            "%s sigma at index 0 (%g) does not match the %s latent (%g).",
            stream,
            now,
            stream,
            was,
        )


def stream_noise(source, latent):
    """A NOISE source's output as a nested pair, whatever shape it hands back."""
    generated = source.generate_noise(latent)
    if generated.is_nested:
        return generated
    shapes = [t.shape for t in latent["samples"].unbind()]
    return comfy.nested_tensor.NestedTensor(
        comfy.utils.unpack_latents(generated, shapes)
    )


def audio_seed(noise_audio, noise_video, added):
    """The seed the audio stream's own sampler noise is drawn from."""
    if not bool(added.any()):
        return noise_video.seed
    return noise_audio.seed


def single_stream(packed, index):
    """One stream of a packed AV latent, as a plain LATENT for decoding on its own."""
    out = {k: v for k, v in packed.items() if k != "noise_mask"}
    out["samples"] = packed["samples"].unbind()[index]
    return out


class DualSamplerCustomAdvanced(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DualSamplerCustomAdvanced",
            display_name="🐧 DualSamplerCustomAdvanced",
            category="SuperNodes/Sampling",
            description="Samples a packed audio-video latent (e.g. MiniMax H3) with a separate sigma schedule per stream.",
            inputs=[
                io.Noise.Input(
                    "noise_video",
                ),
                io.Noise.Input(
                    "noise_audio",
                ),
                io.Guider.Input("guider"),
                DualSamplerType.Input("dual_sampler"),
                io.Sigmas.Input("video_sigmas"),
                io.Sigmas.Input(
                    "audio_sigmas",
                    tooltip="Use the same number of steps as video_sigmas.",
                ),
                io.Latent.Input(
                    "av_latent", tooltip="Packed video + audio latent."
                ),
            ],
            outputs=[
                io.Latent.Output(
                    display_name="output",
                ),
                io.Latent.Output(
                    display_name="denoised_output",
                ),
                io.Latent.Output(display_name="video_output"),
                io.Latent.Output(display_name="audio_output"),
                io.Latent.Output(display_name="denoised_video_output"),
                io.Latent.Output(display_name="denoised_audio_output"),
            ],
        )

    @classmethod
    def execute(
        cls,
        noise_video,
        noise_audio,
        guider,
        dual_sampler,
        video_sigmas,
        audio_sigmas,
        av_latent,
    ) -> io.NodeOutput:
        if not isinstance(dual_sampler, DualSampler):
            raise ValueError("dual_sampler must come from a Dual Sampler node.")
        check_schedules(video_sigmas, audio_sigmas)

        model = guider.model_patcher.model
        if not hasattr(model, "audio_scale"):
            raise ValueError(
                "{} does not sample two streams; use SamplerCustomAdvanced.".format(
                    model.__class__.__name__
                )
            )

        latent = av_latent
        latent_image = latent["samples"]
        latent = latent.copy()
        latent_image = comfy.sample.fix_empty_latent_channels(
            guider.model_patcher,
            latent_image,
            latent.get("downscale_ratio_spacial", None),
            latent.get("downscale_ratio_temporal", None),
        )
        latent["samples"] = latent_image
        if not latent_image.is_nested or len(latent_image.unbind()) < 2:
            raise ValueError(
                "av_latent must be a packed audio-video latent (e.g. EmptyMiniMaxH3LatentAV)."
            )

        # one noise source per stream, so a stream carrying leftover noise from an earlier
        # pass can be resumed while the other is re-noised
        streams = list(stream_noise(noise_video, latent).unbind())
        streams[1] = stream_noise(noise_audio, latent).unbind()[1]
        noise = comfy.nested_tensor.NestedTensor(streams)

        check_resume(latent, video_sigmas, audio_sigmas, streams)
        noise_mask = latent.get("noise_mask", None)

        x0_output = {}
        callback = latent_preview.prepare_callback(
            guider.model_patcher, video_sigmas.shape[-1] - 1, x0_output
        )
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        sampler = dual_sampler.with_audio_sigmas(
            audio_sigmas, audio_seed(noise_audio, noise_video, streams[1])
        )

        # The audio stream is sampled at its own noise level, so process_latent_in/out and the
        # DiT must not carry it onto the video schedule. Every process_latent_out has to run
        # while this is installed, the x0 estimate included, or its audio comes back scaled.
        model.audio_scale = lambda: 1.0
        try:
            samples = guider.sample(
                noise,
                latent_image,
                sampler,
                video_sigmas,
                denoise_mask=noise_mask,
                callback=callback,
                disable_pbar=disable_pbar,
                seed=noise_video.seed,
            )
            samples = samples.to(comfy.model_management.intermediate_device())
            denoised = None
            if "x0" in x0_output:
                x0 = x0_output["x0"]
                if samples.is_nested and not x0.is_nested:
                    shapes = [x.shape for x in samples.unbind()]
                    x0 = comfy.nested_tensor.NestedTensor(
                        comfy.utils.unpack_latents(x0, shapes)
                    )
                denoised = model.process_latent_out(x0.cpu())
        finally:
            del model.audio_scale

        out = latent.copy()
        out.pop("downscale_ratio_spacial", None)
        out.pop("downscale_ratio_temporal", None)
        out["samples"] = samples
        out[END_SIGMAS_KEY] = [float(video_sigmas[-1]), float(audio_sigmas[-1])]

        if denoised is None:
            out_denoised = out
        else:
            out_denoised = latent.copy()
            out_denoised["samples"] = denoised
            out_denoised.pop(
                END_SIGMAS_KEY, None
            )  # x0 is clean, not stopped mid-schedule

        return io.NodeOutput(
            out,
            out_denoised,
            single_stream(out, 0),
            single_stream(out, 1),
            single_stream(out_denoised, 0),
            single_stream(out_denoised, 1),
        )


NODE = [DualSamplerCustomAdvanced]
