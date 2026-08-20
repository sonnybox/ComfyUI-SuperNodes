from comfy_api.latest import io

from .utils import DualSampler, DualSamplerType, dpmpp_sde_stream


class DualSamplerDPMPP_SDE(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DualSamplerDPMPP_SDE",
            display_name="🐧 DualSamplerDPMPP_SDE",
            category="SuperNodes/Sampling",
            description="DPM-Solver++ (stochastic) with separate noise settings for the video and audio streams. Two model evaluations per step.",
            inputs=[
                io.Float.Input(
                    "eta_video",
                    default=1.0,
                    min=0.0,
                    max=100.0,
                    step=0.01,
                    round=False,
                    advanced=True,
                    tooltip="How much of the video stream's step is taken stochastically. 0 makes the step deterministic.",
                ),
                io.Float.Input(
                    "eta_audio",
                    default=1.0,
                    min=0.0,
                    max=100.0,
                    step=0.01,
                    round=False,
                    advanced=True,
                    tooltip="How much of the audio stream's step is taken stochastically. 0 makes the step deterministic.",
                ),
                io.Float.Input(
                    "s_noise_video",
                    default=1.0,
                    min=0.0,
                    max=100.0,
                    step=0.01,
                    round=False,
                    advanced=True,
                    tooltip="Scales the noise added to the video stream.",
                ),
                io.Float.Input(
                    "s_noise_audio",
                    default=1.0,
                    min=0.0,
                    max=100.0,
                    step=0.01,
                    round=False,
                    advanced=True,
                    tooltip="Scales the noise added to the audio stream.",
                ),
                io.Combo.Input(
                    "noise_device",
                    options=["cpu", "gpu"],
                    advanced=True,
                    tooltip="Where each stream's Brownian path is generated. cpu keeps a run reproducible across GPUs.",
                ),
            ],
            outputs=[
                DualSamplerType.Output(
                    tooltip="Sampler for DualSamplerCustomAdvanced."
                ),
            ],
        )

    @classmethod
    def execute(
        cls, eta_video, eta_audio, s_noise_video, s_noise_audio, noise_device
    ) -> io.NodeOutput:
        return io.NodeOutput(
            DualSampler(
                eta_video,
                eta_audio,
                s_noise_video,
                s_noise_audio,
                stream_fn=dpmpp_sde_stream,
                noise_device=noise_device,
            )
        )


NODE = [DualSamplerDPMPP_SDE]
