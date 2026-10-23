from comfy_api.latest import io

from .utils import DualSampler, DualSamplerType, er_sde_stream


class DualSamplerERSDE(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DualSamplerERSDE",
            display_name="🐧 DualSamplerERSDE",
            category="SuperNodes/Sampling",
            description="ER-SDE with separate noise settings for the video and audio streams.",
            inputs=[
                io.Float.Input(
                    "eta_video",
                    default=1.0,
                    min=0.0,
                    max=10.0,
                    step=0.01,
                    round=False,
                    tooltip="Stochastic strength for the video stream. 0 is the deterministic ODE.",
                ),
                io.Float.Input(
                    "eta_audio",
                    default=1.0,
                    min=0.0,
                    max=10.0,
                    step=0.01,
                    round=False,
                    tooltip="Stochastic strength for the audio stream. 0 is the deterministic ODE.",
                ),
                io.Float.Input(
                    "s_noise_video",
                    default=1.0,
                    min=0.0,
                    max=100.0,
                    step=0.01,
                    round=False,
                    tooltip="Scales the noise added to the video stream.",
                ),
                io.Float.Input(
                    "s_noise_audio",
                    default=1.0,
                    min=0.0,
                    max=100.0,
                    step=0.01,
                    round=False,
                    tooltip="Scales the noise added to the audio stream.",
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
        cls, eta_video, eta_audio, s_noise_video, s_noise_audio
    ) -> io.NodeOutput:
        return io.NodeOutput(
            DualSampler(
                eta_video,
                eta_audio,
                s_noise_video,
                s_noise_audio,
                stream_fn=er_sde_stream,
            )
        )


NODE = [DualSamplerERSDE]
