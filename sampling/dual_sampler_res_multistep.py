from comfy_api.latest import io

from .utils import DualSampler, DualSamplerType, res_multistep_stream


class DualSamplerResMultistep(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DualSamplerResMultistep",
            display_name="🐧 DualSamplerResMultistep",
            category="SuperNodes/Sampling",
            description="res_multistep with separate noise settings for the video and audio streams.",
            inputs=[
                io.Float.Input(
                    "eta_video",
                    default=0.0,
                    min=0.0,
                    max=100.0,
                    step=0.01,
                    round=False,
                    tooltip="Ancestral noise for the video stream, using the flow-model split. 0 is res_multistep.",
                ),
                io.Float.Input(
                    "eta_audio",
                    default=0.0,
                    min=0.0,
                    max=100.0,
                    step=0.01,
                    round=False,
                    tooltip="Ancestral noise for the audio stream, using the flow-model split. 0 is res_multistep.",
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
                stream_fn=res_multistep_stream,
            )
        )


NODE = [DualSamplerResMultistep]
