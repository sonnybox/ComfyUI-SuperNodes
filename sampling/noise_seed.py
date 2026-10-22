from comfy_api.latest import io
from comfy_extras.nodes_custom_sampler import (
    Noise_EmptyNoise,
    Noise_RandomNoise,
)


class Noise_SeededEmptyNoise(Noise_EmptyNoise):
    def __init__(self, seed):
        self.seed = seed


class NoiseSeed(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="NoiseSeed",
            display_name="🐧 Noise Seed",
            category="SuperNodes/Sampling",
            description="RandomNoise with a toggle for DisableNoise that is also seeded.",
            inputs=[
                io.Int.Input(
                    "noise_seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                ),
                io.Boolean.Input(
                    "add_noise",
                    default=True,
                ),
            ],
            outputs=[io.Noise.Output()],
        )

    @classmethod
    def execute(cls, noise_seed, add_noise) -> io.NodeOutput:
        if add_noise:
            return io.NodeOutput(Noise_RandomNoise(noise_seed))
        return io.NodeOutput(Noise_SeededEmptyNoise(noise_seed))


NODE = [NoiseSeed]
