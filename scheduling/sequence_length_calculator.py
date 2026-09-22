import math

from comfy_api.latest import io

# (spatial compression, patch size, temporal compression)
FAMILIES = {
    "Flux": (8, 2, 1),  # Flux 1/2, Qwen-Image 1 and 2.1, Krea 2
    # LTX-Video and LTX2: 32x spatial, 8x temporal, no patchify. 2.5 keeps the
    # same geometry but turned dynamic shifting off in the diffusers config
    "LTX2": (32, 1, 8),
}
MANUAL = "Manual"


def _token_count(
    width: int,
    height: int,
    length: int,
    spatial: int,
    patch: int,
    temporal: int,
) -> int:
    # The VAE compresses first, then the transformer patchifies what is left.
    # Both round up, matching pad_to_patch_size and the (h + patch // 2) // patch
    # rounding the models use, so odd sizes count the padded token rather than
    # dropping it. Temporal patch size is 1 on every model that reaches here.
    tokens_w = math.ceil(math.ceil(width / spatial) / patch)
    tokens_h = math.ceil(math.ceil(height / spatial) / patch)
    return math.ceil(length / temporal) * tokens_w * tokens_h


class SequenceLengthCalculator(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SequenceLengthCalculator",
            display_name="🐧 Sequence Length Calculator",
            category="SuperNodes/Scheduling",
            description="Works out the token count to feed a dynamic shift scheduler.",
            search_aliases=[
                "sequence length",
                "seq len",
                "tokens",
                "image_seq_len",
                "dynamic shift",
                "latent tokens",
            ],
            inputs=[
                io.DynamicCombo.Input(
                    "model_type",
                    tooltip="Flux is any image model at 16 px per token. LTX2 is 32 px per token with 8 frames per latent frame.",
                    options=[
                        io.DynamicCombo.Option(
                            "Flux",
                            [
                                io.Int.Input(
                                    "width",
                                    default=1024,
                                    min=16,
                                    max=16_384,
                                    step=16,
                                ),
                                io.Int.Input(
                                    "height",
                                    default=1024,
                                    min=16,
                                    max=16_384,
                                    step=16,
                                ),
                            ],
                        ),
                        io.DynamicCombo.Option(
                            "LTX2",
                            [
                                io.Int.Input(
                                    "width",
                                    default=768,
                                    min=32,
                                    max=16_384,
                                    step=32,
                                ),
                                io.Int.Input(
                                    "height",
                                    default=512,
                                    min=32,
                                    max=16_384,
                                    step=32,
                                ),
                                io.Int.Input(
                                    "length",
                                    default=97,
                                    min=1,
                                    max=16_384,
                                    step=8,
                                ),
                            ],
                        ),
                        io.DynamicCombo.Option(
                            MANUAL,
                            [
                                io.Int.Input(
                                    "width",
                                    default=1024,
                                    min=8,
                                    max=16_384,
                                    step=8,
                                ),
                                io.Int.Input(
                                    "height",
                                    default=1024,
                                    min=8,
                                    max=16_384,
                                    step=8,
                                ),
                                io.Int.Input(
                                    "length",
                                    default=1,
                                    min=1,
                                    max=16_384,
                                    step=1,
                                    tooltip="Frame count. 1 for image models.",
                                ),
                                io.Int.Input(
                                    "spatial_compression",
                                    default=8,
                                    min=1,
                                    max=256,
                                    step=1,
                                    tooltip="How many pixels per latent pixel the VAE folds away.",
                                ),
                                io.Int.Input(
                                    "patch_size",
                                    default=2,
                                    min=1,
                                    max=16,
                                    step=1,
                                    tooltip="How many latent pixels per side the transformer folds into one token.",
                                ),
                                io.Int.Input(
                                    "temporal_compression",
                                    default=1,
                                    min=1,
                                    max=64,
                                    step=1,
                                    tooltip="How many frames per latent frame the VAE folds away.",
                                ),
                            ],
                        ),
                    ],
                ),
            ],
            outputs=[
                io.Int.Output(
                    display_name="seq_length",
                    tooltip="Token count of the latent.",
                ),
            ],
        )

    @classmethod
    def execute(cls, model_type) -> io.NodeOutput:
        family = model_type["model_type"]

        if family == MANUAL:
            spatial = model_type["spatial_compression"]
            patch = model_type["patch_size"]
            temporal = model_type["temporal_compression"]
        else:
            spatial, patch, temporal = FAMILIES[family]

        return io.NodeOutput(
            _token_count(
                model_type["width"],
                model_type["height"],
                # Only the video and manual options carry a frame count
                model_type.get("length", 1),
                spatial,
                patch,
                temporal,
            )
        )


NODE = [SequenceLengthCalculator]
