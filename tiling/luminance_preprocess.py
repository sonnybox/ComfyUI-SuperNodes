from comfy_api.latest import io
import torch

from .utils import rgb_to_lab


class LuminancePreprocess(io.ComfyNode):
    """
    Extracts the perceptual lightness (luminance) from an image
    and outputs it as a 3-channel grayscale image.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SuperLuminancePreprocess",
            display_name="🐧 Luminance Preprocess",
            category="SuperNodes/Tiling",
            inputs=[
                io.Image.Input(
                    "image", tooltip="The original, colored RGB image."
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="image",
                    tooltip="The grayscale luminance image.",
                ),
            ],
        )

    @classmethod
    def execute(cls, image: torch.Tensor) -> io.NodeOutput:
        # 1-2. Convert image from RGB to LAB color space
        lab = rgb_to_lab(image)

        # 3. Extract ONLY the 'L' (Luminance/Lightness) channel. Shape: [B, H, W, 1]
        L = lab[..., 0:1]

        # 4. Normalize L channel from [0, 100] to standard [0.0, 1.0] range
        L_norm = L / 100.0

        # Clamp just to be safe to strictly stay in bounds
        L_norm = torch.clamp(L_norm, min=0.0, max=1.0)

        # 5. Duplicate the single 'L' channel 3 times along the channel dimension
        #    so it looks purely grayscale but remains a 3-channel tensor (R=L, G=L, B=L)
        # Using repeat ensures contiguous memory unlike expand.
        L_3ch = L_norm.repeat(1, 1, 1, 3)

        # 6. Return the final 3-channel grayscale image
        return io.NodeOutput(L_3ch)


NODE = [LuminancePreprocess]
