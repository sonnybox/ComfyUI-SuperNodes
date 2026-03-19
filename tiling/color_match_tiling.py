from comfy_api.latest import io
import torch
import torch.nn.functional as F

from .utils import lab_to_rgb, rgb_to_lab


class ColorMatchLuminance(io.ComfyNode):
    """
    Performs a luminance swap to fix color shifts after tiled upscaling.
    Takes L channel from the target image and matches A/B from reference.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SuperColorMatchLuminance",
            display_name="🐧 Color Match Luminance",
            category="SuperNodes/Tiling",
            inputs=[
                io.Image.Input("target", tooltip="The upscaled image."),
                io.Image.Input(
                    "reference",
                    tooltip="The original image with correct colors.",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
            ],
        )

    @classmethod
    def execute(
        cls, target: torch.Tensor, reference: torch.Tensor
    ) -> io.NodeOutput:
        # 1-2. Check dimensions and resize reference if smaller/different
        b_t, h_t, w_t, c_t = target.shape
        b_r, h_r, w_r, c_r = reference.shape

        if h_r != h_t or w_r != w_t:
            # interpolate accepts [B, C, H, W]
            reference_permuted = reference.permute(0, 3, 1, 2)
            reference_resized = F.interpolate(
                reference_permuted,
                size=(h_t, w_t),
                mode="bicubic",
                align_corners=False,
            )
            reference = reference_resized.permute(0, 2, 3, 1).clamp(0.0, 1.0)

        # 3. Convert both to LAB
        target_lab = rgb_to_lab(target)
        reference_lab = rgb_to_lab(reference)

        # 4, 5, 6. L from target, A, B from reference
        L_target = target_lab[..., 0:1]
        AB_reference = reference_lab[..., 1:3]

        merged_lab = torch.cat([L_target, AB_reference], dim=-1)

        # 7. Convert LAB back to RGB
        merged_rgb = lab_to_rgb(merged_lab)

        # 8. Clamp final RGB values between 0.0 and 1.0
        final_image = torch.clamp(merged_rgb, min=0.0, max=1.0)

        # 9. Return the final merged image
        return io.NodeOutput(final_image)


NODE = [ColorMatchLuminance]
