import re

import comfy.utils
from comfy_api.latest import io
import torch


class SuperPadImage(io.ComfyNode):
    """
    Places an image onto an exact target canvas size and returns:

    IMAGE: Final image at (target_width, target_height)
    MASK:  1.0 = padded pixels, 0.0 = image pixels

    Behavior:
    - Image is first scaled to "contain" inside target (aspect preserved).
    - Then scaled again by scale_factor (0.1–1.0).
        * 1.0 = normal fill/letterbox
        * <1.0 = shrink image to introduce padding on both axes
    - Image is placed using horizontal/vertical shift.
    """

    upscale_methods = [
        "nearest-exact",
        "bilinear",
        "area",
        "bicubic",
        "lanczos",
    ]

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SuperPadImage",
            display_name="🐧 Pad Image Scaled",
            category="SuperNodes/Image",
            inputs=[
                io.Image.Input("image", tooltip="The input image."),
                io.Int.Input(
                    "target_width",
                    default=1024,
                    min=1,
                    max=16384,
                    step=1,
                    tooltip="Final output width in pixels.",
                ),
                io.Int.Input(
                    "target_height",
                    default=1024,
                    min=1,
                    max=16384,
                    step=1,
                    tooltip="Final output height in pixels.",
                ),
                io.Float.Input(
                    "shift_horizontal",
                    default=0.0,
                    min=-1.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Horizontal placement: -1 = far left, 0 = center, 1 = far right.",
                ),
                io.Float.Input(
                    "shift_vertical",
                    default=0.0,
                    min=-1.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Vertical placement: -1 = bottom, 0 = center, 1 = top.",
                ),
                io.Float.Input(
                    "scale_factor",
                    default=1.0,
                    min=0.1,
                    max=1.0,
                    step=0.01,
                    tooltip="Additional scale after fitting. 1.0 = normal fill. Smaller values shrink image to create padding.",
                ),
                io.Combo.Input(
                    "scale_method",
                    options=cls.upscale_methods,
                    default="nearest-exact",
                    tooltip="Resampling method used for resizing.",
                ),
                io.String.Input(
                    "color",
                    default="#808080",
                    multiline=False,
                    tooltip="Padding color as hex (#RRGGBB, RRGGBB, #RGB, RGB). Invalid values default to white.",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="IMAGE"),
                io.Mask.Output(display_name="MASK"),
            ],
        )

    @classmethod
    def execute(
        cls,
        image,
        target_width,
        target_height,
        shift_horizontal,
        shift_vertical,
        scale_factor,
        scale_method,
        color,
    ) -> io.NodeOutput:
        shift_horizontal = float(max(-1.0, min(1.0, shift_horizontal)))
        shift_vertical = float(max(-1.0, min(1.0, shift_vertical)))
        scale_factor = float(max(0.1, min(1.0, scale_factor)))

        b, h, w, c = image.shape
        device = image.device
        dtype = image.dtype

        pad_rgb = cls._parse_hex_color(color)

        # Contain scale
        contain_scale = min(target_width / w, target_height / h)
        cw = max(1, int(round(w * contain_scale)))
        ch = max(1, int(round(h * contain_scale)))

        # Extra scale
        fw = min(target_width, max(1, int(round(cw * scale_factor))))
        fh = min(target_height, max(1, int(round(ch * scale_factor))))

        resized = cls._resize(image[..., :3], fw, fh, scale_method)

        canvas = torch.empty(
            (b, target_height, target_width, 3), device=device, dtype=dtype
        )
        canvas[..., 0].fill_(pad_rgb[0])
        canvas[..., 1].fill_(pad_rgb[1])
        canvas[..., 2].fill_(pad_rgb[2])

        mask = torch.ones(
            (b, target_height, target_width), device=device, dtype=dtype
        )

        dx = target_width - fw
        dy = target_height - fh

        x0 = int(round(((shift_horizontal + 1.0) * 0.5) * dx)) if dx > 0 else 0
        y0 = int(round(((1.0 - shift_vertical) * 0.5) * dy)) if dy > 0 else 0

        x1, y1 = x0, y0
        x2, y2 = x0 + fw, y0 + fh

        canvas[:, y1:y2, x1:x2] = resized
        mask[:, y1:y2, x1:x2] = 0.0

        return io.NodeOutput(canvas, mask)

    @classmethod
    def _resize(cls, img, w, h, method):
        samples = img.movedim(-1, 1)
        out = comfy.utils.common_upscale(samples, w, h, method, "disabled")
        return out.movedim(1, -1)

    @classmethod
    def _parse_hex_color(cls, value):
        if not isinstance(value, str):
            return (1.0, 1.0, 1.0)

        s = value.strip()
        if s.startswith("#"):
            s = s[1:]

        if re.fullmatch(r"[0-9a-fA-F]{3}", s):
            s = "".join([c * 2 for c in s])

        if not re.fullmatch(r"[0-9a-fA-F]{6}", s):
            return (1.0, 1.0, 1.0)

        return (
            int(s[0:2], 16) / 255.0,
            int(s[2:4], 16) / 255.0,
            int(s[4:6], 16) / 255.0,
        )


NODE = [SuperPadImage]
