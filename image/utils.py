import comfy.utils

# Matches core's ImageScale/ImageScaleBy so our nodes behave identically to
# the built-ins users already know.
SCALE_METHODS = [
    "nearest-exact",
    "bilinear",
    "area",
    "bicubic",
    "lanczos",
]


def scale_image(image, width, height, method, crop="disabled"):
    """Resize an IMAGE tensor [B, H, W, C]."""
    samples = image.movedim(-1, 1)
    out = comfy.utils.common_upscale(samples, width, height, method, crop)
    return out.movedim(1, -1)


def scale_mask(mask, width, height, method, crop="disabled"):
    """Resize a MASK tensor [B, H, W], or [B, 1, H, W] as returned by some nodes."""
    samples = mask if mask.dim() == 4 else mask.unsqueeze(1)
    out = comfy.utils.common_upscale(samples, width, height, method, crop)
    # comfy's lanczos squeezes single-channel input and hands back [B, H, W];
    # every other method keeps the channel dim.
    return out.squeeze(1) if out.dim() == 4 else out
