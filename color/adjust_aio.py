from .utils import (
    _apply_brightness_contrast_gamma,
    _apply_saturation_hue,
    _apply_white_balance_cat,
)


class SuperColorAdjustAllInOne:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "brightness": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01},
                ),
                "contrast": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01},
                ),
                "gamma": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.05, "max": 4.0, "step": 0.01},
                ),
                "saturation": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01},
                ),
                "hue_degrees": (
                    "FLOAT",
                    {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.5},
                ),
                "temperature_k": (
                    "INT",
                    {"default": 6500, "min": 1650, "max": 25000, "step": 50},
                ),
                "tint": (
                    "FLOAT",
                    {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "SuperNodes/Color"

    def apply(
        self,
        image,
        brightness=1.0,
        contrast=1.0,
        gamma=1.0,
        saturation=1.0,
        hue_degrees=0.0,
        temperature_k=6500,
        tint=0.0,
    ):
        out = _apply_brightness_contrast_gamma(
            image, brightness, contrast, gamma
        )
        out = _apply_saturation_hue(out, saturation, hue_degrees)
        out = _apply_white_balance_cat(out, float(temperature_k), float(tint))
        return (out,)


NODE_CLASS_MAPPINGS = {"SuperColorAdjustAllInOne": SuperColorAdjustAllInOne}

NODE_DISPLAY_NAME_MAPPINGS = {"SuperColorAdjustAllInOne": "🐧 Adjust Color AIO"}
