from .utils import _apply_brightness_contrast_gamma


class SuperBrightnessContrast:
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
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "SuperNodes/Color"

    def apply(self, image, brightness=1.0, contrast=1.0, gamma=1.0):
        return (
            _apply_brightness_contrast_gamma(
                image, brightness, contrast, gamma
            ),
        )


NODE_CLASS_MAPPINGS = {"SuperBrightnessContrast": SuperBrightnessContrast}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SuperBrightnessContrast": "🐧 Adjust Brightness Contrast Gamma"
}
