from .utils import _apply_saturation_hue


class SuperHueSaturation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "saturation": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01},
                ),
                "hue_degrees": (
                    "FLOAT",
                    {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.5},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "SuperNodes/Color"

    def apply(self, image, saturation=1.0, hue_degrees=0.0):
        return (_apply_saturation_hue(image, saturation, hue_degrees),)


NODE_CLASS_MAPPINGS = {"SuperHueSaturation": SuperHueSaturation}

NODE_DISPLAY_NAME_MAPPINGS = {"SuperHueSaturation": "🐧 Adjust Hue Saturation"}
