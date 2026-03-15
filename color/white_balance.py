from .utils import _apply_white_balance_cat


class SuperWhiteBalanceCAT:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
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

    def apply(self, image, temperature_k=6500, tint=0.0):
        out = _apply_white_balance_cat(image, float(temperature_k), float(tint))
        return (out,)


NODE_CLASS_MAPPINGS = {"SuperWhiteBalanceCAT": SuperWhiteBalanceCAT}

NODE_DISPLAY_NAME_MAPPINGS = {"SuperWhiteBalanceCAT": "🐧 Adjust White Balance"}
