import comfy.model_management


class SetReserveVRAM:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "any": ("*",),
                "reserved_gb": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1024.0, "step": 0.1},
                ),
            }
        }

    RETURN_TYPES = ("*",)
    FUNCTION = "set_vram"
    CATEGORY = "SuperNodes/Tools"
    DESCRIPTION = "Sets --reserve-vram dynamically anywhere in a workflow."

    def set_vram(self, any, reserved_gb):
        comfy.model_management.EXTRA_RESERVED_VRAM = (
            reserved_gb * 1024 * 1024 * 1024
        )
        return (any,)


NODE_CLASS_MAPPINGS = {"SetReserveVRAM": SetReserveVRAM}

NODE_DISPLAY_NAME_MAPPINGS = {"SetReserveVRAM": "🐧 Set Reserve VRAM"}
