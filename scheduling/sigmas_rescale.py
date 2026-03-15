import torch


class SigmasRescale:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sigmas": (
                    "SIGMAS",
                    {"tooltip": "The input sigma schedule to be rescaled."},
                ),
                "max": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 10000.0,
                        "step": 0.01,
                        "tooltip": "The new maximum value (start of the schedule).",
                    },
                ),
                "min": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1000.0,
                        "step": 0.001,
                        "tooltip": "The new minimum value (end of the schedule).",
                    },
                ),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    OUTPUT_TOOLTIPS = ("The rescaled sigma schedule.",)
    FUNCTION = "rescale"

    CATEGORY = "SuperNodes/Scheduling"
    DESCRIPTION = "Rescales a sigma schedule to a new maximum and minimum range while preserving the exact curve of the original schedule."

    def rescale(self, sigmas, max, min):
        # Avoid modifying the original tensor
        s = sigmas.clone()

        # Get the current range of the input sigmas
        # Sigmas usually go from High to Low, so index 0 is max, index -1 is min
        current_max = s[0]
        current_min = s[-1]

        # Handle edge case where max equals min to avoid division by zero
        if current_max == current_min:
            # If the schedule is flat, return a flat schedule at the new max
            return (torch.full_like(s, max),)

        # Normalize the curve to 0.0 - 1.0
        # Formula: (value - min) / (max - min)
        normalized_curve = (s - current_min) / (current_max - current_min)

        # Scale to the new range
        # Formula: normalized * (new_max - new_min) + new_min
        new_sigmas = normalized_curve * (max - min) + min

        return (new_sigmas,)


NODE_CLASS_MAPPINGS = {"SigmasRescale": SigmasRescale}

NODE_DISPLAY_NAME_MAPPINGS = {"SigmasRescale": "🐧 Sigmas Rescale"}
