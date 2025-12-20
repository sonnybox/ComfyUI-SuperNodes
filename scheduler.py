import torch


class SigmaSmoother:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sigmas": (
                    "SIGMAS",
                    {
                        "tooltip": "The input sigmas tensor, typically ending in 0.0."
                    },
                ),
                "smooth_steps": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 100,
                        "tooltip": "Number of additional smoothed steps to insert between the last non-zero sigma and 0.0.",
                    },
                ),
                "interpolation_type": (
                    ["linear", "decay"],
                    {
                        "default": "linear",
                        "tooltip": "Method to calculate the intermediate sigma values.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("SIGMAS",)
    FUNCTION = "smooth_sigmas"
    CATEGORY = "SuperNodes"
    DESCRIPTION = "Inserts smoothed interpolation steps at the end of a sigma schedule before the final zero, useful for refining the final denoising steps."

    def smooth_sigmas(self, sigmas, smooth_steps, interpolation_type):
        # Ensure we are working with a float tensor
        if sigmas.dtype != torch.float32 and sigmas.dtype != torch.float64:
            sigmas = sigmas.float()

        # Check if the last element is zero and remove it for calculation
        if sigmas[-1] == 0.0:
            active_sigmas = sigmas[:-1]
        else:
            active_sigmas = sigmas

        if len(active_sigmas) == 0:
            # Edge case: empty or only zero input
            return (sigmas,)

        last_sigma = active_sigmas[-1].item()
        new_steps = []

        if interpolation_type == "linear":
            # Linear interpolation from last_sigma to 0
            # Total intervals = smooth_steps + 1 (the final drop to 0 is the +1)
            step_size = last_sigma / (smooth_steps + 1)
            for i in range(1, smooth_steps + 1):
                new_val = last_sigma - (step_size * i)
                new_steps.append(new_val)

        elif interpolation_type == "decay":
            # Decay interpolation: previous / 2, then previous / 3, etc.
            # Example: 3 -> 1.5 -> 0.5 -> 0.125
            current_val = last_sigma
            for i in range(1, smooth_steps + 1):
                divisor = i + 1
                current_val = current_val / divisor
                new_steps.append(current_val)

        # Convert new steps to tensor ensuring matching device and type
        new_sigmas_tensor = torch.tensor(
            new_steps, dtype=sigmas.dtype, device=sigmas.device
        )

        # Reconstruct: Old sigmas (minus zero) + New Steps + Zero
        parts = [active_sigmas, new_sigmas_tensor]

        # Always append 0.0 at the end as per ComfyUI sigma standards
        zero_tensor = torch.tensor(
            [0.0], dtype=sigmas.dtype, device=sigmas.device
        )
        parts.append(zero_tensor)

        result_sigmas = torch.cat(parts)

        return (result_sigmas,)


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

    CATEGORY = "SuperNodes"
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
