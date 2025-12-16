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
    RETURN_NAMES = ("sigmas",)
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
            # Decay interpolation: /2, /3, /4, etc.
            # user example: 1 step -> last/2. 2 steps -> last/2, last/3
            for i in range(1, smooth_steps + 1):
                divisor = i + 1
                new_val = last_sigma / divisor
                new_steps.append(new_val)

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
