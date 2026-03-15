class SigmaReplace:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sigmas": (
                    "SIGMAS",
                    {"tooltip": "Input sigma schedule."},
                ),
                "index": (
                    "INT",
                    {
                        "default": 0,
                        "min": -1,
                        "max": 10_000,
                        "step": 1,
                        "tooltip": "Index to replace. 0 = first, -1 = last.",
                    },
                ),
                "value": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 10_000.0,
                        "step": 0.01,
                        "tooltip": "New sigma value to replace at the given index.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    OUTPUT_TOOLTIPS = ("Sigma schedule with one value replaced.",)
    FUNCTION = "replace"

    CATEGORY = "SuperNodes/Scheduling"
    DESCRIPTION = "Replaces a single sigma value at the specified index. Index -1 is last."

    def replace(self, sigmas, index, value):
        s = sigmas.clone()
        length = s.shape[0]

        # Normalize negative index
        if index < 0:
            index = length + index

        # Bounds check
        if index < 0 or index >= length:
            raise IndexError(f"Sigma index out of range: {index}")

        # Validate against previous sigma
        if index > 0:
            prev_sigma = s[index - 1]
            if value > prev_sigma:
                raise ValueError(
                    f"Invalid sigma schedule: sigma[{index}] = {value} "
                    f"is greater than preceding sigma[{index - 1}] = {prev_sigma}"
                )

        # Validate against next sigma
        if index < length - 1:
            next_sigma = s[index + 1]
            if value < next_sigma:
                raise ValueError(
                    f"Invalid sigma schedule: sigma[{index}] = {value} "
                    f"is less than following sigma[{index + 1}] = {next_sigma}"
                )

        s[index] = value
        return (s,)


NODE_CLASS_MAPPINGS = {"SigmaReplace": SigmaReplace}

NODE_DISPLAY_NAME_MAPPINGS = {"SigmaReplace": "🐧 Sigma Replace"}
