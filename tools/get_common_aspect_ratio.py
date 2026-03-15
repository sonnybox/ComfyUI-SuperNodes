class GetCommonAspectRatio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": 1, "max": 65536}),
                "height": ("INT", {"default": 1024, "min": 1, "max": 65536}),
                "1:1": ("BOOLEAN", {"default": True}),
                "4:3": ("BOOLEAN", {"default": True}),
                "3:2": ("BOOLEAN", {"default": True}),
                "5:4": ("BOOLEAN", {"default": True}),
                "16:9": ("BOOLEAN", {"default": True}),
                "16:10": ("BOOLEAN", {"default": True}),
                "21:9": ("BOOLEAN", {"default": True}),
                "2:1": ("BOOLEAN", {"default": True}),
                "1.85:1": ("BOOLEAN", {"default": True}),
                "2.39:1": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("aspect_w", "aspect_h")
    FUNCTION = "get_ratio"
    CATEGORY = "SuperNodes/Tools"

    def get_ratio(self, width, height, **kwargs):
        # 1. Map the string keys to their mathematical ratios
        ratios = {
            "1:1": (1, 1),
            "4:3": (4, 3),
            "3:2": (3, 2),
            "5:4": (5, 4),
            "16:9": (16, 9),
            "16:10": (16, 10),
            "21:9": (21, 9),
            "2:1": (2, 1),
            "1.85:1": (37, 20),
            "2.39:1": (239, 100),
        }

        # 2. Filter enabled ratios
        enabled_ratios = {}
        for key, value in ratios.items():
            # kwargs[key] will be True/False based on the toggle
            if kwargs.get(key, True):
                enabled_ratios[key] = value

        # Fallback to 1:1 if everything is disabled
        if not enabled_ratios:
            enabled_ratios = {"1:1": (1, 1)}

        # 3. Calculate Input Aspect Ratio
        is_portrait = height > width
        if is_portrait:
            input_float = height / width
        else:
            input_float = width / height

        # 4. Find the Closest Match
        best_match_name = None
        min_diff = float("inf")

        for name, (rw, rh) in enabled_ratios.items():
            target_float = max(rw, rh) / min(rw, rh)
            diff = abs(input_float - target_float)

            if diff < min_diff:
                min_diff = diff
                best_match_name = name

        # 5. Retrieve the winner
        if best_match_name is None:
            # fallback to 1:1 if something went wrong
            best_match_name = "1:1"
        target_w, target_h = enabled_ratios[best_match_name]

        # 6. Correct for Orientation
        if is_portrait:
            final_w = min(target_w, target_h)
            final_h = max(target_w, target_h)
        else:
            final_w = max(target_w, target_h)
            final_h = min(target_w, target_h)

        return (final_w, final_h)


NODE_CLASS_MAPPINGS = {"GetCommonAspectRatio": GetCommonAspectRatio}

NODE_DISPLAY_NAME_MAPPINGS = {"GetCommonAspectRatio": "🐧 Get Aspect Ratio"}
