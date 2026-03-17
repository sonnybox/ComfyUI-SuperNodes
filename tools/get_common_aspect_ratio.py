from comfy_api.latest import io


class GetCommonAspectRatio(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="GetCommonAspectRatio",
            display_name="🐧 Get Aspect Ratio",
            category="SuperNodes/Tools",
            inputs=[
                io.Int.Input("width", default=1024, min=1, max=65536),
                io.Int.Input("height", default=1024, min=1, max=65536),
                io.Boolean.Input("1:1", default=True),
                io.Boolean.Input("4:3", default=True),
                io.Boolean.Input("3:2", default=True),
                io.Boolean.Input("5:4", default=True),
                io.Boolean.Input("16:9", default=True),
                io.Boolean.Input("16:10", default=True),
                io.Boolean.Input("21:9", default=True),
                io.Boolean.Input("2:1", default=True),
                io.Boolean.Input("1.85:1", default=True),
                io.Boolean.Input("2.39:1", default=True),
            ],
            outputs=[
                io.Int.Output(display_name="aspect_w"),
                io.Int.Output(display_name="aspect_h"),
            ],
        )

    @classmethod
    def execute(cls, width, height, **kwargs) -> io.NodeOutput:
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

        return io.NodeOutput(final_w, final_h)


NODE = [GetCommonAspectRatio]
