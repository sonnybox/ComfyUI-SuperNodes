import torch


class FaceBBoxToMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "face_bboxes": ("BBOX",),
                "images": ("IMAGE",),
                "extend_up_percent": (
                    "INT",
                    {"default": 0, "min": -100, "max": 500, "step": 1},
                ),
                "extend_down_percent": (
                    "INT",
                    {"default": 0, "min": -100, "max": 500, "step": 1},
                ),
                "extend_left_percent": (
                    "INT",
                    {"default": 0, "min": -100, "max": 500, "step": 1},
                ),
                "extend_right_percent": (
                    "INT",
                    {"default": 0, "min": -100, "max": 500, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASK",)
    FUNCTION = "process"
    CATEGORY = "SuperNodes/Extras"
    DESCRIPTION = "Converts face bounding boxes into a mask batch, with percent-based directional extension."

    def process(
        self,
        face_bboxes,
        images,
        extend_up_percent,
        extend_down_percent,
        extend_left_percent,
        extend_right_percent,
    ):
        batch_size, height, width, _ = images.shape
        masks = torch.zeros((batch_size, height, width), dtype=torch.float32)

        for i in range(min(batch_size, len(face_bboxes))):
            bbox = face_bboxes[i]
            if bbox is None:
                continue

            x1, y1, x2, y2 = bbox
            bw = float(x2 - x1)
            bh = float(y2 - y1)

            if bw <= 1 or bh <= 1:
                continue

            up_px = int(round(bh * (extend_up_percent / 100.0)))
            down_px = int(round(bh * (extend_down_percent / 100.0)))
            left_px = int(round(bw * (extend_left_percent / 100.0)))
            right_px = int(round(bw * (extend_right_percent / 100.0)))

            x1 = int(round(x1)) - left_px
            x2 = int(round(x2)) + right_px
            y1 = int(round(y1)) - up_px
            y2 = int(round(y2)) + down_px

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)

            if x2 > x1 and y2 > y1:
                masks[i, y1:y2, x1:x2] = 1.0

        return (masks,)


NODE_CLASS_MAPPINGS = {"FaceBBoxToMask": FaceBBoxToMask}

NODE_DISPLAY_NAME_MAPPINGS = {"FaceBBoxToMask": "🐧 Convert BBox to Masks"}
