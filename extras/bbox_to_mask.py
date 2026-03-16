from comfy_api.latest import io
import torch


class FaceBBoxToMask(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="FaceBBoxToMask",
            display_name="🐧 Convert BBox to Masks",
            category="SuperNodes/Extras",
            inputs=[
                io.Custom("BBOX").Input("face_bboxes"),
                io.Image.Input("images"),
                io.Int.Input(
                    "extend_up_percent", default=0, min=-100, max=500, step=1
                ),
                io.Int.Input(
                    "extend_down_percent", default=0, min=-100, max=500, step=1
                ),
                io.Int.Input(
                    "extend_left_percent", default=0, min=-100, max=500, step=1
                ),
                io.Int.Input(
                    "extend_right_percent", default=0, min=-100, max=500, step=1
                ),
            ],
            outputs=[
                io.Mask.Output(display_name="MASK"),
            ],
            description="Converts face bounding boxes into a mask batch, with percent-based directional extension.",
        )

    @classmethod
    def execute(
        cls,
        face_bboxes,
        images,
        extend_up_percent,
        extend_down_percent,
        extend_left_percent,
        extend_right_percent,
    ) -> io.NodeOutput:
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

        return io.NodeOutput(masks)


V3_NODES = [FaceBBoxToMask]
