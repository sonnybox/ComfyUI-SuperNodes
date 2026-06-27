from comfy_api.latest import io
import torch
import torch.nn.functional as F


def _parse_pad_color(pad_color, channels, device, dtype):
    """Parse an 'R,G,B' string (0-255) into a [C] tensor in 0-1 range."""
    try:
        parts = [float(p.strip()) for p in str(pad_color).split(",") if p.strip() != ""]
    except ValueError:
        parts = []
    if len(parts) == 0:
        parts = [0.0]
    # Broadcast a single value (grayscale) to all channels.
    if len(parts) == 1:
        parts = parts * channels
    # Clamp / pad to the channel count.
    parts = (parts + [0.0] * channels)[:channels]
    return torch.tensor([p / 255.0 for p in parts], device=device, dtype=dtype)


def _round_up(value, multiple):
    if multiple <= 1:
        return int(value)
    return int(((value + multiple - 1) // multiple) * multiple)


def _extract_window(frame, x1, y1, x2, y2, pad_mode, pad_color_t):
    """
    Extract window [y1:y2, x1:x2] from a single frame [H, W, C].
    The window may extend outside the frame; out-of-bounds regions are filled
    using pad_mode ('color' -> pad_color_t, 'edge' -> replicate edge pixels).
    Returns [win_h, win_w, C].
    """
    H, W, C = frame.shape
    win_w = x2 - x1
    win_h = y2 - y1

    # Valid intersection inside the frame.
    vx1 = max(0, x1)
    vy1 = max(0, y1)
    vx2 = min(W, x2)
    vy2 = min(H, y2)

    if pad_mode == "edge" and (vx2 > vx1) and (vy2 > vy1):
        # Pad the frame by replicating edge pixels, then slice.
        pad_left = max(0, -x1)
        pad_right = max(0, x2 - W)
        pad_top = max(0, -y1)
        pad_bottom = max(0, y2 - H)
        if pad_left or pad_right or pad_top or pad_bottom:
            t = frame.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
            t = F.pad(t, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")
            padded = t.squeeze(0).permute(1, 2, 0)  # [H', W', C]
            return padded[
                y1 + pad_top : y1 + pad_top + win_h,
                x1 + pad_left : x1 + pad_left + win_w,
                :,
            ]
        return frame[y1:y2, x1:x2, :]

    # Color fill (default) — also used by edge mode when there is no valid region.
    canvas = pad_color_t.view(1, 1, C).expand(win_h, win_w, C).clone()
    if vx2 > vx1 and vy2 > vy1:
        dy1 = vy1 - y1
        dx1 = vx1 - x1
        canvas[dy1 : dy1 + (vy2 - vy1), dx1 : dx1 + (vx2 - vx1), :] = frame[
            vy1:vy2, vx1:vx2, :
        ]
    return canvas


class GetBBoxCropFrames(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="GetBBoxCropFrames",
            display_name="🐧 Get BBox Crop Frames",
            category="SuperNodes/Video",
            description=(
                "Crops every frame to a single common size derived from the largest bbox so the "
                "result is a valid uniform image batch. Records each frame's true bbox for precise "
                "restoration."
            ),
            inputs=[
                io.Custom("BBOX").Input(
                    "bboxes",
                    tooltip="Per-frame bounding boxes [x1,y1,x2,y2] (None allowed per frame).",
                ),
                io.Image.Input("frames", tooltip="The source frame batch [B,H,W,C]."),
                io.Boolean.Input(
                    "square",
                    default=True,
                    tooltip="If true, pad the common crop size to a square (max of width/height).",
                ),
                io.Int.Input(
                    "padding",
                    default=0,
                    min=0,
                    max=4096,
                    step=1,
                    tooltip="Grow each bbox by this many pixels on all sides (clamped to the frame). Useful to leave room to feather outside the detection.",
                ),
                io.Int.Input(
                    "multiple_of",
                    default=16,
                    min=1,
                    max=512,
                    step=1,
                    tooltip="Round the output crop width/height up to a multiple of this value.",
                ),
                io.Combo.Input(
                    "pad_mode",
                    options=["color", "edge"],
                    tooltip="How to fill regions outside the frame when the crop window overflows.",
                ),
                io.String.Input(
                    "pad_color",
                    default="0,0,0",
                    tooltip="Fill color as 'R,G,B' (0-255) used when pad_mode is 'color'.",
                ),
            ],
            outputs=[
                io.Custom("BBOX_RESTORE_INFO").Output(
                    display_name="restore_info",
                    tooltip="Per-frame crop metadata required by Restore BBox Crop Frames.",
                ),
                io.Image.Output(
                    display_name="cropped_frames",
                    tooltip="Uniform-size cropped frame batch.",
                ),
            ],
        )

    @classmethod
    def execute(cls, bboxes, frames, square, padding, multiple_of, pad_mode, pad_color) -> io.NodeOutput:
        B, H, W, C = frames.shape
        device = frames.device
        dtype = frames.dtype
        pad_color_t = _parse_pad_color(pad_color, C, device, dtype)

        # 1. Normalize bboxes: grow by padding, clamp to frame, drop degenerate ones.
        #    The padded box becomes the recorded "true" bbox so restoration can
        #    feather across the padded region (outside the original detection).
        norm = []
        for i in range(B):
            bbox = bboxes[i] if bboxes is not None and i < len(bboxes) else None
            if bbox is None:
                norm.append(None)
                continue
            x1, y1, x2, y2 = (int(round(float(v))) for v in bbox)
            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))
            x1 -= padding
            y1 -= padding
            x2 += padding
            y2 += padding
            x1 = max(0, min(W, x1))
            x2 = max(0, min(W, x2))
            y1 = max(0, min(H, y1))
            y2 = max(0, min(H, y2))
            if (x2 - x1) <= 1 or (y2 - y1) <= 1:
                norm.append(None)
            else:
                norm.append((x1, y1, x2, y2))

        # 2. Determine the common crop size from the largest bbox dimensions.
        valid = [b for b in norm if b is not None]
        if valid:
            target_w = max(b[2] - b[0] for b in valid)
            target_h = max(b[3] - b[1] for b in valid)
        else:
            # No usable bboxes anywhere — fall back to the full frame.
            target_w, target_h = W, H

        if square:
            target_w = target_h = max(target_w, target_h)
        target_w = _round_up(target_w, multiple_of)
        target_h = _round_up(target_h, multiple_of)

        # 3. Build a crop window per frame, shifting to stay in-frame where possible.
        crops = []
        info_frames = []
        for i in range(B):
            bbox = norm[i]
            if bbox is None:
                crops.append(
                    torch.zeros((target_h, target_w, C), device=device, dtype=dtype)
                )
                info_frames.append(
                    {"crop_box": None, "bbox": None, "original_size": (H, W)}
                )
                continue

            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            wx1 = int(round(cx - target_w / 2.0))
            wy1 = int(round(cy - target_h / 2.0))

            # Shift the window to stay inside the frame when it fits.
            if target_w <= W:
                wx1 = max(0, min(W - target_w, wx1))
            if target_h <= H:
                wy1 = max(0, min(H - target_h, wy1))
            wx2 = wx1 + target_w
            wy2 = wy1 + target_h

            crop = _extract_window(
                frames[i], wx1, wy1, wx2, wy2, pad_mode, pad_color_t
            )
            crops.append(crop)
            info_frames.append(
                {
                    "crop_box": (wx1, wy1, wx2, wy2),
                    "bbox": (x1, y1, x2, y2),
                    "original_size": (H, W),
                }
            )

        cropped_frames = torch.stack(crops, dim=0)

        restore_info = {
            "frames": info_frames,
            "target_w": target_w,
            "target_h": target_h,
            "square": bool(square),
        }

        return io.NodeOutput(restore_info, cropped_frames)


NODE = [GetBBoxCropFrames]
