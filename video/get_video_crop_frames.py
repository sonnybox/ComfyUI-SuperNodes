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


def _extract_window_into(dest, frame, x1, y1, x2, y2, pad_mode, pad_color_t):
    """
    Fill dest [win_h, win_w, C] with window [y1:y2, x1:x2] of frame [H, W, C],
    writing in place to avoid a per-frame canvas allocation. The window may
    extend outside the frame; out-of-bounds regions are filled using pad_mode
    ('color' -> pad_color_t, 'edge' -> replicate edge pixels).
    """
    H, W, C = frame.shape

    # Valid intersection inside the frame.
    vx1 = max(0, x1)
    vy1 = max(0, y1)
    vx2 = min(W, x2)
    vy2 = min(H, y2)

    # If the intersection is degenerate (entirely out of bounds), fallback to color padding
    if vx2 <= vx1 or vy2 <= vy1:
        dest[:] = pad_color_t.view(1, 1, C)
        return

    if pad_mode == "edge":
        # Extract the valid sub-region first (much smaller memory footprint than padding the whole frame).
        sub_frame = frame[vy1:vy2, vx1:vx2, :]  # Shape [sub_h, sub_w, C]

        # Calculate padding needed relative to the sub-region.
        pad_left = vx1 - x1
        pad_right = x2 - vx2
        pad_top = vy1 - y1
        pad_bottom = y2 - vy2

        if pad_left or pad_right or pad_top or pad_bottom:
            t = sub_frame.permute(2, 0, 1).unsqueeze(0)  # [1, C, sub_h, sub_w]
            t = F.pad(t, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")
            dest[:] = t.squeeze(0).permute(1, 2, 0)  # [win_h, win_w, C]
        else:
            dest[:] = sub_frame
        return

    # Color fill (default)
    dest[:] = pad_color_t.view(1, 1, C)
    dy1 = vy1 - y1
    dx1 = vx1 - x1
    dest[dy1 : dy1 + (vy2 - vy1), dx1 : dx1 + (vx2 - vx1), :] = frame[
        vy1:vy2, vx1:vx2, :
    ]


class GetVideoCropFrames(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="GetVideoCropFrames",
            display_name="🐧 Get Video Crop Frames",
            category="SuperNodes/Video",
            description=(
                "Crops every frame to a single common size derived from the largest bbox or mask area so the "
                "result is a valid uniform image batch. Records each frame's true bbox for precise "
                "restoration."
            ),
            inputs=[
                io.Image.Input("frames", tooltip="The source frame batch [B,H,W,C]."),
                io.Custom("BBOX").Input(
                    "bboxes",
                    optional=True,
                    tooltip="Per-frame bounding boxes [x1,y1,x2,y2]. If a bbox is invalid or empty, the corresponding frame will be black.",
                ),
                io.Mask.Input(
                    "masks",
                    optional=True,
                    tooltip="Per-frame masks [B,H,W] defining regions of interest. If provided, masks are preferred over bboxes.",
                ),
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
                    tooltip="Grow each bbox/mask by this many pixels on all sides (clamped to the frame). Useful to leave room to feather outside the detection.",
                ),
                io.Int.Input(
                    "multiple_of",
                    default=16,
                    min=1,
                    max=512,
                    step=1,
                    tooltip="Round the output crop width/height up to a multiple of this value.",
                ),
                io.Float.Input(
                    "horizontal_offset",
                    default=0.0,
                    min=-5.0,
                    max=5.0,
                    step=0.1,
                    tooltip="Anchor the bbox/mask horizontally within the crop margin. 0 = centered, -5 = flush left, +5 = flush right.",
                ),
                io.Float.Input(
                    "vertical_offset",
                    default=0.0,
                    min=-5.0,
                    max=5.0,
                    step=0.1,
                    tooltip="Anchor the bbox/mask vertically within the crop margin. 0 = centered, +5 = flush top, -5 = flush bottom.",
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
                    tooltip="Per-frame crop metadata required by Restore Video Crop Frames.",
                ),
                io.Image.Output(
                    display_name="cropped_frames",
                    tooltip="Uniform-size cropped frame batch.",
                ),
                io.Mask.Output(
                    display_name="cropped_masks",
                    tooltip="Uniform-size cropped mask batch.",
                ),
            ],
        )

    @classmethod
    @torch.inference_mode()
    def execute(
        cls,
        frames,
        square,
        padding,
        multiple_of,
        horizontal_offset,
        vertical_offset,
        pad_mode,
        pad_color,
        bboxes=None,
        masks=None,
    ) -> io.NodeOutput:
        if bboxes is None and masks is None:
            raise ValueError("Either bboxes or masks must be provided.")

        B, H, W, C = frames.shape
        device = frames.device
        dtype = frames.dtype
        pad_color_t = _parse_pad_color(pad_color, C, device, dtype)

        # Validate bboxes length if provided
        if bboxes is not None and len(bboxes) < B:
            raise ValueError(
                f"Not enough bboxes for the video duration. Video has {B} frames, but only {len(bboxes)} bboxes were provided."
            )

        # Validate masks dimensions and length if provided
        if masks is not None:
            if masks.dim() == 4 and masks.shape[3] == 1:
                masks = masks.squeeze(3)
            if masks.dim() < 3:
                raise ValueError(
                    f"masks must be a 3D tensor of shape [B, H, W], got shape {list(masks.shape)}"
                )
            mask_B, mask_H, mask_W = masks.shape[:3]
            if mask_B < B:
                raise ValueError(
                    f"Not enough masks for the video duration. Video has {B} frames, but only {mask_B} masks were provided."
                )
            if mask_H != H or mask_W != W:
                raise ValueError(
                    f"Mask dimensions ({mask_W}x{mask_H}) must match frame dimensions ({W}x{H})."
                )

        # 1. Extract bboxes: extract from masks if available, otherwise from bboxes.
        #    Optimize mask bounding box extraction using vectorized GPU calculations.
        norm = []
        if masks is not None:
            mask_binary = (masks > 0.5)
            any_x = mask_binary.any(dim=1)  # [B, W]
            any_y = mask_binary.any(dim=2)  # [B, H]
            
            cols = torch.arange(W, device=device)
            rows = torch.arange(H, device=device)
            
            cols_masked_min = cols.view(1, W) * any_x + (~any_x) * W
            cols_masked_max = cols.view(1, W) * any_x + (~any_x) * -1
            
            rows_masked_min = rows.view(1, H) * any_y + (~any_y) * H
            rows_masked_max = rows.view(1, H) * any_y + (~any_y) * -1
            
            x1_t = cols_masked_min.min(dim=1).values
            x2_t = cols_masked_max.max(dim=1).values + 1
            y1_t = rows_masked_min.min(dim=1).values
            y2_t = rows_masked_max.max(dim=1).values + 1
            
            # Single GPU-to-CPU transfer to prevent per-frame loop sync overhead
            x1_arr = x1_t.cpu().tolist()
            x2_arr = x2_t.cpu().tolist()
            y1_arr = y1_t.cpu().tolist()
            y2_arr = y2_t.cpu().tolist()
            
            for i in range(B):
                if x1_arr[i] == W:
                    bbox = None
                else:
                    bbox = (x1_arr[i], y1_arr[i], x2_arr[i], y2_arr[i])
                norm.append(bbox)
        else:
            for i in range(B):
                bbox = bboxes[i] if i < len(bboxes) else None
                norm.append(bbox)

        # Normalize boundaries (padding & clamp)
        for i in range(B):
            bbox = norm[i]
            if bbox is None:
                continue

            x1, y1, x2, y2 = (round(float(v)) for v in bbox)
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
                norm[i] = None
            else:
                norm[i] = (x1, y1, x2, y2)

        # 2. Determine the common crop size from the largest bbox dimensions.
        valid = [b for b in norm if b is not None]
        if valid:
            target_w = max(b[2] - b[0] for b in valid)
            target_h = max(b[3] - b[1] for b in valid)
        else:
            # No usable bboxes/masks anywhere — fall back to the full frame.
            target_w, target_h = W, H

        if square:
            target_w = target_h = max(target_w, target_h)
        target_w = _round_up(target_w, multiple_of)
        target_h = _round_up(target_h, multiple_of)

        # Anchor fractions: where the bbox sits within the crop margin.
        th = min(1.0, max(0.0, (horizontal_offset + 5.0) / 10.0))
        tv = min(1.0, max(0.0, (5.0 - vertical_offset) / 10.0))

        # 3. Pre-allocate output tensors to reduce memory peaks (saves ~50% of the memory footprint)
        cropped_frames = torch.zeros((B, target_h, target_w, C), device=device, dtype=dtype)
        cropped_masks = torch.zeros((B, target_h, target_w), device=device, dtype=dtype)
        
        info_frames: list[dict[str, tuple[int, ...] | None]] = []
        mask_pad_color = torch.tensor([0.0], device=device, dtype=dtype)

        for i in range(B):
            bbox = norm[i]
            if bbox is None:
                info_frames.append({"crop_box": None, "bbox": None})
                continue

            x1, y1, x2, y2 = bbox

            # Position the window by distributing the margin per the anchor.
            margin_x = target_w - (x2 - x1)
            margin_y = target_h - (y2 - y1)
            wx1 = round(x1 - th * margin_x)
            wy1 = round(y1 - tv * margin_y)

            # Shift the window to stay inside the frame when it fits.
            if target_w <= W:
                wx1 = max(0, min(W - target_w, wx1))
            if target_h <= H:
                wy1 = max(0, min(H - target_h, wy1))
            wx2 = wx1 + target_w
            wy2 = wy1 + target_h

            _extract_window_into(
                cropped_frames[i], frames[i], wx1, wy1, wx2, wy2, pad_mode, pad_color_t
            )

            # Process mask cropping (in-place into the preallocated batch)
            if masks is not None:
                _extract_window_into(
                    cropped_masks[i].unsqueeze(-1),
                    masks[i].unsqueeze(-1),
                    wx1, wy1, wx2, wy2, pad_mode, mask_pad_color,
                )
            else:
                # Reconstruct mask from the bbox (the actual bbox area themselves)
                # Fill box region inside the crop window with 1.0, and 0.0 outside.
                bx1 = max(0, x1 - wx1)
                by1 = max(0, y1 - wy1)
                bx2 = min(target_w, x2 - wx1)
                by2 = min(target_h, y2 - wy1)
                if bx2 > bx1 and by2 > by1:
                    cropped_masks[i, by1:by2, bx1:bx2] = 1.0

            info_frames.append(
                {
                    "crop_box": (wx1, wy1, wx2, wy2),
                    "bbox": (x1, y1, x2, y2),
                }
            )

        restore_info = {
            "version": 2,
            "original_size": (H, W),
            "frames": info_frames,
            "target_w": target_w,
            "target_h": target_h,
            "square": bool(square),
        }

        return io.NodeOutput(restore_info, cropped_frames, cropped_masks)


NODE = [GetVideoCropFrames]
