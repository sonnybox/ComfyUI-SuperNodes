from comfy_api.latest import io
import torch

from ..image.utils import SCALE_METHODS, scale_image, scale_mask

try:
    from comfy import model_management
except Exception:
    model_management = None

# Peak VRAM the compositing pass may use for per-chunk tensors.
VRAM_BUDGET_BYTES = 4 * 1024**3

# Where the feather ramp sits relative to the bbox edge: alpha = clamp(d/f + offset)
# with d the signed distance to the edge (positive inside the bbox).
_FEATHER_OFFSET = {"inner": 0.0, "mid": 0.5, "outer": 1.0}


def _compute_device():
    if model_management is not None:
        try:
            return model_management.get_torch_device()
        except Exception:
            pass
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _bbox_feather_alpha(
    py1, py2, px1, px2, ix1, iy1, ix2, iy2, feather, mode, device, dtype
):
    """
    Blend alpha over the paste region [py1:py2, px1:px2] (background coords),
    built from the signed Chebyshev distance to the bbox edge (positive inside).
    feather is the ramp length in pixels; mode places the ramp inside the bbox,
    outside it, or centered on the edge.
    """
    ys = torch.arange(py1, py2, device=device, dtype=dtype)
    xs = torch.arange(px1, px2, device=device, dtype=dtype)
    dy = torch.minimum(ys - iy1, (iy2 - 1) - ys)
    dx = torch.minimum(xs - ix1, (ix2 - 1) - xs)
    d = torch.minimum(dy.view(-1, 1), dx.view(1, -1))
    if feather <= 0:
        return (d >= 0).to(dtype)
    return (d / feather + _FEATHER_OFFSET[mode]).clamp(0.0, 1.0)


class RestoreVideoCropFrames(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="RestoreVideoCropFrames",
            display_name="🐧 Restore Video Crop Frames",
            category="SuperNodes/Video",
            description=(
                "Pastes processed cropped frames back onto the background frames using the crop "
                "metadata from Get Video Crop Frames. Blending uses per-frame masks (mask values are "
                "the paste alpha) or, without masks, a feathered bbox edge. Processes in VRAM-bounded "
                "chunks on the GPU when available."
            ),
            inputs=[
                io.Image.Input(
                    "background_frames",
                    tooltip="The frames to composite onto; defines output size and coordinate space.",
                ),
                io.Image.Input(
                    "cropped_frames",
                    tooltip="The processed cropped frame batch to paste back.",
                ),
                io.Custom("BBOX_RESTORE_INFO").Input(
                    "restore_info",
                    tooltip="Crop metadata produced by Get Video Crop Frames.",
                ),
                io.Combo.Input(
                    "scale_method",
                    options=SCALE_METHODS,
                    tooltip="Interpolation used when resizing crops and crop-space masks.",
                ),
                io.Int.Input(
                    "feather_bbox",
                    default=0,
                    min=0,
                    max=1024,
                    step=1,
                    tooltip=(
                        "Feather radius in pixels (output frame space) applied to the bbox edge when "
                        "'masks' is not connected. 0 = hard edge. Ignored when masks are provided."
                    ),
                ),
                io.Combo.Input(
                    "feather_mode",
                    options=["mid", "inner", "outer"],
                    tooltip=(
                        "Where the bbox feather falls: 'inner' fades inside the bbox, 'outer' fades "
                        "outward past it (needs crop padding to have content there), 'mid' straddles "
                        "the edge. Ignored when masks are provided."
                    ),
                ),
                io.Mask.Input(
                    "masks",
                    optional=True,
                    tooltip=(
                        "Per-frame blend masks; mask values are the paste alpha, so feather the mask "
                        "itself to control edge softness. Coordinate space is auto-detected by size: "
                        "if HxW matches cropped_frames or the original crop size in restore_info "
                        "(e.g. the cropped_masks output of Get Video Crop Frames, even if the crops "
                        "were upscaled since), they are crop-space masks resized with the crop; if "
                        "HxW matches background_frames they are used in place. Anything else is an "
                        "error. Batch must be at least the restored frame count (extras are "
                        "trimmed; too few is an error). Overrides feather_bbox/feather_mode."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="frames",
                    tooltip="The background frames with each crop restored.",
                ),
            ],
        )

    @classmethod
    @torch.inference_mode()
    def execute(
        cls,
        background_frames,
        cropped_frames,
        restore_info,
        scale_method,
        feather_bbox,
        feather_mode,
        masks=None,
    ) -> io.NodeOutput:
        B, bgH, bgW, C = background_frames.shape
        crH, crW = cropped_frames.shape[1], cropped_frames.shape[2]
        info_frames = restore_info.get("frames", [])
        # Output the entire background sequence; only frames that have a matching
        # crop get composited over. Extra background frames pass through untouched.
        n = min(B, len(info_frames), cropped_frames.shape[0])

        # Validate masks: strict per-frame pairing, coordinate space by size.
        mask_space = None
        if masks is not None:
            if masks.dim() == 2:
                masks = masks.unsqueeze(0)
            if masks.dim() == 4 and masks.shape[3] == 1:
                masks = masks.squeeze(3)
            mB, mH, mW = masks.shape[:3]
            tgt_h = restore_info.get("target_h")
            tgt_w = restore_info.get("target_w")
            if (mH == crH and mW == crW) or (mH == tgt_h and mW == tgt_w):
                # Crop space: either the processed crop size or the original
                # (pre-processing) crop size recorded in restore_info — e.g. the
                # cropped_masks output of Get Video Crop Frames fed back in even
                # though the crops were upscaled in between.
                mask_space = "crop"
            elif mH == bgH and mW == bgW:
                mask_space = "background"
            else:
                raise ValueError(
                    f"masks size ({mW}x{mH}) must match the cropped frames "
                    f"({crW}x{crH}), the original crop size ({tgt_w}x{tgt_h}), "
                    f"or the background frames ({bgW}x{bgH})."
                )
            if mB < n:
                raise ValueError(
                    f"Not enough masks: got {mB}, but {n} frames are being restored. "
                    "Provide at least one mask per background/cropped frame (extras are trimmed)."
                )
            masks = masks[:n]

        out = background_frames.clone()
        top_os = restore_info.get("original_size")

        # Precompute per-frame paste geometry in background space.
        jobs = []
        for i in range(n):
            entry = info_frames[i]
            if not entry:
                continue
            crop_box = entry.get("crop_box")
            bbox = entry.get("bbox")
            if crop_box is None or bbox is None:
                continue  # No detection for this frame — leave background untouched.

            orig_h, orig_w = entry.get("original_size") or top_os or (bgH, bgW)
            sx = bgW / orig_w
            sy = bgH / orig_h

            wx1, wy1, wx2, wy2 = crop_box
            cw = int(round((wx2 - wx1) * sx))
            ch = int(round((wy2 - wy1) * sy))
            if cw <= 0 or ch <= 0:
                continue

            x1, y1, x2, y2 = bbox
            jobs.append(
                {
                    "i": i,
                    "cx1": int(round(wx1 * sx)),
                    "cy1": int(round(wy1 * sy)),
                    "cw": cw,
                    "ch": ch,
                    "ix1": int(round(x1 * sx)),
                    "iy1": int(round(y1 * sy)),
                    "ix2": int(round(x2 * sx)),
                    "iy2": int(round(y2 * sy)),
                }
            )

        if not jobs:
            return io.NodeOutput(out)

        device = _compute_device()
        dtype = out.dtype

        # Chunk size from the VRAM budget: per frame we hold the input crop, the
        # resized crop, the blend mask, and composite temporaries on the device.
        max_cw = max(j["cw"] for j in jobs)
        max_ch = max(j["ch"] for j in jobs)
        per_frame_bytes = 4 * (crH * crW * C + max_ch * max_cw * (3 * C + 3))
        budget = VRAM_BUDGET_BYTES
        if model_management is not None and device.type == "cuda":
            try:
                budget = min(budget, int(model_management.get_free_memory(device) * 0.8))
            except Exception:
                pass
        chunk_size = max(1, min(len(jobs), budget // max(1, per_frame_bytes)))

        for start in range(0, len(jobs), chunk_size):
            batch = jobs[start : start + chunk_size]
            idxs = [j["i"] for j in batch]

            crops = cropped_frames[idxs].to(device=device, dtype=dtype)
            sizes = {(j["ch"], j["cw"]) for j in batch}
            uniform = len(sizes) == 1
            if uniform:
                ch0, cw0 = next(iter(sizes))
                if crops.shape[1] != ch0 or crops.shape[2] != cw0:
                    crops = scale_image(crops, cw0, ch0, scale_method)

            crop_masks = None
            if mask_space == "crop":
                crop_masks = torch.stack([masks[i] for i in idxs]).to(
                    device=device, dtype=dtype
                )
                if uniform:
                    if crop_masks.shape[1] != ch0 or crop_masks.shape[2] != cw0:
                        crop_masks = scale_mask(
                            crop_masks, cw0, ch0, scale_method
                        )
                    crop_masks = crop_masks.clamp(0.0, 1.0)

            for k, job in enumerate(batch):
                i = job["i"]
                cx1, cy1, cw, ch = job["cx1"], job["cy1"], job["cw"], job["ch"]

                # Clip the crop window to the background bounds.
                py1 = max(0, cy1)
                px1 = max(0, cx1)
                py2 = min(bgH, cy1 + ch)
                px2 = min(bgW, cx1 + cw)
                if py2 <= py1 or px2 <= px1:
                    continue

                crop = crops[k]
                if not uniform and (crop.shape[0] != ch or crop.shape[1] != cw):
                    crop = scale_image(
                        crop.unsqueeze(0), cw, ch, scale_method
                    )[0]

                sy1 = py1 - cy1
                sx1 = px1 - cx1
                sy2 = sy1 + (py2 - py1)
                sx2 = sx1 + (px2 - px1)
                src = crop[sy1:sy2, sx1:sx2, :]

                if mask_space == "crop":
                    m = crop_masks[k]
                    if not uniform and (m.shape[0] != ch or m.shape[1] != cw):
                        m = scale_mask(
                            m.unsqueeze(0), cw, ch, scale_method
                        )[0].clamp(0.0, 1.0)
                    alpha = m[sy1:sy2, sx1:sx2]
                elif mask_space == "background":
                    alpha = (
                        masks[i, py1:py2, px1:px2]
                        .to(device=device, dtype=dtype)
                        .clamp(0.0, 1.0)
                    )
                else:
                    alpha = _bbox_feather_alpha(
                        py1, py2, px1, px2,
                        job["ix1"], job["iy1"], job["ix2"], job["iy2"],
                        feather_bbox, feather_mode, device, dtype,
                    )

                dest = out[i, py1:py2, px1:px2, :].to(device=device, dtype=dtype)
                alpha = alpha.unsqueeze(-1)
                out[i, py1:py2, px1:px2, :] = (
                    src * alpha + dest * (1.0 - alpha)
                ).to(device=out.device, dtype=out.dtype)

        return io.NodeOutput(out)


NODE = [RestoreVideoCropFrames]
