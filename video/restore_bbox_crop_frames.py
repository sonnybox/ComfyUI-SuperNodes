from comfy_api.latest import io
import numpy as np
from PIL import Image
import torch


def _resize_tensor(tensor, width, height, method, is_mask=False):
    """Resize [B,H,W,C] images or [B,H,W] masks. Lanczos falls back to PIL."""
    if method == "lanczos":
        return _pil_resize_fallback(tensor, width, height, method, is_mask)

    mode = method if method != "nearest-exact" else "nearest"

    if is_mask:
        t = tensor.unsqueeze(1) if tensor.dim() == 3 else tensor
    else:
        t = tensor.permute(0, 3, 1, 2)

    if mode in ["bilinear", "bicubic"]:
        t = torch.nn.functional.interpolate(
            t, size=(height, width), mode=mode, align_corners=False
        )
    else:
        t = torch.nn.functional.interpolate(t, size=(height, width), mode=mode)

    if is_mask:
        return t.squeeze(1)
    return t.permute(0, 2, 3, 1)


def _pil_resize_fallback(tensor_data, width, height, method, is_mask=False):
    pil_method = Image.Resampling.LANCZOS
    if method == "nearest-exact":
        pil_method = Image.Resampling.NEAREST
    elif method == "area":
        pil_method = Image.Resampling.BOX
    elif method == "bicubic":
        pil_method = Image.Resampling.BICUBIC
    elif method == "bilinear":
        pil_method = Image.Resampling.BILINEAR

    results = []
    if is_mask:
        for m in tensor_data:
            i = Image.fromarray(
                (m.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8), mode="L"
            )
            i = i.resize((width, height), resample=pil_method)
            results.append(torch.from_numpy(np.array(i).astype(np.float32) / 255.0))
    else:
        for img in tensor_data:
            i = Image.fromarray(
                (img.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            )
            i = i.resize((width, height), resample=pil_method)
            results.append(torch.from_numpy(np.array(i).astype(np.float32) / 255.0))

    return torch.stack(results).to(tensor_data.device)


class RestoreBBoxCropFrames(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="RestoreBBoxCropFrames",
            display_name="🐧 Restore BBox Crop Frames",
            category="SuperNodes/Video",
            description=(
                "Pastes processed cropped frames back onto the background frames using each frame's "
                "recorded bbox. An optional square mask is resized to each true (inner) bbox and used "
                "to feather the blend."
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
                    tooltip="Crop metadata produced by Get BBox Crop Frames.",
                ),
                io.Combo.Input(
                    "scale_method",
                    options=[
                        "nearest-exact",
                        "bilinear",
                        "area",
                        "bicubic",
                        "lanczos",
                    ],
                    tooltip="Interpolation used when resizing crops and the feather mask.",
                ),
                io.Mask.Input(
                    "mask",
                    optional=True,
                    tooltip="Optional square feather mask, resized to each true bbox for blending. Provide exactly 1 mask (shared by all frames) or at least one per frame (paired from index 0; extras ignored).",
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
    def execute(
        cls, background_frames, cropped_frames, restore_info, scale_method, mask=None
    ) -> io.NodeOutput:
        B, bgH, bgW, C = background_frames.shape
        # Output the entire background sequence; only frames that have a matching
        # crop get composited over. Extra background frames pass through untouched.
        out = background_frames.clone()
        info_frames = restore_info.get("frames", [])
        n = min(B, len(info_frames), cropped_frames.shape[0])

        # The feather mask is either a single mask (shared by every frame) or one
        # mask per frame (paired from index 0; any excess masks are ignored). Only
        # too few masks is an error.
        if mask is not None and mask.shape[0] != 1 and mask.shape[0] < cropped_frames.shape[0]:
            raise ValueError(
                "mask batch size must be exactly 1 or at least the frame count "
                f"({cropped_frames.shape[0]}), got {mask.shape[0]}."
            )

        for i in range(n):
            entry = info_frames[i]
            if entry.get("bbox") is None or entry.get("crop_box") is None:
                continue  # No bbox for this frame — leave background untouched.

            crop = cropped_frames[i].unsqueeze(0)

            orig_h, orig_w = entry["original_size"]
            sx = bgW / orig_w
            sy = bgH / orig_h

            cx1, cy1, cx2, cy2 = entry["crop_box"]
            x1, y1, x2, y2 = entry["bbox"]

            # Scale all coordinates into the background space.
            cx1, cx2 = int(round(cx1 * sx)), int(round(cx2 * sx))
            cy1, cy2 = int(round(cy1 * sy)), int(round(cy2 * sy))
            ix1, ix2 = int(round(x1 * sx)), int(round(x2 * sx))
            iy1, iy2 = int(round(y1 * sy)), int(round(y2 * sy))

            cw = cx2 - cx1
            ch = cy2 - cy1
            if cw <= 0 or ch <= 0:
                continue

            # Resize the crop to fit the crop window in background space.
            if crop.shape[2] != cw or crop.shape[1] != ch:
                crop = _resize_tensor(crop, cw, ch, scale_method, is_mask=False)

            # Build a crop-sized blend mask, active only over the inner true bbox.
            blend = torch.zeros((1, ch, cw), device=out.device, dtype=out.dtype)
            iw = ix2 - ix1
            ih = iy2 - iy1
            ox1 = ix1 - cx1
            oy1 = iy1 - cy1
            if iw > 0 and ih > 0:
                if mask is not None:
                    m_idx = 0 if mask.shape[0] == 1 else i
                    m = mask[m_idx].unsqueeze(0).to(out.device)
                    m = _resize_tensor(m, iw, ih, scale_method, is_mask=True).clamp(
                        0.0, 1.0
                    )
                else:
                    m = torch.ones((1, ih, iw), device=out.device, dtype=out.dtype)
                # Clip the inner region to the crop bounds.
                ty1 = max(0, oy1)
                tx1 = max(0, ox1)
                ty2 = min(ch, oy1 + ih)
                tx2 = min(cw, ox1 + iw)
                if ty2 > ty1 and tx2 > tx1:
                    blend[:, ty1:ty2, tx1:tx2] = m[
                        :, ty1 - oy1 : ty2 - oy1, tx1 - ox1 : tx2 - ox1
                    ]

            # Paste into the background, clipping the crop window to bounds.
            py1 = max(0, cy1)
            px1 = max(0, cx1)
            py2 = min(bgH, cy2)
            px2 = min(bgW, cx2)
            if py2 <= py1 or px2 <= px1:
                continue

            sy1 = py1 - cy1
            sx1 = px1 - cx1
            sy2 = sy1 + (py2 - py1)
            sx2 = sx1 + (px2 - px1)

            dest = out[i, py1:py2, px1:px2, :]
            src = crop[0, sy1:sy2, sx1:sx2, :]
            m_region = blend[0, sy1:sy2, sx1:sx2].unsqueeze(-1)

            out[i, py1:py2, px1:px2, :] = src * m_region + dest * (1.0 - m_region)

        return io.NodeOutput(out)


NODE = [RestoreBBoxCropFrames]
