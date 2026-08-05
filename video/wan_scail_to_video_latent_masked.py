"""WanSCAILToVideoLatentMasked: variant of ComfyUI's WanSCAILToVideo that accepts
optional original_frames + original_frame_masks. Masked areas of the original video
are encoded into the latent and locked via the noise mask, so the sampler is forced
to preserve them instead of being trusted to recreate them.
"""

import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.utils
from comfy_api.latest import io
from comfy_extras.nodes_scail import _extract_mask_to_28ch
import node_helpers
import nodes

# Wan: VAE is 8x, the transformer patches 2x2 -> one token is 16x16 px.
VAE_STRIDE = 8
TOKEN_CELLS = 2
TOKEN_PX = VAE_STRIDE * TOKEN_CELLS


def _wan_frame_spans(t_latent):
    """Pixel frames covered by each Wan latent frame: 1, then 4 each."""
    return [1] + [4] * (t_latent - 1)


def _quantize_keep_mask(m):
    """Per-pixel-frame masks at output resolution -> token blocks + keep + preview.

    m: [N, height, width], already sized to the output. Returns blocks
    [T_lat, 1, h/2, w/2], keep [T_lat, 1, h, w] on the latent cell grid, and a
    [N, height, width] preview of what actually got locked.

    Inner on both axes: min-pool over each 16 px token, then over each latent
    frame's pixel frames, so a cell survives only where every pixel and every
    frame under it is marked. Both are min operations, which compose, so the
    result is the same as one min over the whole 3D cell.
    """
    t_lat = ((m.shape[0] - 1) // 4) + 1
    spans = _wan_frame_spans(t_lat)
    if sum(spans) > m.shape[0]:  # trailing partial group: pad with the last frame
        m = torch.cat([m, m[-1:].expand(sum(spans) - m.shape[0], -1, -1)], 0)

    b = (m > 0.5).float().unsqueeze(1)
    spatial = -F.max_pool2d(-b, TOKEN_PX)  # min-pool: every pixel marked
    groups, at = [], 0
    for span in spans:
        groups.append(spatial[at:at + span].amin(dim=0, keepdim=True))
        at += span
    blocks = torch.cat(groups, dim=0)

    keep = blocks.repeat_interleave(TOKEN_CELLS, dim=-2).repeat_interleave(
        TOKEN_CELLS, dim=-1
    )

    idx = torch.repeat_interleave(
        torch.arange(len(spans), device=blocks.device),
        torch.tensor(spans, device=blocks.device),
    )
    preview = F.interpolate(
        blocks[idx], scale_factor=TOKEN_PX, mode="nearest"
    ).squeeze(1)
    return blocks, keep, preview


class WanSCAILToVideoLatentMasked(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WanSCAILToVideoLatentMasked",
            display_name="🐧 WanSCAILToVideo (Latent Masked)",
            category="SuperNodes/Video",
            description="WanSCAILToVideo with optional original_frames/original_frame_masks inputs.",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Int.Input("width", default=512, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("height", default=896, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("length", default=81, min=1, max=nodes.MAX_RESOLUTION, step=4),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                io.Image.Input("pose_video", optional=True, tooltip="Video used for pose conditioning. Will be downscaled to half the resolution of the main video."),
                io.Image.Input("pose_video_mask", optional=True, tooltip="SCAIL-2 only. Colored per-identity SAM3 mask video at the same resolution as pose_video."),
                io.Boolean.Input("replacement_mode", default=False, optional=True, tooltip="SCAIL-2 only. False = Animation Mode (pose_video_mask should have black background). True = Replacement Mode (pose_video_mask should have white background)."),
                io.Float.Input("pose_strength", default=1.0, min=0.0, max=10.0, step=0.01, tooltip="Strength of the pose latent."),
                io.Float.Input("pose_start", default=0.0, min=0.0, max=1.0, step=0.01, tooltip="Start step of the pose conditioning."),
                io.Float.Input("pose_end", default=1.0, min=0.0, max=1.0, step=0.01, tooltip="End step of the pose conditioning."),
                io.Image.Input("reference_image", optional=True, tooltip="Reference image. The first image is the primary reference (composite all identities onto it). SCAIL-2: extra batch images are used as additional views (back view, close-up, occluded background), each needing a matching reference_image_mask in that identity's color."),
                io.Image.Input("reference_image_mask", optional=True, tooltip="SCAIL-2 only. Colored reference mask, batch matching reference_image (first = primary reference mask, rest = identity masks for the additional reference_image)."),
                io.ClipVisionOutput.Input("clip_vision_output", optional=True, tooltip="CLIP vision features for conditioning. Model is trained with stretch resize to aspect ratio."),
                io.Int.Input("video_frame_offset", default=0, min=0, max=nodes.MAX_RESOLUTION, step=1, tooltip="Cumulative output frame this chunk begins at. Wire from the previous chunk's video_frame_offset output."),
                io.Int.Input("previous_frame_count", default=5, min=1, max=nodes.MAX_RESOLUTION, step=4, tooltip="Tail frames of previous_frames to anchor. SCAIL-2 trained at 5 (81-frame chunks, 76-frame step)."),
                io.Image.Input("previous_frames", optional=True, tooltip="SCAIL-2 only. Full decoded output of the previous chunk. Only the last previous_frame_count are used as the extension anchor."),
                io.Boolean.Input("enable_latent_mask", default=True, optional=True, tooltip="Toggle the original-frame preservation feature. Off = original_frames/original_frame_masks are ignored entirely and the node behaves exactly like stock WanSCAILToVideo."),
                io.Image.Input("original_frames", optional=True, tooltip="Original video frames at the output resolution. Areas covered by original_frame_masks are encoded into the latent and locked so the sampler preserves them exactly. Offset by video_frame_offset like pose_video. Ignored if original_frame_masks is not connected."),
                io.Mask.Input("original_frame_masks", optional=True, tooltip="Per-frame masks matching original_frames (a single mask is broadcast to all frames). White (1.0) = hard-preserve the original video content, black (0.0) = generate normally. Quantized onto the 16 px token grid. Ignored if original_frames is not connected."),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent", tooltip="Latent of the generation size. Contains the encoded original frames (with matching noise mask) where original_frame_masks preserves them; empty elsewhere."),
                io.Int.Output(display_name="video_frame_offset", tooltip="Adjusted offset + length. Wire into the next chunk."),
                io.Mask.Output(display_name="quantized_mask", tooltip="What actually got locked, painted back at output resolution: white = preserved from the original, black = generated. One frame per input mask frame, so frames sharing a latent frame are identical and it plays back at the latent's real temporal resolution. All black if no preservation is active."),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, positive, negative, vae, width, height, length, batch_size, pose_strength, pose_start, pose_end,
                video_frame_offset, previous_frame_count, replacement_mode=False, reference_image=None, clip_vision_output=None, pose_video=None,
                pose_video_mask=None, reference_image_mask=None, previous_frames=None, enable_latent_mask=True, original_frames=None, original_frame_masks=None) -> io.NodeOutput:
        latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        noise_mask = None
        quantized_mask = None

        ref_mask_flag = not replacement_mode
        positive = node_helpers.conditioning_set_values(positive, {"ref_mask_flag": ref_mask_flag})
        negative = node_helpers.conditioning_set_values(negative, {"ref_mask_flag": ref_mask_flag})

        prev_trimmed = None
        if previous_frames is not None and previous_frames.shape[0] > 0:
            prev_trimmed = previous_frames[-previous_frame_count:]
            video_frame_offset -= prev_trimmed.shape[0]
            video_frame_offset = max(0, video_frame_offset)

        if reference_image is not None:
            ref_imgs = comfy.utils.common_upscale(reference_image.movedim(-1, 1), width, height, "bicubic", "center").movedim(1, -1)
            n_ref = ref_imgs.shape[0]
            # SCAIL-2 multi-reference: the first image is the primary ref, the rest are additional references.

            # Replacement Mode: composite each ref on black bg using its mask as alpha matte
            if replacement_mode and reference_image_mask is not None:
                rm = comfy.utils.common_upscale(reference_image_mask.movedim(-1, 1), width, height, "nearest-exact", "center").movedim(1, -1)
                rm = rm[[min(i, rm.shape[0] - 1) for i in range(n_ref)]]
                is_char = (rm[..., :3].max(dim=-1, keepdim=True).values > 0.1).to(ref_imgs.dtype)
                ref_imgs = ref_imgs * is_char
            # encode each ref individually so each stays a single latent frame (a batched encode would be treated as a video)
            ref_latents = [vae.encode(ref_imgs[i:i + 1, :, :, :3]) for i in range(n_ref)]
            positive = node_helpers.conditioning_set_values(positive, {"reference_latents": ref_latents}, append=True)
            negative = node_helpers.conditioning_set_values(negative, {"reference_latents": ref_latents}, append=True)

        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        if pose_video is not None:
            if pose_video.shape[0] <= video_frame_offset:
                pose_video = None
            else:
                pose_video = pose_video[video_frame_offset:]
        if pose_video_mask is not None:
            if pose_video_mask.shape[0] <= video_frame_offset:
                pose_video_mask = None
            else:
                pose_video_mask = pose_video_mask[video_frame_offset:]

        # Truncate pose+mask jointly to the shorter of the two, capped at length.
        ts = [v.shape[0] for v in (pose_video, pose_video_mask) if v is not None]
        if ts:
            T_kept = ((min(min(ts), length) - 1) // 4) * 4 + 1
            if pose_video is not None:
                pose_video = pose_video[:T_kept]
            if pose_video_mask is not None:
                pose_video_mask = pose_video_mask[:T_kept]

        if pose_video is not None:
            pose_video = comfy.utils.common_upscale(pose_video[:length].movedim(-1, 1), width // 2, height // 2, "area", "center").movedim(1, -1)
            pose_video_latent = vae.encode(pose_video[:, :, :, :3]) * pose_strength
            positive = node_helpers.conditioning_set_values_with_timestep_range(positive, {"pose_video_latent": pose_video_latent}, pose_start, pose_end)
            negative = node_helpers.conditioning_set_values_with_timestep_range(negative, {"pose_video_latent": pose_video_latent}, pose_start, pose_end)

        if pose_video_mask is not None:
            mask_video_hw = comfy.utils.common_upscale(pose_video_mask[:length].movedim(-1, 1), width // 2, height // 2, "area", "center").movedim(1, -1)
            driving_mask_28ch = _extract_mask_to_28ch(mask_video_hw)
            positive = node_helpers.conditioning_set_values(positive, {"driving_mask_28ch": driving_mask_28ch})
            negative = node_helpers.conditioning_set_values(negative, {"driving_mask_28ch": driving_mask_28ch})

        # The ref mask binds reference frames to identities, so it only applies when there's a reference image.
        if reference_image_mask is not None and reference_image is not None:
            ref_mask_hw = comfy.utils.common_upscale(reference_image_mask.movedim(-1, 1), width, height, "nearest-exact", "center").movedim(1, -1)
            n_masks = ref_mask_hw.shape[0]
            n_ref = reference_image.shape[0]

            add_masks = [_extract_mask_to_28ch(ref_mask_hw[min(i, n_masks - 1)][None]) for i in range(1, n_ref)]
            ref_mask_1f = _extract_mask_to_28ch(ref_mask_hw[:1])
            zeros = torch.zeros((1, latent.shape[2], 28, ref_mask_1f.shape[-2], ref_mask_1f.shape[-1]), device=ref_mask_1f.device, dtype=ref_mask_1f.dtype)
            ref_mask_28ch = torch.cat(add_masks + [ref_mask_1f, zeros], dim=1)
            positive = node_helpers.conditioning_set_values(positive, {"ref_mask_28ch": ref_mask_28ch})
            negative = node_helpers.conditioning_set_values(negative, {"ref_mask_28ch": ref_mask_28ch})

        # Hard-preserve original video areas: encode original_frames into the latent and
        # zero the noise mask where original_frame_masks marks them. Requires both inputs.
        if enable_latent_mask and original_frames is not None and original_frame_masks is not None and original_frames.shape[0] > 0:
            orig_frames = original_frames
            orig_masks = original_frame_masks
            if orig_masks.ndim == 2:
                orig_masks = orig_masks.unsqueeze(0)
            if tuple(orig_masks.shape[-2:]) != tuple(orig_frames.shape[1:3]):
                raise ValueError(
                    "original_frame_masks are {}x{} but original_frames are "
                    "{}x{}. They must match -- resize them together outside the "
                    "node so the two cannot drift apart.".format(
                        orig_masks.shape[-1], orig_masks.shape[-2],
                        orig_frames.shape[2], orig_frames.shape[1],
                    )
                )
            # A single mask applies to every frame.
            if orig_masks.shape[0] == 1 and orig_frames.shape[0] > 1:
                orig_masks = orig_masks.expand(orig_frames.shape[0], -1, -1)
            elif orig_masks.shape[0] != orig_frames.shape[0]:
                raise ValueError(
                    "got {} original_frame_masks for {} original_frames. Supply "
                    "one per frame, or a single mask to broadcast to all of "
                    "them.".format(orig_masks.shape[0], orig_frames.shape[0])
                )

            # Same chunk-offset logic as pose_video.
            if orig_frames.shape[0] <= video_frame_offset or orig_masks.shape[0] <= video_frame_offset:
                orig_frames = None
            else:
                orig_frames = orig_frames[video_frame_offset:]
                orig_masks = orig_masks[video_frame_offset:]

            if orig_frames is not None:
                # Trim jointly to a 4k+1 frame count so pixel frames group cleanly into latent frames.
                T_kept = ((min(orig_frames.shape[0], orig_masks.shape[0], length) - 1) // 4) * 4 + 1
                orig_frames = orig_frames[:T_kept]
                orig_masks = orig_masks[:T_kept]

                of = comfy.utils.common_upscale(orig_frames.movedim(-1, 1), width, height, "bicubic", "center").movedim(1, -1)
                orig_latent = vae.encode(of[:, :, :, :3])
                t_lat = min(orig_latent.shape[2], latent.shape[2])
                latent[:, :, :t_lat] = orig_latent[:, :, :t_lat].to(device=latent.device, dtype=latent.dtype)

                # Same geometry as the frames above, so the two cannot drift.
                om = comfy.utils.common_upscale(
                    orig_masks.unsqueeze(1).float(), width, height, "bilinear", "center"
                ).squeeze(1)
                _, keep, quantized_mask = _quantize_keep_mask(om.to(latent.device))

                keep5 = keep.movedim(1, 0).unsqueeze(0)  # (1, 1, T_lat_m, h, w)

                noise_mask = torch.ones((1, 1, latent.shape[2], latent.shape[-2], latent.shape[-1]), device=latent.device, dtype=latent.dtype)
                t_use = min(keep5.shape[2], t_lat)
                noise_mask[:, :, :t_use] = 1.0 - keep5[:, :, :t_use].to(device=latent.device, dtype=latent.dtype)

        if prev_trimmed is not None:
            pf = comfy.utils.common_upscale(prev_trimmed.movedim(-1, 1), width, height, "bicubic", "center").movedim(1, -1)
            prev_latent = vae.encode(pf[:, :, :, :3])
            prev_latent_frames = min(prev_latent.shape[2], latent.shape[2])
            # The extension anchor overrides any original-frame preservation on the leading frames.
            latent[:, :, :prev_latent_frames] = prev_latent[:, :, :prev_latent_frames].to(latent.dtype)
            if noise_mask is None:
                noise_mask = torch.ones((1, 1, latent.shape[2], latent.shape[-2], latent.shape[-1]), device=latent.device, dtype=latent.dtype)
            noise_mask[:, :, :prev_latent_frames] = 0.0
            # Reflect the anchor in the preview: those frames are locked too.
            if quantized_mask is not None:
                anchor_px = min(quantized_mask.shape[0], max(1, prev_latent_frames * 4 - 3))
                quantized_mask[:anchor_px] = 1.0

        if quantized_mask is None:
            quantized_mask = torch.zeros((1, height, width), device=latent.device, dtype=torch.float32)

        out_latent = {"samples": latent}
        if noise_mask is not None:
            out_latent["noise_mask"] = noise_mask
        return io.NodeOutput(positive, negative, out_latent, video_frame_offset + length, quantized_mask)


NODE = [WanSCAILToVideoLatentMasked]
