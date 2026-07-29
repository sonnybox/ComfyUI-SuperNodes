import comfy.model_management
import comfy.utils
from comfy_api.latest import io
import torch

SAM3TrackData = io.Custom("SAM3_TRACK_DATA")


def _merge_track_segments(fwd, bwd, n_frames, start_index):
    """
    Stitch a forward run (frames start_index..N-1) and a reversed backward run
    (frames start_index..0) into a single [N, N_obj, ...] packed track result.

    Both runs are seeded from the same mask in the same order, so object slot j
    means the same object in both and the two halves merge directly. Frames no run
    covered stay empty, keeping the output frame count equal to the input count.
    """
    fwd_packed = fwd["packed_masks"] if fwd is not None else None
    bwd_packed = None
    if bwd is not None and bwd["packed_masks"] is not None:
        # Drop the pivot frame (shared with the forward run) and restore original order.
        bwd_packed = bwd["packed_masks"][1:].flip(0)  # frames 0..start_index-1
        if bwd_packed.shape[0] == 0:
            bwd_packed = None

    if fwd_packed is None and bwd_packed is None:
        return {"packed_masks": None, "n_frames": n_frames, "scores": []}

    n_fwd = fwd_packed.shape[1] if fwd_packed is not None else 0
    n_bwd = bwd_packed.shape[1] if bwd_packed is not None else 0
    total = max(n_fwd, n_bwd)

    sample = fwd_packed if fwd_packed is not None else bwd_packed
    out = torch.zeros(
        n_frames, total, *sample.shape[2:], dtype=torch.uint8, device=sample.device
    )
    if fwd_packed is not None:
        out[start_index:, :n_fwd] = fwd_packed.to(out.device)
    if bwd_packed is not None:
        out[:start_index, :n_bwd] = bwd_packed.to(out.device)

    scores = list(fwd.get("scores", []) if fwd is not None else [])[:total]
    scores += [1.0] * (total - len(scores))

    return {"packed_masks": out, "n_frames": n_frames, "scores": scores}


class SAM3VideoTrackIndex(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SAM3VideoTrackIndex",
            display_name="🐧 SAM3 Video Track (Mask Index)",
            category="SuperNodes/Video",
            description=(
                "SAM3 video tracking seeded from a mask on any frame instead of only the first. "
                "Tracks forward from start_index and, with bidirectional enabled, tracks the "
                "earlier frames in reverse and merges both halves. Output mask count always "
                "matches the input frame count; uncovered frames are empty. "
                "For text-prompted detection use the native SAM3 Video Track node."
            ),
            search_aliases=[
                "sam3",
                "video",
                "track",
                "propagate",
                "index",
                "bidirectional",
                "reverse",
            ],
            inputs=[
                io.Image.Input("images", tooltip="Video frames as a batched image [B,H,W,C]."),
                io.Model.Input("model", tooltip="Loaded SAM3 model."),
                io.Mask.Input(
                    "mask",
                    tooltip="Mask(s) for the start_index frame to track (one per object).",
                ),
                io.Int.Input(
                    "start_index",
                    default=0,
                    min=0,
                    max=99999,
                    step=1,
                    tooltip=(
                        "Frame the mask belongs to (0 = first frame, matching the native node). "
                        "Tracking runs forward from here."
                    ),
                ),
                io.Boolean.Input(
                    "bidirectional",
                    default=True,
                    tooltip=(
                        "Also track backwards from start_index to frame 0 by reversing the earlier frames. "
                        "Off leaves those frames empty. Ignored when start_index is 0."
                    ),
                ),
            ],
            outputs=[
                SAM3TrackData.Output(
                    display_name="track_data",
                    tooltip="Track data for SAM3 Track to Mask / SAM3 Track Preview.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        images,
        model,
        mask,
        start_index,
        bidirectional,
    ) -> io.NodeOutput:
        N, H, W, C = images.shape

        comfy.model_management.load_model_gpu(model)
        device = comfy.model_management.get_torch_device()
        dtype = model.model.get_dtype()
        sam3_model = model.model.diffusion_model

        frames_in = images[..., :3].movedim(-1, 1)
        init_masks = mask.unsqueeze(1).to(device=device, dtype=dtype)

        start_index = max(0, min(int(start_index), N - 1))
        run_backward = bool(bidirectional) and start_index > 0

        # The backward run re-processes the pivot frame, so it costs start_index + 1 steps.
        total_steps = (N - start_index) + (start_index + 1 if run_backward else 0)
        pbar = comfy.utils.ProgressBar(total_steps)

        def _track(seq):
            return sam3_model.forward_video(
                images=seq,
                initial_masks=init_masks,
                pbar=pbar,
                text_prompts=None,
                target_device=device,
                target_dtype=dtype,
            )

        fwd = _track(frames_in[start_index:])

        # start_index 0 is the native single-pass behavior, nothing to stitch.
        if start_index == 0:
            fwd["orig_size"] = (H, W)
            return io.NodeOutput(fwd)

        bwd = _track(frames_in[: start_index + 1].flip(0)) if run_backward else None

        result = _merge_track_segments(fwd, bwd, N, start_index)
        result["orig_size"] = (H, W)
        return io.NodeOutput(result)


NODE = [SAM3VideoTrackIndex]
