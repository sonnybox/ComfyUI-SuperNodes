import comfy.model_management
import comfy.utils
from comfy_api.latest import io
import torch

SAM3TrackData = io.Custom("SAM3_TRACK_DATA")


def _merge_track_segments(fwd, bwd, n_frames, start_index, n_shared):
    """
    Stitch a forward run (frames start_index..N-1) and a reversed backward run
    (frames start_index..0) into a single [N, N_obj, ...] packed track result.

    Objects seeded from initial_mask hold the same slot in both runs, so they merge
    into one continuous track. Objects the backward run detected on its own are
    appended as extra slots after the forward objects. Frames no run covered stay
    empty, which keeps the output frame count equal to the input frame count.
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
    # Shared objects keep slots 0..shared-1; backward-only objects start after them.
    shared = min(n_shared, n_bwd)
    base = max(n_fwd, shared)
    total = base + max(0, n_bwd - n_shared)

    sample = fwd_packed if fwd_packed is not None else bwd_packed
    out = torch.zeros(
        n_frames, total, *sample.shape[2:], dtype=torch.uint8, device=sample.device
    )
    if fwd_packed is not None:
        out[start_index:, :n_fwd] = fwd_packed.to(out.device)
    if bwd_packed is not None:
        cols = [j if j < n_shared else base + (j - n_shared) for j in range(n_bwd)]
        out[:start_index, cols] = bwd_packed.to(out.device)

    # Per-object detection scores, aligned to the merged object slots.
    scores = [1.0] * total
    fwd_scores = fwd.get("scores", []) if fwd is not None else []
    bwd_scores = bwd.get("scores", []) if bwd is not None else []
    for i in range(min(n_fwd, len(fwd_scores), total)):
        scores[i] = fwd_scores[i]
    for j in range(n_shared, n_bwd):
        col = base + (j - n_shared)
        if j < len(bwd_scores) and col < total:
            scores[col] = bwd_scores[j]

    return {"packed_masks": out, "n_frames": n_frames, "scores": scores}


class SAM3VideoTrackIndex(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SAM3VideoTrackIndex",
            display_name="🐧 SAM3 Video Track (Index)",
            category="SuperNodes/Video",
            description=(
                "SAM3 video tracking that can start from any frame instead of only the first. "
                "Tracks forward from start_index and, with bidirectional enabled, tracks the "
                "earlier frames in reverse and merges both halves. Output mask count always "
                "matches the input frame count; uncovered frames are empty."
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
                    "initial_mask",
                    optional=True,
                    tooltip="Mask(s) for the start_index frame to track (one per object).",
                ),
                io.Conditioning.Input(
                    "conditioning",
                    optional=True,
                    tooltip="Text conditioning for detecting new objects during tracking.",
                ),
                io.Int.Input(
                    "start_index",
                    default=0,
                    min=0,
                    max=99999,
                    step=1,
                    tooltip=(
                        "Frame the initial_mask belongs to (0 = first frame, matching the native node). "
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
                io.Float.Input(
                    "detection_threshold",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Score threshold for text-prompted detection.",
                ),
                io.Int.Input(
                    "max_objects",
                    default=4,
                    min=0,
                    max=64,
                    step=1,
                    tooltip=(
                        "Max tracked objects per direction. Initial masks count toward this limit. "
                        "0 uses the internal cap of 64."
                    ),
                ),
                io.Int.Input(
                    "detect_interval",
                    default=1,
                    min=1,
                    max=1000,
                    step=1,
                    tooltip="Run detection every N frames (1=every frame). Higher values save compute.",
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
        start_index,
        bidirectional,
        detection_threshold,
        max_objects,
        detect_interval,
        initial_mask=None,
        conditioning=None,
    ) -> io.NodeOutput:
        from comfy_extras.nodes_sam3 import _extract_text_prompts

        N, H, W, C = images.shape

        comfy.model_management.load_model_gpu(model)
        device = comfy.model_management.get_torch_device()
        dtype = model.model.get_dtype()
        sam3_model = model.model.diffusion_model

        frames_in = images[..., :3].movedim(-1, 1)

        init_masks = None
        n_shared = 0
        if initial_mask is not None:
            init_masks = initial_mask.unsqueeze(1).to(device=device, dtype=dtype)
            n_shared = init_masks.shape[0]

        text_prompts = None
        if conditioning is not None and len(conditioning) > 0:
            text_prompts = [
                (emb, mask)
                for emb, mask, _ in _extract_text_prompts(conditioning, device, dtype)
            ]
        elif initial_mask is None:
            raise ValueError("Either initial_mask or conditioning must be provided")

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
                text_prompts=text_prompts,
                new_det_thresh=detection_threshold,
                max_objects=max_objects,
                detect_interval=detect_interval,
                target_device=device,
                target_dtype=dtype,
            )

        fwd = _track(frames_in[start_index:])

        # start_index 0 is the native single-pass behavior, nothing to stitch.
        if start_index == 0:
            fwd["orig_size"] = (H, W)
            return io.NodeOutput(fwd)

        bwd = _track(frames_in[: start_index + 1].flip(0)) if run_backward else None

        result = _merge_track_segments(fwd, bwd, N, start_index, n_shared)
        result["orig_size"] = (H, W)
        return io.NodeOutput(result)


NODE = [SAM3VideoTrackIndex]
