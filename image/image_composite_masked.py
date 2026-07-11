import comfy.utils
import node_helpers
import torch
from comfy_api.latest import io

MAX_RESOLUTION = 16384


def composite(destination, source, x, y, mask=None, multiplier=1, resize_source=False):
    source = source.to(destination.device)
    if resize_source:
        source = torch.nn.functional.interpolate(
            source,
            size=(destination.shape[-2], destination.shape[-1]),
            mode="bilinear",
        )

    source = comfy.utils.repeat_to_batch_size(source, destination.shape[0])

    # clamp so the source can move fully off-canvas in any direction
    x = max(-source.shape[-1] * multiplier, min(x, destination.shape[-1] * multiplier))
    y = max(-source.shape[-2] * multiplier, min(y, destination.shape[-2] * multiplier))

    left, top = (x // multiplier, y // multiplier)

    # negative offsets crop the source, positive offsets shift into the destination
    src_left, src_top = (max(0, -left), max(0, -top))
    dst_left, dst_top = (max(0, left), max(0, top))

    visible_width = min(destination.shape[-1] - dst_left, source.shape[-1] - src_left)
    visible_height = min(destination.shape[-2] - dst_top, source.shape[-2] - src_top)

    if visible_width <= 0 or visible_height <= 0:
        return destination

    if mask is None:
        mask = torch.ones_like(source)
    else:
        mask = mask.to(destination.device, copy=True)
        mask = torch.nn.functional.interpolate(
            mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])),
            size=(source.shape[-2], source.shape[-1]),
            mode="bilinear",
        )
        mask = comfy.utils.repeat_to_batch_size(mask, source.shape[0])

    mask = mask[:, :, src_top:src_top + visible_height, src_left:src_left + visible_width]
    if mask.ndim < source.ndim:
        mask = mask.unsqueeze(1)

    inverse_mask = torch.ones_like(mask) - mask

    source_portion = mask * source[..., src_top:src_top + visible_height, src_left:src_left + visible_width]
    destination_portion = inverse_mask * destination[..., dst_top:dst_top + visible_height, dst_left:dst_left + visible_width]

    destination[..., dst_top:dst_top + visible_height, dst_left:dst_left + visible_width] = source_portion + destination_portion
    return destination


class SuperImageCompositeMasked(io.ComfyNode):
    """
    Vanilla ImageCompositeMasked with negative x/y offsets allowed, so the
    source layer can be moved in any direction (like dragging a Photoshop
    layer). Pixels that fall outside the destination are cropped.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SuperImageCompositeMasked",
            display_name="🐧 Image Composite Masked",
            category="SuperNodes/Image",
            inputs=[
                io.Image.Input("destination", tooltip="The canvas the source is pasted onto."),
                io.Image.Input("source", tooltip="The layer to paste onto the destination."),
                io.Int.Input(
                    "x",
                    default=0,
                    min=-MAX_RESOLUTION,
                    max=MAX_RESOLUTION,
                    step=1,
                    tooltip="Horizontal offset in pixels. Negative moves the source left past the canvas edge.",
                ),
                io.Int.Input(
                    "y",
                    default=0,
                    min=-MAX_RESOLUTION,
                    max=MAX_RESOLUTION,
                    step=1,
                    tooltip="Vertical offset in pixels. Negative moves the source up past the canvas edge.",
                ),
                io.Boolean.Input(
                    "resize_source",
                    default=False,
                    tooltip="Resize the source to match the destination before compositing.",
                ),
                io.Mask.Input("mask", optional=True, tooltip="Optional mask; white areas take the source pixels."),
            ],
            outputs=[io.Image.Output(display_name="IMAGE")],
        )

    @classmethod
    def execute(cls, destination, source, x, y, resize_source, mask=None) -> io.NodeOutput:
        destination, source = node_helpers.image_alpha_fix(destination, source)
        destination = destination.clone().movedim(-1, 1)
        output = composite(destination, source.movedim(-1, 1), x, y, mask, 1, resize_source).movedim(1, -1)
        return io.NodeOutput(output)


NODE = [SuperImageCompositeMasked]
