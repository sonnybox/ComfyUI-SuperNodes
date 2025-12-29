import torch


class WanExtendI2VPlus:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": (
                    "LATENT",
                    {
                        "tooltip": "The output latent from the previous KSampler (the video you want to extend)."
                    },
                ),
                "context_frames": (
                    "INT",
                    {
                        "default": 16,
                        "min": 4,
                        "max": 128,
                        "step": 4,
                        "tooltip": "Number of pixel frames to keep from the previous batch to use as context (overlap). Must be divisible by 4 (Wan default compression).",
                    },
                ),
                "temporal_compression": (
                    "INT",
                    {
                        "default": 4,
                        "min": 1,
                        "max": 32,
                        "step": 1,
                        "tooltip": "The temporal compression factor of the VAE. Wan/Video models usually compress 4 frames into 1 latent.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    OUTPUT_TOOLTIPS = (
        "The new latent batch prepared for the next KSampler step, with context frames masked.",
    )
    FUNCTION = "extend_latent"

    CATEGORY = "Wan/latent"
    DESCRIPTION = "Takes the last N frames from a latent batch, moves them to the front of a new empty batch, and creates a mask so only the new frames are generated. Allows infinite generation loops without VAE decoding."

    def extend_latent(self, samples, context_frames, temporal_compression=4):
        # 1. Calculate Latent Dimensions
        # Wan uses temporal compression (usually 4).
        # If user asks for 16 pixel frames context, we need 16 / 4 = 4 latent frames.
        context_latent_length = context_frames // temporal_compression

        # Get the source latent tensor
        # Shape is usually [Batch, Channels, T (Time), H, W]
        source_samples = samples["samples"]
        batch, channels, total_frames, height, width = source_samples.shape

        # 2. Validation
        if context_latent_length >= total_frames:
            raise ValueError(
                f"Context frames ({context_frames}px / {context_latent_length}lat) cannot be larger than the input video length ({total_frames}lat)."
            )

        # 3. Create New Empty Latent Batch (Same size as input)
        # We start with zeros (empty)
        new_samples = torch.zeros_like(source_samples)

        # 4. Slice and Paste Context
        # Grab the LAST 'context_latent_length' frames from the source
        context_slice = source_samples[:, :, -context_latent_length:, :, :]

        # Paste them into the BEGINNING of the new batch
        new_samples[:, :, :context_latent_length, :, :] = context_slice

        # 5. Create the Noise Mask
        # The mask tells the KSampler which pixels to denoise (1.0) and which to keep fixed (0.0).
        # We need a mask of shape [1, T, H, W] (Channels dim is broadcasted usually, or handled by sampler)

        # Initialize mask with 1.0 (denoise everything / white)
        # Note: SetLatentNoiseMask uses [Batch, 1, H, W] usually, but for video it's [Batch, 1, T, H, W] check?
        # Standard ComfyUI masks in latents are often [Batch, 1, H, W] for images.
        # For Video latents (5D), the mask should typically be 5D or broadcastable.
        # We will create a [1, 1, T, H, W] mask to be safe.

        mask = torch.ones(
            (1, 1, total_frames, height, width),
            dtype=source_samples.dtype,
            device=source_samples.device,
        )

        # Set the context area (frames 0 to context_length) to 0.0 (Block noise/Keep existing)
        mask[:, :, :context_latent_length, :, :] = 0.0

        # 6. Package Output
        # We must clone the dictionary to avoid mutating the original input in the graph
        out = samples.copy()
        out["samples"] = new_samples
        out["noise_mask"] = mask

        # Handle batch_index if it exists (standardize it for the new batch)
        if "batch_index" in out:
            out["batch_index"] = [x for x in range(batch)]

        return (out,)
