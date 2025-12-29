import torch


class WanExtendI2VPlus:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": (
                    "LATENT",
                    {
                        "tooltip": "The output latent from the previous KSampler."
                    },
                ),
                "context_frames": (
                    "INT",
                    {
                        "default": 16,
                        "min": 1,
                        "max": 128,
                        "step": 1,
                        "tooltip": "Pixel frames to keep. Must match your VAE compression (e.g., 16 pixels = 4 latents).",
                    },
                ),
                "temporal_compression": (
                    "INT",
                    {
                        "default": 4,
                        "min": 1,
                        "max": 32,
                        "step": 1,
                        "tooltip": "Wan usually compresses 4 frames into 1 latent.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    OUTPUT_TOOLTIPS = (
        "The new latent batch with context frames 'locked' via a noise mask.",
    )
    FUNCTION = "extend_latent"

    CATEGORY = "Wan/latent"
    DESCRIPTION = "Extends a video latent batch by copying the last N frames to the start of a new batch and masking them so they don't change."

    def extend_latent(self, samples, context_frames, temporal_compression=4):
        source_samples = samples["samples"]

        # Calculate how many LATENT frames correspond to the requested PIXEL frames
        # e.g., 16 pixel frames / 4 compression = 4 latent frames
        ctx_lat_len = context_frames // temporal_compression

        # --- SHAPE DETECTION ---
        # 5D: [Batch, Channels, Time, Height, Width] (Native Video Format)
        # 4D: [Batch, Channels, Height, Width] (Standard Comfy Format where Batch=Time)

        is_5d = len(source_samples.shape) == 5

        if is_5d:
            b, c, t, h, w = source_samples.shape
            total_len = t
        else:
            b, c, h, w = source_samples.shape
            total_len = b

        if ctx_lat_len >= total_len:
            raise ValueError(
                f"Context ({ctx_lat_len} latents) is larger than input video ({total_len} latents)."
            )

        # --- CREATE NEW TENSOR ---
        new_samples = torch.zeros_like(source_samples)

        # --- SLICE & PASTE ---
        if is_5d:
            # Copy last N from source -> Paste to first N of new
            # Shape: [Batch, Channel, TIME, Height, Width]
            context_slice = source_samples[:, :, -ctx_lat_len:, :, :]
            new_samples[:, :, :ctx_lat_len, :, :] = context_slice

            # --- CREATE MASK (5D) ---
            # Mask shape must be [Batch, 1, Time, Height, Width]
            mask = torch.ones(
                (b, 1, t, h, w),
                dtype=source_samples.dtype,
                device=source_samples.device,
            )
            mask[:, :, :ctx_lat_len, :, :] = 0.0  # Lock the context frames

        else:
            # Shape: [BATCH(Time), Channel, Height, Width]
            context_slice = source_samples[-ctx_lat_len:, :, :, :]
            new_samples[:ctx_lat_len, :, :, :] = context_slice

            # --- CREATE MASK (4D) ---
            # Mask shape must be [Batch, 1, Height, Width]
            mask = torch.ones(
                (b, 1, h, w),
                dtype=source_samples.dtype,
                device=source_samples.device,
            )
            mask[:ctx_lat_len, :, :, :] = 0.0  # Lock the context frames

        # --- OUTPUT ---
        out = samples.copy()
        out["samples"] = new_samples
        out["noise_mask"] = mask

        # Handle batch index for correct noise generation in some samplers
        if "batch_index" in out:
            out["batch_index"] = [x for x in range(source_samples.shape[0])]

        return (out,)
