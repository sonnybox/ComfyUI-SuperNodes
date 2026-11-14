# ComfyUI-SuperNodes

Nodes that made my life easier on ComfyUI.

> [!IMPORTANT]
> I no longer publish distinct versions to Comfy registry. The latest git commit / nightly version is the only supported version.

## Current Nodes

### Color

* **Adjust Color AIO**: Adjust all color options at once
* **Adjust Brightness Contrast Gamma**: Adjust brightness, contrast, and gamma in one node
* **Adjust Hue Saturation**: Shift hue and adjust saturation
* **Adjust White Balance**: Correct white balance via temperature and tint

### Extras

* **Convert BBox to Masks**: Converts bounding box coordinates into a solid mask (requires WanAnimatePreprocess)

### Image

* **Crop Image using Mask**: Crops an image to a mask's bounding box, with optional padding and dimension constraints
* **Restore Mask Crop**: Restores a previously cropped masked region back into its original position
* **Pad Image Scaled**: An alternative method to pad images with adjustable positioning
* **Image Composite Masked**: The stock Image Composite Masked node, but allows negative x/y coordinates

### Scheduling

* **Sliced Shift Scheduler**: A simple scheduler with user specified sigma ranges
* **Dynamic Shift Scheduler**: Builds a flow match sigma schedule from a diffusers scheduler config
* **Sequence Length Calculator**: Works out the token count to feed a dynamic shift scheduler
* **Sigma Replace**: Replaces specific sigma values in a noise schedule
* **Sigma Remove**: Remove a sigma at the specified index
* **Sigma Insert**: Add a sigma at the specified index
* **Sigma Stride**: Select every nth sigma
* **Sigmas Smoother**: Add an intermediary step(s) before full denoise
* **Sigmas Rescale**: Rescales a sigma curve
* **Sigmas Interpolate**: Resamples a sigma curve to a different number of steps
* **Sigmas Graph**: Graph the sigma curve
* **Sigma Value**: Get a sigma's float value by index

### Sampling

* **DualSamplerCustomAdvanced**: Samples a packed audio-video latent with a separate sigma schedule per stream.
* **DualSamplerEulerAncestral**: Ancestral Euler with separate noise settings for the video and audio streams.
* **DualSamplerDPMPP_SDE**: DPM-Solver++ SDE with separate noise settings for the video and audio streams.
* **DualSamplerER_SDE**: ER-SDE with separate noise settings for the video and audio streams.
* **DualSamplerResMultistep**: res_multistep with separate noise settings for the video and audio streams.
* **SamplerResMultistepFlow**: res_multistep with an ancestral mode that uses the flow-model noise split.
* **Noise Seed**: RandomNoise with a toggle for DisableNoise that is also seeded.

### Tiling

* **Create Tiles**: Splits an image into an overlapping grid of tiles for tiled processing
* **Stitch Tiles**: Reassembles processed tiles back into a single image with feathered blending
* **SeedVR Calculate Tiles**: Probably useless
* **Color Match Luminance**: Luminance color match
* **Luminance Preprocess**: Grayscale based on luminance

### Tools

* **Get Aspect Ratio**: Snap to the nearest known aspect ratio
* **Image Size Calculator**: Calculates target width and height based on aspect ratio
* **Model Downloader**: Downloads a model from a URL into a selected ComfyUI folder
* **Select LoRA Name**: Picks a LoRA filename to share across multiple LoRA loaders
* **Set Reserve VRAM**: Sets --reserve-vram dynamically anywhere in a workflow
* **List Randomizer**: Selects a word or phrase from a string list
* **Concatenate Multi**: Concatenate multiple strings in one node
* **Show Error Message**: Halts the ComfyUI pipeline execution with a custom error message

### Video

* **Get Video Crop Frames**: Crops a frame batch to a uniform size driven by per-frame bboxes or masks (grown to the largest, optionally squared), recording crop metadata for restoration
* **Restore Video Crop Frames**: Pastes processed crops back onto the background frames using the recorded crop metadata, blending with per-frame masks (crop-space or background-space, auto-detected) or a feathered bbox edge (inner/outer/mid)
* **SAM3 Video Track (Mask Index)**: A modified version of the native video track node that accepts a mask based on a frame index
* **WanSCAILToVideo (Latent Masked)**: WanSCAILToVideo with optional original frames and masks to preserve (experimental)
