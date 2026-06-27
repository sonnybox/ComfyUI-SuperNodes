# ComfyUI SuperNodes

## Current Nodes

### Color

* **Adjust Color AIO**: Adjust everything at once
* **Adjust Brightness Contrast Gamma**: Adjust brightness, contrast, and gamma in one node
* **Adjust Hue Saturation**: Shift hue and adjust saturation
* **Adjust White Balance**: Correct white balance via temperature and tint

### Extras

* **Convert BBox To Mask**: Converts bounding box coordinates into a solid mask (requires WanAnimatePreprocess)

### Image

* **Crop Image using Mask**: Crops an image to a mask's bounding box, with optional padding and dimension constraints
* **Pad Image Scaled**: An alternative method to pad images with adjustable positioning
* **Resize Image & Mask**: Resize an image and mask together with selectable scaling modes
* **Restore Mask Crop**: Restores a previously cropped masked region back into its original position

### Scheduling

* **Sigma Replace**: Replaces specific sigma values in a noise schedule
* **Sigma Smoother**: Add an intermediary step(s) before full denoise
* **Sigmas Graph**: Graph the sigma curve
* **Sigmas Rescale**: Rescales a sigma curve
* **Sigma Remove**: Remove a sigma at the specified index
* **Sigma Insert**: Add a sigma at the specified index

### Tiling

* **Create Tiles**: Splits an image into an overlapping grid of tiles for tiled processing
* **Stitch Tiles**: Reassembles processed tiles back into a single image with feathered blending
* **Color Match Luminance**: Luminance color match
* **Luminance Preprocess**: Grayscale based on luminance
* **Seed VR Calculate Tiles**: Probably useless

### Tools

* **Get Aspect Ratio**: Snap to the nearest known aspect ratio
* **Image Size Calculator**: Calculates target width and height based on aspect ratio
* **Model Downloader**: Downloads a model from a URL into a selected ComfyUI folder
* **Set Reserve VRAM**: Sets --reserve-vram dynamically anywhere in a workflow
* **Show Error Message**: Halts the ComfyUI pipeline execution with a custom error message
* **List Randomizer**: Selects a word or phrase from a string list
* **Concatenate Multi**: Concatenate multiple strings in one node

### Video

* **Get BBox Crop Frames**: Crops a frame batch to a uniform size driven by per-frame bboxes (grown to the largest, optionally squared), recording each frame's bbox for restoration
* **Restore BBox Crop Frames**: Pastes processed crops back onto the background frames using the recorded bboxes, with optional feathered blending
