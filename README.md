# ComfyUI SuperNodes

## Current Nodes

### Color

* **Adjust Color AIO**: Adjust everything at once
* **Adjust Brightness Contrast Gamma**
* **Adjust Hue Saturation**
* **Adjust White Balance**

### Extras

* **Convert BBox To Mask**: Converts bounding box coordinates into a solid mask (requires WanAnimatePreprocess)

### Image

* **Crop Image using Mask**
* **Pad Image Scaled**: An alternative method to pad images with adjustable positioning
* **Resize Image & Mask**
* **Restore Mask Crop**: Restores a previously cropped masked region back into its original position

### Scheduling

* **Sigma Replace**: Replaces specific sigma values in a noise schedule
* **Sigma Smoother**: Add an intermediary step(s) before full denoise
* **Sigmas Graph**: Graph the sigma curve
* **Sigmas Rescale**: Rescales a sigma curve

### Tiling

* **Color Match Tiling**: Luminance color match
* **Luminance Preprocess**: Grayscale based on luminance
* **Seed VR Calculate Tiles**: Probably useless

### Tools

* **Get Aspect Ratio**: Snap to the nearest known aspect ratio
* **Image Size Calculator**: Calculates target width and height based on aspect ratio
* **Model Downloader**
* **Set Reserve VRAM**
* **Show Error Message**: Halts the ComfyUI pipeline execution with a custom error message
* **List Randomizer**: Selects a word or phrase from a string list
* **Concatenate Multi**: Concatenate multiple strings in one node
