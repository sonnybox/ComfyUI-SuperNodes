from .image import ImageMaskCrop, RestoreMaskCrop
from .qol import ImageSizeCalculator, SuperResizeImage
from .scheduler import SigmasGraph, SigmaSmoother, SigmasRescale
from .tiling import CreateTiles, SeedVRCalculateTiles, StitchTiles

NODE_CLASS_MAPPINGS = {
    "SuperCreateTiles": CreateTiles,
    "SuperStitchTiles": StitchTiles,
    "SuperSVRCalcTiles": SeedVRCalculateTiles,
    "SuperResizeImage": SuperResizeImage,
    "SigmaSmoother": SigmaSmoother,
    "SigmasRescale": SigmasRescale,
    "ImageSizeCalculator": ImageSizeCalculator,
    "ImageMaskCrop": ImageMaskCrop,
    "RestoreMaskCrop": RestoreMaskCrop,
    "SigmasGraph": SigmasGraph,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SuperCreateTiles": "🐧 Create Tiles",
    "SuperStitchTiles": "🐧 Stitch Tiles",
    "SuperSVRCalcTiles": "🐧 SeedVR Calculate Tiles",
    "SigmaSmoother": "🐧 Sigma Smoother",
    "SigmasRescale": "🐧 Sigmas Rescale",
    "SuperResizeImage": "🐧 Crop Resize Image",
    "ImageSizeCalculator": "🐧 Image Size Calculator",
    "ImageMaskCrop": "🐧 Image Mask Crop",
    "RestoreMaskCrop": "🐧 Restore Mask Crop",
    "SigmasGraph": "🐧 Sigmas Graph",
}

print("\033[34m[SuperNodes]\033[0m Classes initialized.")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
