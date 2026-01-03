from .image import ImageMaskCrop, RestoreMaskCrop
from .qol import ImageSizeCalculator
from .scheduler import SigmaSmoother, SigmasGraph, SigmasRescale, LoadDiffusersScheduler
from .tiling import CreateTiles, SeedVRCalculateTiles, StitchTiles

NODE_CLASS_MAPPINGS = {
    "SuperCreateTiles": CreateTiles,
    "SuperStitchTiles": StitchTiles,
    "SuperSVRCalcTiles": SeedVRCalculateTiles,
    "SigmaSmoother": SigmaSmoother,
    "SigmasRescale": SigmasRescale,
    "ImageSizeCalculator": ImageSizeCalculator,
    "ImageMaskCrop": ImageMaskCrop,
    "RestoreMaskCrop": RestoreMaskCrop,
    "LoadDiffusersScheduler": LoadDiffusersScheduler,
    "SigmasGraph": SigmasGraph,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SuperCreateTiles": "🐧 Create Tiles",
    "SuperStitchTiles": "🐧 Stitch Tiles",
    "SuperSVRCalcTiles": "🐧 SeedVR Calculate Tiles",
    "SigmaSmoother": "🐧 Sigma Smoother",
    "SigmasRescale": "🐧 Sigmas Rescale",
    "ImageSizeCalculator": "🐧 Image Size Calculator",
    "ImageMaskCrop": "🐧 Image Mask Crop",
    "RestoreMaskCrop": "🐧 Restore Mask Crop",
    "LoadDiffusersScheduler": "🐧 Load Diffusers Scheduler",
    "SigmasGraph": "🐧 Sigmas Graph",
}

print("\033[34m[SuperNodes]\033[0m Classes initialized.")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
