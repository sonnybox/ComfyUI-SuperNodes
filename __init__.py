from .image import ImageMaskCrop, RestoreMaskCrop
from .qol import ImageSizeCalculator
from .scheduler import SigmaSmoother, SigmasRescale
from .tiling import CreateTiles, SeedVRCalculateTiles, StitchTiles
from .video import WanExtendI2VPlus

NODE_CLASS_MAPPINGS = {
    "SuperCreateTiles": CreateTiles,
    "SuperStitchTiles": StitchTiles,
    "SuperSVRCalcTiles": SeedVRCalculateTiles,
    "SigmaSmoother": SigmaSmoother,
    "SigmasRescale": SigmasRescale,
    "ImageSizeCalculator": ImageSizeCalculator,
    "ImageMaskCrop": ImageMaskCrop,
    "RestoreMaskCrop": RestoreMaskCrop,
    "WanExtendI2VPlus": WanExtendI2VPlus,
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
    "WanExtendI2VPlus": "🐧 Wan Extend I2V+",
}

print("\033[34m[SuperNodes]\033[0m Loaded successfully.")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
