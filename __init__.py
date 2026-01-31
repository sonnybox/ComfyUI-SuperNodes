from .image import ImageMaskCrop, RestoreMaskCrop
from .qol import (
    FaceBBoxToMask,
    GetCommonAspectRatio,
    ImageSizeCalculator,
    SuperResizeImage,
)
from .scheduler import SigmaReplace, SigmasGraph, SigmaSmoother, SigmasRescale
from .tiling import CreateTiles, SeedVRCalculateTiles, StitchTiles

NODE_CLASS_MAPPINGS = {
    "SuperCreateTiles": CreateTiles,
    "SuperStitchTiles": StitchTiles,
    "SuperSVRCalcTiles": SeedVRCalculateTiles,
    "SuperResizeImage": SuperResizeImage,
    "SigmaSmoother": SigmaSmoother,
    "SigmasRescale": SigmasRescale,
    "SigmaReplace": SigmaReplace,
    "ImageSizeCalculator": ImageSizeCalculator,
    "ImageMaskCrop": ImageMaskCrop,
    "RestoreMaskCrop": RestoreMaskCrop,
    "SigmasGraph": SigmasGraph,
    "FaceBBoxToMask": FaceBBoxToMask,
    "GetCommonAspectRatio": GetCommonAspectRatio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SuperCreateTiles": "🐧 Create Tiles",
    "SuperStitchTiles": "🐧 Stitch Tiles",
    "SuperSVRCalcTiles": "🐧 SeedVR Calculate Tiles",
    "SigmaSmoother": "🐧 Sigma Smoother",
    "SigmasRescale": "🐧 Sigmas Rescale",
    "SigmaReplace": "🐧 Sigma Replace",
    "SuperResizeImage": "🐧 Crop Resize Image",
    "ImageSizeCalculator": "🐧 Image Size Calculator",
    "ImageMaskCrop": "🐧 Image Mask Crop",
    "RestoreMaskCrop": "🐧 Restore Mask Crop",
    "SigmasGraph": "🐧 Sigmas Graph",
    "FaceBBoxToMask": "🐧 Face BBox Masks",
    "GetCommonAspectRatio": "🐧 Get Aspect Ratio",
}

print("\033[34m[SuperNodes]\033[0m Classes initialized.")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
