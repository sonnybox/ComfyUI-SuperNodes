from .adjustment import (
    SuperBrightnessContrast,
    SuperColorAdjustAllInOne,
    SuperHueSaturation,
    SuperLatentBrightnessContrast,
    SuperLatentChroma,
    SuperLatentColorAdjustAllInOne,
    SuperLatentHueRotate,
    SuperLatentLevelsNormalize,
    SuperLevelsNormalize,
    SuperWhiteBalanceCAT,
)
from .debug import (
    SuperLatentDeltaStats,
    SuperLatentStats,
    SuperLatentStatsPrint,
)
from .image import ImageMaskCrop, RestoreMaskCrop
from .qol import (
    FaceBBoxToMask,
    GetCommonAspectRatio,
    ImageSizeCalculator,
    SetReserveVRAM,
    SuperResizeImage,
)
from .scheduler import SigmaReplace, SigmasGraph, SigmaSmoother, SigmasRescale
from .tiling import CreateTiles, SeedVRCalculateTiles, StitchTiles

NODE_CLASS_MAPPINGS = {
    "SuperCreateTiles": CreateTiles,
    "SuperStitchTiles": StitchTiles,
    "SuperSVRCalcTiles": SeedVRCalculateTiles,
    "SuperResizeImage": SuperResizeImage,
    "SuperBrightnessContrast": SuperBrightnessContrast,
    "SuperHueSaturation": SuperHueSaturation,
    "SuperWhiteBalanceCAT": SuperWhiteBalanceCAT,
    "SuperColorAdjustAllInOne": SuperColorAdjustAllInOne,
    "SuperLevelsNormalize": SuperLevelsNormalize,
    "SuperLatentBrightnessContrast": SuperLatentBrightnessContrast,
    "SuperLatentLevelsNormalize": SuperLatentLevelsNormalize,
    "SuperLatentChroma": SuperLatentChroma,
    "SuperLatentHueRotate": SuperLatentHueRotate,
    "SuperLatentColorAdjustAllInOne": SuperLatentColorAdjustAllInOne,
    "SigmaSmoother": SigmaSmoother,
    "SigmasRescale": SigmasRescale,
    "SigmaReplace": SigmaReplace,
    "ImageSizeCalculator": ImageSizeCalculator,
    "ImageMaskCrop": ImageMaskCrop,
    "RestoreMaskCrop": RestoreMaskCrop,
    "SigmasGraph": SigmasGraph,
    "FaceBBoxToMask": FaceBBoxToMask,
    "GetCommonAspectRatio": GetCommonAspectRatio,
    "SetReserveVRAM": SetReserveVRAM,
    "SuperLatentStats": SuperLatentStats,
    "SuperLatentStatsPrint": SuperLatentStatsPrint,
    "SuperLatentDeltaStats": SuperLatentDeltaStats,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SuperCreateTiles": "🐧 Create Tiles",
    "SuperStitchTiles": "🐧 Stitch Tiles",
    "SuperSVRCalcTiles": "🐧 SeedVR Calculate Tiles",
    "SuperBrightnessContrast": "🐧 Adjust Brightness Contrast Gamma",
    "SuperHueSaturation": "🐧 Adjust Hue Saturation",
    "SuperWhiteBalanceCAT": "🐧 Adjust White Balance",
    "SuperColorAdjustAllInOne": "🐧 Adjust Color AIO",
    "SuperLevelsNormalize": "🐧 Normalize Levels",
    "SuperLatentBrightnessContrast": "🐧 Latent Adjust Basic",
    "SuperLatentLevelsNormalize": "🐧 Latent Normalize Levels",
    "SuperLatentChroma": "🐧 Latent Saturation",
    "SuperLatentHueRotate": "🐧 Latent Hue Rotate",
    "SuperLatentColorAdjustAllInOne": "🐧 Latent Adjust AIO",
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
    "SetReserveVRAM": "🐧 Set Reserve VRAM",
    "SuperLatentStats": "🐧 Latent Stats",
    "SuperLatentStatsPrint": "🐧 Latent Stats (Print)",
    "SuperLatentDeltaStats": "🐧 Latent Delta Stats",
}

print("\033[34m[SuperNodes]\033[0m Classes initialized.")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
