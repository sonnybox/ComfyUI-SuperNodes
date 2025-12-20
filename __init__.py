from .scheduler import SigmaSmoother, SigmasRescale
from .tiling import CreateTiles, SeedVRCalculateTiles, StitchTiles

NODE_CLASS_MAPPINGS = {
    "SuperCreateTiles": CreateTiles,
    "SuperStitchTiles": StitchTiles,
    "SuperSVRCalcTiles": SeedVRCalculateTiles,
    "SigmaSmoother": SigmaSmoother,
    "SigmasRescale": SigmasRescale,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SuperCreateTiles": "🐧 Create Tiles",
    "SuperStitchTiles": "🐧 Stitch Tiles",
    "SuperSVRCalcTiles": "🐧 SeedVR Calculate Tiles",
    "SigmaSmoother": "🐧 Sigma Smoother",
    "SigmasRescale": "🐧 Sigmas Rescale",
}

print("\033[34m[SuperNodes]\033[0m Loaded successfully.")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
