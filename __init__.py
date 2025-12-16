from .scheduler import SigmaSmoother
from .tiling import CreateTiles, SeedVRCalculateTiles, StitchTiles

NODE_CLASS_MAPPINGS = {
    "SuperCreateTiles": CreateTiles,
    "SuperStitchTiles": StitchTiles,
    "SuperSVRCalcTiles": SeedVRCalculateTiles,
    "SigmaSmoother": SigmaSmoother,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SuperCreateTiles": "🐧 Create Tiles",
    "SuperStitchTiles": "🐧 Stitch Tiles",
    "SuperSVRCalcTiles": "🐧 SeedVR Calculate Tiles",
    "SigmaSmoother": "🐧 Sigma Smoother",
}

print("\033[34m[SuperNodes]\033[0m: Loaded Penguin Nodes.")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
