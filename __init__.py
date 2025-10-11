# Import both node classes from our logic files
from .main import CreateTiles
from .main import StitchTiles
from .main import CalculateUpscaleTiles

# A dictionary that maps the node's internal name to its class
NODE_CLASS_MAPPINGS = {
    "SuperCreateTiles": CreateTiles,
    "SuperStitchTiles": StitchTiles,
    "SuperCalculateTiles": CalculateUpscaleTiles
}

# A dictionary that maps the node's internal name to its display name in the UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "SuperCreateTiles": "🐧 Create Tiles",
    "SuperStitchTiles": "🐧 Stitch Tiles",
    "SuperCalculateTiles": "🐧 Calculate Upscale Tiles",
}

# A print statement to show when the nodes are loaded
print("\033[34m[SuperComfyNodes]\033[0m: Loaded nodes.")

# This is required by ComfyUI
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

