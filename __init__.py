import importlib
from pathlib import Path
import traceback

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

this_dir = Path(__file__).parent
for folder in this_dir.iterdir():
    if folder.is_dir() and not folder.name.startswith((".", "__")):
        for file in folder.glob("*.py"):
            if file.name == "utils.py" or file.name.startswith("__"):
                continue
            module_name = f".{folder.name}.{file.stem}"
            try:
                module = importlib.import_module(module_name, package=__name__)
                if hasattr(module, "NODE_CLASS_MAPPINGS"):
                    NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
                if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
                    NODE_DISPLAY_NAME_MAPPINGS.update(
                        module.NODE_DISPLAY_NAME_MAPPINGS
                    )
            except Exception:
                print(f"\n[SuperNodes] Failed to load node: {module_name}")
                traceback.print_exc()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
