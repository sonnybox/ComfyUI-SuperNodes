import importlib
from pathlib import Path
import traceback

from comfy_api.latest import ComfyExtension, io

NODES = []
__all__ = []


this_dir = Path(__file__).parent
for folder in this_dir.iterdir():
    if folder.is_dir() and not folder.name.startswith((".", "__")):
        for file in folder.glob("*.py"):
            if file.name == "utils.py" or file.name.startswith("__"):
                continue
            module_name = f".{folder.name}.{file.stem}"
            try:
                module = importlib.import_module(module_name, package=__name__)
                if hasattr(module, "NODE"):
                    for node_class in module.NODE:
                        node_class.define_schema()
                        NODES.append(node_class)
            except Exception:
                print(
                    f"\n[SuperNodes] Failed to load node module: {module_name}"
                )
                traceback.print_exc()


class SuperNodesExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return NODES


async def comfy_entrypoint() -> ComfyExtension:
    return SuperNodesExtension()


__all__.append("comfy_entrypoint")
