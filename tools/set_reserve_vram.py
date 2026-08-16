import ctypes
import gc

import comfy.model_management
from comfy_api.latest import io

try:
    import comfy_aimdo.control as aimdo_control
except ImportError:
    aimdo_control = None

# Save starting reserves, so 0.0 can put things back the way ComfyUI started.
_DEFAULTS = {}


def _dynamic_headroom():
    """The native DynamicVRAM headroom global, or None when it isn't active."""
    lib = getattr(aimdo_control, "lib", None)
    if lib is None:
        return None
    try:
        return ctypes.c_int64.in_dll(lib, "simple_vram_headroom")
    except ValueError:
        return None


def _snapshot_defaults():
    """Record the untouched reserves. Idempotent, and runs before any write."""
    if "core" not in _DEFAULTS:
        _DEFAULTS["core"] = comfy.model_management.EXTRA_RESERVED_VRAM
    if "aimdo" not in _DEFAULTS:
        headroom = _dynamic_headroom()
        if headroom is not None:
            _DEFAULTS["aimdo"] = headroom.value


_snapshot_defaults()


def _resolve(reserved_gb):
    """Map the widget value onto (core_bytes, aimdo_bytes).

    0.0 restores the boot-time defaults rather than reserving nothing, since
    that is the out-of-box experience people actually want back. Anything
    negative is the real zero, so the scale stays continuous.
    """
    if reserved_gb < 0:
        return 0, 0
    if reserved_gb == 0:
        return _DEFAULTS["core"], _DEFAULTS.get("aimdo")
    reserved_bytes = int(reserved_gb * 1024 * 1024 * 1024)
    return reserved_bytes, reserved_bytes


def _set_dynamic_headroom(reserved_bytes):
    """Apply the reserve to DynamicVRAM's native allocator.

    Core only wires --reserve-vram into aimdo at startup (main.py), so setting
    comfy.model_management.EXTRA_RESERVED_VRAM alone has no effect on the
    dynamic loading path. The native global is read per-allocation, so it can
    be updated mid-workflow. Returns False when DynamicVRAM isn't active.
    """
    lib = getattr(aimdo_control, "lib", None)
    if lib is None or reserved_bytes is None:
        return False

    setter = getattr(lib, "set_simple_vram_headroom", None)
    if setter is None:
        return False

    setter.argtypes = [ctypes.c_int64]
    setter.restype = None
    setter(int(reserved_bytes))
    return True


def _evict_vram():
    """Evict every byte of resident VRAM.

    A raised reserve only governs future allocations, so weights already
    resident have to be pushed out for it to take effect immediately. This
    goes through partially_unload, which for DynamicVRAM models drops vbar
    pages while leaving the RAM pins and disk backing alone, so the model
    streams back on demand rather than reloading from disk. Deliberately not
    unload_all_models(), which detaches models and takes the RAM pins with
    them; partially_unload_ram() is never called for the same reason.
    """
    mm = comfy.model_management
    devices = set(mm.get_all_torch_devices())
    freed_total = 0

    for loaded in list(mm.current_loaded_models):
        if loaded.device not in devices or loaded.is_dead():
            continue
        model = loaded.model
        if model is None:
            continue
        freed_total += model.partially_unload(model.offload_device, 1e30)

    # Evicted weights can leave unreachable tensors holding caching-allocator
    # blocks, which empty_cache can only hand back once they are collected.
    gc.collect()
    mm.soft_empty_cache()  # synchronizes, empties the torch cache, ipc_collect
    return freed_total


class SetReserveVRAM(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SetReserveVRAM",
            display_name="🐧 Set Reserve VRAM",
            category="SuperNodes/Tools",
            description="Sets --reserve-vram dynamically anywhere in a workflow. Use watchers to trigger re-runs when models are loaded.",
            is_output_node=True,
            inputs=[
                io.Custom("*").Input("any", optional=True),
                io.Float.Input(
                    "reserved_gb",
                    default=0.0,
                    min=-1.0,
                    max=128.0,
                    step=0.1,
                    tooltip="Set to 0 to restore values at boot. Set a negative number for true zero reserve.",
                ),
                io.Boolean.Input(
                    "clean_memory",
                    default=False,
                    tooltip="Frees resident VRAM.",
                ),
                io.Autogrow.Input(
                    "watchers",
                    optional=True,
                    template=io.Autogrow.TemplateNames(
                        input=io.Custom("*").Input(
                            "watch",
                        ),
                        names=[f"watch_{i}" for i in range(1, 17)],
                        min=0,
                    ),
                ),
            ],
            outputs=[
                io.Custom("*").Output(display_name="any"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, clean_memory=False, **kwargs):
        # Freeing VRAM is a side effect a cache hit would silently skip, so opt
        # out of caching when it's on. Left cacheable otherwise, or a passthrough
        # would re-run everything downstream on every queue.
        return float("nan") if clean_memory else None

    @classmethod
    def execute(cls, any=None, reserved_gb=0.0, clean_memory=False,
                watchers: io.Autogrow.Type = None) -> io.NodeOutput:
        # watchers are never read. They exist only to pull whatever they are
        # wired to into this node's cache signature, so a change upstream of
        # the sampler re-runs this node rather than skipping it as a cache hit
        # and leaving the previous run's reserve in force.
        _snapshot_defaults()
        core_bytes, aimdo_bytes = _resolve(reserved_gb)

        comfy.model_management.EXTRA_RESERVED_VRAM = core_bytes
        _set_dynamic_headroom(aimdo_bytes)

        if clean_memory:
            _evict_vram()
        return io.NodeOutput(any)


NODE = [SetReserveVRAM]
