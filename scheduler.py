import json
import math
import os
import random
import diffusers
import torch
import folder_paths # type: ignore
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image


class SigmaSmoother:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sigmas": (
                    "SIGMAS",
                    {
                        "tooltip": "The input sigmas tensor, typically ending in 0.0."
                    },
                ),
                "smooth_steps": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 100,
                        "tooltip": "Number of additional smoothed steps to insert between the last non-zero sigma and 0.0.",
                    },
                ),
                "interpolation_type": (
                    ["linear", "decay"],
                    {
                        "default": "linear",
                        "tooltip": "Method to calculate the intermediate sigma values.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("SIGMAS",)
    FUNCTION = "smooth_sigmas"
    CATEGORY = "SuperNodes"
    DESCRIPTION = "Inserts smoothed interpolation steps at the end of a sigma schedule before the final zero, useful for refining the final denoising steps."

    def smooth_sigmas(self, sigmas, smooth_steps, interpolation_type):
        # Ensure we are working with a float tensor
        if sigmas.dtype != torch.float32 and sigmas.dtype != torch.float64:
            sigmas = sigmas.float()

        # Check if the last element is zero and remove it for calculation
        if sigmas[-1] == 0.0:
            active_sigmas = sigmas[:-1]
        else:
            active_sigmas = sigmas

        if len(active_sigmas) == 0:
            # Edge case: empty or only zero input
            return (sigmas,)

        last_sigma = active_sigmas[-1].item()
        new_steps = []

        if interpolation_type == "linear":
            # Linear interpolation from last_sigma to 0
            # Total intervals = smooth_steps + 1 (the final drop to 0 is the +1)
            step_size = last_sigma / (smooth_steps + 1)
            for i in range(1, smooth_steps + 1):
                new_val = last_sigma - (step_size * i)
                new_steps.append(new_val)

        elif interpolation_type == "decay":
            # Decay interpolation: previous / 2, then previous / 3, etc.
            # Example: 3 -> 1.5 -> 0.5 -> 0.125
            current_val = last_sigma
            for i in range(1, smooth_steps + 1):
                divisor = i + 1
                current_val = current_val / divisor
                new_steps.append(current_val)

        # Convert new steps to tensor ensuring matching device and type
        new_sigmas_tensor = torch.tensor(
            new_steps, dtype=sigmas.dtype, device=sigmas.device
        )

        # Reconstruct: Old sigmas (minus zero) + New Steps + Zero
        parts = [active_sigmas, new_sigmas_tensor]

        # Always append 0.0 at the end as per ComfyUI sigma standards
        zero_tensor = torch.tensor(
            [0.0], dtype=sigmas.dtype, device=sigmas.device
        )
        parts.append(zero_tensor)

        result_sigmas = torch.cat(parts)

        return (result_sigmas,)


class SigmasRescale:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sigmas": (
                    "SIGMAS",
                    {"tooltip": "The input sigma schedule to be rescaled."},
                ),
                "max": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 10000.0,
                        "step": 0.01,
                        "tooltip": "The new maximum value (start of the schedule).",
                    },
                ),
                "min": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1000.0,
                        "step": 0.001,
                        "tooltip": "The new minimum value (end of the schedule).",
                    },
                ),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    OUTPUT_TOOLTIPS = ("The rescaled sigma schedule.",)
    FUNCTION = "rescale"

    CATEGORY = "SuperNodes"
    DESCRIPTION = "Rescales a sigma schedule to a new maximum and minimum range while preserving the exact curve of the original schedule."

    def rescale(self, sigmas, max, min):
        # Avoid modifying the original tensor
        s = sigmas.clone()

        # Get the current range of the input sigmas
        # Sigmas usually go from High to Low, so index 0 is max, index -1 is min
        current_max = s[0]
        current_min = s[-1]

        # Handle edge case where max equals min to avoid division by zero
        if current_max == current_min:
            # If the schedule is flat, return a flat schedule at the new max
            return (torch.full_like(s, max),)

        # Normalize the curve to 0.0 - 1.0
        # Formula: (value - min) / (max - min)
        normalized_curve = (s - current_min) / (current_max - current_min)

        # Scale to the new range
        # Formula: normalized * (new_max - new_min) + new_min
        new_sigmas = normalized_curve * (max - min) + min

        return (new_sigmas,)


class LoadDiffusersScheduler:
    @classmethod
    def INPUT_TYPES(s):
        scheduler_type = "scheduler"
        default_scheduler_path = os.path.join(folder_paths.models_dir, scheduler_type)
        folder_paths.add_model_folder_path(scheduler_type, default_scheduler_path)
        paths, current_exts = folder_paths.folder_names_and_paths[scheduler_type]
        if ".json" not in current_exts:
            folder_paths.folder_names_and_paths[scheduler_type] = (paths, current_exts | {".json"})

        return {
            "required": {
                # Now we use the standard get_filename_list with our new key
                "scheduler_name": (folder_paths.get_filename_list(scheduler_type), {"tooltip": "Select a diffusers scheduler JSON config file from 'models/scheduler'."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "The number of steps for the schedule."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The amount of denoising to apply."}),
            },
            "optional": {
                "width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8, "tooltip": "Used for 'dynamic shifting' calculations."}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8, "tooltip": "Used for 'dynamic shifting' calculations."}),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("SIGMAS",)
    CATEGORY = "advanced/loaders"
    FUNCTION = "load_scheduler"
    DESCRIPTION = "Loads a scheduler from a Diffusers JSON config and generates a SIGMAS schedule."

    def load_scheduler(self, scheduler_name, steps, denoise, width=1024, height=1024):
        if diffusers is None:
            raise ImportError("The 'diffusers' library is required to use this node. Please install it via pip.")

        scheduler_type = "scheduler"

        config_path = folder_paths.get_full_path(scheduler_type, scheduler_name)
        
        if config_path is None:
            raise FileNotFoundError(f"Scheduler config '{scheduler_name}' not found. Please ensure it is in 'models/scheduler' or defined in extra_model_paths.yaml.")

        # Load JSON config
        with open(config_path, 'r') as f:
            config = json.load(f)

        class_name = config.get("_class_name", None)
        if not class_name:
            raise ValueError("Invalid scheduler JSON: missing '_class_name'.")

        scheduler_cls = getattr(diffusers, class_name, None)
        if scheduler_cls is None:
            raise ImportError(f"Scheduler class '{class_name}' not found in diffusers library.")

        # --- Dynamic Shifting Logic ---
        if config.get("use_dynamic_shifting", False):
            base_seq_len = config.get("base_image_seq_len", 256)
            max_seq_len = config.get("max_image_seq_len", 4096)
            base_shift = config.get("base_shift", 0.5)
            max_shift = config.get("max_shift", 1.15)
            
            image_seq_len = (width // 16) * (height // 16)
            
            m = image_seq_len
            m1 = base_seq_len ** 2
            m2 = max_seq_len ** 2
            
            # Clamp mu between 0 and 1
            if m2 > m1:
                mu = (m - m1) / (m2 - m1)
            else:
                mu = 0
            
            mu = max(0.0, min(1.0, mu))
            shift = math.exp(math.log(base_shift) + mu * (math.log(max_shift) - math.log(base_shift)))
            
            config["shift"] = shift
            
            # Disable dynamic shifting in the config since we just manually applied it.
            config["use_dynamic_shifting"] = False

        # Instantiate Scheduler
        try:
            scheduler = scheduler_cls.from_config(config)
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate {class_name}: {e}")

        # --- Calculate Sigmas ---
        total_steps = steps
        if denoise < 1.0 and denoise > 0.0:
            total_steps = int(steps / denoise)
        if denoise == 0.0:
            total_steps = steps 
            
        scheduler.set_timesteps(total_steps)
        
        if hasattr(scheduler, "sigmas"):
            sigmas = scheduler.sigmas
        else:
            raise AttributeError(f"Scheduler {class_name} does not expose 'sigmas'.")

        # Truncate if denoise was used
        if denoise < 1.0 and denoise > 0.0:
            sigmas = sigmas[-(steps + 1):]
            
        # Ensure it's a CPU tensor for ComfyUI
        sigmas = sigmas.clone().detach().cpu()
        
        return (sigmas,)


class SigmasGraph:
    def __init__(self):
        # Based on PreviewImage logic here to save to the temp directory
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        self.compress_level = 1

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"tooltip": "The sigma schedule tensor to visualize."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_NODE = True
    FUNCTION = "plot_sigmas"

    CATEGORY = "SuperNodes"
    DESCRIPTION = "Generates a visual graph of the sigma decay schedule and displays it in the node."

    def plot_sigmas(self, sigmas):
        # 1. Prepare Data
        if isinstance(sigmas, torch.Tensor):
            s_data = sigmas.detach().cpu().numpy().flatten()
        elif isinstance(sigmas, list):
            s_data = np.array(sigmas).flatten()
        else:
            s_data = np.array(sigmas).flatten()

        # 2. Generate Plot
        plt.figure(figsize=(8, 7))
        plt.plot(s_data, marker='o', linestyle='-', markersize=4, color='#1f77b4')
        
        steps = len(s_data) - 1 if len(s_data) > 0 else 0
        plt.title(f"Sigma Schedule ({steps} steps)")
        plt.xlabel("Step")
        plt.ylabel("Value")
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()

        # 3. Save Plot to Buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        plt.close()

        # 4. Convert to Tensor (Batch, H, W, C)
        image = Image.open(buf).convert("RGB")
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).unsqueeze(0)

        # 5. Call internal save method to display in UI
        ui_output = self.save_images(image_tensor, filename_prefix="SigmasGraph")
        
        # Return both the UI dictionary and the image tensor
        return {"ui": ui_output["ui"], "result": (image_tensor,)}

    def save_images(self, images, filename_prefix="ComfyUI"):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file), pnginfo=None, compress_level=self.compress_level)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return { "ui": { "images": results } }