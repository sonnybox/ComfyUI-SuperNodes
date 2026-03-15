import io
import os
import random

import folder_paths
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from PIL import Image
import torch


class SigmasGraph:
    def __init__(self):
        # Based on PreviewImage logic here to save to the temp directory
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + "".join(
            random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5)
        )
        self.compress_level = 1

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sigmas": (
                    "SIGMAS",
                    {"tooltip": "The sigma schedule tensor to visualize."},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_NODE = True
    FUNCTION = "plot_sigmas"

    CATEGORY = "SuperNodes/Scheduling"
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
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.plot(
            s_data,
            marker="o",
            linestyle="-",
            linewidth=2.5,
            markersize=4,
            color="#1f77b4",
        )

        ax.set_title("")
        ax.set_xlabel("Step", fontsize=18)
        ax.set_ylabel("Sigma", fontsize=18)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
        ax.tick_params(axis="both", which="major", labelsize=14)
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        plt.tight_layout()

        # 3. Save Plot to Buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100)
        buf.seek(0)
        plt.close()

        # 4. Convert to Tensor (Batch, H, W, C)
        image = Image.open(buf).convert("RGB")
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).unsqueeze(0)

        # 5. Call internal save method to display in UI
        ui_output = self.save_images(
            image_tensor, filename_prefix="SigmasGraph"
        )

        # Return both the UI dictionary and the image tensor
        return {"ui": ui_output["ui"], "result": (image_tensor,)}

    def save_images(self, images, filename_prefix="ComfyUI"):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = (
            folder_paths.get_save_image_path(
                filename_prefix,
                self.output_dir,
                images[0].shape[1],
                images[0].shape[0],
            )
        )
        results = list()
        for batch_number, image in enumerate(images):
            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            filename_with_batch_num = filename.replace(
                "%batch_num%", str(batch_number)
            )
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            img.save(
                os.path.join(full_output_folder, file),
                pnginfo=None,
                compress_level=self.compress_level,
            )
            results.append(
                {"filename": file, "subfolder": subfolder, "type": self.type}
            )
            counter += 1

        return {"ui": {"images": results}}


NODE_CLASS_MAPPINGS = {"SigmasGraph": SigmasGraph}

NODE_DISPLAY_NAME_MAPPINGS = {"SigmasGraph": "🐧 Sigmas Graph"}
