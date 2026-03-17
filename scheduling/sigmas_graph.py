import io as real_io
import random

from comfy_api.latest import io, ui
import folder_paths
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from PIL import Image
import torch


class SigmasGraph(io.ComfyNode):
    def __init__(self):
        # Based on PreviewImage logic here to save to the temp directory
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + "".join(
            random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5)
        )
        self.compress_level = 1

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SigmasGraph",
            display_name="🐧 Sigmas Graph",
            category="SuperNodes/Scheduling",
            is_output_node=True,
            description="Generates a visual graph of the sigma decay schedule and displays it in the node.",
            inputs=[
                io.Custom("SIGMAS").Input(
                    "sigmas", tooltip="The sigma schedule tensor to visualize."
                ),
            ],
            outputs=[
                io.Image.Output(),
            ],
        )

    @classmethod
    def execute(cls, sigmas) -> io.NodeOutput:
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
        buf = real_io.BytesIO()
        plt.savefig(buf, format="png", dpi=100)
        buf.seek(0)
        plt.close()

        # 4. Convert to Tensor (Batch, H, W, C)
        image = Image.open(buf).convert("RGB")
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).unsqueeze(0)

        # 5. Call internal save method to display in UI
        return io.NodeOutput(
            image_tensor, ui=ui.PreviewImage(image_tensor, cls=cls)
        )


NODE = [SigmasGraph]
