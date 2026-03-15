import os
import re
import time
import urllib.parse

import comfy.model_management
import comfy.utils
import folder_paths
import requests


class SuperModelDownloader:
    @classmethod
    def INPUT_TYPES(cls):
        # Dynamically load all valid ComfyUI model directories using the correct attribute
        valid_folders = list(folder_paths.folder_names_and_paths.keys())

        return {
            "required": {
                "url": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "The URL to download the model from.",
                    },
                ),
                "destination": (
                    valid_folders,
                    {
                        "tooltip": "Select the destination folder (e.g., checkpoints, loras)."
                    },
                ),
            },
            "optional": {
                "civitai_api_key": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Optional if CIVITAI_API_KEY is in your environment vars.",
                    },
                ),
                "huggingface_api_key": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Optional if HUGGINGFACE_API_KEY is in your environment vars.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("model_name",)
    FUNCTION = "download_model"
    CATEGORY = "SuperNodes/Tools"

    def download_model(
        self, url, destination, civitai_api_key="", huggingface_api_key=""
    ):
        if not url.strip():
            raise ValueError("No URL provided.")

        headers = {}

        # --------------------------------------------------------------------
        # 1. API KEY & URL PROCESSING
        # --------------------------------------------------------------------
        if "civitai.com" in url:
            match = re.search(r"models/(\d+)", url)
            if match:
                model_id = match.group(1)
                api_key = civitai_api_key or os.environ.get("CIVITAI_API_KEY")
                if not api_key:
                    raise ValueError(
                        "Cannot download from Civitai: API key not found."
                    )
                url = f"https://civitai.com/api/download/models/{model_id}?token={api_key}"

        if "huggingface.co" in url:
            api_key = huggingface_api_key or os.environ.get(
                "HUGGINGFACE_API_KEY"
            )
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

        # --------------------------------------------------------------------
        # 2. INITIATE REQUEST & HANDLE GATED REPOS
        # --------------------------------------------------------------------
        try:
            response = requests.get(
                url, stream=True, headers=headers, allow_redirects=True
            )

            if response.status_code == 401:
                raise RuntimeError(
                    "Error 401 Unauthorized. Your API key is invalid or you need to accept the model terms on HuggingFace."
                )

            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"{e}")

        # Extract filename from Content-Disposition header, fallback to URL parsing
        cd = response.headers.get("content-disposition")
        if cd:
            filenames = re.findall('filename="?([^"]+)"?', cd)
            filename = (
                filenames[0] if filenames else "downloaded_model.safetensors"
            )
        else:
            parsed_url = urllib.parse.urlparse(response.url)
            filename = os.path.basename(parsed_url.path)
            if not filename:
                filename = "downloaded_model.safetensors"

        # --------------------------------------------------------------------
        # 3. FILE EXTENSION VALIDATION
        # --------------------------------------------------------------------
        valid_extensions = {".safetensors", ".pth"}
        ext = os.path.splitext(filename)[1].lower()
        if ext not in valid_extensions:
            raise ValueError(
                f"File type '{ext}' is not supported. Only .safetensors and .pth files are allowed."
            )

        # --------------------------------------------------------------------
        # 4. DESTINATION & RESUME LOGIC
        # --------------------------------------------------------------------
        dest_dirs = folder_paths.get_folder_paths(destination)
        if not dest_dirs:
            raise ValueError(f"Invalid destination folder: {destination}")

        dest_dir = dest_dirs[0]
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, filename)

        total_size = int(response.headers.get("content-length", 0))
        existing_size = (
            os.path.getsize(dest_path) if os.path.exists(dest_path) else 0
        )

        file_mode = "wb"
        initial_pos = 0

        if existing_size > 0:
            if total_size > 0 and existing_size == total_size:
                print(f"✅ File {filename} already exists. Skipping download")
                return (filename,)
            elif total_size > 0 and existing_size < total_size:
                print(f"⚠️ Resuming incomplete download for {filename}")
                headers["Range"] = f"bytes={existing_size}-"
                response = requests.get(
                    url, stream=True, headers=headers, allow_redirects=True
                )
                response.raise_for_status()
                file_mode = "ab"
                initial_pos = existing_size
            else:
                print(
                    f"⚠️ Existing file size mismatch. Redownloading {filename}"
                )

        # --------------------------------------------------------------------
        # 5. DOWNLOAD LOOP WITH PROGRESS & INTERRUPT TRACKING
        # --------------------------------------------------------------------
        pbar = comfy.utils.ProgressBar(100)
        downloaded = initial_pos
        start_time = time.time()
        last_print_time = start_time

        print(f"⬇️ Downloading: {filename} to {destination}")

        with open(dest_path, file_mode) as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                comfy.model_management.throw_exception_if_processing_interrupted()

                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total_size > 0:
                        percent = int((downloaded / total_size) * 100)
                        pbar.update_absolute(percent, 100)

                    current_time = time.time()
                    if current_time - last_print_time >= 2.0:
                        elapsed = current_time - start_time
                        speed_mb = (
                            (downloaded - initial_pos) / elapsed / (1024 * 1024)
                        )
                        dl_mb = downloaded / (1024 * 1024)
                        tot_mb = total_size / (1024 * 1024) if total_size else 0
                        print(
                            f"{dl_mb:.2f}/{tot_mb:.2f} MB ({speed_mb:.2f} MB/s)"
                        )
                        last_print_time = current_time

        print(f"✅ Downloaded {filename}")
        return (filename,)


NODE_CLASS_MAPPINGS = {"SuperModelDownloader": SuperModelDownloader}

NODE_DISPLAY_NAME_MAPPINGS = {"SuperModelDownloader": "🐧 Model Downloader"}
