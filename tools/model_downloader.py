import os
import re
import time
import urllib.parse

import comfy.model_management
import comfy.utils
from comfy_api.latest import io
import folder_paths
import requests


class SuperModelDownloader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        # Dynamically load all valid ComfyUI model directories using the correct attribute
        valid_folders = list(folder_paths.folder_names_and_paths.keys())

        return io.Schema(
            node_id="SuperModelDownloader",
            display_name="🐧 Model Downloader",
            category="SuperNodes/Tools",
            inputs=[
                io.String.Input(
                    "url",
                    default="",
                    tooltip="The URL to download the model from.",
                ),
                io.Combo.Input(
                    "destination",
                    options=valid_folders,
                    tooltip="Select the destination folder (e.g., checkpoints, loras).",
                ),
                io.String.Input(
                    "alias",
                    default="",
                    optional=True,
                    tooltip="Optional custom output name without extension. The original file extension is always preserved.",
                ),
                io.String.Input(
                    "civitai_api_key",
                    default="",
                    optional=True,
                    tooltip="Optional if CIVITAI_API_KEY is in your environment vars.",
                ),
                io.String.Input(
                    "huggingface_api_key",
                    default="",
                    optional=True,
                    tooltip="Optional if HUGGINGFACE_API_KEY is in your environment vars.",
                ),
            ],
            outputs=[
                io.Custom("*").Output(display_name="model_name"),
            ],
        )

    @classmethod
    def execute(
        cls,
        url,
        destination,
        alias="",
        civitai_api_key="",
        huggingface_api_key="",
    ) -> io.NodeOutput:
        if not url.strip():
            raise ValueError("No URL provided.")

        valid_extensions = [".safetensors", ".pth", ".onnx"]
        alias_input = alias.strip()
        alias_raw_name = ""
        alias_base = ""
        if alias_input:
            alias_raw_name = os.path.basename(alias_input)
            alias_base = os.path.splitext(alias_raw_name)[0]
            if not alias_base:
                raise ValueError(
                    "Alias is invalid. Please provide a non-empty filename."
                )

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
            is_offline = isinstance(
                e,
                (
                    requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout,
                ),
            )
            if is_offline:
                if not alias_base:
                    raise RuntimeError(
                        "Offline: cannot reach model URL and no alias was provided to assume an existing local file."
                    )

                dest_dirs = folder_paths.get_folder_paths(destination)
                if not dest_dirs:
                    raise ValueError(
                        f"Invalid destination folder: {destination}"
                    )

                dest_dir = dest_dirs[0]
                os.makedirs(dest_dir, exist_ok=True)

                alias_ext = os.path.splitext(alias_raw_name)[1].lower()
                if alias_ext in valid_extensions:
                    assumed_filename = f"{alias_base}{alias_ext}"
                else:
                    candidate_filenames = [
                        f"{alias_base}{ext}"
                        for ext in valid_extensions
                        if os.path.exists(
                            os.path.join(dest_dir, f"{alias_base}{ext}")
                        )
                    ]
                    if len(candidate_filenames) == 1:
                        assumed_filename = candidate_filenames[0]
                    elif len(candidate_filenames) > 1:
                        raise ValueError(
                            f"Offline and alias '{alias_input}' matches multiple local files in '{destination}'. Please include the extension in alias to disambiguate."
                        )
                    else:
                        assumed_filename = f"{alias_base}{valid_extensions[0]}"

                print(
                    f"⚠️ Offline detected. Assuming model '{assumed_filename}' is already available locally"
                )
                return io.NodeOutput(assumed_filename)
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
        valid_extensions_set = set(valid_extensions)
        ext = os.path.splitext(filename)[1].lower()
        if ext not in valid_extensions_set:
            raise ValueError(
                f"File type '{ext}' is not supported. Only .safetensors and .pth files are allowed."
            )

        if alias_base:
            filename = f"{alias_base}{ext}"

        # --------------------------------------------------------------------
        # 4. DESTINATION & RESUME LOGIC
        # --------------------------------------------------------------------
        dest_dirs = folder_paths.get_folder_paths(destination)
        if not dest_dirs:
            raise ValueError(f"Invalid destination folder: {destination}")

        dest_dir = dest_dirs[0]
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, filename)

        is_alias = bool(alias_base)
        total_size = int(response.headers.get("content-length", 0))
        response_etag = response.headers.get("etag")
        response_last_modified = response.headers.get("last-modified")

        if is_alias and os.path.exists(dest_path):
            existing_final_size = os.path.getsize(dest_path)
            if total_size > 0 and existing_final_size == total_size:
                print(f"✅ File {filename} already exists. Skipping download")
                return io.NodeOutput(filename)
            raise ValueError(
                f"Alias '{alias_input}' is already taken in '{destination}' as '{filename}'. Please choose a different alias or remove the existing file first."
            )

        active_dest_path = f"{dest_path}.part" if is_alias else dest_path

        existing_size = (
            os.path.getsize(active_dest_path)
            if os.path.exists(active_dest_path)
            else 0
        )

        file_mode = "wb"
        initial_pos = 0

        if existing_size > 0:
            if total_size > 0 and existing_size == total_size:
                if is_alias:
                    os.replace(active_dest_path, dest_path)
                    print(f"✅ File {filename} already exists. Finalized alias")
                else:
                    print(
                        f"✅ File {filename} already exists. Skipping download"
                    )
                return io.NodeOutput(filename)
            elif total_size > 0 and existing_size < total_size:
                if response_etag or response_last_modified:
                    print(f"⚠️ Resuming incomplete download for {filename}")
                    resume_headers = dict(headers)
                    resume_headers["Range"] = f"bytes={existing_size}-"
                    resume_headers["If-Range"] = (
                        response_etag
                        if response_etag
                        else response_last_modified
                    )
                    resume_response = requests.get(
                        url,
                        stream=True,
                        headers=resume_headers,
                        allow_redirects=True,
                    )

                    if resume_response.status_code == 401:
                        raise RuntimeError(
                            "Error 401 Unauthorized. Your API key is invalid or you need to accept the model terms on HuggingFace."
                        )

                    if resume_response.status_code == 206:
                        response = resume_response
                        file_mode = "ab"
                        initial_pos = existing_size
                    elif resume_response.status_code == 200:
                        # If-Range failed because remote file changed; restart from byte 0.
                        print(
                            f"⚠️ Remote file changed for {filename}. Restarting full download"
                        )
                        response = resume_response
                        file_mode = "wb"
                        initial_pos = 0
                        total_size = int(
                            response.headers.get("content-length", 0)
                        )
                    else:
                        resume_response.raise_for_status()
                else:
                    print(
                        f"⚠️ Cannot safely resume {filename} (no ETag/Last-Modified). Restarting full download"
                    )
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

        with open(active_dest_path, file_mode) as f:
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

        if is_alias:
            os.replace(active_dest_path, dest_path)

        print(f"✅ Downloaded {filename}")
        return io.NodeOutput(filename)


NODE = [SuperModelDownloader]
