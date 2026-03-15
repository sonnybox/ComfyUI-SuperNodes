import torch


class ImageMaskCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (
                    "IMAGE",
                    {"tooltip": "The source image to be cropped."},
                ),
                "mask": (
                    "MASK",
                    {
                        "tooltip": "The binary mask defining the region of interest to crop."
                    },
                ),
                "padding": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4096,
                        "step": 1,
                        "tooltip": "Amount of padding (in pixels) to add around the mask bounding box.",
                    },
                ),
                "multiple_of": (
                    "INT",
                    {
                        "default": 16,
                        "min": 1,
                        "max": 512,
                        "step": 1,
                        "tooltip": "Ensure the crop dimensions are a multiple of this value (critical for UNet based processing).",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "CROP_INFO")
    RETURN_NAMES = ("cropped_image", "cropped_mask", "uncrop_info")
    OUTPUT_TOOLTIPS = (
        "The cropped image region.",
        "The cropped mask region.",
        "Metadata containing coordinates and original size, required for restoration.",
    )
    FUNCTION = "crop"
    CATEGORY = "SuperNodes/Image"
    DESCRIPTION = "Crops an image based on a mask's bounding box, with optional padding and dimension constraints."

    def crop(self, image, mask, padding, multiple_of):
        # Handle empty mask
        if mask.max() == 0:
            empty_info = {
                "x": 0,
                "y": 0,
                "w": image.shape[2],
                "h": image.shape[1],
                "original_size": (image.shape[1], image.shape[2]),
                "mask_patch": None,
            }
            return (image, mask, empty_info)

        # 1. Binarize Mask (Round to nearest 0 or 1 based on 0.5 threshold)
        mask_binary = (mask > 0.5).float()

        # 2. Calculate Bounding Box
        mask_flat = (
            torch.max(mask_binary, dim=0).values
            if mask.dim() > 2
            else mask_binary
        )
        non_zero = torch.nonzero(mask_flat)

        if non_zero.numel() == 0:
            min_y, min_x = 0, 0
            max_y, max_x = image.shape[1], image.shape[2]
        else:
            min_y = torch.min(non_zero[:, 0]).item()
            max_y = torch.max(non_zero[:, 0]).item() + 1
            min_x = torch.min(non_zero[:, 1]).item()
            max_x = torch.max(non_zero[:, 1]).item() + 1

        # 3. Apply Padding
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(image.shape[2], max_x + padding)
        max_y = min(image.shape[1], max_y + padding)

        # 4. Apply 'multiple_of' constraint
        width = max_x - min_x
        height = max_y - min_y

        if width % multiple_of != 0:
            target_width = ((width // multiple_of) + 1) * multiple_of
            diff = target_width - width
            pad_l = diff // 2
            pad_r = diff - pad_l

            if min_x - pad_l < 0:
                min_x = 0
                max_x = min(image.shape[2], min_x + target_width)
            elif max_x + pad_r > image.shape[2]:
                max_x = image.shape[2]
                min_x = max(0, max_x - target_width)
            else:
                min_x -= pad_l
                max_x += pad_r

        if height % multiple_of != 0:
            target_height = ((height // multiple_of) + 1) * multiple_of
            diff = target_height - height
            pad_t = diff // 2
            pad_b = diff - pad_t

            if min_y - pad_t < 0:
                min_y = 0
                max_y = min(image.shape[1], min_y + target_height)
            elif max_y + pad_b > image.shape[1]:
                max_y = image.shape[1]
                min_y = max(0, max_y - target_height)
            else:
                min_y -= pad_t
                max_y += pad_b

        crop_x, crop_y = min_x, min_y
        crop_w = max_x - min_x
        crop_h = max_y - min_y

        # 5. Crop Image and Mask
        cropped_image = image[
            :, crop_y : crop_y + crop_h, crop_x : crop_x + crop_w, :
        ]
        cropped_mask = mask[
            :, crop_y : crop_y + crop_h, crop_x : crop_x + crop_w
        ]

        # 6. Prepare Uncrop Info
        uncrop_info = {
            "x": crop_x,
            "y": crop_y,
            "w": crop_w,
            "h": crop_h,
            "original_size": (image.shape[1], image.shape[2]),  # H, W
            "mask_patch": cropped_mask,  # Store original mask crop for restoration
        }

        return (cropped_image, cropped_mask, uncrop_info)


NODE_CLASS_MAPPINGS = {"ImageMaskCrop": ImageMaskCrop}

NODE_DISPLAY_NAME_MAPPINGS = {"ImageMaskCrop": "🐧 Crop Image using Mask"}
