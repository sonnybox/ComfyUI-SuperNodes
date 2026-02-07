import torch


def _clamp01(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(0.0, 1.0)


def _srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    # x in [0..1]
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    # x in [0..1] ideally, but allow out-of-range before clamp
    return torch.where(
        x <= 0.0031308,
        x * 12.92,
        1.055 * torch.pow(torch.clamp_min(x, 0.0), 1.0 / 2.4) - 0.055,
    )


def _apply_3x3(img: torch.Tensor, mat: torch.Tensor) -> torch.Tensor:
    """
    img: [...,3]
    mat: [3,3]
    returns [...,3]
    """
    return torch.einsum("...c,dc->...d", img, mat)


def _kelvin_to_xy_approx(k: float) -> tuple[float, float]:
    """
    Practical approximation for CCT (Kelvin) -> CIE xy chromaticity.
    Good enough for a WB slider; not "scientific-grade", but stable and common in tooling.

    Valid-ish range: 1650K..25000K (we clamp to that).
    """
    k = float(max(1650.0, min(25000.0, k)))
    t = k

    # x approximation
    if t <= 4000.0:
        x = (
            (-0.2661239e9 / (t**3))
            - (0.2343580e6 / (t**2))
            + (0.8776956e3 / t)
            + 0.179910
        )
    else:
        x = (
            (-3.0258469e9 / (t**3))
            + (2.1070379e6 / (t**2))
            + (0.2226347e3 / t)
            + 0.240390
        )

    # y approximation (piecewise in x)
    if t <= 2222.0:
        y = (
            -1.1063814 * (x**3)
            - 1.34811020 * (x**2)
            + 2.18555832 * x
            - 0.20219683
        )
    elif t <= 4000.0:
        y = (
            -0.9549476 * (x**3)
            - 1.37418593 * (x**2)
            + 2.09137015 * x
            - 0.16748867
        )
    else:
        y = (
            3.0817580 * (x**3)
            - 5.87338670 * (x**2)
            + 3.75112997 * x
            - 0.37001483
        )

    # clamp to sane range
    x = float(max(1e-6, min(0.999999, x)))
    y = float(max(1e-6, min(0.999999, y)))
    return x, y


def _xy_to_uv(x: float, y: float) -> tuple[float, float]:
    # CIE 1960 UCS u,v (not u',v')
    denom = (-2.0 * x) + (12.0 * y) + 3.0
    if abs(denom) < 1e-8:
        return 0.0, 0.0
    u = (4.0 * x) / denom
    v = (6.0 * y) / denom
    return u, v


def _uv_to_xy(u: float, v: float) -> tuple[float, float]:
    # inverse of CIE 1960 u,v
    denom = (2.0 * u) - (8.0 * v) + 4.0
    if abs(denom) < 1e-8:
        return 0.3127, 0.3290  # fallback D65-ish
    x = (3.0 * u) / denom
    y = (2.0 * v) / denom
    x = float(max(1e-6, min(0.999999, x)))
    y = float(max(1e-6, min(0.999999, y)))
    return x, y


def _xy_to_XYZ_white(x: float, y: float) -> tuple[float, float, float]:
    X = x / y
    Y = 1.0
    Z = (1.0 - x - y) / y
    return X, Y, Z


def _bradford_adaptation_matrix(
    src_white_XYZ: torch.Tensor,  # [3]
    dst_white_XYZ: torch.Tensor,  # [3]
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Returns A: [3,3] Bradford chromatic adaptation matrix for XYZ values.
    """
    M = torch.tensor(
        [
            [0.8951, 0.2664, -0.1614],
            [-0.7502, 1.7135, 0.0367],
            [0.0389, -0.0685, 1.0296],
        ],
        device=device,
        dtype=dtype,
    )
    Minv = torch.tensor(
        [
            [0.9869929, -0.1470543, 0.1599627],
            [0.4323053, 0.5183603, 0.0492912],
            [-0.0085287, 0.0400428, 0.9684867],
        ],
        device=device,
        dtype=dtype,
    )

    src_LMS = M @ src_white_XYZ
    dst_LMS = M @ dst_white_XYZ

    # Avoid divide-by-zero
    scale = dst_LMS / torch.clamp_min(src_LMS, 1e-8)
    D = torch.diag(scale)

    A = Minv @ D @ M
    return A


# sRGB D65 matrices (linear)
_RGB_TO_XYZ = torch.tensor(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=torch.float32,
)

_XYZ_TO_RGB = torch.tensor(
    [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ],
    dtype=torch.float32,
)


def _levels_normalize(
    img: torch.Tensor,
    low_pct: float = 0.5,
    high_pct: float = 99.5,
    per_channel: bool = False,
):
    """
    Percentile-based levels normalization.

    img: [B,H,W,3] float [0..1]
    low_pct / high_pct: percentiles (0..100)
    per_channel: if True, compute per RGB channel, else luminance-style joint
    """

    B = img.shape[0]
    flat = img.view(B, -1, 3)

    lo = low_pct / 100.0
    hi = high_pct / 100.0

    if per_channel:
        lows = []
        highs = []
        for c in range(3):
            chan = flat[..., c]
            lows.append(torch.quantile(chan, lo, dim=1))
            highs.append(torch.quantile(chan, hi, dim=1))
        low = torch.stack(lows, dim=1)
        high = torch.stack(highs, dim=1)
    else:
        # Use luminance-ish average for bounds
        lum = flat.mean(dim=2)
        low = torch.quantile(lum, lo, dim=1).unsqueeze(1).repeat(1, 3)
        high = torch.quantile(lum, hi, dim=1).unsqueeze(1).repeat(1, 3)

    low = low[:, None, None, :]
    high = high[:, None, None, :]

    out = (img - low) / torch.clamp(high - low, min=1e-6)
    return _clamp01(out)


def _get_samples(latent: dict) -> torch.Tensor:
    x = latent["samples"]
    # Some comfy ops may produce nested tensors; unwrap if needed.
    if getattr(x, "is_nested", False):
        x = x.unbind()[0]
    return x


def _set_samples(latent: dict, samples: torch.Tensor) -> dict:
    out = latent.copy()
    out["samples"] = samples
    return out


def _safe_mean_std(x: torch.Tensor, dims, eps=1e-6):
    mean = x.mean(dim=dims, keepdim=True)
    std = x.std(dim=dims, keepdim=True).clamp_min(eps)
    return mean, std


def _latent_brightness_contrast_gamma(
    samples: torch.Tensor, brightness: float, contrast: float, gamma: float
):
    """
    Latent "brightness/contrast" is not image brightness, but it is useful as:
      - brightness: global scale
      - contrast: scale deviation around per-image mean
      - gamma: mild non-linear shaping on normalized latent (optional)
    """
    b = float(brightness)
    c = float(contrast)
    g = float(gamma)

    x = samples
    # brightness as a simple scale
    x = x * b

    # contrast around per-image mean (per batch item, per channel)
    mean = x.mean(dim=(2, 3), keepdim=True)
    x = (x - mean) * c + mean

    if g != 1.0:
        # apply gamma-ish curve to normalized latents to avoid wrecking scale
        m, s = _safe_mean_std(x, dims=(2, 3))
        z = (x - m) / s
        # signed power curve (preserves sign)
        z = torch.sign(z) * torch.pow(torch.abs(z).clamp_min(1e-6), 1.0 / g)
        x = z * s + m

    return x


def _latent_levels_normalize(
    samples: torch.Tensor, low_pct: float, high_pct: float, per_channel: bool
):
    """
    Percentile trim + normalize, then re-match original mean/std (keeps sampler behavior stable).

    low/high are percentiles like 0.5 and 99.5.
    """
    lo = float(low_pct) / 100.0
    hi = float(high_pct) / 100.0

    x = samples
    B, C, H, W = x.shape
    flat = x.reshape(B, C, -1)

    if per_channel:
        low = torch.quantile(flat, lo, dim=2)  # [B,C]
        high = torch.quantile(flat, hi, dim=2)  # [B,C]
    else:
        # compute bounds using a scalar energy measure per pixel
        energy = flat.abs().mean(dim=1)  # [B,N]
        low_s = torch.quantile(energy, lo, dim=1)  # [B]
        high_s = torch.quantile(energy, hi, dim=1)  # [B]
        low = low_s[:, None].repeat(1, C)
        high = high_s[:, None].repeat(1, C)

    low = low[:, :, None, None]
    high = high[:, :, None, None]
    denom = torch.clamp(high - low, min=1e-6)

    y = (x - low) / denom

    # IMPORTANT: don't clamp like an image. Instead, match original distribution.
    orig_mean, orig_std = _safe_mean_std(x, dims=(2, 3))
    y_mean, y_std = _safe_mean_std(y, dims=(2, 3))
    y = (y - y_mean) / y_std
    y = y * orig_std + orig_mean
    return y


def _latent_saturation(samples: torch.Tensor, saturation: float):
    """
    "Saturation" analog:
    - compute per-pixel mean across channels
    - scale channel deviations from that mean

    For many SD-like latents, this behaves like increasing/decreasing chroma.
    """
    s = float(saturation)
    x = samples
    mu = x.mean(dim=1, keepdim=True)  # [B,1,H,W]
    return (x - mu) * s + mu


def _latent_hue_rotate(samples: torch.Tensor, hue_degrees: float):
    """
    "Hue" analog:
    Rotate the latent vector in a fixed 2D subspace.
    Works best when C>=2. For C=4 (SD/Flux-ish), it's a mild creative control.

    This is NOT true hue; it’s a stable channel-space rotation.
    """
    deg = float(hue_degrees)
    if abs(deg) < 1e-6:
        return samples

    x = samples
    B, C, H, W = x.shape
    if C < 2:
        return x

    theta = deg * 3.141592653589793 / 180.0
    cs = torch.cos(torch.tensor(theta, device=x.device, dtype=x.dtype))
    sn = torch.sin(torch.tensor(theta, device=x.device, dtype=x.dtype))

    # rotate channels 0 and 1; keep others unchanged
    c0 = x[:, 0:1, :, :]
    c1 = x[:, 1:2, :, :]

    r0 = cs * c0 - sn * c1
    r1 = sn * c0 + cs * c1

    out = x.clone()
    out[:, 0:1, :, :] = r0
    out[:, 1:2, :, :] = r1
    return out


class SuperLatentBrightnessContrast:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "brightness": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01},
                ),
                "contrast": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01},
                ),
                "gamma": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.05, "max": 4.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "apply"
    CATEGORY = "SuperNodes/Adjustment"

    def apply(self, latent, brightness=1.0, contrast=1.0, gamma=1.0):
        x = _get_samples(latent)
        y = _latent_brightness_contrast_gamma(x, brightness, contrast, gamma)
        return (_set_samples(latent, y),)


class SuperLatentLevelsNormalize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "low_clip": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.1},
                ),
                "high_clip": (
                    "FLOAT",
                    {"default": 99.5, "min": 90.0, "max": 100.0, "step": 0.1},
                ),
                "per_channel": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "apply"
    CATEGORY = "SuperNodes/Adjustment"

    def apply(self, latent, low_clip=0.5, high_clip=99.5, per_channel=False):
        x = _get_samples(latent)
        y = _latent_levels_normalize(x, low_clip, high_clip, per_channel)
        return (_set_samples(latent, y),)


class SuperLatentChroma:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "saturation": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "apply"
    CATEGORY = "SuperNodes/Adjustment"

    def apply(self, latent, saturation=1.0):
        x = _get_samples(latent)
        y = _latent_saturation(x, saturation)
        return (_set_samples(latent, y),)


class SuperLatentHueRotate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "hue_degrees": (
                    "FLOAT",
                    {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.5},
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "apply"
    CATEGORY = "SuperNodes/Adjustment"

    def apply(self, latent, hue_degrees=0.0):
        x = _get_samples(latent)
        y = _latent_hue_rotate(x, hue_degrees)
        return (_set_samples(latent, y),)


class SuperLatentColorAdjustAllInOne:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "brightness": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01},
                ),
                "contrast": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01},
                ),
                "gamma": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.05, "max": 4.0, "step": 0.01},
                ),
                "low_clip": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1},
                ),
                "high_clip": (
                    "FLOAT",
                    {"default": 100.0, "min": 90.0, "max": 100.0, "step": 0.1},
                ),
                "levels_per_channel": ("BOOLEAN", {"default": False}),
                "saturation": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01},
                ),
                "hue_degrees": (
                    "FLOAT",
                    {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.5},
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "apply"
    CATEGORY = "SuperNodes/Adjustment"

    def apply(
        self,
        latent,
        brightness=1.0,
        contrast=1.0,
        gamma=1.0,
        low_clip=0.0,
        high_clip=100.0,
        levels_per_channel=False,
        saturation=1.0,
        hue_degrees=0.0,
    ):
        x = _get_samples(latent)

        # Optional levels normalize (only if user actually trims)
        if float(low_clip) > 0.0 or float(high_clip) < 100.0:
            x = _latent_levels_normalize(
                x, float(low_clip), float(high_clip), bool(levels_per_channel)
            )

        x = _latent_brightness_contrast_gamma(
            x, float(brightness), float(contrast), float(gamma)
        )
        x = _latent_saturation(x, float(saturation))
        x = _latent_hue_rotate(x, float(hue_degrees))

        return (_set_samples(latent, x),)


class SuperLevelsNormalize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "low_clip": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.1},
                ),
                "high_clip": (
                    "FLOAT",
                    {"default": 99.5, "min": 90.0, "max": 100.0, "step": 0.1},
                ),
                "per_channel": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "SuperNodes/Adjustment"

    def apply(self, image, low_clip=0.5, high_clip=99.5, per_channel=False):
        out = _levels_normalize(
            image, float(low_clip), float(high_clip), bool(per_channel)
        )
        return (out,)


def _apply_white_balance_cat(
    img_srgb: torch.Tensor,
    temperature_k: float,
    tint: float,
) -> torch.Tensor:
    """
    img_srgb: [B,H,W,3] float in [0..1], assumed sRGB-ish
    temperature_k: 1650..25000 typical slider
    tint: -1..1 (green..magenta). Implemented as a small shift in CIE 1960 v.
    """
    device = img_srgb.device
    dtype = img_srgb.dtype

    # 1) sRGB -> linear
    lin = _srgb_to_linear(_clamp01(img_srgb))

    # 2) linear RGB -> XYZ
    rgb2xyz = _RGB_TO_XYZ.to(device=device, dtype=dtype)
    xyz = _apply_3x3(lin, rgb2xyz)

    # 3) build src/dst white points in XYZ
    # Source assumed D65 (sRGB reference white)
    # D65 xy:
    src_x, src_y = 0.3127, 0.3290
    src_X, src_Y, src_Z = _xy_to_XYZ_white(src_x, src_y)

    # Destination from Kelvin + tint offset
    dst_x, dst_y = _kelvin_to_xy_approx(float(temperature_k))

    # Tint: shift in UCS v direction (green<->magenta feel)
    # Scale chosen to be "good enough" and not insane.
    # If you want stronger/weaker, tweak 0.05.
    u, v = _xy_to_uv(dst_x, dst_y)
    v = v + float(tint) * 0.05
    # Clamp to sane bounds
    v = float(max(1e-6, min(0.999999, v)))
    dst_x, dst_y = _uv_to_xy(u, v)
    dst_X, dst_Y, dst_Z = _xy_to_XYZ_white(dst_x, dst_y)

    src_white = torch.tensor([src_X, src_Y, src_Z], device=device, dtype=dtype)
    dst_white = torch.tensor([dst_X, dst_Y, dst_Z], device=device, dtype=dtype)

    # 4) Bradford adaptation in XYZ
    A = _bradford_adaptation_matrix(
        src_white, dst_white, device=device, dtype=dtype
    )
    xyz_adapted = _apply_3x3(xyz, A)

    # 5) XYZ -> linear RGB
    xyz2rgb = _XYZ_TO_RGB.to(device=device, dtype=dtype)
    lin_out = _apply_3x3(xyz_adapted, xyz2rgb)

    # 6) linear -> sRGB
    out = _linear_to_srgb(lin_out)
    return _clamp01(out)


def _apply_brightness_contrast_gamma(
    img: torch.Tensor,
    brightness: float = 1.0,
    contrast: float = 1.0,
    gamma: float = 1.0,
) -> torch.Tensor:
    out = img * float(brightness)
    out = (out - 0.5) * float(contrast) + 0.5
    out = _clamp01(out)

    g = float(gamma)
    if g != 1.0:
        out = torch.pow(out.clamp_min(1e-8), 1.0 / g)

    return _clamp01(out)


def _rgb_to_hsv(rgb: torch.Tensor) -> torch.Tensor:
    r, g, b = rgb.unbind(dim=-1)
    maxc = torch.max(rgb, dim=-1).values
    minc = torch.min(rgb, dim=-1).values
    v = maxc
    delt = maxc - minc

    s = torch.where(maxc > 0.0, delt / (maxc + 1e-8), torch.zeros_like(maxc))
    h = torch.zeros_like(maxc)

    mask = delt > 1e-8
    delt_safe = delt + 1e-8

    rc = (maxc - r) / delt_safe
    gc = (maxc - g) / delt_safe
    bc = (maxc - b) / delt_safe

    h_r = (bc - gc) % 6.0
    h_g = rc - bc + 2.0
    h_b = gc - rc + 4.0

    is_r = (maxc == r) & mask
    is_g = (maxc == g) & mask
    is_b = (maxc == b) & mask

    h = torch.where(is_r, h_r, h)
    h = torch.where(is_g, h_g, h)
    h = torch.where(is_b, h_b, h)

    h = (h / 6.0) % 1.0
    return torch.stack((h, s, v), dim=-1)


def _hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
    h, s, v = hsv.unbind(dim=-1)
    h6 = (h % 1.0) * 6.0
    i = torch.floor(h6).to(torch.int64)
    f = h6 - torch.floor(h6)

    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    i_mod = i % 6
    r = torch.where(
        i_mod == 0,
        v,
        torch.where(
            i_mod == 1,
            q,
            torch.where(
                i_mod == 2,
                p,
                torch.where(i_mod == 3, p, torch.where(i_mod == 4, t, v)),
            ),
        ),
    )
    g = torch.where(
        i_mod == 0,
        t,
        torch.where(
            i_mod == 1,
            v,
            torch.where(
                i_mod == 2,
                v,
                torch.where(i_mod == 3, q, torch.where(i_mod == 4, p, p)),
            ),
        ),
    )
    b = torch.where(
        i_mod == 0,
        p,
        torch.where(
            i_mod == 1,
            p,
            torch.where(
                i_mod == 2,
                t,
                torch.where(i_mod == 3, v, torch.where(i_mod == 4, v, q)),
            ),
        ),
    )

    return _clamp01(torch.stack((r, g, b), dim=-1))


def _apply_saturation_hue(
    img: torch.Tensor,
    saturation: float = 1.0,
    hue_degrees: float = 0.0,
) -> torch.Tensor:
    hsv = _rgb_to_hsv(img)
    hsv[..., 0] = (hsv[..., 0] + (float(hue_degrees) / 360.0)) % 1.0
    hsv[..., 1] = (hsv[..., 1] * float(saturation)).clamp(0.0, 1.0)
    return _hsv_to_rgb(hsv)


class SuperBrightnessContrast:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "brightness": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01},
                ),
                "contrast": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01},
                ),
                "gamma": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.05, "max": 4.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "SuperNodes/Adjustment"

    def apply(self, image, brightness=1.0, contrast=1.0, gamma=1.0):
        return (
            _apply_brightness_contrast_gamma(
                image, brightness, contrast, gamma
            ),
        )


class SuperHueSaturation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "saturation": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01},
                ),
                "hue_degrees": (
                    "FLOAT",
                    {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.5},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "SuperNodes/Adjustment"

    def apply(self, image, saturation=1.0, hue_degrees=0.0):
        return (_apply_saturation_hue(image, saturation, hue_degrees),)


class SuperWhiteBalanceCAT:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "temperature_k": (
                    "INT",
                    {"default": 6500, "min": 1650, "max": 25000, "step": 50},
                ),
                "tint": (
                    "FLOAT",
                    {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "SuperNodes/Adjustment"

    def apply(self, image, temperature_k=6500, tint=0.0):
        out = _apply_white_balance_cat(image, float(temperature_k), float(tint))
        return (out,)


class SuperColorAdjustAllInOne:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "brightness": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01},
                ),
                "contrast": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01},
                ),
                "gamma": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.05, "max": 4.0, "step": 0.01},
                ),
                "saturation": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01},
                ),
                "hue_degrees": (
                    "FLOAT",
                    {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.5},
                ),
                "temperature_k": (
                    "INT",
                    {"default": 6500, "min": 1650, "max": 25000, "step": 50},
                ),
                "tint": (
                    "FLOAT",
                    {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "SuperNodes/Adjustment"

    def apply(
        self,
        image,
        brightness=1.0,
        contrast=1.0,
        gamma=1.0,
        saturation=1.0,
        hue_degrees=0.0,
        temperature_k=6500,
        tint=0.0,
    ):
        out = _apply_brightness_contrast_gamma(
            image, brightness, contrast, gamma
        )
        out = _apply_saturation_hue(out, saturation, hue_degrees)
        out = _apply_white_balance_cat(out, float(temperature_k), float(tint))
        return (out,)
