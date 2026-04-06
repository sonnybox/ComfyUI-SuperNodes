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


def _split_rgb_and_extra_channels(
    img: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Split image into RGB channels and optional extra channels (e.g. alpha)."""
    channels = img.shape[-1]
    if channels < 3:
        raise ValueError(
            f"Expected at least 3 channels in the last dimension, got {channels}."
        )
    rgb = img[..., :3]
    extras = img[..., 3:] if channels > 3 else None
    return rgb, extras


def _recombine_rgb_and_extra_channels(
    rgb: torch.Tensor, extras: torch.Tensor | None
) -> torch.Tensor:
    if extras is None:
        return rgb
    return torch.cat((rgb, extras), dim=-1)


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
    rgb_srgb, extras = _split_rgb_and_extra_channels(img_srgb)

    device = img_srgb.device
    dtype = img_srgb.dtype

    # 1) sRGB -> linear
    lin = _srgb_to_linear(_clamp01(rgb_srgb))

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

    # Calculate baseline D65 offset so 6500K is perfectly neutral (D65)
    base_x, base_y = _kelvin_to_xy_approx(6500.0)
    base_u, base_v = _xy_to_uv(base_x, base_y)
    d65_u, d65_v = _xy_to_uv(src_x, src_y)
    u_offset = d65_u - base_u
    v_offset = d65_v - base_v

    # Tint: shift in UCS v direction (green<->magenta feel)
    # Scale chosen to be "good enough" and not insane.
    # If you want stronger/weaker, tweak 0.05.
    u, v = _xy_to_uv(dst_x, dst_y)
    u = u + u_offset
    v = v + v_offset + float(tint) * 0.05
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
    out_rgb = _clamp01(_linear_to_srgb(lin_out))
    return _recombine_rgb_and_extra_channels(out_rgb, extras)


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
    rgb, extras = _split_rgb_and_extra_channels(img)
    hsv = _rgb_to_hsv(rgb)
    hsv[..., 0] = (hsv[..., 0] + (float(hue_degrees) / 360.0)) % 1.0
    hsv[..., 1] = (hsv[..., 1] * float(saturation)).clamp(0.0, 1.0)
    out_rgb = _hsv_to_rgb(hsv)
    return _recombine_rgb_and_extra_channels(out_rgb, extras)
