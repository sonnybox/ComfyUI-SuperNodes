import torch
import torch.nn.functional as F


def rgb_to_xyz(rgb: torch.Tensor) -> torch.Tensor:
    # rgb shape: [..., 3] in range [0, 1]
    mask = rgb > 0.04045
    rgb = torch.where(
        mask,
        torch.pow(((rgb + 0.055) / 1.055).clamp(min=1e-6), 2.4),
        rgb / 12.92,
    )

    # sRGB to XYZ (D65) matrix
    mat = torch.tensor(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193308, 0.1191920, 0.9503041],
        ],
        device=rgb.device,
        dtype=rgb.dtype,
    )

    return F.linear(rgb, mat)


def xyz_to_lab(xyz: torch.Tensor) -> torch.Tensor:
    illuminant = torch.tensor(
        [0.95047, 1.00000, 1.08883], device=xyz.device, dtype=xyz.dtype
    )
    xyz_norm = xyz / illuminant

    mask = xyz_norm > 0.00885645
    f = torch.where(
        mask,
        torch.pow(xyz_norm.clamp(min=1e-6), 1.0 / 3.0),
        7.787037 * xyz_norm + 16.0 / 116.0,
    )

    l = 116.0 * f[..., 1] - 16.0  # noqa: E741
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])

    return torch.stack([l, a, b], dim=-1)


def rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    return xyz_to_lab(rgb_to_xyz(rgb))


def lab_to_xyz(lab: torch.Tensor) -> torch.Tensor:
    l = lab[..., 0]  # noqa: E741
    a = lab[..., 1]
    b = lab[..., 2]

    fy = (l + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0

    fxyz = torch.stack([fx, fy, fz], dim=-1)
    mask = fxyz > 0.2068966

    xyz_norm = torch.where(
        mask,
        torch.pow(fxyz.clamp(min=1e-6), 3.0),
        (fxyz - 16.0 / 116.0) / 7.787037,
    )
    illuminant = torch.tensor(
        [0.95047, 1.00000, 1.08883], device=lab.device, dtype=lab.dtype
    )
    return xyz_norm * illuminant


def xyz_to_rgb(xyz: torch.Tensor) -> torch.Tensor:
    mat = torch.tensor(
        [
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252],
        ],
        device=xyz.device,
        dtype=xyz.dtype,
    )

    rgb = F.linear(xyz, mat)
    mask = rgb > 0.0031308
    rgb = torch.where(
        mask,
        1.055 * torch.pow(rgb.clamp(min=1e-6), 1.0 / 2.4) - 0.055,
        12.92 * rgb,
    )
    return rgb


def lab_to_rgb(lab: torch.Tensor) -> torch.Tensor:
    return xyz_to_rgb(lab_to_xyz(lab))
