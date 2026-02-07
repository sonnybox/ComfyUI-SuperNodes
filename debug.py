import torch


def _get_samples(latent: dict) -> torch.Tensor:
    x = latent["samples"]
    if getattr(x, "is_nested", False):
        x = x.unbind()[0]
    return x


def _fmt(x: float) -> str:
    if abs(x) >= 1000:
        return f"{x:.3e}"
    return f"{x:.6f}"


def _latent_stats_string(x: torch.Tensor, name: str = "latent") -> str:
    with torch.no_grad():
        B, C, H, W = x.shape

        g_min = x.min().item()
        g_max = x.max().item()
        g_mean = x.mean().item()
        g_std = x.std(unbiased=False).item()

        ch_mean = x.mean(dim=(0, 2, 3))
        ch_std = x.std(dim=(0, 2, 3), unbiased=False)

        flat = x.reshape(B, -1)

        # Optional subsample for speed on huge tensors
        if flat.shape[1] > 1_000_000:
            idx = torch.randint(
                0, flat.shape[1], (B, 1_000_000), device=x.device
            )
            flat_s = flat.gather(1, idx)
        else:
            flat_s = flat

        qs = torch.tensor(
            [0.001, 0.01, 0.5, 0.99, 0.999], device=x.device, dtype=x.dtype
        )

        # quantiles: [len(qs), B]
        q = torch.quantile(flat_s, qs, dim=1)

        # average across batch -> [len(qs)]
        q = q.mean(dim=1)

        ch_mean_list = ", ".join(_fmt(v.item()) for v in ch_mean)
        ch_std_list = ", ".join(_fmt(v.item()) for v in ch_std)

        s = (
            f"{name}: shape=[{B},{C},{H},{W}]\n"
            f"global: min={_fmt(g_min)} max={_fmt(g_max)} mean={_fmt(g_mean)} std={_fmt(g_std)}\n"
            f"quantiles(avg batch): q0.1%={_fmt(q[0].item())} q1%={_fmt(q[1].item())} "
            f"q50%={_fmt(q[2].item())} q99%={_fmt(q[3].item())} q99.9%={_fmt(q[4].item())}\n"
            f"channel mean: [{ch_mean_list}]\n"
            f"channel std : [{ch_std_list}]"
        )
        return s


class SuperLatentStats:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "label": ("STRING", {"default": "latent"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"
    CATEGORY = "SuperNodes/adjustment"

    def run(self, latent, label="latent"):
        x = _get_samples(latent)
        return (_latent_stats_string(x, name=str(label)),)


class SuperLatentStatsPrint:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "label": ("STRING", {"default": "latent"}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "run"
    CATEGORY = "SuperNodes/adjustment"

    def run(self, latent, label="latent"):
        x = _get_samples(latent)
        print(_latent_stats_string(x, name=str(label)))
        return (latent,)


class SuperLatentDeltaStats:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent_a": ("LATENT",),
                "latent_b": ("LATENT",),
                "label": ("STRING", {"default": "delta"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"
    CATEGORY = "SuperNodes/adjustment"

    def run(self, latent_a, latent_b, label="delta"):
        a = _get_samples(latent_a)
        b = _get_samples(latent_b)

        if a.shape != b.shape:
            return (
                f"{label}: shapes differ: {tuple(a.shape)} vs {tuple(b.shape)}",
            )

        d = b - a
        with torch.no_grad():
            mean_abs = d.abs().mean().item()
            max_abs = d.abs().max().item()
            mean = d.mean().item()
            std = d.std(unbiased=False).item()

            ch_mean_abs = d.abs().mean(dim=(0, 2, 3))
            ch_list = ", ".join(_fmt(v.item()) for v in ch_mean_abs)

            af = a.flatten()
            bf = b.flatten()
            denom = (af.norm() * bf.norm()).clamp_min(1e-8)
            cos = (af @ bf / denom).item()

        s = (
            f"{label}: delta stats\n"
            f"mean_abs={_fmt(mean_abs)} max_abs={_fmt(max_abs)} mean={_fmt(mean)} std={_fmt(std)}\n"
            f"channel mean_abs: [{ch_list}]\n"
            f"cosine_sim(a,b)={_fmt(cos)}"
        )
        return (s,)
