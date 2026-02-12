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


def _latent_stats_string(
    x: torch.Tensor,
    name: str = "latent",
    top_k: int = 12,
) -> str:
    """
    x: [B,C,H,W]
    Prints extra diagnostics to help design latent-space ops:
      - channels with largest std (most active)
      - channels with largest |mean| (biased)
      - correlation with an "energy" map (structure/detail proxy)
      - top correlated channel pairs (good candidates for rotate/mix)
    """
    with torch.no_grad():
        B, C, H, W = x.shape
        device = str(x.device)
        dtype = str(x.dtype)

        # Global stats
        g_min = x.min().item()
        g_max = x.max().item()
        g_mean = x.mean().item()
        g_std = x.std(unbiased=False).item()

        # Quantiles (global per-batch-item, then avg across batch)
        flat = x.reshape(B, -1)
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
        q = torch.quantile(flat_s, qs, dim=1).mean(dim=1)  # [len(qs)]

        # Per-channel mean/std over (B,H,W)
        ch_mean = x.mean(dim=(0, 2, 3))
        ch_std = x.std(dim=(0, 2, 3), unbiased=False)

        # Top-K channels by std
        k = min(int(top_k), C)
        std_vals, std_idx = torch.topk(ch_std, k=k, largest=True)

        # Top-K channels by |mean|
        absmean_vals, absmean_idx = torch.topk(ch_mean.abs(), k=k, largest=True)

        # "Energy map" proxy: per-pixel magnitude averaged over channels
        # This is a decent proxy for "structure/detail strength"
        energy = x.abs().mean(dim=1)  # [B,H,W]
        energy_flat = energy.reshape(B, -1)

        # Correlation of each channel with energy (avg over batch)
        # corr(c, energy) = cov / (std_c * std_e)
        # We compute per batch item then average.
        x_ch = x.reshape(B, C, -1)  # [B,C,N]
        x_ch_mean = x_ch.mean(dim=2, keepdim=True)
        e_mean = energy_flat.mean(dim=1, keepdim=True)

        x0 = x_ch - x_ch_mean
        e0 = energy_flat - e_mean  # [B,N]

        cov = (x0 * e0[:, None, :]).mean(dim=2)  # [B,C]
        x_std_b = x_ch.std(dim=2, unbiased=False).clamp_min(1e-6)  # [B,C]
        e_std_b = energy_flat.std(dim=1, unbiased=False).clamp_min(1e-6)  # [B]

        corr = (cov / (x_std_b * e_std_b[:, None])).mean(dim=0)  # [C]
        corr_abs = corr.abs()
        corr_vals, corr_idx = torch.topk(corr_abs, k=k, largest=True)

        # Top correlated channel pairs (for rotate/mix candidates)
        # Do this on a reduced subset for speed: use the union of top std and top |corr|
        subset = torch.unique(torch.cat([std_idx, corr_idx], dim=0))
        subC = subset.numel()

        pair_lines = []
        if subC >= 2:
            # Build correlation matrix on subset using flattened vectors (averaged over batch)
            # We standardize per (B) and then average correlations across batch
            xs = x_ch[:, subset, :]  # [B,subC,N]
            xs0 = xs - xs.mean(dim=2, keepdim=True)
            xs_std = xs.std(dim=2, unbiased=False, keepdim=True).clamp_min(1e-6)
            zs = xs0 / xs_std  # [B,subC,N]

            # corr matrix per batch: (Z @ Z^T) / N
            # We'll average over B
            # Result: [subC, subC]
            corr_mat = torch.einsum("bcn,bdn->bcd", zs, zs).mean(dim=0) / float(
                zs.shape[2]
            )

            # Take upper triangle pairs, get top few by |corr|
            iu, ju = torch.triu_indices(subC, subC, offset=1, device=x.device)
            pair_scores = corr_mat[iu, ju].abs()
            top_pairs = min(8, pair_scores.numel())
            ps, pidx = torch.topk(pair_scores, k=top_pairs, largest=True)
            for rank in range(top_pairs):
                i = int(iu[int(pidx[rank])].item())
                j = int(ju[int(pidx[rank])].item())
                ci = int(subset[i].item())
                cj = int(subset[j].item())
                raw_corr = float(corr_mat[i, j].item())
                pair_lines.append(f"({ci},{cj}): corr={_fmt(raw_corr)}")
        else:
            pair_lines.append("(not enough channels in subset)")

        # Helper format lists
        def _idxvals(idx_t, val_t, label):
            parts = []
            for i in range(idx_t.numel()):
                parts.append(
                    f"{int(idx_t[i].item())}:{_fmt(float(val_t[i].item()))}"
                )
            return f"{label}: " + ", ".join(parts)

        ch_mean_list = ", ".join(_fmt(v.item()) for v in ch_mean)
        ch_std_list = ", ".join(_fmt(v.item()) for v in ch_std)

        # Recommendations
        rec_pairs = []
        if pair_lines and "(not enough" not in pair_lines[0]:
            # first couple pairs are good rotate candidates
            rec_pairs = pair_lines[:3]

        s = (
            f"{name}: shape=[{B},{C},{H},{W}] dtype={dtype} device={device}\n"
            f"global: min={_fmt(g_min)} max={_fmt(g_max)} mean={_fmt(g_mean)} std={_fmt(g_std)}\n"
            f"quantiles(avg batch): q0.1%={_fmt(q[0].item())} q1%={_fmt(q[1].item())} "
            f"q50%={_fmt(q[2].item())} q99%={_fmt(q[3].item())} q99.9%={_fmt(q[4].item())}\n"
            f"{_idxvals(std_idx, std_vals, f'top{k}_ch_by_std')}\n"
            f"{_idxvals(absmean_idx, ch_mean.abs()[absmean_idx], f'top{k}_ch_by_|mean|')}\n"
            f"{_idxvals(corr_idx, corr_abs[corr_idx], f'top{k}_ch_by_|corr_with_energy|')}\n"
            f"top_correlated_pairs_in_subset: " + "; ".join(pair_lines) + "\n"
            f"recommended_rotate_pairs: "
            + ("; ".join(rec_pairs) if rec_pairs else "(none)")
            + "\n"
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
                "top_k": (
                    "INT",
                    {"default": 12, "min": 3, "max": 64, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"
    CATEGORY = "SuperNodes/debug"

    def run(self, latent, label="latent", top_k=12):
        x = _get_samples(latent)
        return (_latent_stats_string(x, name=str(label), top_k=int(top_k)),)
