from comfy_api.latest import io, ui
import torch

QUANTITY = "ancestry"
MODEL_TYPES = ["flow", "legacy"]
ETA_HARD_MAX = 100.0


def step_retention(sigma, sigma_next, eta, flow):
    """Fraction of the existing noise field that survives one ancestral step.

    A step is: Euler down to sigma_down, then renoise back up to sigma_next. The old field
    is what the Euler step carried; the rest is freshly drawn. Returns that ratio at the
    new noise level, or None for a terminal step that never renoises.
    """
    if sigma_next <= 0.0 or sigma <= 0.0:
        # sigmas[i + 1] == 0 is the terminal step: x = denoised, nothing is renoised
        return None

    if flow:
        # comfy.k_diffusion.sampling.sample_euler_ancestral_RF
        sigma_down = sigma_next * (1.0 + (sigma_next / sigma - 1.0) * eta)
        alpha_down = 1.0 - sigma_down
        if alpha_down == 0.0:
            return 1.0
        # the Euler step scales the old field by sigma_down / sigma, the renoise rescale
        # by alpha_next / alpha_down; expressed against sigma_next that is:
        return abs(((1.0 - sigma_next) * sigma_down) / (alpha_down * sigma_next))

    # comfy.k_diffusion.sampling.get_ancestral_step
    if not eta:
        return 1.0
    inner = sigma_next**2 * (sigma**2 - sigma_next**2) / sigma**2
    sigma_up = min(sigma_next, eta * max(inner, 0.0) ** 0.5)
    sigma_down = max(sigma_next**2 - sigma_up**2, 0.0) ** 0.5
    return sigma_down / sigma_next


def step_eta_ceiling(sigma, sigma_next, flow):
    """Largest eta for which this step is still a real noise mix.

    For flow models that is where sigma_down reaches 0. Past it sigma_down goes negative,
    the Euler step extrapolates beyond x0, and the retention formula turns around and
    climbs back above 1 - which is not the field surviving, it is the step having left its
    valid domain. The legacy branch clamps sigma_up internally and has no such limit.
    """
    if not flow:
        return ETA_HARD_MAX
    if sigma_next <= 0.0 or sigma <= 0.0 or sigma_next >= sigma:
        return ETA_HARD_MAX
    return sigma / (sigma - sigma_next)


def schedule_eta_ceiling(sigmas, flow):
    """The tightest per-step ceiling. Retention is monotonic in eta below this."""
    ceilings = [
        step_eta_ceiling(sigmas[i], sigmas[i + 1], flow)
        for i in range(len(sigmas) - 1)
        if step_retention(sigmas[i], sigmas[i + 1], 0.0, flow) is not None
    ]
    # clamped: eta above ETA_HARD_MAX is not settable on any sampler anyway
    return min(min(ceilings), ETA_HARD_MAX) if ceilings else ETA_HARD_MAX


def ancestry_of(sigmas, eta, flow):
    """Product of the per-step retentions: what is left of the starting field at the end."""
    keep = 1.0
    for i in range(len(sigmas) - 1):
        r = step_retention(sigmas[i], sigmas[i + 1], eta, flow)
        if r is not None:
            keep *= r
    return keep


def eta_for_ancestry(sigmas, target, flow, ceiling=None):
    """Invert ancestry_of. Returns (eta, reachable).

    Only searches below the ceiling, where retention falls monotonically, so bisection is
    valid. A target below what the ceiling can reach comes back clamped and flagged.
    """
    if ceiling is None:
        ceiling = schedule_eta_ceiling(sigmas, flow)
    hi = ceiling * (1.0 - 1e-9)
    if ancestry_of(sigmas, 0.0, flow) <= target:
        return 0.0, True
    if ancestry_of(sigmas, hi, flow) >= target:
        return hi, False
    lo = 0.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if ancestry_of(sigmas, mid, flow) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0, True


def as_list(sigmas):
    if isinstance(sigmas, torch.Tensor):
        return [float(x) for x in sigmas.detach().cpu().flatten()]
    return [float(x) for x in sigmas]


def build_ladder(values, target, flow, ceiling):
    """Markdown table of eta against the quantity, in even 5% steps.

    Even in eta is the wrong axis - the interesting range is compressed into the low end -
    so the rows step evenly in ancestry and let eta land where it lands.
    """
    lines = [
        "| {} | eta |".format(QUANTITY),
        "| ---: | ---: |",
    ]
    for pct in range(100, -1, -5):
        eta, reachable = eta_for_ancestry(values, pct / 100.0, flow, ceiling)
        cell = "{:.4f}".format(eta) if reachable else "{:.4f} \\*".format(eta)
        if abs(pct - target * 100.0) < 1e-9:
            lines.append("| **{}%** | **{}** |".format(pct, cell))
        else:
            lines.append("| {}% | {} |".format(pct, cell))
    if any(
        not eta_for_ancestry(values, p / 100.0, flow, ceiling)[1]
        for p in range(100, -1, -5)
    ):
        lines += ["", "\\* unreachable on this schedule, clamped to the eta ceiling."]
    return "\n".join(lines)


class SigmaAncestry(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SigmaAncestry",
            display_name="🐧 Sigma Ancestry",
            category="SuperNodes/Scheduling",
            description="Solves the ancestral eta that leaves a chosen share of the starting noise field intact. Sweep a schedule in even ancestry steps instead of in eta.",
            is_output_node=True,
            inputs=[
                io.Custom("SIGMAS").Input(
                    "sigmas",
                    tooltip="The same schedule the sampler gets.",
                ),
                io.Float.Input(
                    "target_ancestry",
                    default=70.0,
                    min=0.0,
                    max=100.0,
                    step=0.5,
                    round=False,
                    tooltip="Percent of the starting noise field to keep.",
                ),
                io.Combo.Input(
                    "model_type",
                    options=MODEL_TYPES,
                    default=MODEL_TYPES[0],
                    tooltip="flow for rectified-flow / CONST models (Flux, SD3, Wan, MiniMax H3). legacy for the older epsilon models on Karras schedules (SD1.5, SDXL).",
                ),
            ],
            outputs=[
                io.Float.Output(
                    display_name="target_eta",
                    tooltip="Wire this into a sampler's eta to hold the schedule at target_ancestry.",
                ),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, sigmas, target_ancestry, model_type) -> io.NodeOutput:
        values = as_list(sigmas)
        if len(values) < 2:
            raise ValueError(
                "sigmas needs at least two entries to have a step, got {}.".format(
                    len(values)
                )
            )
        if all(s <= 0.0 for s in values):
            raise ValueError("every sigma is zero, so nothing is sampled.")

        flow = model_type == "flow"
        target = target_ancestry / 100.0
        ceiling = schedule_eta_ceiling(values, flow)

        target_eta, _ = eta_for_ancestry(values, target, flow, ceiling)
        report = build_ladder(values, target, flow, ceiling)
        return io.NodeOutput(target_eta, report, ui=ui.PreviewText(report))


NODE = [SigmaAncestry]
