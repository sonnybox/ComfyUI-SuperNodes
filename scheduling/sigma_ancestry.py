from comfy_api.latest import io, ui
import torch

QUANTITY = "ancestry"
MODEL_TYPES = ["flow", "legacy"]
ETA_HARD_MAX = 100.0
SEARCH_ITERATIONS = 100
# what percentage points of noise power counts as reached
TOLERANCE = 0.001


def step_noise(sigma, sigma_next, eta, flow):
    """How one ancestral step moves the noise, as (carry, fresh_var).

    The existing noise is scaled by carry, then fresh noise of variance fresh_var (before
    s_noise) is added. Assumes a perfect denoiser, so only the noise matters. None for a
    terminal step, which never renoises.
    """
    if sigma_next <= 0.0 or sigma <= 0.0:
        # sigmas[i + 1] == 0 is the terminal step: x = denoised, nothing is renoised
        return None
    if not eta:
        return sigma_next / sigma, 0.0

    if flow:
        # comfy.k_diffusion.sampling.sample_euler_ancestral_RF
        sigma_down = sigma_next * (1.0 + (sigma_next / sigma - 1.0) * eta)
        alpha_next = 1.0 - sigma_next
        alpha_down = 1.0 - sigma_down
        if alpha_down == 0.0:
            return 1.0, 0.0
        # the Euler step scales the noise by sigma_down / sigma, the renoise rescale by
        # alpha_next / alpha_down, and renoise_coeff tops it back up to sigma_next
        carry = (sigma_down / sigma) * (alpha_next / alpha_down)
        fresh_var = sigma_next**2 - (sigma_down * alpha_next / alpha_down) ** 2
        return carry, max(fresh_var, 0.0)

    # comfy.k_diffusion.sampling.get_ancestral_step + sample_euler_ancestral
    inner = sigma_next**2 * (sigma**2 - sigma_next**2) / sigma**2
    sigma_up = min(sigma_next, eta * max(inner, 0.0) ** 0.5)
    sigma_down = max(sigma_next**2 - sigma_up**2, 0.0) ** 0.5
    if sigma_down == 0.0:
        # core takes x = denoised here and adds nothing: the noise is wiped outright
        return 0.0, 0.0
    return sigma_down / sigma, sigma_up**2


def step_eta_ceiling(sigma, sigma_next, flow):
    """Largest eta for which this step is still a real noise mix.

    For flow models that is where sigma_down reaches 0. Past it sigma_down goes negative
    and the Euler step extrapolates beyond x0. The legacy branch clamps sigma_up internally
    and has no such limit.
    """
    if not flow:
        return ETA_HARD_MAX
    if sigma_next <= 0.0 or sigma <= 0.0 or sigma_next >= sigma:
        return ETA_HARD_MAX
    return sigma / (sigma - sigma_next)


def schedule_eta_ceiling(sigmas, flow):
    """The tightest per-step ceiling. Ancestry is monotonic in eta below this."""
    ceilings = [
        step_eta_ceiling(sigmas[i], sigmas[i + 1], flow)
        for i in range(len(sigmas) - 1)
        if step_noise(sigmas[i], sigmas[i + 1], 0.0, flow) is not None
    ]
    # clamped: eta above ETA_HARD_MAX is not settable on any sampler anyway
    return min(min(ceilings), ETA_HARD_MAX) if ceilings else ETA_HARD_MAX


def has_ancestral_steps(sigmas, flow):
    return any(
        step_noise(sigmas[i], sigmas[i + 1], 0.0, flow) is not None
        for i in range(len(sigmas) - 1)
    )


def ancestry_of(sigmas, eta, flow, s_noise=1.0):
    """Share of the final noise power that still comes from the starting noise.

    Tracks the starting noise's power and the fresh noise's power separately. With s_noise
    1.0 their sum is always sigma**2 and this reduces to the product of the per-step
    carries, but s_noise over- or under-fills the renoise, so the two are tracked apart.
    """
    ancestral = sigmas[0] ** 2
    fresh = 0.0
    for i in range(len(sigmas) - 1):
        step = step_noise(sigmas[i], sigmas[i + 1], eta, flow)
        if step is None:
            continue
        carry, fresh_var = step
        ancestral *= carry**2
        fresh = fresh * carry**2 + s_noise**2 * fresh_var
    total = ancestral + fresh
    return ancestral / total if total > 0.0 else 0.0


def eta_for_ancestry(sigmas, target, flow, s_noise, ceiling):
    """Invert ancestry_of. Returns (eta, reachable).

    Finds the lowest eta that brings ancestry within TOLERANCE of the target, searching only
    below the ceiling, where ancestry falls monotonically, so bisection is valid.
    """
    goal = max(target, TOLERANCE)
    if ancestry_of(sigmas, 0.0, flow, s_noise) <= goal:
        return 0.0, True
    hi = ceiling * (1.0 - 1e-9)
    if ancestry_of(sigmas, hi, flow, s_noise) > goal:
        return hi, False
    lo = 0.0
    for _ in range(SEARCH_ITERATIONS):
        mid = (lo + hi) / 2.0
        if ancestry_of(sigmas, mid, flow, s_noise) > goal:
            lo = mid
        else:
            hi = mid
    # hi is the side already at or under the goal
    return hi, True


def as_list(sigmas):
    if isinstance(sigmas, torch.Tensor):
        return [float(x) for x in sigmas.detach().cpu().flatten()]
    return [float(x) for x in sigmas]


def build_ladder(values, target, flow, s_noise, ceiling):
    """Markdown table of eta against the quantity, in even 5% steps.

    Even in eta is the wrong axis - the interesting range is compressed into the low end -
    so the rows step evenly in ancestry and let eta land where it lands.
    """
    lines = [
        "| {} | eta |".format(QUANTITY),
        "| ---: | ---: |",
    ]
    clamped = False
    for pct in range(100, -1, -5):
        eta, reachable = eta_for_ancestry(
            values, pct / 100.0, flow, s_noise, ceiling
        )
        cell = "{:.4f}".format(eta)
        if not reachable:
            cell += " \\*"
            clamped = True
        if abs(pct - target * 100.0) < 1e-9:
            lines.append("| **{}%** | **{}** |".format(pct, cell))
        else:
            lines.append("| {}% | {} |".format(pct, cell))
    if clamped:
        lines += [
            "",
            "\\* unreachable on this schedule, clamped to the eta ceiling.",
        ]
    return "\n".join(lines)


class SigmaAncestry(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SigmaAncestry",
            display_name="🐧 Sigma Ancestry",
            category="SuperNodes/Scheduling",
            description="Solves the eta that determines the noise deviance from the starting noise for Euler solvers.",
            is_output_node=True,
            inputs=[
                io.Custom("SIGMAS").Input(
                    "sigmas",
                    tooltip="The same schedule the sampler gets.",
                ),
                io.Float.Input(
                    "target_ancestry",
                    default=50.0,
                    min=0.0,
                    max=100.0,
                    step=0.5,
                    round=False,
                    tooltip="% of final noise power from the starting noise. 100% is Euler.",
                ),
                io.Combo.Input(
                    "model_type",
                    options=MODEL_TYPES,
                    default=MODEL_TYPES[0],
                    tooltip="flow: Flux, SD3, Wan, H3. legacy: SD1.5, SDXL.",
                ),
                # last, so saved workflows keep their widget order
                io.Float.Input(
                    "s_noise",
                    default=1.0,
                    min=0.0,
                    max=100.0,
                    step=0.01,
                    round=False,
                    tooltip="Match the sampler's s_noise.",
                ),
            ],
            outputs=[
                io.Float.Output(display_name="target_eta"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(
        cls, sigmas, target_ancestry, model_type, s_noise=1.0
    ) -> io.NodeOutput:
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

        # Two cases where eta cannot move ancestry at all, so any eta is as good as 0
        note = None
        if not has_ancestral_steps(values, flow):
            note = "No ancestral steps on this schedule, so eta has no effect."
        elif s_noise == 0.0:
            note = "s_noise 0 adds no fresh noise, so ancestry stays at 100%."
        if note is not None:
            return io.NodeOutput(0.0, note, ui=ui.PreviewText(note))

        target = target_ancestry / 100.0
        ceiling = schedule_eta_ceiling(values, flow)
        target_eta, _ = eta_for_ancestry(values, target, flow, s_noise, ceiling)
        report = build_ladder(values, target, flow, s_noise, ceiling)
        return io.NodeOutput(target_eta, report, ui=ui.PreviewText(report))


NODE = [SigmaAncestry]
