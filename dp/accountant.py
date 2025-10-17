import math
from typing import Optional, List, Tuple


class PrivacyAccount:
    def __init__(self, eps: float, delta: float, mechanism: str, backend: str, details: dict):
        self.eps = float(eps)
        self.delta = float(delta)
        self.mechanism = mechanism
        self.backend = backend
        self.details = dict(details or {})

    def as_dict(self):
        return {
            "epsilon": self.eps,
            "delta": self.delta,
            "mechanism": self.mechanism,
            "backend": self.backend,
            "details": self.details,
        }


def _try_prv_accountant(noise_multiplier: float, steps: int, sample_rate: float, delta: float) -> Optional[PrivacyAccount]:
    """Try to use prv_accountant if available; return None if import fails.

    Interprets each round as subsampled Gaussian with sampling probability q=sample_rate and noise multiplier sigma.
    """
    try:
        # Lazy import; will fail if not installed.
        from prv_accountant import Accountant, GaussianMechanism

        mech = GaussianMechanism(noise_multiplier=noise_multiplier, sampling_probability=sample_rate)
        acc = Accountant(target_delta=delta)
        for _ in range(steps):
            acc.compose(mech)
        eps = acc.get_epsilon()
        return PrivacyAccount(eps=eps, delta=delta, mechanism="Gaussian", backend="prv_accountant",
                              details={"steps": steps, "q": sample_rate, "sigma": noise_multiplier})
    except Exception:
        return None


def _gdp_from_zcdp(rho: float, delta: float) -> float:
    """Convert zCDP cumulative rho to (epsilon, delta) via standard bound: eps = rho + 2*sqrt(rho*ln(1/delta))."""
    if rho <= 0:
        return 0.0
    return float(rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta)))


def _rdp_gaussian(q: float, noise_multiplier: float, steps: int, orders: List[float]) -> List[float]:
    """RDP of subsampled Gaussian mechanism (Poisson sampling) using a simplified upper bound.

    This follows the TF Privacy style approximation: for each alpha in orders,
    RDP(alpha) <= (1/(alpha-1)) * log(1 + q^2 * alpha * (alpha-1) / (2 * sigma^2)) * steps
    This is a conservative bound; tighter variants exist but require more code.
    """
    rdp = []
    sigma2 = noise_multiplier ** 2
    for alpha in orders:
        if alpha <= 1:
            rdp.append(float("inf"))
            continue
        term = 1.0 + (q ** 2) * alpha * (alpha - 1.0) / (2.0 * sigma2)
        rdp_alpha = steps * math.log(max(term, 1.0)) / (alpha - 1.0)
        rdp.append(rdp_alpha)
    return rdp


def _epsilon_from_rdp(orders: List[float], rdp: List[float], delta: float) -> Tuple[float, float]:
    """Compute epsilon given RDP and delta; picks best alpha.
    Returns (eps, optimal_order).
    """
    best_eps = float("inf")
    best_alpha = None
    for order, rdp_alpha in zip(orders, rdp):
        if order <= 1:
            continue
        eps_alpha = rdp_alpha + math.log(1.0 / delta) / (order - 1.0)
        if eps_alpha < best_eps:
            best_eps = eps_alpha
            best_alpha = order
    return float(best_eps), float(best_alpha or 0)


def compute_privacy(
    *,
    rounds: int,
    noise_multiplier: float,
    delta: float,
    sample_rate: float = 1.0,
    prefer_prv: bool = True,
) -> PrivacyAccount:
    """Compute epsilon for central DP FedAvg with Gaussian noise on aggregated sum.

    Preference order:
      1) prv_accountant (if installed)
      2) GDP/zCDP exact for full participation (q=1)
      3) RDP subsampled Gaussian approximation for q<1
    """
    rounds = int(rounds)
    sample_rate = float(sample_rate)
    sigma = float(noise_multiplier)
    if rounds <= 0 or sigma <= 0:
        return PrivacyAccount(eps=0.0, delta=delta, mechanism="Gaussian", backend="none",
                              details={"rounds": rounds, "q": sample_rate, "sigma": sigma})

    if prefer_prv:
        prv = _try_prv_accountant(noise_multiplier=sigma, steps=rounds, sample_rate=sample_rate, delta=delta)
        if prv is not None:
            return prv

    # GDP/zCDP for q==1
    if abs(sample_rate - 1.0) < 1e-9:
        # Each round: zCDP rho = 1 / (2 * sigma^2); composition is additive.
        rho = rounds * (1.0 / (2.0 * sigma * sigma))
        eps = _gdp_from_zcdp(rho=rho, delta=delta)
        return PrivacyAccount(eps=eps, delta=delta, mechanism="Gaussian", backend="gdp_zcdp",
                              details={"rounds": rounds, "q": sample_rate, "sigma": sigma, "rho": rho})

    # Fallback: RDP subsampled Gaussian approximation
    orders = [1.25, 1.5, 2, 3, 4, 8, 16, 32, 64, 128]
    rdp = _rdp_gaussian(q=sample_rate, noise_multiplier=sigma, steps=rounds, orders=orders)
    eps, alpha = _epsilon_from_rdp(orders, rdp, delta)
    return PrivacyAccount(eps=eps, delta=delta, mechanism="Gaussian", backend="rdp_subsample",
                          details={"rounds": rounds, "q": sample_rate, "sigma": sigma, "alpha": alpha})

