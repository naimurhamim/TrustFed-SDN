"""Renyi Differential Privacy (RDP) accountant for the Sampled Gaussian Mechanism (`Step 05`).

The accountant follows Mironov, Talwar & Zhang (2019), "Renyi Differential Privacy of the
Sampled Gaussian Mechanism", using the exact closed form for integer RDP orders `alpha >= 2`:

    A_alpha = sum_{k=0}^{alpha} C(alpha, k) (1 - q)^(alpha - k) q^k exp((k^2 - k) / (2 sigma^2))
    RDP(alpha) = log(A_alpha) / (alpha - 1)

RDP composes additively over steps. Conversion to `(epsilon, delta)`-DP uses the tighter
bound of Balle et al. (2020), Theorem 21:

    epsilon = RDP_total(alpha) + log((alpha - 1) / alpha) - (log(delta) + log(alpha)) / (alpha - 1)

minimized over the available orders. Only integer orders are used, which gives a valid
(slightly conservative) upper bound on epsilon.

The mechanism assumes Poisson subsampling with rate `q`: every record is included in each
step independently with probability `q`. `trustfed.privacy.dp_sgd` implements exactly that.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Sequence

import numpy as np
from scipy.special import gammaln, logsumexp

DEFAULT_ORDERS: tuple[int, ...] = tuple(list(range(2, 65)) + [80, 96, 128, 160, 192, 256, 384, 512])


@functools.lru_cache(maxsize=512)
def _log_binom_cached(n: int) -> tuple[np.ndarray, np.ndarray]:
    k = np.arange(n + 1, dtype=np.float64)
    lb = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
    return k, lb


@functools.lru_cache(maxsize=1024)
def _rdp_sampled_gaussian_cached(
    q: float,
    noise_multiplier: float,
    orders_tuple: tuple[int, ...],
) -> tuple[float, ...]:
    sigma2 = float(noise_multiplier) ** 2
    out: list[float] = []
    log_q = math.log(q) if 0.0 < q < 1.0 else 0.0
    log_1mq = math.log1p(-q) if 0.0 < q < 1.0 else 0.0
    inv_2sigma2 = 1.0 / (2.0 * sigma2)
    for alpha in orders_tuple:
        if q == 1.0:
            out.append(float(alpha / (2.0 * sigma2)))
            continue
        k, lb = _log_binom_cached(int(alpha))
        log_terms = lb + (alpha - k) * log_1mq + k * log_q + (k * k - k) * inv_2sigma2
        out.append(float(logsumexp(log_terms)) / (alpha - 1))
    return tuple(out)


def rdp_sampled_gaussian(
    q: float,
    noise_multiplier: float,
    orders: Sequence[int] = DEFAULT_ORDERS,
) -> np.ndarray:
    """Return the per-step RDP of the Sampled Gaussian Mechanism at each integer order.

    Args:
        q: Poisson sampling rate in `[0, 1]`.
        noise_multiplier: `sigma`, the ratio of Gaussian noise std to the clipping norm.
        orders: Integer RDP orders `alpha >= 2`.
    """
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"Sampling rate q must be in [0, 1], got {q}.")
    if noise_multiplier < 0.0:
        raise ValueError(f"noise_multiplier must be >= 0, got {noise_multiplier}.")

    orders_tuple = tuple(int(a) for a in orders)
    if any(a < 2 for a in orders_tuple):
        raise ValueError("All RDP orders must be integers >= 2.")

    if q == 0.0:
        return np.zeros(len(orders_tuple), dtype=np.float64)
    if noise_multiplier == 0.0:
        return np.full(len(orders_tuple), np.inf, dtype=np.float64)

    return np.asarray(
        _rdp_sampled_gaussian_cached(float(q), float(noise_multiplier), orders_tuple),
        dtype=np.float64,
    )


def rdp_to_epsilon(
    rdp: np.ndarray,
    delta: float,
    orders: Sequence[int] = DEFAULT_ORDERS,
) -> tuple[float, int]:
    """Convert a cumulative RDP curve to `(epsilon, best_order)` for the given `delta`."""
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta}.")
    orders_arr = np.asarray(orders, dtype=np.float64)
    rdp_arr = np.asarray(rdp, dtype=np.float64)
    if rdp_arr.shape != orders_arr.shape:
        raise ValueError("rdp and orders must have the same length.")

    eps = (
        rdp_arr
        + np.log((orders_arr - 1.0) / orders_arr)
        - (math.log(delta) + np.log(orders_arr)) / (orders_arr - 1.0)
    )
    eps = np.where(np.isnan(eps), np.inf, eps)
    best = int(np.argmin(eps))
    return float(max(0.0, eps[best])), int(orders_arr[best])


def compute_epsilon(
    q: float,
    noise_multiplier: float,
    steps: int,
    delta: float,
    orders: Sequence[int] = DEFAULT_ORDERS,
) -> float:
    """Epsilon after `steps` compositions of the Sampled Gaussian Mechanism."""
    if steps <= 0:
        return 0.0
    rdp = rdp_sampled_gaussian(q, noise_multiplier, orders) * int(steps)
    eps, _ = rdp_to_epsilon(rdp, delta, orders)
    return eps


def calibrate_noise_multiplier(
    target_epsilon: float,
    q: float,
    steps: int,
    delta: float,
    *,
    orders: Sequence[int] = DEFAULT_ORDERS,
    sigma_low: float = 0.05,
    sigma_high: float = 500.0,
    tol: float = 1e-4,
) -> float:
    """Smallest noise multiplier `sigma` whose `(epsilon, delta)` after `steps` is <= target.

    Uses bisection, since epsilon is monotonically decreasing in sigma.
    """
    if target_epsilon <= 0.0:
        raise ValueError(f"target_epsilon must be > 0, got {target_epsilon}.")
    if compute_epsilon(q, sigma_high, steps, delta, orders) > target_epsilon:
        raise ValueError(
            f"Cannot reach epsilon={target_epsilon} with sigma <= {sigma_high} "
            f"(q={q}, steps={steps}, delta={delta})."
        )
    lo, hi = float(sigma_low), float(sigma_high)
    if compute_epsilon(q, lo, steps, delta, orders) <= target_epsilon:
        return lo
    while hi - lo > tol * max(1.0, lo):
        mid = 0.5 * (lo + hi)
        if compute_epsilon(q, mid, steps, delta, orders) <= target_epsilon:
            hi = mid
        else:
            lo = mid
    return hi


class RDPAccountant:
    """Tracks cumulative RDP of a sequence of Sampled Gaussian Mechanism steps."""

    def __init__(self, orders: Sequence[int] = DEFAULT_ORDERS) -> None:
        self.orders: tuple[int, ...] = tuple(int(a) for a in orders)
        self._rdp = np.zeros(len(self.orders), dtype=np.float64)
        self.steps = 0

    def step(self, q: float, noise_multiplier: float, num_steps: int = 1) -> None:
        """Record `num_steps` steps with sampling rate `q` and noise multiplier `sigma`."""
        if num_steps <= 0:
            return
        self._rdp = self._rdp + rdp_sampled_gaussian(q, noise_multiplier, self.orders) * int(
            num_steps
        )
        self.steps += int(num_steps)

    def get_epsilon(self, delta: float) -> float:
        """Return the current epsilon for the given delta."""
        if self.steps == 0:
            return 0.0
        eps, _ = rdp_to_epsilon(self._rdp, delta, self.orders)
        return eps

    def get_epsilon_and_order(self, delta: float) -> tuple[float, int]:
        """Return `(epsilon, best_order)` for the given delta."""
        if self.steps == 0:
            return 0.0, int(self.orders[0])
        return rdp_to_epsilon(self._rdp, delta, self.orders)
