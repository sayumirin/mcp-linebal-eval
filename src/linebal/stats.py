"""Small statistics helpers: Wilson intervals, the unbiased pass^k estimator, exact McNemar and a
paired bootstrap over tasks."""

from __future__ import annotations

import math
import random
from math import comb


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float, float]:
    """Rate and 95% Wilson score interval."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, centre - half), min(1.0, centre + half)


def pass_hat_k(c: int, n: int, k: int) -> float:
    """tau-bench's unbiased estimate of P(all k i.i.d. trials succeed) from c successes in n trials."""
    if k > n:
        raise ValueError("k must not exceed n")
    return comb(c, k) / comb(n, k)


def mean_pass_hat_k(per_task: list[tuple[int, int]], k: int) -> float:
    vals = [pass_hat_k(c, n, k) for c, n in per_task if n >= k]
    return sum(vals) / len(vals) if vals else float("nan")


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts b (only A succeeds) and c (only B)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * p)


def paired_bootstrap(diffs: list[float], iters: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of per-task differences with a 95% percentile bootstrap interval (tasks resampled)."""
    if not diffs:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return sum(diffs) / n, means[int(0.025 * iters)], means[int(0.975 * iters) - 1]


def median_iqr(xs: list[float]) -> tuple[float, float, float]:
    if not xs:
        return float("nan"), float("nan"), float("nan")
    s = sorted(xs)

    def q(p: float) -> float:
        pos = (len(s) - 1) * p
        lo, hi = math.floor(pos), math.ceil(pos)
        return s[lo] + (s[hi] - s[lo]) * (pos - lo)

    return q(0.5), q(0.25), q(0.75)
