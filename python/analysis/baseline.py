"""Baseline statistics: turn N noisy repeated measurements into an honest
summary, with outlier detection, dispersion, and a bootstrap confidence
interval on the mean. Never compares a single sample to another single
sample and calls it a result -- see docs/roadmap.md Phase 4/13.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats


class InsufficientSamplesError(ValueError):
    """Raised when there are too few samples to compute a meaningful baseline."""


@dataclass(frozen=True)
class BaselineStats:
    metric_name: str
    n: int
    mean: float
    median: float
    std: float
    coefficient_of_variation: float
    p50: float
    p95: float
    p99: float
    min: float
    max: float
    ci_low: Optional[float]
    ci_high: Optional[float]
    confidence_level: float
    outlier_indices: list[int]
    outlier_method: str
    high_variance_warning: bool

    def summary_text(self) -> str:
        lines = [
            self.metric_name.upper(),
            f"mean:   {self.mean:.4g}",
            f"median: {self.median:.4g}",
            f"p95:    {self.p95:.4g}",
            f"std:    {self.std:.4g}",
            f"CV:     {self.coefficient_of_variation * 100:.1f}%",
        ]
        if self.ci_low is not None:
            lines.append(f"{self.confidence_level * 100:.0f}% CI: [{self.ci_low:.4g}, {self.ci_high:.4g}]")
        lines.append(f"n = {self.n}")
        if self.outlier_indices:
            lines.append(f"outliers ({self.outlier_method}): {len(self.outlier_indices)} of {self.n}")
        if self.high_variance_warning:
            lines.append("WARNING: high variance (CV > 10%) -- treat this baseline cautiously")
        return "\n".join(lines)


def _iqr_outliers(values: np.ndarray) -> list[int]:
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    if iqr == 0:
        return []
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return [i for i, v in enumerate(values) if v < low or v > high]


def compute_baseline(
    values: list[float],
    metric_name: str = "metric",
    confidence_level: float = 0.95,
    min_samples: int = 5,
    bootstrap_resamples: int = 2000,
    random_state: Optional[int] = None,
) -> BaselineStats:
    if len(values) < min_samples:
        raise InsufficientSamplesError(
            f"{metric_name}: need at least {min_samples} samples for a baseline, got {len(values)}"
        )

    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))
    cv = std / mean if mean != 0 else float("inf")

    ci_low: Optional[float]
    ci_high: Optional[float]
    try:
        result = stats.bootstrap(
            (arr,),
            np.mean,
            confidence_level=confidence_level,
            n_resamples=bootstrap_resamples,
            method="basic",
            random_state=random_state,
        )
        ci_low, ci_high = float(result.confidence_interval.low), float(result.confidence_interval.high)
    except Exception:
        # e.g. degenerate/zero-variance sample -- report honestly rather than crash
        ci_low, ci_high = None, None

    return BaselineStats(
        metric_name=metric_name,
        n=len(arr),
        mean=mean,
        median=float(np.median(arr)),
        std=std,
        coefficient_of_variation=cv,
        p50=float(np.percentile(arr, 50)),
        p95=float(np.percentile(arr, 95)),
        p99=float(np.percentile(arr, 99)),
        min=float(np.min(arr)),
        max=float(np.max(arr)),
        ci_low=ci_low,
        ci_high=ci_high,
        confidence_level=confidence_level,
        outlier_indices=_iqr_outliers(arr),
        outlier_method="IQR (1.5x)",
        high_variance_warning=cv > 0.10,
    )
