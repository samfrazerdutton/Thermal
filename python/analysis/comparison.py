"""Statistical comparison between a baseline and a treatment: the machinery
behind "did it work?". Never reduces a comparison to a single percentage --
every ComparisonResult carries the test statistics, an effect size, and a
confidence interval on the change itself, and explicitly says INCONCLUSIVE
when the evidence doesn't support a verdict either way.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
from scipy import stats


class Verdict(str, Enum):
    IMPROVED = "IMPROVED"
    REGRESSED = "REGRESSED"
    INCONCLUSIVE = "INCONCLUSIVE"
    NO_CHANGE = "NO_CHANGE"


@dataclass(frozen=True)
class ComparisonResult:
    metric_name: str
    baseline_n: int
    treatment_n: int
    baseline_mean: float
    treatment_mean: float
    percent_change: float
    percent_change_ci_low: Optional[float]
    percent_change_ci_high: Optional[float]
    confidence_level: float
    welch_t_statistic: float
    welch_p_value: float
    mannwhitney_p_value: Optional[float]
    cohens_d: float
    higher_is_better: bool
    verdict: Verdict
    warnings: list[str]

    def summary_text(self) -> str:
        lines = [
            f"CONTROL\n{self.baseline_mean:.4g}",
            "",
            f"TREATMENT\n{self.treatment_mean:.4g}",
            "",
            f"CHANGE\n{self.percent_change:+.1f}%",
        ]
        if self.percent_change_ci_low is not None:
            lines.append(
                f"\n{self.confidence_level * 100:.0f}% CI\n"
                f"[{self.percent_change_ci_low:+.1f}%, {self.percent_change_ci_high:+.1f}%]"
            )
        lines.append(f"\nWelch's t-test p-value: {self.welch_p_value:.4g}")
        if self.mannwhitney_p_value is not None:
            lines.append(f"Mann-Whitney U p-value: {self.mannwhitney_p_value:.4g}")
        lines.append(f"effect size (Cohen's d): {self.cohens_d:.2f}")
        lines.append(f"\nn = {self.baseline_n} (control), {self.treatment_n} (treatment)")
        for warning in self.warnings:
            lines.append(f"WARNING: {warning}")
        lines.append(f"\nVERDICT:\n{self.verdict.value}")
        return "\n".join(lines)


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    n1, n2 = len(a), len(b)
    pooled_std = np.sqrt(((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1)) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(b) - np.mean(a)) / pooled_std)


def compare_samples(
    baseline: list[float],
    treatment: list[float],
    metric_name: str = "metric",
    higher_is_better: bool = True,
    confidence_level: float = 0.95,
    min_samples: int = 5,
    bootstrap_resamples: int = 2000,
    random_state: Optional[int] = None,
) -> ComparisonResult:
    warnings: list[str] = []

    if len(baseline) < min_samples or len(treatment) < min_samples:
        raise ValueError(
            f"{metric_name}: need at least {min_samples} samples per arm, "
            f"got {len(baseline)} (control) and {len(treatment)} (treatment)"
        )

    a = np.asarray(baseline, dtype=float)
    b = np.asarray(treatment, dtype=float)
    mean_a, mean_b = float(np.mean(a)), float(np.mean(b))

    percent_change = ((mean_b - mean_a) / mean_a * 100.0) if mean_a != 0 else float("inf")

    t_stat, t_p = stats.ttest_ind(a, b, equal_var=False)
    try:
        _, mw_p = stats.mannwhitneyu(a, b, alternative="two-sided")
    except ValueError:
        mw_p = None
        warnings.append("Mann-Whitney U undefined (identical distributions or too few unique values)")

    effect_size = _cohens_d(a, b)

    def _percent_change_stat(sample_a, sample_b):
        ma = np.mean(sample_a, axis=-1)
        mb = np.mean(sample_b, axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(ma != 0, (mb - ma) / ma * 100.0, np.nan)

    ci_low: Optional[float]
    ci_high: Optional[float]
    try:
        result = stats.bootstrap(
            (a, b),
            _percent_change_stat,
            confidence_level=confidence_level,
            n_resamples=bootstrap_resamples,
            method="basic",
            random_state=random_state,
            paired=False,
        )
        ci_low, ci_high = float(result.confidence_interval.low), float(result.confidence_interval.high)
    except Exception:
        ci_low, ci_high = None, None
        warnings.append("bootstrap confidence interval could not be computed for this sample")

    cv_a = np.std(a, ddof=1) / mean_a if mean_a != 0 else float("inf")
    cv_b = np.std(b, ddof=1) / mean_b if mean_b != 0 else float("inf")
    if cv_a > 0.10 or cv_b > 0.10:
        warnings.append("high variance in at least one arm (CV > 10%) -- treat this comparison cautiously")

    ci_excludes_zero = ci_low is not None and (ci_low > 0) == (ci_high > 0) and ci_low != 0 and ci_high != 0
    statistically_supported = t_p < (1 - confidence_level) and ci_excludes_zero

    direction_is_improvement = (percent_change > 0) == higher_is_better

    if abs(percent_change) < 0.5:
        verdict = Verdict.NO_CHANGE
    elif not statistically_supported:
        verdict = Verdict.INCONCLUSIVE
        warnings.append("change is not statistically well-supported (p-value or CI does not clearly exclude no-difference)")
    elif direction_is_improvement:
        verdict = Verdict.IMPROVED
    else:
        verdict = Verdict.REGRESSED

    return ComparisonResult(
        metric_name=metric_name,
        baseline_n=len(a),
        treatment_n=len(b),
        baseline_mean=mean_a,
        treatment_mean=mean_b,
        percent_change=percent_change,
        percent_change_ci_low=ci_low,
        percent_change_ci_high=ci_high,
        confidence_level=confidence_level,
        welch_t_statistic=float(t_stat),
        welch_p_value=float(t_p),
        mannwhitney_p_value=float(mw_p) if mw_p is not None else None,
        cohens_d=effect_size,
        higher_is_better=higher_is_better,
        verdict=verdict,
        warnings=warnings,
    )
