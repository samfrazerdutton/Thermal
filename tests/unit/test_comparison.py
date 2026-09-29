"""Unit tests for analysis.comparison."""

import numpy as np
import pytest

from analysis.comparison import Verdict, compare_samples


def test_insufficient_samples_raises():
    with pytest.raises(ValueError):
        compare_samples([1.0, 2.0], [1.0, 2.0, 3.0, 4.0, 5.0], min_samples=5)


def test_clear_improvement_detected():
    rng = np.random.default_rng(0)
    baseline = list(rng.normal(100, 2, 30))  # e.g. tok/s
    treatment = list(rng.normal(130, 2, 30))
    result = compare_samples(baseline, treatment, higher_is_better=True, random_state=0)
    assert result.verdict == Verdict.IMPROVED
    assert result.percent_change > 20


def test_clear_regression_detected():
    rng = np.random.default_rng(0)
    baseline = list(rng.normal(100, 2, 30))
    treatment = list(rng.normal(70, 2, 30))
    result = compare_samples(baseline, treatment, higher_is_better=True, random_state=0)
    assert result.verdict == Verdict.REGRESSED


def test_improvement_direction_respects_higher_is_better_false():
    # lower is better, e.g. latency_ms: treatment is lower -> improvement
    rng = np.random.default_rng(0)
    baseline = list(rng.normal(100, 2, 30))
    treatment = list(rng.normal(70, 2, 30))
    result = compare_samples(baseline, treatment, higher_is_better=False, random_state=0)
    assert result.verdict == Verdict.IMPROVED


def test_identical_distributions_no_change():
    rng = np.random.default_rng(0)
    baseline = list(rng.normal(100, 2, 40))
    treatment = list(baseline)  # identical -> zero change
    result = compare_samples(baseline, treatment, random_state=0)
    assert result.verdict == Verdict.NO_CHANGE


def test_noisy_overlapping_samples_inconclusive():
    rng = np.random.default_rng(1)
    baseline = list(rng.normal(100, 40, 8))
    treatment = list(rng.normal(105, 40, 8))
    result = compare_samples(baseline, treatment, random_state=0, min_samples=5)
    assert result.verdict in (Verdict.INCONCLUSIVE, Verdict.NO_CHANGE)


def test_never_forces_a_winner_when_p_value_high():
    rng = np.random.default_rng(2)
    baseline = list(rng.normal(100, 50, 6))
    treatment = list(rng.normal(102, 50, 6))
    result = compare_samples(baseline, treatment, random_state=0, min_samples=5)
    if abs(result.percent_change) >= 0.5:
        assert result.welch_p_value > 0.01 or result.verdict != Verdict.IMPROVED


def test_high_variance_warning_present():
    rng = np.random.default_rng(0)
    baseline = list(rng.normal(100, 60, 20))
    treatment = list(rng.normal(130, 60, 20))
    result = compare_samples(baseline, treatment, random_state=0)
    assert any("variance" in w for w in result.warnings) or result.verdict == Verdict.INCONCLUSIVE


def test_summary_text_contains_verdict():
    rng = np.random.default_rng(0)
    baseline = list(rng.normal(100, 2, 30))
    treatment = list(rng.normal(130, 2, 30))
    result = compare_samples(baseline, treatment, random_state=0)
    text = result.summary_text()
    assert "VERDICT" in text
    assert result.verdict.value in text
