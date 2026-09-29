"""Unit tests for analysis.baseline."""

import numpy as np
import pytest

from analysis.baseline import InsufficientSamplesError, compute_baseline


def test_insufficient_samples_raises():
    with pytest.raises(InsufficientSamplesError):
        compute_baseline([1.0, 2.0], min_samples=5)


def test_constant_values_zero_variance():
    baseline = compute_baseline([42.0] * 10, min_samples=5)
    assert baseline.std == 0
    assert baseline.coefficient_of_variation == 0
    assert baseline.high_variance_warning is False


def test_mean_and_median_are_correct():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    baseline = compute_baseline(values, min_samples=5)
    assert baseline.mean == pytest.approx(3.0)
    assert baseline.median == pytest.approx(3.0)
    assert baseline.n == 5


def test_high_variance_flagged():
    rng = np.random.default_rng(0)
    values = list(rng.normal(100, 50, 30))  # CV ~= 50%
    baseline = compute_baseline(values, min_samples=5)
    assert baseline.high_variance_warning is True


def test_low_variance_not_flagged():
    rng = np.random.default_rng(0)
    values = list(rng.normal(100, 1, 30))  # CV ~= 1%
    baseline = compute_baseline(values, min_samples=5)
    assert baseline.high_variance_warning is False


def test_confidence_interval_contains_mean_for_normal_data():
    rng = np.random.default_rng(0)
    values = list(rng.normal(100, 5, 200))
    baseline = compute_baseline(values, min_samples=5, random_state=0)
    assert baseline.ci_low < baseline.mean < baseline.ci_high


def test_outlier_detected_with_iqr():
    values = [10.0, 11.0, 9.0, 10.5, 9.5, 10.2, 9.8, 100.0]  # 100.0 is a clear outlier
    baseline = compute_baseline(values, min_samples=5)
    assert len(baseline.outlier_indices) >= 1
    assert 7 in baseline.outlier_indices


def test_summary_text_includes_warnings_when_present():
    rng = np.random.default_rng(0)
    values = list(rng.normal(100, 50, 30))
    baseline = compute_baseline(values, min_samples=5)
    text = baseline.summary_text()
    assert "WARNING" in text
    assert f"n = {baseline.n}" in text
