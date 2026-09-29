"""Unit tests for thermal.report (Markdown report generation)."""

import pytest

from thermal.report import generate_experiment_report, generate_report, generate_run_report
from thermal.storage import ExperimentRecord, RunRecord


def _make_run(**overrides):
    defaults = dict(
        workload_name="matmul",
        workload_version="1.0.0",
        device="cpu",
        warmup_iterations=3,
        measurement_iterations=10,
        configuration={"size": 256},
        hardware_fingerprint={"cpu_name": "test-cpu", "gpu_name": None},
        metrics={
            "gflops": {
                "metric_name": "gflops", "n": 10, "mean": 100.0, "median": 99.0, "std": 5.0,
                "coefficient_of_variation": 0.05, "p50": 99.0, "p95": 108.0, "p99": 110.0,
                "min": 90.0, "max": 110.0, "ci_low": 95.0, "ci_high": 105.0, "confidence_level": 0.95,
                "outlier_indices": [], "outlier_method": "IQR (1.5x)", "high_variance_warning": False,
            }
        },
        git_commit="abc1234",
        diagnosis={
            "bottleneck": "MEMORY_BOUND",
            "confidence": 0.9,
            "rationale": "HBM bandwidth saturated.",
            "evidence": [{"feature": "gpu_memory_bandwidth_utilization_percent", "value": 91.0, "threshold": 80.0, "comparison": ">="}],
            "missing_features": [],
        },
        telemetry_path="/tmp/run.jsonl",
    )
    defaults.update(overrides)
    return RunRecord.new(**defaults)


def _make_experiment(**overrides):
    defaults = dict(
        workload_name="matmul",
        metric_name="gflops",
        higher_is_better=True,
        repetitions=15,
        baseline_config={"size": 256},
        treatment_config={"size": 512},
        baseline_values=[100.0] * 15,
        treatment_values=[150.0] * 15,
        baseline_device="cpu",
        treatment_device="cpu",
        comparison={
            "baseline_mean": 100.0, "treatment_mean": 150.0, "percent_change": 50.0,
            "percent_change_ci_low": 40.0, "percent_change_ci_high": 60.0, "confidence_level": 0.95,
            "welch_p_value": 0.0001, "mannwhitney_p_value": 0.0002, "cohens_d": 2.1,
            "baseline_n": 15, "treatment_n": 15, "warnings": [],
        },
        verdict="IMPROVED",
        hypothesis="Bigger is faster.",
        git_commit="abc1234",
    )
    defaults.update(overrides)
    return ExperimentRecord.new(**defaults)


def test_run_report_includes_key_sections():
    report = generate_run_report(_make_run())
    for heading in ("## Hardware", "## Software", "## Configuration", "## Baseline", "## Telemetry", "## Diagnosis", "## Limitations", "## Reproduction"):
        assert heading in report


def test_run_report_includes_real_metric_values():
    report = generate_run_report(_make_run())
    assert "100" in report  # mean gflops
    assert "MEMORY_BOUND" in report


def test_run_report_reproduction_command_is_runnable_shape():
    report = generate_run_report(_make_run())
    assert "thermal workload run matmul --param size=256 --samples 10 --warmup 3" in report


def test_run_report_flags_high_variance_limitation():
    run = _make_run()
    run.metrics["gflops"]["high_variance_warning"] = True
    report = generate_run_report(run)
    assert "High variance" in report
    assert "gflops" in report.split("## Limitations")[1].split("## Reproduction")[0]


def test_run_report_no_diagnosis_says_so():
    run = _make_run(diagnosis=None)
    report = generate_run_report(run)
    assert "No diagnosis recorded" in report


def test_experiment_report_includes_key_sections():
    report = generate_experiment_report(_make_experiment())
    for heading in ("## Hypothesis", "## Configuration", "## Results", "## Statistics", "## Limitations", "## Reproduction"):
        assert heading in report


def test_experiment_report_includes_verdict_and_stats():
    report = generate_experiment_report(_make_experiment())
    assert "IMPROVED" in report
    assert "+50.0%" in report


def test_experiment_report_reproduction_command_is_runnable_shape():
    report = generate_experiment_report(_make_experiment())
    assert "thermal experiment run matmul" in report
    assert "--baseline-param size=256" in report
    assert "--treatment-param size=512" in report
    assert "--metric gflops" in report


def test_generate_report_raises_for_unknown_id(tmp_path, monkeypatch):
    import thermal.report as report_mod

    monkeypatch.setattr(report_mod, "default_db_path", lambda: tmp_path / "thermal.sqlite3")
    with pytest.raises(KeyError):
        generate_report("does-not-exist")


def test_generate_report_finds_run_by_id(tmp_path, monkeypatch):
    import thermal.report as report_mod
    from thermal.storage import SQLiteRunRepository

    db_path = tmp_path / "thermal.sqlite3"
    monkeypatch.setattr(report_mod, "default_db_path", lambda: db_path)

    run = _make_run()
    SQLiteRunRepository(db_path).save(run)

    report = generate_report(run.run_id)
    assert "# THERMAL Report: matmul" in report


def test_generate_report_finds_experiment_when_not_a_run(tmp_path, monkeypatch):
    import thermal.report as report_mod
    from thermal.storage import ExperimentRepository

    db_path = tmp_path / "thermal.sqlite3"
    monkeypatch.setattr(report_mod, "default_db_path", lambda: db_path)

    exp = _make_experiment()
    ExperimentRepository(db_path).save(exp)

    report = generate_report(exp.experiment_id)
    assert "experiment on matmul" in report
