"""Markdown report generation (thermal report <id>).

Renders exactly what's stored for a run or experiment -- every section is a
direct projection of the RunRecord/ExperimentRecord fields already computed
by Phase 4/6/8, plus a literal, copy-pasteable command to reproduce it. A
report never adds a number that wasn't already in the record; a limitation
that applies is stated, not hidden.
"""

from __future__ import annotations

from thermal.storage import ExperimentRecord, ExperimentRepository, RunRecord, SQLiteRunRepository, default_db_path


def _reproduce_workload_command(run: RunRecord) -> str:
    params = " ".join(f"--param {k}={v}" for k, v in run.configuration.items())
    return (
        f"thermal workload run {run.workload_name} {params} "
        f"--samples {run.measurement_iterations} --warmup {run.warmup_iterations}"
    ).strip()


def generate_run_report(run: RunRecord) -> str:
    lines: list[str] = []
    lines.append(f"# THERMAL Report: {run.workload_name}")
    lines.append("")
    lines.append(f"Run `{run.run_id}`, generated from data recorded at run time -- nothing below was recomputed.")
    lines.append("")

    lines.append("## Hardware")
    lines.append("")
    for key, value in run.hardware_fingerprint.items():
        lines.append(f"- **{key.replace('_', ' ')}**: {value if value is not None else '_unavailable_'}")
    lines.append("")

    lines.append("## Software")
    lines.append("")
    lines.append(f"- **workload**: {run.workload_name} v{run.workload_version}")
    lines.append(f"- **git commit**: {run.git_commit or '_unknown (not run inside a git checkout)_'}")
    lines.append("")

    lines.append("## Configuration")
    lines.append("")
    lines.append("```json")
    import json

    lines.append(json.dumps(run.configuration, indent=2))
    lines.append("```")
    lines.append("")

    lines.append("## Baseline")
    lines.append("")
    if run.metrics:
        lines.append("| metric | n | mean | median | p95 | std | CV | 95% CI |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for name, stats in run.metrics.items():
            ci = (
                f"[{stats['ci_low']:.4g}, {stats['ci_high']:.4g}]"
                if stats.get("ci_low") is not None
                else "n/a"
            )
            lines.append(
                f"| {name} | {stats['n']} | {stats['mean']:.4g} | {stats['median']:.4g} | "
                f"{stats['p95']:.4g} | {stats['std']:.4g} | {stats['coefficient_of_variation'] * 100:.1f}% | {ci} |"
            )
    else:
        lines.append("_No metrics recorded._")
    lines.append("")

    lines.append("## Telemetry")
    lines.append("")
    lines.append(
        f"Raw samples: `{run.telemetry_path}`" if run.telemetry_path else "_No telemetry file recorded for this run._"
    )
    lines.append("")

    lines.append("## Diagnosis")
    lines.append("")
    if run.diagnosis:
        lines.append(f"**{run.diagnosis['bottleneck']}** (confidence: {run.diagnosis['confidence']:.0%})")
        lines.append("")
        lines.append(run.diagnosis["rationale"])
        lines.append("")
        if run.diagnosis["evidence"]:
            lines.append("Evidence:")
            for ev in run.diagnosis["evidence"]:
                lines.append(f"- {ev['feature']} = {ev['value']:.2f} ({ev['comparison']} {ev['threshold']})")
            lines.append("")
        if run.diagnosis["missing_features"]:
            lines.append(f"Not measurable on this run: {', '.join(run.diagnosis['missing_features'])}")
            lines.append("")
    else:
        lines.append("_No diagnosis recorded._")
        lines.append("")

    lines.append("## Limitations")
    lines.append("")
    high_variance = [name for name, stats in run.metrics.items() if stats.get("high_variance_warning")]
    if high_variance:
        lines.append(f"- High variance (CV > 10%) in: {', '.join(high_variance)} -- treat those baselines cautiously.")
    if run.diagnosis and run.diagnosis["missing_features"]:
        lines.append(
            f"- The bottleneck classifier could not measure: {', '.join(run.diagnosis['missing_features'])} "
            "on this backend; its confidence reflects that."
        )
    if not high_variance and not (run.diagnosis and run.diagnosis["missing_features"]):
        lines.append("- None recorded.")
    lines.append("")

    lines.append("## Reproduction")
    lines.append("")
    lines.append("```bash")
    lines.append(_reproduce_workload_command(run))
    lines.append("```")

    return "\n".join(lines)


def _reproduce_experiment_command(exp: ExperimentRecord) -> str:
    baseline = " ".join(f"--baseline-param {k}={v}" for k, v in exp.baseline_config.items())
    treatment = " ".join(f"--treatment-param {k}={v}" for k, v in exp.treatment_config.items())
    direction = "" if exp.higher_is_better else " --lower-is-better"
    return (
        f"thermal experiment run {exp.workload_name} {baseline} {treatment} "
        f"--metric {exp.metric_name} --repetitions {exp.repetitions}{direction}"
    ).strip()


def generate_experiment_report(exp: ExperimentRecord) -> str:
    lines: list[str] = []
    lines.append(f"# THERMAL Report: experiment on {exp.workload_name}")
    lines.append("")
    lines.append(f"Experiment `{exp.experiment_id}`.")
    lines.append("")

    if exp.hypothesis:
        lines.append("## Hypothesis")
        lines.append("")
        lines.append(exp.hypothesis)
        lines.append("")

    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- **baseline**: `{exp.baseline_config}` (device: {exp.baseline_device})")
    lines.append(f"- **treatment**: `{exp.treatment_config}` (device: {exp.treatment_device})")
    lines.append(f"- **git commit**: {exp.git_commit or '_unknown_'}")
    lines.append("")

    cmp = exp.comparison
    lines.append("## Results")
    lines.append("")
    lines.append(f"- **control**: {cmp['baseline_mean']:.4g} (n={cmp['baseline_n']})")
    lines.append(f"- **treatment**: {cmp['treatment_mean']:.4g} (n={cmp['treatment_n']})")
    lines.append(f"- **change**: {cmp['percent_change']:+.1f}%")
    if cmp.get("percent_change_ci_low") is not None:
        lines.append(
            f"- **{cmp['confidence_level'] * 100:.0f}% CI**: "
            f"[{cmp['percent_change_ci_low']:+.1f}%, {cmp['percent_change_ci_high']:+.1f}%]"
        )
    lines.append("")

    lines.append("## Statistics")
    lines.append("")
    lines.append(f"- Welch's t-test p-value: {cmp['welch_p_value']:.4g}")
    if cmp.get("mannwhitney_p_value") is not None:
        lines.append(f"- Mann-Whitney U p-value: {cmp['mannwhitney_p_value']:.4g}")
    lines.append(f"- Effect size (Cohen's d): {cmp['cohens_d']:.2f}")
    lines.append("")
    lines.append(f"**Verdict: {exp.verdict}**")
    lines.append("")

    lines.append("## Limitations")
    lines.append("")
    if cmp.get("warnings"):
        for w in cmp["warnings"]:
            lines.append(f"- {w}")
    else:
        lines.append("- None recorded.")
    lines.append("")

    lines.append("## Reproduction")
    lines.append("")
    lines.append("```bash")
    lines.append(_reproduce_experiment_command(exp))
    lines.append("```")

    return "\n".join(lines)


def generate_report(record_id: str) -> str:
    """Looks up record_id as a run first, then an experiment. Raises KeyError
    if it's neither -- a report is never generated from nothing."""
    run = SQLiteRunRepository(default_db_path()).get(record_id)
    if run is not None:
        return generate_run_report(run)

    experiment = ExperimentRepository(default_db_path()).get(record_id)
    if experiment is not None:
        return generate_experiment_report(experiment)

    raise KeyError(f"no run or experiment found with id '{record_id}'")
