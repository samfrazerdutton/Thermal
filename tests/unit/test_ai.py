"""Unit tests for thermal.ai (optional AI explanation layer).

No test here makes a real network call or requires an API key -- the
"unavailable without a key" path is exercised for real (matching this
machine's actual environment), and the "available" path is exercised with a
fake ChatClient that records what it was asked rather than calling out to
Anthropic.
"""

import json

import pytest

from thermal.ai import AIExplainer, AIUnavailableError, SYSTEM_PROMPT, explain_record
from thermal.storage import ExperimentRecord, RunRecord


class _FakeChatClient:
    def __init__(self, reply: str = "This run is memory-bound based on the evidence provided.") -> None:
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.reply


def _make_run(**overrides):
    defaults = dict(
        workload_name="matmul",
        workload_version="1.0.0",
        device="cpu",
        warmup_iterations=3,
        measurement_iterations=10,
        configuration={"size": 256},
        hardware_fingerprint={"cpu_name": "test-cpu"},
        metrics={"gflops": {"mean": 100.0, "n": 10}},
        diagnosis={"bottleneck": "MEMORY_BOUND", "confidence": 0.9, "rationale": "HBM saturated.", "evidence": [], "missing_features": []},
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
        comparison={"percent_change": 50.0, "welch_p_value": 0.0001},
        verdict="IMPROVED",
        hypothesis="Bigger is faster.",
    )
    defaults.update(overrides)
    return ExperimentRecord.new(**defaults)


def test_from_env_unavailable_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    explainer = AIExplainer.from_env()
    assert explainer.available is False


def test_explain_raises_when_unavailable():
    explainer = AIExplainer(client=None)
    with pytest.raises(AIUnavailableError):
        explainer.explain({"foo": "bar"})


def test_explain_diagnosis_never_sends_data_outside_the_record():
    fake = _FakeChatClient()
    explainer = AIExplainer(client=fake)
    run = _make_run()

    result = explainer.explain_diagnosis(run)

    assert result == fake.reply
    assert len(fake.calls) == 1
    system, user = fake.calls[0]
    assert system == SYSTEM_PROMPT
    # every fact the model could cite must actually be present in the sent evidence
    sent_evidence = json.loads(user.split("Evidence (JSON):\n", 1)[1])
    assert sent_evidence["diagnosis"]["bottleneck"] == "MEMORY_BOUND"
    assert sent_evidence["metrics"]["gflops"]["mean"] == 100.0
    assert sent_evidence["configuration"] == {"size": 256}


def test_explain_experiment_includes_verdict_and_hypothesis():
    fake = _FakeChatClient()
    explainer = AIExplainer(client=fake)
    exp = _make_experiment()

    explainer.explain_experiment(exp)

    _, user = fake.calls[0]
    sent_evidence = json.loads(user.split("Evidence (JSON):\n", 1)[1])
    assert sent_evidence["verdict"] == "IMPROVED"
    assert sent_evidence["hypothesis"] == "Bigger is faster."


def test_system_prompt_forbids_fabrication():
    lowered = SYSTEM_PROMPT.lower()
    assert "never state a number" in lowered or "not present in the json" in lowered


def test_explain_record_finds_run(tmp_path, monkeypatch):
    import thermal.ai as ai_mod
    from thermal.storage import SQLiteRunRepository

    db_path = tmp_path / "thermal.sqlite3"
    monkeypatch.setattr(ai_mod, "default_db_path", lambda: db_path)

    run = _make_run()
    SQLiteRunRepository(db_path).save(run)

    fake = _FakeChatClient(reply="explained the run")
    explainer = AIExplainer(client=fake)
    result = explain_record(explainer, run.run_id)
    assert result == "explained the run"


def test_explain_record_finds_experiment_when_not_a_run(tmp_path, monkeypatch):
    import thermal.ai as ai_mod
    from thermal.storage import ExperimentRepository

    db_path = tmp_path / "thermal.sqlite3"
    monkeypatch.setattr(ai_mod, "default_db_path", lambda: db_path)

    exp = _make_experiment()
    ExperimentRepository(db_path).save(exp)

    fake = _FakeChatClient(reply="explained the experiment")
    explainer = AIExplainer(client=fake)
    result = explain_record(explainer, exp.experiment_id)
    assert result == "explained the experiment"


def test_explain_record_raises_for_unknown_id(tmp_path, monkeypatch):
    import thermal.ai as ai_mod

    monkeypatch.setattr(ai_mod, "default_db_path", lambda: tmp_path / "thermal.sqlite3")
    explainer = AIExplainer(client=_FakeChatClient())
    with pytest.raises(KeyError):
        explain_record(explainer, "does-not-exist")
