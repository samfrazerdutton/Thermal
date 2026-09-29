"""Optional AI explanation layer (docs/roadmap.md Phase 16).

Explains structured evidence THERMAL already computed deterministically --
it never invents a measurement, a root cause, or a confidence level of its
own. Every prompt is built entirely from a JSON evidence dict pulled
straight from a RunRecord/ExperimentRecord; the model is explicitly told to
use only those numbers and to say so when something isn't in the evidence.

THERMAL works fully without this: no API key, no internet, no LLM. Every
other subsystem (diagnosis, experiments, reports) is deterministic and does
not import this module.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Optional, Protocol

from thermal.storage import ExperimentRepository, SQLiteRunRepository, default_db_path


class AIUnavailableError(RuntimeError):
    pass


class ChatClient(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class AnthropicChatClient:
    """Thin wrapper so AIExplainer doesn't import the anthropic SDK directly --
    keeps that dependency confined to one place, and keeps AIExplainer testable
    with a fake ChatClient that makes no network call."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-5") -> None:
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def complete(self, system: str, user: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=600,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in response.content if getattr(block, "type", None) == "text")


SYSTEM_PROMPT = (
    "You are THERMAL's evidence explainer. You are given a JSON object of "
    "measurements, diagnosis, and/or experiment results that THERMAL already "
    "computed deterministically, before you were called. Explain it in plain "
    "engineering language for a performance engineer.\n\n"
    "Rules, followed strictly:\n"
    "1. Never state a number that is not present in the JSON.\n"
    "2. If answering well would need something the JSON doesn't contain, say "
    "so explicitly rather than guessing or estimating.\n"
    "3. Do not invent additional measurements, root causes, bottleneck "
    "classes, or confidence levels beyond what is given.\n"
    "4. Keep it to a few sentences of plain prose, no headers or bullet lists."
)


@dataclass
class AIExplainer:
    client: Optional[ChatClient] = None

    @classmethod
    def from_env(cls) -> "AIExplainer":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return cls(client=None)
        try:
            return cls(client=AnthropicChatClient(api_key=api_key))
        except ImportError:
            return cls(client=None)

    @property
    def available(self) -> bool:
        return self.client is not None

    def _require_client(self) -> ChatClient:
        if self.client is None:
            raise AIUnavailableError(
                "AI explanation layer is not configured -- set ANTHROPIC_API_KEY to enable it. "
                "Diagnosis, experiments, and reports all work fully without it."
            )
        return self.client

    def explain(self, evidence: dict[str, Any], question: str = "Explain this evidence.") -> str:
        client = self._require_client()
        user_prompt = f"{question}\n\nEvidence (JSON):\n{json.dumps(evidence, indent=2, default=str)}"
        return client.complete(SYSTEM_PROMPT, user_prompt)

    def explain_diagnosis(self, run) -> str:
        evidence = {
            "workload": run.workload_name,
            "configuration": run.configuration,
            "metrics": run.metrics,
            "diagnosis": run.diagnosis,
        }
        return self.explain(
            evidence,
            "Explain this run's performance diagnosis in plain language for an "
            "engineer who hasn't seen the raw data.",
        )

    def explain_experiment(self, experiment) -> str:
        evidence = {
            "workload": experiment.workload_name,
            "hypothesis": experiment.hypothesis,
            "baseline_config": experiment.baseline_config,
            "treatment_config": experiment.treatment_config,
            "comparison": experiment.comparison,
            "verdict": experiment.verdict,
        }
        return self.explain(
            evidence,
            "Explain this experiment's result in plain language for an engineer "
            "deciding whether to adopt the change.",
        )


def explain_record(explainer: AIExplainer, record_id: str) -> str:
    """Looks up record_id as a run, then an experiment (same order as
    thermal.report.generate_report) and explains whichever is found."""
    run = SQLiteRunRepository(default_db_path()).get(record_id)
    if run is not None:
        return explainer.explain_diagnosis(run)

    experiment = ExperimentRepository(default_db_path()).get(record_id)
    if experiment is not None:
        return explainer.explain_experiment(experiment)

    raise KeyError(f"no run or experiment found with id '{record_id}'")
