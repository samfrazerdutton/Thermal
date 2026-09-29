"""Pydantic request/response schemas for the THERMAL API.

Every request body is validated here -- the API never accepts an arbitrary
command or free-form code, only a workload name (checked against the
registry, itself a fixed set of trusted plugins) plus a bounded parameter
dict. See SECURITY.md.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class WorkloadRunRequest(BaseModel):
    workload_name: str
    params: dict[str, Any] = Field(default_factory=dict)
    samples: Optional[int] = Field(default=None, ge=1, le=1000)
    warmup: Optional[int] = Field(default=None, ge=0, le=1000)


class ExperimentRunRequest(BaseModel):
    workload_name: str
    baseline_params: dict[str, Any] = Field(default_factory=dict)
    treatment_params: dict[str, Any] = Field(default_factory=dict)
    metric_name: str
    higher_is_better: bool = True
    repetitions: int = Field(default=15, ge=5, le=500)
    warmup_iterations: int = Field(default=3, ge=0, le=500)
    hypothesis: str = ""


class OptimizeGridSearchRequest(BaseModel):
    workload_name: str
    baseline_params: dict[str, Any] = Field(default_factory=dict)
    param_name: str
    candidate_values: list[Any]
    metric_name: str
    higher_is_better: bool = True
    repetitions: int = Field(default=10, ge=5, le=200)
    warmup_iterations: int = Field(default=2, ge=0, le=200)
