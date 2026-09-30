// Thin fetch client for the THERMAL API (services/api). No mock data path --
// if a request fails, callers show the real error rather than a placeholder.

const BASE = "/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new ApiError(response.status, body.detail ?? response.statusText);
  }
  return response.json();
}

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

export interface Capability {
  name: string;
  available: boolean;
  value: unknown;
  reason: string | null;
}

export interface HardwareReport {
  cpu: {
    name: string;
    physical_cores: number | null;
    logical_cores: number | null;
    max_frequency_mhz: number | null;
    current_utilization_percent: number | null;
  };
  gpu: {
    available: boolean;
    name: string | null;
    driver_version: string | null;
    cuda_driver_version: string | null;
    memory_total_mb: number | null;
    memory_used_mb: number | null;
    compute_capability: string | null;
    reason: string | null;
    telemetry: Record<string, Capability>;
  };
  toolchain: Record<string, Capability>;
  software: {
    os_name: string;
    os_version: string;
    python_version: string;
    torch_installed: Capability;
    torch_cuda: Capability;
  };
}

export interface WorkloadSpec {
  name: string;
  version: string;
  description: string;
  default_parameters: Record<string, unknown>;
  warmup_iterations: number;
  measurement_iterations: number;
  output_metrics: string[];
}

export interface BaselineStats {
  metric_name: string;
  n: number;
  mean: number;
  median: number;
  std: number;
  coefficient_of_variation: number;
  p50: number;
  p95: number;
  p99: number;
  min: number;
  max: number;
  ci_low: number | null;
  ci_high: number | null;
  confidence_level: number;
  outlier_indices: number[];
  outlier_method: string;
  high_variance_warning: boolean;
}

export interface Evidence {
  feature: string;
  value: number;
  threshold: number;
  comparison: string;
}

export interface Diagnosis {
  bottleneck: string;
  confidence: number;
  rationale: string;
  evidence: Evidence[];
  missing_features: string[];
}

export interface RunRecord {
  run_id: string;
  created_at_ns: number;
  git_commit: string | null;
  workload_name: string;
  workload_version: string;
  device: string;
  warmup_iterations: number;
  measurement_iterations: number;
  configuration: Record<string, unknown>;
  hardware_fingerprint: Record<string, unknown>;
  metrics: Record<string, BaselineStats>;
  telemetry_path: string | null;
  diagnosis: Diagnosis | null;
}

export interface ComparisonResult {
  metric_name: string;
  baseline_n: number;
  treatment_n: number;
  baseline_mean: number;
  treatment_mean: number;
  percent_change: number;
  percent_change_ci_low: number | null;
  percent_change_ci_high: number | null;
  confidence_level: number;
  welch_t_statistic: number;
  welch_p_value: number;
  mannwhitney_p_value: number | null;
  cohens_d: number;
  higher_is_better: boolean;
  verdict: "IMPROVED" | "REGRESSED" | "INCONCLUSIVE" | "NO_CHANGE";
  warnings: string[];
}

export interface ExperimentRecord {
  experiment_id: string;
  created_at_ns: number;
  git_commit: string | null;
  workload_name: string;
  hypothesis: string;
  metric_name: string;
  higher_is_better: boolean;
  repetitions: number;
  baseline_config: Record<string, unknown>;
  treatment_config: Record<string, unknown>;
  baseline_values: number[];
  treatment_values: number[];
  baseline_device: string;
  treatment_device: string;
  comparison: ComparisonResult;
  verdict: string;
}

export interface CausalEdge {
  source: string;
  target: string;
  relationship: "positive" | "negative" | "unknown";
  evidence: {
    kind: string;
    experiment_id: string | null;
    percent_change: number | null;
    confidence_level: number | null;
    note: string;
  };
}

export interface JobRecord {
  job_id: string;
  kind: string;
  status: "pending" | "running" | "completed" | "failed";
  created_at_ns: number;
  started_at_ns: number | null;
  finished_at_ns: number | null;
  request: Record<string, unknown>;
  result: Record<string, unknown> | null;
  error: string | null;
}

export const api = {
  health: () => request<{ status: string }>("/health"),
  hardware: () => request<HardwareReport>("/hardware"),
  workloads: () => request<WorkloadSpec[]>("/workloads"),
  runs: (params?: { workload?: string; limit?: number }) =>
    request<RunRecord[]>(`/runs${toQuery(params)}`),
  run: (id: string) => request<RunRecord>(`/runs/${id}`),
  runWorkload: (body: { workload_name: string; params?: Record<string, unknown>; samples?: number; warmup?: number }) =>
    request<{ run: RunRecord; diagnosis: Diagnosis }>("/workloads/run", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  experiments: (params?: { workload?: string; limit?: number }) =>
    request<ExperimentRecord[]>(`/experiments${toQuery(params)}`),
  experiment: (id: string) => request<ExperimentRecord>(`/experiments/${id}`),
  causalGraph: (workload?: string) =>
    request<{ edges: CausalEdge[] }>(`/causal-graph${toQuery({ workload })}`),
  jobs: (params?: { status?: string; limit?: number }) => request<JobRecord[]>(`/jobs${toQuery(params)}`),
  job: (id: string) => request<JobRecord>(`/jobs/${id}`),
  submitWorkloadJob: (body: { workload_name: string; params?: Record<string, unknown>; samples?: number; warmup?: number }) =>
    request<{ job_id: string; status: string }>("/jobs/workloads/run", {
      method: "POST",
      body: JSON.stringify(body),
    }),
};

function toQuery(params?: Record<string, unknown>): string {
  if (!params) return "";
  const entries = Object.entries(params).filter(([, v]) => v !== undefined && v !== null && v !== "");
  if (entries.length === 0) return "";
  return "?" + new URLSearchParams(entries as [string, string][]).toString();
}
