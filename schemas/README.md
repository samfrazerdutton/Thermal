# schemas/

Versioned JSON Schema definitions for every record THERMAL persists: telemetry
samples, run manifests, experiments, hypotheses, diagnoses, and reports.

Rule: a schema is never edited in place once anything has been recorded against it.
A change adds a new `schema_version` and, if needed, a migration. This directory is
populated starting in Phase 2 (telemetry trace schema) and Phase 5 (storage schema).
