# Security Policy

THERMAL executes workloads and, in server mode, exposes a web API. Treat both as
attack surface.

## Reporting a vulnerability

Please open a private security advisory on GitHub
(`https://github.com/samfrazerdutton/Thermal/security/advisories/new`) rather than a
public issue. Include reproduction steps and affected component (CLI, API, worker,
web console).

## Scope and design constraints

- The web UI and API must never execute arbitrary shell commands from a request body.
  Workload execution goes through a fixed, validated set of registered workload
  plugins — not arbitrary code paths.
- Workload processes run with timeouts and resource limits; a runaway workload must
  not be able to starve the host indefinitely.
- Server-mode deployments require authentication; local single-user mode does not
  expose a network-facing API by default.
- Secrets (database URLs, API keys for the optional AI layer) are read from
  environment variables, never committed.
- Dependencies are periodically scanned (see CI workflows once Phase 17 lands).

This policy will be expanded as the API surface (Phase 12) and server-mode
authentication (Phase 12/18) are implemented.
