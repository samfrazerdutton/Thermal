# apps/web

THERMAL's engineering console: React 19, TypeScript, Vite, Tailwind CSS v4, Recharts.

Design: a dark, blue-slate instrument-panel aesthetic (IBM Plex Sans for UI, IBM Plex
Mono for every measured value), not a generic SaaS dashboard. Color is reserved for
real states — NVIDIA green for a verified improvement, amber for caution/high
variance, red for a regression, blue for interactive elements — never decoration. See
`src/index.css` for the token set.

Every page fetches from the real `services/api` (Phase 12) — there is no mock data
path. A page with nothing to show says so explicitly (see `src/components/AsyncState.tsx`)
rather than rendering a placeholder as if it were data.

## Run it

```bash
# terminal 1: the API
thermal serve

# terminal 2: the web console
cd apps/web
npm install
npm run dev
```

Open the printed local URL. The Vite dev server proxies `/api/*` to
`http://127.0.0.1:8000` by default; set `THERMAL_API_PORT` if you're running the API
on a different port (see `vite.config.ts`).

## Routes

| Route | Status |
|---|---|
| `/` | Overview — the latest run/experiment rendered as the evidence chain (workload → baseline → observation → hypothesis → intervention → experiment → measurement → statistics → verification) |
| `/runs`, `/runs/:id` | Stored runs and full baseline + diagnosis detail |
| `/diagnosis/:id` | Same detail, diagnosis-first |
| `/experiments`, `/experiments/:id` | Stored experiments; detail view includes the raw per-trial scatter plot, not just the headline percentage |
| `/jobs` | Background jobs submitted via `POST /api/jobs/...` (the async, poll-for-status alternative to the blocking and WebSocket-streaming run paths) |
| `/workloads` | Registered workload plugins and their parameters; includes a live "run" button that streams a run over `/api/ws/workloads/run` |
| `/hardware` | Live hardware/toolchain capabilities from `/api/hardware` |
| `/benchmarks` | Runs grouped by workload (full cross-GPU/optimization comparison is Phase 33) |
| `/reports` | Enter a run or experiment id to generate its Markdown report (`GET /api/reports/:id`) |
| `/genome` | Honest "not implemented" page (Phase 23) |
| `/settings` | Local-mode info; nothing configurable yet |

## Build

```bash
npm run build   # tsc -b && vite build
npm run lint    # oxlint
```
