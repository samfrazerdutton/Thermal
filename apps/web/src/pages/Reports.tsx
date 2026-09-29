import { EmptyState } from "../components/AsyncState";

export function Reports() {
  return (
    <div className="flex flex-col gap-6">
      <h1 className="text-lg text-[var(--color-text)]">Reports</h1>
      <EmptyState
        title="Not implemented"
        detail="thermal report <run_id> (docs/roadmap.md Phase 15) hasn't been built yet. The API's /api/reports/:id currently returns 501 for the same reason."
      />
    </div>
  );
}
