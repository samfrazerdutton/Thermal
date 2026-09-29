import { EmptyState } from "../components/AsyncState";

export function Genome() {
  return (
    <div className="flex flex-col gap-6">
      <h1 className="text-lg text-[var(--color-text)]">Genome</h1>
      <EmptyState
        title="Not implemented"
        detail="Workload fingerprinting and similarity search (docs/roadmap.md, spec section 23) hasn't been built yet — this page is not filled in with placeholder data."
      />
    </div>
  );
}
