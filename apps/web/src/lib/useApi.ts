import { useEffect, useState } from "react";
import { ApiError } from "./api";

type State<T> = { status: "loading" } | { status: "error"; message: string } | { status: "ready"; data: T };

export function useApi<T>(fetcher: () => Promise<T>, deps: unknown[]): State<T> & { reload: () => void } {
  const [state, setState] = useState<State<T>>({ status: "loading" });
  const [tick, setTick] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setState({ status: "loading" });
    fetcher()
      .then((data) => {
        if (!cancelled) setState({ status: "ready", data });
      })
      .catch((err) => {
        if (!cancelled) {
          setState({ status: "error", message: err instanceof ApiError ? err.message : String(err) });
        }
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, tick]);

  return { ...state, reload: () => setTick((t) => t + 1) } as State<T> & { reload: () => void };
}
