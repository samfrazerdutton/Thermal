import { useRef, useState } from "react";

export interface StreamEvent {
  event: string;
  [key: string]: unknown;
}

type StreamStatus = "idle" | "connecting" | "running" | "done" | "error";

export function useWorkloadRunStream() {
  const [status, setStatus] = useState<StreamStatus>("idle");
  const [events, setEvents] = useState<StreamEvent[]>([]);
  const [result, setResult] = useState<{ run_id: string; diagnosis: string } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const socketRef = useRef<WebSocket | null>(null);

  function start(body: { workload_name: string; params?: Record<string, unknown>; samples?: number; warmup?: number }) {
    setStatus("connecting");
    setEvents([]);
    setResult(null);
    setError(null);

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const socket = new WebSocket(`${protocol}//${window.location.host}/api/ws/workloads/run`);
    socketRef.current = socket;

    socket.onopen = () => {
      setStatus("running");
      socket.send(JSON.stringify(body));
    };

    socket.onmessage = (raw) => {
      const msg: StreamEvent = JSON.parse(raw.data);
      if (msg.event === "result") {
        const run = msg.run as { run_id: string };
        const diagnosis = msg.diagnosis as { bottleneck: string };
        setResult({ run_id: run.run_id, diagnosis: diagnosis.bottleneck });
        setStatus("done");
      } else if (msg.event === "error") {
        setError(String(msg.detail));
        setStatus("error");
      } else {
        setEvents((prev) => [...prev, msg]);
      }
    };

    socket.onerror = () => {
      setError("connection error");
      setStatus("error");
    };

    socket.onclose = () => {
      // A clean finish already set status to "done"/"error" via onmessage
      // before the server closes the socket -- if we're still "connecting"
      // or "running" here, the connection dropped (network issue, server
      // restart, crash) without ever telling us the run's outcome. Surface
      // that instead of leaving the UI stuck showing "running" forever.
      setStatus((current) => {
        if (current === "connecting" || current === "running") {
          setError("Connection to the server was lost before the run finished.");
          return "error";
        }
        return current;
      });
    };
  }

  return { status, events, result, error, start };
}
