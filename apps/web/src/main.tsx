import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import "./index.css";
import { Shell } from "./components/Shell";
import { Benchmarks } from "./pages/Benchmarks";
import { Dashboard } from "./pages/Dashboard";
import { ExperimentDetail } from "./pages/ExperimentDetail";
import { Experiments } from "./pages/Experiments";
import { Genome } from "./pages/Genome";
import { Hardware } from "./pages/Hardware";
import { Jobs } from "./pages/Jobs";
import { Reports } from "./pages/Reports";
import { RunDetail } from "./pages/RunDetail";
import { Runs } from "./pages/Runs";
import { Settings } from "./pages/Settings";
import { Workloads } from "./pages/Workloads";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <BrowserRouter>
      <Routes>
        <Route element={<Shell />}>
          <Route path="/" element={<Dashboard />} />
          <Route path="/runs" element={<Runs />} />
          <Route path="/runs/:id" element={<RunDetail />} />
          <Route path="/diagnosis/:id" element={<RunDetail focusDiagnosis />} />
          <Route path="/experiments" element={<Experiments />} />
          <Route path="/experiments/:id" element={<ExperimentDetail />} />
          <Route path="/jobs" element={<Jobs />} />
          <Route path="/workloads" element={<Workloads />} />
          <Route path="/hardware" element={<Hardware />} />
          <Route path="/benchmarks" element={<Benchmarks />} />
          <Route path="/genome" element={<Genome />} />
          <Route path="/reports" element={<Reports />} />
          <Route path="/settings" element={<Settings />} />
        </Route>
      </Routes>
    </BrowserRouter>
  </StrictMode>,
);
