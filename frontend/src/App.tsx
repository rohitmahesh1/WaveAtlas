import { useState } from "react";
import AdvancedViewerPage from "./pages/AdvancedViewerPage";
import AdvancedConfigPage from "./pages/AdvancedConfigPage";
import RunsPage from "./pages/RunsPage";
import ConfigDocsPage from "./pages/ConfigDocsPage";
import { JobSessionProvider } from "./context/JobSessionProvider";

export default function App() {
  const [page, setPage] = useState<"viewer" | "advanced" | "runs" | "docs">("viewer");

  return (
    <JobSessionProvider>
      <nav className="app-nav">
        <button
          className={`nav-btn ${page === "viewer" ? "nav-active" : ""}`}
          onClick={() => setPage("viewer")}
        >
          Viewer
        </button>
        <button className={`nav-btn ${page === "runs" ? "nav-active" : ""}`} onClick={() => setPage("runs")}>
          Runs
        </button>
        <button
          className={`nav-btn ${page === "advanced" ? "nav-active" : ""}`}
          onClick={() => setPage("advanced")}
        >
          Advanced
        </button>
        <button className={`nav-btn ${page === "docs" ? "nav-active" : ""}`} onClick={() => setPage("docs")}>
          Docs
        </button>
      </nav>
      <div className={`page ${page === "viewer" ? "page-active" : ""}`}>
        <AdvancedViewerPage onViewAllRuns={() => setPage("runs")} />
      </div>
      <div className={`page ${page === "runs" ? "page-active" : ""}`}>
        <RunsPage onOpenViewer={() => setPage("viewer")} />
      </div>
      <div className={`page ${page === "advanced" ? "page-active" : ""}`}>
        <AdvancedConfigPage />
      </div>
      <div className={`page ${page === "docs" ? "page-active" : ""}`}>
        <ConfigDocsPage />
      </div>
    </JobSessionProvider>
  );
}
