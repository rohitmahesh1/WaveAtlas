// src/pages/RunsPage.tsx
import { PastRunsPanel } from "../components/PastRunsPanel";
import { useJobHistory } from "../hooks/useJobHistory";
import { cancelJob, deleteJob, resumeJob, updateJobName } from "../api";
import { useSharedJobSession } from "../hooks/useSharedJobSession";

export default function RunsPage(props: { onOpenViewer: () => void }) {
  const { onOpenViewer } = props;
  const { jobs, loading, error, refresh } = useJobHistory();
  const { jobId, loadJob, clearSession } = useSharedJobSession();

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="brand">
          <div className="brand-title">WaveAtlas</div>
          <div className="brand-sub">Past Runs</div>
        </div>
      </header>

      <main className="app-main">
        <aside className="sidebar">
          <PastRunsPanel
            jobs={jobs}
            loading={loading}
            error={error}
            currentJobId={jobId}
            includeCurrent
            onRefresh={refresh}
            onLoad={(id) => {
              loadJob(id);
              onOpenViewer();
            }}
            onCancel={async (id) => {
              try {
                await cancelJob(id);
                refresh();
              } catch {
                // no-op
              }
            }}
            onResume={async (id) => {
              try {
                await resumeJob(id);
                loadJob(id);
                onOpenViewer();
              } catch {
                // no-op
              }
            }}
            onDelete={async (id) => {
              const ok = window.confirm("Delete this run and its artifacts? This cannot be undone.");
              if (!ok) return;
              try {
                await deleteJob(id);
                if (id === jobId) {
                  clearSession();
                }
                refresh();
              } catch {
                // no-op
              }
            }}
            onRename={async (id, name) => {
              try {
                await updateJobName(id, name);
                refresh();
              } catch {
                // no-op
              }
            }}
          />
        </aside>
        <section className="viewer">
          <div className="panel">
            <div className="panel-title">Tips</div>
            <div className="panel-body">
              <div className="empty-text">Select a run to open it in the Viewer.</div>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
