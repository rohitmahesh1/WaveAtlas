import React, { useState } from "react";
import type { JobRead } from "../api";

function fmtTime(ts?: string | null) {
  if (!ts) return "—";
  const d = new Date(ts);
  if (Number.isNaN(d.getTime())) return "—";
  return d.toLocaleString();
}

function shortId(id: string) {
  return `${id.slice(0, 8)}…${id.slice(-4)}`;
}

export function PastRunsPanel(props: {
  jobs: JobRead[];
  loading: boolean;
  error: string | null;
  currentJobId: string | null;
  includeCurrent?: boolean;
  limit?: number;
  showSummary?: boolean;
  onViewAll?: () => void;
  onRefresh: () => void;
  onLoad: (id: string) => void;
  onCancel: (id: string) => void;
  onResume?: (id: string) => void;
  onDelete: (id: string) => void;
  onRename?: (id: string, name: string) => void;
}) {
  const {
    jobs,
    loading,
    error,
    currentJobId,
    includeCurrent = false,
    limit,
    showSummary = false,
    onViewAll,
    onRefresh,
    onLoad,
    onCancel,
    onResume,
    onDelete,
    onRename,
  } = props;

  const list = includeCurrent ? jobs : jobs.filter((job) => job.id !== currentJobId);
  const visible = typeof limit === "number" ? list.slice(0, Math.max(0, limit)) : list;
  const hiddenCount = Math.max(0, list.length - visible.length);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editingName, setEditingName] = useState<string>("");

  return (
    <section className="panel">
      <div className="panel-title">Past Runs</div>
      <div className="panel-body">
        <div className="runs-actions">
          <button className="ghost-btn" onClick={onRefresh} disabled={loading}>
            {loading ? "Refreshing…" : "Refresh"}
          </button>
          {error ? <div className="error-text">{error}</div> : null}
        </div>
        {visible.length === 0 ? (
          <div className="empty-text">No runs yet.</div>
        ) : (
          <div className="runs-list">
            {visible.map((job) => {
              const isCurrent = currentJobId === job.id;
              const canCancel = ["queued", "in_progress", "cancel_requested"].includes(job.status);
              const canResume = job.status === "cancelled";
              const canDelete = ["completed", "failed", "cancelled"].includes(job.status);
              const isEditing = editingId === job.id;
              const runName = job.run_name?.trim() || "Untitled run";
              return (
                <div key={job.id} className={`run-item ${isCurrent ? "run-active" : ""}`}>
                  <div className="run-main">
                    {isEditing ? (
                      <div className="run-name-edit">
                        <input
                          className="run-name-input"
                          type="text"
                          value={editingName}
                          onChange={(e) => setEditingName(e.target.value)}
                          placeholder="Untitled run"
                        />
                      </div>
                    ) : (
                      <div>
                        <div className="run-name" title={runName}>
                          {runName}
                        </div>
                        <div className="run-id" title={job.id}>
                          {shortId(job.id)}
                        </div>
                      </div>
                    )}
                    <div className={`run-status status-${job.status}`}>{job.status.replace(/_/g, " ")}</div>
                  </div>
                  <div className="run-meta">
                    {job.input_filename ? <span>File {job.input_filename}</span> : null}
                    <span>Created {fmtTime(job.created_at)}</span>
                    {job.finished_at ? <span>Finished {fmtTime(job.finished_at)}</span> : null}
                  </div>
                  <div className="run-actions">
                    <button className="ghost-btn" onClick={() => onLoad(job.id)}>
                      Load
                    </button>
                    <button className="ghost-btn" onClick={() => onCancel(job.id)} disabled={!canCancel}>
                      Cancel
                    </button>
                    {onResume && canResume ? (
                      <button className="ghost-btn" onClick={() => onResume(job.id)}>
                        Resume
                      </button>
                    ) : null}
                    {onRename ? (
                      isEditing ? (
                        <>
                          <button
                            className="ghost-btn"
                            onClick={() => {
                              const nextName = editingName.trim();
                              if (nextName && nextName !== runName) {
                                onRename(job.id, nextName);
                              }
                              setEditingId(null);
                              setEditingName("");
                            }}
                          >
                            Save
                          </button>
                          <button
                            className="ghost-btn"
                            onClick={() => {
                              setEditingId(null);
                              setEditingName("");
                            }}
                          >
                            Cancel
                          </button>
                        </>
                      ) : (
                        <button
                          className="ghost-btn"
                          onClick={() => {
                            setEditingId(job.id);
                            setEditingName(runName);
                          }}
                        >
                          Rename
                        </button>
                      )
                    ) : null}
                    <button className="ghost-btn danger-btn" onClick={() => onDelete(job.id)} disabled={!canDelete}>
                      Delete
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        )}
        {showSummary && hiddenCount > 0 ? (
          <div className="runs-summary">
            <span>{hiddenCount} not shown.</span>
            {onViewAll ? (
              <button className="ghost-btn" onClick={onViewAll}>
                View all runs
              </button>
            ) : null}
          </div>
        ) : null}
      </div>
    </section>
  );
}
