import React from "react";

export function RunPanel(props: {
  file: File | null;
  onFileChange: (file: File | null) => void;
  onRun: () => void;
  jobId: string | null;
  status?: string;
  runName: string;
  onRunNameChange: (value: string) => void;
  filteredCount: number;
  totalCount: number;
  onCancel?: () => void;
  cancelDisabled?: boolean;
  onResume?: () => void;
  onNewRun?: () => void;
}) {
  const {
    file,
    onFileChange,
    onRun,
    jobId,
    status,
    runName,
    onRunNameChange,
    filteredCount,
    totalCount,
    onCancel,
    cancelDisabled,
    onResume,
    onNewRun,
  } = props;
  const canRun = Boolean(file);
  const isCancelled = status === "cancelled";
  const tracksLabel =
    totalCount > 0
      ? `${filteredCount}/${totalCount}`
      : jobId
      ? ["completed", "failed", "cancelled"].includes(String(status))
        ? "No tracks found"
        : "Waiting for tracks"
      : "—";

  return (
    <section className="panel run-panel">
      <div className="panel-title">Run</div>
      <div className="panel-body">
        <label className="run-name-label">
          Run name
          <input
            className="run-name-input"
            type="text"
            value={runName}
            placeholder="Untitled run"
            onChange={(e) => onRunNameChange(e.target.value)}
          />
        </label>
        <div className="control-row run-file-row">
          <label className="file-button">
            Choose file
            <input
              className="file-hidden"
              type="file"
              accept=".csv,.tsv,.xlsx,.xls"
              onChange={(e) => onFileChange(e.target.files?.[0] || null)}
            />
          </label>
          <div className="file-name">{file?.name ?? "No file selected"}</div>
          <button className="primary-btn" disabled={!canRun} onClick={onRun}>
            Upload + Start
          </button>
        </div>

        <div className="meta-grid">
          <div>
            Job
            <div className="meta-value" title={jobId ?? undefined}>
              {jobId ? `${jobId.slice(0, 8)}…${jobId.slice(-4)}` : "—"}
            </div>
          </div>
          <div>
            Tracks
            <div className="meta-value">
              {tracksLabel}
            </div>
          </div>
        </div>

        {jobId ? (
          <div className="button-row run-actions">
            {isCancelled && onResume ? (
              <button className="ghost-btn" onClick={onResume}>
                Resume
              </button>
            ) : null}
            {onCancel ? (
              <button className="ghost-btn danger-btn" onClick={onCancel} disabled={cancelDisabled}>
                Cancel
              </button>
            ) : null}
            <button className="ghost-btn" onClick={onNewRun}>
              New run
            </button>
          </div>
        ) : null}
      </div>
    </section>
  );
}
