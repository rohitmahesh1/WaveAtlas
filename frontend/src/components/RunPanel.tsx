export function RunPanel(props: {
  file: File | null;
  onFileChange: (file: File | null) => void;
  onRun: () => void;
  imageSizing?: {
    loading: boolean;
    originalWidth: number | null;
    originalHeight: number | null;
    targetWidth: string;
    targetHeight: string;
    helperText: string | null;
    loadError: string | null;
    validationError: string | null;
    valid: boolean;
    onTargetWidthChange: (value: string) => void;
    onTargetHeightChange: (value: string) => void;
  } | null;
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
  onDownloadWaves?: () => void;
  onDownloadHeatmap?: () => void;
  onDownloadOriginalImage?: () => void;
  heatmapDownloadDisabled?: boolean;
  originalImageDownloadDisabled?: boolean;
}) {
  const {
    file,
    onFileChange,
    onRun,
    imageSizing = null,
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
    onDownloadWaves,
    onDownloadHeatmap,
    onDownloadOriginalImage,
    heatmapDownloadDisabled,
    originalImageDownloadDisabled,
  } = props;
  const canRun = Boolean(file) && (!imageSizing || (!imageSizing.loading && imageSizing.valid));
  const normalizedStatus = String(status ?? "");
  const isCancelled = normalizedStatus === "cancelled";
  const isPausePending = normalizedStatus === "cancel_requested";
  const canPause = ["queued", "in_progress"].includes(normalizedStatus);
  const canResume = isCancelled && Boolean(onResume);
  const showTransport = Boolean(jobId && (((canPause || isPausePending) && onCancel) || canResume));
  const transportMode = canResume ? "play" : "pause";
  const transportLabel = canResume ? "Resume run" : isPausePending ? "Stopping run" : "Pause run";
  const transportDisabled = canResume ? false : !canPause || Boolean(cancelDisabled);
  const handleTransport = () => {
    if (canResume) {
      onResume?.();
      return;
    }
    if (canPause && !cancelDisabled) onCancel?.();
  };
  const tracksLabel =
    totalCount > 0
      ? `${filteredCount}/${totalCount}`
      : jobId
      ? ["completed", "failed", "cancelled"].includes(normalizedStatus)
        ? "No tracks found"
        : "Waiting for tracks"
      : "—";

  return (
    <section className="panel run-panel">
      <div className="panel-title-row run-title-row">
        <div className="panel-title">Run</div>
        {showTransport || onNewRun ? (
          <div className="run-title-actions">
            {showTransport ? (
              <button
                className={`transport-btn transport-${transportMode}`}
                onClick={handleTransport}
                disabled={transportDisabled}
                title={transportLabel}
                aria-label={transportLabel}
              >
                <span className={`transport-icon ${transportMode}`} aria-hidden="true" />
                <span>{canResume ? "Resume" : "Pause"}</span>
              </button>
            ) : null}
            {jobId && onNewRun ? (
              <button className="ghost-btn compact-btn" onClick={onNewRun}>
                New
              </button>
            ) : null}
          </div>
        ) : null}
      </div>
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
              accept=".csv,.tsv,.xlsx,.xls,.png,.jpg,.jpeg,.tif,.tiff,.bmp,.webp,image/*"
              onChange={(e) => onFileChange(e.target.files?.[0] || null)}
            />
          </label>
          <div className="file-name">{file?.name ?? "No file selected"}</div>
          <button className="primary-btn" disabled={!canRun} onClick={onRun}>
            Upload + Start
          </button>
        </div>

        {imageSizing ? (
          <div className="image-sizing-card">
            <div className="image-sizing-header">
              <div className="image-sizing-title">Image processing size</div>
              <div className="image-sizing-copy">Set the internal dimensions used before tracking.</div>
            </div>
            <div className="image-sizing-grid">
              <label className="image-sizing-field">
                Width (px)
                <input
                  type="number"
                  min="1"
                  inputMode="numeric"
                  value={imageSizing.targetWidth}
                  onChange={(e) => imageSizing.onTargetWidthChange(e.target.value)}
                  disabled={imageSizing.loading}
                  placeholder={imageSizing.originalWidth ? String(imageSizing.originalWidth) : "e.g. 1024"}
                />
              </label>
              <label className="image-sizing-field">
                Height (px)
                <input
                  type="number"
                  min="1"
                  inputMode="numeric"
                  value={imageSizing.targetHeight}
                  onChange={(e) => imageSizing.onTargetHeightChange(e.target.value)}
                  disabled={imageSizing.loading}
                  placeholder={imageSizing.originalHeight ? String(imageSizing.originalHeight) : "e.g. 768"}
                />
              </label>
            </div>
            {imageSizing.helperText ? <div className="image-sizing-hint">{imageSizing.helperText}</div> : null}
            {imageSizing.loadError ? <div className="image-sizing-error">{imageSizing.loadError}</div> : null}
            {imageSizing.validationError ? (
              <div className="image-sizing-error">{imageSizing.validationError}</div>
            ) : null}
          </div>
        ) : null}

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
          <div className="run-secondary-row">
            <div className="run-transport-note">
              {canResume ? "Stopped. Resume from saved artifacts." : isPausePending ? "Stop requested..." : "Run outputs"}
            </div>
            {onDownloadWaves || onDownloadHeatmap || onDownloadOriginalImage ? (
              <details className="run-download-menu">
                <summary>Downloads</summary>
                <div className="run-download-popover">
                  {onDownloadWaves ? (
                    <button className="ghost-btn download-btn compact-btn" onClick={onDownloadWaves}>
                      Waves CSV
                    </button>
                  ) : null}
                  {onDownloadHeatmap ? (
                    <button
                      className="ghost-btn download-btn compact-btn"
                      onClick={onDownloadHeatmap}
                      disabled={heatmapDownloadDisabled}
                    >
                      Heatmap
                    </button>
                  ) : null}
                  {onDownloadOriginalImage ? (
                    <button
                      className="ghost-btn download-btn compact-btn"
                      onClick={onDownloadOriginalImage}
                      disabled={originalImageDownloadDisabled}
                    >
                      Original image
                    </button>
                  ) : null}
                </div>
              </details>
            ) : null}
          </div>
        ) : null}
      </div>
    </section>
  );
}
