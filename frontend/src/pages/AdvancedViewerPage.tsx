// src/pages/AdvancedViewerPage.tsx
import { useEffect, useMemo, useRef, useState } from "react";

import { OverlayCanvas } from "../OverlayCanvas";
import type { OverlayTrackEvent } from "../OverlayCanvas";
import type { FieldDef, FilterOp } from "../types";
import { stageLabel } from "../utils/format";
import { RunPanel } from "../components/RunPanel";
import { FiltersPanel } from "../components/FiltersPanel";
import { SelectionPanel } from "../components/SelectionPanel";
import { SummaryPanel } from "../components/SummaryPanel";
import { ActivityPanel } from "../components/ActivityPanel";
import { ViewerControls } from "../components/ViewerControls";
import { useJobSession } from "../hooks/useJobSession";
import { useFilters } from "../hooks/useFilters";
import { useTrackDetail } from "../hooks/useTrackDetail";
import { useJobHistory } from "../hooks/useJobHistory";
import { PastRunsPanel } from "../components/PastRunsPanel";
import { cancelJob, deleteJob, resumeJob, updateJobName } from "../api";

const NUMERIC_OPS: FilterOp[] = [">", "<", ">=", "<=", "==", "!=", "between"];
const STRING_OPS: FilterOp[] = ["contains", "==", "!="];

const FILTER_FIELDS: FieldDef[] = [
  { key: "track_index", label: "Track ID", type: "number", ops: NUMERIC_OPS, get: (t) => t.track_index },
  { key: "points", label: "Points", type: "number", ops: NUMERIC_OPS, get: (t) => t.poly?.length ?? 0 },
  { key: "num_peaks", label: "Peaks", type: "number", ops: NUMERIC_OPS, get: (t) => t.metrics?.num_peaks ?? null },
  {
    key: "mean_amplitude",
    label: "Mean amplitude",
    type: "number",
    ops: NUMERIC_OPS,
    get: (t) => t.metrics?.mean_amplitude ?? null,
  },
  {
    key: "dominant_frequency",
    label: "Dominant frequency",
    type: "number",
    ops: NUMERIC_OPS,
    get: (t) => t.metrics?.dominant_frequency ?? null,
  },
  { key: "period", label: "Period", type: "number", ops: NUMERIC_OPS, get: (t) => t.metrics?.period ?? null },
  { key: "sample", label: "Sample", type: "string", ops: STRING_OPS, get: (t) => t.sample ?? "" },
];

export default function AdvancedViewerPage(props: { onViewAllRuns?: () => void }) {
  const { onViewAllRuns } = props;
  const [file, setFile] = useState<File | null>(null);
  const [selectedTrackId, setSelectedTrackId] = useState<string | number | null>(null);
  const [hoveredTrackId, setHoveredTrackId] = useState<string | number | null>(null);
  const [selectedDebugLabel, setSelectedDebugLabel] = useState<string>("none");
  const [debugOpacity, setDebugOpacity] = useState<number>(0.6);
  const [runName, setRunName] = useState<string>("");
  const [runNameAuto, setRunNameAuto] = useState<boolean>(true);
  const runCounterRef = useRef<number>(1);

  const [hideBaseImage, setHideBaseImage] = useState<boolean>(false);
  const defaultOverlayColor = "#008c5a";
  const [overlayColor, setOverlayColor] = useState<string>(defaultOverlayColor);

  const {
    jobId,
    status,
    baseImageUrl,
    tracks,
    activity,
    currentStage,
    stageDetail,
    etaText,
    debugOverlays,
    runJob,
    cancelCurrentJob,
    loadJob,
    clearSession,
  } = useJobSession();

  const { jobs, loading: jobsLoading, error: jobsError, refresh: refreshJobs } = useJobHistory();

  const buildDefaultRunName = (file: File) => {
    const raw = file.name || "run";
    const base = raw.replace(/\.[^/.]+$/, "") || "run";
    const num = runCounterRef.current;
    runCounterRef.current += 1;
    return `${base} #${num}`;
  };

  const handleFileChange = (nextFile: File | null) => {
    setFile(nextFile);
    if (nextFile && (runNameAuto || !runName.trim())) {
      setRunName(buildDefaultRunName(nextFile));
      setRunNameAuto(true);
    }
    if (!nextFile && runNameAuto) {
      setRunName("");
    }
  };

  useEffect(() => {
    if (jobId) refreshJobs();
  }, [jobId, status, refreshJobs]);

  const {
    filters,
    addFilterRule,
    updateFilterRule,
    removeFilterRule,
    clearFilters,
    fieldMap,
    filteredTracks,
    filteredStats,
    setFilters,
  } = useFilters(tracks, FILTER_FIELDS);

  const stageText = stageDetail ? `${stageLabel(currentStage)} — ${stageDetail}` : stageLabel(currentStage);
  const statusLabel = String(status).replace(/_/g, " ");
  const showSpinner = !["completed", "failed", "cancelled", "idle"].includes(String(status));

  const activeSelectedTrackId = useMemo(() => {
    if (selectedTrackId == null) return null;
    const visible = filteredTracks.some((t) => String(t.id ?? t.track_index) === String(selectedTrackId));
    return visible ? selectedTrackId : null;
  }, [filteredTracks, selectedTrackId]);

  const { trackDetail, trackDetailLoading, trackDetailError, resetTrackDetail } = useTrackDetail(
    jobId,
    activeSelectedTrackId
  );

  const selectedTrack = useMemo(() => {
    if (activeSelectedTrackId == null) return null;
    return filteredTracks.find((t) => String(t.id ?? t.track_index) === String(activeSelectedTrackId)) ?? null;
  }, [filteredTracks, activeSelectedTrackId]);

  const hoverColorFn = useMemo(() => {
    if (hoveredTrackId == null) return undefined;
    return (t: OverlayTrackEvent) =>
      String(t.id ?? t.track_index) === String(hoveredTrackId) ? "rgba(255,215,0,0.9)" : undefined;
  }, [hoveredTrackId]);

  const activeDebugLabel = useMemo(() => {
    if (selectedDebugLabel === "none") return "none";
    return debugOverlays.some((o) => o.label === selectedDebugLabel) ? selectedDebugLabel : "none";
  }, [debugOverlays, selectedDebugLabel]);

  const debugImageUrl = useMemo(() => {
    if (activeDebugLabel === "none") return null;
    return debugOverlays.find((o) => o.label === activeDebugLabel)?.url ?? null;
  }, [debugOverlays, activeDebugLabel]);

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="brand">
          <div className="brand-title">WaveAtlas</div>
          <div className="brand-sub">Interactive Viewer</div>
        </div>
        <div className="status-cluster">
          <div className={`status-pill status-${status}`}>Status: {statusLabel}</div>
          <div className="stage-pill">
            {showSpinner ? <span className="spinner" /> : null}
            <span>{stageText}</span>
          </div>
        </div>
      </header>

      <main className="app-main">
        <aside className="sidebar">
          <RunPanel
            file={file}
            onFileChange={handleFileChange}
            onRun={() => {
              if (!file) return;
              setSelectedTrackId(null);
              setHoveredTrackId(null);
              setSelectedDebugLabel("none");
              resetTrackDetail();
              runJob(file, undefined, runName);
              refreshJobs();
            }}
            jobId={jobId}
            status={status}
            runName={runName}
            onRunNameChange={(value) => {
              setRunName(value);
              setRunNameAuto(false);
            }}
            filteredCount={filteredTracks.length}
            totalCount={tracks.length}
            onCancel={cancelCurrentJob}
            cancelDisabled={!jobId || ["completed", "failed", "cancelled"].includes(status)}
            onResume={async () => {
              if (!jobId || status !== "cancelled") return;
              try {
                await resumeJob(jobId);
                loadJob(jobId);
                refreshJobs();
              } catch {
                // no-op for now
              }
            }}
            onNewRun={() => {
              clearSession();
              setFile(null);
              setRunName("");
              setRunNameAuto(true);
              setSelectedTrackId(null);
              setFilters([]);
              setSelectedDebugLabel("none");
              resetTrackDetail();
              refreshJobs();
            }}
          />

          {selectedTrack ? (
            <SelectionPanel
              selectedTrack={selectedTrack}
              trackDetail={trackDetail}
              trackDetailLoading={trackDetailLoading}
              trackDetailError={trackDetailError}
              overlayColor={overlayColor}
              baseImageUrl={baseImageUrl}
              debugImageUrl={debugImageUrl}
              debugOpacity={debugOpacity}
            />
          ) : null}

          <PastRunsPanel
            jobs={jobs}
            loading={jobsLoading}
            error={jobsError}
            currentJobId={jobId}
            limit={1}
            showSummary
            onViewAll={onViewAllRuns}
            onRefresh={refreshJobs}
            onLoad={(id) => {
              loadJob(id);
              setSelectedTrackId(null);
              setSelectedDebugLabel("none");
              resetTrackDetail();
            }}
            onCancel={async (id) => {
              try {
                await cancelJob(id);
                refreshJobs();
              } catch {
                // no-op for now
              }
            }}
            onResume={async (id) => {
              try {
                await resumeJob(id);
                loadJob(id);
                refreshJobs();
              } catch {
                // no-op for now
              }
            }}
            onDelete={async (id) => {
              const ok = window.confirm("Delete this run and its artifacts? This cannot be undone.");
              if (!ok) return;
              try {
                await deleteJob(id);
                if (id === jobId) {
                  clearSession();
                  setSelectedTrackId(null);
                  setSelectedDebugLabel("none");
                  resetTrackDetail();
                }
                refreshJobs();
              } catch {
                // no-op for now
              }
            }}
            onRename={async (id, name) => {
              try {
                await updateJobName(id, name);
                refreshJobs();
              } catch {
                // no-op for now
              }
            }}
          />

          <FiltersPanel
            filters={filters}
            fields={FILTER_FIELDS}
            fieldMap={fieldMap}
            onAdd={addFilterRule}
            onClear={clearFilters}
            onUpdate={updateFilterRule}
            onRemove={removeFilterRule}
          />

          <SummaryPanel stats={filteredStats} />

          <ActivityPanel activity={activity} />
        </aside>

        <section className="viewer">
          <div className="viewer-top">
          <div className="viewer-meta">
            {tracks.length === 0 ? (
              <span>
                {["completed", "failed", "cancelled"].includes(String(status))
                  ? "No tracks found."
                  : "Waiting for tracks…"}
              </span>
            ) : (
              <span>
                Viewing {filteredTracks.length} of {tracks.length} tracks
              </span>
            )}
            {etaText ? <span className="eta-pill">ETA {etaText}</span> : null}
          </div>
            <ViewerControls
              overlayColor={overlayColor}
              onOverlayColorChange={setOverlayColor}
              onOverlayColorReset={() => setOverlayColor(defaultOverlayColor)}
              hideBaseImage={hideBaseImage}
              onHideBaseImageChange={setHideBaseImage}
              debugOverlays={debugOverlays}
              selectedDebugLabel={activeDebugLabel}
              onDebugLabelChange={setSelectedDebugLabel}
              debugOpacity={debugOpacity}
              onDebugOpacityChange={setDebugOpacity}
            />
          </div>

          <div className="canvas-card">
            <OverlayCanvas
              imageUrl={baseImageUrl}
              debugImageUrl={debugImageUrl}
              debugOpacity={debugOpacity}
              tracks={filteredTracks}
              overlayColor={overlayColor}
              hideBaseImage={hideBaseImage}
              selectedTrackId={activeSelectedTrackId}
              onClickTrack={(t) => setSelectedTrackId(t ? (t.id ?? t.track_index) : null)}
              onHoverTrack={(t) => setHoveredTrackId(t ? (t.id ?? t.track_index) : null)}
              colorOverrideFn={hoverColorFn}
            />
          </div>
          {tracks.length > 0 && filteredTracks.length === 0 ? (
            <div className="empty-text">No tracks match the current filters.</div>
          ) : null}
        </section>
      </main>
    </div>
  );
}
