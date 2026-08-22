// src/pages/AdvancedViewerPage.tsx
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { OverlayCanvas, UNASSIGNED_FAMILY_KEY } from "../OverlayCanvas";
import type { OverlayProjection, OverlayTrackEvent } from "../OverlayCanvas";
import type { FieldDef, FilterOp } from "../types";
import { stageLabel } from "../utils/format";
import { RunPanel } from "../components/RunPanel";
import { FiltersPanel } from "../components/FiltersPanel";
import { SelectionPanel } from "../components/SelectionPanel";
import { SummaryPanel } from "../components/SummaryPanel";
import { ActivityPanel } from "../components/ActivityPanel";
import { ViewerControls } from "../components/ViewerControls";
import { useFilters } from "../hooks/useFilters";
import { useTrackDetail } from "../hooks/useTrackDetail";
import { useJobHistory } from "../hooks/useJobHistory";
import { PastRunsPanel } from "../components/PastRunsPanel";
import { cancelJob, deleteJob, jobRippleCsvUrl, jobWavesCsvUrl, resumeJob, updateJobName } from "../api";
import { useImageProcessingPrompt } from "../hooks/useImageProcessingPrompt";
import { useSharedJobSession } from "../hooks/useSharedJobSession";
import { downloadCsv, downloadFromUrl, downloadJson } from "../utils/download";
import { mergeRunConfigWithImageProcessing } from "../utils/imageProcessing";
import {
  buildHeatmapOptionsConfig,
  DEFAULT_HEATMAP_OPTIONS,
  type HeatmapOptions,
} from "../utils/heatmapOptions";
import {
  buildAnalysisOptionsConfig,
  DEFAULT_ANALYSIS_MODE,
  normalizeAnalysisMode,
  type AnalysisMode,
} from "../utils/analysisOptions";

const NUMERIC_OPS: FilterOp[] = [">", "<", ">=", "<=", "==", "!=", "between"];
const STRING_OPS: FilterOp[] = ["contains", "==", "!="];
type SelectionScope = "family" | "track";
const UNASSIGNED_RIPPLE_COLOR = "#87919a";

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
  { key: "family_id", label: "Family", type: "string", ops: STRING_OPS, get: (t) => t.metrics?.family_id ?? "" },
  { key: "direction", label: "Direction", type: "string", ops: STRING_OPS, get: (t) => t.metrics?.direction ?? "" },
  {
    key: "slope_px_per_frame",
    label: "Slope",
    type: "number",
    ops: NUMERIC_OPS,
    get: (t) => t.metrics?.slope_px_per_frame ?? null,
  },
  {
    key: "velocity_px_per_s",
    label: "Velocity",
    type: "number",
    ops: NUMERIC_OPS,
    get: (t) => t.metrics?.velocity_px_per_s ?? null,
  },
  {
    key: "speed_px_per_s",
    label: "Speed",
    type: "number",
    ops: NUMERIC_OPS,
    get: (t) => t.metrics?.speed_px_per_s ?? (
      t.metrics?.velocity_px_per_s != null ? Math.abs(t.metrics.velocity_px_per_s) : null
    ),
  },
  {
    key: "angle_deg",
    label: "Angle",
    type: "number",
    ops: NUMERIC_OPS,
    get: (t) => t.metrics?.angle_from_time_axis_deg ?? t.metrics?.angle_deg ?? null,
  },
  {
    key: "line_rmse_px",
    label: "Line fit error",
    type: "number",
    ops: NUMERIC_OPS,
    get: (t) => t.metrics?.line_rmse_px ?? null,
  },
  { key: "sample", label: "Sample", type: "string", ops: STRING_OPS, get: (t) => t.sample ?? "" },
];

function runStem(id: string | null) {
  return id ? `waveatlas_${id.slice(0, 8)}` : "waveatlas";
}

const RIPPLE_EXPORT_NAMES: Record<"tracks" | "intervals" | "families", { stem: string; label: string }> = {
  tracks: { stem: "tracks", label: "tracks" },
  intervals: { stem: "waves", label: "waves" },
  families: { stem: "families", label: "families" },
};

function hashText(value: string) {
  let hash = 0;
  for (let i = 0; i < value.length; i += 1) {
    hash = ((hash << 5) - hash + value.charCodeAt(i)) | 0;
  }
  return Math.abs(hash);
}

function hslToHex(hue: number, saturation: number, lightness: number) {
  const s = saturation / 100;
  const l = lightness / 100;
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const h = ((hue % 360) + 360) % 360 / 60;
  const x = c * (1 - Math.abs((h % 2) - 1));
  const [r1, g1, b1] =
    h < 1 ? [c, x, 0] :
    h < 2 ? [x, c, 0] :
    h < 3 ? [0, c, x] :
    h < 4 ? [0, x, c] :
    h < 5 ? [x, 0, c] :
    [c, 0, x];
  const m = l - c / 2;
  const toHex = (value: number) => Math.round((value + m) * 255).toString(16).padStart(2, "0");
  return `#${toHex(r1)}${toHex(g1)}${toHex(b1)}`;
}

function familyColor(familyId: string | null | undefined) {
  if (!familyId || familyId === UNASSIGNED_FAMILY_KEY) return undefined;
  const match = familyId.match(/\d+/);
  const seed = match ? Math.max(0, Number(match[0]) - 1) : hashText(familyId);
  const hue = (seed * 137.508 + 162) % 360;
  return hslToHex(hue, 72, 52);
}

function trackFamilyKey(track: OverlayTrackEvent) {
  return track.metrics?.family_id || UNASSIGNED_FAMILY_KEY;
}

function familyLabel(familyId: string) {
  return familyId === UNASSIGNED_FAMILY_KEY ? "Unassigned" : familyId;
}

function familyFilterValue(familyId: string) {
  return familyId === UNASSIGNED_FAMILY_KEY ? "" : familyId;
}

export default function AdvancedViewerPage(props: { onViewAllRuns?: () => void }) {
  const { onViewAllRuns } = props;
  const [file, setFile] = useState<File | null>(null);
  const [selectionJobId, setSelectionJobId] = useState<string | null>(null);
  const [rawSelectedTrackId, setSelectedTrackId] = useState<string | number | null>(null);
  const [rawSelectedFamilyId, setSelectedFamilyId] = useState<string | null>(null);
  const [rawSelectionScope, setSelectionScope] = useState<SelectionScope | null>(null);
  const [debugSelection, setDebugSelection] = useState<{ jobId: string | null; label: string }>({
    jobId: null,
    label: "none",
  });
  const [debugOpacity, setDebugOpacity] = useState<number>(0.6);
  const [runName, setRunName] = useState<string>("");
  const [runNameAuto, setRunNameAuto] = useState<boolean>(true);
  const [heatmapOptions, setHeatmapOptions] = useState<HeatmapOptions>(DEFAULT_HEATMAP_OPTIONS);
  const [runAnalysisMode, setRunAnalysisMode] = useState<AnalysisMode>(DEFAULT_ANALYSIS_MODE);
  const runCounterRef = useRef<number>(1);

  const [hideBaseImage, setHideBaseImage] = useState<boolean>(false);
  const [hideTracks, setHideTracks] = useState<boolean>(false);
  const defaultOverlayColor = "#008c5a";
  const [overlayColor, setOverlayColor] = useState<string>(defaultOverlayColor);
  const [viewerProjection, setViewerProjection] = useState<OverlayProjection | null>(null);

  const {
    jobId,
    status,
    baseImageUrl,
    baseImageInfo,
    heatmapValues,
    analysisMode,
    originalImageUrl,
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
  } = useSharedJobSession();

  const { jobs, loading: jobsLoading, error: jobsError, refresh: refreshJobs } = useJobHistory();
  const {
    imageSizing,
    dimensions: imageProcessingDimensions,
    syncWithFile: syncImageProcessingWithFile,
    reset: resetImageProcessing,
  } = useImageProcessingPrompt();

  const buildDefaultRunName = (file: File) => {
    const raw = file.name || "run";
    const base = raw.replace(/\.[^/.]+$/, "") || "run";
    const num = runCounterRef.current;
    runCounterRef.current += 1;
    return `${base} #${num}`;
  };

  const handleFileChange = (nextFile: File | null) => {
    setFile(nextFile);
    syncImageProcessingWithFile(nextFile);
    if (nextFile && (runNameAuto || !runName.trim())) {
      setRunName(buildDefaultRunName(nextFile));
      setRunNameAuto(true);
    }
    if (!nextFile && runNameAuto) {
      setRunName("");
    }
  };

  const selectedTrackId = selectionJobId === jobId ? rawSelectedTrackId : null;
  const selectedFamilyId = selectionJobId === jobId ? rawSelectedFamilyId : null;
  const selectionScope = selectionJobId === jobId ? rawSelectionScope : null;
  const selectedDebugLabel = debugSelection.jobId === jobId ? debugSelection.label : "none";
  const setSelectedDebugLabel = useCallback((label: string) => {
    setDebugSelection({ jobId, label });
  }, [jobId]);

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
  } = useFilters(tracks, FILTER_FIELDS, jobId ?? "idle");

  const stageText = stageDetail ? `${stageLabel(currentStage)} — ${stageDetail}` : stageLabel(currentStage);
  const statusLabel = String(status).replace(/_/g, " ");
  const showSpinner = !["completed", "failed", "cancelled", "idle"].includes(String(status));
  const activeSelectedTrackId = useMemo(() => {
    if (selectedTrackId == null) return null;
    const visible = filteredTracks.some((t) => String(t.id ?? t.track_index) === String(selectedTrackId));
    return visible ? selectedTrackId : null;
  }, [filteredTracks, selectedTrackId]);
  const activeSelectedFamilyId = useMemo(() => {
    if (selectionScope !== "family" || !selectedFamilyId) return null;
    const visible = filteredTracks.some((t) => trackFamilyKey(t) === selectedFamilyId);
    return visible ? selectedFamilyId : null;
  }, [filteredTracks, selectedFamilyId, selectionScope]);
  const activeSelectionScope: SelectionScope | null = activeSelectedFamilyId
    ? "family"
    : activeSelectedTrackId != null && selectionScope === "track"
      ? "track"
      : null;

  const selectedTrack = useMemo(() => {
    if (activeSelectedTrackId == null) return null;
    return filteredTracks.find((t) => String(t.id ?? t.track_index) === String(activeSelectedTrackId)) ?? null;
  }, [filteredTracks, activeSelectedTrackId]);

  const trackDetailRevision = selectedTrack
    ? [
        status,
        selectedTrack.metrics?.analysis_mode ?? "standard",
        selectedTrack.metrics?.num_peaks ?? selectedTrack.peaks?.length ?? 0,
        selectedTrack.peaks?.length ?? 0,
      ].join(":")
    : String(status);
  const { trackDetail, trackDetailLoading, trackDetailError, resetTrackDetail } = useTrackDetail(
    jobId,
    activeSelectedTrackId,
    trackDetailRevision,
  );

  const familyLegend = useMemo(() => {
    if (analysisMode !== "ripple_family") return [];
    const counts = new Map<string, number>();
    for (const track of filteredTracks) {
      const familyId = trackFamilyKey(track);
      counts.set(familyId, (counts.get(familyId) ?? 0) + 1);
    }
    return Array.from(counts.entries())
      .sort(([left], [right]) => {
        if (left === UNASSIGNED_FAMILY_KEY) return 1;
        if (right === UNASSIGNED_FAMILY_KEY) return -1;
        return left.localeCompare(right, undefined, { numeric: true });
      })
      .map(([familyId, count]) => ({
        familyId,
        label: familyLabel(familyId),
        count,
        color: familyColor(familyId) ?? UNASSIGNED_RIPPLE_COLOR,
      }));
  }, [analysisMode, filteredTracks]);

  const selectedFamilySummary = useMemo(() => {
    if (analysisMode !== "ripple_family" || !activeSelectedFamilyId) return null;
    const familyTracks = filteredTracks.filter((track) => trackFamilyKey(track) === activeSelectedFamilyId);
    if (!familyTracks.length) return null;
    let slopeSum = 0;
    let slopeCount = 0;
    let speedSum = 0;
    let speedCount = 0;
    let angleSum = 0;
    let angleCount = 0;
    let frequencySum = 0;
    let frequencyCount = 0;
    const directions = new Set<string>();
    for (const track of familyTracks) {
      const slope = Number(track.metrics?.slope_px_per_frame);
      const speedValue = track.metrics?.speed_px_per_s ?? (
        track.metrics?.velocity_px_per_s != null ? Math.abs(track.metrics.velocity_px_per_s) : null
      );
      const speed = Number(speedValue);
      const angle = Number(track.metrics?.angle_from_time_axis_deg ?? track.metrics?.angle_deg);
      const frequency = Number(track.metrics?.dominant_frequency);
      if (Number.isFinite(slope)) {
        slopeSum += slope;
        slopeCount += 1;
      }
      if (Number.isFinite(speed)) {
        speedSum += speed;
        speedCount += 1;
      }
      if (Number.isFinite(angle)) {
        angleSum += angle;
        angleCount += 1;
      }
      if (Number.isFinite(frequency)) {
        frequencySum += frequency;
        frequencyCount += 1;
      }
      if (track.metrics?.direction) directions.add(track.metrics.direction);
    }
    return {
      familyId: activeSelectedFamilyId,
      label: familyLabel(activeSelectedFamilyId),
      color: familyColor(activeSelectedFamilyId) ?? UNASSIGNED_RIPPLE_COLOR,
      trackCount: familyTracks.length,
      avgSlope: slopeCount ? slopeSum / slopeCount : null,
      avgSpeed: speedCount ? speedSum / speedCount : null,
      avgAngle: angleCount ? angleSum / angleCount : null,
      avgFrequency: frequencyCount ? frequencySum / frequencyCount : null,
      directions: Array.from(directions).sort(),
    };
  }, [activeSelectedFamilyId, analysisMode, filteredTracks]);

  const familyIsolated = useMemo(() => {
    if (!activeSelectedFamilyId) return false;
    const filterValue = familyFilterValue(activeSelectedFamilyId);
    return filters.some(
      (rule) =>
        rule.field === "family_id"
        && rule.op === "=="
        && String(rule.value ?? "") === filterValue
    );
  }, [activeSelectedFamilyId, filters]);

  const activeDebugLabel = useMemo(() => {
    if (selectedDebugLabel === "none") return "none";
    return debugOverlays.some((o) => o.label === selectedDebugLabel) ? selectedDebugLabel : "none";
  }, [debugOverlays, selectedDebugLabel]);

  const debugImageUrl = useMemo(() => {
    if (activeDebugLabel === "none") return null;
    return debugOverlays.find((o) => o.label === activeDebugLabel)?.url ?? null;
  }, [debugOverlays, activeDebugLabel]);

  const downloadWaves = async (id: string) => {
    try {
      await downloadFromUrl(jobWavesCsvUrl(id), `${runStem(id)}_waves.csv`);
    } catch {
      window.alert("Could not download waves CSV for this run.");
    }
  };

  const downloadRipple = async (id: string, exportName: "tracks" | "intervals" | "families") => {
    const exportInfo = RIPPLE_EXPORT_NAMES[exportName];
    try {
      await downloadFromUrl(jobRippleCsvUrl(id, exportName), `${runStem(id)}_${exportInfo.stem}.csv`);
    } catch {
      window.alert(`Could not download ${exportInfo.label} CSV for this run.`);
    }
  };

  const downloadPrimaryAnalysis = async (id: string) => {
    const run = jobs.find((job) => job.id === id);
    if (normalizeAnalysisMode(run?.analysis_mode) === "ripple_family") {
      await downloadRipple(id, "intervals");
      return;
    }
    await downloadWaves(id);
  };

  const downloadHeatmap = async () => {
    if (!baseImageUrl) return;
    try {
      await downloadFromUrl(baseImageUrl, `${runStem(jobId)}_base_heatmap.png`);
    } catch {
      window.alert("Could not download the base heatmap.");
    }
  };

  const downloadOriginalImage = async () => {
    if (!originalImageUrl) return;
    try {
      await downloadFromUrl(originalImageUrl, `${runStem(jobId)}_original_image.png`);
    } catch {
      window.alert("Could not download the original image.");
    }
  };

  const downloadVisibleTracks = () => {
    if (analysisMode === "ripple_family") {
      const rows = filteredTracks.map((track) => {
        const familyId = trackFamilyKey(track);
        const velocity = track.metrics?.velocity_px_per_s;
        const speed = track.metrics?.speed_px_per_s ?? (velocity != null ? Math.abs(velocity) : "");
        const angle = track.metrics?.angle_from_time_axis_deg ?? track.metrics?.angle_deg ?? "";
        const frequency = track.metrics?.ripple_frequency_hz ?? track.metrics?.frequency_hz ?? track.metrics?.dominant_frequency ?? "";
        const period = track.metrics?.ripple_period_s ?? track.metrics?.period ?? "";
        const lineError = track.metrics?.line_fit_rmse_px ?? track.metrics?.line_rmse_px ?? "";
        return {
          "Track ID": track.track_index,
          Sample: track.sample ?? "",
          Family: familyLabel(familyId),
          Direction: track.metrics?.direction ?? "",
          Points: track.poly?.length ?? 0,
          "Slope (px/frame)": track.metrics?.slope_px_per_frame ?? "",
          "Velocity (pixels/sec)": velocity ?? "",
          "Speed (pixels/sec)": speed,
          "Angle from Time Axis (degrees)": angle,
          "Line Fit Error (RMSE px)": lineError,
          "Neighbor Intervals": track.metrics?.neighbor_interval_count ?? "",
          "Ripple Frequency (Hz)": frequency,
          "Ripple Period (seconds)": period,
          "Frequency Method": track.metrics?.frequency_method ?? "",
          track_index: track.track_index,
          sample: track.sample ?? "",
          family_id: track.metrics?.family_id ?? "",
          family_label: familyLabel(familyId),
          direction: track.metrics?.direction ?? "",
          point_count: track.poly?.length ?? 0,
          slope_px_per_frame: track.metrics?.slope_px_per_frame ?? "",
          velocity_px_per_s: velocity ?? "",
          speed_px_per_s: speed,
          angle_deg: angle,
          angle_from_time_axis_deg: angle,
          line_rmse_px: lineError,
          neighbor_interval_count: track.metrics?.neighbor_interval_count ?? "",
          frequency_hz: frequency,
          period_s: period,
          frequency_method: track.metrics?.frequency_method ?? "",
        };
      });
      downloadCsv(`${runStem(jobId)}_visible_tracks.csv`, rows, [
        "Track ID",
        "Sample",
        "Family",
        "Direction",
        "Points",
        "Slope (px/frame)",
        "Velocity (pixels/sec)",
        "Speed (pixels/sec)",
        "Angle from Time Axis (degrees)",
        "Line Fit Error (RMSE px)",
        "Neighbor Intervals",
        "Ripple Frequency (Hz)",
        "Ripple Period (seconds)",
        "Frequency Method",
        "track_index",
        "sample",
        "family_id",
        "family_label",
        "direction",
        "point_count",
        "slope_px_per_frame",
        "velocity_px_per_s",
        "speed_px_per_s",
        "angle_deg",
        "angle_from_time_axis_deg",
        "line_rmse_px",
        "neighbor_interval_count",
        "frequency_hz",
        "period_s",
        "frequency_method",
      ]);
      return;
    }
    const rows = filteredTracks.map((track) => ({
      track_index: track.track_index,
      sample: track.sample ?? "",
      points: track.poly?.length ?? 0,
      peaks: track.metrics?.num_peaks ?? track.peaks?.length ?? 0,
      mean_amplitude: track.metrics?.mean_amplitude ?? "",
      dominant_frequency: track.metrics?.dominant_frequency ?? "",
      period: track.metrics?.period ?? "",
      poly: track.poly ?? [],
      peak_points: track.peaks ?? [],
    }));
    downloadCsv(`${runStem(jobId)}_visible_tracks.csv`, rows, [
      "track_index",
      "sample",
      "points",
      "peaks",
      "mean_amplitude",
      "dominant_frequency",
      "period",
      "poly",
      "peak_points",
    ]);
  };

  const downloadSelectedTrack = () => {
    if (!selectedTrack || !trackDetail || trackDetail.track_index !== selectedTrack.track_index) return;
    downloadJson(`${runStem(jobId)}_track_${selectedTrack.track_index}.json`, {
      track: selectedTrack,
      detail: trackDetail,
    });
  };

  const clearTrackSelection = useCallback(() => {
    setSelectionJobId(jobId);
    setSelectedTrackId(null);
    setSelectedFamilyId(null);
    setSelectionScope(null);
  }, [jobId]);

  const handleClickTrack = useCallback((t: OverlayTrackEvent | null) => {
    if (!t) {
      clearTrackSelection();
      return;
    }

    const trackId = t.id ?? t.track_index;
    const familyId = analysisMode === "ripple_family" ? trackFamilyKey(t) : null;
    const sameTrack = selectedTrackId != null && String(selectedTrackId) === String(trackId);

    if (familyId) {
      setSelectionJobId(jobId);
      setSelectedTrackId(trackId);
      setSelectedFamilyId(familyId);
      setSelectionScope(sameTrack && selectedFamilyId === familyId && selectionScope === "family" ? "track" : "family");
      return;
    }

    setSelectionJobId(jobId);
    setSelectedTrackId(trackId);
    setSelectedFamilyId(null);
    setSelectionScope("track");
  }, [analysisMode, clearTrackSelection, jobId, selectedFamilyId, selectedTrackId, selectionScope]);

  const handleSelectFamily = useCallback((familyId: string) => {
    const familyTrack = filteredTracks.find((track) => trackFamilyKey(track) === familyId);
    setSelectionJobId(jobId);
    setSelectedFamilyId(familyId);
    setSelectedTrackId(familyTrack ? familyTrack.id ?? familyTrack.track_index : null);
    setSelectionScope("family");
  }, [filteredTracks, jobId]);

  const isolateFamily = useCallback((familyId: string) => {
    setFilters((current) => [
      ...current.filter((rule) => rule.field !== "family_id"),
      {
        id: `family_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
        field: "family_id",
        op: "==",
        value: familyFilterValue(familyId),
        value2: "",
      },
    ]);
    handleSelectFamily(familyId);
  }, [handleSelectFamily, setFilters]);

  const clearFamilyIsolation = useCallback(() => {
    setFilters((current) => current.filter((rule) => rule.field !== "family_id"));
  }, [setFilters]);

  const colorTrackByFamily = useCallback((track: OverlayTrackEvent) => {
    if (analysisMode !== "ripple_family") return undefined;
    return familyColor(trackFamilyKey(track)) ?? UNASSIGNED_RIPPLE_COLOR;
  }, [analysisMode]);

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
              clearTrackSelection();
              setSelectedDebugLabel("none");
              resetTrackDetail();
              runJob(
                file,
                mergeRunConfigWithImageProcessing(
                  {
                    ...buildHeatmapOptionsConfig(heatmapOptions),
                    ...buildAnalysisOptionsConfig(runAnalysisMode),
                  },
                  imageProcessingDimensions
                ),
                runName
              );
              refreshJobs();
            }}
            imageSizing={imageSizing}
            jobId={jobId}
            status={status}
            runName={runName}
            onRunNameChange={(value) => {
              setRunName(value);
              setRunNameAuto(false);
            }}
            heatmapOptions={heatmapOptions}
            onHeatmapOptionsChange={setHeatmapOptions}
            analysisMode={runAnalysisMode}
            onAnalysisModeChange={setRunAnalysisMode}
            filteredCount={filteredTracks.length}
            totalCount={tracks.length}
            onCancel={cancelCurrentJob}
            cancelDisabled={!jobId || ["completed", "failed", "cancelled"].includes(status)}
            onDownloadWaves={jobId && analysisMode !== "ripple_family" ? () => downloadWaves(jobId) : undefined}
            onDownloadRippleTracks={jobId && analysisMode === "ripple_family" ? () => downloadRipple(jobId, "tracks") : undefined}
            onDownloadRippleIntervals={jobId && analysisMode === "ripple_family" ? () => downloadRipple(jobId, "intervals") : undefined}
            onDownloadRippleFamilies={jobId && analysisMode === "ripple_family" ? () => downloadRipple(jobId, "families") : undefined}
            onDownloadHeatmap={downloadHeatmap}
            onDownloadOriginalImage={originalImageUrl ? downloadOriginalImage : undefined}
            heatmapDownloadDisabled={!baseImageUrl}
            originalImageDownloadDisabled={!originalImageUrl}
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
              resetImageProcessing();
              setRunName("");
              setRunNameAuto(true);
              clearTrackSelection();
              setFilters([]);
              setSelectedDebugLabel("none");
              resetTrackDetail();
              refreshJobs();
            }}
          />

          {selectedTrack ? (
            <SelectionPanel
              selectedTrack={selectedTrack}
              selectionScope={activeSelectionScope}
              selectedFamilySummary={selectedFamilySummary}
              familyIsolated={familyIsolated}
              trackDetail={trackDetail}
              trackDetailLoading={trackDetailLoading}
              trackDetailError={trackDetailError}
              overlayColor={overlayColor}
              baseImageUrl={baseImageUrl}
              frameCoordinateHeight={baseImageInfo?.outputHeight ?? baseImageInfo?.sourceRows}
              viewerProjection={viewerProjection}
              debugImageUrl={debugImageUrl}
              debugOpacity={debugOpacity}
              onDownloadTrackDetail={downloadSelectedTrack}
              onIsolateFamily={isolateFamily}
              onClearFamilyIsolation={clearFamilyIsolation}
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
              clearTrackSelection();
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
            onDownload={downloadPrimaryAnalysis}
            onDelete={async (id) => {
              const ok = window.confirm("Delete this run and its artifacts? This cannot be undone.");
              if (!ok) return;
              try {
                await deleteJob(id);
                if (id === jobId) {
                  clearSession();
                  clearTrackSelection();
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

          <SummaryPanel
            stats={filteredStats}
            analysisMode={analysisMode}
            onDownloadTracks={downloadVisibleTracks}
            downloadDisabled={filteredTracks.length === 0}
          />

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
              showOverlayColorControl={analysisMode !== "ripple_family"}
              hideBaseImage={hideBaseImage}
              onHideBaseImageChange={setHideBaseImage}
              hideTracks={hideTracks}
              onHideTracksChange={setHideTracks}
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
              coordInfo={baseImageInfo}
              heatmapValues={heatmapValues}
              debugImageUrl={debugImageUrl}
              debugOpacity={debugOpacity}
              tracks={filteredTracks}
              overlayColor={overlayColor}
              hideBaseImage={hideBaseImage}
              hideTracks={hideTracks}
              selectedTrackId={activeSelectedTrackId}
              selectedFamilyId={activeSelectedFamilyId}
              selectionScope={activeSelectionScope}
              onClickTrack={handleClickTrack}
              colorOverrideFn={analysisMode === "ripple_family" ? colorTrackByFamily : undefined}
              onProjectionChange={setViewerProjection}
            />
          </div>
          {analysisMode === "ripple_family" && familyLegend.length > 0 ? (
            <div className="viewer-family-legend" aria-label="Family colors">
              <div className="viewer-family-legend-title">Families</div>
              <div className="family-legend-items">
                {familyLegend.map((family) => (
                  <button
                    key={family.familyId}
                    type="button"
                    className={family.familyId === activeSelectedFamilyId ? "family-chip active" : "family-chip"}
                    onClick={() => handleSelectFamily(family.familyId)}
                    title={`${family.label}: ${family.count} tracks`}
                  >
                    <span className="family-swatch" style={{ backgroundColor: family.color }} aria-hidden="true" />
                    <span>{family.label}</span>
                    <span className="family-count">{family.count}</span>
                  </button>
                ))}
              </div>
            </div>
          ) : null}
          {tracks.length > 0 && filteredTracks.length === 0 ? (
            <div className="empty-text">No tracks match the current filters.</div>
          ) : null}
        </section>
      </main>
    </div>
  );
}
