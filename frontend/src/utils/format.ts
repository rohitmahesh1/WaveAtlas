export function stageLabel(stage: string | null): string {
  if (!stage) return "Working…";
  const map: Record<string, string> = {
    init: "Starting analysis",
    table_loaded: "Input loaded",
    heatmap: "Generating heatmap",
    heatmap_ready: "Heatmap ready",
    kymo_start: "Extracting tracks",
    kymo_load_image: "Loading heatmap",
    kymo_segmenting: "Segmenting heatmap",
    kymo_masking: "Cleaning mask",
    kymo_skeletonizing: "Skeletonizing tracks",
    kymo_tracking: "Tracing tracks",
    kymo_refining: "Refining tracks",
    kymo_deduping: "Removing duplicates",
    kymo_scaling: "Scaling to original size",
    kymo_saving: "Saving tracks",
    kymo_done: "Tracks extracted",
    processing_tracks: "Analyzing tracks",
    completed: "Completed",
    failed: "Failed",
    cancelled: "Cancelled",
  };
  return map[stage] || stage.replace(/_/g, " ");
}

export function formatEta(secs: number): string {
  const s = Math.max(0, Math.round(secs));
  const mins = Math.floor(s / 60);
  const rem = s % 60;
  return mins > 0 ? `${mins}m ${rem}s` : `${rem}s`;
}
