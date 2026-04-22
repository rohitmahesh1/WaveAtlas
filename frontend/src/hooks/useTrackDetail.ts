import { useEffect, useState } from "react";
import { getTrackDetail } from "../api";
import type { TrackDetail } from "../api";

type TrackDetailResult = {
  requestKey: string;
  detail: TrackDetail | null;
  error: string | null;
};

function isAbortError(err: unknown) {
  return err instanceof DOMException && err.name === "AbortError";
}

export function useTrackDetail(jobId: string | null, selectedTrackId: string | number | null) {
  const [result, setResult] = useState<TrackDetailResult | null>(null);
  const rawTrackIndex = selectedTrackId == null ? null : Number(selectedTrackId);
  const trackIndex = rawTrackIndex != null && Number.isFinite(rawTrackIndex) ? rawTrackIndex : null;
  const invalidSelection = jobId && selectedTrackId != null && trackIndex == null ? "Invalid track id" : null;
  const requestKey = jobId && trackIndex != null ? `${jobId}:${trackIndex}` : null;

  useEffect(() => {
    if (!jobId || trackIndex == null || !requestKey) return;

    const controller = new AbortController();
    let active = true;

    getTrackDetail(jobId, trackIndex, {
      include_sine: true,
      include_residual: false,
      signal: controller.signal,
    })
      .then((detail) => {
        if (!active) return;
        setResult({ requestKey, detail, error: null });
      })
      .catch((err) => {
        if (!active) return;
        if (isAbortError(err)) return;
        setResult({ requestKey, detail: null, error: "Track detail unavailable" });
      });

    return () => {
      active = false;
      controller.abort();
    };
  }, [jobId, trackIndex, requestKey]);

  const resetTrackDetail = () => {
    setResult(null);
  };

  const currentResult = result?.requestKey === requestKey ? result : null;
  const trackDetail = currentResult?.detail ?? null;
  const trackDetailError = invalidSelection ?? currentResult?.error ?? null;
  const trackDetailLoading = Boolean(requestKey && !currentResult && !invalidSelection);

  return {
    trackDetail,
    trackDetailLoading,
    trackDetailError,
    resetTrackDetail,
  };
}
