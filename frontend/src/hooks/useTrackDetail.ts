import { useEffect, useState } from "react";
import { getTrackDetail } from "../api";
import type { TrackDetail } from "../api";

export function useTrackDetail(jobId: string | null, selectedTrackId: string | number | null) {
  const [trackDetail, setTrackDetail] = useState<TrackDetail | null>(null);
  const [trackDetailLoading, setTrackDetailLoading] = useState<boolean>(false);
  const [trackDetailError, setTrackDetailError] = useState<string | null>(null);

  useEffect(() => {
    if (!jobId || selectedTrackId == null) {
      setTrackDetail(null);
      setTrackDetailLoading(false);
      setTrackDetailError(null);
      return;
    }

    const trackIndex = Number(selectedTrackId);
    if (!Number.isFinite(trackIndex)) {
      setTrackDetail(null);
      setTrackDetailLoading(false);
      setTrackDetailError("Invalid track id");
      return;
    }

    const controller = new AbortController();
    let active = true;
    setTrackDetail(null);
    setTrackDetailLoading(true);
    setTrackDetailError(null);

    getTrackDetail(jobId, trackIndex, {
      include_sine: true,
      include_residual: false,
      signal: controller.signal,
    })
      .then((detail) => {
        if (!active) return;
        setTrackDetail(detail);
      })
      .catch((err) => {
        if (!active) return;
        if (err?.name === "AbortError") return;
        setTrackDetailError("Track detail unavailable");
      })
      .finally(() => {
        if (!active) return;
        setTrackDetailLoading(false);
      });

    return () => {
      active = false;
      controller.abort();
    };
  }, [jobId, selectedTrackId]);

  const resetTrackDetail = () => {
    setTrackDetail(null);
    setTrackDetailLoading(false);
    setTrackDetailError(null);
  };

  return {
    trackDetail,
    trackDetailLoading,
    trackDetailError,
    resetTrackDetail,
  };
}
