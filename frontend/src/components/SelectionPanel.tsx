import type { TrackDetail } from "../api";
import type { OverlayTrackEvent } from "../OverlayCanvas";
import { TrackDetailChart } from "./TrackDetailChart";

export function SelectionPanel(props: {
  selectedTrack: OverlayTrackEvent | null;
  trackDetail: TrackDetail | null;
  trackDetailLoading: boolean;
  trackDetailError: string | null;
  overlayColor?: string;
  baseImageUrl?: string | null;
  debugImageUrl?: string | null;
  debugOpacity?: number;
}) {
  const { selectedTrack, trackDetail, trackDetailLoading, trackDetailError, overlayColor, baseImageUrl, debugImageUrl, debugOpacity } =
    props;

  return (
    <section className="panel">
      <div className="panel-title">Selection</div>
      <div className="panel-body">
        {selectedTrack ? (
          <>
            <div className="stats-grid">
              <div>
                ID
                <div className="meta-value">{String(selectedTrack.id ?? selectedTrack.track_index)}</div>
              </div>
              <div>
                Sample
                <div className="meta-value">{selectedTrack.sample ?? "—"}</div>
              </div>
              <div>
                Points
                <div className="meta-value">{selectedTrack.poly?.length ?? 0}</div>
              </div>
              <div>
                Peaks
                <div className="meta-value">{selectedTrack.metrics?.num_peaks ?? 0}</div>
              </div>
              <div>
                Mean amplitude
                <div className="meta-value">
                  {selectedTrack.metrics?.mean_amplitude != null
                    ? selectedTrack.metrics.mean_amplitude.toFixed(2)
                    : "—"}
                </div>
              </div>
              <div>
                Dominant freq
                <div className="meta-value">
                  {selectedTrack.metrics?.dominant_frequency != null
                    ? selectedTrack.metrics.dominant_frequency.toFixed(2)
                    : "—"}
                </div>
              </div>
              <div>
                Period
                <div className="meta-value">
                  {selectedTrack.metrics?.period != null ? selectedTrack.metrics.period.toFixed(2) : "—"}
                </div>
              </div>
            </div>
            <div className="track-detail">
              <div className="track-detail-title">Track preview</div>
              {trackDetailLoading ? (
                <div className="empty-text">Loading track detail…</div>
              ) : trackDetail && trackDetail.track_index === selectedTrack.track_index ? (
                <TrackDetailChart
                  detail={trackDetail}
                  overlayColor={overlayColor}
                  baseImageUrl={baseImageUrl}
                  debugImageUrl={debugImageUrl}
                  debugOpacity={debugOpacity}
                />
              ) : (
                <div className="empty-text">No detail yet for this track.</div>
              )}
              {trackDetailError ? <div className="error-text">{trackDetailError}</div> : null}
            </div>
          </>
        ) : (
          <div className="empty-text">Click a track to inspect its stats.</div>
        )}
      </div>
    </section>
  );
}
