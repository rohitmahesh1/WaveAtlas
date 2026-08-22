import type { TrackDetail } from "../api";
import type { OverlayProjection, OverlayTrackEvent } from "../OverlayCanvas";
import { TrackDetailChart } from "./TrackDetailChart";

export function SelectionPanel(props: {
  selectedTrack: OverlayTrackEvent | null;
  selectionScope?: "family" | "track" | null;
  selectedFamilySummary?: {
    familyId: string;
    label: string;
    color: string;
    trackCount: number;
    avgSlope: number | null;
    avgSpeed: number | null;
    avgAngle: number | null;
    avgFrequency: number | null;
    directions: string[];
  } | null;
  familyIsolated?: boolean;
  trackDetail: TrackDetail | null;
  trackDetailLoading: boolean;
  trackDetailError: string | null;
  overlayColor?: string;
  baseImageUrl?: string | null;
  frameCoordinateHeight?: number | null;
  viewerProjection?: OverlayProjection | null;
  debugImageUrl?: string | null;
  debugOpacity?: number;
  onDownloadTrackDetail?: () => void;
  onIsolateFamily?: (familyId: string) => void;
  onClearFamilyIsolation?: () => void;
}) {
  const {
    selectedTrack,
    selectionScope = null,
    selectedFamilySummary = null,
    familyIsolated = false,
    trackDetail,
    trackDetailLoading,
    trackDetailError,
    overlayColor,
    baseImageUrl,
    frameCoordinateHeight,
    viewerProjection,
    debugImageUrl,
    debugOpacity,
    onDownloadTrackDetail,
    onIsolateFamily,
    onClearFamilyIsolation,
  } = props;
  const detailReady = Boolean(
    selectedTrack && trackDetail && trackDetail.track_index === selectedTrack.track_index
  );
  const rippleMode = selectedTrack?.metrics?.analysis_mode === "ripple_family";
  const largeWaveMode = selectedTrack?.metrics?.analysis_mode === "large_wave";
  const familyHighlighted = rippleMode && selectionScope === "family" && selectedFamilySummary;
  const selectedVelocity = selectedTrack?.metrics?.velocity_px_per_s ?? null;
  const selectedSpeed = selectedTrack?.metrics?.speed_px_per_s ?? (
    selectedVelocity != null ? Math.abs(selectedVelocity) : null
  );
  const selectedAngle = selectedTrack?.metrics?.angle_from_time_axis_deg ?? selectedTrack?.metrics?.angle_deg ?? null;

  return (
    <section className="panel">
      <div className="panel-title-row">
        <div className="panel-title">Selection</div>
        {selectedTrack && onDownloadTrackDetail ? (
          <button
            className="ghost-btn download-btn compact-btn"
            onClick={onDownloadTrackDetail}
            disabled={!detailReady}
          >
            Track JSON
          </button>
        ) : null}
      </div>
      <div className="panel-body">
        {selectedTrack ? (
          <>
            {familyHighlighted ? (
              <div className="family-summary">
                <div className="family-summary-header">
                  <div className="family-summary-title">
                    <span
                      className="family-swatch"
                      style={{ backgroundColor: selectedFamilySummary.color }}
                      aria-hidden="true"
                    />
                    <span>{selectedFamilySummary.label}</span>
                  </div>
                  {familyIsolated ? (
                    <button className="ghost-btn compact-btn" onClick={onClearFamilyIsolation}>
                      Show all
                    </button>
                  ) : (
                    <button
                      className="ghost-btn compact-btn"
                      onClick={() => onIsolateFamily?.(selectedFamilySummary.familyId)}
                    >
                      Isolate
                    </button>
                  )}
                </div>
                <div className="stats-grid">
                  <div>
                    Tracks
                    <div className="meta-value">{selectedFamilySummary.trackCount}</div>
                  </div>
                  <div>
                    Direction
                    <div className="meta-value">{selectedFamilySummary.directions.join(", ") || "—"}</div>
                  </div>
                  <div>
                    Avg slope
                    <div className="meta-value">
                      {selectedFamilySummary.avgSlope != null ? selectedFamilySummary.avgSlope.toFixed(3) : "—"}
                    </div>
                  </div>
                  <div>
                    Avg speed
                    <div className="meta-value">
                      {selectedFamilySummary.avgSpeed != null ? selectedFamilySummary.avgSpeed.toFixed(2) : "—"}
                    </div>
                  </div>
                  <div>
                    Avg angle
                    <div className="meta-value">
                      {selectedFamilySummary.avgAngle != null ? selectedFamilySummary.avgAngle.toFixed(1) : "—"}
                    </div>
                  </div>
                  <div>
                    Avg frequency
                    <div className="meta-value">
                      {selectedFamilySummary.avgFrequency != null
                        ? selectedFamilySummary.avgFrequency.toFixed(3)
                        : "—"}
                    </div>
                  </div>
                </div>
              </div>
            ) : null}
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
              {rippleMode ? (
                <>
                  <div>
                    Family
                    <div className="meta-value">{selectedTrack.metrics?.family_id ?? "Unassigned"}</div>
                  </div>
                  <div>
                    Direction
                    <div className="meta-value">{selectedTrack.metrics?.direction ?? "—"}</div>
                  </div>
                  <div>
                    Slope
                    <div className="meta-value">
                      {selectedTrack.metrics?.slope_px_per_frame != null
                        ? selectedTrack.metrics.slope_px_per_frame.toFixed(3)
                        : "—"}
                    </div>
                  </div>
                  <div>
                    Velocity
                    <div className="meta-value">
                      {selectedVelocity != null
                        ? selectedVelocity.toFixed(2)
                        : "—"}
                    </div>
                  </div>
                  <div>
                    Speed
                    <div className="meta-value">
                      {selectedSpeed != null
                        ? selectedSpeed.toFixed(2)
                        : "—"}
                    </div>
                  </div>
                  <div>
                    Angle
                    <div className="meta-value">
                      {selectedAngle != null
                        ? selectedAngle.toFixed(1)
                        : "—"}
                    </div>
                  </div>
                  <div>
                    Frequency
                    <div className="meta-value">
                      {selectedTrack.metrics?.dominant_frequency != null
                        ? selectedTrack.metrics.dominant_frequency.toFixed(3)
                        : "—"}
                    </div>
                  </div>
                  <div>
                    Line error
                    <div className="meta-value">
                      {selectedTrack.metrics?.line_rmse_px != null
                        ? selectedTrack.metrics.line_rmse_px.toFixed(2)
                        : "—"}
                    </div>
                  </div>
                </>
              ) : (
                <>
                  <div>
                    Peaks
                    <div className="meta-value">{selectedTrack.metrics?.num_peaks ?? 0}</div>
                  </div>
                  <div>
                    Amplitude
                    <div className="meta-value">
                      {selectedTrack.metrics?.mean_amplitude != null
                        ? selectedTrack.metrics.mean_amplitude.toFixed(2)
                        : "—"}
                    </div>
                  </div>
                  <div>
                    Frequency
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
                  {largeWaveMode ? (
                    <div>
                      Recurrence freq
                      <div className="meta-value">
                        {selectedTrack.metrics?.large_wave_recurrence_frequency_hz != null
                          ? selectedTrack.metrics.large_wave_recurrence_frequency_hz.toFixed(2)
                          : "—"}
                      </div>
                    </div>
                  ) : null}
                </>
              )}
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
                  frameCoordinateHeight={frameCoordinateHeight}
                  viewerProjection={viewerProjection}
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
