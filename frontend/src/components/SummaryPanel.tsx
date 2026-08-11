import type { SummaryStats } from "../types";
import type { AnalysisMode } from "../utils/analysisOptions";

export function SummaryPanel(props: {
  stats: SummaryStats;
  analysisMode?: AnalysisMode;
  onDownloadTracks?: () => void;
  downloadDisabled?: boolean;
}) {
  const { stats, analysisMode = "standard", onDownloadTracks, downloadDisabled } = props;
  const rippleMode = analysisMode === "ripple_family";

  return (
    <section className="panel">
      <div className="panel-title-row">
        <div className="panel-title">Summary</div>
        {onDownloadTracks ? (
          <button className="ghost-btn download-btn compact-btn" onClick={onDownloadTracks} disabled={downloadDisabled}>
            Tracks CSV
          </button>
        ) : null}
      </div>
      <div className="panel-body stats-grid">
        <div>
          {rippleMode ? "Tracks" : "Count"}
          <div className="meta-value">{stats.count}</div>
        </div>
        {rippleMode ? (
          <>
            <div>
              Families
              <div className="meta-value">{stats.familyCount}</div>
            </div>
            <div>
              Avg slope
              <div className="meta-value">{stats.avgSlope != null ? stats.avgSlope.toFixed(3) : "—"}</div>
            </div>
            <div>
              Avg speed
              <div className="meta-value">{stats.avgSpeed != null ? stats.avgSpeed.toFixed(2) : "—"}</div>
            </div>
            <div>
              Avg angle
              <div className="meta-value">{stats.avgAngle != null ? stats.avgAngle.toFixed(1) : "—"}</div>
            </div>
          </>
        ) : (
          <>
            <div>
              Points
              <div className="meta-value">{stats.points}</div>
            </div>
            <div>
              Avg amplitude
              <div className="meta-value">{stats.avgAmplitude != null ? stats.avgAmplitude.toFixed(2) : "—"}</div>
            </div>
          </>
        )}
        <div>
          Avg frequency
          <div className="meta-value">{stats.avgFrequency != null ? stats.avgFrequency.toFixed(2) : "—"}</div>
        </div>
      </div>
    </section>
  );
}
