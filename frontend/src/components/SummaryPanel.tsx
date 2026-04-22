import type { SummaryStats } from "../types";

export function SummaryPanel(props: {
  stats: SummaryStats;
  onDownloadTracks?: () => void;
  downloadDisabled?: boolean;
}) {
  const { stats, onDownloadTracks, downloadDisabled } = props;

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
          Count
          <div className="meta-value">{stats.count}</div>
        </div>
        <div>
          Points
          <div className="meta-value">{stats.points}</div>
        </div>
        <div>
          Avg amplitude
          <div className="meta-value">{stats.avgAmplitude != null ? stats.avgAmplitude.toFixed(2) : "—"}</div>
        </div>
        <div>
          Avg frequency
          <div className="meta-value">{stats.avgFrequency != null ? stats.avgFrequency.toFixed(2) : "—"}</div>
        </div>
      </div>
    </section>
  );
}
