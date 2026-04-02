import React from "react";
import type { SummaryStats } from "../types";

export function SummaryPanel({ stats }: { stats: SummaryStats }) {
  return (
    <section className="panel">
      <div className="panel-title">Summary</div>
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
