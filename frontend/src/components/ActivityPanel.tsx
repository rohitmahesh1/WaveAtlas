import React from "react";
import type { LogEntry } from "../types";

export function ActivityPanel({ activity }: { activity: LogEntry[] }) {
  return (
    <section className="panel">
      <div className="panel-title">Activity</div>
      <div className="panel-body">
        {activity.length === 0 ? (
          <div className="empty-text">No updates yet.</div>
        ) : (
          <div className="activity-list">
            {activity.map((item) => (
              <div key={item.id} className={`activity-item activity-${item.level}`}>
                <div className="activity-time">{item.ts}</div>
                <div className="activity-msg">{item.message}</div>
              </div>
            ))}
          </div>
        )}
      </div>
    </section>
  );
}
