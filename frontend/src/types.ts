import type { OverlayTrackEvent } from "./OverlayCanvas";

export type LogEntry = {
  id: string;
  ts: string;
  message: string;
  level: "info" | "warn" | "error";
  stage?: string;
};

export type FieldType = "number" | "string";
export type FilterOp = ">" | "<" | ">=" | "<=" | "==" | "!=" | "between" | "contains";
export type FilterField =
  | "track_index"
  | "points"
  | "num_peaks"
  | "mean_amplitude"
  | "dominant_frequency"
  | "period"
  | "family_id"
  | "direction"
  | "slope_px_per_frame"
  | "velocity_px_per_s"
  | "speed_px_per_s"
  | "angle_deg"
  | "line_rmse_px"
  | "sample";

export type FilterRule = {
  id: string;
  field: FilterField;
  op: FilterOp;
  value?: string;
  value2?: string;
};

export type FieldDef = {
  key: FilterField;
  label: string;
  type: FieldType;
  ops: FilterOp[];
  get: (t: OverlayTrackEvent) => number | string | null | undefined;
};

export type SummaryStats = {
  count: number;
  points: number;
  avgAmplitude: number | null;
  avgFrequency: number | null;
  avgSlope: number | null;
  avgSpeed: number | null;
  avgAngle: number | null;
  familyCount: number;
};
