export type AnalysisMode = "standard" | "ripple_family";

export const DEFAULT_ANALYSIS_MODE: AnalysisMode = "standard";

export function buildAnalysisOptionsConfig(mode: AnalysisMode): Record<string, unknown> {
  return { analysis: { mode } };
}

export function normalizeAnalysisMode(value: unknown): AnalysisMode {
  return value === "ripple_family" || value === "ripple" ? "ripple_family" : "standard";
}
