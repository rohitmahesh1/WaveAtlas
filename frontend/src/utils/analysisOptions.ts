export type AnalysisMode = "standard" | "ripple_family" | "large_wave";

export const DEFAULT_ANALYSIS_MODE: AnalysisMode = "standard";

export function buildAnalysisOptionsConfig(mode: AnalysisMode): Record<string, unknown> {
  return { analysis: { mode } };
}

export function normalizeAnalysisMode(value: unknown): AnalysisMode {
  if (value === "ripple_family" || value === "ripple") return "ripple_family";
  if (
    value === "large_wave"
    || value === "large_waves"
    || value === "large-wave"
    || value === "large-waves"
    || value === "large"
  ) {
    return "large_wave";
  }
  return "standard";
}
