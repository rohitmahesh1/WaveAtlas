import type { AnalysisMode } from "../utils/analysisOptions";

const ANALYSIS_OPTIONS: { value: AnalysisMode; label: string }[] = [
  { value: "standard", label: "Standard" },
  { value: "ripple_family", label: "Ripple waves" },
  { value: "large_wave", label: "Large waves" },
];

export function AnalysisOptionsPanel(props: {
  value: AnalysisMode;
  onChange: (value: AnalysisMode) => void;
}) {
  const { value, onChange } = props;

  return (
    <div className="run-options-section">
      <div className="run-options-title">Analysis</div>
      <fieldset className="option-group">
        <legend>Method</legend>
        <div className="segmented-control">
          {ANALYSIS_OPTIONS.map((option) => (
            <button
              key={option.value}
              type="button"
              className={`segmented-btn ${value === option.value ? "active" : ""}`}
              onClick={() => onChange(option.value)}
              aria-pressed={value === option.value}
            >
              {option.label}
            </button>
          ))}
        </div>
      </fieldset>
    </div>
  );
}
