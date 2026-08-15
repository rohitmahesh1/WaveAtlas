import type { HeatmapOptions, HeatmapPalette, HeatmapProcessingMode } from "../utils/heatmapOptions";

const PROCESSING_OPTIONS: { value: HeatmapProcessingMode; label: string }[] = [
  { value: "auto", label: "Auto" },
  { value: "continuous", label: "Continuous" },
  { value: "binary", label: "Binary" },
];

const PALETTE_OPTIONS: { value: HeatmapPalette; label: string; swatch: string }[] = [
  { value: "default", label: "Analysis default", swatch: "linear-gradient(90deg, #0d0887 0%, #bd3786 50%, #f0f921 100%)" },
  { value: "gray", label: "Greyscale", swatch: "linear-gradient(90deg, #111111 0%, #8a8f94 50%, #ffffff 100%)" },
  { value: "plasma", label: "Plasma", swatch: "linear-gradient(90deg, #0d0887 0%, #bd3786 50%, #f0f921 100%)" },
  { value: "hot", label: "Hot", swatch: "linear-gradient(90deg, #050505 0%, #d7191c 58%, #fff6aa 100%)" },
];

export function HeatmapOptionsPanel(props: {
  value: HeatmapOptions;
  onChange: (value: HeatmapOptions) => void;
}) {
  const { value, onChange } = props;

  const setProcessingMode = (processingMode: HeatmapProcessingMode) => {
    onChange({ ...value, processingMode });
  };

  const setPalette = (palette: HeatmapPalette) => {
    onChange({ ...value, palette });
  };

  return (
    <div className="run-options-section">
      <div className="run-options-title">Table heatmap</div>
      <div className="run-options-grid">
        <fieldset className="option-group">
          <legend>Processing</legend>
          <div className="segmented-control">
            {PROCESSING_OPTIONS.map((option) => (
              <button
                key={option.value}
                type="button"
                className={`segmented-btn ${value.processingMode === option.value ? "active" : ""}`}
                onClick={() => setProcessingMode(option.value)}
                aria-pressed={value.processingMode === option.value}
              >
                {option.label}
              </button>
            ))}
          </div>
        </fieldset>

        <fieldset className="option-group">
          <legend>Palette</legend>
          <div className="palette-grid">
            {PALETTE_OPTIONS.map((option) => (
              <button
                key={option.value}
                type="button"
                className={`palette-btn ${value.palette === option.value ? "active" : ""}`}
                onClick={() => setPalette(option.value)}
                aria-pressed={value.palette === option.value}
              >
                <span className="palette-swatch" style={{ background: option.swatch }} aria-hidden="true" />
                <span>{option.label}</span>
              </button>
            ))}
          </div>
        </fieldset>
      </div>
    </div>
  );
}
