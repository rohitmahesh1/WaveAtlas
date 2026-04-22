export function ViewerControls(props: {
  overlayColor: string;
  onOverlayColorChange: (value: string) => void;
  onOverlayColorReset: () => void;
  hideBaseImage: boolean;
  onHideBaseImageChange: (value: boolean) => void;
  debugOverlays: { label: string; url: string }[];
  selectedDebugLabel: string;
  onDebugLabelChange: (value: string) => void;
  debugOpacity: number;
  onDebugOpacityChange: (value: number) => void;
}) {
  const {
    overlayColor,
    onOverlayColorChange,
    onOverlayColorReset,
    hideBaseImage,
    onHideBaseImageChange,
    debugOverlays,
    selectedDebugLabel,
    onDebugLabelChange,
    debugOpacity,
    onDebugOpacityChange,
  } = props;

  return (
    <div className="viewer-controls">
      <div className="color-control">
        <label>
          <span className="color-icon" aria-hidden="true" />
          Overlay color
          <input
            type="color"
            value={overlayColor}
            onChange={(e) => onOverlayColorChange(e.target.value)}
          />
        </label>
        <button className="ghost-btn color-reset" onClick={onOverlayColorReset}>
          Reset
        </button>
      </div>
      <label className="toggle">
        <input type="checkbox" checked={hideBaseImage} onChange={(e) => onHideBaseImageChange(e.target.checked)} />
        Hide base
      </label>
      <label>
        Debug overlay
        <select
          value={selectedDebugLabel}
          onChange={(e) => onDebugLabelChange(e.target.value)}
          disabled={debugOverlays.length === 0}
        >
          <option value="none">None</option>
          {debugOverlays.map((o) => (
            <option key={o.label} value={o.label}>
              {o.label}
            </option>
          ))}
        </select>
      </label>
      <label>
        Opacity
        <input
          type="range"
          min="0"
          max="1"
          step="0.05"
          value={debugOpacity}
          onChange={(e) => onDebugOpacityChange(Number(e.target.value))}
          disabled={selectedDebugLabel === "none"}
        />
      </label>
    </div>
  );
}
