import { useEffect, useId, useMemo, useRef, useState } from "react";
import type { TrackDetail, TrackPeakRegression } from "../api";

type ChartPoint = { x: number; y: number };
type ChartSize = { width: number; height: number };
type LoadedImageSize = { src: string; width: number; height: number };
type ChartSeries = { name: string; xs: number[]; ys: number[]; color: string; dash?: string };

const MIN_H = 180;
const MAX_H = 520;

function clamp(n: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, n));
}

function niceTicks(min: number, max: number, maxTicks: number): number[] {
  if (!isFinite(min) || !isFinite(max) || max <= min) return [min, max];
  const range = max - min;
  const rough = range / Math.max(1, maxTicks);
  const pow10 = Math.pow(10, Math.floor(Math.log10(rough)));
  const fr = rough / pow10;
  const step = fr < 1.5 ? 1 * pow10 : fr < 3 ? 2 * pow10 : fr < 7 ? 5 * pow10 : 10 * pow10;
  const t0 = Math.ceil(min / step) * step;
  const arr: number[] = [];
  for (let v = t0; v <= max + 1e-9; v += step) arr.push(Number(v.toFixed(12)));
  return arr.slice(0, Math.max(2, maxTicks + 1));
}

function formatTick(val: number, stepGuess?: number) {
  const s = Math.abs(stepGuess ?? 0);
  let decimals = 0;
  if (s > 0 && s < 1) {
    if (s >= 0.5) decimals = 1;
    else if (s >= 0.1) decimals = 1;
    else if (s >= 0.05) decimals = 2;
    else if (s >= 0.01) decimals = 2;
    else decimals = 3;
  }
  if (!isFinite(val)) return "";
  return val.toFixed(decimals);
}

function buildScale(
  width: number,
  height: number,
  pad: number,
  minX: number,
  maxX: number,
  minY: number,
  maxY: number
) {
  const spanX = maxX - minX || 1;
  const spanY = maxY - minY || 1;
  const usableW = width - pad * 2;
  const usableH = height - pad * 2;
  const scale = Math.min(usableW / spanX, usableH / spanY);
  const effW = spanX * scale;
  const effH = spanY * scale;
  const offX = pad + (usableW - effW) / 2;
  const offY = pad + (usableH - effH) / 2;
  return { scale, offX, offY, minX, minY };
}

function toCanvas(point: ChartPoint, scale: { scale: number; offX: number; offY: number; minX: number; minY: number }) {
  return {
    x: scale.offX + (point.x - scale.minX) * scale.scale,
    y: scale.offY + (point.y - scale.minY) * scale.scale,
  };
}

function buildPolyline(
  xs: number[],
  ys: number[],
  scale: { scale: number; offX: number; offY: number; minX: number; minY: number }
): string {
  const parts: string[] = [];
  const n = Math.min(xs.length, ys.length);

  for (let i = 0; i < n; i += 1) {
    const x = xs[i];
    const y = ys[i];
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    const p = toCanvas({ x, y }, scale);
    parts.push(`${p.x.toFixed(2)},${p.y.toFixed(2)}`);
  }

  return parts.join(" ");
}

export function TrackDetailChart({
  detail,
  overlayColor = "#008c5a",
  baseImageUrl,
  debugImageUrl,
  debugOpacity = 0.6,
}: {
  detail: TrackDetail;
  overlayColor?: string;
  baseImageUrl?: string | null;
  debugImageUrl?: string | null;
  debugOpacity?: number;
}) {
  const frames = detail.time_index ?? [];
  const positions = detail.position ?? [];
  const hasUsableTrack = frames.length >= 2 && positions.length >= 2 && frames.length === positions.length;
  const regressions = detail.peak_regressions ?? [];

  const [showAxes, setShowAxes] = useState<boolean>(true);
  const [showBase, setShowBase] = useState<boolean>(false);
  const [showRaw, setShowRaw] = useState<boolean>(true);
  const [showFit, setShowFit] = useState<boolean>(false);
  const [showSine, setShowSine] = useState<boolean>(true);
  const [showPeaks, setShowPeaks] = useState<boolean>(true);
  const [showRegressionWindowOnly, setShowRegressionWindowOnly] = useState<boolean>(true);
  const [selectedPeak, setSelectedPeak] = useState<{ trackIndex: number; peakI: number } | null>(null);
  const [hover, setHover] = useState<{ x: number; y: number; cx: number; cy: number } | null>(null);
  const [baseImg, setBaseImg] = useState<LoadedImageSize | null>(null);
  const [overlayImg, setOverlayImg] = useState<LoadedImageSize | null>(null);

  const selectedPeakI = selectedPeak?.trackIndex === detail.track_index ? selectedPeak.peakI : null;
  const defaultRegression = regressions.find((r) => r.peak_i === detail.strongest_peak_idx || r.is_strongest);
  const selectedRegression: TrackPeakRegression | null =
    regressions.find((r) => r.peak_i === selectedPeakI) ?? defaultRegression ?? regressions[0] ?? null;

  const activeSineFit = selectedRegression?.sine_fit ?? detail.sine_fit ?? null;
  const regressionWindowedSineFit =
    activeSineFit && selectedRegression
      ? activeSineFit.map((value, sliceIdx) => {
          const sliceStart =
            selectedRegression.slice_index != null ? selectedRegression.peak_i - selectedRegression.slice_index : 0;
          const sourceIdx = sliceStart + sliceIdx;
          const fitLo = Number(selectedRegression.fit_window_lo);
          const fitHi = Number(selectedRegression.fit_window_hi);
          return Number.isFinite(fitLo) && Number.isFinite(fitHi) && sourceIdx >= fitLo && sourceIdx <= fitHi
            ? value
            : Number.NaN;
        })
      : activeSineFit;
  const displaySineFit = showRegressionWindowOnly ? regressionWindowedSineFit : activeSineFit;

  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;

  const series: ChartSeries[] = [];
  if (showRaw) {
    series.push({ name: "Raw", xs: positions, ys: frames, color: "#1b242a" });
  }
  if (showFit && detail.baseline?.length === positions.length) {
    series.push({ name: "Fit", xs: detail.baseline, ys: frames, color: "#117a65" });
  }
  if (showSine && hasUsableTrack && displaySineFit && displaySineFit.length === positions.length) {
    series.push({ name: "Sine", xs: displaySineFit, ys: frames, color: "#d56b00", dash: "4 4" });
  }

  if (hasUsableTrack) {
    const rangeSeries = series.length ? series : [{ name: "Raw", xs: positions, ys: frames, color: "#1b242a" }];
    for (const s of rangeSeries) {
      for (const x of s.xs) {
        if (!Number.isFinite(x)) continue;
        minX = Math.min(minX, x);
        maxX = Math.max(maxX, x);
      }
      for (const y of s.ys) {
        if (!Number.isFinite(y)) continue;
        minY = Math.min(minY, y);
        maxY = Math.max(maxY, y);
      }
    }
  }

  const hasValidRange = hasUsableTrack && Number.isFinite(minX + maxX + minY + maxY);
  const safeMinX = hasValidRange ? minX : 0;
  const safeMaxX = hasValidRange ? maxX : 1;
  const safeMinY = hasValidRange ? minY : 0;
  const safeMaxY = hasValidRange ? maxY : 1;

  const spanX = Math.max(1, safeMaxX - safeMinX);
  const spanY = Math.max(1, safeMaxY - safeMinY);
  const pad = 12;

  const wrapRef = useRef<HTMLDivElement | null>(null);
  const [chartSize, setChartSize] = useState<ChartSize>({ width: 320, height: 220 });

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const update = () => {
      const w = Math.max(240, el.clientWidth || 320);
      const ar = spanX / spanY;
      const h = clamp(Math.round(w / Math.max(0.001, ar)), MIN_H, MAX_H);
      setChartSize({ width: w, height: h });
    };
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, [spanX, spanY]);

  const scale = useMemo(
    () => buildScale(chartSize.width, chartSize.height, pad, safeMinX, safeMaxX, safeMinY, safeMaxY),
    [chartSize.width, chartSize.height, pad, safeMinX, safeMaxX, safeMinY, safeMaxY]
  );
  const effW = spanX * scale.scale;
  const effH = spanY * scale.scale;
  const clipId = useId();
  const regressionSelectId = useId();

  const canToggleRegressionPeriod = Boolean(activeSineFit && selectedRegression);
  const periodToggleTitle = !canToggleRegressionPeriod
    ? "No regression period toggle is available for this track."
    : showRegressionWindowOnly
      ? "Showing only the regression fit window. Click to show the full fitted period."
      : "Showing the full fitted period. Click to limit the sine overlay to the regression fit window.";

  useEffect(() => {
    if (!showBase || !baseImageUrl) return;
    let cancelled = false;
    const im = new Image();
    im.crossOrigin = "anonymous";
    im.onload = () => {
      if (!cancelled) setBaseImg({ src: baseImageUrl, width: im.naturalWidth, height: im.naturalHeight });
    };
    im.onerror = () => {
      if (!cancelled) setBaseImg({ src: baseImageUrl, width: 0, height: 0 });
    };
    im.src = baseImageUrl;
    return () => {
      cancelled = true;
    };
  }, [showBase, baseImageUrl]);

  useEffect(() => {
    if (!showBase || !debugImageUrl) return;
    let cancelled = false;
    const im = new Image();
    im.crossOrigin = "anonymous";
    im.onload = () => {
      if (!cancelled) setOverlayImg({ src: debugImageUrl, width: im.naturalWidth, height: im.naturalHeight });
    };
    im.onerror = () => {
      if (!cancelled) setOverlayImg({ src: debugImageUrl, width: 0, height: 0 });
    };
    im.src = debugImageUrl;
    return () => {
      cancelled = true;
    };
  }, [showBase, debugImageUrl]);

  if (!hasValidRange) {
    return <div className="empty-text">Track detail unavailable.</div>;
  }

  const activeBaseImg =
    showBase && baseImageUrl && baseImg?.src === baseImageUrl && baseImg.width > 0 ? baseImg : null;
  const activeOverlayImg =
    showBase && debugImageUrl && overlayImg?.src === debugImageUrl && overlayImg.width > 0 ? overlayImg : null;

  const polylines = series.map((s) => ({
    name: s.name,
    color: s.name === "Raw" ? overlayColor : s.color,
    dash: s.dash,
    points: buildPolyline(s.xs, s.ys, scale),
  }));

  const peakPoints: (ChartPoint & { peak_i?: number })[] = [];
  if (detail.peak_points?.length) {
    for (const peak of detail.peak_points) {
      if (peak.in_slice === false) continue;
      if (!Number.isFinite(peak.position) || !Number.isFinite(peak.frame)) continue;
      peakPoints.push({ x: peak.position, y: peak.frame, peak_i: peak.peak_i });
    }
  } else {
    const peakIndices = detail.peaks_in_slice?.length ? detail.peaks_in_slice : detail.peaks ?? [];
    for (const idx of peakIndices) {
      const i = Number(idx);
      if (!Number.isFinite(i) || i < 0 || i >= positions.length) continue;
      const x = positions[i];
      const y = frames[i];
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      peakPoints.push({ x, y, peak_i: i });
    }
  }

  return (
    <div className="mini-chart">
      <div
        className="mini-controls"
        style={{
          display: "grid",
          gridTemplateColumns: "max-content 1fr",
          columnGap: 8,
          rowGap: 8,
          alignItems: "center",
        }}
      >
        {regressions.length > 1 ? (
          <>
            <label htmlFor={regressionSelectId} style={{ whiteSpace: "nowrap" }}>
              Regression
            </label>
            <select
              id={regressionSelectId}
              className="mini-select"
              value={selectedRegression?.peak_i ?? ""}
              onChange={(e) => setSelectedPeak({ trackIndex: detail.track_index, peakI: Number(e.target.value) })}
            >
              {regressions.map((r) => (
                <option key={r.peak_i} value={r.peak_i}>
                  Peak {r.peak_index} - frame {formatTick(r.frame)} - x {formatTick(r.position)}
                </option>
              ))}
            </select>
          </>
        ) : null}

        <span style={{ whiteSpace: "nowrap" }}>Layers</span>
        <div className="mini-layer-strip" aria-label="Track preview layers">
          <label className={showAxes ? "mini-layer-chip active" : "mini-layer-chip"}>
            <input type="checkbox" checked={showAxes} onChange={(e) => setShowAxes(e.target.checked)} />
            Axes
          </label>
          <label className={showRaw ? "mini-layer-chip active" : "mini-layer-chip"}>
            <input type="checkbox" checked={showRaw} onChange={(e) => setShowRaw(e.target.checked)} />
            Raw
          </label>
          <label className={showFit ? "mini-layer-chip active" : "mini-layer-chip"}>
            <input type="checkbox" checked={showFit} onChange={(e) => setShowFit(e.target.checked)} />
            Fit
          </label>
          <label className={showSine ? "mini-layer-chip active" : "mini-layer-chip"}>
            <input
              type="checkbox"
              checked={showSine}
              onChange={(e) => setShowSine(e.target.checked)}
              disabled={!activeSineFit}
            />
            Sine
          </label>
          <label className={showPeaks ? "mini-layer-chip active" : "mini-layer-chip"}>
            <input type="checkbox" checked={showPeaks} onChange={(e) => setShowPeaks(e.target.checked)} />
            Peaks
          </label>
          <label className={showBase ? "mini-layer-chip active" : "mini-layer-chip"}>
            <input type="checkbox" checked={showBase} onChange={(e) => setShowBase(e.target.checked)} />
            Base
          </label>
        </div>
      </div>

      <div
        ref={wrapRef}
        className="mini-chart-canvas"
        style={{ height: `${chartSize.height}px` }}
        onMouseLeave={() => setHover(null)}
      >
        <svg
          className="mini-svg"
          viewBox={`0 0 ${chartSize.width} ${chartSize.height}`}
          role="img"
          aria-label="Track detail"
          onMouseMove={(e) => {
            const rect = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
            const cx = e.clientX - rect.left;
            const cy = e.clientY - rect.top;
            const inX = cx >= scale.offX && cx <= scale.offX + effW;
            const inY = cy >= scale.offY && cy <= scale.offY + effH;
            if (!inX || !inY) {
              setHover(null);
              return;
            }
            const x = safeMinX + (cx - scale.offX) / scale.scale;
            const y = safeMinY + (cy - scale.offY) / scale.scale;
            setHover({ x, y, cx, cy });
          }}
        >
          <defs>
            <clipPath id={clipId}>
              <rect x={scale.offX} y={scale.offY} width={effW} height={effH} />
            </clipPath>
          </defs>

          {activeBaseImg ? (
            <image
              href={baseImageUrl ?? ""}
              x={scale.offX - safeMinX * scale.scale}
              y={scale.offY - safeMinY * scale.scale}
              width={activeBaseImg.width * scale.scale}
              height={activeBaseImg.height * scale.scale}
              preserveAspectRatio="none"
              opacity={0.7}
              clipPath={`url(#${clipId})`}
            />
          ) : null}

          {activeOverlayImg ? (
            <image
              href={debugImageUrl ?? ""}
              x={scale.offX - safeMinX * scale.scale}
              y={scale.offY - safeMinY * scale.scale}
              width={activeOverlayImg.width * scale.scale}
              height={activeOverlayImg.height * scale.scale}
              preserveAspectRatio="none"
              opacity={debugOpacity}
              clipPath={`url(#${clipId})`}
            />
          ) : null}

          {showAxes ? (
            <>
              <rect
                x={scale.offX}
                y={scale.offY}
                width={effW}
                height={effH}
                fill="none"
                stroke="rgba(148,163,184,0.35)"
                strokeWidth={1}
              />
              {(() => {
                const maxXTicks = clamp(Math.floor(effW / 80), 2, 6);
                const maxYTicks = clamp(Math.floor(effH / 60), 2, 6);
                const xt = niceTicks(safeMinX, safeMaxX, maxXTicks);
                const yt = niceTicks(safeMinY, safeMaxY, maxYTicks);
                const xStep = xt.length >= 2 ? Math.abs(xt[1] - xt[0]) : undefined;
                const yStep = yt.length >= 2 ? Math.abs(yt[1] - yt[0]) : undefined;

                return (
                  <>
                    {xt.map((v) => {
                      const cx = scale.offX + (v - safeMinX) * scale.scale;
                      return (
                        <g key={`xt-${v}`}>
                          <line
                            x1={cx}
                            y1={scale.offY}
                            x2={cx}
                            y2={scale.offY + effH}
                            stroke="rgba(148,163,184,0.18)"
                            strokeDasharray="3 4"
                            strokeWidth={1}
                          />
                          <text
                            x={cx}
                            y={scale.offY + effH + 12}
                            textAnchor="middle"
                            fill="rgba(203,213,225,0.8)"
                            fontSize="10"
                            fontFamily="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
                          >
                            {formatTick(v, xStep)}
                          </text>
                        </g>
                      );
                    })}

                    {yt.map((v) => {
                      const cy = scale.offY + (v - safeMinY) * scale.scale;
                      return (
                        <g key={`yt-${v}`}>
                          <line
                            x1={scale.offX}
                            y1={cy}
                            x2={scale.offX + effW}
                            y2={cy}
                            stroke="rgba(148,163,184,0.18)"
                            strokeDasharray="3 4"
                            strokeWidth={1}
                          />
                          <text
                            x={scale.offX - 6}
                            y={cy + 3}
                            textAnchor="end"
                            fill="rgba(203,213,225,0.8)"
                            fontSize="10"
                            fontFamily="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
                          >
                            {formatTick(v, yStep)}
                          </text>
                        </g>
                      );
                    })}
                  </>
                );
              })()}
            </>
          ) : null}

          {polylines.map((s) => (
            <polyline
              key={s.name}
              points={s.points}
              fill="none"
              stroke={s.color}
              strokeWidth={s.name === "Raw" ? 1.8 : 1.4}
              strokeDasharray={s.dash}
              strokeLinejoin="round"
              strokeLinecap="round"
            />
          ))}

          {showPeaks
            ? peakPoints.map((p, i) => {
                const scaled = toCanvas(p, scale);
                const selected = selectedRegression != null && p.peak_i === selectedRegression.peak_i;
                return (
                  <circle
                    key={`peak-${p.peak_i ?? i}`}
                    cx={scaled.x}
                    cy={scaled.y}
                    r={selected ? 4 : 2.6}
                    className={selected ? "mini-peak selected" : "mini-peak"}
                  />
                );
              })
            : null}
        </svg>

        {hover ? (
          <div className="mini-tooltip" style={{ left: hover.cx + 10, top: hover.cy + 10 }}>
            x {hover.x.toFixed(1)}, frame {hover.y.toFixed(1)}
          </div>
        ) : null}
      </div>

      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 12,
        }}
      >
        <div className="mini-legend">
          {showRaw ? (
            <span className="legend-item">
              <span className="legend-swatch swatch-raw" />
              Raw
            </span>
          ) : null}

          {showFit ? (
            <span className="legend-item">
              <span className="legend-swatch swatch-fit" />
              Fit
            </span>
          ) : null}

          {showSine && displaySineFit ? (
            <span className="legend-item">
              <span className="legend-swatch swatch-sine" />
              Sine
            </span>
          ) : null}

          {showPeaks ? (
            <span className="legend-item">
              <span className="legend-swatch swatch-peak" />
              Peaks
            </span>
          ) : null}
        </div>

        <button
          type="button"
          aria-pressed={showRegressionWindowOnly}
          disabled={!canToggleRegressionPeriod}
          onClick={() => setShowRegressionWindowOnly((value) => !value)}
          title={periodToggleTitle}
          style={{
            marginLeft: "auto",
            padding: "4px 10px",
            borderRadius: 999,
            border: "1px solid rgba(148,163,184,0.28)",
            background: "rgba(15,23,42,0.75)",
            color: "rgba(226,232,240,0.92)",
            fontSize: 12,
            lineHeight: 1.2,
            cursor: canToggleRegressionPeriod ? "pointer" : "not-allowed",
            opacity: canToggleRegressionPeriod ? 1 : 0.5,
            whiteSpace: "nowrap",
          }}
        >
          {showRegressionWindowOnly ? "Full period" : "Fit window"}
        </button>
      </div>
    </div>
  );
}