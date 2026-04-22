import { useEffect, useId, useMemo, useRef, useState } from "react";
import type { TrackDetail } from "../api";

type ChartPoint = { x: number; y: number };
type ChartSize = { width: number; height: number };
type LoadedImageSize = { src: string; width: number; height: number };

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
  const xs = detail.time_index ?? [];
  const ys = detail.position ?? [];
  const hasUsableTrack = xs.length >= 2 && ys.length >= 2 && xs.length === ys.length;

  const [showAxes, setShowAxes] = useState<boolean>(false);
  const [showBase, setShowBase] = useState<boolean>(false);
  const [hover, setHover] = useState<{ x: number; y: number; cx: number; cy: number } | null>(null);
  const [baseImg, setBaseImg] = useState<LoadedImageSize | null>(null);
  const [overlayImg, setOverlayImg] = useState<LoadedImageSize | null>(null);

  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;

  const series: { name: string; ys: number[]; color: string; dash?: string }[] = [
    { name: "Raw", ys, color: "#1b242a" },
    { name: "Fit", ys: detail.baseline ?? [], color: "#117a65" },
  ];
  if (hasUsableTrack && detail.sine_fit && detail.sine_fit.length === ys.length) {
    series.push({ name: "Sine", ys: detail.sine_fit, color: "#d56b00", dash: "4 4" });
  }

  if (hasUsableTrack) {
    for (const v of xs) {
      if (!Number.isFinite(v)) continue;
      minX = Math.min(minX, v);
      maxX = Math.max(maxX, v);
    }

    for (const s of series) {
      for (const v of s.ys) {
        if (!Number.isFinite(v)) continue;
        minY = Math.min(minY, v);
        maxY = Math.max(maxY, v);
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
    points: buildPolyline(xs, s.ys, scale),
  }));

  const peakIndices = detail.peaks_in_slice?.length ? detail.peaks_in_slice : detail.peaks ?? [];
  const peakPoints: ChartPoint[] = [];
  for (const idx of peakIndices) {
    const i = Number(idx);
    if (!Number.isFinite(i) || i < 0 || i >= ys.length) continue;
    const x = xs[i];
    const y = ys[i];
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    peakPoints.push({ x, y });
  }

  return (
    <div className="mini-chart">
      <div className="mini-controls">
        <label className="mini-toggle">
          <input type="checkbox" checked={showAxes} onChange={(e) => setShowAxes(e.target.checked)} />
          Axes
        </label>
        <label className="mini-toggle">
          <input type="checkbox" checked={showBase} onChange={(e) => setShowBase(e.target.checked)} />
          Base + overlay
        </label>
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
          {peakPoints.map((p, i) => {
            const scaled = toCanvas(p, scale);
            return <circle key={`peak-${i}`} cx={scaled.x} cy={scaled.y} r={2.6} className="mini-peak" />;
          })}
        </svg>
        {hover ? (
          <div className="mini-tooltip" style={{ left: hover.cx + 10, top: hover.cy + 10 }}>
            x {hover.x.toFixed(1)}, y {hover.y.toFixed(1)}
          </div>
        ) : null}
      </div>
      <div className="mini-legend">
        <span className="legend-item">
          <span className="legend-swatch swatch-raw" />
          Raw
        </span>
        <span className="legend-item">
          <span className="legend-swatch swatch-fit" />
          Fit
        </span>
        {detail.sine_fit ? (
          <span className="legend-item">
            <span className="legend-swatch swatch-sine" />
            Sine
          </span>
        ) : null}
        <span className="legend-item">
          <span className="legend-swatch swatch-peak" />
          Peaks
        </span>
      </div>
    </div>
  );
}
