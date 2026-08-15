import { useEffect, useId, useMemo, useRef, useState } from "react";
import type { TrackDetail, TrackPeakRegression } from "../api";

type ChartPoint = { x: number; y: number };
type ChartSize = { width: number; height: number };
type LoadedImageSize = { src: string; width: number; height: number };
type ChartMargins = { top: number; right: number; bottom: number; left: number };
type ChartScale = {
  xScale: number;
  yScale: number;
  plotX: number;
  plotY: number;
  plotWidth: number;
  plotHeight: number;
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
};
type ChartSeries = {
  name: "Raw" | "Baseline" | "Regression";
  xs: number[];
  ys: number[];
  color: string;
  dash?: string;
  strokeWidth: number;
};
type PeakChartPoint = ChartPoint & {
  peakI: number;
  peakIndex?: number;
  frame: number;
  position: number;
};
type HoverPoint = { x: number; y: number; cx: number; cy: number; peak?: PeakChartPoint };

const MIN_H = 340;
const MAX_H = 520;
const CHART_MARGINS: ChartMargins = { top: 14, right: 14, bottom: 42, left: 52 };

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
  margins: ChartMargins,
  minX: number,
  maxX: number,
  minY: number,
  maxY: number
): ChartScale {
  const requestedSpanX = Math.max(1e-6, maxX - minX);
  const requestedSpanY = Math.max(1e-6, maxY - minY);
  const plotWidth = Math.max(1, width - margins.left - margins.right);
  const plotHeight = Math.max(1, height - margins.top - margins.bottom);
  const unitScale = Math.min(plotWidth / requestedSpanX, plotHeight / requestedSpanY);
  const spanX = plotWidth / unitScale;
  const spanY = plotHeight / unitScale;
  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;
  const displayMinX = centerX - spanX / 2;
  const displayMinY = centerY - spanY / 2;
  return {
    xScale: unitScale,
    yScale: unitScale,
    plotX: margins.left,
    plotY: margins.top,
    plotWidth,
    plotHeight,
    minX: displayMinX,
    maxX: displayMinX + spanX,
    minY: displayMinY,
    maxY: displayMinY + spanY,
  };
}

function zoomDomain(min: number, max: number, focus: number, factor: number) {
  const span = Math.max(1e-6, max - min);
  const nextSpan = Math.max(1e-3, span / factor);
  const ratio = (focus - min) / span;
  const nextMin = focus - nextSpan * ratio;
  return { min: nextMin, max: nextMin + nextSpan };
}

function clampDomain(min: number, max: number, boundMin: number, boundMax: number) {
  const span = max - min;
  const boundSpan = boundMax - boundMin;
  if (!Number.isFinite(span) || span >= boundSpan) {
    return { min: boundMin, max: boundMax };
  }

  let nextMin = min;
  let nextMax = max;
  if (nextMin < boundMin) {
    nextMin = boundMin;
    nextMax = boundMin + span;
  }
  if (nextMax > boundMax) {
    nextMax = boundMax;
    nextMin = boundMax - span;
  }
  return { min: nextMin, max: nextMax };
}

function toCanvas(
  point: ChartPoint,
  scale: ChartScale
) {
  return {
    x: scale.plotX + (point.x - scale.minX) * scale.xScale,
    y: scale.plotY + (point.y - scale.minY) * scale.yScale,
  };
}

function buildPolyline(
  xs: number[],
  ys: number[],
  scale: ChartScale
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
  frameCoordinateHeight,
  debugImageUrl,
  debugOpacity = 0.6,
}: {
  detail: TrackDetail;
  overlayColor?: string;
  baseImageUrl?: string | null;
  frameCoordinateHeight?: number | null;
  debugImageUrl?: string | null;
  debugOpacity?: number;
}) {
  const rippleMode = detail.analysis_mode === "ripple_family";
  const largeWaveMode = detail.analysis_mode === "large_wave";
  const frames = detail.time_index ?? [];
  const positions = detail.position ?? [];
  const hasUsableTrack = frames.length >= 2 && positions.length >= 2 && frames.length === positions.length;
  const regressions = detail.peak_regressions ?? [];

  const [showAxes, setShowAxes] = useState<boolean>(true);
  const [showBase, setShowBase] = useState<boolean>(false);
  const [showRaw, setShowRaw] = useState<boolean>(true);
  const [showFit, setShowFit] = useState<boolean>(rippleMode);
  const [showSine, setShowSine] = useState<boolean>(!rippleMode);
  const [showPeaks, setShowPeaks] = useState<boolean>(!rippleMode);
  const [showRegressionWindowOnly, setShowRegressionWindowOnly] = useState<boolean>(true);
  const [selectedPeak, setSelectedPeak] = useState<{ trackIndex: number; peakI: number } | null>(null);
  const [hover, setHover] = useState<HoverPoint | null>(null);
  const dragRef = useRef<{ x: number; y: number; minX: number; maxX: number; minY: number; maxY: number } | null>(null);
  const [baseImg, setBaseImg] = useState<LoadedImageSize | null>(null);
  const [overlayImg, setOverlayImg] = useState<LoadedImageSize | null>(null);

  const selectedPeakI = selectedPeak?.trackIndex === detail.track_index ? selectedPeak.peakI : null;
  const defaultRegression = regressions.find((r) => r.peak_i === detail.strongest_peak_idx || r.is_strongest);
  const selectedRegression: TrackPeakRegression | null =
    regressions.find((r) => r.peak_i === selectedPeakI) ?? defaultRegression ?? regressions[0] ?? null;

  const activeSineFit = selectedRegression?.sine_fit ?? detail.sine_fit ?? null;
  const activeBaseline = selectedRegression?.fit_baseline ?? detail.baseline ?? null;
  const regressionWindowedBaseline =
    activeBaseline && selectedRegression?.fit_baseline
      ? activeBaseline.map((value, sliceIdx) => {
          const sliceStart =
            selectedRegression.slice_index != null ? selectedRegression.peak_i - selectedRegression.slice_index : 0;
          const sourceIdx = sliceStart + sliceIdx;
          const fitLo = Number(selectedRegression.fit_window_lo);
          const fitHi = Number(selectedRegression.fit_window_hi);
          return Number.isFinite(fitLo) && Number.isFinite(fitHi) && sourceIdx >= fitLo && sourceIdx <= fitHi
            ? value
            : Number.NaN;
        })
      : activeBaseline;
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
  const displaySineFit = largeWaveMode || showRegressionWindowOnly ? regressionWindowedSineFit : activeSineFit;

  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;

  const series: ChartSeries[] = [];
  if (showRaw) {
    series.push({ name: "Raw", xs: positions, ys: frames, color: overlayColor, strokeWidth: 2.4 });
  }
  if (showFit && regressionWindowedBaseline?.length === positions.length) {
    series.push({ name: "Baseline", xs: regressionWindowedBaseline, ys: frames, color: "#2dd4bf", strokeWidth: 2 });
  }
  if (showSine && hasUsableTrack && displaySineFit && displaySineFit.length === positions.length) {
    series.push({
      name: "Regression",
      xs: displaySineFit,
      ys: frames,
      color: "#ffad33",
      dash: largeWaveMode ? undefined : "7 4",
      strokeWidth: largeWaveMode ? 3.2 : 2.8,
    });
  }

  if (hasUsableTrack) {
    const xSources = [positions, regressionWindowedBaseline ?? []];
    for (const values of xSources) {
      for (const x of values) {
        if (!Number.isFinite(x)) continue;
        minX = Math.min(minX, x);
        maxX = Math.max(maxX, x);
      }
    }
    for (const y of frames) {
      if (!Number.isFinite(y)) continue;
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
    }
  }

  const hasValidRange = hasUsableTrack && Number.isFinite(minX + maxX + minY + maxY);
  const xPadding = hasValidRange ? Math.max(2, (maxX - minX) * 0.08) : 0;
  const safeMinX = hasValidRange ? minX - xPadding : 0;
  const safeMaxX = hasValidRange ? maxX + xPadding : 1;
  const safeMinY = hasValidRange ? minY : 0;
  const safeMaxY = hasValidRange ? maxY : 1;
  const resolvedCoordinateHeight = Number(frameCoordinateHeight);
  const coordinateMaxY = Number.isFinite(resolvedCoordinateHeight) && resolvedCoordinateHeight > 0
    ? resolvedCoordinateHeight - 1
    : safeMaxY;
  const toDisplayFrame = (sourceFrame: number) => coordinateMaxY - sourceFrame;
  const toSourceFrame = (displayFrame: number) => coordinateMaxY - displayFrame;
  const viewKey = `${detail.track_index}:${safeMinX}:${safeMaxX}:${safeMinY}:${safeMaxY}`;
  const [viewDomain, setViewDomain] = useState<{
    key: string;
    minX: number;
    maxX: number;
    minY: number;
    maxY: number;
  } | null>(null);

  const domain = viewDomain?.key === viewKey ? viewDomain : {
    minX: safeMinX,
    maxX: safeMaxX,
    minY: safeMinY,
    maxY: safeMaxY,
  };

  const wrapRef = useRef<HTMLDivElement | null>(null);
  const [chartSize, setChartSize] = useState<ChartSize>({ width: 320, height: 430 });

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const update = () => {
      const w = Math.max(240, el.clientWidth || 320);
      const h = clamp(Math.round(w * 1.35), MIN_H, MAX_H);
      setChartSize({ width: w, height: h });
    };
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, [hasValidRange]);

  const scale = useMemo(
    () =>
      buildScale(
        chartSize.width,
        chartSize.height,
        CHART_MARGINS,
        domain.minX,
        domain.maxX,
        domain.minY,
        domain.maxY
      ),
    [chartSize.width, chartSize.height, domain.minX, domain.maxX, domain.minY, domain.maxY]
  );
  const clipId = useId();
  const regressionSelectId = useId();

  const canToggleRegressionPeriod = !largeWaveMode && Boolean(activeSineFit && selectedRegression);
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
  const isZoomed =
    Math.abs(domain.minX - safeMinX) > 1e-6 ||
    Math.abs(domain.maxX - safeMaxX) > 1e-6 ||
    Math.abs(domain.minY - safeMinY) > 1e-6 ||
    Math.abs(domain.maxY - safeMaxY) > 1e-6;

  const polylines = series.map((s) => ({
    name: s.name,
    color: s.color,
    dash: s.dash,
    strokeWidth: s.strokeWidth,
    points: buildPolyline(s.xs, s.ys, scale),
  }));

  const peakPoints: PeakChartPoint[] = [];
  if (detail.peak_points?.length) {
    for (const peak of detail.peak_points) {
      if (peak.in_slice === false) continue;
      if (!Number.isFinite(peak.position) || !Number.isFinite(peak.frame)) continue;
      peakPoints.push({
        x: peak.position,
        y: peak.frame,
        peakI: peak.peak_i,
        peakIndex: peak.peak_index,
        frame: peak.frame,
        position: peak.position,
      });
    }
  } else {
    const peakIndices = detail.peaks_in_slice?.length ? detail.peaks_in_slice : detail.peaks ?? [];
    for (const idx of peakIndices) {
      const i = Number(idx);
      if (!Number.isFinite(i) || i < 0 || i >= positions.length) continue;
      const x = positions[i];
      const y = frames[i];
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      peakPoints.push({ x, y, peakI: i, frame: y, position: x });
    }
  }

  const selectedWindow = (() => {
    const lo = Number(selectedRegression?.fit_window_lo);
    const hi = Number(selectedRegression?.fit_window_hi);
    if (!Number.isFinite(lo) || !Number.isFinite(hi)) return null;
    const visibleLo = clamp(Math.min(lo, hi), scale.minY, scale.maxY);
    const visibleHi = clamp(Math.max(lo, hi), scale.minY, scale.maxY);
    if (visibleHi <= visibleLo) return null;
    return {
      y: toCanvas({ x: scale.minX, y: visibleLo }, scale).y,
      height: (visibleHi - visibleLo) * scale.yScale,
    };
  })();

  const zoomAtCenter = (factor: number) => {
    setViewDomain((current) => {
      const base = current ?? { minX: safeMinX, maxX: safeMaxX, minY: safeMinY, maxY: safeMaxY };
      const centerX = (base.minX + base.maxX) / 2;
      const centerY = (base.minY + base.maxY) / 2;
      const nextXDomain = zoomDomain(base.minX, base.maxX, centerX, factor);
      const nextYDomain = zoomDomain(base.minY, base.maxY, centerY, factor);
      const nextX = clampDomain(nextXDomain.min, nextXDomain.max, safeMinX, safeMaxX);
      const nextY = clampDomain(nextYDomain.min, nextYDomain.max, safeMinY, safeMaxY);
      return { key: viewKey, minX: nextX.min, maxX: nextX.max, minY: nextY.min, maxY: nextY.max };
    });
  };

  const selectPeak = (peakI: number) => {
    setSelectedPeak({ trackIndex: detail.track_index, peakI });
    setShowSine(true);
  };

  const selectedTitle = selectedRegression
    ? `${largeWaveMode ? "Wave" : "Peak"} ${selectedRegression.peak_index}`
    : `Track ${detail.track_index}`;
  const selectedMeta = selectedRegression
    ? `Frame ${formatTick(toDisplayFrame(selectedRegression.frame))} | Position ${formatTick(selectedRegression.position)} px`
    : `${frames.length} points`;

  return (
    <div className="mini-chart">
      <div className="mini-chart-header">
        <div className="mini-chart-selection" aria-live="polite">
          <span className="mini-chart-selection-title">{selectedTitle}</span>
          <span className="mini-chart-selection-meta">{selectedMeta}</span>
        </div>
        <div className="mini-zoom-strip" role="group" aria-label="Chart zoom">
          <button
            type="button"
            className="mini-icon-btn"
            aria-label="Zoom out"
            title="Zoom out"
            onClick={() => zoomAtCenter(1 / 1.25)}
          >
            -
          </button>
          <button
            type="button"
            className="mini-icon-btn"
            aria-label="Zoom in"
            title="Zoom in"
            onClick={() => zoomAtCenter(1.25)}
          >
            +
          </button>
          <button
            type="button"
            className="mini-reset-btn"
            onClick={() => setViewDomain(null)}
            disabled={!isZoomed}
          >
            Reset
          </button>
        </div>
      </div>

      {!rippleMode && regressions.length > 1 ? (
        <label className="mini-regression-picker" htmlFor={regressionSelectId}>
          <span>{largeWaveMode ? "Selected wave" : "Selected peak"}</span>
          <select
            id={regressionSelectId}
            className="mini-select"
            value={selectedRegression?.peak_i ?? ""}
            onChange={(event) => selectPeak(Number(event.target.value))}
          >
            {regressions.map((regression) => (
              <option key={regression.peak_i} value={regression.peak_i}>
                {largeWaveMode ? "Wave" : "Peak"} {regression.peak_index} | frame {formatTick(toDisplayFrame(regression.frame))} | x {formatTick(regression.position)}
              </option>
            ))}
          </select>
        </label>
      ) : null}

      <div className="mini-controls">
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
            {rippleMode ? "Line fit" : "Baseline"}
          </label>
          {!rippleMode ? (
            <>
              <label className={showSine ? "mini-layer-chip active" : "mini-layer-chip"}>
                <input
                  type="checkbox"
                  checked={showSine}
                  onChange={(e) => setShowSine(e.target.checked)}
                  disabled={!activeSineFit}
                />
                {largeWaveMode ? "Wave fit" : "Sine"}
              </label>
              <label className={showPeaks ? "mini-layer-chip active" : "mini-layer-chip"}>
                <input type="checkbox" checked={showPeaks} onChange={(e) => setShowPeaks(e.target.checked)} />
                Peaks
              </label>
            </>
          ) : null}
          <label className={showBase ? "mini-layer-chip active" : "mini-layer-chip"}>
            <input type="checkbox" checked={showBase} onChange={(e) => setShowBase(e.target.checked)} />
            Image
          </label>
        </div>
      </div>

      <div
        ref={wrapRef}
        className="mini-chart-canvas"
        style={{ height: `${chartSize.height}px` }}
        onMouseLeave={() => {
          dragRef.current = null;
          setHover(null);
        }}
      >
        <svg
          className="mini-svg"
          viewBox={`0 0 ${chartSize.width} ${chartSize.height}`}
          role="img"
          aria-label="Track detail"
          style={{ cursor: isZoomed ? "grab" : "default" }}
          onMouseMove={(e) => {
            const rect = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
            const cx = (e.clientX - rect.left) * (chartSize.width / rect.width);
            const cy = (e.clientY - rect.top) * (chartSize.height / rect.height);
            const inX = cx >= scale.plotX && cx <= scale.plotX + scale.plotWidth;
            const inY = cy >= scale.plotY && cy <= scale.plotY + scale.plotHeight;
            if (dragRef.current) {
              const dx = ((e.clientX - dragRef.current.x) * (chartSize.width / rect.width)) / scale.xScale;
              const dy = ((e.clientY - dragRef.current.y) * (chartSize.height / rect.height)) / scale.yScale;
              const nextX = clampDomain(
                dragRef.current.minX - dx,
                dragRef.current.maxX - dx,
                safeMinX,
                safeMaxX
              );
              const nextY = clampDomain(
                dragRef.current.minY - dy,
                dragRef.current.maxY - dy,
                safeMinY,
                safeMaxY
              );
              setViewDomain({
                key: viewKey,
                minX: nextX.min,
                maxX: nextX.max,
                minY: nextY.min,
                maxY: nextY.max,
              });
            }
            if (!inX || !inY) {
              setHover(null);
              return;
            }
            const x = scale.minX + (cx - scale.plotX) / scale.xScale;
            const sourceY = scale.minY + (cy - scale.plotY) / scale.yScale;
            const y = toDisplayFrame(sourceY);
            let nearestPeak: PeakChartPoint | undefined;
            let nearestDistance = 12;
            for (const peak of peakPoints) {
              const point = toCanvas(peak, scale);
              const distance = Math.hypot(point.x - cx, point.y - cy);
              if (distance <= nearestDistance) {
                nearestPeak = peak;
                nearestDistance = distance;
              }
            }
            setHover({ x, y, cx, cy, peak: nearestPeak });
          }}
          onMouseDown={(e) => {
            if (!isZoomed) return;
            dragRef.current = {
              x: e.clientX,
              y: e.clientY,
              minX: domain.minX,
              maxX: domain.maxX,
              minY: domain.minY,
              maxY: domain.maxY,
            };
          }}
          onMouseUp={() => {
            dragRef.current = null;
          }}
        >
          <defs>
            <clipPath id={clipId}>
              <rect x={scale.plotX} y={scale.plotY} width={scale.plotWidth} height={scale.plotHeight} />
            </clipPath>
          </defs>

          <rect
            x={scale.plotX}
            y={scale.plotY}
            width={scale.plotWidth}
            height={scale.plotHeight}
            className="mini-plot-background"
          />

          {selectedWindow && showSine ? (
            <rect
              x={scale.plotX}
              y={selectedWindow.y}
              width={scale.plotWidth}
              height={selectedWindow.height}
              className="mini-fit-window"
              clipPath={`url(#${clipId})`}
            />
          ) : null}

          {activeBaseImg ? (
            <image
              href={baseImageUrl ?? ""}
              x={scale.plotX - scale.minX * scale.xScale}
              y={scale.plotY - scale.minY * scale.yScale}
              width={activeBaseImg.width * scale.xScale}
              height={activeBaseImg.height * scale.yScale}
              preserveAspectRatio="none"
              opacity={0.7}
              clipPath={`url(#${clipId})`}
            />
          ) : null}

          {activeOverlayImg ? (
            <image
              href={debugImageUrl ?? ""}
              x={scale.plotX - scale.minX * scale.xScale}
              y={scale.plotY - scale.minY * scale.yScale}
              width={activeOverlayImg.width * scale.xScale}
              height={activeOverlayImg.height * scale.yScale}
              preserveAspectRatio="none"
              opacity={debugOpacity}
              clipPath={`url(#${clipId})`}
            />
          ) : null}

          {showAxes ? (
            <>
              <rect
                x={scale.plotX}
                y={scale.plotY}
                width={scale.plotWidth}
                height={scale.plotHeight}
                fill="none"
                stroke="rgba(148,163,184,0.42)"
                strokeWidth={1}
              />
              {(() => {
                const maxXTicks = clamp(Math.floor(scale.plotWidth / 72), 2, 5);
                const maxYTicks = clamp(Math.floor(scale.plotHeight / 64), 2, 7);
                const xt = niceTicks(scale.minX, scale.maxX, maxXTicks);
                const displayY0 = toDisplayFrame(scale.minY);
                const displayY1 = toDisplayFrame(scale.maxY);
                const yt = niceTicks(Math.min(displayY0, displayY1), Math.max(displayY0, displayY1), maxYTicks);
                const xStep = xt.length >= 2 ? Math.abs(xt[1] - xt[0]) : undefined;
                const yStep = yt.length >= 2 ? Math.abs(yt[1] - yt[0]) : undefined;

                return (
                  <>
                    {xt.map((v) => {
                      const cx = scale.plotX + (v - scale.minX) * scale.xScale;
                      return (
                        <g key={`xt-${v}`}>
                          <line
                            x1={cx}
                            y1={scale.plotY}
                            x2={cx}
                            y2={scale.plotY + scale.plotHeight}
                            stroke="rgba(148,163,184,0.18)"
                            strokeDasharray="3 4"
                            strokeWidth={1}
                          />
                          <text
                            className="mini-axis-tick mini-axis-tick-x"
                            x={cx}
                            y={scale.plotY + scale.plotHeight + 17}
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
                      const cy = toCanvas({ x: scale.minX, y: toSourceFrame(v) }, scale).y;
                      return (
                        <g key={`yt-${v}`}>
                          <line
                            x1={scale.plotX}
                            y1={cy}
                            x2={scale.plotX + scale.plotWidth}
                            y2={cy}
                            stroke="rgba(148,163,184,0.18)"
                            strokeDasharray="3 4"
                            strokeWidth={1}
                          />
                          <text
                            className="mini-axis-tick mini-axis-tick-y"
                            x={scale.plotX - 8}
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
                    <text
                      x={scale.plotX + scale.plotWidth}
                      y={chartSize.height - 5}
                      textAnchor="end"
                      className="mini-axis-title"
                    >
                      Position (px)
                    </text>
                    <text
                      x={10}
                      y={scale.plotY + scale.plotHeight / 2}
                      textAnchor="middle"
                      transform={`rotate(-90 10 ${scale.plotY + scale.plotHeight / 2})`}
                      className="mini-axis-title"
                    >
                      Frame
                    </text>
                  </>
                );
              })()}
            </>
          ) : null}

          {polylines.map((seriesLine) => (
            <g key={seriesLine.name} clipPath={`url(#${clipId})`}>
              {seriesLine.name === "Regression" ? (
                <polyline
                  points={seriesLine.points}
                  fill="none"
                  stroke="rgba(4, 7, 12, 0.9)"
                  strokeWidth={seriesLine.strokeWidth + 3.2}
                  strokeDasharray={seriesLine.dash}
                  strokeLinejoin="round"
                  strokeLinecap="round"
                />
              ) : null}
              <polyline
                points={seriesLine.points}
                fill="none"
                stroke={seriesLine.color}
                strokeWidth={seriesLine.strokeWidth}
                strokeDasharray={seriesLine.dash}
                strokeLinejoin="round"
                strokeLinecap="round"
              />
            </g>
          ))}

          {!rippleMode && showPeaks
            ? peakPoints.map((p, i) => {
                const scaled = toCanvas(p, scale);
                const selected = selectedRegression != null && p.peakI === selectedRegression.peak_i;
                return (
                  <g
                    key={`peak-${p.peakI ?? i}`}
                    className="mini-peak-target"
                    clipPath={`url(#${clipId})`}
                    role="button"
                    tabIndex={0}
                    aria-label={`Select peak ${p.peakIndex ?? i + 1} at frame ${formatTick(toDisplayFrame(p.frame))}`}
                    aria-pressed={selected}
                    onMouseDown={(event) => event.stopPropagation()}
                    onClick={(event) => {
                      event.stopPropagation();
                      selectPeak(p.peakI);
                    }}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" || event.key === " ") {
                        event.preventDefault();
                        selectPeak(p.peakI);
                      }
                    }}
                  >
                    <circle cx={scaled.x} cy={scaled.y} r={11} className="mini-peak-hit" />
                    <circle
                      cx={scaled.x}
                      cy={scaled.y}
                      r={selected ? 5.5 : 4}
                      className={selected ? "mini-peak selected" : "mini-peak"}
                    />
                    <title>{`Peak ${p.peakIndex ?? i + 1}: frame ${formatTick(toDisplayFrame(p.frame))}, position ${formatTick(p.position)} px`}</title>
                  </g>
                );
              })
            : null}
        </svg>

        {hover ? (
          <div
            className="mini-tooltip"
            style={{
              left: Math.min(hover.cx + 10, chartSize.width - 126),
              top: Math.min(hover.cy + 10, chartSize.height - 54),
            }}
          >
            {hover.peak ? (
              <>
                <strong>{`Peak ${hover.peak.peakIndex ?? ""}`}</strong>
                <span>{`Frame ${formatTick(toDisplayFrame(hover.peak.frame))} | ${formatTick(hover.peak.position)} px`}</span>
              </>
            ) : (
              <span>{`${hover.x.toFixed(1)} px | frame ${hover.y.toFixed(1)}`}</span>
            )}
          </div>
        ) : null}
      </div>

      <div className="mini-chart-footer">
        <div className="mini-legend">
          {showRaw ? (
            <span className="legend-item">
              <span className="legend-swatch swatch-raw" style={{ backgroundColor: overlayColor }} />
              Raw
            </span>
          ) : null}

          {showFit ? (
            <span className="legend-item">
              <span className="legend-swatch swatch-fit" />
              {rippleMode ? "Line fit" : "Baseline"}
            </span>
          ) : null}

          {!rippleMode && showSine && displaySineFit ? (
            <span className="legend-item">
              <span className="legend-swatch swatch-sine" />
              {largeWaveMode ? "Wave fit" : "Sine"}
            </span>
          ) : null}

          {!rippleMode && showPeaks ? (
            <span className="legend-item">
              <span className="legend-swatch swatch-peak" />
              Peaks
            </span>
          ) : null}
        </div>

        {!rippleMode && !largeWaveMode ? (
          <button
            type="button"
            className="mini-window-btn"
            aria-pressed={showRegressionWindowOnly}
            disabled={!canToggleRegressionPeriod}
            onClick={() => setShowRegressionWindowOnly((value) => !value)}
            title={periodToggleTitle}
          >
            {showRegressionWindowOnly ? "Full period" : "Fit window"}
          </button>
        ) : null}
      </div>
    </div>
  );
}
