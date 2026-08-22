// src/OverlayCanvas.tsx
import { useEffect, useLayoutEffect, useRef, useState } from "react";
import type { MouseEvent, PointerEvent } from "react";

export const UNASSIGNED_FAMILY_KEY = "__unassigned__";

export type OverlayTrackEvent = {
  id?: string | number;
  sample?: string;
  track_index: number;
  poly: { x: number; y: number }[];
  peaks?: { x: number; y: number; amp?: number }[];
  metrics?: {
    mean_amplitude?: number | null;
    dominant_frequency?: number | null;
    period?: number | null;
    num_peaks?: number | null;
    analysis_mode?: string;
    family_id?: string | null;
    direction?: string | null;
    slope_px_per_frame?: number | null;
    velocity_px_per_s?: number | null;
    speed_px_per_s?: number | null;
    angle_deg?: number | null;
    angle_from_time_axis_deg?: number | null;
    line_rmse_px?: number | null;
    line_fit_rmse_px?: number | null;
    line_r2?: number | null;
    duration_s?: number | null;
    neighbor_interval_count?: number | null;
    ripple_period_frames?: number | null;
    ripple_period_s?: number | null;
    frequency_hz?: number | null;
    ripple_frequency_hz?: number | null;
    frequency_method?: string | null;
    large_wave_recurrence_frequency_hz?: number | null;
  };
};

function transformPoint(x: number, y: number) {
  return { x, y };
}

function withAlpha(color: string, alpha: number) {
  const hex = color.trim();
  if (/^#([0-9a-fA-F]{6})$/.test(hex)) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }
  return color;
}

function formatZValue(value: number) {
  if (value === 0) return "0";
  const abs = Math.abs(value);
  if (abs >= 10000 || abs < 0.001) return value.toExponential(3);
  const fixed = abs >= 100 ? value.toFixed(1) : abs >= 10 ? value.toFixed(2) : value.toFixed(3);
  return fixed.replace(/\.?0+$/, "");
}

function trackFamilyKey(track: OverlayTrackEvent) {
  return track.metrics?.family_id || UNASSIGNED_FAMILY_KEY;
}

type HitEntry = {
  id: string | number;
  samples: { x: number; y: number }[];
  bbox: { minX: number; minY: number; maxX: number; maxY: number };
  track: OverlayTrackEvent;
};

type HoverPoint = {
  x: number;
  y: number;
  px: number;
  py: number;
  xLabel: string;
  yLabel: string;
  valueCol: number;
  valueRow: number;
  z: number | null;
  zLabel: string;
};

type CoordInfo = {
  sourceKind?: string | null;
  sourceRows?: number | null;
  sourceCols?: number | null;
  outputWidth?: number | null;
  outputHeight?: number | null;
  coordOrigin?: string | null;
  pixelMapping?: string | null;
  xLabel?: string;
  yLabel?: string;
  zLabel?: string;
  zMin?: number | null;
  zMax?: number | null;
  zVmin?: number | null;
  zVmax?: number | null;
};

export type HeatmapValues = {
  values: Float32Array;
  width: number;
  height: number;
  origin?: string | null;
  label?: string | null;
  min?: number | null;
  max?: number | null;
};

export type OverlayProjection = {
  sourceWidth: number;
  sourceHeight: number;
  displayWidth: number;
  displayHeight: number;
  xScale: number;
  yScale: number;
};

export function OverlayCanvas(props: {
  imageUrl: string | null;
  debugImageUrl?: string | null;
  debugOpacity?: number;
  coordInfo?: CoordInfo | null;
  heatmapValues?: HeatmapValues | null;
  tracks: OverlayTrackEvent[];
  overlayColor?: string;
  hideBaseImage?: boolean;
  hideTracks?: boolean;
  selectedTrackId?: string | number | null;
  selectedFamilyId?: string | null;
  selectionScope?: "family" | "track" | null;
  onClickTrack?: (track: OverlayTrackEvent | null) => void;
  onHoverTrack?: (track: OverlayTrackEvent | null) => void;
  filterFn?: (t: OverlayTrackEvent) => boolean;
  colorOverrideFn?: (t: OverlayTrackEvent) => string | undefined;
  hitRadiusPx?: number;
  onProjectionChange?: (projection: OverlayProjection | null) => void;
}) {
  const {
    imageUrl,
    debugImageUrl = null,
    debugOpacity = 0.5,
    coordInfo = null,
    heatmapValues = null,
    tracks,
    overlayColor = "rgba(0,140,90,0.85)",
    hideBaseImage = false,
    hideTracks = false,
    selectedTrackId = null,
    selectedFamilyId = null,
    selectionScope = null,
    onClickTrack,
    onHoverTrack,
    filterFn,
    colorOverrideFn,
    hitRadiusPx = 8,
    onProjectionChange,
  } = props;

  const imgRef = useRef<HTMLImageElement | null>(null);
  const viewportRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const highlightCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const tooltipRef = useRef<HTMLDivElement | null>(null);
  const hitCacheRef = useRef<HitEntry[] | null>(null);
  const hoverRef = useRef<string | number | null>(null);
  const hoverRaf = useRef<number | null>(null);
  const dragStartRef = useRef<{ x: number; y: number; panX: number; panY: number } | null>(null);
  const [hoverPoint, setHoverPoint] = useState<HoverPoint | null>(null);
  const [tooltipSize, setTooltipSize] = useState<{ width: number; height: number }>({ width: 0, height: 0 });
  const [imageSize, setImageSize] = useState<{ width: number; height: number } | null>(null);
  const [stageSize, setStageSize] = useState<{ width: number; height: number }>({ width: 1, height: 1 });
  const [zoom, setZoom] = useState<number>(1);
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [hoveredTrack, setHoveredTrack] = useState<OverlayTrackEvent | null>(null);

  useEffect(() => {
    const image = imgRef.current;
    if (!image) return;

    const update = () => {
      const width = image.naturalWidth || 0;
      const height = image.naturalHeight || 0;
      setImageSize(width > 0 && height > 0 ? { width, height } : null);
    };

    if (image.complete && image.naturalWidth > 0) {
      update();
      return;
    }

    image.addEventListener("load", update);
    return () => image.removeEventListener("load", update);
  }, [imageUrl]);

  useEffect(() => {
    const viewport = viewportRef.current;
    if (!viewport || !imageSize) return;

    const update = () => {
      const vw = Math.max(1, viewport.clientWidth);
      const vh = Math.max(1, viewport.clientHeight);
      setStageSize({
        // Fill the entire viewer area visually, even when the heatmap aspect
        // ratio differs from the available canvas size.
        width: vw,
        height: vh,
      });
    };

    update();
    const ro = new ResizeObserver(update);
    ro.observe(viewport);
    return () => ro.disconnect();
  }, [imageSize]);

  useEffect(() => {
    if (!onProjectionChange) return;
    if (!imageSize || stageSize.width <= 0 || stageSize.height <= 0) {
      onProjectionChange(null);
      return;
    }
    onProjectionChange({
      sourceWidth: imageSize.width,
      sourceHeight: imageSize.height,
      displayWidth: stageSize.width,
      displayHeight: stageSize.height,
      xScale: stageSize.width / imageSize.width,
      yScale: stageSize.height / imageSize.height,
    });
  }, [imageSize, stageSize.height, stageSize.width, onProjectionChange]);

  useLayoutEffect(() => {
    const tooltip = tooltipRef.current;
    if (!tooltip || !hoverPoint) return;

    setTooltipSize({
      width: tooltip.offsetWidth,
      height: tooltip.offsetHeight,
    });
  }, [hoverPoint]);

  useEffect(() => {
    return () => {
      if (hoverRaf.current) {
        cancelAnimationFrame(hoverRaf.current);
        hoverRaf.current = null;
      }
    };
  }, []);

  function clampPan(nextPan: { x: number; y: number }, nextZoom: number) {
    const extraX = Math.max(0, (stageSize.width * nextZoom - stageSize.width) / 2);
    const extraY = Math.max(0, (stageSize.height * nextZoom - stageSize.height) / 2);
    return {
      x: Math.max(-extraX, Math.min(extraX, nextPan.x)),
      y: Math.max(-extraY, Math.min(extraY, nextPan.y)),
    };
  }

  function updateZoom(nextZoom: number, anchor?: { clientX: number; clientY: number }) {
    const clampedZoom = Math.max(1, Math.min(8, nextZoom));
    if (!viewportRef.current || !anchor || zoom === clampedZoom) {
      setZoom(clampedZoom);
      setPan((current) => clampPan(current, clampedZoom));
      return;
    }

    const rect = viewportRef.current.getBoundingClientRect();
    const anchorX = anchor.clientX - rect.left - rect.width / 2;
    const anchorY = anchor.clientY - rect.top - rect.height / 2;
    const ratio = clampedZoom / zoom;
    setZoom(clampedZoom);
    setPan((current) =>
      clampPan(
        {
          x: anchorX - (anchorX - current.x) * ratio,
          y: anchorY - (anchorY - current.y) * ratio,
        },
        clampedZoom
      )
    );
  }

  function projectHoverPoint(cx: number, cy: number, w: number, h: number): HoverPoint {
    const maxX = Math.max(0, w - 1);
    const maxY = Math.max(0, h - 1);
    const rawX = Math.max(0, Math.min(maxX, cx));
    const rawY = Math.max(0, Math.min(maxY, cy));

    const xLabel = coordInfo?.xLabel || "x";
    const yLabel = coordInfo?.yLabel || "y";
    if (coordInfo?.sourceKind === "table" && coordInfo?.pixelMapping === "table_cell") {
      const gridW = Math.max(1, Math.round(coordInfo.outputWidth || coordInfo.sourceCols || w));
      const gridH = Math.max(1, Math.round(coordInfo.outputHeight || coordInfo.sourceRows || h));
      const col = Math.max(0, Math.min(gridW - 1, Math.floor(rawX)));
      const topRow = Math.max(0, Math.min(gridH - 1, Math.floor(rawY)));
      const row =
        String(coordInfo.coordOrigin || "").toLowerCase() === "lower" ? gridH - 1 - topRow : topRow;
      return {
        x: col,
        y: row,
        px: 0,
        py: 0,
        xLabel,
        yLabel,
        valueCol: col,
        valueRow: topRow,
        z: null,
        zLabel: coordInfo?.zLabel || heatmapValues?.label || "z",
      };
    }

    const col = Math.max(0, Math.floor(rawX));
    const row = Math.max(0, Math.floor(rawY));
    return {
      x: col,
      y: row,
      px: 0,
      py: 0,
      xLabel,
      yLabel,
      valueCol: col,
      valueRow: row,
      z: null,
      zLabel: coordInfo?.zLabel || heatmapValues?.label || "z",
    };
  }

  function lookupHeatmapValue(col: number, row: number): number | null {
    if (!heatmapValues) return null;
    const width = Math.max(0, Math.floor(heatmapValues.width));
    const height = Math.max(0, Math.floor(heatmapValues.height));
    if (width <= 0 || height <= 0 || heatmapValues.values.length < width * height) return null;

    const valueCol = Math.max(0, Math.min(width - 1, col));
    const valueRow = Math.max(0, Math.min(height - 1, row));
    const value = heatmapValues.values[valueRow * width + valueCol];
    return Number.isFinite(value) ? value : null;
  }

  // Draw whenever tracks change, transform changes, or image loads
  useEffect(() => {
    const image = imgRef.current;
    const drawingCanvas = canvasRef.current;
    if (!image || !drawingCanvas) return;

    let cancelled = false;

    function draw(currentImage: HTMLImageElement, currentCanvas: HTMLCanvasElement) {
      if (cancelled) return;

      const w = currentImage.naturalWidth || 1;
      const h = currentImage.naturalHeight || 1;

      currentCanvas.width = w;
      currentCanvas.height = h;

      const ctx = currentCanvas.getContext("2d");
      if (!ctx) return;

      ctx.clearRect(0, 0, w, h);

      ctx.save();

      if (hideTracks) {
        ctx.restore();
        return;
      }

      // draw tracks
      ctx.lineWidth = 2;
      ctx.strokeStyle = withAlpha(overlayColor, 0.85);
      ctx.lineJoin = "round";
      ctx.lineCap = "round";

      const selected: OverlayTrackEvent[] = [];
      const others: OverlayTrackEvent[] = [];
      for (const t of tracks) {
        const trackId = t.id ?? t.track_index;
        const familyId = trackFamilyKey(t);
        const isSelectedTrack =
          selectionScope !== "family" && selectedTrackId != null && String(trackId) === String(selectedTrackId);
        const isSelectedFamily =
          selectionScope === "family"
          && selectedFamilyId != null
          && familyId === selectedFamilyId;
        const isSelected = isSelectedTrack || isSelectedFamily;
        (isSelected ? selected : others).push(t);
      }

      const drawList = [...others, ...selected];
      const hasFamilySelection = selectionScope === "family" && selectedFamilyId != null;
      for (const t of drawList) {
        if (filterFn && !filterFn(t)) continue;
        const pts = t.poly || [];
        if (pts.length < 2) continue;
        const trackId = t.id ?? t.track_index;
        const familyId = trackFamilyKey(t);
        const isSelectedTrack =
          selectionScope !== "family" && selectedTrackId != null && String(trackId) === String(selectedTrackId);
        const isSelectedFamily =
          selectionScope === "family"
          && selectedFamilyId != null
          && familyId === selectedFamilyId;
        const isSelected = isSelectedTrack || isSelectedFamily;

        const customColor = colorOverrideFn?.(t);
        if (isSelected) {
          ctx.lineWidth = isSelectedFamily ? 3 : 3.5;
          ctx.strokeStyle = customColor || "rgba(255,90,20,0.95)";
          ctx.shadowColor = customColor ? withAlpha(customColor, 0.5) : "rgba(255,90,20,0.45)";
          ctx.shadowBlur = 5;
        } else {
          ctx.lineWidth = 1.75;
          ctx.strokeStyle =
            hasFamilySelection && customColor
              ? withAlpha(customColor, 0.36)
              : customColor || withAlpha(overlayColor, 0.85);
          ctx.shadowBlur = 0;
        }

        ctx.beginPath();
        ctx.moveTo(pts[0].x, pts[0].y);
        for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
        ctx.stroke();
        ctx.shadowBlur = 0;
      }

      ctx.restore();
    }

    // If image not loaded yet, wait
    if (!image.complete || image.naturalWidth === 0) {
      const onLoad = () => draw(image, drawingCanvas);
      image.addEventListener("load", onLoad);
      return () => {
        cancelled = true;
        image.removeEventListener("load", onLoad);
      };
    }

    draw(image, drawingCanvas);
    return () => {
      cancelled = true;
    };
  }, [
    imageUrl,
    tracks,
    filterFn,
    colorOverrideFn,
    selectedTrackId,
    selectedFamilyId,
    selectionScope,
    overlayColor,
    hideTracks,
  ]);

  useEffect(() => {
    const canvas = highlightCanvasRef.current;
    if (!canvas || !imageSize) return;

    canvas.width = imageSize.width;
    canvas.height = imageSize.height;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (hideTracks || !hoveredTrack) return;

    const hoveredFamilyId = trackFamilyKey(hoveredTrack);
    const hoverTracks = tracks.filter(
      (track) => trackFamilyKey(track) === hoveredFamilyId && (!filterFn || filterFn(track))
    );

    ctx.save();
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    for (const track of hoverTracks) {
      const pts = track.poly || [];
      if (pts.length < 2) continue;
      const trackId = track.id ?? track.track_index;
      const hoveredId = hoveredTrack.id ?? hoveredTrack.track_index;
      const isHoveredTrack = String(trackId) === String(hoveredId);
      const hoverColor = colorOverrideFn?.(track);
      ctx.lineWidth = isHoveredTrack ? 4 : 2.4;
      ctx.strokeStyle = hoverColor
        ? withAlpha(hoverColor, isHoveredTrack ? 0.95 : 0.58)
        : isHoveredTrack
          ? "rgba(255,215,0,0.9)"
          : "rgba(255,215,0,0.42)";
      ctx.shadowColor = ctx.strokeStyle;
      ctx.shadowBlur = isHoveredTrack ? 7 : 3;
      ctx.beginPath();
      ctx.moveTo(pts[0].x, pts[0].y);
      for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
      ctx.stroke();
    }
    ctx.restore();
  }, [hoveredTrack, imageSize, hideTracks, tracks, filterFn, colorOverrideFn]);

  // Rebuild hit cache when tracks or transforms change
  useEffect(() => {
    if (hideTracks) {
      hitCacheRef.current = [];
      return;
    }

    const img = imgRef.current;
    if (!img) return;

    const buildCache = () => {
      const w = img.naturalWidth;
      const h = img.naturalHeight;
      if (!w || !h) return;

      const entries: HitEntry[] = [];
      const list = filterFn ? tracks.filter(filterFn) : tracks;

      for (const t of list) {
        const pts = t.poly || [];
        if (pts.length < 2) continue;
        const step = Math.max(1, Math.floor(pts.length / 200));
        const samples: { x: number; y: number }[] = [];
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        for (let i = 0; i < pts.length; i += step) {
          const p = transformPoint(pts[i].x, pts[i].y);
          samples.push(p);
          if (p.x < minX) minX = p.x;
          if (p.y < minY) minY = p.y;
          if (p.x > maxX) maxX = p.x;
          if (p.y > maxY) maxY = p.y;
        }
        if (!Number.isFinite(minX + minY + maxX + maxY)) continue;
        entries.push({
          id: t.id ?? t.track_index,
          samples,
          bbox: { minX, minY, maxX, maxY },
          track: t,
        });
      }

      hitCacheRef.current = entries;
    };

    if (!img.complete || img.naturalWidth === 0) {
      const onLoad = () => buildCache();
      img.addEventListener("load", onLoad);
      return () => img.removeEventListener("load", onLoad);
    }

    buildCache();
  }, [tracks, filterFn, imageUrl, hideTracks]);

  const findNearestTrack = (cx: number, cy: number): OverlayTrackEvent | null => {
    const entries = hitCacheRef.current;
    if (!entries || entries.length === 0) return null;
    let best: OverlayTrackEvent | null = null;
    let bestDistSq = Infinity;
    const hitRadiusSq = hitRadiusPx * hitRadiusPx;

    for (const e of entries) {
      const { minX, minY, maxX, maxY } = e.bbox;
      if (cx < minX - hitRadiusPx || cx > maxX + hitRadiusPx) continue;
      if (cy < minY - hitRadiusPx || cy > maxY + hitRadiusPx) continue;
      for (const s of e.samples) {
        const dx = s.x - cx;
        const dy = s.y - cy;
        const dSq = dx * dx + dy * dy;
        if (dSq < bestDistSq) {
          bestDistSq = dSq;
          best = e.track;
        }
      }
    }

    return bestDistSq <= hitRadiusSq ? best : null;
  };

  const handlePointerMove = (ev: PointerEvent<HTMLCanvasElement>) => {
    if (hoverRaf.current) cancelAnimationFrame(hoverRaf.current);
    hoverRaf.current = requestAnimationFrame(() => {
      hoverRaf.current = null;
      if (dragStartRef.current) {
        const dx = ev.clientX - dragStartRef.current.x;
        const dy = ev.clientY - dragStartRef.current.y;
        setPan(
          clampPan(
            {
              x: dragStartRef.current.panX + dx,
              y: dragStartRef.current.panY + dy,
            },
            zoom
          )
        );
      }

      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      const localX = ev.clientX - rect.left;
      const localY = ev.clientY - rect.top;
      const cx = localX * scaleX;
      const cy = localY * scaleY;

      const w = canvas.width || 1;
      const h = canvas.height || 1;
      const projected = projectHoverPoint(cx, cy, w, h);
      const zValue = lookupHeatmapValue(projected.valueCol, projected.valueRow);
      const nextPoint = {
        ...projected,
        px: Math.max(0, Math.min(rect.width, localX)),
        py: Math.max(0, Math.min(rect.height, localY)),
        z: zValue,
        zLabel: projected.zLabel || heatmapValues?.label || coordInfo?.zLabel || "z",
      };
      setHoverPoint(nextPoint);

      const hit = hideTracks ? null : findNearestTrack(cx, cy);
      const hitId = hit ? (hit.id ?? hit.track_index) : null;
      if (String(hitId ?? "") !== String(hoverRef.current ?? "")) {
        hoverRef.current = hitId;
        setHoveredTrack(hit);
        onHoverTrack?.(hit);
      }
    });
  };

  const handlePointerLeave = () => {
    if (hoverRaf.current) {
      cancelAnimationFrame(hoverRaf.current);
      hoverRaf.current = null;
    }
    hoverRef.current = null;
    setHoveredTrack(null);
    onHoverTrack?.(null);
    dragStartRef.current = null;
    setHoverPoint(null);
  };

  const handlePointerDown = (ev: PointerEvent<HTMLCanvasElement>) => {
    if (zoom <= 1) return;
    dragStartRef.current = {
      x: ev.clientX,
      y: ev.clientY,
      panX: pan.x,
      panY: pan.y,
    };
    ev.currentTarget.setPointerCapture(ev.pointerId);
  };

  const handlePointerUp = (ev: PointerEvent<HTMLCanvasElement>) => {
    dragStartRef.current = null;
    if (ev.currentTarget.hasPointerCapture(ev.pointerId)) {
      ev.currentTarget.releasePointerCapture(ev.pointerId);
    }
  };

  const handleClick = (ev: MouseEvent<HTMLCanvasElement>) => {
    if (!onClickTrack) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const cx = (ev.clientX - rect.left) * scaleX;
    const cy = (ev.clientY - rect.top) * scaleY;
    const hit = findNearestTrack(cx, cy);
    onClickTrack(hit);
  };

  const stageTransform = `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`;
  const tooltipOffset = 12;
  const tooltipMargin = 8;
  const tooltipLeft = hoverPoint
    ? Math.max(
        tooltipMargin,
        Math.min(
          stageSize.width - tooltipSize.width - tooltipMargin,
          hoverPoint.px + tooltipOffset
        )
      )
    : 0;
  const tooltipTop = hoverPoint
    ? Math.max(
        tooltipMargin,
        Math.min(
          stageSize.height - tooltipSize.height - tooltipMargin,
          hoverPoint.py + tooltipOffset
        )
      )
    : 0;

  return (
    <div
      style={{
        position: "relative",
        width: "100%",
        background: hideBaseImage ? "#000000" : "transparent",
      }}
    >
      {imageUrl ? (
        <div
          ref={viewportRef}
          className="zoom-viewport"
          style={{
            position: "relative",
            width: "100%",
            height: "clamp(280px, 72vh, 720px)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <div className="zoom-controls">
            <button type="button" className="zoom-btn" onClick={() => updateZoom(zoom / 1.25)}>
              -
            </button>
            <span className="zoom-readout">{Math.round(zoom * 100)}%</span>
            <button type="button" className="zoom-btn" onClick={() => updateZoom(zoom * 1.25)}>
              +
            </button>
            <button
              type="button"
              className="zoom-btn"
              onClick={() => {
                setZoom(1);
                setPan({ x: 0, y: 0 });
              }}
              disabled={zoom === 1 && pan.x === 0 && pan.y === 0}
            >
              Reset
            </button>
          </div>
          <div
            style={{
              position: "relative",
              width: `${stageSize.width}px`,
              height: `${stageSize.height}px`,
              maxWidth: "100%",
              maxHeight: "100%",
              overflow: "hidden",
            }}
          >
            <div
              style={{
                position: "absolute",
                inset: 0,
                transform: stageTransform,
                transformOrigin: "center",
              }}
            >
              <img
                ref={imgRef}
                src={imageUrl}
                alt="base heatmap"
                style={{
                  width: "100%",
                  height: "100%",
                  display: "block",
                  opacity: hideBaseImage ? 0 : 1,
                }}
              />
              {debugImageUrl ? (
                <img
                  src={debugImageUrl}
                  alt="debug overlay"
                  style={{
                    position: "absolute",
                    inset: 0,
                    width: "100%",
                    height: "100%",
                    opacity: debugOpacity,
                    pointerEvents: "none",
                  }}
                />
              ) : null}
              <canvas
                ref={canvasRef}
                style={{
                  position: "absolute",
                  inset: 0,
                  width: "100%",
                  height: "100%",
                  pointerEvents: "auto",
                  cursor: zoom > 1 ? "grab" : hideTracks ? "default" : "crosshair",
                }}
                onPointerMove={handlePointerMove}
                onPointerLeave={handlePointerLeave}
                onPointerDown={handlePointerDown}
                onPointerUp={handlePointerUp}
                onClick={handleClick}
              />
              <canvas
                ref={highlightCanvasRef}
                style={{
                  position: "absolute",
                  inset: 0,
                  width: "100%",
                  height: "100%",
                  pointerEvents: "none",
                }}
              />
            </div>
            {hoverPoint ? (
              <div
                ref={tooltipRef}
                className="coord-tooltip"
                style={{
                  left: tooltipLeft,
                  top: tooltipTop,
                }}
              >
                <span>
                  ({hoverPoint.x}, {hoverPoint.y}
                  {hoverPoint.z != null ? `, ${formatZValue(hoverPoint.z)}` : ""})
                </span>
              </div>
            ) : null}
          </div>
        </div>
      ) : (
        <div className="canvas-empty">Waiting for base heatmap…</div>
      )}
    </div>
  );
}
