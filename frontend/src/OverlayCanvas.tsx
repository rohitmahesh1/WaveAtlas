// src/OverlayCanvas.tsx
import { useEffect, useRef, useState } from "react";
import type { MouseEvent, PointerEvent } from "react";

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
};

export function OverlayCanvas(props: {
  imageUrl: string | null;
  debugImageUrl?: string | null;
  debugOpacity?: number;
  tracks: OverlayTrackEvent[];
  overlayColor?: string;
  hideBaseImage?: boolean;
  selectedTrackId?: string | number | null;
  onClickTrack?: (track: OverlayTrackEvent | null) => void;
  onHoverTrack?: (track: OverlayTrackEvent | null) => void;
  filterFn?: (t: OverlayTrackEvent) => boolean;
  colorOverrideFn?: (t: OverlayTrackEvent) => string | undefined;
  hitRadiusPx?: number;
}) {
  const {
    imageUrl,
    debugImageUrl = null,
    debugOpacity = 0.5,
    tracks,
    overlayColor = "rgba(0,140,90,0.85)",
    hideBaseImage = false,
    selectedTrackId = null,
    onClickTrack,
    onHoverTrack,
    filterFn,
    colorOverrideFn,
    hitRadiusPx = 8,
  } = props;

  const imgRef = useRef<HTMLImageElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const hitCacheRef = useRef<HitEntry[] | null>(null);
  const hoverRef = useRef<string | number | null>(null);
  const hoverRaf = useRef<number | null>(null);
  const [hoverPoint, setHoverPoint] = useState<HoverPoint | null>(null);

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

      // draw tracks
      ctx.lineWidth = 2;
      ctx.strokeStyle = withAlpha(overlayColor, 0.85);
      ctx.lineJoin = "round";
      ctx.lineCap = "round";

      const selected: OverlayTrackEvent[] = [];
      const others: OverlayTrackEvent[] = [];
      for (const t of tracks) {
        const trackId = t.id ?? t.track_index;
        const isSelected = selectedTrackId != null && String(trackId) === String(selectedTrackId);
        (isSelected ? selected : others).push(t);
      }

      const drawList = [...others, ...selected];
      for (const t of drawList) {
        if (filterFn && !filterFn(t)) continue;
        const pts = t.poly || [];
        if (pts.length < 2) continue;
        const trackId = t.id ?? t.track_index;
        const isSelected = selectedTrackId != null && String(trackId) === String(selectedTrackId);

        const customColor = colorOverrideFn?.(t);
        if (isSelected) {
          ctx.lineWidth = 3;
          ctx.strokeStyle = customColor || "rgba(255,90,20,0.95)";
        } else {
          ctx.lineWidth = 1.75;
          ctx.strokeStyle = customColor || withAlpha(overlayColor, 0.85);
        }

        ctx.beginPath();
        ctx.moveTo(pts[0].x, pts[0].y);
        for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
        ctx.stroke();
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
  }, [imageUrl, tracks, filterFn, colorOverrideFn, selectedTrackId, overlayColor]);

  // Rebuild hit cache when tracks or transforms change
  useEffect(() => {
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
  }, [tracks, filterFn, imageUrl]);

  const findNearestTrack = (cx: number, cy: number): OverlayTrackEvent | null => {
    const entries = hitCacheRef.current;
    if (!entries || entries.length === 0) return null;
    let best: OverlayTrackEvent | null = null;
    let bestDist = Infinity;

    for (const e of entries) {
      const { minX, minY, maxX, maxY } = e.bbox;
      if (cx < minX - hitRadiusPx || cx > maxX + hitRadiusPx) continue;
      if (cy < minY - hitRadiusPx || cy > maxY + hitRadiusPx) continue;
      for (const s of e.samples) {
        const dx = s.x - cx;
        const dy = s.y - cy;
        const d = Math.sqrt(dx * dx + dy * dy);
        if (d < bestDist) {
          bestDist = d;
          best = e.track;
        }
      }
    }

    return bestDist <= hitRadiusPx ? best : null;
  };

  const handlePointerMove = (ev: PointerEvent<HTMLCanvasElement>) => {
    if (hoverRaf.current) cancelAnimationFrame(hoverRaf.current);
    hoverRaf.current = requestAnimationFrame(() => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      const localX = ev.clientX - rect.left;
      const localY = ev.clientY - rect.top;
      const cx = localX * scaleX;
      const cy = localY * scaleY;
      const hit = findNearestTrack(cx, cy);
      if (onHoverTrack) {
        const hitId = hit ? (hit.id ?? hit.track_index) : null;
        if (String(hitId ?? "") !== String(hoverRef.current ?? "")) {
          hoverRef.current = hitId;
          onHoverTrack(hit);
        }
      }

      const w = canvas.width || 1;
      const h = canvas.height || 1;
      const nextPoint = {
        x: Math.max(0, Math.min(w, cx)),
        y: Math.max(0, Math.min(h, cy)),
        px: Math.max(0, Math.min(rect.width, localX)),
        py: Math.max(0, Math.min(rect.height, localY)),
      };
      setHoverPoint(nextPoint);
    });
  };

  const handlePointerLeave = () => {
    if (onHoverTrack) {
      hoverRef.current = null;
      onHoverTrack(null);
    }
    setHoverPoint(null);
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

  const imgTransform = "none";

  return (
    <div
      style={{
        position: "relative",
        width: "100%",
        background: hideBaseImage ? "#000000" : "transparent",
      }}
    >
      {imageUrl ? (
        <>
          <img
            ref={imgRef}
            src={imageUrl}
            alt="base heatmap"
            style={{
              width: "100%",
              height: "auto",
              display: "block",
              opacity: hideBaseImage ? 0 : 1,
              transform: imgTransform,
              transformOrigin: "center",
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
                objectFit: "contain",
                opacity: debugOpacity,
                transform: imgTransform,
                transformOrigin: "center",
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
              cursor: "crosshair",
            }}
            onPointerMove={handlePointerMove}
            onPointerLeave={handlePointerLeave}
            onClick={handleClick}
          />
          {hoverPoint ? (
            <div
              className="coord-tooltip"
              style={{
                left: hoverPoint.px + 12,
                top: hoverPoint.py + 12,
              }}
            >
              x {Math.round(hoverPoint.x)}, y {Math.round(hoverPoint.y)}
            </div>
          ) : null}
        </>
      ) : (
        <div className="canvas-empty">Waiting for base heatmap…</div>
      )}
    </div>
  );
}
