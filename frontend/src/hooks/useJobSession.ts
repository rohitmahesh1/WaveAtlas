import { useEffect, useRef, useState } from "react";
import {
  createJob,
  startJob,
  listArtifacts,
  uploadViaApi,
  createUploadSession,
  uploadToResumableUrl,
  uploadComplete,
  getJob,
  wsUrl,
  cancelJob,
  API_BASE,
  isApiError,
} from "../api";
import type { OverlayTrackEvent } from "../OverlayCanvas";
import type { LogEntry } from "../types";
import { formatEta } from "../utils/format";

type WsMsg =
  | { type: "snapshot"; payload?: unknown; seq?: number; job_id?: string }
  | { type: "overlay_track"; payload?: unknown; seq?: number; job_id?: string }
  | { type: "status"; payload?: unknown; seq?: number; job_id?: string }
  | { type: "progress"; payload?: unknown; seq?: number; job_id?: string }
  | { type: "user_log"; payload?: unknown; seq?: number; job_id?: string }
  | { type: "ping"; payload?: unknown; seq?: number; job_id?: string }
  | { type: string; payload?: unknown; seq?: number; job_id?: string };

type UnknownRecord = Record<string, unknown>;
type OverlayPeak = NonNullable<OverlayTrackEvent["peaks"]>[number];

function asRecord(value: unknown): UnknownRecord | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as UnknownRecord)
    : null;
}

function finiteNumber(value: unknown): number | null {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

const SESSION_KEY = "waveatlas:lastSession";

function loadSession(): { jobId: string; lastSeq: number } | null {
  try {
    const raw = localStorage.getItem(SESSION_KEY);
    if (!raw) return null;
    const s = JSON.parse(raw);
    if (!s?.jobId) return null;
    return { jobId: String(s.jobId), lastSeq: Number(s.lastSeq || 0) };
  } catch {
    return null;
  }
}

function saveSession(jobId: string, lastSeq: number) {
  localStorage.setItem(SESSION_KEY, JSON.stringify({ jobId, lastSeq }));
}

function isImageFile(file: File) {
  if (file.type.startsWith("image/")) return true;
  return /\.(png|jpe?g|tiff?|bmp|webp)$/i.test(file.name);
}

function normalizeOverlayTrack(payload: unknown): OverlayTrackEvent | null {
  const data = asRecord(payload);
  if (!data) return null;

  const idx = Number(data.track_index);
  if (!Number.isFinite(idx)) return null;

  const peaks: OverlayPeak[] = Array.isArray(data.peaks)
    ? data.peaks.flatMap((entry) => {
        const peak = asRecord(entry);
        if (!peak) return [];
        const x = finiteNumber(peak.x);
        const y = finiteNumber(peak.y);
        if (x == null || y == null) return [];
        const amp = finiteNumber(peak.amp);
        return [{ x, y, ...(amp != null ? { amp } : {}) }];
      })
    : [];
  const ampVals = peaks.map((p) => Number(p.amp)).filter((v) => Number.isFinite(v));
  const meanAmp = ampVals.length ? ampVals.reduce((a, b) => a + b, 0) / ampVals.length : null;
  const poly = Array.isArray(data.poly)
    ? data.poly.flatMap((entry) => {
        const point = asRecord(entry);
        if (!point) return [];
        const x = finiteNumber(point.x);
        const y = finiteNumber(point.y);
        return x != null && y != null ? [{ x, y }] : [];
      })
    : [];

  return {
    id: idx,
    sample: typeof data.sample === "string" ? data.sample : undefined,
    track_index: idx,
    poly,
    peaks,
    metrics: {
      mean_amplitude: meanAmp,
      dominant_frequency: finiteNumber(data.freq_hz),
      period: finiteNumber(data.period),
      num_peaks: peaks.length,
    },
  };
}

export function useJobSession(options?: { resumeOnMount?: boolean }) {
  const [initialSession] = useState(() => (options?.resumeOnMount ? loadSession() : null));
  const [jobId, setJobId] = useState<string | null>(initialSession?.jobId ?? null);
  const [status, setStatus] = useState<string>(initialSession ? "resuming…" : "idle");
  const [baseImageUrl, setBaseImageUrl] = useState<string | null>(null);
  const [originalImageUrl, setOriginalImageUrl] = useState<string | null>(null);
  const [tracks, setTracks] = useState<OverlayTrackEvent[]>([]);
  const [activity, setActivity] = useState<LogEntry[]>([]);
  const [currentStage, setCurrentStage] = useState<string>(initialSession ? "resuming" : "idle");
  const [stageDetail, setStageDetail] = useState<string | null>(null);
  const [etaText, setEtaText] = useState<string | null>(null);
  const [debugOverlays, setDebugOverlays] = useState<{ label: string; url: string }[]>([]);

  const wsRef = useRef<WebSocket | null>(null);
  const lastSeqRef = useRef<number>(initialSession?.lastSeq ?? 0);
  const wsTokenRef = useRef<number>(0);
  const reconnectRef = useRef<{ stop: boolean; tries: number }>({ stop: false, tries: 0 });
  const lastStageRef = useRef<string>("");
  const pollTokenRef = useRef<number>(0);
  const resumeStartedRef = useRef<boolean>(false);
  const jobIdRef = useRef<string | null>(initialSession?.jobId ?? null);
  const pollForBaseHeatmapRef = useRef<(id: string) => void>(() => undefined);
  const connectWsWithRetryRef = useRef<(id: string, afterSeq: number) => void>(() => undefined);
  const resumeSavedJobRef = useRef<(id: string) => void>(() => undefined);

  const addActivity = (message: string, level: "info" | "warn" | "error" = "info", stage?: string) => {
    const ts = new Date().toLocaleTimeString();
    const id = `${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    setActivity((x) => [{ id, ts, message, level, stage }, ...x].slice(0, 200));
  };

  function upsertTrack(t: OverlayTrackEvent) {
    setTracks((prev) => {
      const m = new Map<number, OverlayTrackEvent>();
      for (const p of prev) m.set(p.track_index, p);
      m.set(t.track_index, t);
      return Array.from(m.values()).sort((a, b) => a.track_index - b.track_index);
    });
  }

  function upsertDebugOverlay(entry: { label: string; url: string }) {
    setDebugOverlays((prev) => {
      const m = new Map<string, { label: string; url: string }>();
      for (const p of prev) m.set(p.label, p);
      m.set(entry.label, entry);
      return Array.from(m.values()).sort((a, b) => a.label.localeCompare(b.label));
    });
  }

  function normalizeOverlayLabel(label: string) {
    return label.includes(":") ? label.split(":").slice(1).join(":") : label;
  }

  function normalizeOverlayUrl(url: string) {
    if (!url) return url;
    if (url.startsWith("http")) return url;
    if (url.startsWith("/") && API_BASE) return `${API_BASE}${url}`;
    return url;
  }

  function clearMissingJobSession(id: string) {
    if (jobIdRef.current !== id) return;

    closeWs("missing-job");
    pollTokenRef.current += 1;
    localStorage.removeItem(SESSION_KEY);
    jobIdRef.current = null;
    setJobId(null);
    setStatus("idle");
    setTracks([]);
    setBaseImageUrl(null);
    setOriginalImageUrl(null);
    lastSeqRef.current = 0;
    setCurrentStage("idle");
    setStageDetail(null);
    setEtaText(null);
    setDebugOverlays([]);
    setActivity([]);
    addActivity("Saved run is no longer available; cleared session", "warn", "resume");
  }

  async function refreshDebugOverlays(id: string) {
    try {
      const overlayArts = await listArtifacts(id, { kind: "overlay", limit: 2000 });
      const overlays = overlayArts
        .filter((a) => {
          if (!a.label || !a.download_url) return false;
          const label = String(a.label);
          if (label.endsWith(":stats") || label === "stats") return false;
          if (a.content_type && !a.content_type.startsWith("image/")) return false;
          return true;
        })
        .map((a) => {
          const label = normalizeOverlayLabel(String(a.label));
          return { label, url: normalizeOverlayUrl(a.download_url) };
        });
      if (overlays.length) {
        const uniq = new Map<string, string>();
        for (const o of overlays) uniq.set(o.label, o.url);
        const list = Array.from(uniq.entries())
          .map(([label, url]) => ({ label, url }))
          .sort((a, b) => a.label.localeCompare(b.label));
        setDebugOverlays(list);
      }
    } catch (error) {
      if (isApiError(error, 404)) {
        clearMissingJobSession(id);
      }
    }
  }

  async function refreshOriginalImage(id: string, showAsBase = false) {
    try {
      const imageUploads = await listArtifacts(id, { kind: "upload_image", limit: 1 });
      const original = imageUploads.find((a) => a.kind === "upload_image" || a.label === "upload");
      if (original?.download_url) {
        const url = normalizeOverlayUrl(original.download_url);
        setOriginalImageUrl(url);
        if (showAsBase) setBaseImageUrl(url);
      }
    } catch (error) {
      if (isApiError(error, 404)) {
        clearMissingJobSession(id);
      }
    }
  }

  async function pollForBaseHeatmap(id: string) {
    const myToken = ++pollTokenRef.current;
    for (let i = 0; i < 60; i++) {
      if (pollTokenRef.current !== myToken) return;
      try {
        const [overlayArts, baseArts, imageUploads] = await Promise.all([
          listArtifacts(id, { kind: "overlay", limit: 2000 }),
          listArtifacts(id, { kind: "base_heatmap", limit: 1 }),
          listArtifacts(id, { kind: "upload_image", limit: 1 }),
        ]);
        if (pollTokenRef.current !== myToken) return;
        const overlays = overlayArts
          .filter((a) => {
            if (!a.label || !a.download_url) return false;
            const label = String(a.label);
            if (label.endsWith(":stats") || label === "stats") return false;
            if (a.content_type && !a.content_type.startsWith("image/")) return false;
            return true;
          })
          .map((a) => {
            const label = normalizeOverlayLabel(String(a.label));
            return { label, url: normalizeOverlayUrl(a.download_url) };
          });
        if (overlays.length) {
          const uniq = new Map<string, string>();
          for (const o of overlays) uniq.set(o.label, o.url);
          const list = Array.from(uniq.entries())
            .map(([label, url]) => ({ label, url }))
            .sort((a, b) => a.label.localeCompare(b.label));
          setDebugOverlays(list);
        }
        const base = baseArts.find((a) => a.kind === "base_heatmap" || a.label === "base_heatmap");
        if (base?.download_url) {
          setBaseImageUrl(base.download_url);
        }
        const original = imageUploads.find((a) => a.kind === "upload_image" || a.label === "upload");
        if (original?.download_url) {
          const url = normalizeOverlayUrl(original.download_url);
          setOriginalImageUrl(url);
          if (!base?.download_url) setBaseImageUrl(url);
        }
      } catch (error) {
        if (isApiError(error, 404)) {
          clearMissingJobSession(id);
          return;
        }
        // Keep polling transient failures quietly; user log stays clean.
      }
      await new Promise((r) => setTimeout(r, 1000));
    }
    if (pollTokenRef.current === myToken) {
      addActivity("Heatmap not found after polling", "warn");
    }
  }

  function closeWs(reason = "close") {
    wsTokenRef.current += 1;
    reconnectRef.current.stop = true;
    reconnectRef.current.tries = 0;

    try {
      wsRef.current?.close(1000, reason);
    } catch {
      // ignore
    }
    wsRef.current = null;
  }

  function connectWsWithRetry(id: string, afterSeq: number) {
    reconnectRef.current.stop = false;
    reconnectRef.current.tries = 0;

    const myToken = ++wsTokenRef.current;

    const doConnect = (seq: number) => {
      if (myToken !== wsTokenRef.current) return;

      try {
        wsRef.current?.close(1000, "reconnect");
      } catch {
        // ignore
      }

      lastSeqRef.current = seq;
      saveSession(id, lastSeqRef.current);

      const url = wsUrl(`/api/ws/jobs/${id}`, { after_seq: seq, poll_interval: 0.2 });

      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        if (myToken !== wsTokenRef.current) return;
        reconnectRef.current.tries = 0;
      };

      ws.onmessage = (ev) => {
        if (myToken !== wsTokenRef.current) return;

        const msg = JSON.parse(ev.data) as WsMsg;

        if (typeof msg.seq === "number") {
          lastSeqRef.current = Math.max(lastSeqRef.current, msg.seq);
          saveSession(id, lastSeqRef.current);
        }

        if (msg.type === "snapshot") {
          const payload = asRecord(msg.payload);
          const st = payload?.status;
          if (st) setStatus(String(st));
          const prog = asRecord(payload?.progress);
          if (prog?.stage) {
            const stage = String(prog.stage);
            setCurrentStage(stage);
            setStageDetail(prog.detail ? String(prog.detail) : null);
            lastStageRef.current = stage;
          }
          return;
        }

        if (msg.type === "status") {
          const payload = asRecord(msg.payload);
          const st = payload?.status;
          if (st) {
            setStatus(String(st));
          }
          return;
        }

        if (msg.type === "user_log") {
          const payload = asRecord(msg.payload);
          const message = String(payload?.message || "");
          if (message) {
            const rawLevel = payload?.level;
            const level =
              rawLevel === "warn" || rawLevel === "error" || rawLevel === "info" ? rawLevel : "info";
            const stage = payload?.stage ? String(payload.stage) : undefined;
            addActivity(message, level, stage);
          }
          return;
        }

        if (msg.type === "overlay_track") {
          const t = normalizeOverlayTrack(msg.payload);
          if (t) upsertTrack(t);
          return;
        }

        if (msg.type === "overlay_ready") {
          const payload = asRecord(msg.payload);
          const art = asRecord(payload?.artifact);
          const label = typeof art?.label === "string" ? normalizeOverlayLabel(art.label) : "";
          const url = typeof art?.download_url === "string" ? normalizeOverlayUrl(art.download_url) : "";
          const contentType = typeof art?.content_type === "string" ? art.content_type : "";
          if (!label || !url) return;
          if (label.endsWith(":stats") || label === "stats") return;
          if (contentType && !contentType.startsWith("image/")) return;
          upsertDebugOverlay({ label, url });
          return;
        }

        if (msg.type === "progress") {
          const p = asRecord(msg.payload);
          if (p?.stage) {
            const stage = String(p.stage);
            let detail: string | null = p.detail ? String(p.detail) : null;
            let eta: string | null = null;
            const etaSecs = finiteNumber(p.eta_secs);
            if (etaSecs != null && etaSecs > 0) {
              eta = formatEta(etaSecs);
            }
            const total = finiteNumber(p.total);
            const processed = finiteNumber(p.processed) ?? 0;
            if (stage === "processing_tracks" && total != null && total > 0) {
              detail = `${processed}/${total} tracks`;
              if (eta) {
                detail += ` · ETA ${eta}`;
              }
            }
            if (stage === "kymo_tracking") {
              const parts: string[] = [];
              const pct = finiteNumber(p.pct);
              if (pct != null) {
                parts.push(`${Math.round(pct * 100)}%`);
              }
              if (eta) {
                parts.push(`ETA ${eta}`);
              }
              const tracksFound = finiteNumber(p.tracks_found);
              if (tracksFound != null && tracksFound > 0) {
                parts.push(`${tracksFound} tracks`);
              }
              if (parts.length > 0) {
                detail = parts.join(" · ");
              }
            } else if (stage !== "processing_tracks") {
              eta = null;
            }
            setCurrentStage(stage);
            setStageDetail(detail);
            setEtaText(eta);
            lastStageRef.current = stage;
            if (stage === "kymo_done") {
              refreshDebugOverlays(id);
            }
          }
          return;
        }

        if (msg.type === "ping") return;
      };

      ws.onerror = () => {
        if (myToken !== wsTokenRef.current) return;
      };

      ws.onclose = (ev) => {
        if (myToken !== wsTokenRef.current) return;

        if (reconnectRef.current.stop) return;

        if (ev.code === 4404) {
          clearMissingJobSession(id);
          return;
        }

        if (ev.code === 4401) {
          addActivity("Live updates unavailable (auth/ownership)", "warn");
          return;
        }

        reconnectRef.current.tries += 1;
        const delay = Math.min(5000, 250 * 2 ** reconnectRef.current.tries);

        setTimeout(() => {
          if (myToken !== wsTokenRef.current) return;
          if (reconnectRef.current.stop) return;
          doConnect(lastSeqRef.current);
        }, delay);
      };
    };

    doConnect(afterSeq);
  }

  async function runJob(file: File, configOverride?: unknown, runName?: string) {
    setStatus("creating job…");
    setTracks([]);
    setBaseImageUrl(null);
    setOriginalImageUrl(null);
    setCurrentStage("init");
    setStageDetail(null);
    setActivity([]);
    setDebugOverlays([]);
    addActivity("Creating job");

    const safeName = (runName || "").trim() || "untitled";
    const job = await createJob(safeName, configOverride ?? {});
    jobIdRef.current = job.id;
    setJobId(job.id);

    lastSeqRef.current = 0;
    saveSession(job.id, 0);

    try {
      setStatus("creating upload session…");
      const sess = await createUploadSession(job.id, file);
      addActivity("Upload session created");

      setStatus("uploading to storage…");
      await uploadToResumableUrl(sess.upload_url, file);
      addActivity("Uploading file to storage");

      setStatus("finalizing upload…");
      await uploadComplete(job.id, sess.blob_path, file);
      addActivity("Upload complete");
    } catch {
      addActivity("Resumable upload unavailable, using direct upload", "warn");
      setStatus("uploading via api…");
      await uploadViaApi(job.id, file);
      addActivity("Direct upload complete");
    }

    if (isImageFile(file)) {
      await refreshOriginalImage(job.id, true);
    }

    setStatus("starting job…");
    await startJob(job.id);

    connectWsWithRetry(job.id, 0);
    pollForBaseHeatmap(job.id);
    setStatus("running…");
  }

  async function cancelCurrentJob() {
    if (!jobId) return;
    try {
      await cancelJob(jobId);
      addActivity("Cancel requested", "warn", "cancel");
    } catch {
      addActivity("Cancel failed", "error", "cancel");
    }
  }

  function loadJob(id: string) {
    jobIdRef.current = id;
    setJobId(id);
    setStatus("resuming…");
    setTracks([]);
    setBaseImageUrl(null);
    setOriginalImageUrl(null);
    setCurrentStage("resuming");
    setStageDetail(null);
    setActivity([]);
    setDebugOverlays([]);
    lastSeqRef.current = 0;
    saveSession(id, 0);
    pollForBaseHeatmap(id);
    connectWsWithRetry(id, 0);
  }

  async function resumeSavedJob(id: string) {
    try {
      await getJob(id);
    } catch (error) {
      if (isApiError(error, 404)) {
        clearMissingJobSession(id);
        return;
      }
      // For transient startup/network failures, keep the old retry path alive.
    }

    lastSeqRef.current = 0;
    saveSession(id, 0);
    pollForBaseHeatmapRef.current(id);
    connectWsWithRetryRef.current(id, 0);
  }

  function reconnect() {
    if (!jobId) return;
    addActivity("Reconnecting to live updates");
    connectWsWithRetry(jobId, 0);
    pollForBaseHeatmap(jobId);
  }

  function clearSession() {
    closeWs("clear-session");
    pollTokenRef.current += 1;
    localStorage.removeItem(SESSION_KEY);
    jobIdRef.current = null;
    setJobId(null);
    setStatus("idle");
    setTracks([]);
    setBaseImageUrl(null);
    setOriginalImageUrl(null);
    lastSeqRef.current = 0;
    setCurrentStage("idle");
    setStageDetail(null);
    setEtaText(null);
    setActivity([]);
    setDebugOverlays([]);
    addActivity("Session cleared");
  }

  useEffect(() => {
    jobIdRef.current = jobId;
  }, [jobId]);

  useEffect(() => {
    pollForBaseHeatmapRef.current = pollForBaseHeatmap;
    connectWsWithRetryRef.current = connectWsWithRetry;
    resumeSavedJobRef.current = (id: string) => {
      void resumeSavedJob(id);
    };
  });

  useEffect(() => {
    if (!initialSession || resumeStartedRef.current) return;

    resumeStartedRef.current = true;
    const id = initialSession.jobId;
    const timeout = window.setTimeout(() => resumeSavedJobRef.current(id), 0);
    return () => window.clearTimeout(timeout);
  }, [initialSession]);

  useEffect(() => {
    return () => {
      pollTokenRef.current += 1;
      closeWs("unmount");
    };
  }, []);

  return {
    jobId,
    status,
    baseImageUrl,
    originalImageUrl,
    tracks,
    activity,
    currentStage,
    stageDetail,
    etaText,
    debugOverlays,
    runJob,
    cancelCurrentJob,
    loadJob,
    reconnect,
    clearSession,
    addActivity,
    setStatus,
  };
}

export type JobSessionState = ReturnType<typeof useJobSession>;
