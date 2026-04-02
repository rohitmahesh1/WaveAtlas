import { useEffect, useRef, useState } from "react";
import {
  createJob,
  startJob,
  listArtifacts,
  uploadViaApi,
  createUploadSession,
  uploadToResumableUrl,
  uploadComplete,
  wsUrl,
  cancelJob,
  API_BASE,
} from "../api";
import type { OverlayTrackEvent } from "../OverlayCanvas";
import type { LogEntry } from "../types";
import { formatEta } from "../utils/format";

type WsMsg =
  | { type: "snapshot"; payload: any; seq: number; job_id?: string }
  | { type: "overlay_track"; payload: any; seq: number; job_id?: string }
  | { type: "status"; payload: any; seq: number; job_id?: string }
  | { type: "progress"; payload: any; seq: number; job_id?: string }
  | { type: "user_log"; payload: any; seq: number; job_id?: string }
  | { type: "ping"; payload: any; seq: number; job_id?: string }
  | { type: string; payload: any; seq: number; job_id?: string };

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

function normalizeOverlayTrack(payload: any): OverlayTrackEvent | null {
  if (!payload) return null;

  const idx = Number(payload.track_index);
  if (!Number.isFinite(idx)) return null;

  const peaks = Array.isArray(payload.peaks)
    ? payload.peaks.map((x: any) => ({ x: x.x, y: x.y, amp: x.amp }))
    : [];
  const ampVals = peaks.map((p) => Number(p.amp)).filter((v) => Number.isFinite(v));
  const meanAmp = ampVals.length ? ampVals.reduce((a, b) => a + b, 0) / ampVals.length : null;

  return {
    id: idx,
    sample: payload.sample,
    track_index: idx,
    poly: Array.isArray(payload.poly) ? payload.poly : [],
    peaks,
    metrics: {
      mean_amplitude: meanAmp,
      dominant_frequency: Number.isFinite(payload.freq_hz) ? Number(payload.freq_hz) : null,
      period: Number.isFinite(payload.period) ? Number(payload.period) : null,
      num_peaks: peaks.length,
    },
  };
}

export function useJobSession(options?: { resumeOnMount?: boolean }) {
  const resumeOnMount = options?.resumeOnMount ?? false;
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<string>("idle");
  const [baseImageUrl, setBaseImageUrl] = useState<string | null>(null);
  const [tracks, setTracks] = useState<OverlayTrackEvent[]>([]);
  const [activity, setActivity] = useState<LogEntry[]>([]);
  const [currentStage, setCurrentStage] = useState<string>("idle");
  const [stageDetail, setStageDetail] = useState<string | null>(null);
  const [etaText, setEtaText] = useState<string | null>(null);
  const [debugOverlays, setDebugOverlays] = useState<{ label: string; url: string }[]>([]);

  const wsRef = useRef<WebSocket | null>(null);
  const lastSeqRef = useRef<number>(0);
  const wsTokenRef = useRef<number>(0);
  const reconnectRef = useRef<{ stop: boolean; tries: number }>({ stop: false, tries: 0 });
  const lastStageRef = useRef<string>("");
  const pollTokenRef = useRef<number>(0);

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
    } catch {
      // ignore
    }
  }

  async function pollForBaseHeatmap(id: string) {
    const myToken = ++pollTokenRef.current;
    for (let i = 0; i < 60; i++) {
      if (pollTokenRef.current !== myToken) return;
      try {
        const [overlayArts, baseArts] = await Promise.all([
          listArtifacts(id, { kind: "overlay", limit: 2000 }),
          listArtifacts(id, { kind: "base_heatmap", limit: 1 }),
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
      } catch (e: any) {
        // Keep polling quietly; user log stays clean.
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

        const msg: WsMsg = JSON.parse(ev.data);

        if (typeof msg.seq === "number") {
          lastSeqRef.current = Math.max(lastSeqRef.current, msg.seq);
          saveSession(id, lastSeqRef.current);
        }

        if (msg.type === "snapshot") {
          const st = msg.payload?.status;
          if (st) setStatus(String(st));
          const prog = msg.payload?.progress;
          if (prog?.stage) {
            const stage = String(prog.stage);
            setCurrentStage(stage);
            setStageDetail(prog.detail ? String(prog.detail) : null);
            lastStageRef.current = stage;
          }
          return;
        }

        if (msg.type === "status") {
          const st = msg.payload?.status;
          if (st) {
            setStatus(String(st));
          }
          return;
        }

        if (msg.type === "user_log") {
          const message = String(msg.payload?.message || "");
          if (message) {
            const level = (msg.payload?.level as "info" | "warn" | "error") || "info";
            const stage = msg.payload?.stage ? String(msg.payload.stage) : undefined;
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
          const art = msg.payload?.artifact || {};
          const label = typeof art.label === "string" ? normalizeOverlayLabel(art.label) : "";
          const url = typeof art.download_url === "string" ? normalizeOverlayUrl(art.download_url) : "";
          const contentType = typeof art.content_type === "string" ? art.content_type : "";
          if (!label || !url) return;
          if (label.endsWith(":stats") || label === "stats") return;
          if (contentType && !contentType.startsWith("image/")) return;
          upsertDebugOverlay({ label, url });
          return;
        }

        if (msg.type === "progress") {
          const p = msg.payload || {};
          if (p.stage) {
            const stage = String(p.stage);
            let detail: string | null = p.detail ? String(p.detail) : null;
            let eta: string | null = null;
            if (Number.isFinite(p.eta_secs) && Number(p.eta_secs) > 0) {
              eta = formatEta(Number(p.eta_secs));
            }
            if (stage === "processing_tracks" && Number.isFinite(p.total) && Number(p.total) > 0) {
              detail = `${Number(p.processed || 0)}/${Number(p.total)} tracks`;
              if (eta) {
                detail += ` · ETA ${eta}`;
              }
            }
            if (stage === "kymo_tracking") {
              const parts: string[] = [];
              if (Number.isFinite(p.pct)) {
                parts.push(`${Math.round(Number(p.pct) * 100)}%`);
              }
              if (eta) {
                parts.push(`ETA ${eta}`);
              }
              if (Number.isFinite(p.tracks_found) && Number(p.tracks_found) > 0) {
                parts.push(`${Number(p.tracks_found)} tracks`);
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
            if (jobId && stage === "kymo_done") {
              refreshDebugOverlays(jobId);
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

        if (ev.code === 4401 || ev.code === 4404) {
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
    setCurrentStage("init");
    setStageDetail(null);
    setActivity([]);
    setDebugOverlays([]);
    addActivity("Creating job");

    const safeName = (runName || "").trim() || "untitled";
    const job = await createJob(safeName, configOverride ?? {});
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
    } catch (e: any) {
      addActivity("Resumable upload unavailable, using direct upload", "warn");
      setStatus("uploading via api…");
      await uploadViaApi(job.id, file);
      addActivity("Direct upload complete");
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
    } catch (err: any) {
      addActivity("Cancel failed", "error", "cancel");
    }
  }

  function loadJob(id: string) {
    setJobId(id);
    setStatus("resuming…");
    setTracks([]);
    setBaseImageUrl(null);
    setCurrentStage("resuming");
    setStageDetail(null);
    setActivity([]);
    setDebugOverlays([]);
    lastSeqRef.current = 0;
    saveSession(id, 0);
    pollForBaseHeatmap(id);
    connectWsWithRetry(id, 0);
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
    setJobId(null);
    setStatus("idle");
    setTracks([]);
    setBaseImageUrl(null);
    lastSeqRef.current = 0;
    setCurrentStage("idle");
    setStageDetail(null);
    setEtaText(null);
    setActivity([]);
    setDebugOverlays([]);
    addActivity("Session cleared");
  }

  useEffect(() => {
    if (!resumeOnMount) return;
    const sess = loadSession();
    if (!sess) return;

    setJobId(sess.jobId);
    setStatus("resuming…");
    setTracks([]);
    setBaseImageUrl(null);
    setCurrentStage("resuming");
    setStageDetail(null);
    setEtaText(null);

    lastSeqRef.current = 0;
    saveSession(sess.jobId, 0);

    pollForBaseHeatmap(sess.jobId);
    connectWsWithRetry(sess.jobId, 0);
  }, []);

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
