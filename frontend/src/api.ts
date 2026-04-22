// src/api.ts
export const API_BASE = import.meta.env.VITE_API_BASE || ""; // "" => same origin

export type JsonPrimitive = string | number | boolean | null;
export type JsonValue = JsonPrimitive | JsonValue[] | { [key: string]: JsonValue };
export type JobProgress = Record<string, JsonValue>;

export type Job = { id: string; status: string; progress: JobProgress | null };
export type JobRead = {
  id: string;
  owner_session_id: string;
  run_name: string;
  status: string;
  cancel_requested: boolean;
  error?: string | null;
  error_code?: string | null;
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  updated_at: string;
  progress: JobProgress | null;
  tracks_total?: number | null;
  tracks_done: number;
  waves_done: number;
  peaks_done: number;
  input_filename?: string | null;
};
export type ArtifactView = {
  id: string;
  kind: string;
  label?: string | null;
  download_url: string;
  content_type?: string | null;
};

export type TrackDetail = {
  track_index: number;
  time_index: number[];
  position: number[];
  baseline: number[];
  residual?: number[] | null;
  sine_fit?: number[] | null;
  peaks: number[];
  peaks_in_slice: number[];
  strongest_peak_idx?: number | null;
  metrics: {
    dominant_frequency?: number | null;
    period?: number | null;
    num_peaks?: number | null;
    mean_amplitude?: number | null;
  };
};

export async function createJob(runName = "mvp", config: unknown = {}): Promise<JobRead> {
  const res = await fetch(`${API_BASE}/api/jobs`, {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ run_name: runName, config }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function startJob(jobId: string): Promise<JobRead> {
  const res = await fetch(`${API_BASE}/api/jobs/${jobId}/start`, {
    method: "POST",
    credentials: "include",
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function listArtifacts(
  jobId: string,
  opts?: { kind?: string; label?: string; limit?: number }
): Promise<ArtifactView[]> {
  const params = new URLSearchParams();
  if (opts?.kind) params.set("kind", opts.kind);
  if (opts?.label) params.set("label", opts.label);
  if (opts?.limit) params.set("limit", String(opts.limit));
  const qs = params.toString();
  const res = await fetch(`${API_BASE}/api/jobs/${jobId}/artifacts${qs ? `?${qs}` : ""}`, {
    method: "GET",
    credentials: "include",
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function listJobs(limit = 50, offset = 0): Promise<JobRead[]> {
  const res = await fetch(`${API_BASE}/api/jobs?limit=${limit}&offset=${offset}`, {
    method: "GET",
    credentials: "include",
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function cancelJob(jobId: string): Promise<JobRead> {
  const res = await fetch(`${API_BASE}/api/jobs/${jobId}/cancel`, {
    method: "POST",
    credentials: "include",
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function deleteJob(jobId: string): Promise<{ ok: boolean; deleted?: unknown; blob_errors?: string[] }> {
  const res = await fetch(`${API_BASE}/api/jobs/${jobId}`, {
    method: "DELETE",
    credentials: "include",
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function resumeJob(jobId: string): Promise<JobRead> {
  const res = await fetch(`${API_BASE}/api/jobs/${jobId}/resume`, {
    method: "POST",
    credentials: "include",
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function updateJobName(jobId: string, runName: string): Promise<JobRead> {
  const res = await fetch(`${API_BASE}/api/jobs/${jobId}/name`, {
    method: "PATCH",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ run_name: runName }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getTrackDetail(
  jobId: string,
  trackIndex: number,
  opts?: { include_sine?: boolean; include_residual?: boolean; range?: string; signal?: AbortSignal }
): Promise<TrackDetail> {
  const params = new URLSearchParams();
  if (opts?.include_sine) params.set("include_sine", "true");
  if (opts?.include_residual) params.set("include_residual", "true");
  if (opts?.range) params.set("range", opts.range);
  const qs = params.toString();
  const res = await fetch(
    `${API_BASE}/api/jobs/${jobId}/tracks/${trackIndex}/detail${qs ? `?${qs}` : ""}`,
    {
      method: "GET",
      credentials: "include",
      signal: opts?.signal,
    }
  );
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// Small/dev upload (goes through backend)
export async function uploadViaApi(jobId: string, file: File): Promise<unknown> {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(`${API_BASE}/api/jobs/${jobId}/upload`, {
    method: "POST",
    credentials: "include",
    body: fd,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// Large/prod upload (direct to GCS resumable)
export async function createUploadSession(jobId: string, file: File) {
  const params = new URLSearchParams({
    filename: file.name,
    content_type: file.type || "text/csv",
  });
  const res = await fetch(`${API_BASE}/api/jobs/${jobId}/upload-session?${params}`, {
    method: "POST",
    credentials: "include",
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<{ upload_url: string; blob_path: string; content_type: string }>;
}

export async function uploadComplete(jobId: string, blobPath: string, file: File): Promise<unknown> {
  const res = await fetch(`${API_BASE}/api/jobs/${jobId}/upload-complete`, {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      blob_path: blobPath,
      filename: file.name,
      content_type: file.type || "text/csv",
      byte_size: file.size,
    }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// Minimal resumable upload: one-shot PUT (works for many cases, not fully chunk-resumable)
// Good enough for “backend test”; you can upgrade to chunked later.
export async function uploadToResumableUrl(uploadUrl: string, file: File) {
  const res = await fetch(uploadUrl, {
    method: "PUT",
    headers: {
      "Content-Type": file.type || "application/octet-stream",
      "Content-Range": `bytes 0-${file.size - 1}/${file.size}`,
    },
    body: file,
  });
  if (!(res.status === 200 || res.status === 201)) {
    throw new Error(`Resumable upload failed: ${res.status} ${await res.text()}`);
  }
}

export function wsUrl(path: string, params?: Record<string, string | number | boolean | null | undefined>) {
  const base = API_BASE || window.location.origin;
  const u = new URL(base);

  // switch scheme
  u.protocol = u.protocol === "https:" ? "wss:" : "ws:";

  // IMPORTANT: pathname must NOT include "?"
  u.pathname = path;
  u.search = "";

  if (params) {
    for (const [k, v] of Object.entries(params)) {
      if (v === null || v === undefined) continue;
      u.searchParams.set(k, String(v));
    }
  }

  return u.toString();
}
