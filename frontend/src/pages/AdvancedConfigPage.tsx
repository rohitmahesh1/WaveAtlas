// src/pages/AdvancedConfigPage.tsx
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { stageLabel } from "../utils/format";
import { API_BASE, resumeJob } from "../api";
import { RunPanel } from "../components/RunPanel";
import { ActivityPanel } from "../components/ActivityPanel";
import { useJobSession } from "../hooks/useJobSession";

const DEFAULT_CONFIG_TEXT = `# Loading config...`;

function escapeHtml(text: string) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function highlightYaml(text: string) {
  const lines = text.split("\n");
  const out: string[] = [];
  for (const line of lines) {
    const trimmed = line.trimStart();
    if (trimmed.startsWith("#")) {
      out.push(`<span class="yaml-comment">${escapeHtml(line)}</span>`);
      continue;
    }
    const match = line.match(/^(\s*)([^:#]+?):(.*)$/);
    if (match) {
      const [, indent, key, rest] = match;
      out.push(
        `${escapeHtml(indent)}<span class="yaml-key">${escapeHtml(key)}</span>:${escapeHtml(rest)}`
      );
    } else {
      out.push(escapeHtml(line));
    }
  }
  return out.join("\n");
}

export default function AdvancedConfigPage() {
  const [file, setFile] = useState<File | null>(null);
  const [runName, setRunName] = useState<string>("");
  const [runNameAuto, setRunNameAuto] = useState<boolean>(true);
  const runCounterRef = useRef<number>(1);
  const [configText, setConfigText] = useState<string>(DEFAULT_CONFIG_TEXT);
  const [configDirty, setConfigDirty] = useState<boolean>(false);
  const [configLoading, setConfigLoading] = useState<boolean>(true);
  const [configError, setConfigError] = useState<string | null>(null);
  const [configValidatedText, setConfigValidatedText] = useState<string | null>(null);
  const [validationState, setValidationState] = useState<"idle" | "validating" | "valid" | "error">("idle");
  const [validationMessage, setValidationMessage] = useState<string | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const preRef = useRef<HTMLPreElement | null>(null);

  const highlightedHtml = useMemo(() => highlightYaml(configText), [configText]);

  const {
    jobId,
    status,
    activity,
    currentStage,
    stageDetail,
    tracks,
    runJob,
    cancelCurrentJob,
    loadJob,
    clearSession,
  } = useJobSession();

  const buildDefaultRunName = (file: File) => {
    const raw = file.name || "run";
    const base = raw.replace(/\.[^/.]+$/, "") || "run";
    const num = runCounterRef.current;
    runCounterRef.current += 1;
    return `${base} #${num}`;
  };

  const handleFileChange = (nextFile: File | null) => {
    setFile(nextFile);
    if (nextFile && (runNameAuto || !runName.trim())) {
      setRunName(buildDefaultRunName(nextFile));
      setRunNameAuto(true);
    }
    if (!nextFile && runNameAuto) {
      setRunName("");
    }
  };

  const stageText = stageDetail ? `${stageLabel(currentStage)} — ${stageDetail}` : stageLabel(currentStage);
  const statusLabel = String(status).replace(/_/g, " ");
  const showSpinner = !["completed", "failed", "cancelled", "idle"].includes(String(status));

  const loadDefaultConfig = useCallback(
    async (force = false) => {
      setConfigLoading(true);
      setConfigError(null);
      try {
        const base = API_BASE || "";
        const res = await fetch(`${base}/api/config/default`, { credentials: "include" });
        if (!res.ok) throw new Error(await res.text());
        const text = await res.text();
        if (!configDirty || force) {
          setConfigText(text);
          setConfigDirty(false);
          setConfigValidatedText(null);
          setValidationState("idle");
          setValidationMessage(null);
        }
      } catch {
        setConfigError("Failed to load default config");
      } finally {
        setConfigLoading(false);
      }
    },
    [configDirty]
  );

  useEffect(() => {
    loadDefaultConfig(false);
  }, [loadDefaultConfig]);

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="brand">
          <div className="brand-title">WaveAtlas</div>
          <div className="brand-sub">Advanced Config</div>
        </div>
        <div className="status-cluster">
          <div className={`status-pill status-${status}`}>Status: {statusLabel}</div>
          <div className="stage-pill">
            {showSpinner ? <span className="spinner" /> : null}
            <span>{stageText}</span>
          </div>
        </div>
      </header>

      <main className="app-main">
        <aside className="sidebar">
          <RunPanel
            file={file}
            onFileChange={handleFileChange}
            onRun={() => {
              if (!file) return;
              if (!configValidatedText || configValidatedText !== configText) {
                setValidationState("error");
                setValidationMessage("Please submit config before running.");
                return;
              }
              runJob(file, configValidatedText.trim(), runName);
            }}
            jobId={jobId}
            status={status}
            runName={runName}
            onRunNameChange={(value) => {
              setRunName(value);
              setRunNameAuto(false);
            }}
            filteredCount={tracks.length}
            totalCount={tracks.length}
            onCancel={cancelCurrentJob}
            cancelDisabled={!jobId || ["completed", "failed", "cancelled"].includes(status)}
            onResume={async () => {
              if (!jobId || status !== "cancelled") return;
              try {
                await resumeJob(jobId);
                loadJob(jobId);
              } catch {
                // no-op
              }
            }}
            onNewRun={() => {
              clearSession();
              setFile(null);
              setRunName("");
              setRunNameAuto(true);
            }}
          />

          <ActivityPanel activity={activity} />
        </aside>

        <section className="viewer">
          <div className="viewer-top">
            <div className="viewer-meta">Edit config before running</div>
          </div>
          <section className="panel">
            <div className="panel-title">Config (YAML or JSON)</div>
            <div className="panel-body">
              <div className="config-editor">
                <textarea
                  ref={textareaRef}
                  className="config-textarea"
                  value={configText}
                  onChange={(e) => {
                    setConfigText(e.target.value);
                    setConfigDirty(true);
                    setConfigValidatedText(null);
                    setValidationState("idle");
                    setValidationMessage(null);
                  }}
                  onScroll={() => {
                    if (preRef.current && textareaRef.current) {
                      preRef.current.scrollTop = textareaRef.current.scrollTop;
                      preRef.current.scrollLeft = textareaRef.current.scrollLeft;
                    }
                  }}
                  spellCheck={false}
                />
                <pre
                  ref={preRef}
                  className="config-pre"
                  aria-hidden="true"
                  dangerouslySetInnerHTML={{ __html: highlightedHtml + "\n" }}
                />
              </div>
              <div className="config-hint">
                {configLoading ? "Loading config from disk…" : "Config text is sent as-is and parsed on the server."}
                {configError ? ` ${configError}` : ""}
                {validationState === "error" && validationMessage ? ` ${validationMessage}` : ""}
                {validationState === "valid" ? " Config validated." : ""}
              </div>
              <div className="config-actions">
                <button className="ghost-btn" onClick={() => setConfigText("")}>
                  Clear
                </button>
                <button className="ghost-btn" onClick={() => loadDefaultConfig(true)}>
                  Reload default
                </button>
                <button
                  className="primary-btn"
                  onClick={async () => {
                    setValidationState("validating");
                    setValidationMessage(null);
                    try {
                      const base = API_BASE || "";
                      const res = await fetch(`${base}/api/config/validate`, {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        credentials: "include",
                        body: JSON.stringify({ config: configText }),
                      });
                      if (!res.ok) throw new Error(await res.text());
                      setValidationState("valid");
                      setValidationMessage(null);
                      setConfigValidatedText(configText);
                      setConfigDirty(false);
                    } catch {
                      setValidationState("error");
                      setValidationMessage("Config validation failed");
                    }
                  }}
                  disabled={configLoading || validationState === "validating"}
                >
                  {validationState === "validating" ? "Validating…" : "Submit config"}
                </button>
              </div>
            </div>
          </section>
        </section>
      </main>
    </div>
  );
}
