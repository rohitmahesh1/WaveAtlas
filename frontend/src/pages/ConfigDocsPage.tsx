// src/pages/ConfigDocsPage.tsx
import React, { useEffect, useMemo, useState } from "react";
import { API_BASE } from "../api";

type Block =
  | { type: "h1" | "h2" | "h3" | "h4" | "p"; text: string }
  | { type: "ul" | "ol"; items: string[] };

function parseMarkdown(md: string): Block[] {
  const lines = md.split(/\r?\n/);
  const blocks: Block[] = [];
  let list: { type: "ul" | "ol"; items: string[] } | null = null;
  let paragraph: string[] = [];

  const flushParagraph = () => {
    if (paragraph.length) {
      blocks.push({ type: "p", text: paragraph.join(" ") });
      paragraph = [];
    }
  };

  const flushList = () => {
    if (list) {
      blocks.push(list);
      list = null;
    }
  };

  for (const raw of lines) {
    const line = raw.trim();
    if (!line) {
      flushParagraph();
      flushList();
      continue;
    }

    const headerMatch = line.match(/^(#{1,6})\s+(.*)$/);
    if (headerMatch) {
      const level = headerMatch[1].length;
      const text = headerMatch[2].trim();
      flushParagraph();
      flushList();
      if (level === 1) blocks.push({ type: "h1", text });
      else if (level === 2) blocks.push({ type: "h2", text });
      else if (level === 3) blocks.push({ type: "h3", text });
      else blocks.push({ type: "h4", text });
      continue;
    }

    if (/^\d+\.\s+/.test(line)) {
      flushParagraph();
      if (!list || list.type !== "ol") {
        flushList();
        list = { type: "ol", items: [] };
      }
      list.items.push(line.replace(/^\d+\.\s+/, ""));
      continue;
    }

    if (line.startsWith("- ")) {
      flushParagraph();
      if (!list || list.type !== "ul") {
        flushList();
        list = { type: "ul", items: [] };
      }
      list.items.push(line.slice(2));
      continue;
    }

    flushList();
    paragraph.push(line);
  }

  flushParagraph();
  flushList();
  return blocks;
}

function renderInline(text: string) {
  const parts = text.split(/`/g);
  const out: React.ReactNode[] = [];

  const pushEmphasis = (chunk: string, keyPrefix: string) => {
    let i = 0;
    while (i < chunk.length) {
      if (chunk.startsWith("**", i)) {
        const end = chunk.indexOf("**", i + 2);
        if (end !== -1) {
          const content = chunk.slice(i + 2, end);
          out.push(
            <strong key={`${keyPrefix}-b-${i}`}>{content}</strong>
          );
          i = end + 2;
          continue;
        }
      }
      if (chunk.startsWith("*", i)) {
        const end = chunk.indexOf("*", i + 1);
        if (end !== -1) {
          const content = chunk.slice(i + 1, end);
          out.push(
            <em key={`${keyPrefix}-i-${i}`}>{content}</em>
          );
          i = end + 1;
          continue;
        }
      }
      const next = chunk.indexOf("*", i);
      const sliceEnd = next === -1 ? chunk.length : next;
      const plain = chunk.slice(i, sliceEnd);
      if (plain) out.push(<span key={`${keyPrefix}-t-${i}`}>{plain}</span>);
      i = sliceEnd;
    }
  };

  parts.forEach((part, idx) => {
    if (idx % 2 === 1) {
      out.push(<code key={`code-${idx}-${part.slice(0, 6)}`}>{part}</code>);
    } else {
      pushEmphasis(part, `txt-${idx}`);
    }
  });

  return out;
}

export default function ConfigDocsPage() {
  const [content, setContent] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(`${API_BASE}/api/docs/config`, { credentials: "include" });
        if (!res.ok) throw new Error(await res.text());
        const text = await res.text();
        if (!cancelled) setContent(text);
      } catch (err: any) {
        if (!cancelled) setError("Failed to load config docs.");
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    load();
    return () => {
      cancelled = true;
    };
  }, []);

  const blocks = useMemo(() => parseMarkdown(content), [content]);

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="brand">
          <div className="brand-title">WaveAtlas</div>
          <div className="brand-sub">Config Docs</div>
        </div>
      </header>

      <main className="app-main docs-main">
        <section className="viewer docs-viewer">
          <div className="panel docs-panel">
            <div className="panel-title">Config Reference</div>
            <div className="panel-body">
              {loading ? <div className="empty-text">Loading docs…</div> : null}
              {error ? <div className="error-text">{error}</div> : null}
              {!loading && !error ? (
                <div className="docs-content">
                  {blocks.map((block, idx) => {
                    if (block.type === "h1") return <h1 key={idx}>{renderInline(block.text)}</h1>;
                    if (block.type === "h2") return <h2 key={idx}>{renderInline(block.text)}</h2>;
                    if (block.type === "h3") return <h3 key={idx}>{renderInline(block.text)}</h3>;
                    if (block.type === "h4") return <h4 key={idx}>{renderInline(block.text)}</h4>;
                    if (block.type === "p") return <p key={idx}>{renderInline(block.text)}</p>;
                    if (block.type === "ul") {
                      return (
                        <ul key={idx}>
                          {block.items.map((item, i) => (
                            <li key={i}>{renderInline(item)}</li>
                          ))}
                        </ul>
                      );
                    }
                    return (
                      <ol key={idx}>
                        {block.items.map((item, i) => (
                          <li key={i}>{renderInline(item)}</li>
                        ))}
                      </ol>
                    );
                  })}
                </div>
              ) : null}
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
