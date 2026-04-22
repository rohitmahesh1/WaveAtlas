import { API_BASE, apiUrl } from "../api";

type CsvRow = Record<string, unknown>;

function saveBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
}

function clickUrl(url: string, filename: string) {
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.target = "_blank";
  anchor.rel = "noopener noreferrer";
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
}

function shouldFetchDownload(url: string) {
  try {
    const target = new URL(url, window.location.href);
    const apiOrigin = API_BASE ? new URL(API_BASE, window.location.href).origin : window.location.origin;
    return target.origin === window.location.origin || target.origin === apiOrigin;
  } catch {
    return true;
  }
}

function csvCell(value: unknown) {
  if (value == null) return "";
  const text = typeof value === "object" ? JSON.stringify(value) : String(value);
  return /[",\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

export function rowsToCsv(rows: CsvRow[], headers: string[]) {
  return [
    headers.map(csvCell).join(","),
    ...rows.map((row) => headers.map((header) => csvCell(row[header])).join(",")),
  ].join("\n");
}

export function downloadText(filename: string, text: string, contentType = "text/plain;charset=utf-8") {
  saveBlob(new Blob([text], { type: contentType }), filename);
}

export function downloadJson(filename: string, data: unknown) {
  downloadText(filename, `${JSON.stringify(data, null, 2)}\n`, "application/json;charset=utf-8");
}

export function downloadCsv(filename: string, rows: CsvRow[], headers: string[]) {
  downloadText(filename, `${rowsToCsv(rows, headers)}\n`, "text/csv;charset=utf-8");
}

export async function downloadFromUrl(url: string, filename: string) {
  const resolvedUrl = apiUrl(url);
  if (!shouldFetchDownload(resolvedUrl)) {
    clickUrl(resolvedUrl, filename);
    return;
  }

  const response = await fetch(resolvedUrl, { credentials: "include" });
  if (!response.ok) throw new Error(await response.text());
  saveBlob(await response.blob(), filename);
}
