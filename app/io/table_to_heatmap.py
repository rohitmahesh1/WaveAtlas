# app/io/table_to_heatmap.py
from __future__ import annotations

import io
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Headless backend for servers/Cloud Run (mirrors the intent of your current module)
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib  # noqa: E402
matplotlib.use("Agg", force=True)  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def _is_xlsx_magic(header: bytes) -> bool:
    # XLSX files are ZIP archives; ZIP magic is PK\x03\x04
    return header.startswith(b"PK\x03\x04")


def _is_xls_magic(header: bytes) -> bool:
    # Legacy Excel (.xls OLE) magic: D0 CF 11 E0 A1 B1 1A E1
    return header.startswith(b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1")


def _decode_text_table(data: bytes) -> str:
    """
    Decode bytes into a string for pandas read_csv. Be permissive for MVP.
    """
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    # last resort: replace invalid chars
    return data.decode("utf-8", errors="replace")


def _load_table_bytes(table_bytes: bytes, *, filename_hint: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load a table from raw bytes. Returns (df, meta).
    Meta includes best-effort info about parsing.
    """
    header = table_bytes[:8]
    meta: Dict[str, Any] = {"filename_hint": filename_hint}

    # True XLSX
    if _is_xlsx_magic(header):
        try:
            df = pd.read_excel(io.BytesIO(table_bytes), header=None, engine="openpyxl")
            meta["format"] = "xlsx"
            return df, meta
        except ImportError as e:
            raise RuntimeError(
                "Input appears to be .xlsx but 'openpyxl' is not installed. Install with: pip install openpyxl"
            ) from e

    # True XLS
    if _is_xls_magic(header):
        try:
            df = pd.read_excel(io.BytesIO(table_bytes), header=None, engine="xlrd")
            meta["format"] = "xls"
            return df, meta
        except ImportError as e:
            raise RuntimeError(
                "Input appears to be .xls but 'xlrd' is not installed. Install with: pip install xlrd"
            ) from e
        except Exception:
            # Some odd/corrupt XLS; fall back to text parsing
            meta["format"] = "xls_fallback_to_text"

    # Text table (CSV/TSV or mislabeled Excel)
    text = _decode_text_table(table_bytes)
    try:
        # sep=None lets pandas sniff delimiter (python engine)
        df = pd.read_csv(io.StringIO(text), sep=None, engine="python", header=None)
        meta["format"] = "text_sniffed"
        meta["delimiter_sniffed"] = True
        return df, meta
    except Exception:
        # One more attempt: assume TSV
        df = pd.read_csv(io.StringIO(text), sep="\t", header=None)
        meta["format"] = "text_tsv_fallback"
        meta["delimiter_sniffed"] = False
        return df, meta


def _keep_extremes_zero_middle(arr: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """
    Zero out values within [lower, upper]; keep extreme values as-is.
    Mirrors your previous 'keep_extreme_values' logic. :contentReference[oaicite:1]{index=1}
    """
    out = arr.copy()
    mask = (out >= lower) & (out <= upper)
    out[mask] = 0
    return out


def table_to_heatmap_bytes(
    table_bytes: bytes,
    *,
    config: Optional[Dict[str, Any]] = None,
    filename_hint: Optional[str] = None,
) -> Tuple[bytes, Dict[str, Any]]:
    """
    Ideal pipeline API:
      table_bytes -> heatmap PNG bytes

    Config: either provide keys at top-level, or under config["heatmap"].
    Supported keys (with defaults):
      lower: float = -1e20
      upper: float =  1e16
      binarize: bool = True
      origin: str = "lower"
      cmap: str = "hot"
      dpi: int = 180

    Returns:
      (png_bytes, meta)
    """
    cfg = config or {}
    heat_cfg = cfg.get("heatmap", cfg)

    lower = float(heat_cfg.get("lower", -1e20))
    upper = float(heat_cfg.get("upper", 1e16))
    binarize = bool(heat_cfg.get("binarize", True))
    origin = str(heat_cfg.get("origin", "lower"))
    cmap = str(heat_cfg.get("cmap", "hot"))
    dpi = heat_cfg.get("dpi", 180)
    dpi_val: Optional[int] = int(dpi) if dpi is not None else None

    df, load_meta = _load_table_bytes(table_bytes, filename_hint=filename_hint)

    # Convert to float and sanitize NaN/Inf
    data = df.to_numpy(dtype=float)
    if not np.isfinite(data).all():
        data = np.nan_to_num(data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # Keep extremes and optionally binarize
    filtered = _keep_extremes_zero_middle(data, lower, upper)
    filtered = np.abs(filtered)
    if binarize:
        filtered = (filtered > 0).astype(int)

    nrows, ncols = filtered.shape
    vmax = float(np.max(filtered)) if filtered.size else 1.0

    # Render to PNG bytes
    buf = io.BytesIO()
    plt.figure(figsize=(8, 6), dpi=(dpi_val if dpi_val else None))
    plt.imshow(filtered, cmap=cmap, interpolation="nearest", vmin=0, vmax=vmax, origin=origin)
    plt.axis("off")
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=(dpi_val if dpi_val else None))
    plt.close()
    buf.seek(0)

    png_bytes = buf.read()

    meta: Dict[str, Any] = {
        **load_meta,
        "nrows": int(nrows),
        "ncols": int(ncols),
        "lower": lower,
        "upper": upper,
        "binarize": binarize,
        "origin": origin,
        "cmap": cmap,
        "dpi": dpi_val,
        "vmax": vmax,
        "png_bytes": len(png_bytes),
    }
    return png_bytes, meta


def table_to_heatmap_file(
    table_bytes: bytes,
    *,
    out_path: str,
    config: Optional[Dict[str, Any]] = None,
    filename_hint: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Optional compatibility helper:
    If any legacy code still wants a file on disk (scratch), this saves it
    and returns meta. Durable publishing should still be done via ArtifactStore.
    """
    png, meta = table_to_heatmap_bytes(table_bytes, config=config, filename_hint=filename_hint)
    with open(out_path, "wb") as f:
        f.write(png)
    meta["out_path"] = out_path
    return meta
