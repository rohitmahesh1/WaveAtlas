# app/io/table_to_heatmap.py
from __future__ import annotations

import io
import os
import re
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

# Headless backend for servers/Cloud Run (mirrors the intent of your current module)
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib  # noqa: E402
matplotlib.use("Agg", force=True)  # noqa: E402


_AREA_FILENAME_RE = re.compile(r"(^|[^a-z0-9])area([^a-z0-9]|$)", re.IGNORECASE)
_EXTREME_TABLE_MODES = {"extreme", "extremes", "extreme_mask", "intensity", "legacy"}
_CONTINUOUS_TABLE_MODES = {"area", "continuous", "raw"}


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


def _looks_like_area_table(filename_hint: Optional[str]) -> bool:
    if not filename_hint:
        return False
    return bool(_AREA_FILENAME_RE.search(os.path.basename(str(filename_hint))))


def _resolve_table_mode(heat_cfg: Dict[str, Any], filename_hint: Optional[str]) -> Tuple[str, str]:
    requested = str(heat_cfg.get("table_mode", "auto")).strip().lower() or "auto"
    if requested == "auto":
        return requested, "area" if _looks_like_area_table(filename_hint) else "extreme_mask"
    if requested in _EXTREME_TABLE_MODES:
        return requested, "extreme_mask"
    if requested in _CONTINUOUS_TABLE_MODES:
        return requested, "area" if requested == "area" else "continuous"
    raise ValueError(
        "Unsupported heatmap.table_mode "
        f"{requested!r}. Use auto, area, continuous, intensity, or legacy."
    )


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


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
      table_mode: str = "auto"
      lower: float = -1e20
      upper: float =  1e16
      binarize: bool = True
      origin: str = "lower"
      cmap: str = "hot"
      dpi: int = 180
      area: dict = area-specific overrides for auto-detected area tables

    Returns:
      (png_bytes, meta)
    """
    cfg = config or {}
    heat_cfg = cfg.get("heatmap", cfg)
    requested_table_mode, resolved_table_mode = _resolve_table_mode(heat_cfg, filename_hint)

    lower = float(heat_cfg.get("lower", -1e20))
    upper = float(heat_cfg.get("upper", 1e16))
    dpi = heat_cfg.get("dpi", 180)
    dpi_val: Optional[int] = int(dpi) if dpi is not None else None

    df, load_meta = _load_table_bytes(table_bytes, filename_hint=filename_hint)

    # Convert to float and sanitize NaN/Inf
    data = df.to_numpy(dtype=float)
    if not np.isfinite(data).all():
        data = np.nan_to_num(data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    if resolved_table_mode in {"area", "continuous"}:
        mode_cfg = {}
        if resolved_table_mode == "area" and isinstance(heat_cfg.get("area"), dict):
            mode_cfg = dict(heat_cfg["area"])
        filtered = data
        binarize = bool(mode_cfg.get("binarize", False))
        origin = str(mode_cfg.get("origin", heat_cfg.get("origin", "lower")))
        cmap = str(mode_cfg.get("cmap", "plasma" if resolved_table_mode == "area" else heat_cfg.get("cmap", "viridis")))
        vmin = _optional_float(mode_cfg.get("vmin", heat_cfg.get("vmin")))
        vmax = _optional_float(mode_cfg.get("vmax", heat_cfg.get("vmax")))
        if binarize:
            filtered = (filtered > 0).astype(int)
    else:
        binarize = bool(heat_cfg.get("binarize", True))
        origin = str(heat_cfg.get("origin", "lower"))
        cmap = str(heat_cfg.get("cmap", "hot"))
        vmin = 0.0
        # Keep extremes and optionally binarize
        filtered = _keep_extremes_zero_middle(data, lower, upper)
        filtered = np.abs(filtered)
        if binarize:
            filtered = (filtered > 0).astype(int)
        vmax = float(np.max(filtered)) if filtered.size else 1.0

    nrows, ncols = filtered.shape
    if vmin is None:
        vmin = float(np.min(filtered)) if filtered.size else 0.0
    if vmax is None:
        vmax = float(np.max(filtered)) if filtered.size else 1.0

    # Render a direct pixel-for-cell PNG so the output image dimensions match
    # the submitted table dimensions exactly.
    render = np.flipud(filtered) if origin == "lower" else filtered
    norm = np.zeros_like(render, dtype=np.float32)
    scale = float(vmax) - float(vmin)
    if scale > 0:
        norm = np.clip((render.astype(np.float32) - float(vmin)) / scale, 0.0, 1.0)

    cmap_fn = matplotlib.colormaps.get_cmap(cmap)
    rgba = np.asarray(cmap_fn(norm, bytes=True), dtype=np.uint8)
    out_img = Image.fromarray(rgba, mode="RGBA")

    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    meta: Dict[str, Any] = {
        **load_meta,
        "nrows": int(nrows),
        "ncols": int(ncols),
        "source_kind": "table",
        "source_rows": int(nrows),
        "source_cols": int(ncols),
        "output_width": int(ncols),
        "output_height": int(nrows),
        "pixel_mapping": "table_cell",
        "coord_origin": origin,
        "coord_x_label": "col",
        "coord_y_label": "row",
        "table_mode": requested_table_mode,
        "resolved_table_mode": resolved_table_mode,
        "lower": lower,
        "upper": upper,
        "binarize": binarize,
        "origin": origin,
        "cmap": cmap,
        "dpi": dpi_val,
        "vmin": vmin,
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
