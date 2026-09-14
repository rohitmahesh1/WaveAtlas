# app/io/table_to_heatmap.py
from __future__ import annotations

import io
import os
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

# Headless backend for servers/Cloud Run (mirrors the intent of your current module)
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib  # noqa: E402
matplotlib.use("Agg", force=True)  # noqa: E402

from ..cancel import CancellationRequested
from ..heatmap_values import encode_heatmap_values


_EXTREME_TABLE_MODES = {"binary", "extreme", "extremes", "extreme_mask", "intensity", "legacy"}
_CONTINUOUS_TABLE_MODES = {"area", "continuous", "raw"}


def _check_cancel(cancel_cb: Optional[Callable[[], bool]]) -> None:
    if cancel_cb is not None and cancel_cb():
        raise CancellationRequested("cancel_requested")


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


def _resolve_table_mode(heat_cfg: Dict[str, Any]) -> Tuple[str, str]:
    requested = str(heat_cfg.get("table_mode", "auto")).strip().lower() or "auto"
    if requested == "auto":
        return requested, "continuous"
    if requested in _EXTREME_TABLE_MODES:
        return requested, "extreme_mask"
    if requested in _CONTINUOUS_TABLE_MODES:
        return requested, "area" if requested == "area" else "continuous"
    raise ValueError(
        "Unsupported heatmap.table_mode "
        f"{requested!r}. Use auto, area, continuous, binary, intensity, or legacy."
    )


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def _value_range(values: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    if not values.size:
        return None, None
    return float(np.min(values)), float(np.max(values))


def table_to_heatmap_payload(
    table_bytes: bytes,
    *,
    config: Optional[Dict[str, Any]] = None,
    filename_hint: Optional[str] = None,
    cancel_cb: Optional[Callable[[], bool]] = None,
) -> Tuple[bytes, Dict[str, Any], bytes, Dict[str, Any]]:
    """
    Ideal pipeline API:
      table_bytes -> heatmap PNG bytes + exact display values

    Config: either provide keys at top-level, or under config["heatmap"].
    Supported keys (with defaults):
      table_mode: str = "auto"
      lower: float = -1e20
      upper: float =  1e16
      binarize: bool = True
      origin: str = "lower"
      cmap: str = "plasma"
      dpi: int = 180
      continuous: dict = overrides for continuous table rendering
      area: dict = overrides for explicitly selected area rendering

    Returns:
      (png_bytes, meta, value_bytes, value_meta)
    """
    _check_cancel(cancel_cb)
    cfg = config or {}
    heat_cfg = cfg.get("heatmap", cfg)
    requested_table_mode, resolved_table_mode = _resolve_table_mode(heat_cfg)

    lower = float(heat_cfg.get("lower", -1e20))
    upper = float(heat_cfg.get("upper", 1e16))
    dpi = heat_cfg.get("dpi", 180)
    dpi_val: Optional[int] = int(dpi) if dpi is not None else None

    _check_cancel(cancel_cb)
    df, load_meta = _load_table_bytes(table_bytes, filename_hint=filename_hint)
    _check_cancel(cancel_cb)

    # Convert to float, then apply the declared missing/invalid-value policy.
    data = df.to_numpy(dtype=float)
    _check_cancel(cancel_cb)
    finite_mask = np.isfinite(data)
    non_finite_count = int(np.size(data) - np.count_nonzero(finite_mask))
    nan_count = int(np.count_nonzero(np.isnan(data)))
    positive_infinity_count = int(np.count_nonzero(np.isposinf(data)))
    negative_infinity_count = int(np.count_nonzero(np.isneginf(data)))
    non_finite_policy = str(heat_cfg.get("non_finite_policy", "reject")).strip().lower()
    if non_finite_policy not in {"reject", "zero"}:
        raise ValueError("heatmap.non_finite_policy must be 'reject' or 'zero'")
    if non_finite_count and non_finite_policy == "reject":
        raise ValueError(
            "Input table contains "
            f"{non_finite_count} non-finite numeric cell(s) "
            f"(NaN={nan_count}, +Infinity={positive_infinity_count}, "
            f"-Infinity={negative_infinity_count}). Fix the input or explicitly set "
            "heatmap.non_finite_policy to 'zero'."
        )
    if non_finite_count:
        data = np.nan_to_num(data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    if resolved_table_mode in {"area", "continuous"}:
        mode_cfg = {}
        if isinstance(heat_cfg.get(resolved_table_mode), dict):
            mode_cfg = dict(heat_cfg[resolved_table_mode])
        filtered = data
        binarize = bool(mode_cfg.get("binarize", False))
        origin = str(mode_cfg.get("origin", heat_cfg.get("origin", "lower"))).strip().lower()
        cmap = str(mode_cfg.get("cmap", heat_cfg.get("cmap", "plasma")))
        vmin = _optional_float(mode_cfg.get("vmin", heat_cfg.get("vmin")))
        vmax = _optional_float(mode_cfg.get("vmax", heat_cfg.get("vmax")))
        if binarize:
            filtered = (filtered > 0).astype(int)
    else:
        binarize = bool(heat_cfg.get("binarize", True))
        origin = str(heat_cfg.get("origin", "lower")).strip().lower()
        cmap = str(heat_cfg.get("cmap", "plasma"))
        vmin = 0.0
        # Keep extremes and optionally binarize
        filtered = _keep_extremes_zero_middle(data, lower, upper)
        filtered = np.abs(filtered)
        if binarize:
            filtered = (filtered > 0).astype(int)
        vmax = float(np.max(filtered)) if filtered.size else 1.0

    if origin not in {"lower", "upper"}:
        raise ValueError("heatmap.origin must be 'lower' or 'upper'")

    _check_cancel(cancel_cb)
    nrows, ncols = filtered.shape
    if vmin is None:
        vmin = float(np.min(filtered)) if filtered.size else 0.0
    if vmax is None:
        vmax = float(np.max(filtered)) if filtered.size else 1.0
    z_min, z_max = _value_range(filtered)

    # Render a direct pixel-for-cell PNG so the output image dimensions match
    # the submitted table dimensions exactly.
    render = np.flipud(filtered) if origin == "lower" else filtered
    norm = np.zeros_like(render, dtype=np.float32)
    scale = float(vmax) - float(vmin)
    if scale > 0:
        norm = np.clip((render.astype(np.float32) - float(vmin)) / scale, 0.0, 1.0)
    _check_cancel(cancel_cb)

    cmap_fn = matplotlib.colormaps.get_cmap(cmap)
    rgba = np.asarray(cmap_fn(norm, bytes=True), dtype=np.uint8)
    out_img = Image.fromarray(rgba)
    _check_cancel(cancel_cb)

    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    png_bytes = buf.getvalue()
    _check_cancel(cancel_cb)

    meta: Dict[str, Any] = {
        **load_meta,
        "nrows": int(nrows),
        "ncols": int(ncols),
        "source_kind": "table",
        "input_representation": "numeric_matrix",
        "analysis_representation": "scalar_float32",
        "analysis_value_source": "processed_numeric_matrix",
        "analysis_image_normalization": "finite_min_max",
        "quantitative_information": "native_numeric",
        "lossy_analysis_input": bool(binarize or non_finite_count),
        "source_rows": int(nrows),
        "source_cols": int(ncols),
        "analysis_rows": int(nrows),
        "analysis_cols": int(ncols),
        "output_width": int(ncols),
        "output_height": int(nrows),
        "pixel_mapping": "table_cell",
        "coord_origin": "lower",
        "render_origin": origin,
        "coord_x_label": "col",
        "coord_y_label": "row",
        "table_mode": requested_table_mode,
        "resolved_table_mode": resolved_table_mode,
        "non_finite_policy": non_finite_policy,
        "non_finite_count": non_finite_count,
        "nan_count": nan_count,
        "positive_infinity_count": positive_infinity_count,
        "negative_infinity_count": negative_infinity_count,
        "non_finite_disposition": "zero_imputed" if non_finite_count else "none",
        "lower": lower,
        "upper": upper,
        "binarize": binarize,
        "origin": origin,
        "cmap": cmap,
        "dpi": dpi_val,
        "vmin": vmin,
        "vmax": vmax,
        "z_label": "z",
        "z_min": z_min,
        "z_max": z_max,
        "z_vmin": vmin,
        "z_vmax": vmax,
        "png_bytes": len(png_bytes),
    }
    _check_cancel(cancel_cb)
    value_bytes = encode_heatmap_values(filtered)
    _check_cancel(cancel_cb)
    value_meta: Dict[str, Any] = {
        "source_kind": "table",
        "input_representation": "numeric_matrix",
        "analysis_representation": "scalar_float32",
        "analysis_value_source": "processed_numeric_matrix",
        "analysis_image_normalization": "finite_min_max",
        "quantitative_information": "native_numeric",
        "lossy_analysis_input": bool(binarize or non_finite_count),
        "source_rows": int(nrows),
        "source_cols": int(ncols),
        "analysis_rows": int(nrows),
        "analysis_cols": int(ncols),
        "output_width": int(ncols),
        "output_height": int(nrows),
        "pixel_mapping": "table_cell",
        "coord_origin": "lower",
        "render_origin": origin,
        "coord_x_label": "col",
        "coord_y_label": "row",
        "table_mode": requested_table_mode,
        "resolved_table_mode": resolved_table_mode,
        "non_finite_policy": non_finite_policy,
        "non_finite_count": non_finite_count,
        "nan_count": nan_count,
        "positive_infinity_count": positive_infinity_count,
        "negative_infinity_count": negative_infinity_count,
        "non_finite_disposition": "zero_imputed" if non_finite_count else "none",
        "value_encoding": "float32_le",
        "value_dtype": "float32",
        "value_order": "row_major",
        "value_row_order": "top_to_bottom_source",
        "value_count": int(nrows * ncols),
        "value_nbytes": len(value_bytes),
        "z_label": "z",
        "z_min": z_min,
        "z_max": z_max,
        "z_vmin": vmin,
        "z_vmax": vmax,
    }
    return png_bytes, meta, value_bytes, value_meta


def table_to_heatmap_bytes(
    table_bytes: bytes,
    *,
    config: Optional[Dict[str, Any]] = None,
    filename_hint: Optional[str] = None,
    cancel_cb: Optional[Callable[[], bool]] = None,
) -> Tuple[bytes, Dict[str, Any]]:
    png_bytes, meta, _, _ = table_to_heatmap_payload(
        table_bytes,
        config=config,
        filename_hint=filename_hint,
        cancel_cb=cancel_cb,
    )
    return png_bytes, meta


def table_to_heatmap_file(
    table_bytes: bytes,
    *,
    out_path: str,
    config: Optional[Dict[str, Any]] = None,
    filename_hint: Optional[str] = None,
    cancel_cb: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """
    Optional compatibility helper:
    If any legacy code still wants a file on disk (scratch), this saves it
    and returns meta. Durable publishing should still be done via ArtifactStore.
    """
    png, meta = table_to_heatmap_bytes(
        table_bytes,
        config=config,
        filename_hint=filename_hint,
        cancel_cb=cancel_cb,
    )
    _check_cancel(cancel_cb)
    with open(out_path, "wb") as f:
        f.write(png)
    _check_cancel(cancel_cb)
    meta["out_path"] = out_path
    return meta
