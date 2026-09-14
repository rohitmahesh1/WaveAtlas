# app/io/image_to_heatmap.py
from __future__ import annotations

import io
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from PIL import Image

from ..cancel import CancellationRequested
from ..heatmap_values import encode_heatmap_values


def _check_cancel(cancel_cb: Optional[Callable[[], bool]]) -> None:
    if cancel_cb is not None and cancel_cb():
        raise CancellationRequested("cancel_requested")


def _parse_hex_color(value: Optional[str]) -> Optional[np.ndarray]:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    if raw.startswith("#"):
        raw = raw[1:]
    if len(raw) == 3:
        raw = "".join(ch * 2 for ch in raw)
    if len(raw) != 6:
        raise ValueError(f"Expected a 6-digit hex color, got {value!r}")
    try:
        rgb = [int(raw[i : i + 2], 16) for i in (0, 2, 4)]
    except ValueError as exc:
        raise ValueError(f"Invalid hex color {value!r}") from exc
    return np.asarray(rgb, dtype=np.float32) / 255.0


def _composite_rgba(img: Image.Image, background_hex: str) -> Image.Image:
    rgba = img.convert("RGBA")
    bg_rgb = _parse_hex_color(background_hex)
    if bg_rgb is None:
        bg_rgb = np.zeros(3, dtype=np.float32)
    bg = Image.new("RGBA", rgba.size, tuple(int(round(float(c) * 255.0)) for c in bg_rgb) + (255,))
    return Image.alpha_composite(bg, rgba).convert("RGB")


def _rgb_to_luminance(rgb01: np.ndarray) -> np.ndarray:
    weights = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    return np.tensordot(rgb01, weights, axes=([-1], [0])).astype(np.float32)


def _rgb_to_hex_projection(rgb01: np.ndarray, low_rgb: np.ndarray, high_rgb: np.ndarray) -> np.ndarray:
    axis = high_rgb.astype(np.float32) - low_rgb.astype(np.float32)
    denom = float(np.dot(axis, axis))
    if denom <= 0:
        raise ValueError("image_input.low_hex and image_input.high_hex must be different colors")
    projected = np.tensordot(rgb01 - low_rgb, axis, axes=([-1], [0])) / denom
    return np.clip(projected, 0.0, 1.0).astype(np.float32)


def _input_representation(
    *,
    original_mode: str,
    grayscale: bool,
    binary_grayscale: bool,
    color_projection: bool,
) -> Tuple[str, str, bool]:
    if binary_grayscale:
        return "binary_image", "binary_scalar_raster", True
    if color_projection:
        return "rendered_color_image", "recovered_color_projection", True
    if original_mode in {"1", "L"} and grayscale:
        return "scalar_image", "native_scalar_raster", False
    if original_mode in {"I", "F", "I;16", "I;16B", "I;16L"} and grayscale:
        return "scalar_image", "scalar_raster_quantized_to_8_bit", True
    return "color_image", "derived_luminance", True


def image_to_heatmap_payload(
    image_bytes: bytes,
    *,
    config: Optional[Dict[str, Any]] = None,
    filename_hint: Optional[str] = None,
    cancel_cb: Optional[Callable[[], bool]] = None,
) -> Tuple[bytes, Dict[str, Any], bytes, Dict[str, Any]]:
    _check_cancel(cancel_cb)
    cfg = config or {}
    image_cfg = dict(cfg.get("image_input") or {})
    resize_fields = (
        "target_width",
        "target_height",
        "internal_width",
        "internal_height",
    )
    configured_resize = [
        field for field in resize_fields if image_cfg.get(field) not in (None, "")
    ]
    if configured_resize:
        names = ", ".join(f"image_input.{field}" for field in configured_resize)
        raise ValueError(
            "Quantitative image analysis requires the original image dimensions; "
            f"remove {names}."
        )

    grayscale = bool(image_cfg.get("grayscale", True))
    binary_grayscale = bool(image_cfg.get("binary_grayscale", False))
    binary_threshold = float(image_cfg.get("binary_threshold", 0.5))
    if not 0.0 <= binary_threshold <= 1.0:
        raise ValueError("image_input.binary_threshold must be between 0.0 and 1.0")

    invert = bool(image_cfg.get("invert", False))
    alpha_background = str(image_cfg.get("alpha_background", "#000000"))
    low_hex = image_cfg.get("low_hex")
    high_hex = image_cfg.get("high_hex")
    if (low_hex in (None, "")) != (high_hex in (None, "")):
        raise ValueError("image_input.low_hex and image_input.high_hex must be supplied together")
    if not grayscale and (low_hex not in (None, "") or high_hex not in (None, "")):
        raise ValueError("image_input.low_hex/high_hex require image_input.grayscale=true")
    if binary_grayscale and not grayscale:
        raise ValueError("image_input.binary_grayscale requires image_input.grayscale=true")
    origin = str(
        image_cfg.get("origin", (cfg.get("heatmap") or {}).get("origin", "lower"))
    ).strip().lower()
    if origin not in {"lower", "upper"}:
        raise ValueError("image coordinate origin must be 'lower' or 'upper'")

    _check_cancel(cancel_cb)
    with Image.open(io.BytesIO(image_bytes)) as img:
        original_mode = img.mode
        original_width, original_height = img.size
        rgb = _composite_rgba(img, alpha_background)
    _check_cancel(cancel_cb)

    output_width, output_height = original_width, original_height
    _check_cancel(cancel_cb)

    rgb01 = np.asarray(rgb, dtype=np.float32) / 255.0
    method = "rgb_passthrough"

    low_rgb = _parse_hex_color(low_hex)
    high_rgb = _parse_hex_color(high_hex)
    color_projection = bool(grayscale and low_rgb is not None and high_rgb is not None)
    if grayscale:
        if low_rgb is not None and high_rgb is not None:
            gray01 = _rgb_to_hex_projection(rgb01, low_rgb, high_rgb)
            method = "hex_projection"
        else:
            gray01 = _rgb_to_luminance(rgb01)
            method = "luminance"

        if invert:
            gray01 = 1.0 - gray01
            method = f"{method}_inverted"

        if binary_grayscale:
            gray01 = (gray01 >= binary_threshold).astype(np.float32)
            method = f"{method}_binary"

        out_img = Image.fromarray((gray01 * 255.0).astype(np.uint8))
    else:
        # Extraction still needs a declared scalar raster. This mirrors the
        # configured luminance interpretation while leaving the display RGB.
        gray01 = _rgb_to_luminance(rgb01)
        if invert:
            gray01 = 1.0 - gray01
        method = "luminance_from_rgb_display_inverted" if invert else "luminance_from_rgb_display"
        out_img = rgb
    _check_cancel(cancel_cb)

    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    png_bytes = buf.getvalue()
    _check_cancel(cancel_cb)

    input_representation, quantitative_information, lossy = _input_representation(
        original_mode=original_mode,
        grayscale=grayscale,
        binary_grayscale=binary_grayscale,
        color_projection=color_projection,
    )
    value_bytes = encode_heatmap_values(gray01)
    shared_meta: Dict[str, Any] = {
        "filename_hint": filename_hint,
        "format": "image",
        "original_mode": original_mode,
        "original_width": int(original_width),
        "original_height": int(original_height),
        "output_width": int(output_width),
        "output_height": int(output_height),
        "source_kind": "image",
        "input_representation": input_representation,
        "analysis_representation": "scalar_float32",
        "analysis_value_source": f"image_{method}",
        "analysis_image_normalization": "finite_min_max",
        "quantitative_information": quantitative_information,
        "lossy_analysis_input": lossy,
        "source_rows": int(original_height),
        "source_cols": int(original_width),
        "analysis_rows": int(output_height),
        "analysis_cols": int(output_width),
        "pixel_mapping": "source_pixel",
        "coord_origin": "lower",
        "source_origin": origin,
        "render_origin": "upper",
        "coord_x_label": "position",
        "coord_y_label": "frame",
        "grayscale": grayscale,
        "binary_grayscale": binary_grayscale,
        "binary_threshold": binary_threshold if binary_grayscale else None,
        "grayscale_method": method,
        "low_hex": str(low_hex) if low_hex else None,
        "high_hex": str(high_hex) if high_hex else None,
        "invert": invert,
        "alpha_background": alpha_background,
        "value_encoding": "float32_le",
        "value_dtype": "float32",
        "value_order": "row_major",
        "value_row_order": "top_to_bottom_image",
        "value_count": int(output_height * output_width),
        "value_nbytes": len(value_bytes),
        "z_label": "image intensity",
        "z_min": float(np.min(gray01)) if gray01.size else None,
        "z_max": float(np.max(gray01)) if gray01.size else None,
        "z_vmin": 0.0,
        "z_vmax": 1.0,
    }
    meta: Dict[str, Any] = {
        **shared_meta,
        "png_bytes": len(png_bytes),
    }
    value_meta = dict(shared_meta)
    return png_bytes, meta, value_bytes, value_meta


def image_to_heatmap_bytes(
    image_bytes: bytes,
    *,
    config: Optional[Dict[str, Any]] = None,
    filename_hint: Optional[str] = None,
    cancel_cb: Optional[Callable[[], bool]] = None,
) -> Tuple[bytes, Dict[str, Any]]:
    png_bytes, meta, _, _ = image_to_heatmap_payload(
        image_bytes,
        config=config,
        filename_hint=filename_hint,
        cancel_cb=cancel_cb,
    )
    return png_bytes, meta
