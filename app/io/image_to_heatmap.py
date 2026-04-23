# app/io/image_to_heatmap.py
from __future__ import annotations

import io
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image


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


def _resize_size(width: int, height: int, cfg: Dict[str, Any]) -> Tuple[int, int]:
    target_w = cfg.get("target_width", cfg.get("internal_width"))
    target_h = cfg.get("target_height", cfg.get("internal_height"))
    target_w = int(target_w) if target_w not in (None, "") else None
    target_h = int(target_h) if target_h not in (None, "") else None

    if target_w and target_w > 0 and target_h and target_h > 0:
        return target_w, target_h
    if target_w and target_w > 0:
        scale = target_w / max(1, width)
        return target_w, max(1, int(round(height * scale)))
    if target_h and target_h > 0:
        scale = target_h / max(1, height)
        return max(1, int(round(width * scale))), target_h
    return width, height


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


def image_to_heatmap_bytes(
    image_bytes: bytes,
    *,
    config: Optional[Dict[str, Any]] = None,
    filename_hint: Optional[str] = None,
) -> Tuple[bytes, Dict[str, Any]]:
    cfg = config or {}
    image_cfg = dict(cfg.get("image_input") or {})

    grayscale = bool(image_cfg.get("grayscale", True))
    binary_grayscale = bool(image_cfg.get("binary_grayscale", False))
    binary_threshold = float(image_cfg.get("binary_threshold", 0.5))
    if not 0.0 <= binary_threshold <= 1.0:
        raise ValueError("image_input.binary_threshold must be between 0.0 and 1.0")

    invert = bool(image_cfg.get("invert", False))
    alpha_background = str(image_cfg.get("alpha_background", "#000000"))
    low_hex = image_cfg.get("low_hex")
    high_hex = image_cfg.get("high_hex")

    with Image.open(io.BytesIO(image_bytes)) as img:
        original_mode = img.mode
        original_width, original_height = img.size
        rgb = _composite_rgba(img, alpha_background)

    output_width, output_height = _resize_size(original_width, original_height, image_cfg)
    if (output_width, output_height) != rgb.size:
        rgb = rgb.resize((output_width, output_height), Image.Resampling.BILINEAR)

    rgb01 = np.asarray(rgb, dtype=np.float32) / 255.0
    method = "rgb_passthrough"

    if grayscale:
        low_rgb = _parse_hex_color(low_hex)
        high_rgb = _parse_hex_color(high_hex)
        if low_rgb is not None and high_rgb is not None:
            gray01 = _rgb_to_hex_projection(rgb01, low_rgb, high_rgb)
            method = "hex_projection"
        else:
            gray01 = _rgb_to_luminance(rgb01)
            method = "luminance"

        if invert:
            gray01 = 1.0 - gray01

        if binary_grayscale:
            gray01 = (gray01 >= binary_threshold).astype(np.float32)
            method = f"{method}_binary"

        out_img = Image.fromarray((gray01 * 255.0).astype(np.uint8), mode="L")
    else:
        out_img = rgb

    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    meta: Dict[str, Any] = {
        "filename_hint": filename_hint,
        "format": "image",
        "original_mode": original_mode,
        "original_width": int(original_width),
        "original_height": int(original_height),
        "output_width": int(output_width),
        "output_height": int(output_height),
        "grayscale": grayscale,
        "binary_grayscale": binary_grayscale,
        "binary_threshold": binary_threshold if binary_grayscale else None,
        "grayscale_method": method,
        "low_hex": str(low_hex) if low_hex else None,
        "high_hex": str(high_hex) if high_hex else None,
        "invert": invert,
        "alpha_background": alpha_background,
        "png_bytes": len(png_bytes),
    }
    return png_bytes, meta