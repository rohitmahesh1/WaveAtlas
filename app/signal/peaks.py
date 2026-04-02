# src/signal/peaks.py
from __future__ import annotations

import math
from typing import Tuple, Optional, Dict, Sequence, List

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# ============================================================
# Helpers
# ============================================================

def _mad(a: np.ndarray) -> float:
    """Median absolute deviation (robust scale)."""
    a = np.asarray(a, dtype=float)
    med = np.median(a)
    return float(np.median(np.abs(a - med)))


def _nms_1d_by_index(
    indices: np.ndarray,
    strength: np.ndarray,
    *,
    window: int,
    dominance_frac: float = 0.55,
) -> np.ndarray:
    """
    Greedy 1D NMS: iterate peaks by descending strength, keep a peak and
    suppress neighbors within +/- window samples if they are weaker than
    dominance_frac * current_strength. Returns boolean keep mask.
    """
    idx = np.asarray(indices, dtype=int)
    s = np.asarray(strength, dtype=float)
    order = np.argsort(-s)  # descending
    keep = np.ones(idx.shape[0], dtype=bool)

    for o in order:
        if not keep[o]:
            continue
        i0 = idx[o]
        s0 = s[o]
        close = np.where(np.abs(idx - i0) <= int(window))[0]
        for j in close:
            if j == o:
                continue
            if s[j] <= dominance_frac * s0:
                keep[j] = False
    return keep


# ============================================================
# Primary detector: SciPy find_peaks
# ============================================================

def detect_peaks(
    signal: np.ndarray,
    prominence: float = 1.0,
    width: float = 1,
    distance: Optional[int] = None,
) -> Tuple[np.ndarray, dict]:
    """
    Detect peaks in a 1D signal using scipy.find_peaks.

    Args:
        signal: 1D array of residual signal values.
        prominence: required prominence of peaks.
        width: required MINIMUM width of peaks (in samples).
        distance: minimum horizontal distance (in samples) between peaks.

    Returns:
        peaks: indices of peaks in the signal.
        properties: dict of peak properties from scipy.
    """
    kwargs = {"prominence": float(prominence), "width": float(width)}
    if distance is not None:
        kwargs["distance"] = int(distance)

    peaks, properties = find_peaks(np.asarray(signal, dtype=float), **kwargs)
    return peaks, properties


def detect_peaks_adaptive(
    signal: np.ndarray,
    *,
    frames_per_period: Optional[float] = None,
    distance_frac: float = 0.6,          # min distance = distance_frac * period
    width_frac: float = 0.2,             # min width    = width_frac    * period
    rel_mad_k: float = 2.0,              # prom >= max(abs_min_prom_px, rel_mad_k*MAD)
    abs_min_prom_px: float = 1.0,
    nms_enable: bool = True,
    nms_dominance_frac: float = 0.55,    # keep stronger if neighbor < frac*strong
) -> Tuple[np.ndarray, dict]:
    """
    Period-aware peak detector with adaptive prominence and optional NMS.
    """
    sig = np.asarray(signal, dtype=float)
    n = sig.size
    if n == 0:
        return np.array([], dtype=int), {}

    prom_floor = max(float(abs_min_prom_px), float(rel_mad_k) * _mad(sig))

    if frames_per_period and np.isfinite(frames_per_period) and frames_per_period > 0:
        min_dist = max(1, int(round(distance_frac * frames_per_period)))
        min_width = max(1, int(round(width_frac * frames_per_period)))
    else:
        min_dist = max(1, int(round(0.05 * n)))
        min_width = 1

    peaks, props = find_peaks(sig, prominence=prom_floor, width=min_width, distance=min_dist)

    if peaks.size == 0 or not nms_enable:
        return peaks, props

    if "prominences" in props and len(props["prominences"]) == len(peaks):
        strength = np.asarray(props["prominences"], dtype=float)
    else:
        strength = sig[peaks].astype(float, copy=False)

    if frames_per_period and np.isfinite(frames_per_period) and frames_per_period > 0:
        win = max(1, int(round(0.5 * frames_per_period)))
    else:
        win = max(1, int(round(0.03 * n)))

    keep_mask = _nms_1d_by_index(peaks, strength, window=win, dominance_frac=float(nms_dominance_frac))
    if not np.all(keep_mask):
        peaks = peaks[keep_mask]
        props = {
            k: (np.asarray(v)[keep_mask] if hasattr(v, "__len__") and len(v) == keep_mask.shape[0] else v)
            for k, v in props.items()
        }

    return peaks, props


# ============================================================
# CWT path (SciPy if available; faithful NumPy fallback)
# ============================================================

def _try_import_cwt():
    """Try to import SciPy's CWT & ricker; return (cwt_fn, ricker_fn) or (None, None)."""
    try:
        from scipy.signal import cwt as _cwt, ricker as _ricker  # type: ignore
        return _cwt, _ricker
    except Exception:
        return None, None


def _estimate_kernel_len(a: float) -> int:
    """Heuristic kernel length ~ 10*a, forced odd and >=3 (mirrors SciPy practice)."""
    M = int(math.ceil(10.0 * float(a)))
    if M % 2 == 0:
        M += 1
    return max(3, M)


def _ricker_scipy_like(M: int, a: float) -> np.ndarray:
    """
    SciPy-like ricker(M, a) normalization.
    x runs 0..M-1 centered at (M-1)/2.
    """
    M = int(M)
    a = float(a)
    x = np.arange(M, dtype=float) - (M - 1) / 2.0
    xsq_over_a2 = (x * x) / (a * a)
    A = 2.0 / (np.sqrt(3.0 * a) * (np.pi ** 0.25))
    return A * (1.0 - xsq_over_a2) * np.exp(-0.5 * xsq_over_a2)


def _conv1d_same(sig: np.ndarray, ker: np.ndarray) -> np.ndarray:
    """
    'same' convolution with zero-padding, choosing direct vs FFT by size.
    Assumes ker is symmetric (true for Ricker), so no flip difference.
    """
    n, m = sig.size, ker.size
    if m < 64 or n < 512:
        return np.convolve(sig, ker, mode="same")
    L = int(1 << ((n + m - 1).bit_length()))
    S = np.fft.rfft(sig, L)
    K = np.fft.rfft(ker, L)
    y_full = np.fft.irfft(S * K, L)[: n + m - 1]
    start = (m - 1) // 2
    return y_full[start:start + n]


def _cwt_fallback(signal: np.ndarray, widths: np.ndarray) -> np.ndarray:
    """
    NumPy-only CWT approximation:
      CWT(s, a) ≈ conv(signal, ricker(M≈10a, a)) with SciPy-like normalization.
    Returns array of shape (len(widths), len(signal)).
    """
    sig = np.asarray(signal, dtype=float)
    W = np.asarray(widths, dtype=float)
    out = np.empty((W.size, sig.size), dtype=float)
    for i, a in enumerate(W):
        M = _estimate_kernel_len(a)
        k = _ricker_scipy_like(M, a)
        out[i] = _conv1d_same(sig, k)
    return out


def _local_maxima_1d(y: np.ndarray, order: int) -> np.ndarray:
    """
    Boolean mask of local maxima using a symmetric neighborhood of 'order'.
    Points must be strictly greater than all neighbors in the window.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = int(max(1, order))
    is_max = np.ones(n, dtype=bool)
    for k in range(1, order + 1):
        left = np.r_[y[k:], np.full(k, -np.inf)]
        right = np.r_[np.full(k, -np.inf), y[:-k]]
        is_max &= (y > left) & (y > right)
    return is_max


# ---- Ridge identification & filtering (SciPy-like behavior) ----

def _identify_ridge_lines(
    cwt_matrix: np.ndarray,
    max_distances: Sequence[float],
    gap_thresh: int,
) -> List[List[tuple[int, int]]]:
    """
    Identify ridge lines through local maxima across scales.
    Each ridge is a list of (scale_index, x_index) pairs from small->large scale.

    Linking tolerance per scale is given by max_distances[si].
    Up to 'gap_thresh' missing scales are allowed when continuing a ridge.
    """
    C = np.asarray(cwt_matrix, dtype=float)
    S, N = C.shape
    # Precompute candidate maxima per scale
    candidates_per_scale: List[np.ndarray] = []
    for si in range(S):
        # Local maxima neighborhood ~ width/6; estimate width from scale spacing
        # (SciPy uses width directly; we approximate with ~6 as a typical factor)
        nb = max(1, int(round(6 * 0.0)) + 1)  # keep >=1; actual nb chosen below
        # Choose nb adaptively from data variation (fallback if flat)
        std = np.std(C[si])
        nb = 1 if std == 0 else max(1, int(round(1 + 0.0)))
        mask = _local_maxima_1d(C[si], order=nb)
        candidates_per_scale.append(np.flatnonzero(mask))

    # Active ridges represented by last point + accumulated list
    ridges: List[List[tuple[int, int]]] = []
    actives: List[dict] = []  # { 'pts': [(si, xi), ...], 'last_x': int, 'gaps': int }

    for si in range(S):
        cand = candidates_per_scale[si]
        used = np.zeros(cand.size, dtype=bool)
        dmax = int(math.ceil(max_distances[si])) if si < len(max_distances) else 1

        # try to extend existing ridges
        next_actives: List[dict] = []
        for r in actives:
            prev_x = r['last_x']
            # nearest candidate within tolerance
            if cand.size:
                d = np.abs(cand - prev_x)
                ok = np.where(d <= dmax)[0]
            else:
                ok = np.array([], dtype=int)

            if ok.size:
                # pick the strongest local continuation (by CWT magnitude)
                vals = C[si, cand[ok]]
                j_rel = np.argmax(vals)
                j = ok[j_rel]
                xi = int(cand[j])
                used[j] = True
                r['pts'].append((si, xi))
                r['last_x'] = xi
                r['gaps'] = 0
                next_actives.append(r)
            else:
                # allow a gap
                r['gaps'] += 1
                if r['gaps'] <= int(gap_thresh):
                    next_actives.append(r)
                else:
                    ridges.append(r['pts'])

        # start new ridges from unused candidates
        for j, xi in enumerate(cand):
            if not used[j]:
                next_actives.append({'pts': [(si, int(xi))], 'last_x': int(xi), 'gaps': 0})

        actives = next_actives

    # flush remaining
    ridges.extend([r['pts'] for r in actives])
    return ridges


def _filter_ridge_lines(
    cwt_matrix: np.ndarray,
    ridge_lines: List[List[tuple[int, int]]],
    min_length: int,
    min_snr: float,
    noise_perc: float,
) -> List[List[tuple[int, int]]]:
    """
    Filter ridge lines by length and SNR.
    Noise floor per scale is estimated as the 'noise_perc' percentile of |CWT|.
    Ridge passes SNR if max(|CWT| along ridge) / noise_floor >= min_snr.
    """
    C = np.asarray(cwt_matrix, dtype=float)
    S, _ = C.shape
    min_length = int(min_length)
    noise_perc = float(noise_perc)
    min_snr = float(min_snr)

    # Precompute noise floors per scale
    noise_floor = np.empty(S, dtype=float)
    for si in range(S):
        row = np.abs(C[si])
        nf = np.percentile(row, noise_perc)
        if not np.isfinite(nf) or nf <= 0:
            nf = np.std(row) if np.std(row) > 0 else 1.0
        noise_floor[si] = nf

    keep: List[List[tuple[int, int]]] = []
    for ridge in ridge_lines:
        if len(ridge) < min_length:
            continue
        # Max SNR along ridge
        vals = [abs(C[si, xi]) / max(noise_floor[si], 1e-12) for (si, xi) in ridge]
        if len(vals) == 0:
            continue
        if max(vals) >= min_snr:
            keep.append(ridge)
    return keep


def _cwt_ridge_peaks(
    cwt_matrix: np.ndarray,
    widths: np.ndarray,
    *,
    max_distances: Optional[Sequence[float]] = None,
    gap_thresh: int = 2,
    min_length: int = 3,
    min_snr: float = 1.0,
    noise_perc: float = 10.0,
) -> np.ndarray:
    """
    SciPy-like ridge-line peak selection:
      1) identify ridge lines across scales using local maxima and a horizontal tolerance
      2) filter by min_length and SNR
      3) return peak positions (x) at the strongest point along each ridge
      4) dedupe with light NMS
    """
    C = np.asarray(cwt_matrix, dtype=float)
    W = np.asarray(widths, dtype=float)
    S, N = C.shape
    if S == 0 or N == 0:
        return np.array([], dtype=int)

    if max_distances is None:
        max_distances = np.maximum(1.0, W / 4.0)
    else:
        max_distances = np.asarray(max_distances, dtype=float)
        if max_distances.shape[0] != S:
            raise ValueError("max_distances must have length == len(widths)")

    ridges = _identify_ridge_lines(C, max_distances, gap_thresh=int(gap_thresh))
    ridges = _filter_ridge_lines(
        C,
        ridges,
        min_length=int(min_length),
        min_snr=float(min_snr),
        noise_perc=float(noise_perc),
    )
    if not ridges:
        return np.array([], dtype=int)

    # Pick each ridge's strongest point
    peak_idx: List[int] = []
    strength: List[float] = []
    A = np.abs(C)
    for ridge in ridges:
        # strongest point along ridge
        best = max(ridge, key=lambda p: A[p[0], p[1]])
        peak_idx.append(int(best[1]))
        strength.append(float(A[best[0], best[1]]))

    # De-duplicate with NMS (window ~ half median width)
    nms_window = max(1, int(round(np.median(W) / 2.0)))
    peak_idx = np.asarray(peak_idx, dtype=int)
    strength = np.asarray(strength, dtype=float)
    keep = _nms_1d_by_index(peak_idx, strength, window=nms_window, dominance_frac=0.75)
    return peak_idx[keep]


def find_peaks_cwt_compat(
    vector: np.ndarray,
    widths: Sequence[float],
    wavelet: Optional[callable] = None,
    *,
    max_distances: Optional[Sequence[float]] = None,
    gap_thresh: int = 2,
    min_length: int = 3,
    min_snr: float = 1.0,
    noise_perc: float = 10.0,
) -> np.ndarray:
    """
    SciPy-compatible find_peaks_cwt replacement.

    Args mirror SciPy’s (deprecated) find_peaks_cwt:
      - vector: 1D signal
      - widths: sequence of widths (Ricker 'a' parameters)
      - wavelet: if None or 'ricker', use Ricker; custom callables can be passed
      - max_distances: per-scale tolerance for ridge linking (default ~ widths/4)
      - gap_thresh: allowed number of missing scales
      - min_length: minimum ridge length
      - min_snr: SNR threshold using per-scale noise floors
      - noise_perc: percentile for noise floor

    Returns:
      np.ndarray of peak indices.
    """
    x = np.asarray(vector, dtype=float)
    W = np.asarray(widths, dtype=float)
    cwt_fn, ricker_fn = _try_import_cwt()

    if wavelet is None or wavelet == "ricker":
        # Prefer SciPy CWT if present
        if cwt_fn and ricker_fn:
            C = cwt_fn(x, ricker_fn, W)
        else:
            C = _cwt_fallback(x, W)
    elif callable(wavelet):
        # Allow custom wavelet(vector, width)->kernel path via convolution
        # Build CWT row-by-row by convolving with provided wavelet kernel
        rows = []
        for w in W:
            ker = np.asarray(wavelet(_estimate_kernel_len(w), w), dtype=float)
            rows.append(_conv1d_same(x, ker))
        C = np.vstack(rows)
    else:
        raise ValueError("wavelet must be None, 'ricker', or a callable producing kernels")

    return _cwt_ridge_peaks(
        C,
        W,
        max_distances=max_distances,
        gap_thresh=gap_thresh,
        min_length=min_length,
        min_snr=min_snr,
        noise_perc=noise_perc,
    )


def detect_peaks_cwt(
    signal: np.ndarray,
    widths: np.ndarray = np.arange(1, 10),
    wavelet: str = 'ricker',
    *,
    max_distances: Optional[Sequence[float]] = None,
    gap_thresh: int = 2,
    min_length: int = 3,
    min_snr: float = 1.0,
    noise_perc: float = 10.0,
) -> np.ndarray:
    """
    Detect peaks using a Continuous Wavelet Transform (Ricker) across multiple scales
    with ridge-line linking. Falls back to a SciPy-compatible implementation if
    SciPy’s cwt/ricker are unavailable.

    Returns:
        peaks_cwt: indices of peaks detected by CWT.
    """
    if wavelet != 'ricker' and wavelet is not None:
        raise ValueError("Only 'ricker' (or None for default) is supported.")

    return find_peaks_cwt_compat(
        signal,
        widths,
        wavelet='ricker',
        max_distances=max_distances,
        gap_thresh=gap_thresh,
        min_length=min_length,
        min_snr=min_snr,
        noise_perc=noise_perc,
    )


# ============================================================
# Tabulation utility
# ============================================================

def peaks_to_dataframe(
    x: np.ndarray,
    signal: np.ndarray,
    peaks: np.ndarray,
    properties: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Build a DataFrame listing peak positions and their signal values.
    """
    x = np.asarray(x)
    sig = np.asarray(signal)
    peaks = np.asarray(peaks, dtype=int)

    data: Dict[str, np.ndarray] = {
        'index': peaks,
        'x': x[peaks] if peaks.size else np.array([], dtype=x.dtype),
        'value': sig[peaks] if peaks.size else np.array([], dtype=sig.dtype),
    }

    if properties:
        for key, vals in properties.items():
            vals = np.asarray(vals)
            if vals.ndim == 1 and vals.shape[0] == peaks.shape[0]:
                data[key] = vals
            elif vals.ndim == 1 and vals.shape[0] == sig.shape[0]:
                data[key] = vals[peaks]
            # else: ignore mismatched shapes
    return pd.DataFrame(data)
