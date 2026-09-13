from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np
from scipy.fft import rfft, rfftfreq
import pandas as pd


@dataclass(frozen=True)
class FrameSamplingAssessment:
    valid: bool
    failure_reason: Optional[str]
    frame_count: int
    frame_span: Optional[float]
    missing_frame_count: int
    max_frame_gap: Optional[float]
    coverage_fraction: Optional[float]

    def metadata(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FrequencyEstimate:
    value: Optional[float]
    method: str
    valid: bool
    failure_reason: Optional[str]
    frame_sampling: FrameSamplingAssessment

    def metadata(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "method": self.method,
            "valid": self.valid,
            "failure_reason": self.failure_reason,
            "frame_sampling": self.frame_sampling.metadata(),
        }


def assess_frame_sampling(frame: np.ndarray) -> FrameSamplingAssessment:
    frames = np.asarray(frame, dtype=float).reshape(-1)
    count = int(frames.size)
    if count == 0:
        return FrameSamplingAssessment(
            valid=False,
            failure_reason="empty_track",
            frame_count=0,
            frame_span=None,
            missing_frame_count=0,
            max_frame_gap=None,
            coverage_fraction=None,
        )
    if not np.all(np.isfinite(frames)):
        return FrameSamplingAssessment(
            valid=False,
            failure_reason="non_finite_frame_coordinates",
            frame_count=count,
            frame_span=None,
            missing_frame_count=0,
            max_frame_gap=None,
            coverage_fraction=None,
        )
    if count < 2:
        return FrameSamplingAssessment(
            valid=False,
            failure_reason="insufficient_frames",
            frame_count=count,
            frame_span=0.0,
            missing_frame_count=0,
            max_frame_gap=None,
            coverage_fraction=1.0,
        )

    deltas = np.diff(frames)
    span = float(frames[-1] - frames[0])
    max_gap = float(np.max(deltas))
    if np.any(deltas <= 0):
        return FrameSamplingAssessment(
            valid=False,
            failure_reason="frames_not_strictly_increasing",
            frame_count=count,
            frame_span=span,
            missing_frame_count=0,
            max_frame_gap=max_gap,
            coverage_fraction=None,
        )

    rounded_deltas = np.rint(deltas)
    integer_spaced = bool(np.all(np.isclose(deltas, rounded_deltas, rtol=0.0, atol=1e-6)))
    missing = int(np.sum(np.maximum(rounded_deltas - 1.0, 0.0))) if integer_spaced else 0
    coverage = float(count / (count + missing)) if integer_spaced else None
    if not bool(np.all(np.isclose(deltas, 1.0, rtol=0.0, atol=1e-6))):
        reason = "missing_frames" if integer_spaced and missing > 0 else "non_unit_frame_spacing"
        return FrameSamplingAssessment(
            valid=False,
            failure_reason=reason,
            frame_count=count,
            frame_span=span,
            missing_frame_count=missing,
            max_frame_gap=max_gap,
            coverage_fraction=coverage,
        )
    return FrameSamplingAssessment(
        valid=True,
        failure_reason=None,
        frame_count=count,
        frame_span=span,
        missing_frame_count=0,
        max_frame_gap=max_gap,
        coverage_fraction=1.0,
    )


def estimate_valid_dominant_frequency(
    residual: np.ndarray,
    *,
    frame: np.ndarray,
    sampling_rate: float = 1.0,
    min_freq: float = None,
    max_freq: float = None,
) -> FrequencyEstimate:
    """Estimate Standard-mode frequency only when its FFT assumptions hold."""
    sampling = assess_frame_sampling(frame)
    if not sampling.valid:
        return FrequencyEstimate(None, "fft_uniform", False, sampling.failure_reason, sampling)

    try:
        resolved_sampling_rate = float(sampling_rate)
    except (TypeError, ValueError):
        resolved_sampling_rate = float("nan")
    if not np.isfinite(resolved_sampling_rate) or resolved_sampling_rate <= 0:
        return FrequencyEstimate(None, "fft_uniform", False, "invalid_sampling_rate", sampling)

    values = np.asarray(residual, dtype=float).reshape(-1)
    if values.size != sampling.frame_count:
        return FrequencyEstimate(None, "fft_uniform", False, "frame_signal_length_mismatch", sampling)
    if values.size < 4:
        return FrequencyEstimate(None, "fft_uniform", False, "insufficient_signal_samples", sampling)
    if not np.all(np.isfinite(values)):
        return FrequencyEstimate(None, "fft_uniform", False, "non_finite_signal", sampling)
    centered = values - float(np.mean(values))
    scale = max(1.0, float(np.max(np.abs(values))))
    if float(np.max(np.abs(centered))) <= np.finfo(float).eps * scale * 16.0:
        return FrequencyEstimate(None, "fft_uniform", False, "no_signal_variation", sampling)

    try:
        value = float(
            estimate_dominant_frequency(
                centered,
                sampling_rate=resolved_sampling_rate,
                min_freq=min_freq,
                max_freq=max_freq,
            )
        )
    except (TypeError, ValueError, FloatingPointError) as exc:
        return FrequencyEstimate(
            None,
            "fft_uniform",
            False,
            f"estimation_failed:{type(exc).__name__}",
            sampling,
        )
    if not np.isfinite(value) or value <= 0:
        return FrequencyEstimate(None, "fft_uniform", False, "non_positive_frequency", sampling)
    return FrequencyEstimate(value, "fft_uniform", True, None, sampling)


def estimate_dominant_frequency(
    residual: np.ndarray,
    sampling_rate: float = 1.0,
    min_freq: float = None,
    max_freq: float = None
) -> float:
    """
    Estimate the dominant frequency of a signal via FFT of the residual.

    Args:
        residual: 1D detrended data (numpy array).
        sampling_rate: samples per unit x (e.g., frames per second).
        min_freq: lower bound to consider (Hz).
        max_freq: upper bound to consider (Hz).

    Returns:
        freq: dominant frequency in same units as sampling_rate.
    """
    # zero-mean
    res = residual - np.mean(residual)
    n = len(res)

    # FFT
    yf = rfft(res)
    xf = rfftfreq(n, d=1.0 / sampling_rate)

    # magnitude spectrum
    mag = np.abs(yf)
    # ignore DC
    mag[0] = 0

    # apply frequency bounds mask
    mask = np.ones_like(xf, dtype=bool)
    if min_freq is not None:
        mask &= (xf >= min_freq)
    if max_freq is not None:
        mask &= (xf <= max_freq)

    # find peak in masked spectrum
    if not np.any(mask):
        raise ValueError("No frequencies in the specified range.")
    idx = np.argmax(mag * mask)
    return xf[idx]


def frequency_to_period(
    frequency: float
) -> float:
    """
    Convert frequency to period.

    Args:
        frequency: frequency value (Hz or cycles per frame).

    Returns:
        period: reciprocal of frequency.
    """
    if frequency == 0:
        return np.inf
    return 1.0 / frequency


def resolve_positive_frequency(
    frequency: float,
    *,
    frame: np.ndarray = None,
    sampling_rate: float = 1.0,
    min_freq: float = None,
    max_freq: float = None,
) -> float:
    """
    Return a usable positive frequency for downstream sine fitting.

    FFT estimation can legitimately fail or return 0 for flat/noisy tracks.
    In that case, fall back to one cycle across the visible track duration,
    then clamp to configured frequency bounds when present.
    """
    try:
        freq = float(frequency)
    except Exception:
        freq = float("nan")

    if not np.isfinite(freq) or freq <= 0:
        fallback = float("nan")
        if frame is not None:
            frames = np.asarray(frame, dtype=float)
            finite = frames[np.isfinite(frames)]
            if finite.size >= 2 and sampling_rate and sampling_rate > 0:
                frame_span = float(np.nanmax(finite) - np.nanmin(finite))
                if frame_span > 0:
                    fallback = float(sampling_rate) / frame_span
        if not np.isfinite(fallback) or fallback <= 0:
            try:
                min_freq_value = float(min_freq) if min_freq is not None else float("nan")
            except Exception:
                min_freq_value = float("nan")
            fallback = min_freq_value if np.isfinite(min_freq_value) and min_freq_value > 0 else 1.0
        freq = fallback

    try:
        min_freq_value = float(min_freq) if min_freq is not None else float("nan")
    except Exception:
        min_freq_value = float("nan")
    try:
        max_freq_value = float(max_freq) if max_freq is not None else float("nan")
    except Exception:
        max_freq_value = float("nan")
    if np.isfinite(min_freq_value) and min_freq_value > 0:
        freq = max(freq, min_freq_value)
    if np.isfinite(max_freq_value) and max_freq_value > 0:
        freq = min(freq, max_freq_value)
    return float(freq)


def estimate_period_from_residual(
    residual: np.ndarray,
    sampling_rate: float = 1.0,
    min_freq: float = None,
    max_freq: float = None
) -> float:
    """
    Combine dominant frequency estimation and period conversion.

    Args:
        residual: 1D detrended data.
        sampling_rate: samples per unit x.
        min_freq: lower freq bound.
        max_freq: upper freq bound.

    Returns:
        period: estimated period in x units.
    """
    freq = estimate_dominant_frequency(residual, sampling_rate, min_freq, max_freq)
    return frequency_to_period(freq)


def spectrum_dataframe(
    residual: np.ndarray,
    sampling_rate: float = 1.0
) -> pd.DataFrame:
    """
    Return the full FFT spectrum as a DataFrame for inspection or plotting.

    Args:
        residual: 1D detrended data.
        sampling_rate: samples per unit x.

    Returns:
        df_spectrum: DataFrame with columns ['frequency', 'magnitude'].
    """
    res = residual - np.mean(residual)
    n = len(res)
    yf = rfft(res)
    xf = rfftfreq(n, d=1.0 / sampling_rate)
    mag = np.abs(yf)
    df = pd.DataFrame({'frequency': xf, 'magnitude': mag})
    return df
