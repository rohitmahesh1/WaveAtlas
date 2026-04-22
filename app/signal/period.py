import numpy as np
from scipy.fft import rfft, rfftfreq
import pandas as pd


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
