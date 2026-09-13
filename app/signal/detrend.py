from dataclasses import dataclass
from typing import Any, Dict, Optional
import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class BaselineFitResult:
    model: Pipeline
    method: str
    fallback_used: bool
    fallback_reason: Optional[str]
    residual_threshold: float
    min_samples: int
    inlier_count: Optional[int]
    total_count: int

    def metadata(self) -> Dict[str, Any]:
        inlier_fraction = (
            float(self.inlier_count) / float(self.total_count)
            if self.inlier_count is not None and self.total_count > 0
            else None
        )
        return {
            "method": self.method,
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
            "residual_threshold": self.residual_threshold,
            "min_samples": self.min_samples,
            "inlier_count": self.inlier_count,
            "inlier_fraction": inlier_fraction,
            "total_count": self.total_count,
        }


@dataclass(frozen=True)
class DetrendResult:
    baseline: np.ndarray
    residual: np.ndarray
    fit: BaselineFitResult


def _mad(a: np.ndarray) -> float:
    med = np.median(a)
    return float(np.median(np.abs(a - med)))


def _make_ransac(estimator, **kwargs) -> RANSACRegressor:
    """
    Create a RANSACRegressor compatible with both old (base_estimator=)
    and new (estimator=) scikit-learn versions.
    """
    try:
        # Newer scikit-learn (>=1.1)
        return RANSACRegressor(estimator=estimator, **kwargs)
    except TypeError:
        # Older scikit-learn (<1.1)
        return RANSACRegressor(base_estimator=estimator, **kwargs)


def _fit_poly_baseline(x: np.ndarray, y: np.ndarray, degree: int) -> Pipeline:
    """
    Plain polynomial least-squares baseline (no RANSAC), used as fallback.
    """
    model = Pipeline([
        ("poly", PolynomialFeatures(degree, include_bias=False)),
        ("lin", LinearRegression()),
    ])
    model.fit(x.reshape(-1, 1), y)
    return model


def _fit_ransac_baseline(
    x: np.ndarray,
    y: np.ndarray,
    *,
    degree: int,
    min_samples: int,
    residual_threshold: float,
    random_state: int,
) -> Pipeline:
    ransac = _make_ransac(
        LinearRegression(),
        min_samples=min_samples,
        residual_threshold=residual_threshold,
        random_state=random_state,
    )
    model = Pipeline([
        ("poly", PolynomialFeatures(degree, include_bias=False)),
        ("ransac", ransac),
    ])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        model.fit(x.reshape(-1, 1), y)
    return model


def fit_baseline(
    x: np.ndarray,
    y: np.ndarray,
    degree: int = 1,
    min_samples: float | int = 0.5,
    residual_threshold: float | None = None,
    random_state: int = 42,
) -> BaselineFitResult:
    """Fit RANSAC and report the actual estimator, including any LSQ fallback."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) == 0:
        raise ValueError("Baseline inputs must be non-empty one-dimensional arrays of equal length")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Baseline inputs must contain only finite values")
    n = len(x)

    # Default residual threshold = MAD(y); fallback to 0.5*std if MAD==0
    if residual_threshold is None:
        rt = _mad(y)
        if not np.isfinite(rt) or rt == 0.0:
            s = float(np.std(y))
            rt = 0.5 * s if s > 0 else 1.0
        residual_threshold = float(rt)

    # Compute a safe integer min_samples
    if isinstance(min_samples, float):
        ms = int(np.ceil(min_samples * n))
    else:
        ms = int(min_samples)
    ms = max(ms, degree + 1, 2)       # at least 2 points, and enough for the polynomial
    ms = min(ms, n)                   # cannot exceed available samples

    try:
        model = _fit_ransac_baseline(
            x,
            y,
            degree=degree,
            min_samples=ms,
            residual_threshold=float(residual_threshold),
            random_state=random_state,
        )
        inlier_mask = getattr(model.named_steps["ransac"], "inlier_mask_", None)
        inlier_count = int(np.count_nonzero(inlier_mask)) if inlier_mask is not None else None
        return BaselineFitResult(
            model=model,
            method="ransac_polynomial",
            fallback_used=False,
            fallback_reason=None,
            residual_threshold=float(residual_threshold),
            min_samples=ms,
            inlier_count=inlier_count,
            total_count=n,
        )
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
        model = _fit_poly_baseline(x, y, degree)
        return BaselineFitResult(
            model=model,
            method="least_squares_polynomial",
            fallback_used=True,
            fallback_reason=f"{type(exc).__name__}: {exc}",
            residual_threshold=float(residual_threshold),
            min_samples=ms,
            inlier_count=None,
            total_count=n,
        )


def fit_baseline_ransac(
    x: np.ndarray,
    y: np.ndarray,
    degree: int = 1,
    min_samples: float | int = 0.5,
    residual_threshold: float | None = None,
    random_state: int = 42,
) -> Pipeline:
    """Compatibility wrapper returning the fitted estimator only."""

    return fit_baseline(
        x,
        y,
        degree=degree,
        min_samples=min_samples,
        residual_threshold=residual_threshold,
        random_state=random_state,
    ).model


def detrend_with_fit(
    x: np.ndarray,
    y: np.ndarray,
    degree: int = 1,
    **ransac_kwargs,
) -> DetrendResult:
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    fit = fit_baseline(x_values, y_values, degree=degree, **ransac_kwargs)
    baseline = fit.model.predict(x_values.reshape(-1, 1)).astype(float)
    return DetrendResult(baseline=baseline, residual=y_values - baseline, fit=fit)


def detrend_residual(
    x: np.ndarray,
    y: np.ndarray,
    degree: int = 1,
    **ransac_kwargs
) -> np.ndarray:
    """
    Remove baseline trend from y by fitting and subtracting a robust baseline.
    Falls back to LSQ poly if RANSAC fails.
    """
    return detrend_with_fit(x, y, degree=degree, **ransac_kwargs).residual


def detrend_dataframe(
    df: pd.DataFrame,
    x_col: str = 'frame',
    y_col: str = 'position',
    degree: int = 1,
    **ransac_kwargs
) -> pd.DataFrame:
    """
    Add a 'residual' column to df by subtracting a robust polynomial baseline.
    """
    df_out = df.copy()
    x = df_out[x_col].to_numpy()
    y = df_out[y_col].to_numpy()
    df_out['residual'] = detrend_residual(x, y, degree=degree, **ransac_kwargs)
    return df_out
