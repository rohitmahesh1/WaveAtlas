# app/modules/kb_adapter.py
from __future__ import annotations

from pathlib import Path
import time
from typing import Callable, Dict, List, Optional, Tuple, Union, Iterable

import cv2
import networkx as nx
import numpy as np
from skimage.filters import apply_hysteresis_threshold
from skimage.morphology import thin as _thin

from ..cancel import CancellationRequested
from .kymobutler_pt import get_kymobutler, filter_components, prob_to_mask, prune_endpoints
from .tracker import CrossingTracker, Track, enforce_one_point_per_row


def _check_cancel(cancel_cb: Optional[Callable[[], bool]]) -> None:
    if cancel_cb is not None and cancel_cb():
        raise CancellationRequested("cancel_requested")


# ---------------------------
# Morphology helpers
# ---------------------------

def _to_cv(mask01: np.ndarray) -> np.ndarray:
    return (mask01.astype(np.uint8) * 255)


def _from_cv(mask255: np.ndarray) -> np.ndarray:
    return (mask255 > 0).astype(np.uint8)


def morph_classic(mask01: np.ndarray, k: int = 3) -> np.ndarray:
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(k), int(k)))
    m = _to_cv(mask01)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, se, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, se, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, se, iterations=1)
    return _from_cv(m)


def morph_directional(mask01: np.ndarray, kv: int, kh: int, diag_bridge: bool) -> np.ndarray:
    m = _to_cv(mask01)
    v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(1, int(kv))))
    h = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, int(kh)), 1))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, v, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, h, iterations=1)
    if diag_bridge:
        d = np.ones((3, 3), np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, d, iterations=1)
    return _from_cv(m)


def weak_only_shave(mask01: np.ndarray, prob: np.ndarray, p_shave: float = 0.12) -> np.ndarray:
    weak = (prob < float(p_shave))
    m = mask01.astype(bool)
    sub = (m & weak).astype(np.uint8) * 255
    k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    sub = cv2.morphologyEx(sub, cv2.MORPH_OPEN, k1, iterations=1)
    sub = cv2.morphologyEx(sub, cv2.MORPH_OPEN, k2, iterations=1)
    m_weak = (sub > 0)
    m_strong = m & (~weak)
    return (m_weak | m_strong).astype(np.uint8)


def apply_morphology(
    mask01: np.ndarray,
    prob: np.ndarray,
    *,
    mode: str = "classic",  # classic | directional | none
    classic_kernel: int = 3,
    dir_kv: int = 4,
    dir_kh: int = 3,
    diag_bridge: bool = True,
    weak_shave_enable: bool = True,
    p_shave: float = 0.12,
) -> np.ndarray:
    mode = (mode or "classic").lower()
    if mode == "none":
        m = mask01
    elif mode == "classic":
        m = morph_classic(mask01, k=int(classic_kernel))
    else:
        m = morph_directional(mask01, kv=int(dir_kv), kh=int(dir_kh), diag_bridge=bool(diag_bridge))
        if weak_shave_enable:
            m = weak_only_shave(m, prob, p_shave=float(p_shave))
    return m


# ---------------------------
# Auto-threshold
# ---------------------------

def _auto_threshold(
    prob: np.ndarray,
    sweep: Tuple[float, float, int] = (0.12, 0.30, 19),
    target_mask_pct: Tuple[float, float] = (15.0, 25.0),
    cancel_cb: Optional[Callable[[], bool]] = None,
) -> float:
    lo, hi, n = float(sweep[0]), float(sweep[1]), int(sweep[2])
    thr_candidates = np.linspace(lo, hi, max(2, n))
    target_mid = 0.5 * (float(target_mask_pct[0]) + float(target_mask_pct[1]))
    best_thr, best_err = float(thr_candidates[0]), 1e18
    for t in thr_candidates:
        _check_cancel(cancel_cb)
        m = prob_to_mask(prob, thr=float(t))
        pct = float(m.mean()) * 100.0
        err = abs(pct - target_mid)
        if err < best_err:
            best_thr, best_err = float(t), err
    return float(best_thr)


# ---------------------------
# Track quality / dedupe
# ---------------------------

def _track_score(prob: np.ndarray, t: Track) -> float:
    if not t.points:
        return 0.0
    ys, xs = zip(*t.points)
    ys = np.asarray(ys, dtype=int)
    xs = np.asarray(xs, dtype=int)
    return float(np.median(prob[ys, xs]))


def _row_overlap(a_pts: List[Tuple[int, int]], b_pts: List[Tuple[int, int]]) -> float:
    ay = {y for y, _ in a_pts}
    by = {y for y, _ in b_pts}
    if not ay or not by:
        return 0.0
    inter = len(ay & by)
    return inter / float(min(len(ay), len(by)))


def _mean_dx_on_overlap(a_pts: List[Tuple[int, int]], b_pts: List[Tuple[int, int]]) -> float:
    from collections import defaultdict

    ax = defaultdict(list)
    bx = defaultdict(list)
    for y, x in a_pts:
        ax[y].append(x)
    for y, x in b_pts:
        bx[y].append(x)

    ys = sorted(set(ax) & set(bx))
    if not ys:
        return 1e9

    diffs: List[float] = []
    for y in ys:
        xa = min(ax[y], key=lambda v: abs(v - np.median(ax[y])))
        xb = min(bx[y], key=lambda v: abs(v - np.median(bx[y])))
        diffs.append(abs(xa - xb))
    return float(np.mean(diffs)) if diffs else 1e9


def _dedupe_row_index(pts: List[Tuple[int, int]]) -> Tuple[set[int], Dict[int, int]]:
    from collections import defaultdict

    by_row = defaultdict(list)
    for y, x in pts:
        by_row[y].append(x)

    row_x: Dict[int, int] = {}
    for y, xs in by_row.items():
        median_x = np.median(xs)
        row_x[y] = min(xs, key=lambda v: abs(v - median_x))
    return set(by_row.keys()), row_x


def _row_overlap_index(a_rows: set[int], b_rows: set[int]) -> float:
    if not a_rows or not b_rows:
        return 0.0
    inter = len(a_rows & b_rows)
    return inter / float(min(len(a_rows), len(b_rows)))


def _mean_dx_on_overlap_index(a_row_x: Dict[int, int], b_row_x: Dict[int, int]) -> float:
    ys = sorted(set(a_row_x) & set(b_row_x))
    if not ys:
        return 1e9
    diffs = [abs(a_row_x[y] - b_row_x[y]) for y in ys]
    return float(np.mean(diffs)) if diffs else 1e9


def filter_and_dedupe_tracks(
    tracks: List[Track],
    prob: np.ndarray,
    *,
    min_rows: int = 30,
    min_score: float = 0.11,
    overlap_iou: float = 0.80,
    dx_tol: float = 2.5,
    cancel_cb: Optional[Callable[[], bool]] = None,
) -> List[Track]:
    enriched = []
    for t in tracks:
        _check_cancel(cancel_cb)
        pts = sorted(t.points, key=lambda p: (p[0], p[1]))
        if len(pts) < min_rows:
            continue
        score = _track_score(prob, t)
        if score < min_score:
            continue
        row_set, row_x = _dedupe_row_index(pts)
        enriched.append((t, pts, score, len(pts), row_set, row_x))

    enriched.sort(key=lambda z: (z[2], z[3]), reverse=True)

    kept = []
    for t, pts, score, ln, row_set, row_x in enriched:
        _check_cancel(cancel_cb)
        dup = False
        for kt, kpts, kscore, kln, krow_set, krow_x in kept:
            _check_cancel(cancel_cb)
            if (
                _row_overlap_index(row_set, krow_set) >= overlap_iou
                and _mean_dx_on_overlap_index(row_x, krow_x) <= dx_tol
            ):
                dup = True
                break
        if not dup:
            kept.append((t, pts, score, ln, row_set, row_x))

    return [z[0] for z in kept]


# ---------------------------
# Skeleton cleanup + bridging
# ---------------------------

_OFFSETS_8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def _neighbors8(y: int, x: int, h: int, w: int):
    for dy, dx in _OFFSETS_8:
        ny, nx = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx < w:
            yield ny, nx


def _degree_map(skel: np.ndarray) -> np.ndarray:
    k = np.ones((3, 3), np.uint8)
    k[1, 1] = 0
    return cv2.filter2D(skel.astype(np.uint8), ddepth=cv2.CV_8U, kernel=k, borderType=cv2.BORDER_CONSTANT)


def _junction_nms(
    skel: np.ndarray,
    prob: np.ndarray,
    *,
    cancel_cb: Optional[Callable[[], bool]] = None,
) -> np.ndarray:
    h, w = skel.shape
    deg = _degree_map(skel)
    keep = skel.copy().astype(np.uint8)
    ys, xs = np.where((skel == 1) & (deg >= 3))
    for y, x in zip(ys, xs):
        _check_cancel(cancel_cb)
        p0 = prob[y, x]
        y0, y1 = max(0, y - 1), min(h, y + 2)
        x0, x1 = max(0, x - 1), min(w, x + 2)
        win = (skel[y0:y1, x0:x1] == 1) & (prob[y0:y1, x0:x1] > p0)
        if np.any(win):
            keep[y, x] = 0
    return keep


def _endpoints(
    skel: np.ndarray,
    *,
    cancel_cb: Optional[Callable[[], bool]] = None,
) -> List[Tuple[int, int]]:
    h, w = skel.shape
    out: List[Tuple[int, int]] = []
    for y, x in zip(*np.where(skel == 1)):
        _check_cancel(cancel_cb)
        deg = 0
        for ny, nx in _neighbors8(int(y), int(x), h, w):
            if skel[ny, nx] == 1:
                deg += 1
        if deg == 1:
            out.append((int(y), int(x)))
    return out


def _bresenham(y0: int, x0: int, y1: int, x1: int) -> List[Tuple[int, int]]:
    pts: List[Tuple[int, int]] = []
    dy = abs(y1 - y0)
    dx = abs(x1 - x0)
    sy = 1 if y0 < y1 else -1
    sx = 1 if x0 < x1 else -1
    err = dy - dx
    while True:
        pts.append((y0, x0))
        if y0 == y1 and x0 == x1:
            break
        e2 = 2 * err
        if e2 > -dx:
            err -= dx
            y0 += sy
        if e2 < dy:
            err += dy
            x0 += sx
    return pts


def _bridge_skeleton_gaps(
    skel: np.ndarray,
    prob: np.ndarray,
    *,
    max_gap_rows: int = 18,
    max_dx: int = 7,
    prob_min: float = 0.11,
    max_bridges: int = 2000,
    cancel_cb: Optional[Callable[[], bool]] = None,
) -> np.ndarray:
    h, w = skel.shape
    ends = _endpoints(skel, cancel_cb=cancel_cb)
    if not ends:
        return skel

    by_row: Dict[int, List[Tuple[int, int]]] = {}
    for y, x in ends:
        by_row.setdefault(y, []).append((y, x))

    bridges = 0
    out = skel.copy().astype(np.uint8)

    for y0, x0 in sorted(ends):
        _check_cancel(cancel_cb)
        for dy in range(1, max_gap_rows + 1):
            _check_cancel(cancel_cb)
            y1 = y0 + dy
            if y1 >= h or y1 not in by_row:
                break
            for yy, xx in by_row[y1]:
                _check_cancel(cancel_cb)
                if abs(xx - x0) > max_dx:
                    continue
                pts = _bresenham(y0, x0, yy, xx)
                yyv, xxv = zip(*pts)
                if out[yyv, xxv].mean() > 0.25:
                    continue
                pmean = float(prob[yyv, xxv].mean()) if pts else 0.0
                if pmean < prob_min:
                    continue
                out[yyv, xxv] = 1
                bridges += 1
                if bridges >= max_bridges:
                    break
            if bridges >= max_bridges:
                break
        if bridges >= max_bridges:
            break

    out = _thin(out.astype(bool)).astype(np.uint8)
    return out


# ---------------------------
# Track refinement (extend + merge)
# ---------------------------

def _extend_one_end(
    prob: np.ndarray,
    start_y: int,
    start_x: int,
    step: int,
    *,
    max_rows: int = 8,
    dx_win: int = 3,
    prob_min: float = 0.12,
    cancel_cb: Optional[Callable[[], bool]] = None,
) -> List[Tuple[int, int]]:
    h, w = prob.shape
    y, x = int(start_y), int(start_x)
    out: List[Tuple[int, int]] = []
    for _ in range(int(max_rows)):
        _check_cancel(cancel_cb)
        y2 = y + int(step)
        if not (0 <= y2 < h):
            break
        x0, x1 = max(0, x - int(dx_win)), min(w - 1, x + int(dx_win))
        row = prob[y2, x0:x1 + 1]
        if row.size == 0:
            break
        x2 = x0 + int(np.argmax(row))
        if float(prob[y2, x2]) < float(prob_min):
            break
        out.append((int(y2), int(x2)))
        y, x = int(y2), int(x2)
    return out


def _merge_pairwise(
    tracks: List[Track],
    prob: np.ndarray,
    *,
    max_gap_rows: int = 6,
    max_dx: int = 4,
    prob_bridge_min: float = 0.10,
    cancel_cb: Optional[Callable[[], bool]] = None,
) -> List[Track]:
    if not tracks:
        return []

    def start(t: Track) -> Tuple[int, int]:
        return t.points[0]

    def end(t: Track) -> Tuple[int, int]:
        return t.points[-1]

    used = [False] * len(tracks)
    merged: List[Track] = []
    order = sorted(range(len(tracks)), key=lambda i: start(tracks[i])[0])

    for i in order:
        _check_cancel(cancel_cb)
        if used[i]:
            continue
        ti = tracks[i]
        changed = True
        while changed:
            _check_cancel(cancel_cb)
            changed = False
            ey, ex = end(ti)
            for j in order:
                _check_cancel(cancel_cb)
                if used[j] or j == i:
                    continue
                sjy, sjx = start(tracks[j])
                gap = sjy - ey
                if 0 < gap <= max_gap_rows and abs(sjx - ex) <= max_dx:
                    n = max(1, gap)
                    ys = np.linspace(ey, sjy, n + 2, dtype=int)[1:-1]
                    xs = np.linspace(ex, sjx, n + 2, dtype=int)[1:-1]
                    bridge_p = float(prob[ys, xs].mean()) if len(ys) else 1.0
                    if bridge_p >= prob_bridge_min:
                        ti = type(ti)(points=ti.points + tracks[j].points, id=ti.id)
                        used[j] = True
                        changed = True
                        break
        merged.append(ti)
        used[i] = True

    return merged


def refine_tracks(
    tracks: List[Track],
    prob: np.ndarray,
    *,
    extend_rows: int = 10,
    dx_win: int = 3,
    prob_min: float = 0.12,
    max_gap_rows: int = 6,
    max_dx: int = 4,
    prob_bridge_min: float = 0.10,
    cancel_cb: Optional[Callable[[], bool]] = None,
) -> List[Track]:
    if not tracks:
        return []

    refined: List[Track] = []
    for t in tracks:
        _check_cancel(cancel_cb)
        if not t.points:
            continue
        pts = sorted(t.points, key=lambda p: (p[0], p[1]))
        hy, hx = pts[0]
        ty, tx = pts[-1]
        head_ext = _extend_one_end(
            prob,
            hy,
            hx,
            step=-1,
            max_rows=extend_rows,
            dx_win=dx_win,
            prob_min=prob_min,
            cancel_cb=cancel_cb,
        )
        tail_ext = _extend_one_end(
            prob,
            ty,
            tx,
            step=+1,
            max_rows=extend_rows,
            dx_win=dx_win,
            prob_min=prob_min,
            cancel_cb=cancel_cb,
        )
        pts = list(reversed(head_ext)) + pts + tail_ext
        refined.append(type(t)(points=pts, id=t.id))

    merged = _merge_pairwise(
        refined,
        prob,
        max_gap_rows=max_gap_rows,
        max_dx=max_dx,
        prob_bridge_min=prob_bridge_min,
        cancel_cb=cancel_cb,
    )
    _check_cancel(cancel_cb)
    return [type(t)(points=enforce_one_point_per_row(t.points), id=t.id) for t in merged]


def _track_slope(points: List[Tuple[int, int]], *, head: bool, fit_rows: int) -> float:
    if len(points) < 2:
        return 0.0
    pts = points[:fit_rows] if head else points[-fit_rows:]
    if len(pts) < 2:
        return 0.0

    ys = np.asarray([p[0] for p in pts], dtype=np.float64)
    xs = np.asarray([p[1] for p in pts], dtype=np.float64)
    y0 = ys - float(ys.mean())
    denom = float(np.dot(y0, y0))
    if denom <= 1e-9:
        return 0.0
    return float(np.dot(y0, xs - float(xs.mean())) / denom)


def _track_linearity_score(points: List[Tuple[int, int]]) -> float:
    if len(points) < 2:
        return 0.0
    ys = np.asarray([p[0] for p in points], dtype=np.float64)
    xs = np.asarray([p[1] for p in points], dtype=np.float64)
    y0 = ys - float(ys.mean())
    denom = float(np.dot(y0, y0))
    if denom <= 1e-9:
        return 0.0
    slope = float(np.dot(y0, xs - float(xs.mean())) / denom)
    intercept = float(xs.mean()) - slope * float(ys.mean())
    rmse = float(np.sqrt(np.mean((xs - (slope * ys + intercept)) ** 2)))
    return 1.0 / (1.0 + max(0.0, rmse))


def _row_line_points(y0: int, x0: int, y1: int, x1: int, *, include_ends: bool) -> List[Tuple[int, int]]:
    if y1 < y0:
        return []
    if y1 == y0:
        return [(int(y0), int(round(0.5 * (x0 + x1))))] if include_ends else []

    start = y0 if include_ends else y0 + 1
    stop = y1 + 1 if include_ends else y1
    pts: List[Tuple[int, int]] = []
    for y in range(int(start), int(stop)):
        frac = float(y - y0) / float(y1 - y0)
        x = int(round(float(x0) + frac * float(x1 - x0)))
        pts.append((int(y), x))
    return pts


def _mean_prob_at_points(prob: np.ndarray, points: List[Tuple[int, int]]) -> float:
    if not points:
        return 1.0
    h, w = prob.shape
    vals = [
        float(prob[y, min(w - 1, max(0, x))])
        for y, x in points
        if 0 <= y < h
    ]
    return float(np.mean(vals)) if vals else 0.0


def _track_row_x(points: List[Tuple[int, int]]) -> Dict[int, int]:
    return _dedupe_row_index(points)[1]


def _consensus_overlap_points(
    prob: np.ndarray,
    a_row_x: Dict[int, int],
    b_row_x: Dict[int, int],
    rows: List[int],
) -> List[Tuple[int, int]]:
    h, w = prob.shape
    out: List[Tuple[int, int]] = []
    for y in rows:
        ax = int(a_row_x[y])
        bx = int(b_row_x[y])
        if 0 <= y < h:
            ap = float(prob[y, min(w - 1, max(0, ax))])
            bp = float(prob[y, min(w - 1, max(0, bx))])
        else:
            ap = bp = 0.0

        if ap + bp > 1e-9:
            x = int(round((float(ax) * ap + float(bx) * bp) / (ap + bp)))
        else:
            x = int(round(0.5 * (float(ax) + float(bx))))
        out.append((int(y), x))
    return out


def _build_owner_index(tracks: List[Track]) -> Dict[Tuple[int, int], set[int]]:
    owners: Dict[Tuple[int, int], set[int]] = {}
    for idx, t in enumerate(tracks):
        for y, x in t.points:
            owners.setdefault((int(y), int(x)), set()).add(idx)
    return owners


def _bridge_conflict_fraction(
    bridge_points: List[Tuple[int, int]],
    owners: Dict[Tuple[int, int], set[int]],
    *,
    source_idx: int,
    target_idx: int,
    shape: Tuple[int, int],
    radius: int = 1,
) -> float:
    if not bridge_points:
        return 0.0

    h, w = shape
    conflicts = 0
    allowed = {int(source_idx), int(target_idx)}
    for y, x in bridge_points:
        found = False
        for yy in range(max(0, y - radius), min(h, y + radius + 1)):
            for xx in range(max(0, x - radius), min(w, x + radius + 1)):
                pix_owners = owners.get((int(yy), int(xx)))
                if pix_owners and (pix_owners - allowed):
                    found = True
                    break
            if found:
                break
        if found:
            conflicts += 1

    return float(conflicts) / float(len(bridge_points))


def link_track_endpoints(
    tracks: List[Track],
    prob: np.ndarray,
    *,
    max_gap_rows: int = 35,
    max_dx: float = 6.0,
    min_bridge_prob: float = 0.10,
    max_slope_delta: float = 0.45,
    fit_rows: int = 12,
    max_conflict_fraction: float = 0.15,
    insert_bridge_points: bool = True,
    overlap_enabled: bool = True,
    min_overlap_rows: int = 5,
    max_overlap_rows: int = 45,
    overlap_dx_tol: float = 3.0,
    prefer_long_linear: bool = False,
    length_weight: float = 0.25,
    linearity_weight: float = 0.25,
    min_abs_slope: float = 0.05,
    cancel_cb: Optional[Callable[[], bool]] = None,
) -> Tuple[List[Track], Dict[str, object]]:
    _check_cancel(cancel_cb)
    if not tracks:
        return [], {
            "input_tracks": 0,
            "candidate_links": 0,
            "candidate_gap_links": 0,
            "candidate_overlap_links": 0,
            "accepted_links": 0,
            "accepted_gap_links": 0,
            "accepted_overlap_links": 0,
            "output_tracks": 0,
        }

    norm_tracks: List[Track] = []
    for t in tracks:
        _check_cancel(cancel_cb)
        if t.points:
            norm_tracks.append(
                type(t)(points=enforce_one_point_per_row(sorted(t.points, key=lambda p: (p[0], p[1]))), id=t.id)
            )
    if not norm_tracks:
        return [], {
            "input_tracks": len(tracks),
            "candidate_links": 0,
            "candidate_gap_links": 0,
            "candidate_overlap_links": 0,
            "accepted_links": 0,
            "accepted_gap_links": 0,
            "accepted_overlap_links": 0,
            "output_tracks": 0,
        }

    starts_by_row: Dict[int, List[int]] = {}
    starts: List[Tuple[int, int]] = []
    ends: List[Tuple[int, int]] = []
    row_xs: List[Dict[int, int]] = []
    head_slopes: List[float] = []
    tail_slopes: List[float] = []
    full_slopes: List[float] = []
    linearity_scores: List[float] = []
    length_scores: List[float] = []
    max_track_length = max(len(t.points) for t in norm_tracks)
    for idx, t in enumerate(norm_tracks):
        _check_cancel(cancel_cb)
        starts.append(t.points[0])
        ends.append(t.points[-1])
        starts_by_row.setdefault(int(t.points[0][0]), []).append(idx)
        row_xs.append(_track_row_x(t.points))
        head_slopes.append(_track_slope(t.points, head=True, fit_rows=int(fit_rows)))
        tail_slopes.append(_track_slope(t.points, head=False, fit_rows=int(fit_rows)))
        full_slopes.append(_track_slope(t.points, head=True, fit_rows=len(t.points)))
        linearity_scores.append(_track_linearity_score(t.points))
        length_scores.append(float(len(t.points)) / float(max(1, max_track_length)))

    owners = _build_owner_index(norm_tracks)
    candidates: List[Tuple[float, int, int, str, List[Tuple[int, int]], Optional[int], Optional[int]]] = []
    max_gap = max(0, int(max_gap_rows))
    max_dx_f = max(0.0, float(max_dx))
    max_slope_delta_f = max(0.0, float(max_slope_delta))
    min_overlap = max(1, int(min_overlap_rows))
    max_overlap = max(min_overlap, int(max_overlap_rows))
    overlap_dx_tol_f = max(0.0, float(overlap_dx_tol))
    candidate_gap_links = 0
    candidate_overlap_links = 0

    for i, (ey, ex) in enumerate(ends):
        _check_cancel(cancel_cb)
        for gap in range(1, max_gap + 1):
            _check_cancel(cancel_cb)
            for j in starts_by_row.get(int(ey) + gap, []):
                _check_cancel(cancel_cb)
                if i == j:
                    continue

                sy, sx = starts[j]
                tail_slope = tail_slopes[i]
                head_slope = head_slopes[j]
                if prefer_long_linear:
                    if abs(full_slopes[i]) < float(min_abs_slope) or abs(full_slopes[j]) < float(min_abs_slope):
                        continue
                    if full_slopes[i] * full_slopes[j] <= 0:
                        continue
                slope_delta = abs(tail_slope - head_slope)
                if slope_delta > max_slope_delta_f:
                    continue

                pred_head_x = float(ex) + tail_slope * float(gap)
                pred_tail_x = float(sx) - head_slope * float(gap)
                projected_dx = max(abs(float(sx) - pred_head_x), abs(float(ex) - pred_tail_x))
                if projected_dx > max_dx_f:
                    continue

                score_line = _row_line_points(int(ey), int(ex), int(sy), int(sx), include_ends=True)
                bridge_prob = _mean_prob_at_points(prob, score_line)
                if bridge_prob < float(min_bridge_prob):
                    continue

                bridge_points = _row_line_points(int(ey), int(ex), int(sy), int(sx), include_ends=False)
                conflict_fraction = _bridge_conflict_fraction(
                    bridge_points,
                    owners,
                    source_idx=i,
                    target_idx=j,
                    shape=prob.shape,
                )
                if conflict_fraction > float(max_conflict_fraction):
                    continue

                score = (
                    bridge_prob
                    - 0.25 * (projected_dx / max(max_dx_f, 1e-6))
                    - 0.15 * (slope_delta / max(max_slope_delta_f, 1e-6))
                    - 0.10 * (float(gap) / max(float(max_gap), 1.0))
                    - 0.50 * conflict_fraction
                )
                if prefer_long_linear:
                    anchor_length = 0.75 * length_scores[i] + 0.25 * length_scores[j]
                    pair_linearity = 0.5 * (linearity_scores[i] + linearity_scores[j])
                    score += float(length_weight) * anchor_length + float(linearity_weight) * pair_linearity
                candidates.append((float(score), i, j, "gap", bridge_points, None, None))
                candidate_gap_links += 1

        if not overlap_enabled:
            continue

        iy0 = int(starts[i][0])
        for overlap_start_y in range(max(iy0 + 1, int(ey) - max_overlap + 1), int(ey) + 1):
            _check_cancel(cancel_cb)
            for j in starts_by_row.get(overlap_start_y, []):
                _check_cancel(cancel_cb)
                if i == j:
                    continue

                jy0, _ = starts[j]
                jy1, _ = ends[j]
                if int(jy0) <= iy0 or int(jy1) <= int(ey):
                    continue

                rows = sorted(y for y in set(row_xs[i]) & set(row_xs[j]) if int(jy0) <= y <= int(ey))
                if len(rows) < min_overlap or len(rows) > max_overlap:
                    continue

                diffs = [abs(row_xs[i][y] - row_xs[j][y]) for y in rows]
                mean_dx = float(np.mean(diffs)) if diffs else 1e9
                max_row_dx = float(max(diffs)) if diffs else 1e9
                if mean_dx > overlap_dx_tol_f or max_row_dx > 2.0 * overlap_dx_tol_f:
                    continue

                tail_slope = tail_slopes[i]
                head_slope = head_slopes[j]
                if prefer_long_linear:
                    if abs(full_slopes[i]) < float(min_abs_slope) or abs(full_slopes[j]) < float(min_abs_slope):
                        continue
                    if full_slopes[i] * full_slopes[j] <= 0:
                        continue
                slope_delta = abs(tail_slope - head_slope)
                if slope_delta > max_slope_delta_f:
                    continue

                consensus_points = _consensus_overlap_points(prob, row_xs[i], row_xs[j], rows)
                overlap_prob = _mean_prob_at_points(prob, consensus_points)
                if overlap_prob < float(min_bridge_prob):
                    continue

                conflict_fraction = _bridge_conflict_fraction(
                    consensus_points,
                    owners,
                    source_idx=i,
                    target_idx=j,
                    shape=prob.shape,
                )
                if conflict_fraction > float(max_conflict_fraction):
                    continue

                score = (
                    overlap_prob
                    + 0.20 * (float(len(rows)) / max(float(max_overlap), 1.0))
                    - 0.25 * (mean_dx / max(overlap_dx_tol_f, 1e-6))
                    - 0.15 * (slope_delta / max(max_slope_delta_f, 1e-6))
                    - 0.50 * conflict_fraction
                )
                if prefer_long_linear:
                    anchor_length = 0.75 * length_scores[i] + 0.25 * length_scores[j]
                    pair_linearity = 0.5 * (linearity_scores[i] + linearity_scores[j])
                    score += float(length_weight) * anchor_length + float(linearity_weight) * pair_linearity
                candidates.append((float(score), i, j, "overlap", consensus_points, int(rows[0]), int(rows[-1])))
                candidate_overlap_links += 1

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected_candidates = candidates
    if prefer_long_linear and candidates:
        candidate_by_pair = {(item[1], item[2]): item for item in candidates}
        graph = nx.Graph()
        for score, source, target, *_rest in candidates:
            graph.add_edge(("end", source), ("start", target), weight=float(score))
        matches = nx.max_weight_matching(graph, maxcardinality=False, weight="weight")
        matched_pairs: set[Tuple[int, int]] = set()
        for left, right in matches:
            if left[0] == "end":
                matched_pairs.add((int(left[1]), int(right[1])))
            else:
                matched_pairs.add((int(right[1]), int(left[1])))
        selected_candidates = sorted(
            (candidate_by_pair[pair] for pair in matched_pairs if pair in candidate_by_pair),
            key=lambda item: item[0],
            reverse=True,
        )
    linked_from: Dict[int, int] = {}
    linked_to: Dict[int, int] = {}
    links: Dict[Tuple[int, int], Tuple[str, List[Tuple[int, int]], Optional[int], Optional[int]]] = {}
    accepted_gap_links = 0
    accepted_overlap_links = 0
    for _, i, j, kind, connector_points, overlap_start, overlap_end in selected_candidates:
        _check_cancel(cancel_cb)
        if i in linked_from or j in linked_to:
            continue
        linked_from[i] = j
        linked_to[j] = i
        links[(i, j)] = (kind, connector_points, overlap_start, overlap_end)
        if kind == "overlap":
            accepted_overlap_links += 1
        else:
            accepted_gap_links += 1

    out: List[Track] = []
    seen: set[int] = set()
    chain_starts = [idx for idx in range(len(norm_tracks)) if idx not in linked_to]
    chain_starts.sort(key=lambda idx: (norm_tracks[idx].points[0][0], norm_tracks[idx].points[0][1]))

    for start_idx in chain_starts:
        _check_cancel(cancel_cb)
        if start_idx in seen:
            continue
        chain_points: List[Tuple[int, int]] = []
        cur = start_idx
        skip_until_y: Optional[int] = None
        while cur not in seen:
            _check_cancel(cancel_cb)
            seen.add(cur)
            cur_points = norm_tracks[cur].points
            if skip_until_y is not None:
                cur_points = [p for p in cur_points if int(p[0]) > int(skip_until_y)]

            if chain_points and cur_points and chain_points[-1] == cur_points[0]:
                chain_points.extend(cur_points[1:])
            else:
                chain_points.extend(cur_points)

            nxt = linked_from.get(cur)
            if nxt is None:
                break
            kind, connector_points, overlap_start, overlap_end = links.get((cur, nxt), ("gap", [], None, None))
            skip_until_y = None
            if kind == "overlap" and overlap_start is not None and overlap_end is not None:
                chain_points = [
                    p for p in chain_points
                    if not (int(overlap_start) <= int(p[0]) <= int(overlap_end))
                ]
                chain_points.extend(connector_points)
                skip_until_y = int(overlap_end)
            elif insert_bridge_points:
                chain_points.extend(connector_points)
            cur = nxt

        out.append(type(norm_tracks[start_idx])(points=enforce_one_point_per_row(chain_points), id=norm_tracks[start_idx].id))

    for idx, t in enumerate(norm_tracks):
        _check_cancel(cancel_cb)
        if idx not in seen:
            out.append(t)

    out.sort(key=lambda t: (t.points[0][0], t.points[0][1]) if t.points else (0, 0))
    return out, {
        "input_tracks": len(tracks),
        "candidate_links": len(candidates),
        "candidate_gap_links": candidate_gap_links,
        "candidate_overlap_links": candidate_overlap_links,
        "accepted_links": len(linked_from),
        "accepted_gap_links": accepted_gap_links,
        "accepted_overlap_links": accepted_overlap_links,
        "output_tracks": len(out),
    }


# ---------------------------
# Geometry + IO
# ---------------------------

def _track_len_rows(t: Track) -> int:
    if not t.points:
        return 0
    ys = [p[0] for p in t.points]
    return int(max(ys) - min(ys) + 1)


def _scale_tracks_to_original(
    tracks: List[Track],
    seg_hw: Tuple[int, int],
    orig_hw: Tuple[int, int],
    *,
    cancel_cb: Optional[Callable[[], bool]] = None,
) -> List[Track]:
    seg_h, seg_w = seg_hw
    h, w = orig_hw
    sy, sx = h / seg_h, w / seg_w
    out: List[Track] = []
    for t in tracks:
        _check_cancel(cancel_cb)
        # Center-aware scaling to reduce systematic pixel-center bias.
        pts = [
            (int(round((y + 0.5) * sy - 0.5)), int(round((x + 0.5) * sx - 0.5)))
            for (y, x) in t.points
        ]
        out.append(Track(points=pts, id=t.id))
    return out


def _save_npy_tracks(
    tracks: List[Track],
    out_dir: Path,
    *,
    min_length: int,
    cancel_cb: Optional[Callable[[], bool]] = None,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for i, t in enumerate(tracks):
        _check_cancel(cancel_cb)
        arr = np.asarray(t.points, dtype=float)
        if arr.shape[0] < int(min_length):
            continue
        np.save(out_dir / f"{i}.npy", arr)
        saved += 1
    return saved


# ---------------------------
# Runner
# ---------------------------

def run_kymobutler(
    heatmap_path: Union[str, Path],
    *,
    output_dir: Union[str, Path],
    export_dir: Optional[Union[str, Path]] = None,
    providers: Optional[Iterable[str]] = None,
    seg_size: int = 256,
    min_length: int = 30,
    verbose: bool = False,
    force_mode: Optional[str] = "bi",  # "uni" | "bi" | None
    thr: float = 0.20,
    thr_uni: Optional[float] = None,
    thr_bi: Optional[float] = None,
    auto_threshold: bool = True,
    auto_target_pct: Tuple[float, float] = (15.0, 25.0),
    auto_sweep: Tuple[float, float, int] = (0.12, 0.30, 19),
    auto_trigger_pct: Tuple[float, float] = (5.0, 35.0),
    hysteresis_enable: bool = True,
    hysteresis_low: float = 0.10,
    hysteresis_high: float = 0.20,
    morph_mode: str = "directional",  # classic | directional | none
    classic_kernel: int = 3,
    dir_kv: int = 5,
    dir_kh: int = 5,
    diag_bridge: bool = True,
    weak_shave_enable: bool = True,
    weak_shave_p: float = 0.12,
    comp_min_px: int = 10,
    comp_min_rows: int = 10,
    prune_iters: int = 2,
    fuse_uni_into_bi: bool = True,
    fuse_uni_weight: float = 0.7,
    skel_keep_ratio: float = 0.60,
    skel_keep_min_px: Optional[int] = None,
    skel_prob_floor_min: float = 0.06,
    skel_prob_floor_max: float = 0.10,
    decision_thr: float = 0.50,
    refine_enable: bool = True,
    extend_rows: int = 22,
    dx_win: int = 4,
    refine_prob_min: float = 0.11,
    max_gap_rows: int = 13,
    max_dx: int = 6,
    prob_bridge_min: float = 0.11,
    endpoint_link_enable: bool = False,
    endpoint_link_max_gap_rows: int = 35,
    endpoint_link_max_dx: float = 6.0,
    endpoint_link_min_bridge_prob: float = 0.10,
    endpoint_link_max_slope_delta: float = 0.45,
    endpoint_link_fit_rows: int = 12,
    endpoint_link_max_conflict_fraction: float = 0.15,
    endpoint_link_insert_bridge_points: bool = True,
    endpoint_link_overlap_enabled: bool = True,
    endpoint_link_min_overlap_rows: int = 5,
    endpoint_link_max_overlap_rows: int = 45,
    endpoint_link_overlap_dx_tol: float = 3.0,
    endpoint_link_prefer_long_linear: bool = False,
    endpoint_link_length_weight: float = 0.25,
    endpoint_link_linearity_weight: float = 0.25,
    endpoint_link_min_abs_slope: float = 0.05,
    dedupe_enable: bool = True,
    dedupe_min_rows: Optional[int] = None,
    dedupe_min_score: float = 0.11,
    dedupe_overlap_iou: float = 0.80,
    dedupe_dx_tol: float = 2.5,
    debug_save_images: bool = True,
    save_overlay_tracks: bool = True,
    progress_cb: Optional[Callable[[str, Dict[str, object]], None]] = None,
    cancel_cb: Optional[Callable[[], bool]] = None,
    **_: object,
) -> Path:
    """
    Compute tracks and overlay layers from a heatmap image.
    """
    last_cancel_check = 0.0

    def _cancelled_throttled() -> bool:
        nonlocal last_cancel_check
        if cancel_cb is None:
            return False
        now = time.monotonic()
        if (now - last_cancel_check) < 0.25:
            return False
        last_cancel_check = now
        return cancel_cb()

    def _check_cancel() -> None:
        if cancel_cb is not None and cancel_cb():
            raise CancellationRequested("cancel_requested")

    def _progress(stage: str, **data: object) -> None:
        _check_cancel()
        if progress_cb is None:
            return
        try:
            progress_cb(stage, data)
        except CancellationRequested:
            raise
        except Exception:
            # Don't fail the pipeline on progress hooks.
            return
        _check_cancel()

    heatmap_path = Path(heatmap_path)
    base_dir = Path(output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    out_dir = base_dir / "kymobutler_output"
    dbg_dir = base_dir / "debug"
    dbg_dir.mkdir(parents=True, exist_ok=True)

    _progress("load_image")
    gray_orig = cv2.imread(str(heatmap_path), cv2.IMREAD_GRAYSCALE)
    if gray_orig is None:
        raise FileNotFoundError(heatmap_path)
    h0, w0 = gray_orig.shape

    _progress("segmenting")
    kb = get_kymobutler(export_dir=export_dir, seg_size=int(seg_size), providers=providers)
    _check_cancel()

    cls = kb.classify(gray_orig, cancel_cb=_cancelled_throttled)
    _check_cancel()
    mode = "bi" if cls.get("label", 1) == 1 else "uni"
    if force_mode in {"uni", "bi"}:
        mode = force_mode

    t_uni = float(thr if thr_uni is None else thr_uni)
    t_bi = float(thr if thr_bi is None else thr_bi)

    if mode == "uni":
        out = kb.segment_uni_full(gray_orig, cancel_cb=_cancelled_throttled)
        _check_cancel()
        prob = np.maximum(out["ant"], out["ret"]).astype(np.float32)
        used_thr = t_uni
    else:
        prob_bi = kb.segment_bi_full(gray_orig, cancel_cb=_cancelled_throttled).astype(np.float32)
        _check_cancel()
        if fuse_uni_into_bi:
            outu = kb.segment_uni_full(gray_orig, cancel_cb=_cancelled_throttled)
            _check_cancel()
            prob_uni = np.maximum(outu["ant"], outu["ret"]).astype(np.float32)
            if prob_uni.shape != prob_bi.shape:
                prob_uni = cv2.resize(prob_uni, (prob_bi.shape[1], prob_bi.shape[0]), interpolation=cv2.INTER_LINEAR)
            prob = np.maximum(prob_bi, float(fuse_uni_weight) * prob_uni)
        else:
            prob = prob_bi
        used_thr = t_bi

    _progress("masking")
    mask0 = prob_to_mask(prob, thr=float(used_thr))
    hmask = None

    if hysteresis_enable:
        try:
            hmask = apply_hysteresis_threshold(prob.astype(np.float32), float(hysteresis_low), float(hysteresis_high))
            mask0 = hmask.astype(np.uint8)
        except Exception:
            hmask = None

    pct0 = float(mask0.mean()) * 100.0
    if auto_threshold and (pct0 < auto_trigger_pct[0] or pct0 > auto_trigger_pct[1]):
        used_thr = _auto_threshold(
            prob,
            sweep=auto_sweep,
            target_mask_pct=auto_target_pct,
            cancel_cb=_cancelled_throttled,
        )
        mask0 = prob_to_mask(prob, thr=float(used_thr))

    _check_cancel()
    mask = apply_morphology(
        mask0,
        prob,
        mode=str(morph_mode),
        classic_kernel=int(classic_kernel),
        dir_kv=int(dir_kv),
        dir_kh=int(dir_kh),
        diag_bridge=bool(diag_bridge),
        weak_shave_enable=bool(weak_shave_enable),
        p_shave=float(weak_shave_p),
    )
    _check_cancel()

    mask_f = filter_components(
        mask,
        min_px=int(comp_min_px),
        min_rows=int(comp_min_rows),
        cancel_cb=_cancelled_throttled,
    )
    _check_cancel()

    _progress("skeletonizing")
    skel_base = _thin(mask_f.astype(bool)).astype(np.uint8)
    _check_cancel()
    base_px = int(skel_base.sum())
    keep_floor = (
        max(2000, int(float(skel_keep_ratio) * max(1, base_px)))
        if skel_keep_min_px is None
        else int(skel_keep_min_px)
    )

    skel = skel_base.copy()
    if base_px > 0:
        vals = prob[skel == 1]
        p10 = float(np.percentile(vals, 10.0)) if vals.size else 0.08
        lo = float(skel_prob_floor_min)
        hi = float(skel_prob_floor_max)
        if hi < lo:
            hi, lo = lo, hi
        prob_floor = max(lo, min(hi, p10))

        deg = _degree_map(skel)
        corridor = (skel == 1) & (deg <= 2)
        skel[corridor & (prob < prob_floor)] = 0

        skel = _thin(skel.astype(bool)).astype(np.uint8)
        if int(skel.sum()) < keep_floor:
            skel = skel_base.copy()

        skel_nms = _junction_nms(skel, prob, cancel_cb=_cancelled_throttled)
        skel_nms = _thin(skel_nms.astype(bool)).astype(np.uint8)
        _check_cancel()
        if int(skel_nms.sum()) >= keep_floor:
            skel = skel_nms

    if int(prune_iters) > 0:
        skel = prune_endpoints(skel, iterations=int(prune_iters), cancel_cb=_cancelled_throttled)

    skel = _bridge_skeleton_gaps(
        skel,
        prob,
        max_gap_rows=int(max_gap_rows),
        max_dx=int(max_dx),
        prob_min=float(prob_bridge_min),
        cancel_cb=_cancelled_throttled,
    )
    _check_cancel()

    if debug_save_images:
        _check_cancel()
        cv2.imwrite(str(dbg_dir / "prob.png"), (prob * 255).astype(np.uint8))
        _check_cancel()
        cv2.imwrite(str(dbg_dir / "mask_raw.png"), (mask0 * 255))
        _check_cancel()
        cv2.imwrite(str(dbg_dir / "mask_clean.png"), (mask * 255))
        _check_cancel()
        cv2.imwrite(str(dbg_dir / "mask_filtered.png"), (mask_f * 255))
        _check_cancel()
        cv2.imwrite(str(dbg_dir / "skeleton.png"), (skel * 255))
        if hmask is not None:
            _check_cancel()
            cv2.imwrite(str(dbg_dir / "mask_hysteresis.png"), (hmask.astype(np.uint8) * 255))
        with open(dbg_dir / "stats.txt", "w") as f:
            f.write(f"prob_min={float(prob.min()):.6f} prob_max={float(prob.max()):.6f}\n")
            f.write(f"thr_used={float(used_thr):.6f}\n")
            f.write(f"mask_raw_pct={float(mask0.mean()) * 100.0:.2f}\n")
            f.write(f"mask_clean_pct={float(mask.mean()) * 100.0:.2f}\n")
            f.write(f"mask_filtered_pct={float(mask_f.mean()) * 100.0:.2f}\n")
            f.write(f"skel_px_base={base_px} skel_px_final={int(skel.sum())} keep_floor={keep_floor}\n")

    _progress("tracking")
    gray_seg = kb.preproc_for_seg(gray_orig, hw=prob.shape)
    _check_cancel()
    tracker = CrossingTracker(
        kb,
        max_branch_steps=256,
        min_track_len=max(5, int(min_length) // 3),
        decision_recent_tail=16,
        decision_thr=float(decision_thr),
    )
    def _tracking_progress(data: Dict[str, object]) -> None:
        _progress("tracking", **data)

    tracks_seg = tracker.extract_tracks(
        gray_seg,
        skel,
        progress_cb=_tracking_progress,
        progress_every_secs=1.0,
        cancel_cb=cancel_cb,
    )

    if refine_enable and tracks_seg:
        _check_cancel()
        _progress("refining")
        tracks_seg = refine_tracks(
            tracks_seg,
            prob,
            extend_rows=int(extend_rows),
            dx_win=int(dx_win),
            prob_min=float(refine_prob_min),
            max_gap_rows=int(max_gap_rows),
            max_dx=int(max_dx),
            prob_bridge_min=float(prob_bridge_min),
            cancel_cb=_cancelled_throttled,
        )

    endpoint_link_stats: Dict[str, object] = {}
    if endpoint_link_enable and tracks_seg:
        _check_cancel()
        _progress("endpoint_linking", before_tracks=len(tracks_seg))
        tracks_seg, endpoint_link_stats = link_track_endpoints(
            tracks_seg,
            prob,
            max_gap_rows=int(endpoint_link_max_gap_rows),
            max_dx=float(endpoint_link_max_dx),
            min_bridge_prob=float(endpoint_link_min_bridge_prob),
            max_slope_delta=float(endpoint_link_max_slope_delta),
            fit_rows=int(endpoint_link_fit_rows),
            max_conflict_fraction=float(endpoint_link_max_conflict_fraction),
            insert_bridge_points=bool(endpoint_link_insert_bridge_points),
            overlap_enabled=bool(endpoint_link_overlap_enabled),
            min_overlap_rows=int(endpoint_link_min_overlap_rows),
            max_overlap_rows=int(endpoint_link_max_overlap_rows),
            overlap_dx_tol=float(endpoint_link_overlap_dx_tol),
            prefer_long_linear=bool(endpoint_link_prefer_long_linear),
            length_weight=float(endpoint_link_length_weight),
            linearity_weight=float(endpoint_link_linearity_weight),
            min_abs_slope=float(endpoint_link_min_abs_slope),
            cancel_cb=_cancelled_throttled,
        )
        _progress("endpoint_linking_done", **endpoint_link_stats)

    if dedupe_enable and tracks_seg:
        _check_cancel()
        _progress("deduping")
        tracks_seg = filter_and_dedupe_tracks(
            tracks_seg,
            prob,
            min_rows=int(dedupe_min_rows if dedupe_min_rows is not None else min_length),
            min_score=float(dedupe_min_score),
            overlap_iou=float(dedupe_overlap_iou),
            dx_tol=float(dedupe_dx_tol),
            cancel_cb=_cancelled_throttled,
        )

    if debug_save_images and endpoint_link_stats:
        _check_cancel()
        with open(dbg_dir / "stats.txt", "a") as f:
            f.write(
                "endpoint_link "
                f"input_tracks={endpoint_link_stats.get('input_tracks', 0)} "
                f"candidate_links={endpoint_link_stats.get('candidate_links', 0)} "
                f"candidate_gap_links={endpoint_link_stats.get('candidate_gap_links', 0)} "
                f"candidate_overlap_links={endpoint_link_stats.get('candidate_overlap_links', 0)} "
                f"accepted_links={endpoint_link_stats.get('accepted_links', 0)} "
                f"accepted_gap_links={endpoint_link_stats.get('accepted_gap_links', 0)} "
                f"accepted_overlap_links={endpoint_link_stats.get('accepted_overlap_links', 0)} "
                f"output_tracks={endpoint_link_stats.get('output_tracks', 0)}\n"
            )

    _progress("scaling")
    tracks = _scale_tracks_to_original(
        tracks_seg,
        seg_hw=prob.shape,
        orig_hw=(h0, w0),
        cancel_cb=_cancelled_throttled,
    )
    _progress("saving")
    _save_npy_tracks(tracks, out_dir, min_length=int(min_length), cancel_cb=_cancelled_throttled)

    if save_overlay_tracks:
        _check_cancel()
        overlay = cv2.cvtColor(gray_orig, cv2.COLOR_GRAY2BGR)
        for t in tracks:
            _check_cancel()
            for y, x in t.points:
                cv2.circle(overlay, (int(x), int(y)), 1, (0, 255, 0), -1)
        _check_cancel()
        cv2.imwrite(str(base_dir / "overlay_tracks.png"), overlay)

    if verbose:
        lengths = [_track_len_rows(t) for t in tracks] if tracks else []
        if lengths:
            print(
                f"[kb_adapter] mode={mode} tracks={len(tracks)} "
                f"p50={int(np.percentile(lengths, 50))} p90={int(np.percentile(lengths, 90))} max={max(lengths)} "
                f"thr_used={float(used_thr):.4f}"
            )
        else:
            print(f"[kb_adapter] mode={mode} tracks=0 thr_used={float(used_thr):.4f}")

    return base_dir
