# app/modules/kb_adapter.py
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union, Iterable

import cv2
import numpy as np
from skimage.filters import apply_hysteresis_threshold
from skimage.morphology import thin as _thin

from .kymobutler_pt import get_kymobutler, filter_components, prob_to_mask, prune_endpoints
from .tracker import CrossingTracker, Track, enforce_one_point_per_row


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
) -> float:
    lo, hi, n = float(sweep[0]), float(sweep[1]), int(sweep[2])
    thr_candidates = np.linspace(lo, hi, max(2, n))
    target_mid = 0.5 * (float(target_mask_pct[0]) + float(target_mask_pct[1]))
    best_thr, best_err = float(thr_candidates[0]), 1e18
    for t in thr_candidates:
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


def filter_and_dedupe_tracks(
    tracks: List[Track],
    prob: np.ndarray,
    *,
    min_rows: int = 30,
    min_score: float = 0.11,
    overlap_iou: float = 0.80,
    dx_tol: float = 2.5,
) -> List[Track]:
    enriched = []
    for t in tracks:
        pts = sorted(t.points, key=lambda p: (p[0], p[1]))
        if len(pts) < min_rows:
            continue
        score = _track_score(prob, t)
        if score < min_score:
            continue
        enriched.append((t, pts, score, len(pts)))

    enriched.sort(key=lambda z: (z[2], z[3]), reverse=True)

    kept = []
    for t, pts, score, ln in enriched:
        dup = False
        for kt, kpts, kscore, kln in kept:
            if _row_overlap(pts, kpts) >= overlap_iou and _mean_dx_on_overlap(pts, kpts) <= dx_tol:
                dup = True
                break
        if not dup:
            kept.append((t, pts, score, ln))

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


def _junction_nms(skel: np.ndarray, prob: np.ndarray) -> np.ndarray:
    h, w = skel.shape
    deg = _degree_map(skel)
    keep = skel.copy().astype(np.uint8)
    ys, xs = np.where((skel == 1) & (deg >= 3))
    for y, x in zip(ys, xs):
        p0 = prob[y, x]
        y0, y1 = max(0, y - 1), min(h, y + 2)
        x0, x1 = max(0, x - 1), min(w, x + 2)
        win = (skel[y0:y1, x0:x1] == 1) & (prob[y0:y1, x0:x1] > p0)
        if np.any(win):
            keep[y, x] = 0
    return keep


def _endpoints(skel: np.ndarray) -> List[Tuple[int, int]]:
    h, w = skel.shape
    out: List[Tuple[int, int]] = []
    for y, x in zip(*np.where(skel == 1)):
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
) -> np.ndarray:
    h, w = skel.shape
    ends = _endpoints(skel)
    if not ends:
        return skel

    by_row: Dict[int, List[Tuple[int, int]]] = {}
    for y, x in ends:
        by_row.setdefault(y, []).append((y, x))

    bridges = 0
    out = skel.copy().astype(np.uint8)

    for y0, x0 in sorted(ends):
        for dy in range(1, max_gap_rows + 1):
            y1 = y0 + dy
            if y1 >= h or y1 not in by_row:
                break
            for yy, xx in by_row[y1]:
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
) -> List[Tuple[int, int]]:
    h, w = prob.shape
    y, x = int(start_y), int(start_x)
    out: List[Tuple[int, int]] = []
    for _ in range(int(max_rows)):
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
        if used[i]:
            continue
        ti = tracks[i]
        changed = True
        while changed:
            changed = False
            ey, ex = end(ti)
            for j in order:
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
) -> List[Track]:
    if not tracks:
        return []

    refined: List[Track] = []
    for t in tracks:
        if not t.points:
            continue
        pts = sorted(t.points, key=lambda p: (p[0], p[1]))
        hy, hx = pts[0]
        ty, tx = pts[-1]
        head_ext = _extend_one_end(prob, hy, hx, step=-1, max_rows=extend_rows, dx_win=dx_win, prob_min=prob_min)
        tail_ext = _extend_one_end(prob, ty, tx, step=+1, max_rows=extend_rows, dx_win=dx_win, prob_min=prob_min)
        pts = list(reversed(head_ext)) + pts + tail_ext
        refined.append(type(t)(points=pts, id=t.id))

    merged = _merge_pairwise(
        refined,
        prob,
        max_gap_rows=max_gap_rows,
        max_dx=max_dx,
        prob_bridge_min=prob_bridge_min,
    )
    return [type(t)(points=enforce_one_point_per_row(t.points), id=t.id) for t in merged]


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
) -> List[Track]:
    seg_h, seg_w = seg_hw
    h, w = orig_hw
    sy, sx = h / seg_h, w / seg_w
    out: List[Track] = []
    for t in tracks:
        # Center-aware scaling to reduce systematic pixel-center bias.
        pts = [
            (int(round((y + 0.5) * sy - 0.5)), int(round((x + 0.5) * sx - 0.5)))
            for (y, x) in t.points
        ]
        out.append(Track(points=pts, id=t.id))
    return out


def _save_npy_tracks(tracks: List[Track], out_dir: Path, *, min_length: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for i, t in enumerate(tracks):
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
    dedupe_enable: bool = True,
    dedupe_min_rows: Optional[int] = None,
    dedupe_min_score: float = 0.11,
    dedupe_overlap_iou: float = 0.80,
    dedupe_dx_tol: float = 2.5,
    debug_save_images: bool = True,
    save_overlay_tracks: bool = True,
    progress_cb: Optional[Callable[[str, Dict[str, object]], None]] = None,
    **_: object,
) -> Path:
    """
    Compute tracks and overlay layers from a heatmap image.
    """
    def _progress(stage: str, **data: object) -> None:
        if progress_cb is None:
            return
        try:
            progress_cb(stage, data)
        except Exception:
            # Don't fail the pipeline on progress hooks.
            return

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

    cls = kb.classify(gray_orig)
    mode = "bi" if cls.get("label", 1) == 1 else "uni"
    if force_mode in {"uni", "bi"}:
        mode = force_mode

    t_uni = float(thr if thr_uni is None else thr_uni)
    t_bi = float(thr if thr_bi is None else thr_bi)

    if mode == "uni":
        out = kb.segment_uni_full(gray_orig)
        prob = np.maximum(out["ant"], out["ret"]).astype(np.float32)
        used_thr = t_uni
    else:
        prob_bi = kb.segment_bi_full(gray_orig).astype(np.float32)
        if fuse_uni_into_bi:
            outu = kb.segment_uni_full(gray_orig)
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
        used_thr = _auto_threshold(prob, sweep=auto_sweep, target_mask_pct=auto_target_pct)
        mask0 = prob_to_mask(prob, thr=float(used_thr))

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

    mask_f = filter_components(mask, min_px=int(comp_min_px), min_rows=int(comp_min_rows))

    _progress("skeletonizing")
    skel_base = _thin(mask_f.astype(bool)).astype(np.uint8)
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

        skel_nms = _junction_nms(skel, prob)
        skel_nms = _thin(skel_nms.astype(bool)).astype(np.uint8)
        if int(skel_nms.sum()) >= keep_floor:
            skel = skel_nms

    if int(prune_iters) > 0:
        skel = prune_endpoints(skel, iterations=int(prune_iters))

    skel = _bridge_skeleton_gaps(
        skel,
        prob,
        max_gap_rows=int(max_gap_rows),
        max_dx=int(max_dx),
        prob_min=float(prob_bridge_min),
    )

    if debug_save_images:
        cv2.imwrite(str(dbg_dir / "prob.png"), (prob * 255).astype(np.uint8))
        cv2.imwrite(str(dbg_dir / "mask_raw.png"), (mask0 * 255))
        cv2.imwrite(str(dbg_dir / "mask_clean.png"), (mask * 255))
        cv2.imwrite(str(dbg_dir / "mask_filtered.png"), (mask_f * 255))
        cv2.imwrite(str(dbg_dir / "skeleton.png"), (skel * 255))
        if hmask is not None:
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
    )

    if refine_enable and tracks_seg:
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
        )

    if dedupe_enable and tracks_seg:
        _progress("deduping")
        tracks_seg = filter_and_dedupe_tracks(
            tracks_seg,
            prob,
            min_rows=int(dedupe_min_rows if dedupe_min_rows is not None else min_length),
            min_score=float(dedupe_min_score),
            overlap_iou=float(dedupe_overlap_iou),
            dx_tol=float(dedupe_dx_tol),
        )

    _progress("scaling")
    tracks = _scale_tracks_to_original(tracks_seg, seg_hw=prob.shape, orig_hw=(h0, w0))
    _progress("saving")
    _save_npy_tracks(tracks, out_dir, min_length=int(min_length))

    if save_overlay_tracks:
        overlay = cv2.cvtColor(gray_orig, cv2.COLOR_GRAY2BGR)
        for t in tracks:
            for y, x in t.points:
                cv2.circle(overlay, (int(x), int(y)), 1, (0, 255, 0), -1)
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
