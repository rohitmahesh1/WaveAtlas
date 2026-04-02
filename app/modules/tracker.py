# tracker.py
from __future__ import annotations
import os, csv
from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional, Dict, Iterable

import numpy as np
import time
import cv2

# Only need the class type hint (and decision/preproc helper at runtime)
from .kymobutler_pt import KymoButlerPT

Coord = Tuple[int, int]  # (y, x)

# 8-connected neighborhood
_OFFSETS_8 = [(-1, -1), (-1, 0), (-1, 1),
              ( 0, -1),          ( 0, 1),
              ( 1, -1), ( 1, 0), ( 1, 1)]

# Direction -> index map for per-edge visitation
_DIR_TO_IDX = {d: i for i, d in enumerate(_OFFSETS_8)}

# ---------------------------
# Small pixel-graph utilities
# ---------------------------

def neighbors8(y: int, x: int, H: int, W: int) -> Iterable[Coord]:
    for dy, dx in _OFFSETS_8:
        ny, nx = y + dy, x + dx
        if 0 <= ny < H and 0 <= nx < W:
            yield ny, nx

def degree_at(skel: np.ndarray, y: int, x: int) -> int:
    H, W = skel.shape
    return sum(1 for ny, nx in neighbors8(y, x, H, W) if skel[ny, nx] == 1)

def find_endpoints_and_junctions(skel: np.ndarray) -> Tuple[List[Coord], List[Coord]]:
    endpoints, junctions = [], []
    ys, xs = np.where(skel == 1)
    for y, x in zip(ys, xs):
        d = degree_at(skel, y, x)
        if d == 1:
            endpoints.append((y, x))
        elif d >= 3:
            junctions.append((y, x))
    return endpoints, junctions

def crop_with_pad(arr: np.ndarray, cy: int, cx: int, hh: int, hw: int) -> np.ndarray:
    """Center crop with zero-padding if window runs off the borders."""
    H, W = arr.shape[:2]
    y0, y1 = cy - hh, cy + hh
    x0, x1 = cx - hw, cx + hw
    pad_top   = max(0, -y0)
    pad_left  = max(0, -x0)
    pad_bot   = max(0, y1 - H + 1)
    pad_right = max(0, x1 - W + 1)
    y0c, y1c = max(0, y0), min(H - 1, y1)
    x0c, x1c = max(0, x0), min(W - 1, x1)
    crop = arr[y0c:y1c + 1, x0c:x1c + 1]
    if pad_top or pad_bot or pad_left or pad_right:
        crop = np.pad(
            crop,
            ((pad_top, pad_bot), (pad_left, pad_right)),
            mode="constant",
            constant_values=0,
        )
    return crop

def path_mask(shape: Tuple[int, int], pts: List[Coord], radius: int = 0) -> np.ndarray:
    """Binary mask from a list of points; optional dilation radius to provide a corridor."""
    m = np.zeros(shape, dtype=np.uint8)
    if pts:
        ys, xs = zip(*pts)
        m[ys, xs] = 1
    if radius > 0:
        k = 2 * radius + 1
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return m

def enforce_one_point_per_row(track: List[Coord]) -> List[Coord]:
    """Collapse any multi-pixel rows to a single x (median for the first row, then nearest to previous)."""
    if not track:
        return track
    rows: Dict[int, List[int]] = {}
    for y, x in track:
        rows.setdefault(y, []).append(x)
    ys_sorted = sorted(rows.keys())
    cleaned: List[Coord] = []
    prev_x: Optional[int] = None
    for y in ys_sorted:
        xs = rows[y]
        if prev_x is None:
            x = int(np.median(xs))
        else:
            x = min(xs, key=lambda v: abs(v - prev_x))
        cleaned.append((y, x))
        prev_x = x
    return cleaned

def _norm01(img: np.ndarray) -> np.ndarray:
    f = img.astype(np.float32)
    if f.max() > 1:
        f /= 255.0
    return np.clip(f, 0.0, 1.0)


# ---------------------------
# Data class
# ---------------------------

@dataclass
class Track:
    points: List[Coord]
    id: int


# ---------------------------
# Crossing-aware tracker (edge-visited)
# ---------------------------

class CrossingTracker:
    def __init__(self, kb: KymoButlerPT,
                 decision_crop_hw: Tuple[int, int] = (48, 48),
                 max_branch_steps: int = 64,
                 min_track_len: int = 8,
                 decision_recent_tail: int = 12,
                 decision_thr: float = 0.0,
                 decision_topk: int = 8,          # NEW: top-K mean for branch scoring
                 seed_interior: bool = True,
                 max_iters: int = 50):
        """
        Crossing-aware tracker with edge-level visitation.
        Iteratively harvests tracks from the skeleton until leftovers dry up.
        """
        self.kb = kb
        self.ch, self.cw = decision_crop_hw
        self.max_branch_steps = int(max_branch_steps)
        self.min_track_len = int(min_track_len)
        self.decision_recent_tail = int(decision_recent_tail)
        self.decision_thr = float(decision_thr)
        self.decision_topk = max(1, int(decision_topk))
        self.seed_interior = bool(seed_interior)
        self.max_iters = int(max_iters)
        self._raw01_cache: Optional[np.ndarray] = None  # set per extract() call

    # -------- edge-visited helpers --------
    @staticmethod
    def _edge_is_used(visited_edge: np.ndarray, y: int, x: int, ny: int, nx: int) -> bool:
        dy, dx = ny - y, nx - x
        if (dy, dx) not in _DIR_TO_IDX:
            return True
        idx = _DIR_TO_IDX[(dy, dx)]
        return visited_edge[y, x, idx] == 1

    @staticmethod
    def _mark_edge(visited_edge: np.ndarray, y: int, x: int, ny: int, nx: int):
        dy, dx = ny - y, nx - x
        if (dy, dx) not in _DIR_TO_IDX:
            return
        i1 = _DIR_TO_IDX[(dy, dx)]
        i2 = _DIR_TO_IDX[(-dy, -dx)]
        visited_edge[y, x, i1] = 1
        visited_edge[ny, nx, i2] = 1

    # -------- decision scoring --------
    def _score_branches(self, raw01: np.ndarray, skel_all: np.ndarray,
                        curr_pts: List[Coord], junction: Coord,
                        cand_starts: List[Coord]) -> Optional[Coord]:
        """
        Score candidate branch starts using the decision map; return the best (or None).
        Gate: if a branch's max probability along its preview path is below decision_thr,
              it is rejected outright (WL-like confidence gate).
        Score: mean of top-K probabilities along the preview path (stable vs. length).
        """
        H, W = raw01.shape
        jy, jx = junction
        tail_pts = curr_pts[-self.decision_recent_tail:] if curr_pts else []
        curr_mask = path_mask((H, W), tail_pts, radius=0)
        hh, hw = self.ch // 2, self.cw // 2

        # one decision map per junction
        crop_raw  = crop_with_pad(raw01, jy, jx, hh, hw)
        crop_all  = crop_with_pad(skel_all, jy, jx, hh, hw)
        crop_curr = crop_with_pad(curr_mask, jy, jx, hh, hw)
        dmap = self.kb.decision_map(crop_raw, crop_all, crop_curr)  # [Hc,Wc] in 0..1

        best_start: Optional[Coord] = None
        best_score: float = -1.0

        for sy, sx in cand_starts:
            branch_pts = self._walk_branch_preview(skel_all, junction, (sy, sx), self.max_branch_steps)
            if not branch_pts:
                continue

            ps: List[float] = []
            for y, x in branch_pts:
                cy, cx = y - (jy - hh), x - (jx - hw)
                if 0 <= cy < dmap.shape[0] and 0 <= cx < dmap.shape[1]:
                    ps.append(float(dmap[cy, cx]))

            if not ps:
                continue

            pmax = max(ps)
            if pmax < self.decision_thr:
                # hard gate: this branch is not trusted
                continue

            # Stable score: mean of top-K probs along the preview
            k = min(self.decision_topk, len(ps))
            score = float(np.mean(sorted(ps, reverse=True)[:k]))

            if score > best_score:
                best_score = score
                best_start = (sy, sx)

        return best_start  # may be None -> triggers fallback straightness

    def _walk_branch_preview(self, skel: np.ndarray, prev: Coord,
                             start: Coord, max_steps: int) -> List[Coord]:
        """Preview a corridor from 'start' until junction/end or max_steps."""
        H, W = skel.shape
        path: List[Coord] = []
        prev_y, prev_x = prev
        y, x = start
        steps = 0
        while steps < max_steps and skel[y, x] == 1:
            path.append((y, x))
            steps += 1
            deg = degree_at(skel, y, x)
            nbrs = [(ny, nx) for ny, nx in neighbors8(y, x, H, W)
                    if skel[ny, nx] == 1 and (ny, nx) != (prev_y, prev_x)]
            if deg != 2 or not nbrs:
                break
            prev_y, prev_x = y, x
            y, x = nbrs[0]
        return path

    # -------- edge-aware walking & growing --------
    def _walk_one_dir(self, start: Coord, prev: Coord,
                      skel: np.ndarray,
                      visited_px: np.ndarray,
                      visited_edge: np.ndarray) -> List[Coord]:
        """
        Walk in one direction, marking edges as used at junctions, and pixels in corridors.
        Uses decision-map scoring at junctions.
        """
        assert self._raw01_cache is not None, "raw01 cache must be set before walking"
        raw01 = self._raw01_cache

        H, W = skel.shape
        y, x = start
        prev_y, prev_x = prev
        acc: List[Coord] = []

        while True:
            if skel[y, x] == 0:
                break

            deg = degree_at(skel, y, x)

            # Corridor pixels can be pixel-visited; junctions keep pixel free (edge-wise control)
            if deg <= 2:
                if visited_px[y, x] == 1:
                    break
                visited_px[y, x] = 1

            acc.append((y, x))

            # Candidate neighbors (excluding immediate back-step)
            nbrs = [(ny, nx) for ny, nx in neighbors8(y, x, H, W)
                    if skel[ny, nx] == 1 and (ny, nx) != (prev_y, prev_x)]

            # Prefer neighbors via UNUSED edges
            nbrs = [(ny, nx) for (ny, nx) in nbrs if not self._edge_is_used(visited_edge, y, x, ny, nx)]

            if not nbrs:
                break

            if deg == 1:
                ny, nx = nbrs[0]
            elif deg == 2:
                ny, nx = nbrs[0]
            else:
                chosen = self._score_branches(raw01, skel, acc, (y, x), nbrs)
                if chosen is None:
                    # fallback: pick the neighbor that best continues the incoming direction
                    vy, vx = y - prev_y, x - prev_x
                    def straightness(n):
                        dy, dx = n[0] - y, n[1] - x
                        return -(vy * dy + vx * dx)
                    ny, nx = min(nbrs, key=straightness)
                else:
                    ny, nx = chosen

            # Mark the edge we will traverse (both directions)
            self._mark_edge(visited_edge, y, x, ny, nx)

            prev_y, prev_x = y, x
            y, x = ny, nx

        return acc

    def _grow_from(self, raw01: np.ndarray, skel: np.ndarray, seed: Coord,
                   visited_px: np.ndarray, visited_edge: np.ndarray, bidir: bool) -> List[Coord]:
        """Grow a track from 'seed'. If bidir=True, grow forward and backward from the seed."""
        H, W = skel.shape
        y0, x0 = seed
        if skel[y0, x0] == 0:
            return []

        # neighbors from seed that have UNUSED edges
        nbrs0 = [(ny, nx) for ny, nx in neighbors8(y0, x0, H, W)
                 if skel[ny, nx] == 1 and not self._edge_is_used(visited_edge, y0, x0, ny, nx)]

        if not nbrs0:
            # allow single-pixel capture if corridor & not visited
            if degree_at(skel, y0, x0) <= 2 and visited_px[y0, x0] == 0:
                visited_px[y0, x0] = 1
                return [(y0, x0)]
            return []

        if bidir:
            fwd_start = nbrs0[0]
            bwd_start = nbrs0[1] if len(nbrs0) >= 2 else None
            fwd = self._walk_one_dir(fwd_start, prev=(y0, x0), skel=skel,
                                     visited_px=visited_px, visited_edge=visited_edge)
            bwd = self._walk_one_dir(bwd_start, prev=(y0, x0), skel=skel,
                                     visited_px=visited_px, visited_edge=visited_edge) if bwd_start else []
            pts = list(reversed(bwd)) + [(y0, x0)] + fwd
        else:
            pts = [(y0, x0)] + self._walk_one_dir(nbrs0[0], prev=(y0, x0), skel=skel,
                                                  visited_px=visited_px, visited_edge=visited_edge)

        return enforce_one_point_per_row(pts)

    @staticmethod
    def _subtract_tracks(src: np.ndarray, trks: List[Track]) -> np.ndarray:
        """Remove tracked pixels from skeleton for the next iteration."""
        if not trks:
            return src
        m = np.zeros_like(src, dtype=np.uint8)
        for t in trks:
            if t.points:
                ys, xs = zip(*t.points)
                m[ys, xs] = 1
        return (src & (~m)).astype(np.uint8)

    # -------- main extraction loop --------
    def extract_tracks(
        self,
        raw_gray: np.ndarray,
        skel: np.ndarray,
        *,
        progress_cb: Optional[Callable[[Dict[str, float | int]], None]] = None,
        progress_every_secs: float = 1.0,
    ) -> List[Track]:
        """
        Iteratively extract tracks:
          1) Endpoints pass (unidirectional).
          2) Interior seeding pass (bidirectional).
          3) Subtract explained pixels, repeat until dry or max_iters.
        """
        raw01 = _norm01(raw_gray)
        self._raw01_cache = raw01

        remaining = skel.copy().astype(np.uint8)
        total_px = int(np.count_nonzero(remaining))
        start_ts = time.monotonic()
        last_emit = start_ts
        last_rate_ts = start_ts
        last_processed_px = 0
        ema_rate_px: Optional[float] = None
        ema_alpha = 0.2

        def emit_progress(force: bool = False) -> None:
            nonlocal last_emit, last_rate_ts, last_processed_px, ema_rate_px
            if progress_cb is None:
                return
            now = time.monotonic()
            if not force and (now - last_emit) < float(progress_every_secs):
                return
            remaining_px = int(np.count_nonzero(remaining))
            processed_px = max(0, total_px - remaining_px)
            pct = (processed_px / total_px) if total_px > 0 else 1.0
            rate_elapsed = max(1e-6, now - last_rate_ts)
            delta = processed_px - last_processed_px
            inst_rate = delta / rate_elapsed if delta > 0 else 0.0
            if inst_rate > 0:
                ema_rate_px = inst_rate if ema_rate_px is None else (
                    ema_alpha * inst_rate + (1.0 - ema_alpha) * ema_rate_px
                )
            last_rate_ts = now
            last_processed_px = processed_px

            rate = ema_rate_px or 0.0
            eta_secs = (remaining_px / rate) if rate > 0 else 0.0
            try:
                progress_cb(
                    {
                        "remaining_px": remaining_px,
                        "total_px": total_px,
                        "processed_px": processed_px,
                        "pct": float(pct),
                        "eta_secs": float(eta_secs),
                        "tracks_found": len(tracks),
                        "iter": it,
                    }
                )
            except Exception:
                # Progress hooks should never break tracking.
                pass
            last_emit = now
        tracks: List[Track] = []
        tid = 0

        it = 0
        emit_progress(force=True)
        while it < self.max_iters:
            it += 1
            new_tracks: List[Track] = []

            # (re)initialize visitation for this pass
            visited_px = np.zeros_like(remaining, dtype=np.uint8)
            visited_edge = np.zeros((*remaining.shape, 8), dtype=np.uint8)

            # Pass 1: endpoints first (more stable)
            endpoints, _ = find_endpoints_and_junctions(remaining)
            for seed in endpoints:
                y, x = seed
                if remaining[y, x] == 0:
                    continue
                trk = self._grow_from(raw01, remaining, seed, visited_px, visited_edge, bidir=False)
                if len(trk) >= self.min_track_len:
                    new_tracks.append(Track(points=trk, id=tid)); tid += 1

            # Pass 2: interior seeding — any pixel that still has an UNUSED edge
            if self.seed_interior:
                H, W = remaining.shape
                ys, xs = np.where(remaining == 1)
                for y, x in zip(ys, xs):
                    # quick check: does (y,x) have any unused outgoing edge?
                    has_free = False
                    for ny, nx in neighbors8(y, x, H, W):
                        if remaining[ny, nx] == 1 and not self._edge_is_used(visited_edge, y, x, ny, nx):
                            has_free = True
                            break
                    if not has_free:
                        continue
                    trk = self._grow_from(raw01, remaining, (y, x), visited_px, visited_edge, bidir=True)
                    if len(trk) >= self.min_track_len:
                        new_tracks.append(Track(points=trk, id=tid)); tid += 1

            if not new_tracks:
                break  # nothing else to harvest

            # Accumulate and subtract for next iteration
            tracks.extend(new_tracks)
            remaining = self._subtract_tracks(remaining, new_tracks)
            emit_progress()

            # Done if nothing left
            if int(remaining.sum()) <= 0:
                break

        # clear cache
        emit_progress(force=True)
        self._raw01_cache = None
        return tracks

    # -------- utility I/O --------
    @staticmethod
    def save_tracks_csv(tracks: List[Track], out_csv: str):
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["track_id", "frame", "x"])
            for t in tracks:
                for (y, x) in t.points:
                    w.writerow([t.id, y, x])
