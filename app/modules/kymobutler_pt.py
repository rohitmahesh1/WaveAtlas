# app/modules/kymobutler_pt.py
from __future__ import annotations

import os
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np
import onnx
import onnxruntime as ort

from skimage.measure import label, regionprops
from skimage.morphology import skeletonize as _skel
from skimage.morphology import thin as _thin


REQUIRED_ONNX = ("uni_seg.onnx", "bi_seg.onnx", "classifier.onnx", "decision.onnx")


# ---------------------------
# Export dir + caching
# ---------------------------

_KB_CACHE_LOCK = threading.Lock()
_KB_CACHE: dict[tuple, "KymoButlerPT"] = {}


def resolve_export_dir(export_dir: Optional[str | Path]) -> Path:
    """
    Returns a directory containing REQUIRED_ONNX.
    """
    if export_dir is None:
        env = os.getenv("KYMO_EXPORT_DIR")
        if not env:
            raise FileNotFoundError(
                "ONNX export directory not provided. Pass export_dir=... or set KYMO_EXPORT_DIR."
            )
        export_dir = env

    p = Path(export_dir).expanduser().resolve()
    if not p.is_dir():
        raise FileNotFoundError(f"export_dir is not a directory: {p}")

    missing = [name for name in REQUIRED_ONNX if not (p / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing ONNX files {missing} in export_dir: {p}")

    return p


def get_kymobutler(
    *,
    export_dir: Optional[str | Path],
    seg_size: int = 256,
    tile_stride: int = 128,
    tile_round_to: int = 16,
    use_tiling: bool = True,
    providers: Optional[Iterable[str]] = None,
) -> "KymoButlerPT":
    """
    Returns a cached KymoButlerPT instance for the current process.
    """
    p = resolve_export_dir(export_dir)
    providers_key = tuple(providers) if providers is not None else ("CPUExecutionProvider",)
    key = (str(p), int(seg_size), int(tile_stride), int(tile_round_to), bool(use_tiling), providers_key)

    with _KB_CACHE_LOCK:
        kb = _KB_CACHE.get(key)
        if kb is not None:
            return kb
        kb = KymoButlerPT(
            export_dir=p,
            seg_size=int(seg_size),
            tile_stride=int(tile_stride),
            tile_round_to=int(tile_round_to),
            use_tiling=bool(use_tiling),
            providers=list(providers_key),
        )
        _KB_CACHE[key] = kb
        return kb


# ---------------------------
# Basic image helpers
# ---------------------------

def _to_01_gray(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return (img.astype(np.float32) / 255.0).clip(0.0, 1.0)
    if img.dtype in (np.float32, np.float64):
        return img.astype(np.float32).clip(0.0, 1.0)
    return img.astype(np.float32)


def _resize_hw(img01: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    h, w = int(hw[0]), int(hw[1])
    return cv2.resize(img01, (w, h), interpolation=cv2.INTER_AREA)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _as_prob(y: np.ndarray) -> np.ndarray:
    """
    Normalizes common ONNX output encodings into [0,1] probability maps.
    """
    y = np.asarray(y)
    if y.dtype.kind in ("f", "c"):
        # Heuristic: logits -> sigmoid, prob -> unchanged
        if y.min() < 0.0 or y.max() > 1.0:
            return _sigmoid(y)
        return y
    return y.astype(np.float32)


def _is_negated_like_wl(img01: np.ndarray) -> bool:
    return float(img01.mean()) > 0.5


def _normlines_like_wl(img01: np.ndarray) -> np.ndarray:
    """
    Simple contrast normalization akin to WL pipeline behavior.
    """
    x = img01.astype(np.float32)
    lo, hi = np.percentile(x, 5.0), np.percentile(x, 95.0)
    if hi <= lo:
        return x
    x = (x - lo) / (hi - lo)
    return x.clip(0.0, 1.0)


def _preproc_like_wl(img_gray: np.ndarray) -> np.ndarray:
    img01 = _to_01_gray(img_gray)
    if _is_negated_like_wl(img01):
        img01 = 1.0 - img01
    img01 = _normlines_like_wl(img01)
    return img01


def _hann1d(n: int) -> np.ndarray:
    n = int(n)
    if n <= 1:
        return np.ones((n,), dtype=np.float32)
    w = 0.5 - 0.5 * np.cos(2.0 * math.pi * np.arange(n, dtype=np.float32) / (n - 1))
    return (0.25 + 0.75 * w).astype(np.float32)


def _hann2d(h: int, w: int) -> np.ndarray:
    wy = _hann1d(int(h))
    wx = _hann1d(int(w))
    return np.outer(wy, wx).astype(np.float32)


def _pad_to_at_least(img01: np.ndarray, min_h: int, min_w: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = img01.shape
    top = 0
    left = 0
    bottom = max(0, int(min_h) - h)
    right = max(0, int(min_w) - w)
    out = cv2.copyMakeBorder(img01, top, bottom, left, right, borderType=cv2.BORDER_REFLECT_101)
    return out, (top, bottom, left, right)


def _pad_to_multiple(img01: np.ndarray, multiple: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = img01.shape
    mh = int(multiple)
    mw = int(multiple)
    pad_h = (mh - (h % mh)) % mh
    pad_w = (mw - (w % mw)) % mw
    out = cv2.copyMakeBorder(img01, 0, pad_h, 0, pad_w, borderType=cv2.BORDER_REFLECT_101)
    return out, (0, pad_h, 0, pad_w)


# ---------------------------
# Skeleton utilities
# ---------------------------

def prob_to_mask(prob: np.ndarray, thr: float = 0.20) -> np.ndarray:
    return (prob >= float(thr)).astype(np.uint8)


def prune_endpoints(skel: np.ndarray, iterations: int = 1) -> np.ndarray:
    """
    Iteratively removes endpoints from a binary skeleton.
    """
    sk = skel.astype(np.uint8).copy()
    H, W = sk.shape

    def endpoints(arr: np.ndarray) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        ys, xs = np.where(arr == 1)
        for y, x in zip(ys, xs):
            y = int(y)
            x = int(x)
            y0, y1 = max(0, y - 1), min(H, y + 2)
            x0, x1 = max(0, x - 1), min(W, x + 2)
            nb = int(np.sum(arr[y0:y1, x0:x1])) - 1
            if nb == 1:
                out.append((y, x))
        return out

    for _ in range(int(iterations)):
        to_zero = endpoints(sk)
        if not to_zero:
            break
        for y, x in to_zero:
            sk[y, x] = 0
    return sk


def filter_components(mask: np.ndarray, min_px: int, min_rows: int) -> np.ndarray:
    """
    Removes small connected components by pixel count and vertical span.
    """
    lab = label(mask.astype(np.uint8), connectivity=2)
    keep = np.zeros_like(mask, dtype=np.uint8)

    for r in regionprops(lab):
        y0, x0, y1, x1 = r.bbox
        vertical_span = int(y1 - y0)
        if int(r.area) >= int(min_px) and vertical_span >= int(min_rows):
            rr, cc = zip(*r.coords)
            keep[tuple(rr), tuple(cc)] = 1

    return keep


# ---------------------------
# ORT model wrapper
# ---------------------------

class ORTModel:
    """
    ONNXRuntime inference wrapper that supplies defaults for non-image inputs.
    """
    def __init__(self, path: Path, providers: Optional[list[str]] = None):
        self.path = str(path)

        self.model = onnx.load(self.path)
        self.init_names = {init.name for init in self.model.graph.initializer}

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.sess = ort.InferenceSession(
            self.path,
            sess_options=so,
            providers=(providers or ["CPUExecutionProvider"]),
        )

        self.inputs_meta = self.sess.get_inputs()
        self.outputs = [o.name for o in self.sess.get_outputs()]

        # Choose first float-ish input as the data input.
        self.data_input = self.inputs_meta[0]

    def _default_for(self, arg) -> Optional[np.ndarray]:
        name = arg.name
        if name in self.init_names:
            return None
        shp = getattr(arg, "shape", None)
        t = (getattr(arg, "type", "") or "").lower()

        if shp is None:
            return None
        if isinstance(shp, list):
            shp = tuple(shp)
        if any(d is None for d in shp):
            shp = tuple(1 for _ in shp)

        lname = name.lower()
        if lname in {"trainingmode", "is_training", "training", "train"} or "trainingmode" in lname:
            return np.array(False, dtype=np.bool_)
        if "bool" in t:
            return np.zeros(shp, dtype=np.bool_)
        if "float16" in t:
            return np.zeros(shp, dtype=np.float16)
        if "double" in t:
            return np.zeros(shp, dtype=np.float64)
        if "int64" in t:
            return np.zeros(shp, dtype=np.int64)
        if "int32" in t:
            return np.zeros(shp, dtype=np.int32)
        if "uint8" in t:
            return np.zeros(shp, dtype=np.uint8)
        return np.zeros(shp, dtype=np.float32)

    def run(self, x: np.ndarray) -> dict[str, np.ndarray]:
        feed = {self.data_input.name: x.astype(np.float32, copy=False)}
        for arg in self.inputs_meta:
            if arg.name == self.data_input.name:
                continue
            default = self._default_for(arg)
            if default is not None:
                feed[arg.name] = default
        outs = self.sess.run(self.outputs, feed)
        return {name: arr for name, arr in zip(self.outputs, outs)}


# ---------------------------
# Main ONNX runner
# ---------------------------

class KymoButlerPT:
    """
    ORT-backed KymoButler runner.
    """
    def __init__(
        self,
        *,
        export_dir: Path,
        seg_size: int = 256,
        tile_stride: int = 128,
        tile_round_to: int = 16,
        use_tiling: bool = True,
        providers: Optional[list[str]] = None,
    ):
        self.export_dir = resolve_export_dir(export_dir)

        paths = {
            "uni": self.export_dir / "uni_seg.onnx",
            "bi": self.export_dir / "bi_seg.onnx",
            "clf": self.export_dir / "classifier.onnx",
            "dec": self.export_dir / "decision.onnx",
        }

        self.uni = ORTModel(paths["uni"], providers=providers)
        self.bi = ORTModel(paths["bi"], providers=providers)
        self.clf = ORTModel(paths["clf"], providers=providers)
        self.dec = ORTModel(paths["dec"], providers=providers)

        self.seg_hw = (int(seg_size), int(seg_size))
        self.uni_hw = self.seg_hw
        self.bi_hw = self.seg_hw
        self.clf_hw = (64, 64)
        self.dec_hw = (48, 48)

        self.tile_stride = int(tile_stride)
        self.tile_round_to = int(tile_round_to)
        self.use_tiling = bool(use_tiling)

    def preproc_for_seg(self, img_gray: np.ndarray, hw: tuple[int, int] | None = None) -> np.ndarray:
        img01 = _preproc_like_wl(img_gray)
        if hw is not None:
            img01 = _resize_hw(img01, hw)
        return img01

    def _prep_gray(self, img_gray: np.ndarray, hw: tuple[int, int], wl_preproc: bool = True) -> np.ndarray:
        img01 = _preproc_like_wl(img_gray) if wl_preproc else _to_01_gray(img_gray)
        img01 = _resize_hw(img01, hw)
        return img01[None, None, ...].astype(np.float32)

    def _tile_infer_2d(self, img01: np.ndarray, run_fn, *, out_kind: str):
        seg_h, seg_w = self.seg_hw
        H, W = img01.shape

        stride = self.tile_stride
        wy = list(range(0, max(1, H - seg_h + 1), stride))
        wx = list(range(0, max(1, W - seg_w + 1), stride))
        if wy[-1] != H - seg_h:
            wy.append(H - seg_h)
        if wx[-1] != W - seg_w:
            wx.append(W - seg_w)

        w2d = _hann2d(seg_h, seg_w)
        eps = 1e-6

        if out_kind == "bi":
            acc = np.zeros((H, W), dtype=np.float32)
            wsum = np.zeros((H, W), dtype=np.float32)
        else:
            acc_ant = np.zeros((H, W), dtype=np.float32)
            acc_ret = np.zeros((H, W), dtype=np.float32)
            wsum = np.zeros((H, W), dtype=np.float32)

        for y in wy:
            for x in wx:
                tile = img01[y : y + seg_h, x : x + seg_w]
                tile_nchw = tile[None, None, ...].astype(np.float32)
                out = run_fn(tile_nchw)

                if out_kind == "bi":
                    ypred = list(out.values())[0]
                    ypred = np.squeeze(ypred)
                    prob = _as_prob(ypred).astype(np.float32)
                    acc[y : y + seg_h, x : x + seg_w] += prob * w2d
                    wsum[y : y + seg_h, x : x + seg_w] += w2d
                else:
                    if "ant" in out and "ret" in out:
                        ant = _as_prob(out["ant"]).squeeze().astype(np.float32)
                        ret = _as_prob(out["ret"]).squeeze().astype(np.float32)
                    else:
                        ypred = list(out.values())[0]
                        if ypred.ndim == 4 and ypred.shape[-1] == 2:
                            ant = _as_prob(ypred[..., 0]).squeeze().astype(np.float32)
                            ret = _as_prob(ypred[..., 1]).squeeze().astype(np.float32)
                        elif ypred.ndim == 4 and ypred.shape[1] == 2:
                            ant = _as_prob(ypred[:, 0]).squeeze().astype(np.float32)
                            ret = _as_prob(ypred[:, 1]).squeeze().astype(np.float32)
                        else:
                            raise RuntimeError(f"Unexpected uni_seg output shape: {ypred.shape}")
                    acc_ant[y : y + seg_h, x : x + seg_w] += ant * w2d
                    acc_ret[y : y + seg_h, x : x + seg_w] += ret * w2d
                    wsum[y : y + seg_h, x : x + seg_w] += w2d

        if out_kind == "bi":
            prob_full = acc / np.maximum(wsum, eps)
            return prob_full.astype(np.float32)
        else:
            ant_full = acc_ant / np.maximum(wsum, eps)
            ret_full = acc_ret / np.maximum(wsum, eps)
            return {"ant": ant_full.astype(np.float32), "ret": ret_full.astype(np.float32)}

    def classify(self, img_gray: np.ndarray) -> dict:
        x = self._prep_gray(img_gray, self.clf_hw, wl_preproc=True)
        out = self.clf.run(x)
        y = list(out.values())[0]
        y = np.asarray(y)
        if y.ndim >= 2 and y.shape[-1] >= 2:
            probs = _softmax(y, axis=-1).reshape(-1)
            label = int(np.argmax(probs))
            return {"label": label, "probs": probs.tolist()}
        prob = float(_as_prob(y).reshape(-1)[0])
        label = 1 if prob >= 0.5 else 0
        return {"label": label, "probs": [1.0 - prob, prob]}

    def decision_map(self, crop_raw01: np.ndarray, crop_skel_all: np.ndarray, crop_skel_curr: np.ndarray) -> np.ndarray:
        x = np.stack(
            [
                _resize_hw(_to_01_gray(crop_raw01), self.dec_hw),
                _resize_hw(_to_01_gray(crop_skel_all), self.dec_hw),
                _resize_hw(_to_01_gray(crop_skel_curr), self.dec_hw),
            ],
            axis=0,
        )[None, ...].astype(np.float32)
        out = self.dec.run(x)
        y = list(out.values())[0]
        y = np.asarray(y)
        if y.ndim == 4 and y.shape[-1] == 2:
            prob = _softmax(y, axis=-1)[0, ..., 1]
        elif y.ndim == 4 and y.shape[1] == 2:
            prob = _softmax(np.moveaxis(y, 1, -1), axis=-1)[0, ..., 1]
        else:
            prob = _as_prob(y).squeeze()
        return prob.astype(np.float32)

    def segment_bi_full(self, img_gray: np.ndarray) -> np.ndarray:
        img01 = _preproc_like_wl(img_gray)
        seg_h, seg_w = self.seg_hw

        img01, pads1 = _pad_to_at_least(img01, seg_h, seg_w)
        img01, pads2 = _pad_to_multiple(img01, self.tile_round_to)

        def run(tile_nchw: np.ndarray) -> dict[str, np.ndarray]:
            return self.bi.run(tile_nchw)

        if self.use_tiling:
            prob_full = self._tile_infer_2d(img01, run, out_kind="bi")
        else:
            x = self._prep_gray(img_gray, self.bi_hw, wl_preproc=True)
            out = self.bi.run(x)
            y = np.squeeze(list(out.values())[0])
            prob = _as_prob(y).astype(np.float32)
            H0, W0 = img_gray.shape
            prob = cv2.resize(prob, (W0, H0), interpolation=cv2.INTER_LINEAR)
            return prob.astype(np.float32)

        # Tiled path: crop off padding (no resize) to avoid alignment shift.
        H0, W0 = img_gray.shape
        pt1, pb1, pl1, pr1 = pads1
        pt2, pb2, pl2, pr2 = pads2
        pt = pt1 + pt2
        pl = pl1 + pl2
        prob = prob_full[pt : pt + H0, pl : pl + W0]
        return prob.astype(np.float32)

    def segment_uni_full(self, img_gray: np.ndarray) -> dict[str, np.ndarray]:
        img01 = _preproc_like_wl(img_gray)
        seg_h, seg_w = self.seg_hw

        img01, pads1 = _pad_to_at_least(img01, seg_h, seg_w)
        img01, pads2 = _pad_to_multiple(img01, self.tile_round_to)

        def run(tile_nchw: np.ndarray) -> dict[str, np.ndarray]:
            return self.uni.run(tile_nchw)

        if self.use_tiling:
            out_full = self._tile_infer_2d(img01, run, out_kind="uni")
        else:
            x = self._prep_gray(img_gray, self.uni_hw, wl_preproc=True)
            out = self.uni.run(x)
            y = list(out.values())[0]
            y = np.asarray(y)
            if y.ndim == 4 and y.shape[-1] == 2:
                ant = _as_prob(y[..., 0]).squeeze().astype(np.float32)
                ret = _as_prob(y[..., 1]).squeeze().astype(np.float32)
            elif y.ndim == 4 and y.shape[1] == 2:
                ant = _as_prob(y[:, 0]).squeeze().astype(np.float32)
                ret = _as_prob(y[:, 1]).squeeze().astype(np.float32)
            else:
                raise RuntimeError(f"Unexpected uni_seg output shape: {y.shape}")
            H0, W0 = img_gray.shape
            ant = cv2.resize(ant, (W0, H0), interpolation=cv2.INTER_LINEAR)
            ret = cv2.resize(ret, (W0, H0), interpolation=cv2.INTER_LINEAR)
            return {"ant": ant.astype(np.float32), "ret": ret.astype(np.float32)}

        # Tiled path: crop off padding (no resize) to avoid alignment shift.
        H0, W0 = img_gray.shape
        pt1, pb1, pl1, pr1 = pads1
        pt2, pb2, pl2, pr2 = pads2
        pt = pt1 + pt2
        pl = pl1 + pl2
        ant = out_full["ant"][pt : pt + H0, pl : pl + W0]
        ret = out_full["ret"][pt : pt + H0, pl : pl + W0]
        return {"ant": ant.astype(np.float32), "ret": ret.astype(np.float32)}

    def segment_bi(self, img_gray: np.ndarray) -> np.ndarray:
        x = self._prep_gray(img_gray, self.bi_hw, wl_preproc=True)
        out = self.bi.run(x)
        y = np.squeeze(list(out.values())[0])
        return _as_prob(y).astype(np.float32)

    def segment_uni(self, img_gray: np.ndarray) -> dict[str, np.ndarray]:
        x = self._prep_gray(img_gray, self.uni_hw, wl_preproc=True)
        out = self.uni.run(x)
        y = list(out.values())[0]
        y = np.asarray(y)
        if y.ndim == 4 and y.shape[-1] == 2:
            ant = _as_prob(y[..., 0]).squeeze().astype(np.float32)
            ret = _as_prob(y[..., 1]).squeeze().astype(np.float32)
        elif y.ndim == 4 and y.shape[1] == 2:
            ant = _as_prob(y[:, 0]).squeeze().astype(np.float32)
            ret = _as_prob(y[:, 1]).squeeze().astype(np.float32)
        else:
            raise RuntimeError(f"Unexpected uni_seg output shape: {y.shape}")
        return {"ant": ant.astype(np.float32), "ret": ret.astype(np.float32)}
