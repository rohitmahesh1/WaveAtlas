from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "waveatlas-mplconfig"))

import numpy as np
import yaml
from PIL import Image

from app.extract_core import _detect_peak_sets, _flatten_onnx_cfg_for_runner, process_track
from app.api.routes_jobs import _detect_peak_sets_for_detail, _detail_fit_meta_for_original_polarity
from app.io.table_to_heatmap import table_to_heatmap_bytes
from app.job_store import _PEAK_MODEL_KEYS, _WAVE_MODEL_KEYS, _json_safe, _row_for_metric_model
from app.modules.kb_adapter import link_track_endpoints
from app.modules.tracker import Track


def _synthetic_track_path(tmp: str, position: np.ndarray) -> Path:
    base = Path(tmp) / "synthetic_sample" / "kymobutler_output"
    base.mkdir(parents=True)
    track_path = base / "0.npy"
    frame = np.arange(position.size, dtype=float)
    np.save(track_path, np.column_stack([frame, position]))
    return track_path


def _base_config(*, fit_target: str | None = None, event_polarity: str = "both") -> dict:
    features = {"fit_window_period_frac": 0.5, "compare_fit_targets": True}
    if fit_target is not None:
        features["fit_target"] = fit_target
    return {
        "io": {"sampling_rate": 10.0},
        "kymo": {"backend": "onnx", "track_xy_order": "yx"},
        "detrend": {"degree": 1, "min_samples": 0.5, "random_state": 42},
        "peaks": {
            "event_polarity": event_polarity,
            "adaptive": False,
            "minimum_per_track": 1,
            "prominence": 2.0,
            "width": 1,
            "distance": 6,
        },
        "period": {"sampling_rate": 10.0, "min_freq": 0.2, "max_freq": 2.0},
        "features": features,
    }


class BackendCoreTests(unittest.TestCase):
    def test_default_config_enables_requested_extraction_defaults(self) -> None:
        config = yaml.safe_load(Path("configs/default.yaml").read_text())

        self.assertEqual(config["peaks"]["event_polarity"], "both")
        self.assertEqual(config["features"]["fit_target"], "raw_wave")
        self.assertTrue(config["features"]["compare_fit_targets"])
        self.assertEqual(config["heatmap"]["table_mode"], "auto")
        self.assertEqual(config["heatmap"]["area"]["cmap"], "plasma")
        self.assertFalse(config["heatmap"]["area"]["binarize"])
        endpoint_link = config["kymo"]["onnx"]["postproc"]["endpoint_link"]
        self.assertTrue(endpoint_link["enabled"])
        self.assertNotIn("level", endpoint_link)
        self.assertEqual(endpoint_link["max_gap_rows"], 60)
        self.assertEqual(endpoint_link["max_dx"], 10.0)
        self.assertEqual(endpoint_link["min_bridge_prob"], 0.075)
        self.assertEqual(endpoint_link["max_slope_delta"], 0.7)
        self.assertEqual(endpoint_link["fit_rows"], 16)
        self.assertEqual(endpoint_link["max_conflict_fraction"], 0.15)
        self.assertTrue(endpoint_link["overlap_enabled"])
        runner_cfg = _flatten_onnx_cfg_for_runner(config["kymo"]["onnx"])
        self.assertEqual(runner_cfg["endpoint_link_max_gap_rows"], 60)
        self.assertEqual(runner_cfg["endpoint_link_max_dx"], 10.0)
        self.assertEqual(runner_cfg["endpoint_link_min_bridge_prob"], 0.075)
        self.assertEqual(runner_cfg["endpoint_link_max_slope_delta"], 0.7)
        self.assertEqual(runner_cfg["endpoint_link_fit_rows"], 16)
        self.assertTrue(runner_cfg["endpoint_link_overlap_enabled"])

    def test_area_named_table_uses_continuous_heatmap_mode(self) -> None:
        csv = b"0,0.5,1\n0.25,0.75,1\n"

        png, meta = table_to_heatmap_bytes(
            csv,
            config={
                "heatmap": {
                    "table_mode": "auto",
                    "origin": "upper",
                    "area": {"cmap": "gray", "vmin": 0, "vmax": 1},
                }
            },
            filename_hint="DCPM2-DIES-CD1-CON-1-BH_Area_Vertical_Edge.csv",
        )

        image = Image.open(io.BytesIO(png)).convert("RGBA")
        row = [image.getpixel((x, 0))[0] for x in range(3)]

        self.assertEqual(image.size, (3, 2))
        self.assertEqual(meta["resolved_table_mode"], "area")
        self.assertFalse(meta["binarize"])
        self.assertEqual(meta["cmap"], "gray")
        self.assertLess(row[0], row[1])
        self.assertLess(row[1], row[2])

    def test_non_area_table_keeps_extreme_mask_heatmap_mode(self) -> None:
        csv = b"0,2e16,-3e20\n1,2,3\n"

        png, meta = table_to_heatmap_bytes(
            csv,
            config={
                "heatmap": {
                    "table_mode": "auto",
                    "lower": -1e20,
                    "upper": 1e16,
                    "binarize": True,
                    "origin": "upper",
                    "cmap": "gray",
                }
            },
            filename_hint="DCPM2-DIES-CD1-CON-1-BH_Mean_intensities_Vertical_Edge_Filtered.csv",
        )

        image = Image.open(io.BytesIO(png)).convert("RGBA")
        first_row = [image.getpixel((x, 0))[0] for x in range(3)]
        second_row = [image.getpixel((x, 1))[0] for x in range(3)]

        self.assertEqual(meta["resolved_table_mode"], "extreme_mask")
        self.assertTrue(meta["binarize"])
        self.assertEqual(first_row, [0, 255, 255])
        self.assertEqual(second_row, [0, 0, 0])

    def test_endpoint_link_levels_resolve_from_preset_and_allow_overrides(self) -> None:
        maximal = _flatten_onnx_cfg_for_runner({
            "postproc": {"endpoint_link": {"enabled": True, "level": "maximal"}}
        })
        aggressive = _flatten_onnx_cfg_for_runner({
            "postproc": {"endpoint_link": {"enabled": True, "level": "aggressive"}}
        })
        custom = _flatten_onnx_cfg_for_runner({
            "postproc": {"endpoint_link": {"enabled": True, "level": "aggressive", "max_gap_rows": 42}}
        })

        self.assertEqual(maximal["endpoint_link_max_gap_rows"], 35)
        self.assertEqual(maximal["endpoint_link_max_dx"], 6.0)
        self.assertGreater(aggressive["endpoint_link_max_gap_rows"], maximal["endpoint_link_max_gap_rows"])
        self.assertGreater(aggressive["endpoint_link_max_dx"], maximal["endpoint_link_max_dx"])
        self.assertLess(aggressive["endpoint_link_min_bridge_prob"], maximal["endpoint_link_min_bridge_prob"])
        self.assertEqual(custom["endpoint_link_max_gap_rows"], 42)

    def test_process_track_extracts_min_and_max_with_raw_wave_primary_fit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            frame = np.arange(0, 80, dtype=float)
            position = 40.0 + 0.15 * frame + 5.0 * np.sin(2.0 * np.pi * frame / 10.0)
            track_path = _synthetic_track_path(tmp, position)

            track_row, wave_rows, peak_rows, overlay = process_track(
                job_id=uuid4(),
                track_index=0,
                track_path=track_path,
                config=_base_config(),
                heatmap_meta={"output_height": 100, "coord_origin": "upper", "pixel_mapping": "synthetic"},
            )

        self.assertEqual(track_row["metrics"]["event_polarity"], "both")
        self.assertGreater(track_row["metrics"]["num_maxima"], 0)
        self.assertGreater(track_row["metrics"]["num_minima"], 0)
        self.assertIn("max", {row["event_kind"] for row in wave_rows})
        self.assertIn("min", {row["event_kind"] for row in wave_rows})
        self.assertTrue(any(float(row["value"]) < 0 for row in peak_rows if row["event_kind"] == "min"))
        self.assertIn("min", {peak["kind"] for peak in overlay["peaks"]})

        max_wave = next(row for row in wave_rows if row["event_kind"] == "max")
        metrics = max_wave["metrics"]
        self.assertEqual(metrics["fit_target"], "raw_wave")
        self.assertTrue(metrics["compare_fit_targets"])
        self.assertIn("raw_fit_error_vnmse", metrics)
        self.assertIn("residual_fit_error_vnmse", metrics)
        self.assertEqual(metrics["fit_error_vnmse"], metrics["raw_fit_error_vnmse"])
        self.assertAlmostEqual(max_wave["error"], metrics["raw_fit_error_vnmse"])

    def test_residual_fit_target_changes_primary_fit_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            frame = np.arange(0, 80, dtype=float)
            position = 40.0 + 0.15 * frame + 5.0 * np.sin(2.0 * np.pi * frame / 10.0)
            track_path = _synthetic_track_path(tmp, position)

            _track_row, wave_rows, _peak_rows, _overlay = process_track(
                job_id=uuid4(),
                track_index=0,
                track_path=track_path,
                config=_base_config(fit_target="residual"),
                heatmap_meta={"output_height": 100, "coord_origin": "upper", "pixel_mapping": "synthetic"},
            )

        max_wave = next(row for row in wave_rows if row["event_kind"] == "max")
        metrics = max_wave["metrics"]
        self.assertEqual(metrics["fit_target"], "residual")
        self.assertIn("raw_fit_error_vnmse", metrics)
        self.assertIn("residual_fit_error_vnmse", metrics)
        self.assertEqual(metrics["fit_error_vnmse"], metrics["residual_fit_error_vnmse"])
        self.assertAlmostEqual(max_wave["error"], metrics["residual_fit_error_vnmse"])

    def test_track_detail_peak_detection_respects_both_polarities(self) -> None:
        residual = np.array([0.0, 4.0, 0.0, -5.0, 0.0, 3.0, 0.0, -4.0, 0.0])
        peaks_cfg = {
            "event_polarity": "both",
            "adaptive": False,
            "minimum_per_track": 1,
            "prominence": 1.0,
            "width": 1,
        }

        peak_sets = _detect_peak_sets_for_detail(residual, peaks_cfg, frames_per_period=None)
        by_kind = {peak_set["event_kind"]: peak_set for peak_set in peak_sets}

        self.assertEqual(set(by_kind), {"max", "min"})
        self.assertEqual(by_kind["max"]["peaks_idx"].tolist(), [1, 5])
        self.assertEqual(by_kind["min"]["peaks_idx"].tolist(), [3, 7])
        self.assertGreater(float(by_kind["min"]["signal"][3]), 0.0)

    def test_cross_polarity_peak_filter_keeps_stronger_close_extrema(self) -> None:
        residual = np.array([0.0, 5.0, 0.0, -2.0, 0.0, -6.0, 0.0, 4.0, 0.0])
        peaks_cfg = {
            "event_polarity": "both",
            "adaptive": False,
            "minimum_per_track": 0,
            "prominence": 1.0,
            "width": 1,
            "distance": 1,
            "cross_polarity_min_distance": 3,
            "cross_polarity_policy": "stronger",
        }

        peak_sets = _detect_peak_sets(residual, peaks_cfg, frames_per_period=None)
        by_kind = {peak_set["event_kind"]: peak_set for peak_set in peak_sets}

        self.assertEqual(by_kind["max"]["peaks_idx"].tolist(), [1])
        self.assertEqual(by_kind["min"]["peaks_idx"].tolist(), [5])

    def test_track_detail_minima_fit_metadata_maps_back_to_original_sign(self) -> None:
        fit_meta = {
            "fit_amp_A": 2.0,
            "fit_offset_c": 1.0,
            "fit_peak_error": 0.25,
            "fit_peak_value": 4.0,
        }

        out = _detail_fit_meta_for_original_polarity(
            fit_meta,
            sign=-1,
            original_peak_value=-4.0,
        )

        self.assertEqual(out["fit_signal_sign"], -1)
        self.assertEqual(out["fit_event_value"], 4.0)
        self.assertEqual(out["fit_peak_value"], -4.0)
        self.assertEqual(out["fit_amp_A"], -2.0)
        self.assertEqual(out["fit_offset_c"], -1.0)
        self.assertEqual(out["fit_peak_error"], -0.25)

    def test_lower_origin_uses_raw_frames_for_wave_motion_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            frame = np.arange(0, 120, dtype=float)
            position = 100.0 + 0.5 * frame + 5.0 * np.sin(2.0 * np.pi * frame / 12.0)
            track_path = _synthetic_track_path(tmp, position)

            _track_row, wave_rows, _peak_rows, _overlay = process_track(
                job_id=uuid4(),
                track_index=0,
                track_path=track_path,
                config=_base_config(event_polarity="maxima"),
                heatmap_meta={"output_height": 1500, "coord_origin": "lower", "pixel_mapping": "table_cell"},
            )

        interior_rows = [
            row for row in wave_rows
            if row["metrics"]["previous_peak_i"] is not None
            and row["metrics"]["next_peak_i"] is not None
            and not row["metrics"]["boundary_extrapolated"]
        ]
        self.assertGreater(len(interior_rows), 0)
        metrics = interior_rows[0]["metrics"]
        self.assertGreater(abs(metrics["delta_pos_px"]), 0.1)
        self.assertGreater(metrics["wavelength_px"], 0.1)
        self.assertLess(metrics["frame1_raw"], 120.0)
        self.assertGreater(metrics["frame1"], 1000.0)

    def test_metric_model_rows_keep_event_labels_as_columns_and_fit_scores_as_metrics(self) -> None:
        wave = _row_for_metric_model(
            {
                "wave_index": 1,
                "event_kind": "min",
                "fit_error_vnmse": 0.2,
                "metrics": {"fit_target": "raw_wave", "empty_quality": float("nan")},
            },
            model_keys=_WAVE_MODEL_KEYS,
        )
        peak = _row_for_metric_model(
            {"pos": 3.0, "value": -1.0, "event_polarity": "minima", "metrics": {}},
            model_keys=_PEAK_MODEL_KEYS,
        )

        self.assertEqual(wave["event_kind"], "min")
        self.assertEqual(wave["metrics"]["fit_error_vnmse"], 0.2)
        self.assertEqual(wave["metrics"]["fit_target"], "raw_wave")
        self.assertIsNone(wave["metrics"]["empty_quality"])
        self.assertEqual(peak["event_polarity"], "minima")
        json.dumps(wave, allow_nan=False)
        json.dumps(peak, allow_nan=False)

    def test_json_safe_converts_non_finite_values_for_postgres_json(self) -> None:
        payload = {
            "nan": float("nan"),
            "inf": np.float32(np.inf),
            "array": np.array([1.0, np.nan, np.inf]),
            "nested": [{"count": np.int64(4), "ok": True}],
        }

        safe = _json_safe(payload)

        self.assertIsNone(safe["nan"])
        self.assertIsNone(safe["inf"])
        self.assertEqual(safe["array"], [1.0, None, None])
        self.assertEqual(safe["nested"][0]["count"], 4)
        json.dumps(safe, allow_nan=False)

    def test_endpoint_linker_merges_clean_fragments(self) -> None:
        prob = np.full((16, 16), 0.2, dtype=np.float32)
        tracks = [
            Track(points=[(0, 5), (1, 5), (2, 5)], id=0),
            Track(points=[(6, 5), (7, 5), (8, 5)], id=1),
        ]

        linked, stats = link_track_endpoints(
            tracks,
            prob,
            max_gap_rows=5,
            max_dx=1,
            min_bridge_prob=0.1,
            max_slope_delta=0.1,
            fit_rows=3,
            max_conflict_fraction=0.0,
            insert_bridge_points=True,
        )

        self.assertEqual(stats["accepted_links"], 1)
        self.assertEqual(len(linked), 1)
        self.assertEqual(
            linked[0].points,
            [(0, 5), (1, 5), (2, 5), (3, 5), (4, 5), (5, 5), (6, 5), (7, 5), (8, 5)],
        )


if __name__ == "__main__":
    unittest.main()
