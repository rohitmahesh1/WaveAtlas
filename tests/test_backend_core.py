from __future__ import annotations

import csv
import io
import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "waveatlas-mplconfig"))

import numpy as np
import yaml
from fastapi import HTTPException
from PIL import Image
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine, select

from app import pipeline as pipeline_module
from app.artifact_store import LocalArtifactStore
from app.cancel import CancellationRequested
from app.analysis_mode import RIPPLE_ANALYSIS_MODE, STANDARD_ANALYSIS_MODE, resolve_analysis_mode
from app.config_validation import normalize_and_validate_config
from app.extract_core import detect_peak_sets, _flatten_onnx_cfg_for_runner, process_track
from app.api.routes_jobs import (
    ConfigValidatePayload,
    _detect_peak_sets_for_detail,
    _detail_fit_meta_for_original_polarity,
    _fit_anchored_sine,
    _fit_anchored_wave_basin,
    _large_wave_peak_events_for_detail,
    validate_config,
)
from app.io.image_to_heatmap import image_to_heatmap_bytes
from app.io.table_to_heatmap import table_to_heatmap_bytes, table_to_heatmap_payload
from app.heatmap_values import read_cv_image, write_cv_image
from app.job_store import JobStore, _PEAK_MODEL_KEYS, _WAVE_MODEL_KEYS, _json_safe, _row_for_metric_model
from app.large_wave_extraction import (
    _RidgeTrace,
    _ridge_candidate_masks,
    _resolve_track_conflicts,
    run_large_wave_extraction,
)
from app.large_wave_analysis import (
    LargeWavePreparedTrack,
    STANDARD_WAVE_FIELDS,
    _assign_recurrence_periods,
    _dedupe_track_measurements,
    analyze_large_wave_events,
    prepare_large_wave_track,
)
from app.large_wave_fit import fit_large_wave
from app.modules.kb_adapter import link_track_endpoints
from app.modules.kymo_interface import _canonicalize_wolfram_track
from app.modules.kymobutler_pt import KymoButlerPT
from app.modules.tracker import CrossingTracker, Track
from app.models import ArtifactKind, Artifact, JobRead, JobStatus, Track as TrackModel, Wave
from app.measurement_schema import (
    MEASUREMENT_DEFINITIONS,
    WAVE_EXPORT_FAMILIAR_HEADERS,
    descriptive_ripple_csv,
    measurement_schema_identity,
    profile_csv_columns,
    wave_export_descriptive_keys,
)
from app.pipeline import PipelineSettings
from app.ripple_analysis import analyze_ripple_tracks
from app.ripple_extraction import _Trace, _dedupe_and_extend
from app.signal.peaks import ensure_minimum_peaks
from app.signal import detrend as detrend_module
from app.signal.detrend import fit_baseline
from app.sampling import normalize_sampling_rate_config, resolve_sampling_rate
from app.run_manifest import sha256_bytes
from app.time_utils import utc_isoformat
from app.track_coordinates import (
    point_order_from_artifact,
    point_order_from_config,
    track_artifact_metadata,
    track_coordinates_from_points,
)


def _synthetic_track_path(tmp: str, position: np.ndarray) -> Path:
    base = Path(tmp) / "synthetic_sample" / "kymobutler_output"
    base.mkdir(parents=True)
    track_path = base / "0.npy"
    frame = np.arange(position.size, dtype=float)
    np.save(track_path, np.column_stack([frame, position]))
    return track_path


def _synthetic_ripple_track(base: Path, track_index: int, *, slope: float, intercept: float) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    frame = np.arange(0, 121, dtype=float)
    position = slope * frame + intercept
    path = base / f"track_{track_index}.npy"
    np.save(path, np.column_stack([frame, position]))
    return path


def _ripple_dedupe_test_config() -> dict:
    return {
        "dedupe": {
            "allow_cross_phase": True,
            "min_overlap_rows": 35,
            "min_overlap_fraction": 0.45,
            "min_partial_overlap_rows": 18,
            "min_short_overlap_fraction": 0.55,
            "max_slope_delta": 0.35,
            "max_median_distance_px": 6.0,
            "max_p90_distance_px": 10.0,
            "min_spatial_overlap_rows": 12,
            "close_distance_px": 7.0,
            "min_close_fraction": 0.68,
            "partial_max_median_distance_px": 7.0,
            "cross_phase_max_median_distance_px": 4.0,
            "cross_phase_max_p90_distance_px": 7.0,
            "cross_phase_min_spatial_overlap_rows": 16,
            "cross_phase_close_distance_px": 5.0,
            "cross_phase_min_close_fraction": 0.75,
            "cross_phase_partial_max_median_distance_px": 5.0,
        }
    }


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
        "period": {"min_freq": 0.2, "max_freq": 2.0},
        "features": features,
    }


class BackendCoreTests(unittest.TestCase):
    def test_sampling_rate_config_has_one_canonical_authority(self) -> None:
        canonical = normalize_sampling_rate_config(
            {"io": {"sampling_rate": 5}, "period": {"sampling_rate": 5, "min_freq": 0.1}}
        )

        self.assertEqual(canonical["io"]["sampling_rate"], 5.0)
        self.assertNotIn("sampling_rate", canonical["period"])
        self.assertEqual(resolve_sampling_rate(canonical), 5.0)

    def test_legacy_period_sampling_rate_is_migrated(self) -> None:
        canonical = normalize_sampling_rate_config({"period": {"sampling_rate": 7.5}})

        self.assertEqual(canonical["io"]["sampling_rate"], 7.5)
        self.assertNotIn("sampling_rate", canonical["period"])

    def test_conflicting_sampling_rates_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Conflicting sampling rates"):
            normalize_sampling_rate_config(
                {"io": {"sampling_rate": 5}, "period": {"sampling_rate": 10}}
            )

    def test_sampling_rate_must_be_finite_and_positive(self) -> None:
        for value in (0, -1, float("nan"), float("inf"), "not-a-number"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite positive"):
                    resolve_sampling_rate({"io": {"sampling_rate": value}})

    def test_critical_config_fields_are_normalized_and_validated(self) -> None:
        normalized = normalize_and_validate_config({"analysis": {"mode": "ripple"}})
        self.assertEqual(normalized["analysis"]["mode"], "ripple_family")

        invalid_configs = (
            ({"analysis": {"mode": "typo"}}, "analysis.mode"),
            ({"period": {"min_freq": 2, "max_freq": 1}}, "must not exceed"),
            ({"heatmap": {"vmin": 2, "vmax": 1}}, "must not exceed"),
            ({"heatmap": {"continuous": {"origin": "sideways"}}}, "origin"),
            ({"kymo": {"track_xy_order": "row-first-ish"}}, "track_xy_order"),
            ({"heatmap": {"non_finite_policy": "ignore"}}, "non_finite_policy"),
            ({"detrend": {"min_samples": 1.5}}, "must not exceed 1"),
            ({"features": {"fit_target": "mystery-fit"}}, "fit_target"),
        )
        for config, message in invalid_configs:
            with self.subTest(config=config):
                with self.assertRaisesRegex(ValueError, message):
                    normalize_and_validate_config(config)

    def test_config_endpoint_validates_overrides_against_defaults(self) -> None:
        with self.assertRaises(HTTPException) as caught:
            validate_config(ConfigValidatePayload(config={"period": {"min_freq": 3.0}}))

        self.assertEqual(caught.exception.status_code, 400)
        self.assertIn("must not exceed", str(caught.exception.detail))

    def test_detrend_fallback_reports_the_actual_estimator_and_reason(self) -> None:
        original = detrend_module._fit_ransac_baseline

        def fail_ransac(*_args, **_kwargs):
            raise ValueError("no consensus set")

        try:
            detrend_module._fit_ransac_baseline = fail_ransac
            result = fit_baseline(
                np.arange(10, dtype=float),
                np.linspace(2.0, 5.0, 10),
            )
        finally:
            detrend_module._fit_ransac_baseline = original

        meta = result.metadata()
        self.assertEqual(meta["method"], "least_squares_polynomial")
        self.assertTrue(meta["fallback_used"])
        self.assertIn("no consensus set", meta["fallback_reason"])
        self.assertIsNone(meta["inlier_fraction"])

    def test_non_finite_table_values_require_an_explicit_policy(self) -> None:
        table = b"1,nan\ninf,-inf\n"
        with self.assertRaisesRegex(ValueError, "3 non-finite numeric cell"):
            table_to_heatmap_payload(table)

        _, meta, value_bytes, value_meta = table_to_heatmap_payload(
            table,
            config={
                "heatmap": {
                    "table_mode": "continuous",
                    "non_finite_policy": "zero",
                    "vmin": 0,
                    "vmax": 1,
                }
            },
        )
        values = np.frombuffer(value_bytes, dtype="<f4")
        np.testing.assert_array_equal(values, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        self.assertEqual(meta["non_finite_count"], 3)
        self.assertEqual(meta["nan_count"], 1)
        self.assertEqual(meta["positive_infinity_count"], 1)
        self.assertEqual(meta["negative_infinity_count"], 1)
        self.assertEqual(meta["non_finite_disposition"], "zero_imputed")
        self.assertEqual(value_meta["non_finite_count"], 3)

    def test_large_wave_result_replacement_rolls_back_as_one_snapshot(self) -> None:
        engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        SQLModel.metadata.create_all(engine)

        with Session(engine) as session:
            store = JobStore(session)
            job = store.create_job(owner_session_id=uuid4(), run_name="atomic replacement")
            track_rows = [{
                "track_index": 0,
                "amplitude": 4.0,
                "frequency": 0.5,
                "metrics": {"analysis_mode": "large_wave", "version": "original"},
                "overlay": {"points": [[1, 2]]},
            }]
            wave_rows = [{
                "wave_index": 1,
                "amplitude": 4.0,
                "metrics": {"track_index": 0, "measurement": "original"},
            }]
            artifacts = [
                {
                    "label": label,
                    "blob_path": f"/staged/original-{label}.csv",
                    "content_type": "text/csv",
                    "byte_size": 12,
                    "meta": {"publication_id": "original"},
                }
                for label in (
                    "large_wave_tracks",
                    "large_wave_measurements",
                    "large_wave_events",
                )
            ]
            store.replace_large_wave_results(
                job.id,
                track_rows=track_rows,
                wave_rows=wave_rows,
                artifacts=artifacts,
            )

            with self.assertRaisesRegex(ValueError, "unknown track 99"):
                store.replace_large_wave_results(
                    job.id,
                    track_rows=[{**track_rows[0], "metrics": {"version": "replacement"}}],
                    wave_rows=[{"wave_index": 2, "metrics": {"track_index": 99}}],
                    artifacts=[
                        {**artifact, "blob_path": artifact["blob_path"].replace("original", "replacement")}
                        for artifact in artifacts
                    ],
                )

            tracks = list(session.exec(select(TrackModel).where(TrackModel.job_id == job.id)).all())
            waves = list(session.exec(select(Wave).where(Wave.job_id == job.id)).all())
            stored_artifacts = store.list_artifacts(
                job.id,
                kind=ArtifactKind.other,
                label="large_wave_measurements",
            )
            refreshed_job = store.get_job(job.id)

        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].metrics["version"], "original")
        self.assertEqual(len(waves), 1)
        self.assertEqual(waves[0].metrics["measurement"], "original")
        self.assertEqual(
            [artifact.blob_path for artifact in stored_artifacts],
            ["/staged/original-large_wave_measurements.csv"],
        )
        self.assertEqual(refreshed_job.tracks_done, 1)
        self.assertEqual(refreshed_job.waves_done, 1)

    def test_scientific_track_coordinates_are_bottom_left_for_every_render_origin(self) -> None:
        points = np.array([[0.0, 10.0], [1.0, 11.0], [3.0, 13.0]])

        for render_origin in ("lower", "upper"):
            coordinates = track_coordinates_from_points(
                points,
                heatmap_meta={
                    "output_height": 4,
                    "coord_origin": render_origin,
                    "render_origin": render_origin,
                },
            )

            np.testing.assert_array_equal(coordinates.frame, np.array([0.0, 2.0, 3.0]))
            np.testing.assert_array_equal(coordinates.image_row, np.array([3.0, 1.0, 0.0]))
            np.testing.assert_array_equal(coordinates.position, np.array([13.0, 11.0, 10.0]))

    def test_all_current_extractor_backends_default_to_row_column_points(self) -> None:
        self.assertEqual(point_order_from_config({"kymo": {"backend": "onnx"}}), "yx")
        self.assertEqual(point_order_from_config({"kymo": {"backend": "wolfram"}}), "yx")

    def test_wolfram_tracks_are_canonicalized_from_one_based_row_column_points(self) -> None:
        canonical = _canonicalize_wolfram_track(np.array([[1.0, 1.0], [4.0, 7.0]]))
        np.testing.assert_array_equal(canonical, np.array([[0.0, 0.0], [3.0, 6.0]]))

    def test_track_artifact_metadata_records_coordinate_contract(self) -> None:
        meta = track_artifact_metadata(
            7,
            {"output_height": 120, "coord_origin": "upper", "render_origin": "upper"},
        )

        self.assertEqual(meta["track_schema_version"], 1)
        self.assertEqual(meta["coordinate_order"], "image_row_position")
        self.assertEqual(meta["coordinate_index_base"], 0)
        self.assertEqual(meta["coord_origin"], "lower")
        self.assertEqual(meta["image_height"], 120.0)

        with self.assertRaisesRegex(ValueError, "Unsupported track schema version"):
            point_order_from_artifact(
                {**meta, "track_schema_version": 2},
                config={},
            )

    def test_image_io_supports_unicode_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "Workspace ü with spaces" / "heatmap.png"
            path.parent.mkdir()
            expected = np.arange(64, dtype=np.uint8).reshape((8, 8))
            write_cv_image(path, expected)
            actual = read_cv_image(path, 0)
            self.assertIsNotNone(actual)
            np.testing.assert_array_equal(actual, expected)

    def test_image_ingestion_declares_bottom_left_coordinates_by_default(self) -> None:
        image = Image.fromarray(np.zeros((4, 6), dtype=np.uint8))
        payload = io.BytesIO()
        image.save(payload, format="PNG")

        _png, meta = image_to_heatmap_bytes(
            payload.getvalue(),
            config={"image_input": {"origin": "upper"}},
        )

        self.assertEqual(meta["source_kind"], "image")
        self.assertEqual(meta["pixel_mapping"], "processed_pixel")
        self.assertEqual(meta["coord_origin"], "lower")
        self.assertEqual(meta["source_origin"], "upper")
        self.assertEqual(meta["coord_x_label"], "position")
        self.assertEqual(meta["coord_y_label"], "frame")

    def test_analysis_mode_defaults_to_standard_and_accepts_ripple_aliases(self) -> None:
        self.assertEqual(resolve_analysis_mode({}), STANDARD_ANALYSIS_MODE)
        self.assertEqual(resolve_analysis_mode({"analysis": {"mode": "ripple"}}), RIPPLE_ANALYSIS_MODE)
        self.assertEqual(resolve_analysis_mode({"analysis": {"mode": "ripple_family"}}), RIPPLE_ANALYSIS_MODE)

    def test_default_config_enables_requested_extraction_defaults(self) -> None:
        config = yaml.safe_load(Path("configs/default.yaml").read_text(encoding="utf-8"))

        self.assertEqual(config["peaks"]["event_polarity"], "both")
        self.assertEqual(config["features"]["fit_target"], "raw_wave")
        self.assertTrue(config["features"]["compare_fit_targets"])
        self.assertEqual(config["heatmap"]["table_mode"], "auto")
        self.assertTrue(config["heatmap"]["binarize"])
        self.assertEqual(config["heatmap"]["cmap"], "plasma")
        self.assertEqual(config["heatmap"]["continuous"]["cmap"], "plasma")
        self.assertEqual(config["heatmap"]["area"]["cmap"], "plasma")
        self.assertFalse(config["heatmap"]["area"]["binarize"])
        self.assertEqual(config["analysis"]["mode"], "standard")
        self.assertTrue(config["analysis"]["ripple"]["extraction"]["enabled"])
        self.assertEqual(config["analysis"]["ripple"]["extraction"]["max_tracks"], 500)
        large_wave_bias = config["analysis"]["large_wave"]["extraction"]["intensity_bias"]
        self.assertTrue(large_wave_bias["enabled"])
        self.assertEqual(large_wave_bias["strength"], 0.65)
        self.assertEqual(large_wave_bias["selection_weight"], 0.75)
        ripple_dedupe = config["analysis"]["ripple"]["extraction"]["dedupe"]
        self.assertTrue(ripple_dedupe["allow_cross_phase"])
        self.assertEqual(ripple_dedupe["min_overlap_rows"], 35)
        self.assertEqual(ripple_dedupe["min_overlap_fraction"], 0.45)
        self.assertEqual(ripple_dedupe["min_partial_overlap_rows"], 18)
        self.assertEqual(ripple_dedupe["min_short_overlap_fraction"], 0.55)
        self.assertEqual(ripple_dedupe["max_median_distance_px"], 6.0)
        self.assertEqual(ripple_dedupe["min_spatial_overlap_rows"], 12)
        self.assertEqual(ripple_dedupe["close_distance_px"], 7.0)
        self.assertEqual(ripple_dedupe["min_close_fraction"], 0.68)
        self.assertEqual(ripple_dedupe["cross_phase_min_spatial_overlap_rows"], 16)
        self.assertTrue(config["analysis"]["ripple"]["endpoint_link"]["prefer_long_linear"])
        self.assertEqual(config["analysis"]["ripple"]["family"]["min_tracks"], 2)
        large_extraction = config["analysis"]["large_wave"]["extraction"]
        self.assertTrue(large_extraction["enabled"])
        self.assertEqual(large_extraction["spatial_sigma_fraction"], 0.02)
        self.assertEqual(large_extraction["prominence"], 0.01)
        self.assertEqual(large_extraction["min_component_rows"], 50)
        self.assertEqual(large_extraction["endpoint_bridge"]["max_step_dx_px_per_row"], 4.0)
        large_ensemble = large_extraction["ensemble"]
        self.assertTrue(large_ensemble["enabled"])
        self.assertEqual(large_ensemble["cusp_spatial_sigma_ratio"], 0.35)
        self.assertEqual(large_ensemble["max_step_dx_px_per_row"], 12.0)
        self.assertEqual(large_ensemble["min_cusp_arm_displacement_px"], 5.0)
        self.assertEqual(large_ensemble["conflicts"]["max_overlap_distance_px"], 3.0)
        self.assertEqual(
            config["analysis"]["large_wave"]["events"]["fit_window_width_multiplier"],
            1.0,
        )
        self.assertEqual(
            config["analysis"]["large_wave"]["events"]["fit_boundary_smoothing_sigma_rows"],
            4.0,
        )
        self.assertEqual(
            config["analysis"]["large_wave"]["events"][
                "max_period_boundary_error_fraction"
            ],
            0.5,
        )
        large_peaks = config["analysis"]["large_wave"]["peaks"]
        self.assertEqual(large_peaks["minimum_per_track"], 1)
        self.assertEqual(large_peaks["minimum_scope"], "track")
        self.assertEqual(large_peaks["cross_polarity_min_distance"], 30)
        self.assertEqual(large_peaks["edge_margin_frames"], 40)
        large_events = config["analysis"]["large_wave"]["events"]
        self.assertEqual(large_events["fit_boundary_smoothing_scales"], [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(large_events["fit_candidate_error_tolerance"], 0.25)
        self.assertEqual(large_events["min_peak_separation_frames"], 30)
        self.assertEqual(large_events["peak_separation_width_fraction"], 0.3)
        self.assertEqual(large_events["duplicate_window_overlap_fraction"], 0.65)
        self.assertEqual(large_events["duplicate_support_overlap_fraction"], 0.8)
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
        self.assertEqual(endpoint_link["max_chord_slope_px_per_row"], 2.0)
        self.assertEqual(endpoint_link["max_step_dx_px_per_row"], 4.0)
        self.assertEqual(endpoint_link["max_manifest_rejections"], 500)
        runner_cfg = _flatten_onnx_cfg_for_runner(config["kymo"]["onnx"])
        self.assertEqual(runner_cfg["endpoint_link_max_gap_rows"], 60)
        self.assertEqual(runner_cfg["endpoint_link_max_dx"], 10.0)
        self.assertEqual(runner_cfg["endpoint_link_min_bridge_prob"], 0.075)
        self.assertEqual(runner_cfg["endpoint_link_max_slope_delta"], 0.7)
        self.assertEqual(runner_cfg["endpoint_link_fit_rows"], 16)
        self.assertTrue(runner_cfg["endpoint_link_overlap_enabled"])
        self.assertEqual(runner_cfg["endpoint_link_max_chord_slope_px_per_row"], 2.0)
        self.assertEqual(runner_cfg["endpoint_link_max_step_dx_px_per_row"], 4.0)
        self.assertFalse(runner_cfg["endpoint_link_prefer_long_linear"])
        self.assertFalse(runner_cfg["endpoint_link_prefer_smooth_curves"])

        ripple_runner_cfg = _flatten_onnx_cfg_for_runner(
            config["kymo"]["onnx"],
            analysis_cfg={"mode": "ripple_family", "ripple": {}},
        )
        self.assertTrue(ripple_runner_cfg["endpoint_link_prefer_long_linear"])
        self.assertFalse(ripple_runner_cfg["endpoint_link_prefer_smooth_curves"])

        large_runner_cfg = _flatten_onnx_cfg_for_runner(
            config["kymo"]["onnx"],
            analysis_cfg=config["analysis"] | {"mode": "large_wave"},
        )
        self.assertFalse(large_runner_cfg["endpoint_link_prefer_long_linear"])
        self.assertTrue(large_runner_cfg["endpoint_link_prefer_smooth_curves"])

    def test_auto_table_uses_continuous_heatmap_regardless_of_filename(self) -> None:
        csv = b"0,0.5,1\n0.25,0.75,1\n"

        png, meta = table_to_heatmap_bytes(
            csv,
            config={
                "heatmap": {
                    "table_mode": "auto",
                    "origin": "upper",
                    "continuous": {"cmap": "gray", "vmin": 0, "vmax": 1},
                    "area": {"cmap": "hot", "vmin": 0, "vmax": 1},
                }
            },
            filename_hint="DCPM2-DIES-CD1-CON-1-BH_Area_Vertical_Edge.csv",
        )

        image = Image.open(io.BytesIO(png)).convert("RGBA")
        row = [image.getpixel((x, 0))[0] for x in range(3)]

        self.assertEqual(image.size, (3, 2))
        self.assertEqual(meta["resolved_table_mode"], "continuous")
        self.assertFalse(meta["binarize"])
        self.assertEqual(meta["cmap"], "gray")
        self.assertEqual(meta["coord_origin"], "lower")
        self.assertEqual(meta["render_origin"], "upper")
        self.assertLess(row[0], row[1])
        self.assertLess(row[1], row[2])

    def test_continuous_intensity_heatmap_defaults_to_plotly_plasma(self) -> None:
        png, meta = table_to_heatmap_bytes(
            b"0,1\n",
            config={
                "heatmap": {
                    "table_mode": "continuous",
                    "origin": "upper",
                    "cmap": "hot",
                    "continuous": {"cmap": "plasma"},
                }
            },
            filename_hint="intensity.csv",
        )

        image = Image.open(io.BytesIO(png)).convert("RGBA")

        self.assertEqual(meta["cmap"], "plasma")
        np.testing.assert_allclose(image.getpixel((0, 0)), (13, 8, 135, 255), atol=1)
        np.testing.assert_allclose(image.getpixel((1, 0)), (240, 249, 33, 255), atol=1)

    def test_large_wave_auto_heatmap_resolves_to_continuous(self) -> None:
        _, meta, value_bytes, value_meta = table_to_heatmap_payload(
            b"-20,0,20\n5,11,-11\n",
            config={
                "analysis": {"mode": "large_wave"},
                "heatmap": {
                    "table_mode": "auto",
                    "lower": -10,
                    "upper": 10,
                    "origin": "upper",
                },
            },
            filename_hint="mean_intensities.csv",
        )

        values = np.frombuffer(value_bytes, dtype="<f4").reshape((2, 3))

        self.assertEqual(meta["table_mode"], "auto")
        self.assertEqual(meta["resolved_table_mode"], "continuous")
        self.assertFalse(meta["binarize"])
        self.assertEqual(meta["cmap"], "plasma")
        self.assertEqual(value_meta["resolved_table_mode"], "continuous")
        np.testing.assert_allclose(values, np.array([[-20, 0, 20], [5, 11, -11]], dtype=np.float32))

    def test_ripple_auto_heatmap_resolves_to_continuous(self) -> None:
        _, meta, value_bytes, value_meta = table_to_heatmap_payload(
            b"-20,0,20\n",
            config={
                "analysis": {"mode": "ripple_family"},
                "heatmap": {
                    "table_mode": "auto",
                    "lower": -10,
                    "upper": 10,
                    "origin": "upper",
                },
            },
            filename_hint="mean_intensities.csv",
        )

        values = np.frombuffer(value_bytes, dtype="<f4").reshape((1, 3))

        self.assertEqual(meta["table_mode"], "auto")
        self.assertEqual(meta["resolved_table_mode"], "continuous")
        self.assertFalse(meta["binarize"])
        self.assertEqual(meta["cmap"], "plasma")
        self.assertEqual(value_meta["resolved_table_mode"], "continuous")
        np.testing.assert_allclose(values, np.array([[-20, 0, 20]], dtype=np.float32))

    def test_large_wave_explicit_binary_heatmap_is_respected(self) -> None:
        _, meta, _, _ = table_to_heatmap_payload(
            b"-20,0,20\n",
            config={
                "analysis": {"mode": "large_wave"},
                "heatmap": {
                    "table_mode": "binary",
                    "lower": -10,
                    "upper": 10,
                    "origin": "upper",
                },
            },
            filename_hint="mean_intensities.csv",
        )

        self.assertEqual(meta["table_mode"], "binary")
        self.assertEqual(meta["resolved_table_mode"], "extreme_mask")
        self.assertTrue(meta["binarize"])

    def test_table_heatmap_payload_preserves_continuous_display_values_by_default(self) -> None:
        csv = b"0,0.5,1\n0.25,0.75,1\n"

        _, meta, value_bytes, value_meta = table_to_heatmap_payload(
            csv,
            config={
                "heatmap": {
                    "table_mode": "area",
                    "origin": "lower",
                    "area": {"cmap": "gray", "vmin": 0, "vmax": 1},
                }
            },
            filename_hint="cells_area.csv",
        )

        values = np.frombuffer(value_bytes, dtype="<f4").reshape((2, 3))

        self.assertEqual(meta["z_min"], 0.0)
        self.assertEqual(meta["z_max"], 1.0)
        self.assertFalse(meta["binarize"])
        self.assertEqual(value_meta["value_encoding"], "float32_le")
        self.assertEqual(value_meta["value_count"], 6)
        np.testing.assert_allclose(values, np.array([[0, 0.5, 1], [0.25, 0.75, 1]], dtype=np.float32))

    def test_explicit_binary_uses_legacy_binarized_intensity_mode(self) -> None:
        csv = b"-20,0,20\n5,11,-11\n"

        png, meta, value_bytes, value_meta = table_to_heatmap_payload(
            csv,
            config={
                "heatmap": {
                    "table_mode": "binary",
                    "lower": -10,
                    "upper": 10,
                    "origin": "upper",
                    "cmap": "gray",
                }
            },
            filename_hint="DCPM2-DIES-CD1-CON-1-BH_Mean_intensities_Vertical_Edge_Filtered.csv",
        )

        image = Image.open(io.BytesIO(png)).convert("RGBA")
        first_row = [image.getpixel((x, 0))[0] for x in range(3)]
        second_row = [image.getpixel((x, 1))[0] for x in range(3)]
        values = np.frombuffer(value_bytes, dtype="<f4").reshape((2, 3))

        self.assertEqual(meta["resolved_table_mode"], "extreme_mask")
        self.assertTrue(meta["binarize"])
        self.assertEqual(first_row, [255, 0, 255])
        self.assertEqual(second_row, [0, 255, 255])
        self.assertEqual(value_meta["resolved_table_mode"], "extreme_mask")
        np.testing.assert_allclose(values, np.array([[1, 0, 1], [0, 1, 1]], dtype=np.float32))

    def test_standard_auto_uses_plasma_continuous_default(self) -> None:
        config = yaml.safe_load(Path("configs/default.yaml").read_text(encoding="utf-8"))

        png, meta, value_bytes, value_meta = table_to_heatmap_payload(
            b"0,0.5,1\n",
            config=config,
            filename_hint="mean_intensities.csv",
        )
        image = Image.open(io.BytesIO(png)).convert("RGBA")
        values = np.frombuffer(value_bytes, dtype="<f4")

        self.assertEqual(meta["resolved_table_mode"], "continuous")
        self.assertFalse(meta["binarize"])
        self.assertEqual(meta["cmap"], "plasma")
        self.assertEqual(value_meta["resolved_table_mode"], "continuous")
        np.testing.assert_allclose(values, np.array([0.0, 0.5, 1.0], dtype=np.float32))
        np.testing.assert_allclose(image.getpixel((0, 0)), (13, 8, 135, 255), atol=1)
        np.testing.assert_allclose(image.getpixel((2, 0)), (240, 249, 33, 255), atol=1)

    def test_api_timestamp_serialization_marks_utc_explicitly(self) -> None:
        naive_utc = datetime(2026, 8, 6, 20, 8, 21)
        local_offset = timezone(timedelta(hours=-5))
        aware_local = datetime(2026, 8, 6, 15, 8, 21, tzinfo=local_offset)

        self.assertEqual(utc_isoformat(naive_utc), "2026-08-06T20:08:21Z")
        self.assertEqual(utc_isoformat(aware_local), "2026-08-06T20:08:21Z")
        self.assertEqual(_json_safe({"updated_at": naive_utc}), {"updated_at": "2026-08-06T20:08:21Z"})

        job = JobRead(
            id=uuid4(),
            owner_session_id=uuid4(),
            run_name="timezone test",
            status=JobStatus.completed,
            cancel_requested=False,
            error=None,
            error_code=None,
            created_at=naive_utc,
            started_at=None,
            finished_at=aware_local,
            updated_at=naive_utc,
            progress={},
            tracks_total=None,
            tracks_done=0,
            waves_done=0,
            peaks_done=0,
        )

        dumped = job.model_dump(mode="json")
        self.assertEqual(dumped["created_at"], "2026-08-06T20:08:21Z")
        self.assertIsNone(dumped["started_at"])
        self.assertEqual(dumped["finished_at"], "2026-08-06T20:08:21Z")
        self.assertEqual(dumped["updated_at"], "2026-08-06T20:08:21Z")

    def test_crossing_tracker_honors_cancel_callback(self) -> None:
        tracker = CrossingTracker(object(), max_iters=1)
        raw = np.zeros((5, 5), dtype=np.uint8)
        skel = np.zeros((5, 5), dtype=np.uint8)
        skel[1:4, 2] = 1

        with self.assertRaises(CancellationRequested):
            tracker.extract_tracks(raw, skel, cancel_cb=lambda: True)

    def test_heatmap_converters_honor_cancel_callback(self) -> None:
        with self.assertRaises(CancellationRequested):
            table_to_heatmap_bytes(b"0,1\n", cancel_cb=lambda: True)

        buf = io.BytesIO()
        Image.new("RGB", (2, 2), color=(255, 255, 255)).save(buf, format="PNG")
        with self.assertRaises(CancellationRequested):
            image_to_heatmap_bytes(buf.getvalue(), cancel_cb=lambda: True)

    def test_ripple_analysis_groups_directional_families_and_measures_line_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "ripple_sample" / "kymobutler_output"
            paths = [
                _synthetic_ripple_track(base, 0, slope=1.0, intercept=0.0),
                _synthetic_ripple_track(base, 1, slope=1.0, intercept=-20.0),
                _synthetic_ripple_track(base, 2, slope=1.0, intercept=-40.0),
                _synthetic_ripple_track(base, 3, slope=-1.0, intercept=200.0),
                _synthetic_ripple_track(base, 4, slope=-1.0, intercept=220.0),
            ]
            stages: list[str] = []
            result = analyze_ripple_tracks(
                job_id=uuid4(),
                track_paths=paths,
                config={
                    "io": {"sampling_rate": 5.0},
                    "kymo": {"backend": "onnx", "track_xy_order": "yx"},
                    "analysis": {
                        "mode": "ripple_family",
                        "ripple": {
                            "min_track_rows": 30,
                            "min_abs_slope": 0.05,
                            "max_line_rmse_px": 2.0,
                            "family": {
                                "min_tracks": 2,
                                "max_angle_delta_deg": 5.0,
                                "min_x_overlap_px": 20.0,
                                "min_x_overlap_fraction": 0.1,
                                "max_reference_gap_frames": 100.0,
                            },
                            "frequency": {
                                "sample_count": 11,
                                "min_period_frames": 5.0,
                                "max_period_frames": 50.0,
                                "max_gap_cv": 0.1,
                            },
                        },
                    },
                },
                progress_cb=lambda stage, _processed, _total: stages.append(stage),
            )

        self.assertEqual(len(result.track_rows), 5)
        self.assertEqual(len(result.families), 2)
        self.assertEqual({row["direction"] for row in result.families}, {"positive", "negative"})
        self.assertEqual(len(result.intervals), 3)
        for interval in result.intervals:
            self.assertAlmostEqual(interval["period_frames"], 20.0, places=6)
            self.assertAlmostEqual(interval["period_s"], 4.0, places=6)
            self.assertAlmostEqual(interval["frequency_hz"], 0.25, places=6)
        self.assertIn("ripple_track_geometry", stages)
        self.assertIn("ripple_family_grouping", stages)
        self.assertIn("ripple_interval_analysis", stages)
        self.assertIn(b"family_id", result.tracks_csv)
        self.assertIn(b"frequency_hz", result.intervals_csv)

        track_csv_rows = list(csv.DictReader(io.StringIO(result.tracks_csv.decode("utf-8"))))
        interval_csv_rows = list(csv.DictReader(io.StringIO(result.intervals_csv.decode("utf-8"))))
        family_csv_rows = list(csv.DictReader(io.StringIO(result.families_csv.decode("utf-8"))))

        self.assertIn("Velocity (pixels/sec)", track_csv_rows[0])
        self.assertIn("Speed (pixels/sec)", track_csv_rows[0])
        self.assertIn("Angle from Time Axis (degrees)", track_csv_rows[0])
        self.assertAlmostEqual(float(track_csv_rows[0]["Velocity (pixels/sec)"]), 5.0, places=6)
        self.assertAlmostEqual(float(track_csv_rows[0]["Speed (pixels/sec)"]), 5.0, places=6)
        self.assertAlmostEqual(float(track_csv_rows[0]["Angle from Time Axis (degrees)"]), 45.0, places=6)

        self.assertIn("Velocity (pixels/sec)", interval_csv_rows[0])
        self.assertIn("Speed (pixels/sec)", interval_csv_rows[0])
        self.assertIn("Angle from Time Axis (degrees)", interval_csv_rows[0])
        self.assertTrue(all(abs(float(row["Speed (pixels/sec)"]) - 5.0) < 1e-6 for row in interval_csv_rows))
        self.assertIn("Median Velocity (pixels/sec)", family_csv_rows[0])
        self.assertIn("Median Angle from Time Axis (degrees)", family_csv_rows[0])

    def test_ripple_direction_uses_bottom_left_frame_axis_while_overlay_uses_image_rows(self) -> None:
        height = 121
        image_row = np.arange(height, dtype=float)
        scientific_frame = (height - 1.0) - image_row
        position = 20.0 + 0.5 * scientific_frame

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "ripple_sample" / "kymobutler_output"
            base.mkdir(parents=True)
            track_path = base / "track_0.npy"
            np.save(track_path, np.column_stack([image_row, position]))
            result = analyze_ripple_tracks(
                job_id=uuid4(),
                track_paths=[track_path],
                config={
                    "io": {"sampling_rate": 5.0},
                    "kymo": {"backend": "wolfram"},
                    "analysis": {
                        "mode": "ripple_family",
                        "ripple": {
                            "min_track_rows": 30,
                            "min_abs_slope": 0.05,
                            "max_line_rmse_px": 2.0,
                        },
                    },
                },
                heatmap_meta={"output_height": height, "coord_origin": "lower"},
            )

        metrics = result.track_rows[0]["metrics"]
        self.assertEqual(metrics["direction"], "positive")
        self.assertAlmostEqual(metrics["slope_px_per_frame"], 0.5, places=6)
        self.assertAlmostEqual(metrics["velocity_px_per_s"], 2.5, places=6)
        self.assertEqual(result.overlay_events[0]["poly"][0]["y"], height - 1)
        self.assertEqual(result.overlay_events[0]["poly"][-1]["y"], 0.0)

    def test_ripple_analysis_honors_cancellation(self) -> None:
        with self.assertRaises(CancellationRequested):
            analyze_ripple_tracks(
                job_id=uuid4(),
                track_paths=[],
                config={"analysis": {"mode": "ripple_family"}},
                cancel_cb=lambda: True,
            )

    def test_ripple_extraction_dedupe_collapses_near_cross_phase_duplicates(self) -> None:
        rows = np.arange(0, 140, dtype=np.float32)
        base = np.column_stack([rows, 0.45 * rows + 20.0]).astype(np.float32)
        duplicate = np.column_stack([rows, 0.45 * rows + 22.0]).astype(np.float32)
        neighboring_ripple = np.column_stack([rows, 0.45 * rows + 42.0]).astype(np.float32)

        traces = [
            _Trace("bright", base, 0.95, "guided", phase_contrast=0.09),
            _Trace("dark", duplicate, 0.90, "linear", phase_contrast=0.08),
            _Trace("bright", neighboring_ripple, 0.92, "guided", phase_contrast=0.09),
        ]

        deduped = _dedupe_and_extend(
            traces,
            _ripple_dedupe_test_config(),
            cancel_cb=None,
        )

        self.assertEqual(len(deduped), 2)
        x_at_midpoint = sorted(
            float(np.interp(70.0, trace.points[:, 0], trace.points[:, 1]))
            for trace in deduped
        )
        self.assertGreater(x_at_midpoint[1] - x_at_midpoint[0], 15.0)

    def test_ripple_extraction_dedupe_collapses_same_length_close_overlaps(self) -> None:
        rows = np.arange(0, 150, dtype=np.float32)
        base = np.column_stack([rows, 0.5 * rows + 18.0]).astype(np.float32)
        duplicate = np.column_stack([rows, 0.5 * rows + 23.0]).astype(np.float32)
        neighboring_ripple = np.column_stack([rows, 0.5 * rows + 36.0]).astype(np.float32)

        deduped = _dedupe_and_extend(
            [
                _Trace("bright", base, 0.94, "guided", phase_contrast=0.10),
                _Trace("bright", duplicate, 0.91, "linear", phase_contrast=0.08),
                _Trace("bright", neighboring_ripple, 0.92, "guided", phase_contrast=0.10),
            ],
            _ripple_dedupe_test_config(),
            cancel_cb=None,
        )

        self.assertEqual(len(deduped), 2)
        x_at_midpoint = sorted(
            float(np.interp(75.0, trace.points[:, 0], trace.points[:, 1]))
            for trace in deduped
        )
        self.assertGreater(x_at_midpoint[1] - x_at_midpoint[0], 10.0)

    def test_ripple_extraction_dedupe_prefers_support_over_length_tie(self) -> None:
        rows = np.arange(0, 150, dtype=np.float32)
        lower_support = np.column_stack([rows, 0.5 * rows + 18.0]).astype(np.float32)
        higher_support = np.column_stack([rows, 0.5 * rows + 23.0]).astype(np.float32)

        deduped = _dedupe_and_extend(
            [
                _Trace("bright", lower_support, 0.80, "guided"),
                _Trace("bright", higher_support, 0.95, "linear"),
            ],
            _ripple_dedupe_test_config(),
            cancel_cb=None,
        )

        self.assertEqual(len(deduped), 1)
        self.assertAlmostEqual(
            float(np.interp(75.0, deduped[0].points[:, 0], deduped[0].points[:, 1])),
            60.5,
            places=6,
        )

    def test_ripple_extraction_dedupe_treats_local_spatial_overlap_as_dealbreaker(self) -> None:
        rows = np.arange(0, 150, dtype=np.float32)
        base_x = 0.5 * rows + 18.0
        local_overlap_x = base_x + 18.0
        local_overlap_x[64:82] = base_x[64:82] + 3.0
        neighboring_x = base_x + 36.0

        deduped = _dedupe_and_extend(
            [
                _Trace("bright", np.column_stack([rows, base_x]).astype(np.float32), 0.94, "guided"),
                _Trace("bright", np.column_stack([rows, local_overlap_x]).astype(np.float32), 0.91, "linear"),
                _Trace("bright", np.column_stack([rows, neighboring_x]).astype(np.float32), 0.92, "guided"),
            ],
            _ripple_dedupe_test_config(),
            cancel_cb=None,
        )

        self.assertEqual(len(deduped), 2)

    def test_ripple_extraction_dedupe_rejects_spatial_overlap_across_different_slopes(self) -> None:
        rows = np.arange(0, 150, dtype=np.float32)
        base_x = 0.5 * rows + 18.0
        different_family_x = 0.1 * rows + 46.0
        neighboring_x = base_x + 36.0

        deduped = _dedupe_and_extend(
            [
                _Trace("bright", np.column_stack([rows, base_x]).astype(np.float32), 0.94, "guided"),
                _Trace("bright", np.column_stack([rows, different_family_x]).astype(np.float32), 0.91, "linear"),
                _Trace("bright", np.column_stack([rows, neighboring_x]).astype(np.float32), 0.92, "guided"),
            ],
            _ripple_dedupe_test_config(),
            cancel_cb=None,
        )

        self.assertEqual(len(deduped), 2)

    def test_ripple_extraction_dedupe_collapses_braided_local_overlaps(self) -> None:
        rows = np.arange(0, 150, dtype=np.float32)
        base_x = 0.5 * rows + 18.0
        duplicate_x = base_x + 4.5
        duplicate_x[114:] += 8.0
        neighboring_x = base_x + 22.0
        traces = [
            _Trace("bright", np.column_stack([rows, base_x]).astype(np.float32), 0.94, "guided", phase_contrast=0.10),
            _Trace(
                "bright",
                np.column_stack([rows, duplicate_x]).astype(np.float32),
                0.91,
                "linear",
                phase_contrast=0.08,
            ),
            _Trace(
                "bright",
                np.column_stack([rows, neighboring_x]).astype(np.float32),
                0.92,
                "guided",
                phase_contrast=0.10,
            ),
        ]

        deduped = _dedupe_and_extend(
            traces,
            _ripple_dedupe_test_config(),
            cancel_cb=None,
        )

        self.assertEqual(len(deduped), 2)

    def test_ripple_extraction_dedupe_keeps_adjacent_parallel_ripples(self) -> None:
        rows = np.arange(0, 150, dtype=np.float32)
        traces = [
            _Trace("bright", np.column_stack([rows, 0.5 * rows + 18.0]).astype(np.float32), 0.94, "guided"),
            _Trace("bright", np.column_stack([rows, 0.5 * rows + 31.0]).astype(np.float32), 0.94, "guided"),
        ]

        deduped = _dedupe_and_extend(
            traces,
            _ripple_dedupe_test_config(),
            cancel_cb=None,
        )

        self.assertEqual(len(deduped), 2)

    def test_ripple_pipeline_uses_ripple_extractor_with_continuous_heatmap_values(self) -> None:
        engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        SQLModel.metadata.create_all(engine)

        captured: dict[str, object] = {}
        old_extractor = pipeline_module.run_ripple_extraction

        def fake_ripple_extraction(**kwargs):
            heatmap_value_bytes = kwargs.get("heatmap_value_bytes")
            captured["value_bytes_len"] = len(heatmap_value_bytes or b"")
            captured["value_meta"] = dict(kwargs.get("heatmap_value_meta") or {})
            base_dir = Path(kwargs["scratch_dir"]) / Path(kwargs["heatmap_path"]).stem
            output_dir = base_dir / "kymobutler_output"
            paths = [
                _synthetic_ripple_track(output_dir, 0, slope=0.5, intercept=10.0),
                _synthetic_ripple_track(output_dir, 1, slope=0.5, intercept=0.0),
            ]
            (base_dir / "ripple_track_manifest.json").write_text(
                json.dumps({
                    "extractor": "ripple_multiscale_hough",
                    "value_source": "continuous_table_values",
                    "track_count": len(paths),
                }),
                encoding="utf-8",
            )
            return SimpleNamespace(
                image_id=Path(kwargs["heatmap_path"]).stem,
                base_dir=base_dir,
                track_paths=paths,
                track_metadata={},
            )

        try:
            pipeline_module.run_ripple_extraction = fake_ripple_extraction
            with tempfile.TemporaryDirectory() as tmp:
                with Session(engine) as session:
                    store = JobStore(session)
                    artifact_store = LocalArtifactStore(str(Path(tmp) / "artifacts"))
                    config = {
                        "io": {"sampling_rate": 5.0},
                        "kymo": {"backend": "onnx", "track_xy_order": "yx"},
                        "heatmap": {
                            "table_mode": "area",
                            "origin": "lower",
                            "area": {"cmap": "gray", "vmin": 0.0, "vmax": 1.0},
                        },
                        "analysis": {
                            "mode": "ripple_family",
                            "ripple": {
                                "min_track_rows": 10,
                                "min_abs_slope": 0.05,
                                "max_line_rmse_px": 1.0,
                                "family": {
                                    "min_tracks": 2,
                                    "max_angle_delta_deg": 5.0,
                                    "min_x_overlap_px": 5.0,
                                    "min_x_overlap_fraction": 0.1,
                                    "max_reference_gap_frames": 50.0,
                                },
                                "frequency": {
                                    "sample_count": 7,
                                    "min_period_frames": 3.0,
                                    "max_period_frames": 40.0,
                                    "max_gap_cv": 0.1,
                                },
                            },
                        },
                        "overlay": {"max_points": 50},
                        "track_detail": {"store_npy": True},
                        "service": {"resume": {"enabled": True}},
                    }
                    job = store.create_job(owner_session_id=uuid4(), run_name="ripple pipeline", config=config)
                    table = "\n".join(
                        ",".join(f"{((x + y) % 17) / 16:.3f}" for x in range(32))
                        for y in range(32)
                    ).encode("utf-8")
                    blob_path, byte_size = artifact_store.put_bytes(
                        job_id=job.id,
                        kind=ArtifactKind.upload_csv.value,
                        filename="area.csv",
                        data=table,
                        content_type="text/csv",
                        label="upload",
                    )
                    store.create_artifact(
                        job_id=job.id,
                        kind=ArtifactKind.upload_csv,
                        blob_path=blob_path,
                        label="upload",
                        content_type="text/csv",
                        byte_size=byte_size,
                        meta={"filename": "area.csv", "input_type": "table"},
                    )

                    pipeline_module.run_job(
                        job.id,
                        job_store=store,
                        artifact_store=artifact_store,
                        config=config,
                        settings=PipelineSettings(scratch_root=Path(tmp) / "scratch"),
                    )

                    finished = store.get_job(job.id)
                    artifacts = list(session.exec(select(Artifact).where(Artifact.job_id == job.id)).all())
                    labels = {artifact.label for artifact in artifacts}
                    filename_by_label = {
                        artifact.label: (artifact.meta or {}).get("filename")
                        for artifact in artifacts
                        if artifact.label
                    }
                    tracks = list(session.exec(select(TrackModel).where(TrackModel.job_id == job.id)).all())

            self.assertEqual(finished.status, JobStatus.completed)
            self.assertGreater(int(captured["value_bytes_len"]), 0)
            self.assertEqual((captured["value_meta"] or {}).get("value_encoding"), "float32_le")
            self.assertIn("ripple_tracks", labels)
            self.assertIn("ripple_intervals", labels)
            self.assertIn("ripple_families", labels)
            self.assertEqual(filename_by_label["ripple_tracks"], "tracks.csv")
            self.assertEqual(filename_by_label["ripple_intervals"], "waves.csv")
            self.assertEqual(filename_by_label["ripple_families"], "families.csv")
            self.assertIn("base_heatmap:ripple_track_manifest", labels)
            self.assertEqual(len(tracks), 2)
            self.assertTrue(all(track.metrics.get("analysis_mode") == "ripple_family" for track in tracks))
        finally:
            pipeline_module.run_ripple_extraction = old_extractor

    def test_large_wave_pipeline_skips_standard_track_analysis(self) -> None:
        engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        SQLModel.metadata.create_all(engine)

        old_extractor = pipeline_module.run_large_wave_extraction
        old_process_track = pipeline_module.process_track
        old_prepare_track = pipeline_module.prepare_large_wave_track
        prepared_calls = 0

        def fake_large_wave_extraction(**kwargs):
            frame = np.arange(0, 201, dtype=float)
            shape = 12.0 * np.exp(-0.5 * ((frame - 100.0) / 22.0) ** 2)
            output_dir = Path(kwargs["scratch_dir"]) / "large_wave" / "kymobutler_output"
            output_dir.mkdir(parents=True)
            track_path = output_dir / "track_0000.npy"
            np.save(track_path, np.column_stack([frame, 30.0 + 0.05 * frame + shape]))
            return SimpleNamespace(
                image_id="large_wave",
                base_dir=output_dir.parent,
                track_paths=[track_path],
                track_metadata={},
            )

        def fail_standard_analysis(**_kwargs):
            raise AssertionError("large-wave mode invoked standard process_track")

        def counting_prepare_track(**kwargs):
            nonlocal prepared_calls
            prepared_calls += 1
            return old_prepare_track(**kwargs)

        class InterruptibleArtifactStore(LocalArtifactStore):
            interrupt_store: JobStore | None = None
            interrupt_job_id = None
            interrupt_next_publication = False

            def put_bytes(self, **kwargs):
                result = super().put_bytes(**kwargs)
                label = str(kwargs.get("label") or "")
                if self.interrupt_next_publication and "-staging-" in label:
                    self.interrupt_next_publication = False
                    assert self.interrupt_store is not None
                    assert self.interrupt_job_id is not None
                    self.interrupt_store.request_cancel(self.interrupt_job_id, emit_event=False)
                return result

        try:
            pipeline_module.run_large_wave_extraction = fake_large_wave_extraction
            pipeline_module.process_track = fail_standard_analysis
            pipeline_module.prepare_large_wave_track = counting_prepare_track
            with tempfile.TemporaryDirectory() as tmp:
                with Session(engine) as session:
                    store = JobStore(session)
                    artifact_store = InterruptibleArtifactStore(str(Path(tmp) / "artifacts"))
                    config = _base_config(event_polarity="both")
                    config.update({
                        "heatmap": {
                            "table_mode": "area",
                            "origin": "lower",
                            "area": {"cmap": "gray", "vmin": 0.0, "vmax": 1.0},
                        },
                        "analysis": {
                            "mode": "large_wave",
                            "large_wave": {
                                "peaks": {
                                    "event_polarity": "both",
                                    "adaptive": False,
                                    "minimum_per_track": 1,
                                    "minimum_scope": "track",
                                    "prominence": 2.0,
                                    "width": 5,
                                    "distance": 30,
                                },
                                "events": {
                                    "min_tracks": 1,
                                    "min_amplitude_px": 1.0,
                                    "min_prominence_px": 1.0,
                                    "min_width_frames": 2.0,
                                    "fit_boundary_smoothing_sigma_rows": 1.0,
                                    "endpoint_anchor_rows": 3,
                                },
                            },
                        },
                        "overlay": {"max_points": 50},
                        "track_detail": {"store_npy": True},
                        "service": {"resume": {"enabled": True}},
                    })
                    job = store.create_job(owner_session_id=uuid4(), run_name="large wave pipeline", config=config)
                    table = "\n".join(
                        ",".join(f"{((x + y) % 17) / 16:.3f}" for x in range(32))
                        for y in range(32)
                    ).encode("utf-8")
                    blob_path, byte_size = artifact_store.put_bytes(
                        job_id=job.id,
                        kind=ArtifactKind.upload_csv.value,
                        filename="area.csv",
                        data=table,
                        content_type="text/csv",
                        label="upload",
                    )
                    store.create_artifact(
                        job_id=job.id,
                        kind=ArtifactKind.upload_csv,
                        blob_path=blob_path,
                        label="upload",
                        content_type="text/csv",
                        byte_size=byte_size,
                        meta={"filename": "area.csv", "input_type": "table"},
                    )

                    pipeline_module.run_job(
                        job.id,
                        job_store=store,
                        artifact_store=artifact_store,
                        config=config,
                        settings=PipelineSettings(scratch_root=Path(tmp) / "scratch"),
                    )

                    first_measurements = store.list_artifacts(
                        job.id,
                        kind=ArtifactKind.other,
                        label="large_wave_measurements",
                        limit=2,
                    )
                    self.assertEqual(len(first_measurements), 1)
                    first_blob_path = first_measurements[0].blob_path
                    first_csv = artifact_store.get_bytes(first_blob_path)
                    first_wave_count = len(
                        session.exec(select(Wave).where(Wave.job_id == job.id)).all()
                    )
                    files_before_interruption = {
                        path for path in (Path(tmp) / "artifacts").rglob("*") if path.is_file()
                    }

                    artifact_store.interrupt_store = store
                    artifact_store.interrupt_job_id = job.id
                    artifact_store.interrupt_next_publication = True
                    pipeline_module.run_job(
                        job.id,
                        job_store=store,
                        artifact_store=artifact_store,
                        config=config,
                        settings=PipelineSettings(scratch_root=Path(tmp) / "scratch"),
                        resume=True,
                    )
                    self.assertEqual(store.get_job(job.id).status, JobStatus.cancelled)
                    self.assertTrue(Path(first_blob_path).exists())
                    self.assertEqual(
                        {
                            path
                            for path in (Path(tmp) / "artifacts").rglob("*")
                            if path.is_file()
                        },
                        files_before_interruption,
                    )
                    self.assertEqual(
                        len(session.exec(select(Wave).where(Wave.job_id == job.id)).all()),
                        first_wave_count,
                    )

                    store.clear_cancel(job.id, emit_event=False)

                    pipeline_module.run_job(
                        job.id,
                        job_store=store,
                        artifact_store=artifact_store,
                        config=config,
                        settings=PipelineSettings(scratch_root=Path(tmp) / "scratch"),
                        resume=True,
                    )

                    finished = store.get_job(job.id)
                    artifacts = list(session.exec(select(Artifact).where(Artifact.job_id == job.id)).all())
                    tracks = list(session.exec(select(TrackModel).where(TrackModel.job_id == job.id)).all())
                    waves = list(session.exec(select(Wave).where(Wave.job_id == job.id)).all())
                    second_measurements = [
                        artifact
                        for artifact in artifacts
                        if artifact.label == "large_wave_measurements"
                    ]
                    self.assertEqual(len(second_measurements), 1)
                    second_csv = artifact_store.get_bytes(second_measurements[0].blob_path)
                    old_blob_was_removed = not Path(first_blob_path).exists()
                    result_manifests = [
                        artifact for artifact in artifacts if artifact.label == "result_manifest"
                    ]
                    self.assertEqual(len(result_manifests), 1)
                    result_manifest_bytes = artifact_store.get_bytes(
                        result_manifests[0].blob_path
                    )
                    result_manifest = json.loads(result_manifest_bytes.decode("utf-8"))
                    result_manifest_sha256 = (result_manifests[0].meta or {}).get("sha256")
                    expected_input_sha256 = sha256_bytes(table)

            self.assertEqual(finished.status, JobStatus.completed)
            self.assertGreater(finished.waves_done, 0)
            self.assertEqual(finished.peaks_done, 0)
            self.assertEqual(len(tracks), 1)
            self.assertEqual(len(waves), first_wave_count)
            self.assertEqual(tracks[0].metrics.get("analysis_mode"), "large_wave")
            self.assertEqual(prepared_calls, 3)
            self.assertEqual(second_csv, first_csv)
            self.assertTrue(old_blob_was_removed)
            labels = [artifact.label for artifact in artifacts]
            self.assertEqual(labels.count("large_wave_tracks"), 1)
            self.assertEqual(labels.count("large_wave_measurements"), 1)
            self.assertEqual(labels.count("large_wave_events"), 1)
            self.assertEqual(labels.count("result_manifest"), 1)
            self.assertEqual(result_manifest["schema_version"], 2)
            self.assertEqual(result_manifest["analysis_mode"], "large_wave")
            self.assertEqual(result_manifest["measurement_schema"], measurement_schema_identity())
            self.assertEqual(result_manifest["input"]["sha256"], expected_input_sha256)
            self.assertEqual(result_manifest_sha256, sha256_bytes(result_manifest_bytes))
            self.assertEqual(result_manifest["outputs"]["database"]["waves"]["count"], len(waves))
            self.assertTrue(
                all(len(row["sha256"]) == 64 for row in result_manifest["outputs"]["artifacts"])
            )
            self.assertEqual(
                result_manifest["method"]["config"]["io"]["sampling_rate"],
                10.0,
            )
            self.assertNotIn("sampling_rate", result_manifest["method"]["config"]["period"])
        finally:
            pipeline_module.run_large_wave_extraction = old_extractor
            pipeline_module.process_track = old_process_track
            pipeline_module.prepare_large_wave_track = old_prepare_track

    def test_kymobutler_tiled_inference_honors_cancel_callback(self) -> None:
        class DummyKymo:
            seg_hw = (2, 2)
            tile_stride = 1

        with self.assertRaises(CancellationRequested):
            KymoButlerPT._tile_infer_2d(
                DummyKymo(),
                np.ones((3, 3), dtype=np.float32),
                lambda _: {},
                out_kind="bi",
                cancel_cb=lambda: True,
            )

    def test_cancel_check_reads_current_database_state(self) -> None:
        engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        SQLModel.metadata.create_all(engine)

        with Session(engine) as worker_session:
            worker_store = JobStore(worker_session)
            job = worker_store.create_job(owner_session_id=uuid4(), run_name="cancel test")
            worker_store.get_job(job.id)

            with Session(engine) as request_session:
                JobStore(request_session).request_cancel(job.id)

            self.assertTrue(worker_store.is_cancel_requested(job.id))

    def test_endpoint_link_levels_resolve_from_preset_and_allow_overrides(self) -> None:
        maximal = _flatten_onnx_cfg_for_runner({
            "postproc": {"endpoint_link": {"enabled": True, "level": "maximal"}}
        })
        aggressive = _flatten_onnx_cfg_for_runner({
            "postproc": {"endpoint_link": {"enabled": True, "level": "aggressive"}}
        })
        custom = _flatten_onnx_cfg_for_runner({
            "postproc": {"endpoint_link": {
                "enabled": True,
                "level": "aggressive",
                "max_gap_rows": 42,
                "max_step_dx_px_per_row": 3.5,
            }}
        })

        self.assertEqual(maximal["endpoint_link_max_gap_rows"], 35)
        self.assertEqual(maximal["endpoint_link_max_dx"], 6.0)
        self.assertGreater(aggressive["endpoint_link_max_gap_rows"], maximal["endpoint_link_max_gap_rows"])
        self.assertGreater(aggressive["endpoint_link_max_dx"], maximal["endpoint_link_max_dx"])
        self.assertLess(aggressive["endpoint_link_min_bridge_prob"], maximal["endpoint_link_min_bridge_prob"])
        self.assertEqual(custom["endpoint_link_max_gap_rows"], 42)
        self.assertEqual(custom["endpoint_link_max_step_dx_px_per_row"], 3.5)

    def test_ripple_endpoint_linking_does_not_join_opposite_slopes(self) -> None:
        probability = np.ones((24, 24), dtype=np.float32)
        rising = Track(points=[(y, y) for y in range(0, 11)], id=1)
        falling = Track(points=[(y, 21 - y) for y in range(11, 22)], id=2)

        standard, _ = link_track_endpoints(
            [rising, falling],
            probability,
            max_gap_rows=2,
            max_dx=2.0,
            min_bridge_prob=0.0,
            max_slope_delta=3.0,
            overlap_enabled=False,
        )
        ripple, _ = link_track_endpoints(
            [rising, falling],
            probability,
            max_gap_rows=2,
            max_dx=2.0,
            min_bridge_prob=0.0,
            max_slope_delta=3.0,
            overlap_enabled=False,
            prefer_long_linear=True,
            min_abs_slope=0.05,
        )

        self.assertEqual(len(standard), 1)
        self.assertEqual(len(ripple), 2)

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
        self.assertIn(track_row["metrics"]["detrend"]["method"], {
            "ransac_polynomial",
            "least_squares_polynomial",
        })
        self.assertEqual(
            metrics["detrend_method"],
            track_row["metrics"]["detrend"]["method"],
        )

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

        peak_sets = detect_peak_sets(residual, peaks_cfg, frames_per_period=None)
        by_kind = {peak_set["event_kind"]: peak_set for peak_set in peak_sets}

        self.assertEqual(by_kind["max"]["peaks_idx"].tolist(), [1])
        self.assertEqual(by_kind["min"]["peaks_idx"].tolist(), [5])

    def test_required_fallback_peak_prefers_track_interior(self) -> None:
        signal = np.zeros(101, dtype=float)
        signal[0] = 10.0
        signal[50] = 8.0

        peaks, props = ensure_minimum_peaks(
            signal,
            np.asarray([], dtype=int),
            minimum=1,
            edge_margin=25,
            edge_score_floor=0.1,
        )

        self.assertEqual(peaks.tolist(), [50])
        self.assertEqual(np.asarray(props["fallback_peak"], dtype=bool).tolist(), [True])

    def test_large_wave_track_minimum_creates_one_fallback_across_both_polarities(self) -> None:
        peak_sets = detect_peak_sets(
            np.zeros(101, dtype=float),
            {
                "event_polarity": "both",
                "adaptive": False,
                "minimum_per_track": 1,
                "minimum_scope": "track",
                "prominence": 10.0,
                "width": 5,
                "distance": 30,
                "cross_polarity_min_distance": 30,
                "edge_margin_frames": 30,
            },
            frames_per_period=None,
        )

        detected = [
            (peak_set["event_kind"], int(peak_i), bool(fallback))
            for peak_set in peak_sets
            for peak_i, fallback in zip(
                np.asarray(peak_set["peaks_idx"], dtype=int).tolist(),
                np.asarray(peak_set["peak_props"].get("fallback_peak", []), dtype=bool).tolist(),
            )
        ]

        self.assertEqual(len(detected), 1)
        self.assertTrue(detected[0][2])

    def test_large_wave_multiscale_fit_spans_a_broad_basin_across_a_local_notch(self) -> None:
        frame = np.arange(301, dtype=float)
        baseline = 20.0 + 0.03 * frame
        shape = np.zeros_like(frame)
        lo, center, hi = 40, 120, 250
        shape[lo : center + 1] = np.cos(
            0.5 * np.pi * (center - frame[lo : center + 1]) / (center - lo)
        )
        shape[center : hi + 1] = np.cos(
            0.5 * np.pi * (frame[center : hi + 1] - center) / (hi - center)
        )
        shape *= 15.0
        shape[165:172] -= 18.0

        fit = fit_large_wave(
            frame=frame,
            position=baseline + shape,
            global_baseline=baseline,
            center_idx=center,
            event_kind="max",
            sampling_rate=10.0,
            boundary_smoothing_sigma_rows=2.0,
            boundary_smoothing_scales=[1.0, 2.0, 3.0, 4.0],
            refine_apex=False,
        )

        self.assertIsNotNone(fit)
        assert fit is not None
        self.assertGreaterEqual(fit.metrics["fit_window_width_frames"], 200.0)
        self.assertEqual(fit.metrics["fit_window_selection"], "longest_coherent_multiscale_basin")
        self.assertGreater(fit.metrics["fit_window_candidate_count"], 1)

    def test_large_wave_support_basin_suppresses_opposite_local_wiggle(self) -> None:
        candidates = [
            {
                "peak_i": 146,
                "peak_frame": 146.0,
                "event_kind": "max",
                "fit_window_lo": 61,
                "fit_window_hi": 228,
                "fit_window_width_frames": 167.0,
                "fit_support_window_lo": 59,
                "fit_support_window_hi": 221,
                "amplitude_px": 18.0,
                "fit_error_vnmse": 0.29,
            },
            {
                "peak_i": 195,
                "peak_frame": 195.0,
                "event_kind": "min",
                "fit_window_lo": 189,
                "fit_window_hi": 202,
                "fit_window_width_frames": 13.0,
                "fit_support_window_lo": 59,
                "fit_support_window_hi": 221,
                "amplitude_px": 3.0,
                "fit_error_vnmse": 0.16,
            },
        ]

        selected = _dedupe_track_measurements(
            candidates,
            cfg={
                "min_peak_separation_frames": 30,
                "peak_separation_width_fraction": 0.3,
                "duplicate_window_overlap_fraction": 0.65,
                "duplicate_support_overlap_fraction": 0.8,
            },
            track_frame_min=0.0,
            track_frame_max=292.0,
        )

        self.assertEqual([candidate["peak_i"] for candidate in selected], [146])

    def test_large_wave_peak_selection_suppresses_close_opposite_extrema_and_penalizes_edges(self) -> None:
        candidates = [
            {
                "peak_i": 5,
                "peak_frame": 5.0,
                "event_kind": "max",
                "fit_window_lo": 0,
                "fit_window_hi": 55,
                "fit_window_width_frames": 55.0,
                "amplitude_px": 20.0,
                "fit_error_vnmse": 0.0,
            },
            {
                "peak_i": 35,
                "peak_frame": 35.0,
                "event_kind": "min",
                "fit_window_lo": 10,
                "fit_window_hi": 70,
                "fit_window_width_frames": 60.0,
                "amplitude_px": 15.0,
                "fit_error_vnmse": 0.0,
            },
            {
                "peak_i": 130,
                "peak_frame": 130.0,
                "event_kind": "max",
                "fit_window_lo": 105,
                "fit_window_hi": 155,
                "fit_window_width_frames": 50.0,
                "amplitude_px": 12.0,
                "fit_error_vnmse": 0.0,
            },
        ]

        selected = _dedupe_track_measurements(
            candidates,
            cfg={
                "min_peak_separation_frames": 30,
                "peak_separation_width_fraction": 0.3,
                "duplicate_window_overlap_fraction": 0.65,
                "edge_margin_frames": 40,
                "edge_score_floor": 0.2,
            },
            track_frame_min=0.0,
            track_frame_max=180.0,
        )

        self.assertEqual([row["peak_i"] for row in selected], [35, 130])
        self.assertLess(candidates[0]["peak_selection_edge_weight"], candidates[1]["peak_selection_edge_weight"])

    def test_large_wave_peak_selection_always_keeps_best_available_candidate(self) -> None:
        only_candidate = {
            "peak_i": 2,
            "peak_frame": 2.0,
            "event_kind": "max",
            "fit_window_lo": 0,
            "fit_window_hi": 20,
            "fit_window_width_frames": 20.0,
            "amplitude_px": 1.0,
            "fit_error_vnmse": 1.0,
        }

        selected = _dedupe_track_measurements(
            [only_candidate],
            cfg={"edge_margin_frames": 40, "edge_score_floor": 0.1},
            track_frame_min=0.0,
            track_frame_max=100.0,
        )

        self.assertEqual(selected, [only_candidate])

    def test_large_wave_fallback_produces_a_measured_peak_for_each_track(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_row = np.arange(0, 201, dtype=float)
            frame = 200.0 - image_row
            shape = np.exp(-0.5 * ((frame - 100.0) / 22.0) ** 2)
            position = 30.0 + 0.05 * frame + 0.8 * shape
            track_path = _synthetic_track_path(tmp, position)
            config = _base_config(event_polarity="both")
            config["analysis"] = {
                "mode": "large_wave",
                "large_wave": {
                    "peaks": {
                        "event_polarity": "both",
                        "minimum_per_track": 1,
                        "prominence": 100.0,
                        "width": 5,
                        "distance": 30,
                        "cross_polarity_min_distance": 30,
                        "edge_margin_frames": 40,
                        "edge_score_floor": 0.2,
                    },
                    "events": {
                        "min_tracks": 1,
                        "min_amplitude_px": 3.0,
                        "min_prominence_px": 2.0,
                        "min_width_frames": 5.0,
                        "fit_window_width_multiplier": 1.0,
                        "fit_boundary_smoothing_sigma_rows": 2.0,
                        "endpoint_anchor_rows": 3,
                        "fallback_minimum_width_frames": 30,
                        "min_peak_separation_frames": 30,
                        "peak_separation_width_fraction": 0.3,
                        "duplicate_window_overlap_fraction": 0.65,
                        "edge_margin_frames": 40,
                        "edge_score_floor": 0.2,
                    },
                },
            }
            prepared = prepare_large_wave_track(
                job_id=uuid4(),
                track_index=0,
                track_path=track_path,
                config=config,
                heatmap_meta={"output_height": 500, "coord_origin": "lower"},
            )
            result = analyze_large_wave_events(
                prepared_tracks=[prepared],
                config=config,
            )

        self.assertGreaterEqual(len(prepared.candidate_rows), 1)
        self.assertTrue(any(bool(row["metrics"].get("fallback_peak")) for row in prepared.candidate_rows))
        self.assertGreaterEqual(len(result.measurements), 1)
        self.assertTrue(any(bool(row.get("fallback_peak")) for row in result.measurements))
        measurement = result.measurements[0]
        self.assertAlmostEqual(
            measurement["peak_frame_y_axis"],
            measurement["peak_frame"],
        )
        self.assertAlmostEqual(
            measurement["peak_frame_raw"],
            499.0 - measurement["peak_frame"],
        )
        self.assertGreater(measurement["baseline_slope_px_per_frame"], 0.0)

    def test_large_wave_preparation_preserves_peak_seeds_without_standard_fits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            frame = np.arange(0, 301, dtype=float)
            position = (
                30.0
                + 0.03 * frame
                + 9.0 * np.exp(-0.5 * ((frame - 90.0) / 15.0) ** 2)
                - 7.0 * np.exp(-0.5 * ((frame - 220.0) / 18.0) ** 2)
            )
            track_path = _synthetic_track_path(tmp, position)
            config = _base_config(event_polarity="both")
            large_peaks = {
                "event_polarity": "both",
                "adaptive": False,
                "minimum_per_track": 1,
                "minimum_scope": "track",
                "prominence": 2.0,
                "width": 5,
                "distance": 30,
                "smoothing_sigma_rows": 2.0,
                "cross_polarity_min_distance": 30,
            }
            config["analysis"] = {"mode": "large_wave", "large_wave": {"peaks": large_peaks}}
            standard_config = {**config, "peaks": {**config["peaks"], **large_peaks}}

            _track_row, old_rows, _peak_rows, _overlay = process_track(
                job_id=uuid4(),
                track_index=0,
                track_path=track_path,
                config=standard_config,
            )
            prepared = prepare_large_wave_track(
                job_id=uuid4(),
                track_index=0,
                track_path=track_path,
                config=config,
            )

        old_seeds = [
            (row["metrics"]["peak_i"], row["event_kind"], row["metrics"]["fallback_peak"])
            for row in old_rows
        ]
        new_seeds = [
            (row["metrics"]["peak_i"], row["event_kind"], row["metrics"]["fallback_peak"])
            for row in prepared.candidate_rows
        ]
        self.assertEqual(new_seeds, old_seeds)
        self.assertTrue(prepared.candidate_rows)
        self.assertTrue(all("fit_error_vnmse" not in row["metrics"] for row in prepared.candidate_rows))
        self.assertTrue(all("raw_fit_error_vnmse" not in row["metrics"] for row in prepared.candidate_rows))

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

    def test_large_wave_detail_fit_uses_measured_width_as_minimum(self) -> None:
        residual = np.zeros(700, dtype=float)
        residual[350] = -24.0
        wave = SimpleNamespace(
            metrics={
                "peak_i": 350,
                "event_kind": "min",
                "large_wave_width_frames": 320.0,
            },
            event_kind="min",
            wave_index=1,
        )
        events = _large_wave_peak_events_for_detail([wave], residual)

        self.assertEqual(events[0]["peak_i"], 350)
        self.assertEqual(events[0]["fit_signal_sign"], -1)
        self.assertEqual(events[0]["fit_window_frames"], 320.0)
        self.assertEqual(events[0]["fit_window_lo"], 190)
        self.assertEqual(events[0]["fit_window_hi"], 510)
        self.assertEqual(events[0]["fit_window_source"], "large_wave_baseline_basin")
        fit_result = _fit_anchored_wave_basin(
            -residual,
            np.arange(len(residual), dtype=float),
            350,
            window_lo=int(events[0]["fit_window_lo"]),
            window_hi=int(events[0]["fit_window_hi"]),
        )
        self.assertIsNotNone(fit_result)
        _, fit_meta = fit_result
        self.assertEqual(fit_meta["fit_window_hi"] - fit_meta["fit_window_lo"], 320)
        self.assertEqual(fit_meta["fit_method"], "asymmetric_half_cosine_basin")
        self.assertTrue(fit_meta["fit_passes_peak"])
        self.assertEqual(fit_meta["fit_offset_c"], 0.0)

    def test_large_wave_detail_fit_spans_full_same_side_baseline_basin(self) -> None:
        residual = np.ones(1000, dtype=float)
        residual[300:701] = -2.0
        residual[420] = -5.0
        residual[500] = -12.0
        residual[580] = -4.0
        wave = SimpleNamespace(
            metrics={
                "peak_i": 500,
                "event_kind": "min",
                "large_wave_width_frames": 30.0,
            },
            event_kind="min",
            wave_index=1,
        )

        events = _large_wave_peak_events_for_detail(
            [wave],
            residual,
            boundary_smoothing_sigma_rows=0.0,
        )

        self.assertEqual(events[0]["fit_window_lo"], 300)
        self.assertEqual(events[0]["fit_window_hi"], 700)
        self.assertEqual(events[0]["fit_window_frames"], 400.0)

    def test_large_wave_local_chord_fit_is_invariant_to_linear_track_drift(self) -> None:
        frame = np.arange(0, 201, dtype=float)
        lo, center, hi = 40, 90, 160
        shape = np.zeros_like(frame)
        shape[lo : center + 1] = np.cos(
            0.5 * np.pi * (center - frame[lo : center + 1]) / (center - lo)
        ) ** 1.5
        shape[center : hi + 1] = np.cos(
            0.5 * np.pi * (frame[center : hi + 1] - center) / (hi - center)
        ) ** 0.8
        baseline = 25.0 + 0.2 * frame
        position = baseline + 14.0 * shape

        fit = fit_large_wave(
            frame=frame,
            position=position,
            global_baseline=baseline,
            center_idx=center,
            event_kind="max",
            sampling_rate=10.0,
            fixed_window=(lo, hi),
            endpoint_anchor_rows=3,
            refine_apex=False,
        )
        shifted_baseline = baseline + 30.0 + 0.4 * frame
        shifted_fit = fit_large_wave(
            frame=frame,
            position=position + 30.0 + 0.4 * frame,
            global_baseline=shifted_baseline,
            center_idx=center,
            event_kind="max",
            sampling_rate=10.0,
            fixed_window=(lo, hi),
            endpoint_anchor_rows=3,
            refine_apex=False,
        )

        self.assertIsNotNone(fit)
        self.assertIsNotNone(shifted_fit)
        assert fit is not None and shifted_fit is not None
        self.assertAlmostEqual(fit.metrics["amplitude_px"], 14.0, delta=0.2)
        self.assertAlmostEqual(
            fit.metrics["amplitude_px"], shifted_fit.metrics["amplitude_px"], places=6
        )
        self.assertAlmostEqual(
            fit.metrics["width_half_prominence_frames"],
            shifted_fit.metrics["width_half_prominence_frames"],
            places=6,
        )
        self.assertAlmostEqual(fit.metrics["baseline_slope_px_per_frame"], 0.2, delta=0.01)
        self.assertAlmostEqual(shifted_fit.metrics["baseline_slope_px_per_frame"], 0.6, delta=0.01)
        self.assertEqual(fit.metrics["fit_method"], "asymmetric_half_cosine_local_chord")
        self.assertTrue(fit.metrics["period_estimate_valid"])
        self.assertEqual(fit.metrics["period_source"], "equivalent_sinusoid_from_lobe")
        self.assertAlmostEqual(fit.metrics["period_frames"], 240.0, places=6)
        self.assertAlmostEqual(fit.metrics["period_s"], 24.0, places=6)
        self.assertAlmostEqual(fit.metrics["frequency_hz"], 1.0 / 24.0, places=6)
        self.assertAlmostEqual(fit.metrics["period_from_left_frames"], 200.0, places=6)
        self.assertAlmostEqual(fit.metrics["period_from_right_frames"], 280.0, places=6)
        self.assertAlmostEqual(fit.metrics["period_asymmetry"], 1.0 / 6.0, places=6)
        self.assertAlmostEqual(
            fit.metrics["equivalent_cycle_frame2"] - fit.metrics["equivalent_cycle_frame1"],
            fit.metrics["period_frames"],
            places=6,
        )
        self.assertAlmostEqual(fit.metrics["velocity_px_per_s"], 2.0, delta=0.1)
        self.assertAlmostEqual(shifted_fit.metrics["velocity_px_per_s"], 6.0, delta=0.1)

    def test_large_wave_analysis_exports_one_standard_compatible_row_per_broad_fit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            frame = np.arange(0, 201, dtype=float)
            lo, center, hi = 55, 80, 115
            shape = np.zeros_like(frame)
            shape[lo : center + 1] = np.cos(
                0.5 * np.pi * (center - frame[lo : center + 1]) / (center - lo)
            )
            shape[center : hi + 1] = np.cos(
                0.5 * np.pi * (frame[center : hi + 1] - center) / (hi - center)
            )
            position = 40.0 + 0.15 * frame + 12.0 * shape
            track_path = _synthetic_track_path(tmp, position)
            candidate = {
                "wave_index": 1,
                "event_kind": "max",
                "metrics": {
                    "track_index": 0,
                    "peak_i": center,
                    "event_kind": "max",
                    "bulge_width_frames": 30.0,
                    "bulge_prominence_px": 10.0,
                    "peak_frame_raw": float(center),
                    "peak_frame_y_axis": float(center),
                    "peak_position_raw": float(position[center]),
                    "peak_position_x_axis": float(position[center]),
                },
            }
            duplicate = {
                "wave_index": 2,
                "event_kind": "max",
                "metrics": {**candidate["metrics"], "peak_i": center + 2},
            }
            config = {
                "io": {"sampling_rate": 10.0},
                "kymo": {"backend": "onnx", "track_xy_order": "yx"},
                "detrend": {"degree": 1, "min_samples": 0.5, "random_state": 42},
                "analysis": {
                    "large_wave": {
                        "events": {
                            "min_tracks": 1,
                            "max_peak_frame_gap": 20.0,
                            "min_amplitude_px": 1.0,
                            "min_prominence_px": 1.0,
                            "min_width_frames": 2.0,
                            "fit_window_width_multiplier": 1.0,
                            "fit_boundary_smoothing_sigma_rows": 1.0,
                            "endpoint_anchor_rows": 3,
                        }
                    }
                },
            }

            track_rows = [{"track_index": 0, "metrics": {"num_peaks": 2}}]
            prepared = LargeWavePreparedTrack(
                track_index=0,
                track_path=track_path,
                frame=frame,
                position=position,
                residual=position - (40.0 + 0.15 * frame),
                candidate_rows=[candidate, duplicate],
                track_row=track_rows[0],
                overlay_event={},
            )
            result = analyze_large_wave_events(
                prepared_tracks=[prepared],
                config=config,
            )

        self.assertEqual(len(result.measurements), 1)
        self.assertEqual(len(result.wave_rows), 1)
        self.assertEqual(track_rows[0]["metrics"]["num_peaks"], 1)
        self.assertEqual(track_rows[0]["metrics"]["large_wave_measurement_count"], 1)
        measurement = result.measurements[0]
        self.assertEqual(measurement["fit_method"], "asymmetric_half_cosine_local_chord")
        self.assertAlmostEqual(measurement["amplitude_px"], measurement["fit_amp_A"], places=6)
        self.assertEqual(measurement["period_source"], "equivalent_sinusoid_from_lobe")
        self.assertIsNone(measurement["recurrence_period_frames"])
        self.assertAlmostEqual(
            measurement["frame2_raw"] - measurement["frame1_raw"],
            measurement["period_frames"],
            places=6,
        )
        self.assertAlmostEqual(
            measurement["frame2_time_s"] - measurement["frame1_time_s"],
            measurement["period_s"],
            places=6,
        )
        csv_rows = list(csv.DictReader(io.StringIO(result.measurements_csv.decode("utf-8"))))
        self.assertEqual(len(csv_rows), 1)
        self.assertEqual(list(csv_rows[0].keys())[: len(STANDARD_WAVE_FIELDS)], STANDARD_WAVE_FIELDS)
        self.assertEqual(csv_rows[0]["Fit Target"], "large_wave_local_chord")
        self.assertAlmostEqual(
            float(csv_rows[0]["Amplitude (Pixels)"]), measurement["amplitude_px"], places=6
        )
        self.assertAlmostEqual(
            float(csv_rows[0]["Frame 2 Raw"]) - float(csv_rows[0]["Frame 1 Raw"]),
            float(csv_rows[0]["Period In Frames (Frame 1- Frame 2)"]),
            places=6,
        )
        self.assertEqual(csv_rows[0]["Period Source"], "equivalent_sinusoid_from_lobe")
        self.assertEqual(csv_rows[0]["Recurrence Period (frames)"], "")

    def test_large_wave_equivalent_period_is_blank_for_a_clipped_lobe(self) -> None:
        frame = np.arange(0, 101, dtype=float)
        center, hi = 20, 40
        shape = np.zeros_like(frame)
        shape[: center + 1] = np.cos(
            0.5 * np.pi * (center - frame[: center + 1]) / center
        )
        shape[center : hi + 1] = np.cos(
            0.5 * np.pi * (frame[center : hi + 1] - center) / (hi - center)
        )
        baseline = 20.0 + 0.1 * frame
        fit = fit_large_wave(
            frame=frame,
            position=baseline + 8.0 * shape,
            global_baseline=baseline,
            center_idx=center,
            event_kind="max",
            sampling_rate=5.0,
            fixed_window=(0, hi),
            endpoint_anchor_rows=2,
            refine_apex=False,
        )

        self.assertIsNotNone(fit)
        assert fit is not None
        self.assertFalse(fit.metrics["period_estimate_valid"])
        self.assertIsNone(fit.metrics["period_frames"])
        self.assertIsNone(fit.metrics["frequency_hz"])
        self.assertTrue(fit.metrics["fit_boundary_extrapolated"])

    def test_large_wave_recurrence_metrics_do_not_replace_equivalent_sinusoid_period(self) -> None:
        measurements = [
            {
                "track_index": 0,
                "event_kind": "max",
                "peak_frame": 10.0,
                "period_frames": 100.0,
                "period_s": 20.0,
                "frequency_hz": 0.05,
            },
            {
                "track_index": 0,
                "event_kind": "max",
                "peak_frame": 30.0,
                "period_frames": 100.0,
                "period_s": 20.0,
                "frequency_hz": 0.05,
            },
        ]

        _assign_recurrence_periods(measurements, sampling_rate=5.0)

        for measurement in measurements:
            self.assertEqual(measurement["period_frames"], 100.0)
            self.assertEqual(measurement["frequency_hz"], 0.05)
            self.assertEqual(measurement["recurrence_period_frames"], 20.0)
            self.assertEqual(measurement["recurrence_period_s"], 4.0)
            self.assertEqual(measurement["recurrence_frequency_hz"], 0.25)

    def test_lower_origin_uses_bottom_left_frames_for_wave_motion_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            height = 120
            image_row = np.arange(height, dtype=float)
            frame = (height - 1.0) - image_row
            position = 100.0 + 0.5 * frame + 5.0 * np.sin(2.0 * np.pi * frame / 12.0)
            base = Path(tmp) / "synthetic_sample" / "kymobutler_output"
            base.mkdir(parents=True)
            track_path = base / "0.npy"
            np.save(track_path, np.column_stack([image_row, position]))

            track_row, wave_rows, _peak_rows, overlay = process_track(
                job_id=uuid4(),
                track_index=0,
                track_path=track_path,
                config=_base_config(event_polarity="maxima"),
                heatmap_meta={"output_height": height, "coord_origin": "lower", "pixel_mapping": "table_cell"},
            )

        interior_rows = [
            row for row in wave_rows
            if row["metrics"]["previous_peak_i"] is not None
            and row["metrics"]["next_peak_i"] is not None
            and not row["metrics"]["boundary_extrapolated"]
        ]
        self.assertGreater(len(interior_rows), 0)
        metrics = interior_rows[0]["metrics"]
        self.assertGreater(metrics["delta_pos_px"], 0.1)
        self.assertGreater(metrics["velocity_px_per_s"], 0.0)
        self.assertGreater(metrics["orientation_deg"], 0.0)
        self.assertGreater(metrics["wavelength_px"], 0.1)
        self.assertLess(metrics["frame1_raw"], 120.0)
        self.assertGreaterEqual(metrics["frame1"], 0.0)
        self.assertEqual(track_row["y0"], 0)
        self.assertEqual(overlay["poly"][0]["y"], height - 1)
        self.assertEqual(overlay["poly"][-1]["y"], 0.0)

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

    def test_large_wave_endpoint_linker_accepts_a_supported_smooth_curve(self) -> None:
        probability = np.ones((48, 64), dtype=np.float32)

        def curve_x(y: int) -> int:
            return int(round(20.0 + 0.04 * float(y - 20) ** 2))

        fragments = [
            Track(points=[(y, curve_x(y)) for y in range(0, 13)], id=1),
            Track(points=[(y, curve_x(y)) for y in range(28, 41)], id=2),
        ]
        standard, _ = link_track_endpoints(
            fragments,
            probability,
            max_gap_rows=20,
            max_dx=20.0,
            min_bridge_prob=0.0,
            max_slope_delta=0.2,
            overlap_enabled=False,
        )
        curved, stats = link_track_endpoints(
            fragments,
            probability,
            max_gap_rows=20,
            max_dx=20.0,
            min_bridge_prob=0.0,
            max_slope_delta=0.2,
            overlap_enabled=False,
            prefer_smooth_curves=True,
            curve_max_turn_deg=120.0,
            curve_max_curvature=0.35,
        )
        curved_again, repeated_stats = link_track_endpoints(
            fragments,
            probability,
            max_gap_rows=20,
            max_dx=20.0,
            min_bridge_prob=0.0,
            max_slope_delta=0.2,
            overlap_enabled=False,
            prefer_smooth_curves=True,
            curve_max_turn_deg=120.0,
            curve_max_curvature=0.35,
        )

        self.assertEqual(len(standard), 2)
        self.assertEqual(len(curved), 1)
        self.assertEqual(stats["link_mode"], "smooth_curve")
        self.assertEqual(stats["accepted_links"], 1)
        self.assertEqual(curved[0].points, curved_again[0].points)
        self.assertEqual(stats["manifest"], repeated_stats["manifest"])

    def test_large_wave_endpoint_linker_rejects_adjacent_row_teleport(self) -> None:
        probability = np.ones((50, 700), dtype=np.float32)
        fragments = [
            Track(points=[(y, 592) for y in range(0, 21)], id=1),
            Track(points=[(y, 60) for y in range(21, 41)], id=2),
        ]

        linked, stats = link_track_endpoints(
            fragments,
            probability,
            max_gap_rows=2,
            min_bridge_prob=0.0,
            overlap_enabled=False,
            prefer_smooth_curves=True,
            curve_max_curvature=1e9,
            max_chord_slope_px_per_row=2.0,
            max_step_dx_px_per_row=4.0,
        )

        self.assertEqual(len(linked), 2)
        self.assertEqual(stats["accepted_links"], 0)
        self.assertEqual(stats["manifest"]["rejection_counts"]["hard_max_chord_slope"], 1)

    def test_endpoint_linker_splits_discontinuous_input_tracks(self) -> None:
        probability = np.ones((30, 120), dtype=np.float32)
        corrupted = Track(
            points=[
                *[(y, 10) for y in range(0, 10)],
                *[(y, 100) for y in range(10, 20)],
            ],
            id=1,
        )

        linked, stats = link_track_endpoints(
            [corrupted],
            probability,
            max_gap_rows=2,
            min_bridge_prob=0.0,
            overlap_enabled=False,
            prefer_smooth_curves=True,
            curve_max_curvature=1e9,
            max_chord_slope_px_per_row=2.0,
            max_step_dx_px_per_row=4.0,
        )

        self.assertEqual(stats["input_discontinuity_splits"], 1)
        self.assertEqual(len(linked), 2)
        for track in linked:
            for (y0, x0), (y1, x1) in zip(track.points, track.points[1:]):
                self.assertLessEqual(abs(x1 - x0) / (y1 - y0), 4.0)

    def test_large_wave_extractor_follows_broad_curves_and_chevrons(self) -> None:
        height, width = 240, 180
        rows = np.arange(height, dtype=np.float32)
        cols = np.arange(width, dtype=np.float32)[None, :]
        broad_curve = 45.0 + 12.0 * np.sin(rows / 55.0)
        chevron_curve = 105.0 + 0.28 * np.abs(rows - 120.0)
        values = np.zeros((height, width), dtype=np.float32)
        values -= np.exp(-0.5 * ((cols - broad_curve[:, None]) / 6.0) ** 2)
        values -= 0.8 * np.exp(-0.5 * ((cols - chevron_curve[:, None]) / 5.0) ** 2)
        values += 0.01 * np.sin(cols / 9.0)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            heatmap = tmp_path / "synthetic_heatmap.png"
            rendered = np.asarray(
                255.0 * (values - values.min()) / (values.max() - values.min()),
                dtype=np.uint8,
            )
            Image.fromarray(rendered).save(heatmap)
            config = {
                "analysis": {
                    "mode": "large_wave",
                    "large_wave": {
                        "extraction": {
                            "enabled": True,
                            "normalize_percentiles": [0.0, 100.0],
                            "spatial_sigma_px": 4.0,
                            "background_sigma_px": 14.0,
                            "temporal_sigma_rows": 1.0,
                            "prominence": 0.03,
                            "peak_min_distance_px": 12,
                            "min_component_rows": 40,
                            "min_component_support_fraction": 0.8,
                            "min_track_rows": 80,
                            "max_tracks": 10,
                            "endpoint_bridge": {
                                "max_gap_rows": 15,
                                "min_bridge_response": 0.0,
                                "max_conflict_fraction": 0.05,
                                "max_turn_deg": 150.0,
                                "max_curvature_px_per_row2": 1.0,
                                "max_chord_slope_px_per_row": 2.0,
                                "max_step_dx_px_per_row": 4.0,
                            },
                        },
                        "endpoint_link": {},
                    },
                }
            }
            result = run_large_wave_extraction(
                heatmap_path=heatmap,
                scratch_dir=tmp_path / "scratch",
                config=config,
                heatmap_value_bytes=np.ascontiguousarray(values, dtype="<f4").tobytes(),
                heatmap_value_meta={
                    "source_rows": height,
                    "source_cols": width,
                    "coord_origin": "upper",
                    "value_row_order": "top_to_bottom_source",
                },
            )
            tracks = [np.load(path) for path in result.track_paths]

        long_tracks = [track for track in tracks if len(track) >= 200]
        self.assertGreaterEqual(len(long_tracks), 2)
        errors = []
        for expected in (broad_curve, chevron_curve):
            track_errors = []
            for track in long_tracks:
                y = track[:, 0].astype(int)
                track_errors.append(float(np.median(np.abs(track[:, 1] - expected[y]))))
            errors.append(min(track_errors))
        self.assertTrue(all(error <= 2.0 for error in errors), errors)
        for track in tracks:
            if len(track) < 2:
                continue
            step = np.abs(np.diff(track[:, 1])) / np.diff(track[:, 0])
            self.assertLessEqual(float(np.max(step)), 4.0)

    def test_large_wave_ensemble_recovers_a_sharp_cusp_without_duplicate_broad_track(self) -> None:
        height, width = 220, 620
        rows = np.arange(height, dtype=np.float32)
        cols = np.arange(width, dtype=np.float32)[None, :]
        broad_curve = 90.0 + 8.0 * np.sin(rows / 45.0)
        cusp_curve = 250.0 + 3.0 * np.abs(rows - 110.0)
        values = np.zeros((height, width), dtype=np.float32)
        values -= np.exp(-0.5 * ((cols - broad_curve[:, None]) / 7.0) ** 2)
        values -= 0.9 * np.exp(-0.5 * ((cols - cusp_curve[:, None]) / 4.0) ** 2)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            heatmap = tmp_path / "sharp_cusp_heatmap.png"
            rendered = np.asarray(
                255.0 * (values - values.min()) / (values.max() - values.min()),
                dtype=np.uint8,
            )
            Image.fromarray(rendered).save(heatmap)
            config = {
                "analysis": {
                    "mode": "large_wave",
                    "large_wave": {
                        "extraction": {
                            "normalize_percentiles": [0.0, 100.0],
                            "spatial_sigma_px": 8.0,
                            "background_sigma_px": 24.0,
                            "temporal_sigma_rows": 1.0,
                            "prominence": 0.02,
                            "peak_min_distance_px": 18,
                            "min_component_rows": 50,
                            "min_component_support_fraction": 0.8,
                            "min_track_rows": 60,
                            "track_smoothing_sigma_rows": 1.0,
                            "max_tracks": 20,
                            "ensemble": {
                                "enabled": True,
                                "cusp_spatial_sigma_ratio": 0.4,
                                "min_spatial_sigma_px": 3.0,
                                "max_spatial_sigma_px": 5.0,
                                "temporal_sigma_rows": 0.5,
                                "prominence": 0.02,
                                "peak_min_distance_px": 10,
                                "max_step_dx_px_per_row": 6.0,
                                "min_track_rows": 100,
                                "cusp_half_window_rows": 8,
                                "min_cusp_arm_displacement_px": 12.0,
                                "dedupe_min_overlap_rows": 40,
                                "dedupe_max_distance_px": 8.0,
                                "dedupe_min_close_fraction": 0.8,
                                "dedupe_min_broad_coverage": 0.5,
                            },
                            "endpoint_bridge": {
                                "max_gap_rows": 15,
                                "min_bridge_response": 0.0,
                                "max_conflict_fraction": 0.05,
                                "max_turn_deg": 150.0,
                                "max_curvature_px_per_row2": 1.0,
                                "max_chord_slope_px_per_row": 2.0,
                                "max_step_dx_px_per_row": 4.0,
                            },
                        },
                        "endpoint_link": {},
                    },
                }
            }
            result = run_large_wave_extraction(
                heatmap_path=heatmap,
                scratch_dir=tmp_path / "scratch",
                config=config,
                heatmap_value_bytes=np.ascontiguousarray(values, dtype="<f4").tobytes(),
                heatmap_value_meta={
                    "source_rows": height,
                    "source_cols": width,
                    "coord_origin": "upper",
                    "value_row_order": "top_to_bottom_source",
                },
            )
            tracks = [np.load(path) for path in result.track_paths]
            metadata = list(result.track_metadata.values())

        cusp_tracks = [
            (track, item)
            for track, item in zip(tracks, metadata)
            if item["detector"] == "cusp"
        ]
        self.assertEqual(len(cusp_tracks), 1)
        cusp_track, cusp_metadata = cusp_tracks[0]
        self.assertEqual(len(cusp_track), height)
        y = cusp_track[:, 0].astype(int)
        self.assertLessEqual(float(np.median(np.abs(cusp_track[:, 1] - cusp_curve[y]))), 2.0)
        self.assertGreater(cusp_metadata["cusp_score"], 12.0)
        self.assertEqual(
            sum(
                float(np.median(np.abs(track[:, 1] - cusp_curve[track[:, 0].astype(int)]))) <= 8.0
                for track in tracks
                if len(track) == height
            ),
            1,
        )

    def test_large_wave_conflict_resolver_rejects_overlaps_and_intersections(self) -> None:
        long_track = _RidgeTrace(
            phase="dark",
            track=Track(points=[(y, 20 + y) for y in range(80)], id="long"),
            ridge_strength=0.8,
            support_fraction=1.0,
        )
        overlapping = _RidgeTrace(
            phase="bright",
            track=Track(points=[(y, 22 + y) for y in range(20, 70)], id="overlap"),
            ridge_strength=0.9,
            support_fraction=1.0,
        )
        intersecting = _RidgeTrace(
            phase="dark",
            track=Track(points=[(y, 100 - y) for y in range(30, 71)], id="intersection"),
            ridge_strength=0.9,
            support_fraction=1.0,
        )
        separate = _RidgeTrace(
            phase="bright",
            track=Track(points=[(y, 180 + y) for y in range(60)], id="separate"),
            ridge_strength=0.5,
            support_fraction=1.0,
        )

        kept, summary = _resolve_track_conflicts(
            [overlapping, intersecting, separate, long_track],
            config={
                "min_overlap_rows": 12,
                "max_overlap_distance_px": 3.0,
                "min_close_fraction": 0.6,
                "min_close_run_rows": 8,
                "intersection_tolerance_px": 0.5,
            },
        )

        self.assertEqual({trace.track.id for trace in kept}, {"long", "separate"})
        self.assertEqual(summary["rejected_track_count"], 2)
        self.assertEqual(summary["rejection_counts"], {"overlap": 1, "intersection": 1})

    def test_large_wave_intensity_bias_prefers_high_intensity_ridges(self) -> None:
        row = np.asarray([0.8, 0.9, 0.8, 0.4, 0.0, 0.1, 0.0], dtype=np.float32)
        smooth = np.repeat(row[None, :], 3, axis=0)

        unbiased, _ = _ridge_candidate_masks(
            smooth,
            prominence=0.08,
            distance=1,
            cancel_cb=None,
        )
        biased, _ = _ridge_candidate_masks(
            smooth,
            prominence=0.08,
            distance=1,
            intensity_bias={
                "enabled": True,
                "strength": 1.0,
                "min_prominence_factor": 0.5,
                "max_prominence_factor": 2.0,
            },
            cancel_cb=None,
        )

        self.assertTrue(np.all(unbiased["bright"][:, 1] == 1))
        self.assertTrue(np.all(unbiased["bright"][:, 5] == 1))
        self.assertTrue(np.all(biased["bright"][:, 1] == 1))
        self.assertTrue(np.all(biased["bright"][:, 5] == 0))

    def test_large_wave_intensity_bias_can_select_brighter_conflict_survivor(self) -> None:
        longer_dim = _RidgeTrace(
            phase="bright",
            track=Track(points=[(y, 20 + y) for y in range(80)], id="longer-dim"),
            ridge_strength=0.8,
            support_fraction=1.0,
            intensity_score=0.1,
        )
        shorter_bright = _RidgeTrace(
            phase="bright",
            track=Track(points=[(y, 22 + y) for y in range(70)], id="shorter-bright"),
            ridge_strength=0.8,
            support_fraction=1.0,
            intensity_score=0.9,
        )

        kept, summary = _resolve_track_conflicts(
            [longer_dim, shorter_bright],
            config={
                "min_overlap_rows": 12,
                "max_overlap_distance_px": 3.0,
                "min_close_fraction": 0.6,
                "min_close_run_rows": 8,
                "intersection_tolerance_px": 0.5,
            },
            intensity_weight=0.75,
        )

        self.assertEqual([trace.track.id for trace in kept], ["shorter-bright"])
        self.assertEqual(summary["rejected_track_count"], 1)
        self.assertEqual(summary["intensity_selection_weight"], 0.75)

    def test_measurement_schema_has_stable_unique_identity(self) -> None:
        identity = measurement_schema_identity()

        self.assertEqual(identity["version"], 1)
        self.assertEqual(identity["default_column_labels"], "familiar")
        self.assertEqual(identity["available_column_labels"], ["familiar", "descriptive"])
        self.assertEqual(len(identity["sha256"]), 64)
        self.assertIn("standard_track_spectral_frequency_hz", MEASUREMENT_DEFINITIONS)
        self.assertIn("ripple_intertrack_arrival_rate_hz", MEASUREMENT_DEFINITIONS)
        self.assertIn("large_wave_equivalent_lobe_frequency_hz", MEASUREMENT_DEFINITIONS)

    def test_descriptive_wave_columns_remove_only_legacy_aliases(self) -> None:
        familiar_values = list(range(len(WAVE_EXPORT_FAMILIAR_HEADERS)))
        descriptive_keys = wave_export_descriptive_keys("large_wave")

        familiar_headers, familiar_result = profile_csv_columns(
            WAVE_EXPORT_FAMILIAR_HEADERS,
            descriptive_keys,
            familiar_values,
            "familiar",
        )
        descriptive_headers, descriptive_result = profile_csv_columns(
            WAVE_EXPORT_FAMILIAR_HEADERS,
            descriptive_keys,
            familiar_values,
            "descriptive",
        )

        self.assertEqual(familiar_headers, list(WAVE_EXPORT_FAMILIAR_HEADERS))
        self.assertEqual(familiar_result, familiar_values)
        self.assertEqual(len(descriptive_headers), len(set(descriptive_headers)))
        self.assertEqual(len(descriptive_result), len(descriptive_headers))
        self.assertIn("large_wave_equivalent_lobe_frequency_hz", descriptive_headers)
        self.assertIn("large_wave_local_recurrence_rate_hz", descriptive_headers)

    def test_descriptive_ripple_csv_uses_one_precise_column_per_value(self) -> None:
        familiar = io.StringIO()
        writer = csv.DictWriter(
            familiar,
            fieldnames=[
                "Ripple Wave ID", "interval_index", "family_id", "family_label", "direction",
                "earlier_track_index", "later_track_index", "x_overlap_start_px", "x_overlap_end_px",
                "sample_count", "slope_px_per_frame", "velocity_px_per_s", "speed_px_per_s",
                "angle_deg", "angle_from_time_axis_deg", "period_frames", "period_s", "frequency_hz",
                "gap_mad_frames", "gap_cv", "measurement_method",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "Ripple Wave ID": 1,
            "interval_index": 1,
            "family_id": "family_1",
            "family_label": "Family 1",
            "direction": "right",
            "earlier_track_index": 4,
            "later_track_index": 5,
            "x_overlap_start_px": 10,
            "x_overlap_end_px": 20,
            "sample_count": 25,
            "slope_px_per_frame": 0.5,
            "velocity_px_per_s": 5.0,
            "speed_px_per_s": 5.0,
            "angle_deg": 26.565,
            "angle_from_time_axis_deg": 26.565,
            "period_frames": 20,
            "period_s": 2.0,
            "frequency_hz": 0.5,
            "gap_mad_frames": 1.0,
            "gap_cv": 0.05,
            "measurement_method": "median_shared_x_frame_gap",
        })

        result = list(csv.DictReader(io.StringIO(
            descriptive_ripple_csv(familiar.getvalue().encode("utf-8"), "intervals").decode("utf-8")
        )))

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["ripple_intertrack_interval_s"], "2.0")
        self.assertEqual(result[0]["ripple_intertrack_arrival_rate_hz"], "0.5")
        self.assertEqual(result[0]["ripple_angle_from_time_axis_deg"], "26.565")
        self.assertNotIn("Frequency (Hertz)", result[0])
        self.assertNotIn("angle_deg", result[0])


if __name__ == "__main__":
    unittest.main()
