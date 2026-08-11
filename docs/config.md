# WaveAtlas Config Reference

This page explains the settings in `configs/default.yaml`, what they do, and how to change them.

**How to change the config**
1. Edit `configs/default.yaml` for the default settings used by the backend.
2. Set `PIPELINE_CONFIG_PATH` to point at a different YAML file if you want to swap configs per environment.
3. Use the **Advanced** page in the frontend to paste YAML/JSON and submit overrides at run time.

Notes:
- YAML anchors are used in the default config. For example `io.sampling_rate` is anchored as `&sr` and re-used by `period.sampling_rate`.
- Some sections are currently used by the legacy pipeline or reserved for future use. Those are called out below.

## logging
- `logging.level` (string): Python log level. Typical values: `DEBUG`, `INFO`, `WARNING`, `ERROR`. Change when: you need more/less verbosity for debugging.
- `logging.jsonl_events` (bool): Reserved for JSONL event logging. Not currently used by the new pipeline. Change when: you enable legacy JSONL event logging.

## io
- `io.sampling_rate` (number): Frames per second (or samples per second) used by frequency/period estimation. This value is also referenced by `period.sampling_rate`. Change when: the true sampling rate of your data differs.
- `io.image_globs` (list of strings): File patterns for image inputs in legacy/CLI workflows. Not used by the API pipeline. Change when: you use the legacy CLI with nonstandard image extensions.
- `io.track_glob` (string): File pattern for track `.npy` files in legacy/CLI workflows. Not used by the API pipeline. Change when: your legacy track files use a different naming pattern.
- `io.table_globs` (list of strings): File patterns for table uploads in legacy/CLI workflows. Not used by the API pipeline. Change when: your legacy table inputs use different extensions.

## heatmap
These settings control how tabular data is converted into a heatmap image.
- `heatmap.table_mode` (string): Table rendering mode. `auto` keeps the original intensity-style extreme mask for most tables, but switches files with `Area` in the filename to continuous area rendering.
- `heatmap.lower` (number): Lower bound for “mid-range” intensity values that get zeroed out. Values outside `[lower, upper]` are kept before optional binarization.
- `heatmap.upper` (number): Upper bound for “mid-range” intensity values that get zeroed out.
- `heatmap.binarize` (bool): When true, converts the filtered intensity heatmap into a 0/1 mask. The default is `true`, matching the original intensity behavior.
- `heatmap.origin` (string): Image origin for rendering. Typical values: `lower` or `upper`. Change when: the heatmap appears vertically flipped.
- `heatmap.cmap` (string): Matplotlib colormap name used when rendering the default intensity path. The default is `hot`.
- `heatmap.area` (object): Overrides used for auto-detected area tables. By default, area tables are rendered continuously with no extreme filtering or binarization.
- `heatmap.area.cmap` (string): Matplotlib colormap for area tables. The default is `plasma`, which is close to the Plotly-exported HTML behavior.
- `heatmap.area.vmin` / `heatmap.area.vmax` (number or null): Color scale bounds for area tables. `null` means use the table's own min/max.
Optional (not in the default file):
- `heatmap.dpi` (number): DPI for the rendered heatmap PNG. Change when: you need higher-resolution heatmaps or want smaller files/faster processing.

## image_input
Used when an uploaded image is converted into the canonical base heatmap artifact.
- `image_input.grayscale` (bool): Convert uploaded images to grayscale before tracking. Change when: you want to pass the uploaded RGB image through directly.
- `image_input.target_width` / `image_input.target_height` (int or null): Resize image input to internal processing dimensions. If only one is set, aspect ratio is preserved.
- `image_input.low_hex` / `image_input.high_hex` (hex string or null): Optional color endpoints for mapping a false-color image into intensity. Use these for dark-purple-to-yellow image scales.
- `image_input.invert` (bool): Invert the computed intensity after grayscale conversion.
- `image_input.alpha_background` (hex string): Background color used when compositing transparent image uploads.
Notes:
- When both `low_hex` and `high_hex` are supplied, the grayscale conversion projects each pixel along that color axis. Otherwise it uses standard luminance.

## analysis
Selects the scientific analysis performed after track extraction.
- `analysis.mode` (string): `standard` runs the existing per-track detrending, FFT, and peak analysis. `ripple_family` runs the ripple-wave geometry and family workflow.
- `analysis.ripple.min_track_rows` (int): Minimum temporal span for a track to participate in family grouping.
- `analysis.ripple.min_abs_slope` (number): Minimum absolute slope in pixels per frame for a directional ripple track.
- `analysis.ripple.max_line_rmse_px` (number): Maximum robust line-fit error for family eligibility. All extracted tracks remain visible and exported, even when ineligible.
- `analysis.ripple.extraction.enabled` (bool): Use the ripple-specific multiscale/Hough track extractor for ripple runs. If disabled, ripple runs use the configured KymoButler extraction path and still run ripple family analysis afterward.
- `analysis.ripple.extraction.max_tracks` (int): Maximum number of ripple tracks kept after deduplication and validation.
- `analysis.ripple.extraction.min_abs_slope` / `max_abs_slope` (number): Slope range accepted by the ripple extractor. Increase `min_abs_slope` to ignore near-vertical/stationary texture; lower `max_abs_slope` if steep noise is being traced.
- `analysis.ripple.extraction.temporal_sigma_rows` / `peak_min_distance_px` / `min_phase_contrast` (number): Core smoothing, seed spacing, and final contrast filters for extracted ripple lines.
- `analysis.ripple.extraction.dedupe.allow_cross_phase` (bool): Compare bright and dark candidates when removing near-identical ripple tracks. This catches duplicate ridge centerlines produced by different scales or trace methods while keeping true neighboring bands separate through tight distance thresholds.
- `analysis.ripple.extraction.dedupe.min_overlap_rows` / `min_overlap_fraction` (number): Required shared temporal span before two ripple traces can be treated as duplicates.
- `analysis.ripple.extraction.dedupe.min_partial_overlap_rows` / `min_short_overlap_fraction` (number): Shorter, containment-style duplicate gate for local fragments that mostly ride on top of a longer trace.
- `analysis.ripple.extraction.dedupe.max_slope_delta` / `max_median_distance_px` / `max_p90_distance_px` (number): Line-shape tolerances for duplicate ripple traces. Lower distance values preserve nearby parallel ripples; higher values collapse more overlapping tracks.
- `analysis.ripple.extraction.dedupe.min_spatial_overlap_rows` (number): Hard spatial-overlap gate. When two traces share at least this many close rows, dedupe keeps the better-ranked trace and drops the other, even if they would otherwise land in different families.
- `analysis.ripple.extraction.dedupe.close_distance_px` / `min_close_fraction` / `partial_max_median_distance_px` (number): Local duplicate tolerances. These catch braided/stacked copies without requiring the whole overlap to pass a strict 90th-percentile distance check.
- `analysis.ripple.extraction.dedupe.cross_phase_max_median_distance_px` / `cross_phase_max_p90_distance_px` (number): Extra-tight spatial tolerances for bright-vs-dark dedupe so alternating ripple bands are not merged accidentally.
- `analysis.ripple.extraction.dedupe.cross_phase_min_spatial_overlap_rows` / `cross_phase_close_distance_px` / `cross_phase_min_close_fraction` / `cross_phase_partial_max_median_distance_px` (number): Extra-tight local duplicate tolerances for bright-vs-dark comparisons.
- `analysis.ripple.endpoint_link.prefer_long_linear` (bool): Enables ripple-specific global endpoint matching with preference for longer, straighter tracks.
- `analysis.ripple.endpoint_link.length_weight` / `linearity_weight` (number): Weights applied to anchor length and line-fit quality during ripple endpoint matching.
- `analysis.ripple.family.min_tracks` (int): Minimum number of tracks required for an assigned family ID.
- `analysis.ripple.family.max_angle_delta_deg` (number): Maximum orientation difference between tracks connected in the family graph.
- `analysis.ripple.family.min_x_overlap_px` / `min_x_overlap_fraction` (number): Required spatial overlap between candidate family members.
- `analysis.ripple.family.max_reference_gap_frames` (number): Largest temporal separation allowed between candidate family members.
- `analysis.ripple.frequency.sample_count` (int): Number of shared spatial positions used to estimate each adjacent-track gap.
- `analysis.ripple.frequency.min_period_frames` / `max_period_frames` (number): Accepted temporal-gap range for ripple intervals.
- `analysis.ripple.frequency.max_gap_cv` (number): Maximum robust gap variability accepted for an interval.
Notes:
- Ripple frequency is calculated from the median temporal gap between adjacent, same-direction tracks. The ripple workflow does not run the standard FFT/peak calculations.
- Ripple runs publish separate track, interval, and family CSV artifacts. The interval/waves CSV includes period, frequency, signed velocity, absolute speed, and line angle derived from the neighboring ripple tracks.

## detrend
Used when removing baseline trends from track signals (RANSAC + polynomial fit).
- `detrend.degree` (int): Polynomial degree for baseline fit. Change when: baselines are curved and a higher-degree fit is needed (or overfitting requires lowering).
- `detrend.min_samples` (float): RANSAC min_samples parameter (fraction or absolute count). Change when: the baseline fit is too sensitive to noise (increase) or missing true trend (decrease).
- `detrend.residual_threshold` (number or null): RANSAC residual threshold. `null` lets RANSAC choose. Change when: the baseline fit rejects too many points or includes too many outliers.
- `detrend.random_state` (int): Random seed for RANSAC reproducibility. Change when: you want deterministic results across runs.

## peaks
Used for peak detection on the detrended residual.
- `peaks.event_polarity` (string): Which residual events to extract. Use `both` for maxima and minima, `maxima` for positive peaks only, or `minima` for negative troughs only. Change when: you want a single event polarity or need to compare against older positive-peak-only output.
- `peaks.adaptive` (bool): Use period-aware adaptive peak thresholds instead of the explicit prominence/width/distance values. Change when: tracks vary enough that fixed thresholds miss peaks or over-detect noise.
- `peaks.minimum_per_track` (int): Minimum peak anchors required for each non-empty track. The default keeps one fallback anchor when thresholds miss every peak.
- `peaks.prominence` (number): Minimum peak prominence for detection. Change when: you’re getting too many weak peaks (increase) or missing real peaks (decrease).
- `peaks.width` (number): Minimum peak width (in samples). Change when: peaks are too narrow/wide relative to your signal.
- `peaks.distance` (int): Minimum distance between peaks (in samples). Change when: peaks are too clustered (increase) or too sparse (decrease).
Notes:
- When `peaks.adaptive` is `true`, the pipeline also accepts period-aware keys such as `distance_frac`, `width_frac`, `rel_mad_k`, and `abs_min_prom_px`.

## period
Used for dominant frequency estimation and period calculation.
- `period.sampling_rate` (number): Sampling rate for frequency estimation. In the default config this reuses `io.sampling_rate`. Change when: the actual sampling rate differs from default.
- `period.min_freq` (number): Lower bound on estimated frequency (Hz). Change when: you want to ignore very slow trends or low-frequency noise.
- `period.max_freq` (number): Upper bound on estimated frequency (Hz). Change when: you want to ignore high-frequency noise or constrain expected oscillations.

## viz
Visualization toggles used by the legacy pipeline (not currently used by the new API).
- `viz.enabled` (bool): Master switch for visualization outputs. Change when: you want legacy plots on/off.
- `viz.per_track.detrended_with_peaks` (bool): Per-track plot of detrended signal with peaks. Change when: you need per-track diagnostics.
- `viz.per_track.spectrum` (bool): Per-track frequency spectrum plot. Change when: you want per-track frequency inspection.
- `viz.summary.histograms` (bool): Summary histograms. Change when: you want global distribution plots.
- `viz.wave_windows.save` (bool): Save windowed wave slices. Change when: you want to inspect per-wave windows.
- `viz.wave_windows.max_per_track` (int): Max windows per track. Change when: you want more/less wave windows saved.
- `viz.wave_windows.stride` (int): Window stride in frames. Change when: you want denser/sparser window sampling.
- `viz.hist_bins` (int): Histogram bin count. Change when: you want coarser/finer histograms.
- `viz.dpi` (int): DPI for visualization renders. Change when: you want higher resolution or smaller images.

## features
Controls derived metrics and wave classification.
- `features.fit_target` (string): Primary local sine fit target. `raw_wave` fits the base track position and is the default; `residual` fits the detrended residual. Change when: you want the reported `fit_*` columns to use residual-based fitting for older workflows.
- `features.compare_fit_targets` (bool): Also compute the non-primary fit target and include `raw_fit_*` / `residual_fit_*` metadata. Change when: you want smaller output or faster feature computation and only need the primary fit.
- `features.fit_window_period_frac` (number): Window size (as a fraction of period) used for local fits when computing features. Change when: fits are too noisy (increase) or oversmoothed (decrease).
- `features.classify.ripple_max_deg` (number): Max angle (degrees) to classify a wave as ripple-like. Change when: ripple classification is too strict/lenient.
- `features.classify.surf_min_deg` (number): Min angle (degrees) to classify a wave as surf-like. Change when: surf classification is too strict/lenient.
- `features.classify.prominence_min_px` (number): Minimum peak prominence (in pixels) required for ripple classification. Change when: weak peaks are being misclassified.
Notes:
- Wave rows are peak-centered: each detected peak creates one wave row, with frame windows estimated from neighboring peaks or the global period.
- With the default config, both minima and maxima are extracted, both raw-wave and residual fit metrics are recorded, and the unprefixed `fit_*` fields come from the configured primary target.

## service
Service and pipeline runtime settings.
- `service.partial_every_tracks` (int): Legacy setting for partial results cadence. Not used by the new pipeline. Change when: you enable legacy partial outputs and want a different cadence.
- `service.write_progress_every_secs` (int): Legacy progress write cadence. Not used by the new pipeline (progress is event-driven). Change when: you need more/less frequent legacy progress writes.
- `service.overlay.write_every_tracks` (int): Legacy overlay event cadence. Not used by the new pipeline. Change when: you enable legacy overlays and want a different cadence.
- `service.overlay.format` (string): Legacy overlay format. Not used by the new pipeline. Change when: a legacy consumer expects a specific overlay format.
- `service.resume.enabled` (bool): Enables resume behavior in the new pipeline. Change when: you want to allow resuming cancelled runs (true) or always start fresh (false).
- `service.resume.marker_dir` (string): Legacy resume marker directory. Not used by the new pipeline. Change when: using legacy resume markers.
- `service.resume.progress_file` (string): Legacy resume progress file. Not used by the new pipeline. Change when: using legacy resume progress files.
- `service.resume.safe_skip` (bool): Legacy safe-skip mode. Not used by the new pipeline. Change when: using legacy resume and want safer skipping.
- `service.retention.enabled` (bool): Legacy retention cleanup. Not used by the new pipeline. Change when: you want automatic cleanup of legacy run outputs.
- `service.retention.keep_last_runs` (int): Legacy retention setting. Not used by the new pipeline. Change when: you want to keep more/less legacy runs.
- `service.retention.max_run_age_days` (number or null): Legacy retention setting. Not used by the new pipeline. Change when: you want to expire legacy runs by age.
- `service.artifacts.write_run_json` (bool): Legacy artifact export setting. Not used by the new pipeline. Change when: you want legacy run metadata exported.
- `service.artifacts.write_events_ndjson` (bool): Legacy artifact export setting. Not used by the new pipeline. Change when: you want legacy event logs exported.

## kymo
Controls track extraction (KymoButler).
- `kymo.backend` (string): Backend selection. `onnx` uses the ONNX pipeline, `wolfram` uses WolframScript. Change when: you want to switch between ONNX and Wolfram backends.

### kymo.onnx
These settings feed the ONNX-backed KymoButler runner.
- `kymo.onnx.export_dir` (string or null): Path to ONNX model exports. `null` uses the default resolution path. Change when: your model files live in a custom directory.
- `kymo.onnx.seg_size` (int): Segmentation tile size in pixels. Change when: you need smaller tiles for memory or larger tiles for accuracy.
- `kymo.onnx.force_mode` (string): Force `bi` or `uni` mode; `null` auto-selects. Change when: auto mode selection is wrong for your data.
- `kymo.onnx.fuse_uni_into_bi` (bool): Whether to fuse uni-directional prediction into bi-directional mask. Change when: uni predictions help or hurt the bi mask.
- `kymo.onnx.fuse_uni_weight` (number): Weighting factor used when fusing uni into bi. Change when: the fused mask is too aggressive or too weak.

#### kymo.onnx.thresholds
- `kymo.onnx.thresholds.thr_default` (number): Default probability threshold for mask creation. Change when: the mask is too thick (raise) or too thin (lower).
- `kymo.onnx.thresholds.thr_bi` (number): Threshold override for bi-directional mode. Change when: bi mode needs different sensitivity than default.
- `kymo.onnx.thresholds.thr_uni` (number): Threshold override for uni-directional mode. Change when: uni mode needs different sensitivity than default.

#### kymo.onnx.auto_threshold
Automatically adjusts thresholds when mask coverage is too small/large.
- `kymo.onnx.auto_threshold.enabled` (bool): Enable automatic threshold sweeping. Change when: you want dynamic thresholds or need fixed thresholds for reproducibility.
- `kymo.onnx.auto_threshold.sweep` (list): `[min, max, steps]` for sweep range. Change when: the sweep range doesn’t reach a usable threshold.
- `kymo.onnx.auto_threshold.target_mask_pct` (list): Desired mask coverage range in percent. Change when: you want a denser or sparser mask.
- `kymo.onnx.auto_threshold.trigger_pct` (list): Trigger range for enabling auto-thresholding. Change when: auto-threshold is firing too often or not often enough.

#### kymo.onnx.hysteresis
Hysteresis thresholding for stronger mask continuity.
- `kymo.onnx.hysteresis.enabled` (bool): Enable hysteresis thresholding. Change when: you want smoother masks (enable) or sharper masks (disable).
- `kymo.onnx.hysteresis.low` (number): Low threshold for hysteresis. Change when: you want to keep more faint pixels (lower) or remove noise (raise).
- `kymo.onnx.hysteresis.high` (number): High threshold for hysteresis. Change when: you want to keep fewer weak pixels (raise) or be more permissive (lower).

#### kymo.onnx.morphology
Morphology applied to binary masks.
- `kymo.onnx.morphology.mode` (string): `directional`, `classic`, or `none`. Change when: you need different cleanup behavior for mask artifacts.
- `kymo.onnx.morphology.directional.kv` (int): Vertical kernel size for directional morphology. Change when: you need more/less vertical bridging.
- `kymo.onnx.morphology.directional.kh` (int): Horizontal kernel size for directional morphology. Change when: you need more/less horizontal bridging.
- `kymo.onnx.morphology.directional.diag_bridge` (bool): Whether to bridge diagonals. Change when: diagonal bridges create false connections.
- `kymo.onnx.morphology.classic.kernel` (int): Kernel size for classic morphology. Change when: classic morphology is enabled and needs tuning.

#### kymo.onnx.components
Filters out small connected components.
- `kymo.onnx.components.min_px` (int): Minimum pixel count for a component. Change when: small noise blobs remain (raise) or true small tracks are lost (lower).
- `kymo.onnx.components.min_rows` (int): Minimum vertical span (rows) for a component. Change when: short vertical noise remains (raise) or true short tracks are lost (lower).

#### kymo.onnx.skeleton
Skeletonization and pruning behavior.
- `kymo.onnx.skeleton.method` (string): Skeletonization method (`thin` is used in the current implementation). Change when: you add support for another method.
- `kymo.onnx.skeleton.prune_iters` (int): Pruning iterations for removing short stubs. Change when: too many stubs remain (increase) or tracks get over-pruned (decrease).
- `kymo.onnx.skeleton.keep_ratio` (number): Ratio of skeleton to keep when pruning. Change when: pruning is too aggressive (increase) or too lax (decrease).
- `kymo.onnx.skeleton.keep_min_px` (int): Minimum skeleton size to keep regardless of ratio. Change when: small real structures are dropped or noise is kept.
- `kymo.onnx.skeleton.prob_floor_min` (number): Lower probability floor for skeleton masking. Change when: skeletons include low-probability noise (raise) or lose faint tracks (lower).
- `kymo.onnx.skeleton.prob_floor_max` (number): Upper probability floor for skeleton masking. Change when: skeleton masking is too strict (raise) or too permissive (lower).

#### kymo.onnx.tracking
Tracking/trace extraction settings.
- `kymo.onnx.tracking.min_length` (int): Minimum track length in rows. Change when: short spurious tracks remain (raise) or real short tracks are lost (lower).
- `kymo.onnx.tracking.max_branch_steps` (int): Max steps for exploring branches. Change when: branching search is too shallow (increase) or too slow (decrease).
- `kymo.onnx.tracking.decision_recent_tail` (int): Window size for recent-track decisions. Change when: tracking decisions are too jittery (increase) or too slow to adapt (decrease).

#### kymo.onnx.postproc
Post-processing after skeletonization.
- `kymo.onnx.postproc.extend_rows` (int): Extend tracks by this many rows. Change when: tracks are under-extended (increase) or over-extended (decrease).
- `kymo.onnx.postproc.dx_win` (int): Horizontal window when extending. Change when: extensions should allow more/less horizontal drift.
- `kymo.onnx.postproc.prob_min` (number): Minimum probability for extension. Change when: extension is too permissive (raise) or too strict (lower).
- `kymo.onnx.postproc.max_gap_rows` (int): Max gap (rows) to bridge. Change when: you want to bridge larger gaps (increase) or avoid over-bridging (decrease).
- `kymo.onnx.postproc.max_dx` (int): Max horizontal delta allowed when bridging. Change when: you want to allow more/less horizontal deviation.
- `kymo.onnx.postproc.prob_bridge_min` (number): Minimum probability to bridge gaps. Change when: bridging is too aggressive (raise) or too conservative (lower).
- `kymo.onnx.postproc.endpoint_link.enabled` (bool): Link compatible track endpoints after the normal extension pass. Change when: you want to merge fragmented or slightly overlapping traces (true) or preserve raw extracted fragments (false).
- `kymo.onnx.postproc.endpoint_link.level` (string): Preset strength for endpoint linking. Available levels are `minimal`, `conservative`, `balanced`, `maximal`, `aggressive`, and `experimental`. The default is `maximal`; `aggressive` and `experimental` are more permissive. Change when: links are too sparse or too eager and you want a simpler knob than tuning every value below.
- `kymo.onnx.postproc.endpoint_link.max_gap_rows` (int): Largest row gap considered for endpoint linking. Change when: desired fragments are farther apart or false merges appear over long gaps.
- `kymo.onnx.postproc.endpoint_link.max_dx` (number): Max projected horizontal mismatch for endpoint links. Change when: valid tracks drift more/less between fragments.
- `kymo.onnx.postproc.endpoint_link.min_bridge_prob` (number): Minimum average probability along a proposed link. Change when: links are too permissive (raise) or missed across faint regions (lower).
- `kymo.onnx.postproc.endpoint_link.max_slope_delta` (number): Max slope mismatch between linked tails and heads. Change when: good links bend more sharply or bad links pass through crossings.
- `kymo.onnx.postproc.endpoint_link.fit_rows` (int): Number of rows used to estimate endpoint slope. Change when: slope estimates are too noisy or too slow to react.
- `kymo.onnx.postproc.endpoint_link.max_conflict_fraction` (number): Max fraction of proposed link points that conflict with other tracks. Change when: dense regions need stricter or looser conflict rejection.
- `kymo.onnx.postproc.endpoint_link.insert_bridge_points` (bool): Insert interpolated bridge points for gap links. Change when: you want continuous saved tracks instead of preserving a gap.
- `kymo.onnx.postproc.endpoint_link.overlap_enabled` (bool): Allow endpoint linking through short overlapping track sections. Change when: duplicate/overlapping fragments should be merged.
- `kymo.onnx.postproc.endpoint_link.min_overlap_rows` / `max_overlap_rows` (int): Overlap-size range considered for overlap links. Change when: valid overlaps are shorter or longer.
- `kymo.onnx.postproc.endpoint_link.overlap_dx_tol` (number): Horizontal tolerance for overlap-link consensus. Change when: overlapping fragments are slightly offset or false overlap links appear.
- Endpoint-link presets: `minimal` is shortest/strictest, `conservative` is cautious for dense or noisy images, `balanced` is moderate, `maximal` is the default, `aggressive` goes beyond `maximal` for very fragmented traces, and `experimental` is the most permissive inspection mode.
- Numeric endpoint-link keys can still be set next to `level`; explicit values override the preset for that one run.
- `kymo.onnx.postproc.dedupe.enabled` (bool): Remove duplicate tracks. Change when: duplicates appear (enable) or you want to keep overlapping tracks (disable).
- `kymo.onnx.postproc.dedupe.min_rows` (int): Minimum rows for dedupe consideration. Change when: shorter tracks should be deduped or ignored.
- `kymo.onnx.postproc.dedupe.min_score` (number): Minimum score to keep a track during dedupe. Change when: dedupe is too aggressive (raise) or too lax (lower).
- `kymo.onnx.postproc.dedupe.overlap_iou` (number): IoU threshold for duplicate detection. Change when: duplicates slip through (lower) or real tracks are removed (raise).
- `kymo.onnx.postproc.dedupe.dx_tol` (number): Horizontal tolerance for dedupe alignment. Change when: alignment is too strict or too permissive.

#### kymo.onnx.debug
- `kymo.onnx.debug.save_debug_images` (bool): Saves debug masks and overlays that appear in the UI. Change when: you want UI debug overlays (true) or need to reduce disk usage (false).
- `kymo.onnx.debug.save_overlay_tracks` (bool): Saves the final track overlay image. Change when: you want a compact visual check of final linked tracks (true) or fewer artifacts (false).

### kymo.wolfram
Used only when `kymo.backend` is set to `wolfram`.
- `kymo.wolfram.scripts_dir` (string): Path to WolframScript files. Change when: scripts live in a non-default directory.
- `kymo.wolfram.executable` (string): Wolfram executable name/path. Change when: WolframScript is installed somewhere nonstandard.
- `kymo.wolfram.script_name` (string): Script file to execute from `scripts_dir`. Change when: you want to run a different Wolfram wrapper.
- `kymo.wolfram.min_length` (int): Minimum saved Wolfram track length. Change when: short tracks should be kept or filtered.
- `kymo.wolfram.verbose` (bool): Print Wolfram stdout/stderr. Change when: debugging Wolfram execution.
