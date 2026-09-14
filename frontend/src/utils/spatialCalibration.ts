export function buildSpatialCalibrationConfig(value: string): Record<string, unknown> {
  const trimmed = value.trim();
  if (!trimmed) return {};
  const micrometersPerPixel = Number(trimmed);
  if (!Number.isFinite(micrometersPerPixel) || micrometersPerPixel <= 0) return {};
  return {
    io: {
      spatial_calibration: {
        micrometers_per_pixel: micrometersPerPixel,
      },
    },
  };
}

export function spatialCalibrationError(value: string): string | null {
  const trimmed = value.trim();
  if (!trimmed) return null;
  const micrometersPerPixel = Number(trimmed);
  return Number.isFinite(micrometersPerPixel) && micrometersPerPixel > 0
    ? null
    : "Spatial scale must be a positive number.";
}
