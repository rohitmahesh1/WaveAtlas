export type HeatmapProcessingMode = "auto" | "continuous" | "binary";
export type HeatmapPalette = "default" | "gray" | "plasma" | "hot";

export type HeatmapOptions = {
  processingMode: HeatmapProcessingMode;
  palette: HeatmapPalette;
};

export const DEFAULT_HEATMAP_OPTIONS: HeatmapOptions = {
  processingMode: "auto",
  palette: "plasma",
};

const PALETTE_CMAP: Record<Exclude<HeatmapPalette, "default">, string> = {
  gray: "gray",
  plasma: "plasma",
  hot: "hot",
};

export function buildHeatmapOptionsConfig(options: HeatmapOptions): Record<string, unknown> {
  const heatmap: Record<string, unknown> = {};
  const area: Record<string, unknown> = {};

  if (options.processingMode === "continuous") {
    heatmap.table_mode = "continuous";
    heatmap.binarize = false;
    area.binarize = false;
  } else if (options.processingMode === "binary") {
    heatmap.table_mode = "binary";
    heatmap.binarize = true;
    area.binarize = true;
  }

  if (options.palette !== "default") {
    const cmap = PALETTE_CMAP[options.palette];
    heatmap.cmap = cmap;
    area.cmap = cmap;
  }

  if (Object.keys(area).length > 0) {
    heatmap.area = area;
  }

  return Object.keys(heatmap).length > 0 ? { heatmap } : {};
}
