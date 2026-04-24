export type ImageProcessingDimensions = {
  width: number;
  height: number;
};

function asRecord(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function deepMerge(base: Record<string, unknown>, override: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = { ...base };
  for (const [key, value] of Object.entries(override)) {
    const baseValue = out[key];
    const baseObject = asRecord(baseValue);
    const overrideObject = asRecord(value);
    out[key] = baseObject && overrideObject ? deepMerge(baseObject, overrideObject) : value;
  }
  return out;
}

export function isImageUploadFile(file: File | null): file is File {
  if (!file) return false;
  if (file.type.startsWith("image/")) return true;
  return /\.(png|jpe?g|tiff?|bmp|webp)$/i.test(file.name);
}

export function readImageDimensions(file: File): Promise<ImageProcessingDimensions> {
  return new Promise((resolve, reject) => {
    const imageUrl = URL.createObjectURL(file);
    const image = new Image();

    image.onload = () => {
      URL.revokeObjectURL(imageUrl);
      resolve({
        width: image.naturalWidth,
        height: image.naturalHeight,
      });
    };

    image.onerror = () => {
      URL.revokeObjectURL(imageUrl);
      reject(new Error("Could not read image dimensions"));
    };

    image.src = imageUrl;
  });
}

export function buildImageProcessingConfig(
  dimensions: ImageProcessingDimensions
): Record<string, Record<string, number>> {
  return {
    image_input: {
      target_width: dimensions.width,
      target_height: dimensions.height,
    },
  };
}

export function mergeRunConfigWithImageProcessing(
  baseConfig: unknown,
  dimensions: ImageProcessingDimensions | null
): unknown {
  if (!dimensions) return baseConfig ?? {};

  const imageConfig = buildImageProcessingConfig(dimensions);
  if (typeof baseConfig === "string") {
    const trimmed = baseConfig.trim();
    if (!trimmed) return imageConfig;

    try {
      const parsed = JSON.parse(trimmed);
      const parsedObject = asRecord(parsed);
      if (parsedObject) {
        return deepMerge(parsedObject, imageConfig);
      }
    } catch {
      // Fall through to YAML-friendly text append.
    }

    return `${trimmed}

image_input:
  target_width: ${dimensions.width}
  target_height: ${dimensions.height}
`;
  }

  const baseObject = asRecord(baseConfig);
  if (baseObject) {
    return deepMerge(baseObject, imageConfig);
  }

  return imageConfig;
}
