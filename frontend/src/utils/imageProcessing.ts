export type ImageProcessingDimensions = {
  width: number;
  height: number;
};

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

export function mergeRunConfigWithImageProcessing(
  baseConfig: unknown,
  _dimensions: ImageProcessingDimensions | null
): unknown {
  // Dimensions are read for user feedback only. Scientific runs use the
  // uploaded raster at native resolution, so no resize config is injected.
  return baseConfig ?? {};
}
