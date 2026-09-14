import { useCallback, useMemo, useRef, useState } from "react";

import {
  isImageUploadFile,
  readImageDimensions,
  type ImageProcessingDimensions,
} from "../utils/imageProcessing";

type ImagePromptState = {
  visible: boolean;
  loading: boolean;
  originalWidth: number | null;
  originalHeight: number | null;
  loadError: string | null;
};

const EMPTY_STATE: ImagePromptState = {
  visible: false,
  loading: false,
  originalWidth: null,
  originalHeight: null,
  loadError: null,
};

export function useImageProcessingPrompt() {
  const [state, setState] = useState<ImagePromptState>(EMPTY_STATE);
  const loadTokenRef = useRef(0);

  const syncWithFile = useCallback((file: File | null) => {
    loadTokenRef.current += 1;
    const token = loadTokenRef.current;

    if (!isImageUploadFile(file)) {
      setState(EMPTY_STATE);
      return;
    }

    setState({
      visible: true,
      loading: true,
      originalWidth: null,
      originalHeight: null,
      loadError: null,
    });

    readImageDimensions(file)
      .then(({ width, height }) => {
        if (loadTokenRef.current !== token) return;
        setState({
          visible: true,
          loading: false,
          originalWidth: width,
          originalHeight: height,
          loadError: null,
        });
      })
      .catch(() => {
        if (loadTokenRef.current !== token) return;
        setState({
          visible: true,
          loading: false,
          originalWidth: null,
          originalHeight: null,
          loadError: "Couldn't preview the image size. WaveAtlas will read it when analysis starts.",
        });
      });
  }, []);

  const reset = useCallback(() => {
    loadTokenRef.current += 1;
    setState(EMPTY_STATE);
  }, []);

  const dimensions = useMemo<ImageProcessingDimensions | null>(() => {
    if (state.originalWidth == null || state.originalHeight == null) return null;
    return { width: state.originalWidth, height: state.originalHeight };
  }, [state.originalHeight, state.originalWidth]);

  const helperText = useMemo(() => {
    if (!state.visible) return null;
    if (state.loading) return "Reading image size...";
    if (state.originalWidth != null && state.originalHeight != null) {
      return `Original image: ${state.originalWidth} x ${state.originalHeight}px. Analysis will use this native resolution.`;
    }
    return "Analysis will use the uploaded image at its native resolution.";
  }, [state.loading, state.originalHeight, state.originalWidth, state.visible]);

  return {
    imageSizing: state.visible
      ? {
          loading: state.loading,
          originalWidth: state.originalWidth,
          originalHeight: state.originalHeight,
          helperText,
          loadError: state.loadError,
          valid: !state.loading,
        }
      : null,
    dimensions,
    syncWithFile,
    reset,
  };
}
