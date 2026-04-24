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
  targetWidth: string;
  targetHeight: string;
  loadError: string | null;
};

const EMPTY_STATE: ImagePromptState = {
  visible: false,
  loading: false,
  originalWidth: null,
  originalHeight: null,
  targetWidth: "",
  targetHeight: "",
  loadError: null,
};

function parsePositiveInt(value: string): number | null {
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : null;
}

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
      targetWidth: "",
      targetHeight: "",
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
          targetWidth: String(width),
          targetHeight: String(height),
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
          targetWidth: "",
          targetHeight: "",
          loadError: "Couldn't read the uploaded image size. Enter the processing dimensions manually.",
        });
      });
  }, []);

  const reset = useCallback(() => {
    loadTokenRef.current += 1;
    setState(EMPTY_STATE);
  }, []);

  const setTargetWidth = useCallback((value: string) => {
    setState((current) => ({ ...current, targetWidth: value }));
  }, []);

  const setTargetHeight = useCallback((value: string) => {
    setState((current) => ({ ...current, targetHeight: value }));
  }, []);

  const dimensions = useMemo<ImageProcessingDimensions | null>(() => {
    if (!state.visible || state.loading) return null;
    const width = parsePositiveInt(state.targetWidth);
    const height = parsePositiveInt(state.targetHeight);
    if (width == null || height == null) return null;
    return { width, height };
  }, [state.loading, state.targetHeight, state.targetWidth, state.visible]);

  const helperText = useMemo(() => {
    if (!state.visible) return null;
    if (state.loading) return "Reading image size...";
    if (state.originalWidth != null && state.originalHeight != null) {
      return `Original image: ${state.originalWidth} x ${state.originalHeight}px. Adjust the internal processing size before starting.`;
    }
    return "Provide the internal image size to use for processing.";
  }, [state.loading, state.originalHeight, state.originalWidth, state.visible]);

  const validationError = useMemo(() => {
    if (!state.visible || state.loading) return null;
    if (dimensions) return null;
    return "Enter both width and height in pixels before starting this image run.";
  }, [dimensions, state.loading, state.visible]);

  return {
    imageSizing: state.visible
      ? {
          loading: state.loading,
          originalWidth: state.originalWidth,
          originalHeight: state.originalHeight,
          targetWidth: state.targetWidth,
          targetHeight: state.targetHeight,
          helperText,
          loadError: state.loadError,
          validationError,
          valid: Boolean(dimensions),
          onTargetWidthChange: setTargetWidth,
          onTargetHeightChange: setTargetHeight,
        }
      : null,
    dimensions,
    syncWithFile,
    reset,
  };
}
