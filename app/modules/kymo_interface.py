import subprocess
from pathlib import Path
from typing import Union, List, Optional
import ast
import numpy as np

# Directory containing the Mathematica/WolframScript kymobutler scripts
SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent / "kymobutler_scripts"


def _extract_wolfram_list_block(s: str) -> Optional[str]:
    """
    Return the substring containing the first full Wolfram list that starts at the
    first '{{{' and ends at its matching closing brace (balanced), or None if not found.
    """
    start = s.find("{{{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(s)):
        c = s[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                # include this closing brace
                return s[start : i + 1]
    return None  # unterminated / not balanced


def _parse_mathematica_arrays(stdout_text: str) -> List[np.ndarray]:
    """
    Parse KymoButler's printed coordinate lists from Wolfram stdout into
    a list of (N,2) float ndarrays. Robust to extra nesting and noise.

    Strategy:
      1) Extract the first balanced Wolfram list starting at '{{{'.
      2) Convert braces -> brackets and literal_eval to a Python object.
      3) For each item, try to coerce to an (N,2) array, with fallbacks:
         - direct (N,2)
         - first element (extra nesting)
         - reshape-flat to pairs as a last resort
    """
    block = _extract_wolfram_list_block(stdout_text)
    if not block:
        return []

    # Wolfram lists -> Python lists
    py_list_text = block.replace("{", "[").replace("}", "]")

    try:
        data = ast.literal_eval(py_list_text)
    except Exception:
        return []

    arrays: List[np.ndarray] = []

    def _to_pairs(x) -> Optional[np.ndarray]:
        try:
            a = np.asarray(x, dtype=float)
        except Exception:
            return None

        if a.ndim == 2 and a.shape[1] == 2:
            return a

        # Try to coerce the first element when there's an extra nesting level
        if isinstance(x, (list, tuple)) and len(x) > 0:
            try:
                a0 = np.asarray(x[0], dtype=float)
                if a0.ndim == 2 and a0.shape[1] == 2:
                    return a0
            except Exception:
                pass

        # Last resort: flatten into pairs
        if a.size % 2 == 0:
            a = a.reshape(-1, 2)
            if a.ndim == 2 and a.shape[1] == 2:
                return a

        return None

    # Expect a list of tracks, but be flexible
    for item in data:
        cand = _to_pairs(item)
        if cand is not None:
            arrays.append(cand)

    return arrays


def run_kymobutler(
    heatmap_path: Union[str, Path],
    min_length: int = 30,
    verbose: bool = False,
    script_name: str = "Run_Kymobutler.wls",
) -> Path:
    """
    Invoke the KymoButler WolframScript on a heatmap image, parse stdout for
    track coordinates, and save each track as .npy under:
        <heatmap_dir>/<heatmap_stem>/kymobutler_output/

    Returns the base_dir (parent folder that contains kymobutler_output/).
    """
    heatmap_path = Path(heatmap_path)
    base_dir = heatmap_path.parent / heatmap_path.stem
    out_dir = base_dir / "kymobutler_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    script = SCRIPT_DIR / script_name
    cmd = ["wolframscript", "-script", str(script), str(heatmap_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    if verbose:
        if proc.stdout:
            print(proc.stdout)
        if proc.stderr:
            print(proc.stderr)

    proc.check_returncode()

    arrays = _parse_mathematica_arrays(proc.stdout or "")
    saved = 0
    for i, arr in enumerate(arrays):
        # Filter out very short traces right here, if desired
        if arr.shape[0] < min_length:
            continue
        np.save(out_dir / f"{i}.npy", arr)
        saved += 1

    if verbose:
        print(
            f"[kymo_interface] Parsed {len(arrays)} track(s); "
            f"saved {saved} ≥ min_length to {out_dir}"
        )

    return base_dir


def preprocess_image(
    image_path: Union[str, Path],
    verbose: bool = False,
    script_name: str = "Run_Kymobutler_Save_Image.wls",
    output_suffix: str = "_processed_2.png",
) -> Path:
    """
    Run the Mathematica preprocessing script to prepare a raw kymograph image.

    Returns the processed image path in the 'proc' subfolder.

    Parameters
    ----------
    image_path : str | Path
        Input raw kymograph.
    verbose : bool
        If True, print Wolfram stdout/stderr.
    script_name : str
        WolframScript filename to invoke from SCRIPT_DIR.
    output_suffix : str
        Expected suffix of the output filename produced by the Wolfram script.
        Defaults to '_processed_2.png' to match prior behavior.
    """
    image_path = Path(image_path)
    proc_dir = image_path.parent / image_path.stem / "proc"
    proc_dir.mkdir(parents=True, exist_ok=True)

    script = SCRIPT_DIR / script_name
    cmd = ["wolframscript", "-script", str(script), str(proc_dir), str(image_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    if verbose:
        if proc.stdout:
            print(proc.stdout)
        if proc.stderr:
            print(proc.stderr)

    proc.check_returncode()

    processed = proc_dir / f"{image_path.stem}{output_suffix}"
    if not processed.exists():
        raise FileNotFoundError(f"Processed image not found: {processed}")

    return processed
