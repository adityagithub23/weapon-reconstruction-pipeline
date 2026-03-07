"""
sketch.py
---------
Converts a reconstructed weapon image into a forensic-style
black-and-white edge sketch using OpenCV.

No ML model needed — pure classical image processing.

Pipeline:
    Reconstructed color image
    → Grayscale conversion
    → Bilateral filter (preserve edges, remove noise)
    → Canny edge detection
    → Morphological cleanup
    → Forensic sketch output

Usage:
    python sketch_generation/sketch.py \
        --image output/reconstructed/img001_reconstructed.jpg \
        --output output/sketches/img001_sketch.png \
        --style forensic

    Styles: "forensic" (clean lines), "pencil" (softer), "technical" (precise)
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from PIL import Image


# ── Sketch style presets ─────────────────────────────────────────────────────
STYLE_PARAMS = {
    "forensic": {
        # Bilateral filter — smooths while keeping edges sharp
        "bilateral_d": 9,
        "bilateral_sigma_color": 75,
        "bilateral_sigma_space": 75,
        # Canny thresholds — low=fine detail, high=strong edges only
        "canny_low":  30,
        "canny_high": 100,
        # Dilation kernel for line thickness
        "dilate_size": 2,
        # Final inversion: white background, black lines (forensic style)
        "invert": True,
    },
    "pencil": {
        "bilateral_d": 15,
        "bilateral_sigma_color": 80,
        "bilateral_sigma_space": 80,
        "canny_low":  20,
        "canny_high": 80,
        "dilate_size": 3,
        "invert": True,
    },
    "technical": {
        "bilateral_d": 5,
        "bilateral_sigma_color": 50,
        "bilateral_sigma_space": 50,
        "canny_low":  50,
        "canny_high": 150,
        "dilate_size": 1,
        "invert": True,
    },
}


def image_to_forensic_sketch(
    image: np.ndarray,
    style: str = "forensic",
    custom_params: dict = None,
) -> dict:
    """
    Convert a color or grayscale image to a forensic-style sketch.

    Args:
        image:         BGR or grayscale numpy array
        style:         One of "forensic", "pencil", "technical"
        custom_params: Override any style parameters

    Returns dict with:
        sketch       → H×W uint8 sketch image (white bg, black lines)
        edges_raw    → Raw Canny edges before cleanup
        gray         → Grayscale version of input
    """
    params = STYLE_PARAMS.get(style, STYLE_PARAMS["forensic"]).copy()
    if custom_params:
        params.update(custom_params)

    # Step 1: Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Step 2: Bilateral filter — smooths flat regions, preserves edges
    # This is critical for clean sketches (Gaussian blur would blur edges too)
    filtered = cv2.bilateralFilter(
        gray,
        d=params["bilateral_d"],
        sigmaColor=params["bilateral_sigma_color"],
        sigmaSpace=params["bilateral_sigma_space"],
    )

    # Step 3: Canny edge detection
    edges = cv2.Canny(filtered, params["canny_low"], params["canny_high"])

    # Step 4: Morphological operations to clean up and strengthen edges
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_dilate = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (params["dilate_size"], params["dilate_size"])
    )

    # Close small gaps in edge lines
    sketch = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close)

    # Slightly thicken lines for visibility
    if params["dilate_size"] > 1:
        sketch = cv2.dilate(sketch, kernel_dilate, iterations=1)

    # Step 5: Remove isolated noise pixels (small blobs)
    sketch = remove_small_blobs(sketch, min_area=10)

    # Step 6: Invert for white background, black lines (forensic document style)
    if params["invert"]:
        sketch = cv2.bitwise_not(sketch)

    return {
        "sketch":    sketch,
        "edges_raw": edges,
        "gray":      gray,
        "filtered":  filtered,
    }


def remove_small_blobs(binary_image: np.ndarray, min_area: int = 10) -> np.ndarray:
    """Remove isolated noise pixels smaller than min_area."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image)
    cleaned = np.zeros_like(binary_image)
    for i in range(1, num_labels):  # skip background (label 0)
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
    return cleaned


def generate_comparison_sheet(
    original: np.ndarray,
    reconstructed: np.ndarray,
    sketch: np.ndarray,
    output_path: str,
) -> np.ndarray:
    """
    Create a 3-panel comparison: [Occluded Input | Reconstructed | Forensic Sketch]
    Great for presentations and evaluations.
    """
    target_h = 512
    target_w = 512

    def prep(img):
        """Resize to target, convert to 3-channel."""
        resized = cv2.resize(img, (target_w, target_h))
        if len(resized.shape) == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        return resized

    orig_panel   = prep(original)
    recon_panel  = prep(reconstructed)
    sketch_panel = prep(sketch)

    # Add labels
    def add_label(img, text):
        result = img.copy()
        cv2.rectangle(result, (0, target_h - 35), (target_w, target_h), (0, 0, 0), -1)
        cv2.putText(result, text, (10, target_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return result

    orig_panel   = add_label(orig_panel,   "1. Input (Occluded)")
    recon_panel  = add_label(recon_panel,  "2. SD Reconstruction")
    sketch_panel = add_label(sketch_panel, "3. Forensic Sketch")

    # Add dividers
    divider = np.ones((target_h, 4, 3), dtype=np.uint8) * 128
    comparison = np.hstack([orig_panel, divider, recon_panel, divider, sketch_panel])

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), comparison)
    print(f"[Sketch] Comparison sheet saved to {out_path}")
    return comparison


def sketch_from_pil(pil_image: Image.Image, style: str = "forensic") -> Image.Image:
    """
    Convenience function: PIL Image in → PIL Image out (sketch).
    Used by pipeline.py and Streamlit app.
    """
    import numpy as np
    bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    result = image_to_forensic_sketch(bgr, style=style)
    sketch_bgr = result["sketch"]
    if len(sketch_bgr.shape) == 2:
        sketch_rgb = cv2.cvtColor(sketch_bgr, cv2.COLOR_GRAY2RGB)
    else:
        sketch_rgb = cv2.cvtColor(sketch_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(sketch_rgb)


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forensic sketch generation.")
    parser.add_argument("--image",   required=True, help="Input image (reconstructed weapon)")
    parser.add_argument("--output",  default="output/sketches/sketch.png",
                        help="Output path for sketch")
    parser.add_argument("--style",   default="forensic",
                        choices=["forensic", "pencil", "technical"],
                        help="Sketch style preset")
    parser.add_argument("--canny_low",  type=int, default=None, help="Override Canny low threshold")
    parser.add_argument("--canny_high", type=int, default=None, help="Override Canny high threshold")
    parser.add_argument("--show_steps", action="store_true",
                        help="Save intermediate processing steps")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise ValueError(f"Cannot read image: {args.image}")

    custom_params = {}
    if args.canny_low:  custom_params["canny_low"]  = args.canny_low
    if args.canny_high: custom_params["canny_high"] = args.canny_high

    result = image_to_forensic_sketch(image, style=args.style, custom_params=custom_params or None)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), result["sketch"])
    print(f"[Sketch] Saved sketch to {out_path}")

    if args.show_steps:
        stem = out_path.stem
        parent = out_path.parent
        cv2.imwrite(str(parent / f"{stem}_gray.png"),     result["gray"])
        cv2.imwrite(str(parent / f"{stem}_filtered.png"), result["filtered"])
        cv2.imwrite(str(parent / f"{stem}_edges.png"),    result["edges_raw"])
        print(f"[Sketch] Saved intermediate steps to {parent}/")
