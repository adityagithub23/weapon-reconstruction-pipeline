"""
evaluate_pipeline.py
--------------------
Evaluates weapon reconstruction pipeline output.
Computes SSIM, PSNR, MSE, IoU and saves:
  - Updated comparison image with IoU overlap panel
  - CSV with all scores per image
  - Summary statistics

Usage:
    python evaluation/evaluate_pipeline.py \
        --results_dir output/results \
        --gt_dir datasets/raw/train \
        --csv_output evaluation/results.csv
"""

import cv2
import csv
import argparse
import numpy as np
from pathlib import Path

# Import metrics from your existing metrics.py
import sys
sys.path.append(str(Path(__file__).parent))
from metrics import compute_iou, compute_psnr, compute_ssim, compute_mse


# ── IoU Overlap Visualization ─────────────────────────────────────────────────
def make_iou_overlap_panel(gt_image, reconstructed, size=(512, 512)):
    """
    Creates a visual panel showing overlap between ground truth and reconstruction.
    - Green  = correctly reconstructed region (true positive)
    - Red    = reconstructed but not in GT (false positive)
    - Blue   = in GT but not reconstructed (false negative)
    """
    gt_r = cv2.resize(gt_image, size)
    rc_r = cv2.resize(reconstructed, size)

    # Convert to grayscale and threshold
    gt_gray = cv2.cvtColor(gt_r, cv2.COLOR_BGR2GRAY)
    rc_gray = cv2.cvtColor(rc_r, cv2.COLOR_BGR2GRAY)

    # Edge-based comparison — more meaningful than raw pixel comparison
    gt_edges = cv2.Canny(cv2.GaussianBlur(gt_gray, (5, 5), 0), 50, 150)
    rc_edges = cv2.Canny(cv2.GaussianBlur(rc_gray, (5, 5), 0), 50, 150)

    gt_bin = gt_edges > 0
    rc_bin = rc_edges > 0

    # Build color overlay
    overlay = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255  # white bg

    # True positive — green
    tp = np.logical_and(rc_bin, gt_bin)
    overlay[tp] = [0, 200, 0]

    # False positive — red
    fp = np.logical_and(rc_bin, ~gt_bin)
    overlay[fp] = [0, 0, 220]

    # False negative — blue
    fn = np.logical_and(~rc_bin, gt_bin)
    overlay[fn] = [220, 0, 0]

    # Compute edge IoU
    intersection = tp.sum()
    union = np.logical_or(rc_bin, gt_bin).sum()
    edge_iou = intersection / union if union > 0 else 0.0

    # Add IoU text on panel
    label = f"Edge IoU: {edge_iou:.3f}"
    cv2.putText(overlay, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Add legend
    cv2.putText(overlay, "Green=Match  Red=Extra  Blue=Missing",
                (10, size[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)

    return overlay, edge_iou


def make_comparison_with_iou(sample_dir, gt_image_path, output_path):
    """
    Loads existing pipeline outputs and creates a 4-panel comparison:
    1. Input (occluded)
    2. SD Reconstruction
    3. Forensic Sketch
    4. IoU Overlap Visualization
    """
    stem = sample_dir.name

    # Load pipeline outputs
    crop_path  = sample_dir / f"{stem}_crop.jpg"
    recon_path = sample_dir / f"{stem}_reconstructed.jpg"
    sketch_path = sample_dir / f"{stem}_sketch.png"

    if not recon_path.exists():
        print(f"  [SKIP] No reconstruction found for {stem}")
        return None, None

    crop  = cv2.imread(str(crop_path))   if crop_path.exists()   else None
    recon = cv2.imread(str(recon_path))
    sketch = cv2.imread(str(sketch_path)) if sketch_path.exists() else None
    gt    = cv2.imread(str(gt_image_path))

    if recon is None or gt is None:
        return None, None

    # Standard panel size
    SIZE = (512, 512)

    # Panel 1 — Input
    p1 = cv2.resize(crop if crop is not None else gt, SIZE)
    cv2.putText(p1, "1. Input (Occluded)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Panel 2 — Reconstruction
    p2 = cv2.resize(recon, SIZE)
    cv2.putText(p2, "2. SD Reconstruction", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Panel 3 — Sketch
    if sketch is not None:
        p3 = cv2.resize(sketch, SIZE)
        if len(p3.shape) == 2:
            p3 = cv2.cvtColor(p3, cv2.COLOR_GRAY2BGR)
    else:
        p3 = np.ones((SIZE[1], SIZE[0], 3), dtype=np.uint8) * 200
    cv2.putText(p3, "3. Forensic Sketch", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Panel 4 — IoU Overlap
    # p4, edge_iou = make_iou_overlap_panel(gt, recon, SIZE)
    # Use crop as ground truth for IoU, not full scene
    gt_for_iou = cv2.imread(str(sample_dir / f"{stem}_crop.jpg"))
    if gt_for_iou is None:
        gt_for_iou = gt
    p4, edge_iou = make_iou_overlap_panel(gt_for_iou, recon, SIZE)
    cv2.putText(p4, "4. IoU Overlap", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Combine all 4 panels
    comparison = np.hstack([p1, p2, p3, p4])

    # Save
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), comparison)

    return comparison, edge_iou


# ── Single sample evaluation ──────────────────────────────────────────────────
def evaluate_sample(sample_dir, gt_image_path):
    """
    Compute all metrics for one sample.
    Returns dict of scores.
    """
    stem = sample_dir.name
    recon_path  = sample_dir / f"{stem}_reconstructed.jpg"
    sketch_path = sample_dir / f"{stem}_sketch.png"
    mask_path   = sample_dir / f"{stem}_mask_visible.png"

    if not recon_path.exists():
        return None

    recon = cv2.imread(str(recon_path))
    gt    = cv2.imread(str(gt_image_path))

    if recon is None or gt is None:
        return None

    # Resize to same size for comparison
    gt_resized = cv2.resize(gt, (recon.shape[1], recon.shape[0]))

    scores = {"sample": stem}

    # Reconstruction quality
    scores["SSIM"]  = round(compute_ssim(recon, gt_resized), 4)
    scores["PSNR"]  = round(compute_psnr(recon, gt_resized), 2)
    scores["MSE"]   = round(compute_mse(recon, gt_resized), 2)

    # Edge IoU
    _, edge_iou = make_iou_overlap_panel(gt, recon)
    scores["Edge_IoU"] = round(edge_iou, 4)

    # Mask IoU (if mask exists)
    if mask_path.exists():
        mask_pred = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        gt_gray   = cv2.cvtColor(gt_resized, cv2.COLOR_BGR2GRAY)
        _, gt_mask = cv2.threshold(gt_gray, 10, 255, cv2.THRESH_BINARY)
        mask_pred_resized = cv2.resize(mask_pred, (gt_mask.shape[1], gt_mask.shape[0]))
        scores["Mask_IoU"] = round(compute_iou(mask_pred_resized, gt_mask), 4)

    return scores


# ── Batch evaluation ──────────────────────────────────────────────────────────
def batch_evaluate(results_dir, gt_dir, csv_output):
    """
    Evaluate all samples in results_dir against ground truth in gt_dir.
    Saves updated comparison images and CSV.
    """
    results_dir = Path(results_dir)
    gt_dir      = Path(gt_dir)
    csv_output  = Path(csv_output)

    sample_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir()])
    print(f"\nFound {len(sample_dirs)} samples to evaluate.")

    all_scores = []

    for sample_dir in sample_dirs:
        stem = sample_dir.name
        print(f"\nEvaluating: {stem}")

        # Find ground truth image
        gt_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            # Search recursively in gt_dir
            matches = list(gt_dir.rglob(f"{stem}{ext}"))
            if matches:
                gt_path = matches[0]
                break

        if gt_path is None:
            print(f"  [SKIP] No ground truth found for {stem}")
            continue

        # Compute metrics
        scores = evaluate_sample(sample_dir, gt_path)
        if scores is None:
            print(f"  [SKIP] Could not evaluate {stem}")
            continue

        all_scores.append(scores)
        print(f"  SSIM={scores['SSIM']:.3f}  PSNR={scores['PSNR']:.1f}dB  "
              f"Edge_IoU={scores['Edge_IoU']:.3f}")

        # Save updated 4-panel comparison with IoU panel
        comp_path = sample_dir / f"{stem}_comparison_with_iou.jpg"
        make_comparison_with_iou(sample_dir, gt_path, comp_path)
        print(f"  Saved: {comp_path.name}")

    if not all_scores:
        print("\nNo samples evaluated. Check your paths.")
        return

    # Print aggregate statistics
    print("\n" + "=" * 55)
    print("  AGGREGATE RESULTS")
    print("=" * 55)
    for metric in ["SSIM", "PSNR", "MSE", "Edge_IoU", "Mask_IoU"]:
        vals = [s[metric] for s in all_scores if metric in s]
        if vals:
            print(f"  {metric:12s}: mean={np.mean(vals):.4f}  "
                  f"std={np.std(vals):.4f}  "
                  f"min={np.min(vals):.4f}  "
                  f"max={np.max(vals):.4f}")
    print("=" * 55)

    # Save CSV
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    keys = list(all_scores[0].keys())
    with open(csv_output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_scores)

    print(f"\nResults saved to: {csv_output}")
    print(f"Total samples evaluated: {len(all_scores)}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True,
                        help="Pipeline output directory (e.g. output/results)")
    parser.add_argument("--gt_dir",      required=True,
                        help="Ground truth images directory (e.g. datasets/raw/train)")
    parser.add_argument("--csv_output",  default="evaluation/results.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    batch_evaluate(args.results_dir, args.gt_dir, args.csv_output)