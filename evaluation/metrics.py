"""
metrics.py
----------
Evaluation metrics for the weapon reconstruction pipeline.

Computes:
  - SSIM  (Structural Similarity Index)   → reconstruction quality vs ground truth
  - PSNR  (Peak Signal-to-Noise Ratio)    → pixel-level accuracy
  - L1 Loss                               → mean absolute pixel error
  - Edge Similarity (F1)                  → how well sketch edges match ground truth
  - IoU / Dice                            → segmentation mask quality

Usage:
    python evaluation/metrics.py \
        --reconstructed output/results/img001/img001_reconstructed.jpg \
        --ground_truth  datasets/raw/pistol/img001.jpg \
        --sketch        output/results/img001/img001_sketch.png \
        --mask_pred     output/results/img001/img001_mask_visible.png

    # Batch evaluation:
    python evaluation/metrics.py \
        --batch_results_dir output/batch \
        --ground_truth_dir  datasets/raw \
        --csv_output        evaluation/results.csv
"""

import argparse
import cv2
import csv
import json
import numpy as np
from pathlib import Path

try:
    from skimage.metrics import structural_similarity as ssim_fn
    from skimage.metrics import peak_signal_noise_ratio as psnr_fn
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("[Metrics] Warning: scikit-image not installed. pip install scikit-image")


# ── Individual metric functions ───────────────────────────────────────────────

def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Structural Similarity Index.
    1.0 = identical, 0.0 = completely different.
    Both images must be same size. Resize img1 to match img2 if needed.
    """
    if not SKIMAGE_AVAILABLE:
        return -1.0

    img1 = _prep_gray(img1)
    img2 = _prep_gray(img2)
    img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))

    score, _ = ssim_fn(img1, img2, full=True, data_range=255)
    return float(score)


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Peak Signal-to-Noise Ratio (dB).
    Higher is better. >30dB is generally good. inf = identical images.
    """
    if not SKIMAGE_AVAILABLE:
        return _psnr_manual(img1, img2)

    img1 = _prep_gray(img1)
    img2 = _prep_gray(img2)
    img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))

    return float(psnr_fn(img2, img1, data_range=255))


def _psnr_manual(img1: np.ndarray, img2: np.ndarray) -> float:
    """PSNR without skimage."""
    img1 = _prep_gray(img1).astype(np.float64)
    img2 = _prep_gray(img2).astype(np.float64)
    img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return float(20 * np.log10(255.0 / np.sqrt(mse)))


def compute_l1(img1: np.ndarray, img2: np.ndarray) -> float:
    """Mean Absolute Error between two images (normalized 0-1)."""
    img1 = _prep_gray(img1).astype(np.float64) / 255.0
    img2 = _prep_gray(img2).astype(np.float64) / 255.0
    img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))
    return float(np.mean(np.abs(img1 - img2)))


def compute_iou(mask_pred: np.ndarray, mask_gt: np.ndarray) -> float:
    """
    Intersection over Union for binary segmentation masks.
    Both masks: 0=background, 255=foreground.
    """
    pred = (mask_pred > 127).astype(bool)
    gt   = (mask_gt   > 127).astype(bool)

    if pred.shape != gt.shape:
        pred_resized = cv2.resize(mask_pred, (gt.shape[1], gt.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
        pred = (pred_resized > 127).astype(bool)

    intersection = np.logical_and(pred, gt).sum()
    union        = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


def compute_dice(mask_pred: np.ndarray, mask_gt: np.ndarray) -> float:
    """
    Dice coefficient (F1 score) for segmentation.
    2 * |A ∩ B| / (|A| + |B|)
    """
    pred = (mask_pred > 127).astype(bool)
    gt   = (mask_gt   > 127).astype(bool)

    if pred.shape != gt.shape:
        pred_resized = cv2.resize(mask_pred, (gt.shape[1], gt.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
        pred = (pred_resized > 127).astype(bool)

    intersection = np.logical_and(pred, gt).sum()
    total = pred.sum() + gt.sum()
    if total == 0:
        return 1.0
    return float(2 * intersection / total)


def compute_edge_similarity(sketch: np.ndarray, ground_truth: np.ndarray,
                             canny_low: int = 30, canny_high: int = 100) -> dict:
    """
    Compare sketch edges against Canny edges extracted from ground truth image.
    Returns precision, recall, F1 score.

    Higher F1 = sketch edges closely match the true weapon structure.
    """
    sketch_edges = _get_edges(sketch, canny_low, canny_high)
    gt_edges     = _get_edges(ground_truth, canny_low, canny_high)

    # Resize to same size
    gt_edges = cv2.resize(gt_edges, (sketch_edges.shape[1], sketch_edges.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

    sketch_bin = (sketch_edges > 127)
    gt_bin     = (gt_edges > 127)

    tp = np.logical_and(sketch_bin, gt_bin).sum()
    fp = np.logical_and(sketch_bin, ~gt_bin).sum()
    fn = np.logical_and(~sketch_bin, gt_bin).sum()

    precision = float(tp / (tp + fp + 1e-8))
    recall    = float(tp / (tp + fn + 1e-8))
    f1        = float(2 * precision * recall / (precision + recall + 1e-8))

    return {"precision": precision, "recall": recall, "f1": f1}


def _get_edges(image: np.ndarray, low: int, high: int) -> np.ndarray:
    """Extract Canny edges from image (gray or color)."""
    gray = _prep_gray(image)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blurred, low, high)


def _prep_gray(image: np.ndarray) -> np.ndarray:
    """Convert to uint8 grayscale if needed."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.astype(np.uint8)


# ── Full evaluation for one sample ───────────────────────────────────────────

def evaluate_sample(reconstructed_path: str, ground_truth_path: str,
                    sketch_path: str = None, mask_pred_path: str = None,
                    mask_gt_path: str = None) -> dict:
    """
    Compute all metrics for one reconstructed image vs ground truth.

    Returns dict of all metric scores.
    """
    recon = cv2.imread(str(reconstructed_path))
    gt    = cv2.imread(str(ground_truth_path))

    if recon is None:
        raise ValueError(f"Cannot read: {reconstructed_path}")
    if gt is None:
        raise ValueError(f"Cannot read: {ground_truth_path}")

    scores = {
        "sample":    Path(reconstructed_path).stem,
        "ssim":      compute_ssim(recon, gt),
        "psnr":      compute_psnr(recon, gt),
        "l1_loss":   compute_l1(recon, gt),
    }

    # Edge similarity (sketch vs ground truth)
    if sketch_path and Path(sketch_path).exists():
        sketch = cv2.imread(str(sketch_path))
        edge_scores = compute_edge_similarity(sketch, gt)
        scores.update({
            "edge_precision": edge_scores["precision"],
            "edge_recall":    edge_scores["recall"],
            "edge_f1":        edge_scores["f1"],
        })

    # Segmentation quality
    if mask_pred_path and mask_gt_path:
        if Path(mask_pred_path).exists() and Path(mask_gt_path).exists():
            mask_pred = cv2.imread(str(mask_pred_path), cv2.IMREAD_GRAYSCALE)
            mask_gt_img = cv2.imread(str(mask_gt_path), cv2.IMREAD_GRAYSCALE)
            scores["iou"]  = compute_iou(mask_pred, mask_gt_img)
            scores["dice"] = compute_dice(mask_pred, mask_gt_img)

    return scores


def print_scores(scores: dict):
    """Pretty-print metric scores."""
    print("\n" + "─" * 45)
    print(f"  Evaluation: {scores.get('sample', 'unknown')}")
    print("─" * 45)
    if "ssim"   in scores: print(f"  SSIM            : {scores['ssim']:.4f}  (1.0 = perfect)")
    if "psnr"   in scores: print(f"  PSNR            : {scores['psnr']:.2f} dB (>30 = good)")
    if "l1_loss" in scores:print(f"  L1 Loss         : {scores['l1_loss']:.4f}  (0.0 = perfect)")
    if "edge_f1" in scores:
        print(f"  Edge F1 Score   : {scores['edge_f1']:.4f}")
        print(f"  Edge Precision  : {scores['edge_precision']:.4f}")
        print(f"  Edge Recall     : {scores['edge_recall']:.4f}")
    if "iou"  in scores: print(f"  IoU             : {scores['iou']:.4f}  (1.0 = perfect)")
    if "dice" in scores: print(f"  Dice Score      : {scores['dice']:.4f}  (1.0 = perfect)")
    print("─" * 45)


def batch_evaluate(results_dir: str, gt_dir: str, csv_output: str = None) -> list:
    """
    Evaluate all samples in a batch output directory.
    Expects results_dir/<stem>/<stem>_reconstructed.jpg structure from pipeline.py
    """
    results_dir = Path(results_dir)
    gt_dir = Path(gt_dir)

    all_scores = []
    sample_dirs = [d for d in sorted(results_dir.iterdir()) if d.is_dir()]
    print(f"[Eval] Found {len(sample_dirs)} sample directories")

    for sample_dir in sample_dirs:
        stem = sample_dir.name
        recon_path = sample_dir / f"{stem}_reconstructed.jpg"
        sketch_path = sample_dir / f"{stem}_sketch.png"
        mask_pred = sample_dir / f"{stem}_mask_visible.png"

        # Find ground truth — look in all class subdirectories
        gt_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            for cls_dir in gt_dir.iterdir():
                candidate = cls_dir / f"{stem}{ext}"
                if candidate.exists():
                    gt_path = candidate
                    break
            if gt_path:
                break

        if not recon_path.exists():
            print(f"  [SKIP] No reconstruction found for {stem}")
            continue
        if gt_path is None:
            print(f"  [SKIP] No ground truth found for {stem}")
            continue

        try:
            scores = evaluate_sample(
                str(recon_path), str(gt_path),
                sketch_path=str(sketch_path) if sketch_path.exists() else None,
                mask_pred_path=str(mask_pred) if mask_pred.exists() else None,
            )
            all_scores.append(scores)
            print(f"  ✓ {stem}: SSIM={scores['ssim']:.3f}  PSNR={scores['psnr']:.1f}dB")
        except Exception as e:
            print(f"  [ERROR] {stem}: {e}")

    if all_scores:
        # Aggregate statistics
        print("\n" + "=" * 50)
        print("  AGGREGATE STATISTICS")
        print("=" * 50)
        for metric in ["ssim", "psnr", "l1_loss", "edge_f1", "iou", "dice"]:
            vals = [s[metric] for s in all_scores if metric in s]
            if vals:
                print(f"  {metric:16s}: mean={np.mean(vals):.4f}  "
                      f"std={np.std(vals):.4f}  "
                      f"min={np.min(vals):.4f}  max={np.max(vals):.4f}")

        # Save CSV
        if csv_output:
            csv_path = Path(csv_output)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            keys = list(all_scores[0].keys())
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(all_scores)
            print(f"\n[Eval] Results saved to {csv_path}")

    return all_scores


# ── Standalone runner ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate weapon reconstruction pipeline.")
    parser.add_argument("--reconstructed",    default=None, help="Reconstructed image")
    parser.add_argument("--ground_truth",     default=None, help="Ground truth original image")
    parser.add_argument("--sketch",           default=None, help="Generated sketch")
    parser.add_argument("--mask_pred",        default=None, help="Predicted segmentation mask")
    parser.add_argument("--mask_gt",          default=None, help="Ground truth mask (optional)")
    parser.add_argument("--batch_results_dir",default=None, help="Batch output directory")
    parser.add_argument("--ground_truth_dir", default=None, help="Ground truth images directory")
    parser.add_argument("--csv_output",       default="evaluation/results.csv",
                        help="CSV output path for batch evaluation")
    args = parser.parse_args()

    if args.batch_results_dir and args.ground_truth_dir:
        batch_evaluate(args.batch_results_dir, args.ground_truth_dir, args.csv_output)

    elif args.reconstructed and args.ground_truth:
        scores = evaluate_sample(
            args.reconstructed, args.ground_truth,
            args.sketch, args.mask_pred, args.mask_gt,
        )
        print_scores(scores)

    else:
        parser.error("Provide either (--reconstructed + --ground_truth) or "
                     "(--batch_results_dir + --ground_truth_dir)")
