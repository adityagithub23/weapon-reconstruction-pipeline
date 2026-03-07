"""
sam_segmentation.py
--------------------
Uses Meta's Segment Anything Model (SAM) to generate a pixel-level mask
of the visible weapon region from an RT-DETR bounding box.

No training required. SAM is used at inference time as a zero-shot segmenter.

Setup (run once):
    pip install segment-anything
    pip install git+https://github.com/facebookresearch/segment-anything.git

    # Download SAM checkpoint (mobile_sam is M1-friendly and fast):
    # Option A — MobileSAM (recommended for M1, fast, ~40MB):
    pip install git+https://github.com/ChaoningZhang/MobileSAM.git
    wget https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt -O models/mobile_sam.pt

    # Option B — SAM ViT-B (larger, more accurate, ~375MB):
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O models/sam_vit_b.pth

Usage (standalone test):
    python segmentation/sam_segmentation.py \
        --image path/to/image.jpg \
        --bbox "100,150,400,500" \
        --model models/mobile_sam.pt \
        --output_dir output/masks

Returns:
    visible_mask   → binary mask of the weapon (255 = weapon pixel)
    missing_mask   → inverted mask (255 = region to inpaint)
    cropped_image  → 512x512 weapon crop for SD Inpainting
    cropped_mask   → 512x512 missing mask for SD Inpainting
"""

import argparse
import os
import cv2
import numpy as np
from pathlib import Path
import torch


# ── Determine device ─────────────────────────────────────────────────────────
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Load SAM or MobileSAM ─────────────────────────────────────────────────────
def load_sam_model(model_path: str):
    """
    Auto-detects whether to use MobileSAM or original SAM based on file name.
    Falls back gracefully with an informative error.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"\n[SAM] Model checkpoint not found at '{model_path}'.\n"
            f"  → Download MobileSAM (recommended for M1):\n"
            f"     pip install git+https://github.com/ChaoningZhang/MobileSAM.git\n"
            f"     wget https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt "
            f"-O models/mobile_sam.pt\n"
        )

    device = get_device()
    print(f"[SAM] Using device: {device}")

    model_name = model_path.name.lower()

    # Try MobileSAM first (faster, M1-optimized)
    if "mobile" in model_name:
        try:
            from mobile_sam import sam_model_registry, SamPredictor
            sam = sam_model_registry["vit_t"](checkpoint=str(model_path))
            sam.to(device)
            sam.eval()
            predictor = SamPredictor(sam)
            print(f"[SAM] Loaded MobileSAM from {model_path}")
            return predictor
        except ImportError:
            print("[SAM] MobileSAM not installed, trying original SAM...")

    # Fallback to original SAM
    try:
        from segment_anything import sam_model_registry, SamPredictor
        if "vit_h" in model_name:
            model_type = "vit_h"
        elif "vit_l" in model_name:
            model_type = "vit_l"
        else:
            model_type = "vit_b"  # default
        sam = sam_model_registry[model_type](checkpoint=str(model_path))
        sam.to(device)
        sam.eval()
        predictor = SamPredictor(sam)
        print(f"[SAM] Loaded SAM ({model_type}) from {model_path}")
        return predictor
    except ImportError:
        raise ImportError(
            "[SAM] Neither MobileSAM nor segment-anything is installed.\n"
            "  → pip install git+https://github.com/ChaoningZhang/MobileSAM.git"
        )


# ── Core segmentation function ────────────────────────────────────────────────
def segment_weapon(predictor, image_bgr: np.ndarray, bbox: list,
                   padding: int = 20) -> dict:
    """
    Use SAM to segment the weapon inside a bounding box.

    Args:
        predictor:   SAM SamPredictor instance
        image_bgr:   Full image in BGR (OpenCV format)
        bbox:        [x1, y1, x2, y2] from RT-DETR
        padding:     Extra pixels to add around bbox before segmentation

    Returns dict with keys:
        visible_mask     → H×W uint8 (255=weapon, 0=background)
        missing_mask     → H×W uint8 (255=inpaint region, 0=keep)
        bbox_padded      → [x1, y1, x2, y2] after padding
        crop_512         → 512×512 BGR image of weapon region
        mask_512         → 512×512 binary mask (visible)
        missing_512      → 512×512 binary mask (missing/to-inpaint)
    """
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = bbox

    # Add padding, clamp to image bounds
    x1p = max(0, x1 - padding)
    y1p = max(0, y1 - padding)
    x2p = min(w, x2 + padding)
    y2p = min(h, y2 + padding)

    # Convert to RGB for SAM
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    # SAM box prompt
    input_box = np.array([x1p, y1p, x2p, y2p])
    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=True,
    )

    # Pick the highest-scoring mask
    best_idx = np.argmax(scores)
    visible_mask_full = (masks[best_idx] * 255).astype(np.uint8)

    # Clean up mask with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    visible_mask_full = cv2.morphologyEx(visible_mask_full, cv2.MORPH_CLOSE, kernel)
    visible_mask_full = cv2.morphologyEx(visible_mask_full, cv2.MORPH_OPEN, kernel)

    # Missing mask = bounding box area MINUS the visible weapon mask
    missing_mask_full = np.zeros((h, w), dtype=np.uint8)
    missing_mask_full[y1p:y2p, x1p:x2p] = 255         # start with full bbox
    missing_mask_full[visible_mask_full == 255] = 0     # remove visible weapon pixels

    # Crop to padded bounding box and resize to 512×512
    crop_bgr = image_bgr[y1p:y2p, x1p:x2p]
    crop_vis = visible_mask_full[y1p:y2p, x1p:x2p]
    crop_mis = missing_mask_full[y1p:y2p, x1p:x2p]

    crop_512 = cv2.resize(crop_bgr, (512, 512), interpolation=cv2.INTER_LANCZOS4)
    mask_512 = cv2.resize(crop_vis, (512, 512), interpolation=cv2.INTER_NEAREST)
    missing_512 = cv2.resize(crop_mis, (512, 512), interpolation=cv2.INTER_NEAREST)

    # Ensure binary after resize
    _, mask_512 = cv2.threshold(mask_512, 127, 255, cv2.THRESH_BINARY)
    _, missing_512 = cv2.threshold(missing_512, 127, 255, cv2.THRESH_BINARY)

    return {
        "visible_mask":   visible_mask_full,
        "missing_mask":   missing_mask_full,
        "bbox_padded":    [x1p, y1p, x2p, y2p],
        "crop_512":       crop_512,
        "mask_512":       mask_512,
        "missing_512":    missing_512,
        "sam_score":      float(scores[best_idx]),
    }


def save_segmentation_outputs(result: dict, output_dir: str, stem: str):
    """Save all segmentation outputs to disk."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out / f"{stem}_crop512.jpg"),      result["crop_512"])
    cv2.imwrite(str(out / f"{stem}_mask_visible.png"), result["mask_512"])
    cv2.imwrite(str(out / f"{stem}_mask_missing.png"), result["missing_512"])

    # Save a debug visualization
    debug = result["crop_512"].copy()
    overlay = debug.copy()
    overlay[result["mask_512"] == 255] = [0, 255, 0]    # green = visible weapon
    overlay[result["missing_512"] == 255] = [0, 0, 255] # red   = missing region
    debug_vis = cv2.addWeighted(debug, 0.6, overlay, 0.4, 0)
    cv2.imwrite(str(out / f"{stem}_debug_overlay.jpg"), debug_vis)

    print(f"[SAM] Saved outputs to {out}/")
    print(f"[SAM] Segmentation score: {result['sam_score']:.3f}")


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM weapon segmentation.")
    parser.add_argument("--image",      required=True,  help="Input image path")
    parser.add_argument("--bbox",       required=True,
                        help="Bounding box as 'x1,y1,x2,y2' (pixel coords from RT-DETR)")
    parser.add_argument("--model",      default="models/mobile_sam.pt",
                        help="SAM model checkpoint path")
    parser.add_argument("--output_dir", default="output/segmentation",
                        help="Directory to save outputs")
    parser.add_argument("--padding",    type=int, default=20,
                        help="Padding in pixels around bounding box")
    args = parser.parse_args()

    # Parse bbox
    bbox = [int(v) for v in args.bbox.split(",")]
    assert len(bbox) == 4, "bbox must be 'x1,y1,x2,y2'"

    # Load image
    image = cv2.imread(args.image)
    if image is None:
        raise ValueError(f"Cannot read image: {args.image}")

    # Run SAM
    predictor = load_sam_model(args.model)
    result = segment_weapon(predictor, image, bbox, padding=args.padding)

    # Save outputs
    stem = Path(args.image).stem
    save_segmentation_outputs(result, args.output_dir, stem)

    print("\n[SAM] Segmentation complete!")
    print(f"  Visible mask coverage: "
          f"{(result['mask_512'] > 0).sum() / (512*512) * 100:.1f}% of 512×512 crop")
    print(f"  Missing region:        "
          f"{(result['missing_512'] > 0).sum() / (512*512) * 100:.1f}% of 512×512 crop")

def segment_from_bbox(predictor, image, bbox):
    """
    Simple helper used by Streamlit demo.
    Runs SAM segmentation from bounding box.
    """

    import numpy as np
    import cv2

    if image is None:
        raise ValueError("Image is None")

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    predictor.set_image(img_rgb)

    box = np.array(bbox)

    masks, scores, logits = predictor.predict(
        box=box,
        multimask_output=False
    )

    mask = masks[0].astype("uint8") * 255

    return mask