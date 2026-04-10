"""
rtdetr_inference.py
-------------------
Runs RT-DETR on a single image and returns bounding boxes + class labels.
Designed as a drop-in module for pipeline.py.

Usage (standalone test):
    python detection/rtdetr_inference.py --image path/to/image.jpg --model models/rtdetr_weapon.pt

Returns list of dicts:
    [{"class_id": 4, "class_name": "pistol", "confidence": 0.91,
      "bbox": [x1, y1, x2, y2]}, ...]
"""

import argparse
import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import RTDETR

# ── class names — must match your training data.yaml order ───────────────────
CLASS_NAMES = {0: "fire", 1: "firearm", 2: "grenade", 3: "knife", 4: "pistol", 5: "rocket"}

# Classes that are actual weapons we care about for reconstruction
WEAPON_CLASSES = {0, 1, 2, 3, 4, 5}  # exclude "fire" (class 0)

# Text prompts for SD Inpainting — one per weapon class
SD_PROMPTS = {
    "pistol":   "complete realistic pistol handgun, full weapon visible, clean white background, "
                "forensic technical drawing, high detail, professional",
    "firearm":  "complete realistic firearm rifle assault weapon, full weapon visible, "
                "clean background, forensic technical drawing, high detail",
    "knife":    "complete realistic knife blade, full weapon visible, clean white background, "
                "forensic technical drawing, high detail, professional",
    "grenade":  "complete realistic grenade, full weapon visible, clean white background, "
                "forensic technical drawing, high detail",
    "rocket":   "complete realistic rocket launcher missile, full weapon visible, "
                "clean background, forensic technical drawing, high detail",
    "fire":     "controlled fire flame, complete image, natural background",  # fallback
}


def load_model(model_path: str) -> RTDETR:
    """Load RT-DETR model from .pt weights file."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at '{model_path}'.\n"
            f"  → Place your trained weights at: models/rtdetr_weapon.pt"
        )
    print(f"[RT-DETR] Loading model from {model_path}")
    model = RTDETR(str(model_path))
    return model


def run_inference(model: RTDETR, image_path: str, conf_threshold: float = 0.35,
                  weapon_only: bool = True) -> list:
    """
    Run RT-DETR on one image.

    Args:
        model:           Loaded RT-DETR model
        image_path:      Path to input image
        conf_threshold:  Minimum confidence to keep a detection
        weapon_only:     If True, filter out non-weapon class detections (e.g. fire)

    Returns:
        List of detection dicts sorted by confidence (highest first)
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    results = model.predict(source=str(image_path), conf=conf_threshold, verbose=False)
    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]

            x1, y1, x2, y2 = [int(v) for v in xyxy]
            class_name = CLASS_NAMES.get(cls_id, f"unknown_{cls_id}")

            if weapon_only and cls_id not in WEAPON_CLASSES:
                continue

            detections.append({
                "class_id":   cls_id,
                "class_name": class_name,
                "confidence": round(conf, 4),
                "bbox":       [x1, y1, x2, y2],
                "prompt":     SD_PROMPTS.get(class_name, SD_PROMPTS["firearm"]),
            })

    # Sort by confidence descending
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections


def draw_detections(image_path: str, detections: list, output_path: str = None) -> np.ndarray:
    """Draw bounding boxes on image. Optionally save to output_path."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    colors = {
        "pistol": (0, 100, 255), "firearm": (0, 200, 100),
        "knife": (255, 100, 0), "grenade": (200, 0, 200),
        "rocket": (0, 180, 255), "fire": (0, 80, 200),
    }

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        color = colors.get(det["class_name"], (255, 255, 255))
        label = f"{det['class_name']} {det['confidence']:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(image, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if output_path:
        import os
        os.makedirs(Path(output_path).parent, exist_ok=True)
        cv2.imwrite(str(output_path), image)
        print(f"[RT-DETR] Saved annotated image to {output_path}")

    return image


def get_primary_weapon(detections: list) -> dict | None:
    """Return the highest-confidence weapon detection, or None if no detections."""
    weapon_dets = [d for d in detections if d["class_id"] in WEAPON_CLASSES]
    return weapon_dets[0] if weapon_dets else None


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RT-DETR weapon detection inference.")
    parser.add_argument("--image",  required=True, help="Path to input image")
    parser.add_argument("--model",  default="models/rtdetr_weapon.pt",
                        help="Path to RT-DETR weights")
    parser.add_argument("--conf",   type=float, default=0.35,
                        help="Confidence threshold (default 0.35)")
    parser.add_argument("--output", default=None,
                        help="Optional path to save annotated image")
    args = parser.parse_args()

    model = load_model(args.model)
    dets  = run_inference(model, args.image, conf_threshold=args.conf)

    if not dets:
        print("[RT-DETR] No weapon detections above confidence threshold.")
    else:
        print(f"[RT-DETR] Found {len(dets)} detection(s):")
        for d in dets:
            print(f"  → {d['class_name']:10s}  conf={d['confidence']:.3f}  "
                  f"bbox={d['bbox']}")

        primary = get_primary_weapon(dets)
        if primary:
            print(f"\n[Primary weapon] {primary['class_name']} "
                  f"(conf={primary['confidence']:.3f})")
            print(f"[SD Prompt]      {primary['prompt']}")

    out = args.output or "detection_output.jpg"
    draw_detections(args.image, dets, out)
