"""
generate_occlusions.py
----------------------
Takes clean weapon images + YOLO bounding box annotations and generates
artificially occluded versions for testing the reconstruction pipeline.

Usage:
    python segmentation/generate_occlusions.py \
        --images_dir datasets/raw \
        --labels_dir datasets/labels \
        --output_dir datasets/occluded \
        --num_occlusions 3

Folder structure expected:
    datasets/raw/pistol/img001.jpg
    datasets/labels/pistol/img001.txt   ← YOLO format .txt
"""

import os
import cv2
import random
import argparse
import numpy as np
from pathlib import Path


# ── class names matching your RT-DETR training order ──────────────────────────
CLASS_NAMES = {0: "fire", 1: "firearm", 2: "grenade", 3: "knife", 4: "pistol", 5: "rocket"}


def random_rectangle_occlusion(image, bbox, intensity=0.4):
    """Cover a random portion of the bounding box with a black rectangle."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    result = image.copy()

    box_w = x2 - x1
    box_h = y2 - y1

    # Cover between 30-60% of the bounding box area
    occ_w = int(box_w * random.uniform(0.3, 0.6))
    occ_h = int(box_h * random.uniform(0.3, 0.6))

    # Random position within the bounding box
    occ_x = random.randint(x1, max(x1, x2 - occ_w))
    occ_y = random.randint(y1, max(y1, y2 - occ_h))

    cv2.rectangle(result, (occ_x, occ_y), (occ_x + occ_w, occ_y + occ_h), (0, 0, 0), -1)
    return result, (occ_x, occ_y, occ_x + occ_w, occ_y + occ_h)


def random_polygon_occlusion(image, bbox):
    """Cover a random polygon region inside the bounding box."""
    x1, y1, x2, y2 = bbox
    result = image.copy()

    # Random polygon with 4-6 vertices inside the bounding box
    num_pts = random.randint(4, 6)
    pts = np.array([
        [random.randint(x1, x2), random.randint(y1, y2)]
        for _ in range(num_pts)
    ], dtype=np.int32)

    cv2.fillPoly(result, [pts], (0, 0, 0))
    return result


def edge_crop_occlusion(image, bbox):
    """Simulate weapon partially going out of frame — crop from one edge."""
    x1, y1, x2, y2 = bbox
    result = image.copy()
    h, w = image.shape[:2]

    edge = random.choice(["left", "right", "top", "bottom"])
    crop_frac = random.uniform(0.25, 0.5)

    if edge == "left":
        cut = x1 + int((x2 - x1) * crop_frac)
        result[:, :cut] = 0
    elif edge == "right":
        cut = x2 - int((x2 - x1) * crop_frac)
        result[:, cut:] = 0
    elif edge == "top":
        cut = y1 + int((y2 - y1) * crop_frac)
        result[:cut, :] = 0
    elif edge == "bottom":
        cut = y2 - int((y2 - y1) * crop_frac)
        result[cut:, :] = 0

    return result


def yolo_to_pixel(label_line, img_w, img_h):
    """Convert YOLO normalized bbox to pixel coordinates."""
    parts = label_line.strip().split()
    if len(parts) < 5:
        return None, None
    cls_id = int(parts[0])
    cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

    x1 = int((cx - bw / 2) * img_w)
    y1 = int((cy - bh / 2) * img_h)
    x2 = int((cx + bw / 2) * img_w)
    y2 = int((cy + bh / 2) * img_h)

    # Clamp to image bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_w, x2), min(img_h, y2)
    return cls_id, (x1, y1, x2, y2)


def process_image(img_path, label_path, output_dir, num_occlusions=3):
    """Generate multiple occluded versions of one image."""
    image = cv2.imread(str(img_path))
    if image is None:
        print(f"  [SKIP] Cannot read {img_path}")
        return 0

    h, w = image.shape[:2]

    if not label_path.exists():
        print(f"  [SKIP] No label file for {img_path.name}")
        return 0

    with open(label_path) as f:
        lines = f.readlines()

    if not lines:
        return 0

    # Use the first (primary) weapon annotation
    cls_id, bbox = yolo_to_pixel(lines[0], w, h)
    if bbox is None:
        return 0

    class_name = CLASS_NAMES.get(cls_id, f"class{cls_id}")
    out_class_dir = Path(output_dir) / class_name
    out_class_dir.mkdir(parents=True, exist_ok=True)

    stem = img_path.stem
    suffix = img_path.suffix
    count = 0

    occlusion_fns = [random_rectangle_occlusion, random_polygon_occlusion, edge_crop_occlusion]

    for i in range(num_occlusions):
        fn = occlusion_fns[i % len(occlusion_fns)]
        try:
            if fn == random_rectangle_occlusion:
                occluded, _ = fn(image, bbox)
            else:
                occluded = fn(image, bbox)

            out_path = out_class_dir / f"{stem}_occ{i}{suffix}"
            cv2.imwrite(str(out_path), occluded)
            count += 1
        except Exception as e:
            print(f"  [ERROR] {fn.__name__} on {img_path.name}: {e}")

    return count


def main():
    parser = argparse.ArgumentParser(description="Generate occluded weapon images.")
    parser.add_argument("--images_dir", default="datasets/raw",
                        help="Root folder with class subfolders of clean images")
    parser.add_argument("--labels_dir", default="datasets/labels",
                        help="Root folder with class subfolders of YOLO .txt labels")
    parser.add_argument("--output_dir", default="datasets/occluded",
                        help="Output folder for occluded images")
    parser.add_argument("--num_occlusions", type=int, default=3,
                        help="Number of occluded variants per image")
    args = parser.parse_args()

    images_root = Path(args.images_dir)
    labels_root = Path(args.labels_dir)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    total_generated = 0
    img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    # Walk through all class subdirectories
    for class_dir in sorted(images_root.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        label_class_dir = labels_root / class_name
        print(f"\n[CLASS] {class_name}")

        images = [p for p in class_dir.iterdir() if p.suffix.lower() in img_extensions]
        print(f"  Found {len(images)} images")

        for img_path in sorted(images):
            label_path = label_class_dir / (img_path.stem + ".txt")
            n = process_image(img_path, label_path, str(output_root), args.num_occlusions)
            total_generated += n

    print(f"\n✅ Done. Generated {total_generated} occluded images in '{output_root}'")


if __name__ == "__main__":
    main()
