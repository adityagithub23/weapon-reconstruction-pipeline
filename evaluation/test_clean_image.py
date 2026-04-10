# # """
# # test_clean_image.py
# # -------------------
# # Takes a clean weapon image, artificially occludes it,
# # runs it through the pipeline, then computes IoU against
# # the original clean image for a fair evaluation.

# # This gives meaningful metrics because ground truth is known exactly.

# # Usage:
# #     python evaluation/test_clean_image.py \
# #         --image datasets/raw/grenade/grenade3.jpg \
# #         --class_name grenade \
# #         --output_dir output/clean_test
# # """

# # import cv2
# # import sys
# # import argparse
# # import numpy as np
# # from pathlib import Path

# # sys.path.append(str(Path(__file__).parent))
# # from metrics import compute_iou, compute_psnr, compute_ssim, compute_mse


# # def apply_occlusion(image, occlusion_ratio=0.4):
# #     """
# #     Artificially occlude the right half of the weapon image.
# #     Returns occluded image and the exact occlusion mask.
# #     """
# #     h, w = image.shape[:2]
# #     occluded = image.copy()

# #     # Occlude right portion of the image
# #     x_start = int(w * (1 - occlusion_ratio))
# #     occluded[:, x_start:] = 0  # black out right side

# #     # Occlusion mask — white = hidden region
# #     mask = np.zeros((h, w), dtype=np.uint8)
# #     mask[:, x_start:] = 255

# #     return occluded, mask


# # def make_iou_panel(original, reconstructed, size=(512, 512)):
# #     """
# #     Compare edges of original vs reconstructed.
# #     Green = matched, Red = extra in recon, Blue = missing from recon.
# #     """
# #     orig_r  = cv2.resize(original,     size)
# #     recon_r = cv2.resize(reconstructed, size)

# #     orig_gray  = cv2.cvtColor(orig_r,  cv2.COLOR_BGR2GRAY)
# #     recon_gray = cv2.cvtColor(recon_r, cv2.COLOR_BGR2GRAY)

# #     orig_edges  = cv2.Canny(cv2.GaussianBlur(orig_gray,  (5, 5), 0), 50, 150)
# #     recon_edges = cv2.Canny(cv2.GaussianBlur(recon_gray, (5, 5), 0), 50, 150)

# #     orig_bin  = orig_edges  > 0
# #     recon_bin = recon_edges > 0

# #     overlay = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255

# #     tp = np.logical_and(recon_bin, orig_bin)
# #     fp = np.logical_and(recon_bin, ~orig_bin)
# #     fn = np.logical_and(~recon_bin, orig_bin)

# #     overlay[tp] = [0, 200, 0]    # green  = match
# #     overlay[fp] = [0, 0, 220]    # red    = extra
# #     overlay[fn] = [220, 0, 0]    # blue   = missing

# #     intersection = tp.sum()
# #     union = np.logical_or(recon_bin, orig_bin).sum()
# #     edge_iou = intersection / union if union > 0 else 0.0

# #     # cv2.putText(overlay, f"Edge IoU: {edge_iou:.3f}", (10, 35),
# #     #             cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
# #     # cv2.putText(overlay, "Green=Match  Red=Extra  Blue=Missing",
# #     #             (10, size[1] - 15),
# #     #             cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)

# #     return overlay, edge_iou


# # def run_test(image_path, class_name, output_dir, steps=20):
# #     """
# #     Full test pipeline on a clean image.
# #     """
# #     image_path = Path(image_path)
# #     output_dir = Path(output_dir)
# #     output_dir.mkdir(parents=True, exist_ok=True)

# #     print(f"\nClean Image Test")
# #     print(f"Image : {image_path.name}")
# #     print(f"Class : {class_name}")
# #     print("-" * 40)

# #     # Load clean image
# #     original = cv2.imread(str(image_path))
# #     if original is None:
# #         raise ValueError(f"Cannot read: {image_path}")

# #     print(f"Image size: {original.shape[1]}x{original.shape[0]}")

# #     # Step 1 — Artificially occlude
# #     print("\n[1] Applying artificial occlusion (right 40%)...")
# #     occluded, occ_mask = apply_occlusion(original, occlusion_ratio=0.4)
# #     cv2.imwrite(str(output_dir / "1_original.jpg"), original)
# #     cv2.imwrite(str(output_dir / "2_occluded.jpg"), occluded)
# #     cv2.imwrite(str(output_dir / "3_occlusion_mask.png"), occ_mask)
# #     print("   Saved: 1_original.jpg, 2_occluded.jpg, 3_occlusion_mask.png")

# #     # Step 2 — SD Inpainting
# #     print("\n[2] Running SD Inpainting...")
# #     sys.path.append(str(Path(__file__).parent.parent / "diffusion_completion"))
# #     from inpaint import inpaint_weapon

# #     from PIL import Image as PILImage
# #     pil_occluded = PILImage.fromarray(cv2.cvtColor(occluded, cv2.COLOR_BGR2RGB))
# #     pil_mask     = PILImage.fromarray(occ_mask)

# #     results = inpaint_weapon(
# #         crop_image=pil_occluded,
# #         missing_mask=pil_mask,
# #         class_name=class_name,
# #         num_inference_steps=steps,
# #         guidance_scale=7.5,
# #         seed=42,
# #     )

# #     reconstructed_pil = results[0]
# #     reconstructed = cv2.cvtColor(np.array(reconstructed_pil), cv2.COLOR_RGB2BGR)
# #     reconstructed = cv2.resize(reconstructed, (original.shape[1], original.shape[0]))
# #     cv2.imwrite(str(output_dir / "4_reconstructed.jpg"), reconstructed)
# #     print("   Saved: 4_reconstructed.jpg")

# #     # Step 3 — Sketch
# #     print("\n[3] Generating forensic sketch...")
# #     gray    = cv2.cvtColor(reconstructed, cv2.COLOR_BGR2GRAY)
# #     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# #     edges   = cv2.Canny(blurred, 50, 150)
# #     kernel  = np.ones((2, 2), np.uint8)
# #     sketch  = cv2.bitwise_not(cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel))
# #     cv2.imwrite(str(output_dir / "5_sketch.png"), sketch)
# #     print("   Saved: 5_sketch.png")

# #     # Step 4 — Metrics
# #     print("\n[4] Computing metrics...")
# #     orig_resized = cv2.resize(original, (reconstructed.shape[1], reconstructed.shape[0]))

# #     ssim_score = compute_ssim(reconstructed, orig_resized)
# #     psnr_score = compute_psnr(reconstructed, orig_resized)
# #     mse_score  = compute_mse(reconstructed,  orig_resized)

# #     # IoU on occluded region only — most meaningful metric
# #     occ_mask_resized = cv2.resize(occ_mask, (reconstructed.shape[1], reconstructed.shape[0]))
# #     region_recon = cv2.bitwise_and(reconstructed, reconstructed, mask=occ_mask_resized)
# #     region_orig  = cv2.bitwise_and(orig_resized,  orig_resized,  mask=occ_mask_resized)

# #     region_recon_gray = cv2.cvtColor(region_recon, cv2.COLOR_BGR2GRAY)
# #     region_orig_gray  = cv2.cvtColor(region_oarig,  cv2.COLOR_BGR2GRAY)
# #     _, recon_bin = cv2.threshold(region_recon_gray, 10, 255, cv2.THRESH_BINARY)
# #     _, orig_bin  = cv2.threshold(region_orig_gray,  10, 255, cv2.THRESH_BINARY)
# #     region_iou   = compute_iou(recon_bin, orig_bin)

# #     # Edge IoU panel
# #     iou_panel, edge_iou = make_iou_panel(original, reconstructed)
# #     cv2.imwrite(str(output_dir / "6_iou_overlap.jpg"), iou_panel)

# #     print(f"\n{'='*45}")
# #     print(f"  EVALUATION RESULTS — {image_path.name}")
# #     print(f"{'='*45}")
# #     print(f"  SSIM          : {ssim_score:.4f}  (1.0 = perfect)")
# #     print(f"  PSNR          : {psnr_score:.2f} dB  (>20 = acceptable)")
# #     print(f"  MSE           : {mse_score:.2f}")
# #     print(f"  Region IoU    : {region_iou:.4f}  (occluded area only)")
# #     print(f"  Edge IoU      : {edge_iou:.4f}  (structure similarity)")
# #     print(f"{'='*45}")

# #     # Step 5 — Final 5-panel comparison
# #     SIZE = (400, 400)

# #     def panel(img, label, is_gray=False):
# #         if is_gray:
# #             img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# #         p = cv2.resize(img, SIZE)
# #         cv2.putText(p, label, (8, 28),
# #                     cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
# #         cv2.putText(p, label, (8, 28),
# #                     cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1)
# #         return p

# #     iou_panel_resized = cv2.resize(iou_panel, SIZE)
# #     # cv2.putText(iou_panel_resized, f"Edge IoU: {edge_iou:.3f}", (8, 28),
# #     #             cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

# #     comparison = np.hstack([
# #         panel(original,     "1. Original (GT)"),
# #         panel(occluded,     "2. Occluded Input"),
# #         panel(reconstructed,"3. Reconstruction"),
# #         panel(sketch,       "4. Sketch", is_gray=True),
# #         iou_panel_resized,
# #     ])

# #     # Add metrics bar at bottom
# #     metrics_bar = np.ones((50, comparison.shape[1], 3), dtype=np.uint8) * 40
# #     metrics_text = (f"SSIM: {ssim_score:.3f}   "
# #                     f"PSNR: {psnr_score:.1f}dB   "
# #                     f"Region IoU: {region_iou:.3f}   "
# #                     f"Edge IoU: {edge_iou:.3f}")
# #     cv2.putText(metrics_bar, metrics_text, (10, 33),
# #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

# #     final = np.vstack([comparison, metrics_bar])
# #     cv2.imwrite(str(output_dir / "7_final_comparison.jpg"), final)
# #     print(f"\n  Final comparison saved: {output_dir}/7_final_comparison.jpg")
# #     print(f"\nAll outputs saved to: {output_dir}")


# # if __name__ == "__main__":
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument("--image",      required=True,  help="Path to clean weapon image")
# #     parser.add_argument("--class_name", required=True,  help="Weapon class (grenade, knife, pistol...)")
# #     parser.add_argument("--output_dir", default="output/clean_test")
# #     parser.add_argument("--steps",      type=int, default=20)
# #     args = parser.parse_args()

# #     run_test(args.image, args.class_name, args.output_dir, args.steps)

# """
# test_clean_image.py
# -------------------
# Takes a clean weapon image, artificially occludes ONLY the weapon
# region using RT-DETR bounding box, runs reconstruction, then computes
# metrics against the original clean image for fair evaluation.

# Usage:
#     python evaluation/test_clean_image.py \
#         --image datasets/raw/grenade/grenade3.jpg \
#         --class_name grenade \
#         --output_dir output/clean_test
# """

# import cv2
# import sys
# import argparse
# import numpy as np
# from pathlib import Path

# sys.path.append(str(Path(__file__).parent))
# sys.path.append(str(Path(__file__).parent.parent))

# from metrics import compute_iou, compute_psnr, compute_ssim, compute_mse


# def get_weapon_bbox(image_path, rtdetr_model_path="models/rtdetr_weapon.pt", conf=0.20):
#     """
#     Use RT-DETR to detect weapon bounding box in the image.
#     Returns [x1, y1, x2, y2] or None if not detected.
#     """
#     try:
#         from detection.rtdetr_inference import load_model, run_inference, get_primary_weapon
#         model = load_model(rtdetr_model_path)
#         detections = run_inference(model, str(image_path), conf_threshold=conf)
#         primary = get_primary_weapon(detections)
#         if primary:
#             print(f"   RT-DETR detected: {primary['class_name']} "
#                   f"(conf: {primary['confidence']:.2f})")
#             return primary["bbox"]
#     except Exception as e:
#         print(f"   RT-DETR detection failed: {e}")
#     return None


# def apply_occlusion(image, occlusion_ratio=0.4, bbox=None):
#     """
#     Artificially occlude the weapon region in the image.

#     If bbox is provided: occludes only the right portion of the
#     weapon bounding box — intelligent, weapon-specific occlusion.

#     If bbox is None: falls back to simple right-side horizontal
#     occlusion of the full image.

#     Returns occluded image and exact occlusion mask.
#     """
#     h, w = image.shape[:2]
#     occluded = image.copy()
#     mask = np.zeros((h, w), dtype=np.uint8)

#     if bbox is not None:
#         x1, y1, x2, y2 = [int(v) for v in bbox]

#         # Clamp bbox to image bounds
#         x1 = max(0, x1)
#         y1 = max(0, y1)
#         x2 = min(w, x2)
#         y2 = min(h, y2)

#         bbox_w = x2 - x1

#         # Occlude right portion of the bounding box only
#         occ_x_start = x1 + int(bbox_w * (1 - occlusion_ratio))

#         occluded[y1:y2, occ_x_start:x2] = 0
#         mask[y1:y2, occ_x_start:x2] = 255

#         print(f"   Bbox occlusion: ({occ_x_start},{y1}) to ({x2},{y2})")
#         print(f"   Occluded {int(occlusion_ratio*100)}% of weapon bbox width")

#     else:
#         # Fallback — simple horizontal occlusion of full image
#         print("   No bbox found, using full image horizontal occlusion")
#         x_start = int(w * (1 - occlusion_ratio))
#         occluded[:, x_start:] = 0
#         mask[:, x_start:] = 255

#     return occluded, mask


# def make_iou_panel(original, reconstructed, size=(512, 512)):
#     """
#     Compare edges of original vs reconstructed.
#     Green = matched, Red = extra in recon, Blue = missing from recon.
#     Clean panel — no text overlay.
#     """
#     orig_r  = cv2.resize(original,      size)
#     recon_r = cv2.resize(reconstructed, size)

#     orig_gray  = cv2.cvtColor(orig_r,  cv2.COLOR_BGR2GRAY)
#     recon_gray = cv2.cvtColor(recon_r, cv2.COLOR_BGR2GRAY)

#     orig_edges  = cv2.Canny(cv2.GaussianBlur(orig_gray,  (5, 5), 0), 50, 150)
#     recon_edges = cv2.Canny(cv2.GaussianBlur(recon_gray, (5, 5), 0), 50, 150)

#     orig_bin  = orig_edges  > 0
#     recon_bin = recon_edges > 0

#     overlay = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255

#     tp = np.logical_and(recon_bin,  orig_bin)
#     fp = np.logical_and(recon_bin, ~orig_bin)
#     fn = np.logical_and(~recon_bin, orig_bin)

#     overlay[tp] = [0, 200, 0]   # green  = match
#     overlay[fp] = [0, 0, 220]   # red    = extra
#     overlay[fn] = [220, 0, 0]   # blue   = missing

#     intersection = tp.sum()
#     union = np.logical_or(recon_bin, orig_bin).sum()
#     edge_iou = intersection / union if union > 0 else 0.0

#     return overlay, edge_iou


# def run_test(image_path, class_name, output_dir, steps=20,
#              rtdetr_model_path="models/rtdetr_weapon.pt",
#              occlusion_ratio=0.4):
#     """
#     Full test pipeline on a clean image.
#     """
#     image_path = Path(image_path)
#     output_dir = Path(output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)

#     print(f"\nClean Image Test")
#     print(f"Image : {image_path.name}")
#     print(f"Class : {class_name}")
#     print("-" * 40)

#     # Load clean image
#     original = cv2.imread(str(image_path))
#     if original is None:
#         raise ValueError(f"Cannot read: {image_path}")

#     print(f"Image size: {original.shape[1]}x{original.shape[0]}")

#     # ── Step 1 — Detect weapon bbox then occlude ──────────────────────────────
#     print("\n[1] Detecting weapon bbox for intelligent occlusion...")
#     bbox = get_weapon_bbox(image_path, rtdetr_model_path, conf=0.20)

#     if bbox is not None:
#         print(f"   Using bbox-based occlusion: {bbox}")
#     else:
#         print("   Falling back to full-image horizontal occlusion")

#     occluded, occ_mask = apply_occlusion(
#         original,
#         occlusion_ratio=occlusion_ratio,
#         bbox=bbox
#     )

#     cv2.imwrite(str(output_dir / "1_original.jpg"),       original)
#     cv2.imwrite(str(output_dir / "2_occluded.jpg"),       occluded)
#     cv2.imwrite(str(output_dir / "3_occlusion_mask.png"), occ_mask)
#     print("   Saved: 1_original.jpg, 2_occluded.jpg, 3_occlusion_mask.png")

#     # ── Step 2 — SD Inpainting ────────────────────────────────────────────────
#     print("\n[2] Running SD Inpainting...")
#     sys.path.append(str(Path(__file__).parent.parent / "diffusion_completion"))
#     from inpaint import inpaint_weapon
#     from PIL import Image as PILImage

#     pil_occluded = PILImage.fromarray(cv2.cvtColor(occluded, cv2.COLOR_BGR2RGB))
#     pil_mask     = PILImage.fromarray(occ_mask)

#     results = inpaint_weapon(
#         crop_image=pil_occluded,
#         missing_mask=pil_mask,
#         class_name=class_name,
#         num_inference_steps=steps,
#         guidance_scale=7.5,
#         seed=42,
#     )

#     reconstructed_pil = results[0]
#     reconstructed = cv2.cvtColor(np.array(reconstructed_pil), cv2.COLOR_RGB2BGR)
#     reconstructed = cv2.resize(reconstructed, (original.shape[1], original.shape[0]))
#     cv2.imwrite(str(output_dir / "4_reconstructed.jpg"), reconstructed)
#     print("   Saved: 4_reconstructed.jpg")

#     # ── Step 3 — Forensic Sketch ──────────────────────────────────────────────
#     print("\n[3] Generating forensic sketch...")
#     gray    = cv2.cvtColor(reconstructed, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     edges   = cv2.Canny(blurred, 50, 150)
#     kernel  = np.ones((2, 2), np.uint8)
#     sketch  = cv2.bitwise_not(cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel))
#     cv2.imwrite(str(output_dir / "5_sketch.png"), sketch)
#     print("   Saved: 5_sketch.png")

#     # ── Step 4 — Metrics ──────────────────────────────────────────────────────
#     print("\n[4] Computing metrics...")
#     orig_resized = cv2.resize(original, (reconstructed.shape[1], reconstructed.shape[0]))

#     ssim_score = compute_ssim(reconstructed, orig_resized)
#     psnr_score = compute_psnr(reconstructed, orig_resized)
#     mse_score  = compute_mse(reconstructed,  orig_resized)

#     # IoU on occluded region only — most meaningful metric
#     occ_mask_resized  = cv2.resize(occ_mask, (reconstructed.shape[1], reconstructed.shape[0]))
#     region_recon      = cv2.bitwise_and(reconstructed, reconstructed, mask=occ_mask_resized)
#     region_orig       = cv2.bitwise_and(orig_resized,  orig_resized,  mask=occ_mask_resized)

#     region_recon_gray = cv2.cvtColor(region_recon, cv2.COLOR_BGR2GRAY)
#     region_orig_gray  = cv2.cvtColor(region_orig,  cv2.COLOR_BGR2GRAY)
#     _, recon_bin = cv2.threshold(region_recon_gray, 10, 255, cv2.THRESH_BINARY)
#     _, orig_bin  = cv2.threshold(region_orig_gray,  10, 255, cv2.THRESH_BINARY)
#     region_iou   = compute_iou(recon_bin, orig_bin)

#     # Edge IoU panel — clean, no text
#     iou_panel, edge_iou = make_iou_panel(original, reconstructed)
#     cv2.imwrite(str(output_dir / "6_iou_overlap.jpg"), iou_panel)

#     print(f"\n{'='*45}")
#     print(f"  EVALUATION RESULTS — {image_path.name}")
#     print(f"{'='*45}")
#     print(f"  SSIM          : {ssim_score:.4f}  (1.0 = perfect)")
#     print(f"  PSNR          : {psnr_score:.2f} dB  (>20 = acceptable)")
#     print(f"  MSE           : {mse_score:.2f}")
#     print(f"  Region IoU    : {region_iou:.4f}  (occluded area only)")
#     print(f"  Edge IoU      : {edge_iou:.4f}  (structure similarity)")
#     print(f"{'='*45}")

#     # ── Step 5 — Final 5-panel comparison ─────────────────────────────────────
#     SIZE = (400, 400)

#     def panel(img, label, is_gray=False):
#         if is_gray:
#             img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#         p = cv2.resize(img, SIZE)
#         cv2.putText(p, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
#         cv2.putText(p, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1)
#         return p

#     # IoU panel — clean, no text written on it
#     iou_panel_resized = cv2.resize(iou_panel, SIZE)

#     comparison = np.hstack([
#         panel(original,      "1. Original (GT)"),
#         panel(occluded,      "2. Occluded Input"),
#         panel(reconstructed, "3. Reconstruction"),
#         panel(sketch,        "4. Sketch", is_gray=True),
#         iou_panel_resized,
#     ])

#     # Metrics bar at bottom — dark background, white text
#     metrics_bar = np.ones((50, comparison.shape[1], 3), dtype=np.uint8) * 40
#     metrics_text = (f"SSIM: {ssim_score:.3f}   "
#                     f"PSNR: {psnr_score:.1f}dB   "
#                     f"Region IoU: {region_iou:.3f}   "
#                     f"Edge IoU: {edge_iou:.3f}")
#     cv2.putText(metrics_bar, metrics_text, (10, 33),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

#     final = np.vstack([comparison, metrics_bar])
#     cv2.imwrite(str(output_dir / "7_final_comparison.jpg"), final)
#     print(f"\n  Final comparison saved: {output_dir}/7_final_comparison.jpg")
#     print(f"\nAll outputs saved to: {output_dir}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--image",      required=True,  help="Path to clean weapon image")
#     parser.add_argument("--class_name", required=True,  help="Weapon class")
#     parser.add_argument("--output_dir", default="output/clean_test")
#     parser.add_argument("--steps",      type=int,   default=20)
#     parser.add_argument("--rtdetr",     default="models/rtdetr_weapon.pt")
#     parser.add_argument("--occ_ratio",  type=float, default=0.4)
#     args = parser.parse_args()

#     run_test(
#         image_path=args.image,
#         class_name=args.class_name,
#         output_dir=args.output_dir,
#         steps=args.steps,
#         rtdetr_model_path=args.rtdetr,
#         occlusion_ratio=args.occ_ratio,
#     )

"""
test_clean_image.py
-------------------
Takes a clean weapon image, uses RT-DETR to detect the weapon bbox
and class automatically, artificially occludes only the weapon region,
runs SD reconstruction, generates forensic sketch, and computes metrics.
"""

import cv2
import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from metrics import compute_iou, compute_psnr, compute_ssim, compute_mse


def detect_weapon(image_path, rtdetr_model_path="models/rtdetr_weapon.pt", conf=0.20):
    """
    Use RT-DETR to detect weapon. Returns (class_name, bbox) or (None, None).
    """
    try:
        from detection.rtdetr_inference import load_model, run_inference, get_primary_weapon
        model = load_model(rtdetr_model_path)
        detections = run_inference(model, str(image_path), conf_threshold=conf)
        primary = get_primary_weapon(detections)
        if primary:
            print(f"   RT-DETR detected: {primary['class_name']} (conf: {primary['confidence']:.2f})")
            return primary["class_name"], primary["bbox"]
    except Exception as e:
        print(f"   RT-DETR detection failed: {e}")
    return None, None


def apply_occlusion(image, occlusion_ratio=0.4, bbox=None):
    """
    Occlude the weapon region.
    If bbox provided: occludes right portion of weapon bbox only.
    If no bbox: falls back to full image horizontal occlusion.
    Returns occluded image and binary mask (white = hidden).
    """
    h, w = image.shape[:2]
    occluded = image.copy()
    mask = np.zeros((h, w), dtype=np.uint8)

    if bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        occ_x_start = x1 + int((x2 - x1) * (1 - occlusion_ratio))
        occluded[y1:y2, occ_x_start:x2] = 0
        mask[y1:y2, occ_x_start:x2] = 255
        print(f"   Occluded right {int(occlusion_ratio*100)}% of weapon bbox")
    else:
        print("   No bbox found — using full image horizontal occlusion")
        x_start = int(w * (1 - occlusion_ratio))
        occluded[:, x_start:] = 0
        mask[:, x_start:] = 255

    return occluded, mask


def make_iou_panel(original, reconstructed, size=(512, 512)):
    """
    Clean edge IoU visualization — no text on panel.
    Green = match, Red = extra, Blue = missing.
    """
    orig_r  = cv2.resize(original,      size)
    recon_r = cv2.resize(reconstructed, size)

    orig_edges  = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(orig_r,  cv2.COLOR_BGR2GRAY), (5, 5), 0), 50, 150)
    recon_edges = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(recon_r, cv2.COLOR_BGR2GRAY), (5, 5), 0), 50, 150)

    orig_bin  = orig_edges  > 0
    recon_bin = recon_edges > 0

    overlay = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255
    overlay[np.logical_and(recon_bin,  orig_bin)] = [0, 200, 0]
    overlay[np.logical_and(recon_bin, ~orig_bin)] = [0, 0, 220]
    overlay[np.logical_and(~recon_bin, orig_bin)] = [220, 0, 0]

    intersection = np.logical_and(recon_bin, orig_bin).sum()
    union = np.logical_or(recon_bin, orig_bin).sum()
    edge_iou = intersection / union if union > 0 else 0.0

    return overlay, edge_iou


def run_test(image_path, class_name=None, output_dir="output/clean_test",
             steps=20, rtdetr_model_path="models/rtdetr_weapon.pt",
             occlusion_ratio=0.4):
    """
    Full evaluation pipeline on a clean image.
    class_name is auto-detected by RT-DETR if not provided or set to 'auto'.
    Returns dict of metric scores.
    """
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nClean Image Test: {image_path.name}")
    print("-" * 40)

    original = cv2.imread(str(image_path))
    if original is None:
        raise ValueError(f"Cannot read: {image_path}")

    # ── Step 1 — Detect weapon ────────────────────────────────────────────────
    print("\n[1] Running RT-DETR detection...")
    detected_class, bbox = detect_weapon(image_path, rtdetr_model_path)

    if class_name is None or class_name == "auto":
        class_name = detected_class if detected_class else "firearm"
        if not detected_class:
            print(f"   No detection — using fallback class: {class_name}")

    print(f"   Using class: {class_name}")

    # ── Step 2 — Occlude ─────────────────────────────────────────────────────
    print(f"\n[2] Applying occlusion ({int(occlusion_ratio*100)}%)...")
    occluded, occ_mask = apply_occlusion(original, occlusion_ratio, bbox)

    cv2.imwrite(str(output_dir / "1_original.jpg"),       original)
    cv2.imwrite(str(output_dir / "2_occluded.jpg"),       occluded)
    cv2.imwrite(str(output_dir / "3_occlusion_mask.png"), occ_mask)

    # ── Step 3 — SD Inpainting ────────────────────────────────────────────────
    print("\n[3] Running SD Inpainting...")
    sys.path.append(str(Path(__file__).parent.parent / "diffusion_completion"))
    from inpaint import inpaint_weapon
    from PIL import Image as PILImage

    results = inpaint_weapon(
        crop_image=PILImage.fromarray(cv2.cvtColor(occluded, cv2.COLOR_BGR2RGB)),
        missing_mask=PILImage.fromarray(occ_mask),
        class_name=class_name,
        num_inference_steps=steps,
        guidance_scale=7.5,
        seed=42,
    )

    reconstructed = cv2.cvtColor(np.array(results[0]), cv2.COLOR_RGB2BGR)
    reconstructed = cv2.resize(reconstructed, (original.shape[1], original.shape[0]))
    cv2.imwrite(str(output_dir / "4_reconstructed.jpg"), reconstructed)

    # ── Step 4 — Forensic Sketch ──────────────────────────────────────────────
    print("\n[4] Generating forensic sketch...")
    edges  = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(reconstructed, cv2.COLOR_BGR2GRAY), (5, 5), 0), 50, 150)
    sketch = cv2.bitwise_not(cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8)))
    cv2.imwrite(str(output_dir / "5_sketch.png"), sketch)

    # ── Step 5 — Metrics ──────────────────────────────────────────────────────
    print("\n[5] Computing metrics...")
    orig_r = cv2.resize(original, (reconstructed.shape[1], reconstructed.shape[0]))

    ssim_score = compute_ssim(reconstructed, orig_r)
    psnr_score = compute_psnr(reconstructed, orig_r)
    mse_score  = compute_mse(reconstructed,  orig_r)

    occ_r = cv2.resize(occ_mask, (reconstructed.shape[1], reconstructed.shape[0]))
    region_recon = cv2.bitwise_and(reconstructed, reconstructed, mask=occ_r)
    region_orig  = cv2.bitwise_and(orig_r, orig_r, mask=occ_r)
    _, rb = cv2.threshold(cv2.cvtColor(region_recon, cv2.COLOR_BGR2GRAY), 10, 255, cv2.THRESH_BINARY)
    _, ob = cv2.threshold(cv2.cvtColor(region_orig,  cv2.COLOR_BGR2GRAY), 10, 255, cv2.THRESH_BINARY)
    region_iou = compute_iou(rb, ob)

    iou_panel, edge_iou = make_iou_panel(original, reconstructed)
    cv2.imwrite(str(output_dir / "6_iou_overlap.jpg"), iou_panel)

    print(f"\n{'='*45}")
    print(f"  RESULTS — {image_path.name}")
    print(f"{'='*45}")
    print(f"  Class         : {class_name}")
    print(f"  SSIM          : {ssim_score:.4f}  (1.0 = perfect)")
    print(f"  PSNR          : {psnr_score:.2f} dB")
    print(f"  MSE           : {mse_score:.2f}")
    print(f"  Region IoU    : {region_iou:.4f}")
    print(f"  Edge IoU      : {edge_iou:.4f}")
    print(f"{'='*45}")

    # ── Step 6 — Final comparison image ───────────────────────────────────────
    SIZE = (400, 400)

    def panel(img, label, is_gray=False):
        if is_gray:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        p = cv2.resize(img, SIZE)
        cv2.putText(p, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(p, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1)
        return p

    comparison = np.hstack([
        panel(original,      "1. Original (GT)"),
        panel(occluded,      "2. Occluded Input"),
        panel(reconstructed, "3. Reconstruction"),
        panel(sketch,        "4. Sketch", is_gray=True),
        cv2.resize(iou_panel, SIZE),
    ])

    metrics_bar = np.ones((50, comparison.shape[1], 3), dtype=np.uint8) * 40
    cv2.putText(metrics_bar,
                f"Class: {class_name}   SSIM: {ssim_score:.3f}   PSNR: {psnr_score:.1f}dB   Region IoU: {region_iou:.3f}   Edge IoU: {edge_iou:.3f}",
                (10, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    cv2.imwrite(str(output_dir / "7_final_comparison.jpg"), np.vstack([comparison, metrics_bar]))
    print(f"\n  All outputs saved to: {output_dir}")

    return {
        "class_name": class_name,
        "ssim":       ssim_score,
        "psnr":       psnr_score,
        "mse":        mse_score,
        "region_iou": region_iou,
        "edge_iou":   edge_iou,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",      required=True)
    parser.add_argument("--class_name", default="auto", help="Weapon class or 'auto' to detect")
    parser.add_argument("--output_dir", default="output/clean_test")
    parser.add_argument("--steps",      type=int,   default=20)
    parser.add_argument("--rtdetr",     default="models/rtdetr_weapon.pt")
    parser.add_argument("--occ_ratio",  type=float, default=0.4)
    args = parser.parse_args()

    run_test(
        image_path=args.image,
        class_name=args.class_name,
        output_dir=args.output_dir,
        steps=args.steps,
        rtdetr_model_path=args.rtdetr,
        occlusion_ratio=args.occ_ratio,
    )