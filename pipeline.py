"""
pipeline.py
-----------
End-to-end weapon reconstruction pipeline with adaptive canvas expansion.

Flow:
Image → RT-DETR detect → SAM visible mask → Adaptive canvas expansion
→ Stable Diffusion reconstruction → Forensic sketch
"""

import argparse
import sys
import cv2
import json
import time
import numpy as np
from pathlib import Path
from PIL import Image
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from detection.rtdetr_inference import load_model, run_inference, get_primary_weapon
from segmentation.sam_segmentation import load_sam_model, segment_weapon
from segmentation.adaptive_crop import compute_adaptive_crop
from diffusion_completion.inpaint import load_inpainting_pipeline, inpaint_weapon
from sketch_generation.sketch import image_to_forensic_sketch, generate_comparison_sheet


DEFAULT_CONFIG = {
    "rtdetr_model": "models/rtdetr_weapon.pt",
    "sam_model": "models/mobile_sam.pt",
    "conf_threshold": 0.35,
    "sam_padding": 20,
    "sd_steps": 30,
    "sd_guidance": 7.5,
    "sd_seed": 42,
    "sketch_style": "forensic",
    "save_intermediates": True,
}


class WeaponReconstructionPipeline:

    def __init__(self, config=None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.rtdetr_model = None
        self.sam_predictor = None
        self.sd_pipeline = None
        self._models_loaded = False

    def load_models(self):

        if self._models_loaded:
            return

        print("\n============================================================")
        print("  Loading Weapon Reconstruction Pipeline Models")
        print("============================================================")

        print("\n[1/3] Loading RT-DETR...")
        self.rtdetr_model = load_model(self.config["rtdetr_model"])

        print("\n[2/3] Loading SAM...")
        self.sam_predictor = load_sam_model(self.config["sam_model"])

        print("\n[3/3] Loading Stable Diffusion...")
        self.sd_pipeline = load_inpainting_pipeline()

        self._models_loaded = True
        print("\n✅ All models loaded. Pipeline ready.\n")

    def process_image(self, image_path, output_dir="output/results"):

        if not self._models_loaded:
            self.load_models()

        image_path = Path(image_path)
        if not image_path.exists():
            return {"success": False, "error": "Image not found"}

        stem = image_path.stem
        out_dir = Path(output_dir) / stem
        out_dir.mkdir(parents=True, exist_ok=True)

        t_start = time.time()

        print(f"\n[Pipeline] Processing: {image_path.name}")

        image_bgr = cv2.imread(str(image_path))

        # --------------------------------------------------------
        # STEP 1 — DETECTION
        # --------------------------------------------------------

        print("[Step 1] RT-DETR detection...")

        detections = run_inference(
            self.rtdetr_model,
            str(image_path),
            conf_threshold=self.config["conf_threshold"],
        )

        primary = get_primary_weapon(detections)

        if primary is None:
            print("No weapon detected.")
            return {"success": False}

        print(f"Detected: {primary['class_name']} ({primary['confidence']:.3f})")

        bbox = primary["bbox"]

        # --------------------------------------------------------
        # STEP 2 — SAM SEGMENTATION
        # --------------------------------------------------------

        print("[Step 2] SAM segmentation...")

        seg = segment_weapon(
            self.sam_predictor,
            image_bgr,
            bbox,
            padding=self.config["sam_padding"],
        )

        print(f"SAM score: {seg['sam_score']:.3f}")

        # --------------------------------------------------------
        # STEP 3 — ADAPTIVE CANVAS EXPANSION
        # --------------------------------------------------------

        print("[Step 3] Adaptive canvas expansion...")

        crop512, visible_mask, missing_mask = compute_adaptive_crop(
            image_bgr,
            bbox,
            expansion_factor=2.8,
            target_size=512,
        )

        # --------------------------------------------------------
        # SAVE INTERMEDIATE FILES
        # --------------------------------------------------------

        paths = {}

        if self.config["save_intermediates"]:

            crop_path = out_dir / f"{stem}_crop512.jpg"
            mask_vis_path = out_dir / f"{stem}_mask_visible.png"
            mask_mis_path = out_dir / f"{stem}_mask_missing.png"

            cv2.imwrite(str(crop_path), crop512)
            cv2.imwrite(str(mask_vis_path), visible_mask)
            cv2.imwrite(str(mask_mis_path), missing_mask)

            paths["crop"] = str(crop_path)
            paths["mask_visible"] = str(mask_vis_path)
            paths["mask_missing"] = str(mask_mis_path)

        # --------------------------------------------------------
        # STEP 4 — DIFFUSION RECONSTRUCTION
        # --------------------------------------------------------

        print("[Step 4] Stable Diffusion reconstruction...")

        reconstructed_list = inpaint_weapon(
            crop_image=crop512,
            missing_mask=missing_mask,
            class_name=primary["class_name"],
            num_inference_steps=self.config["sd_steps"],
            guidance_scale=self.config["sd_guidance"],
            seed=self.config["sd_seed"],
        )

        reconstructed = reconstructed_list[0]

        recon_path = out_dir / f"{stem}_reconstructed.jpg"
        reconstructed.save(str(recon_path))

        paths["reconstructed"] = str(recon_path)

        print("Reconstruction saved.")

        # --------------------------------------------------------
        # STEP 5 — FORENSIC SKETCH
        # --------------------------------------------------------

        print("[Step 5] Forensic sketch generation...")

        recon_bgr = cv2.cvtColor(np.array(reconstructed), cv2.COLOR_RGB2BGR)

        sketch_data = image_to_forensic_sketch(
            recon_bgr,
            style=self.config["sketch_style"],
        )

        sketch = sketch_data["sketch"]

        sketch_path = out_dir / f"{stem}_sketch.png"
        cv2.imwrite(str(sketch_path), sketch)

        paths["sketch"] = str(sketch_path)

        # --------------------------------------------------------
        # STEP 6 — COMPARISON IMAGE
        # --------------------------------------------------------

        comparison_path = out_dir / f"{stem}_comparison.jpg"

        generate_comparison_sheet(
            original=image_bgr,
            reconstructed=recon_bgr,
            sketch=sketch,
            output_path=str(comparison_path),
        )

        paths["comparison"] = str(comparison_path)

        # --------------------------------------------------------
        # METADATA
        # --------------------------------------------------------

        elapsed = time.time() - t_start

        metadata = {
            "image": str(image_path),
            "class": primary["class_name"],
            "confidence": primary["confidence"],
            "bbox": bbox,
            "sam_score": seg["sam_score"],
            "processing_time": elapsed,
            "timestamp": datetime.now().isoformat(),
        }

        with open(out_dir / f"{stem}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\n✅ Done in {elapsed:.1f}s")
        print("Output folder:", out_dir)

        return {"success": True, "paths": paths}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--image")
    parser.add_argument("--output_dir", default="output/results")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--steps", type=int, default=30)

    args = parser.parse_args()

    config = {
        "conf_threshold": args.conf,
        "sd_steps": args.steps,
    }

    pipeline = WeaponReconstructionPipeline(config)

    pipeline.load_models()

    pipeline.process_image(args.image, args.output_dir)