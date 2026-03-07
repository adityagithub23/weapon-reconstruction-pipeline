# Weapon Reconstruction Pipeline — Setup & Usage Guide

## Your 6-Class Dataset
Classes trained: fire(0), firearm(1), grenade(2), knife(3), pistol(4), rocket(5)
Model: RT-DETR-L  |  mAP50 ~0.85  |  mAP50-95 ~0.65  ✅

---

## Step 1: Mac M1 Environment Setup

```bash
# Create and activate virtual environment (recommended)
python3 -m venv weapon_env
source weapon_env/bin/activate

# Core dependencies
pip install torch torchvision          # MPS backend included
pip install ultralytics                # RT-DETR via YOLO framework
pip install diffusers transformers accelerate   # Stable Diffusion
pip install opencv-python numpy Pillow
pip install scikit-image               # SSIM/PSNR metrics
pip install streamlit                  # Demo app

# SAM — choose ONE option:

# Option A: MobileSAM (RECOMMENDED for M1 — fast, 40MB)
pip install git+https://github.com/ChaoningZhang/MobileSAM.git
# Then download weights:
mkdir -p models
curl -L https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt \
     -o models/mobile_sam.pt

# Option B: Original SAM ViT-B (larger, ~375MB)
pip install git+https://github.com/facebookresearch/segment-anything.git
curl -L https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \
     -o models/sam_vit_b.pth

# Verify MPS
python3 -c "import torch; print('MPS:', torch.backends.mps.is_available())"
# Should print: MPS: True
```

---

## Step 2: Place Your Model Weights

```
weapon_project/
└── models/
    ├── rtdetr_weapon.pt    ← COPY YOUR TRAINED RT-DETR WEIGHTS HERE
    └── mobile_sam.pt       ← Downloaded by setup above
```

Download your RT-DETR weights from Kaggle and place at `models/rtdetr_weapon.pt`.

---

## Step 3: Organize Your Dataset

Your dataset should follow this structure:
```
weapon_project/
└── datasets/
    ├── raw/
    │   ├── pistol/       ← clean weapon images
    │   ├── firearm/
    │   ├── knife/
    │   ├── grenade/
    │   ├── fire/
    │   └── rocket/
    └── labels/
        ├── pistol/       ← matching YOLO .txt annotation files
        ├── firearm/
        └── ...
```

---

## Step 4: Generate Occluded Test Images

```bash
cd weapon_project
python segmentation/generate_occlusions.py \
    --images_dir datasets/raw \
    --labels_dir datasets/labels \
    --output_dir datasets/occluded \
    --num_occlusions 3
```

---

## Step 5: Test RT-DETR Detection

```bash
python detection/rtdetr_inference.py \
    --image datasets/raw/pistol/some_image.jpg \
    --model models/rtdetr_weapon.pt \
    --output output/test_detection.jpg
```

---

## Step 6: Test SAM Segmentation

```bash
# Use the bbox output from Step 5
python segmentation/sam_segmentation.py \
    --image datasets/raw/pistol/some_image.jpg \
    --bbox "100,150,400,500" \
    --model models/mobile_sam.pt \
    --output_dir output/test_segmentation
```

---

## Step 7: Test Inpainting

```bash
# Use crop and mask from Step 6
python diffusion_completion/inpaint.py \
    --image output/test_segmentation/some_image_crop512.jpg \
    --mask  output/test_segmentation/some_image_mask_missing.png \
    --class_name pistol \
    --output output/test_reconstruction/result.jpg
# NOTE: First run downloads ~4GB from HuggingFace — needs internet connection
```

---

## Step 8: Test Sketch Generation

```bash
python sketch_generation/sketch.py \
    --image output/test_reconstruction/result.jpg \
    --output output/test_sketch/sketch.png \
    --style forensic \
    --show_steps
```

---

## Step 9: Run Full Pipeline (End-to-End)

```bash
# Single image
python pipeline.py \
    --image datasets/occluded/pistol/some_image_occ0.jpg \
    --output_dir output/results

# Batch mode (all images in a directory)
python pipeline.py \
    --batch_dir datasets/occluded/pistol \
    --output_dir output/batch_results
```

Output per image:
```
output/results/<stem>/
├── <stem>_crop512.jpg          ← 512×512 weapon crop
├── <stem>_mask_visible.png     ← SAM visible mask
├── <stem>_mask_missing.png     ← Region to inpaint
├── <stem>_reconstructed.jpg    ← SD Inpainting result  ⭐
├── <stem>_sketch.png           ← Forensic sketch       ⭐
├── <stem>_comparison.jpg       ← 3-panel comparison    ⭐
└── <stem>_metadata.json        ← Detection + timing info
```

---

## Step 10: Evaluate Results

```bash
# Single image
python evaluation/metrics.py \
    --reconstructed output/results/img001/img001_reconstructed.jpg \
    --ground_truth  datasets/raw/pistol/img001.jpg \
    --sketch        output/results/img001/img001_sketch.png

# Batch evaluation → saves CSV
python evaluation/metrics.py \
    --batch_results_dir output/batch_results \
    --ground_truth_dir  datasets/raw \
    --csv_output        evaluation/results.csv
```

---

## Step 11: Run Streamlit Demo

```bash
cd weapon_project
streamlit run app_demo/app.py
# Opens at http://localhost:8501
```

---

## Troubleshooting

### MPS out of memory during SD Inpainting
```python
# In diffusion_completion/inpaint.py, the pipeline already has attention slicing enabled.
# If you still get OOM, reduce steps:
python pipeline.py --image ... --steps 20
```

### SAM model not found
```bash
# Re-download:
curl -L https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt \
     -o models/mobile_sam.pt
```

### RT-DETR confidence too low on occluded images
```bash
# Lower the threshold:
python pipeline.py --image ... --conf 0.20
```

### SD Inpainting first-run download fails
The model downloads from HuggingFace (~4GB). Ensure stable internet.
Once downloaded, it's cached at `~/.cache/huggingface/` and never downloads again.

---

## Project Structure

```
weapon_project/
├── models/
│   ├── rtdetr_weapon.pt         ← YOUR MODEL HERE
│   └── mobile_sam.pt            ← SAM weights
├── datasets/
│   ├── raw/                     ← clean weapon images (by class)
│   ├── occluded/                ← artificially occluded versions
│   ├── labels/                  ← YOLO .txt annotations
│   └── masks/                   ← SAM-generated masks (optional cache)
├── detection/
│   └── rtdetr_inference.py      ← Step 1: detect + classify
├── segmentation/
│   ├── generate_occlusions.py   ← Occlusion data augmentation
│   └── sam_segmentation.py      ← Step 2: SAM masking
├── diffusion_completion/
│   └── inpaint.py               ← Step 3: SD reconstruction
├── sketch_generation/
│   └── sketch.py                ← Step 4: forensic sketch
├── evaluation/
│   └── metrics.py               ← SSIM, PSNR, IoU, Dice, Edge F1
├── app_demo/
│   └── app.py                   ← Streamlit interactive demo
├── pipeline.py                  ← End-to-end runner
└── README.md                    ← This file
```

---

## Ethical Statement (For Faculty)
- Dataset used for academic forensic research only
- No weapon blueprint generation — focus is reconstruction under occlusion
- All models are publicly available pretrained checkpoints
- Intended use: forensic analysis, security research, academic publication
