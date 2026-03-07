"""
app.py
------
Streamlit demo for the weapon reconstruction pipeline.

Run:
    cd weapon_project
    streamlit run app_demo/app.py

Features:
    - Upload any weapon image
    - See RT-DETR detection result with bounding box
    - See SAM segmentation mask overlaid
    - See SD Inpainting reconstruction
    - See forensic sketch
    - Download all outputs
    - Evaluation metrics vs ground truth (optional)
"""

import sys
import cv2
import json
import numpy as np
from pathlib import Path
from PIL import Image
import streamlit as st

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Weapon Reconstruction Pipeline",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS styling ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title { font-size: 2rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0.2rem; }
    .sub-title  { font-size: 1rem; color: #555; margin-bottom: 2rem; }
    .step-header { background: #f0f4ff; padding: 0.5rem 1rem;
                   border-left: 4px solid #4a6cf7; border-radius: 4px;
                   font-weight: 600; margin: 1rem 0 0.5rem 0; }
    .metric-box { background: #f8f9fa; padding: 1rem; border-radius: 8px;
                  text-align: center; border: 1px solid #dee2e6; }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #4a6cf7; }
    .metric-label { font-size: 0.85rem; color: #666; }
    .success-banner { background: #d4edda; color: #155724; padding: 0.75rem 1rem;
                      border-radius: 6px; border: 1px solid #c3e6cb; }
    .error-banner   { background: #f8d7da; color: #721c24; padding: 0.75rem 1rem;
                      border-radius: 6px; border: 1px solid #f5c6cb; }
</style>
""", unsafe_allow_html=True)


# ── Load pipeline (cached — only loads models once) ───────────────────────────
@st.cache_resource
def get_pipeline(rtdetr_path: str, sam_path: str, sd_steps: int, sketch_style: str):
    from pipeline import WeaponReconstructionPipeline
    config = {
        "rtdetr_model":   rtdetr_path,
        "sam_model":      sam_path,
        "sd_steps":       sd_steps,
        "sketch_style":   sketch_style,
        "save_intermediates": True,
    }
    pipe = WeaponReconstructionPipeline(config)
    pipe.load_models()
    return pipe


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    rtdetr_path = st.text_input(
        "RT-DETR Model Path",
        value="models/rtdetr_weapon.pt",
        help="Path to your trained RT-DETR .pt weights file"
    )
    sam_path = st.text_input(
        "SAM Model Path",
        value="models/mobile_sam.pt",
        help="Path to MobileSAM or SAM checkpoint"
    )

    st.markdown("---")
    st.markdown("**Inpainting Settings**")
    sd_steps = st.slider("SD Inference Steps", 10, 50, 30,
                         help="More steps = better quality but slower")
    guidance = st.slider("Guidance Scale", 3.0, 15.0, 7.5, step=0.5,
                         help="How closely to follow the text prompt")
    seed = st.number_input("Seed", value=42, help="For reproducibility")
    conf_threshold = st.slider("Detection Confidence", 0.1, 0.9, 0.35,
                               help="RT-DETR confidence threshold")

    st.markdown("---")
    st.markdown("**Sketch Style**")
    sketch_style = st.selectbox("Style", ["forensic", "pencil", "technical"],
                                help="Forensic = clean lines, Pencil = softer, Technical = precise")

    st.markdown("---")
    st.markdown("**About**")
    st.markdown("""
    **Pipeline:**
    1. RT-DETR → detect weapon
    2. SAM → segment visible region
    3. Stable Diffusion → reconstruct
    4. Canny → forensic sketch

    *Capstone Project — CSE*
    """)


# ── Main UI ───────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🔍 Forensic Weapon Reconstruction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Class-Conditioned Diffusion-Based Reconstruction of Occluded Weapons</div>', unsafe_allow_html=True)

# Upload
uploaded_file = st.file_uploader(
    "Upload a weapon image (occluded or partial)",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
)

if uploaded_file is not None:
    # Save upload to temp file
    import tempfile, os
    tmp_dir = Path(tempfile.mkdtemp())
    tmp_img  = tmp_dir / uploaded_file.name
    with open(tmp_img, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Show original
    st.markdown('<div class="step-header">📥 Input Image</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(str(tmp_img), caption="Uploaded image", use_column_width=True)

    # Run pipeline
    if st.button("🚀 Run Pipeline", type="primary", use_container_width=True):
        with st.spinner("Loading models (first run downloads ~4GB)..."):
            try:
                pipeline = get_pipeline(rtdetr_path, sam_path, sd_steps, sketch_style)
            except Exception as e:
                st.markdown(f'<div class="error-banner">❌ Model loading failed: {e}</div>',
                            unsafe_allow_html=True)
                st.stop()

        output_dir = str(tmp_dir / "results")

        # Override config for this run
        pipeline.config.update({
            "conf_threshold": conf_threshold,
            "sd_steps": sd_steps,
            "sd_guidance": guidance,
            "sd_seed": int(seed),
            "sketch_style": sketch_style,
        })

        with st.spinner("Running pipeline... (inpainting may take 1-3 min on M1)"):
            result = pipeline.process_image(str(tmp_img), output_dir)

        if not result["success"]:
            st.markdown(f'<div class="error-banner">❌ {result["error"]}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="success-banner">✅ Pipeline complete in {result["elapsed"]:.1f}s</div>',
                        unsafe_allow_html=True)

            # Detection info
            st.markdown('<div class="step-header">🎯 Step 1: RT-DETR Detection</div>',
                        unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f'<div class="metric-box"><div class="metric-value">{result["class_name"].upper()}</div><div class="metric-label">Detected Class</div></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="metric-box"><div class="metric-value">{result["confidence"]:.1%}</div><div class="metric-label">Confidence</div></div>', unsafe_allow_html=True)
            with m3:
                bbox = result["bbox"]
                st.markdown(f'<div class="metric-box"><div class="metric-value">{bbox[2]-bbox[0]}×{bbox[3]-bbox[1]}</div><div class="metric-label">BBox Size (px)</div></div>', unsafe_allow_html=True)

            # Visual pipeline results
            st.markdown('<div class="step-header">🖼️ Pipeline Results</div>', unsafe_allow_html=True)

            paths = result["paths"]
            col_a, col_b, col_c = st.columns(3)

            with col_a:
                st.markdown("**Step 2: SAM Segmentation**")
                if "crop" in paths:
                    st.image(paths["crop"], caption="512×512 weapon crop", use_column_width=True)
                if "mask_missing" in paths:
                    st.image(paths["mask_missing"], caption="Missing region mask (white = inpaint)", use_column_width=True)

            with col_b:
                st.markdown("**Step 3: SD Inpainting**")
                if "reconstructed" in paths:
                    st.image(paths["reconstructed"], caption="Reconstructed full weapon", use_column_width=True)

            with col_c:
                st.markdown("**Step 4: Forensic Sketch**")
                if "sketch" in paths:
                    st.image(paths["sketch"], caption=f"Forensic sketch ({sketch_style} style)", use_column_width=True)

            # Comparison sheet
            if "comparison" in paths:
                st.markdown('<div class="step-header">📊 Full Comparison</div>', unsafe_allow_html=True)
                st.image(paths["comparison"], caption="Input → Reconstruction → Sketch", use_column_width=True)

            # Optional evaluation vs ground truth
            st.markdown('<div class="step-header">📈 Evaluation (Optional)</div>', unsafe_allow_html=True)
            gt_file = st.file_uploader(
                "Upload ground truth (original un-occluded image) for metrics",
                type=["jpg", "jpeg", "png"],
                key="gt_upload"
            )
            if gt_file and "reconstructed" in paths:
                gt_tmp = tmp_dir / f"gt_{gt_file.name}"
                with open(gt_tmp, "wb") as f:
                    f.write(gt_file.getbuffer())

                try:
                    sys.path.insert(0, str(Path(__file__).parent.parent))
                    from evaluation.metrics import evaluate_sample
                    scores = evaluate_sample(
                        paths["reconstructed"],
                        str(gt_tmp),
                        sketch_path=paths.get("sketch"),
                    )
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    mc1.metric("SSIM",     f"{scores['ssim']:.4f}",    help="1.0 = perfect")
                    mc2.metric("PSNR",     f"{scores['psnr']:.1f} dB", help=">30dB = good")
                    mc3.metric("L1 Loss",  f"{scores['l1_loss']:.4f}", help="0 = perfect")
                    if "edge_f1" in scores:
                        mc4.metric("Edge F1", f"{scores['edge_f1']:.4f}", help="Sketch edge quality")
                except Exception as e:
                    st.warning(f"Metrics computation failed: {e}")

            # Downloads
            st.markdown('<div class="step-header">💾 Downloads</div>', unsafe_allow_html=True)
            dcols = st.columns(3)
            for i, (label, key) in enumerate([
                ("Download Reconstruction", "reconstructed"),
                ("Download Sketch",         "sketch"),
                ("Download Comparison",     "comparison"),
            ]):
                if key in paths:
                    with open(paths[key], "rb") as f:
                        dcols[i].download_button(
                            label=label,
                            data=f,
                            file_name=Path(paths[key]).name,
                            mime="image/jpeg" if key != "sketch" else "image/png",
                        )

else:
    # Landing state
    st.info("👆 Upload a weapon image to begin. The pipeline will detect, segment, reconstruct, and sketch it.")

    st.markdown("---")
    st.markdown("### How it works")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown("**1. RT-DETR**\nDetects weapon class and bounding box")
    c2.markdown("**2. SAM**\nSegments the visible weapon pixels")
    c3.markdown("**3. Stable Diffusion**\nRecostructs missing/occluded region")
    c4.markdown("**4. Canny Edge**\nGenerates forensic sketch output")
