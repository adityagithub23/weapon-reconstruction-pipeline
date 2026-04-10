"""
app.py
------
Streamlit demo for the weapon reconstruction pipeline.
RT-DETR auto-detects weapon class — no manual selection needed.

Run:
    cd "weapon_project copy"
    streamlit run app_demo/app.py
"""

import sys
import time
import cv2
import numpy as np
from pathlib import Path
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="Forensic Weapon Reconstruction",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-title   { font-size: 2rem; font-weight: 700; color: #D1D1F1; margin-bottom: 0.2rem; }
    .sub-title    { font-size: 1rem; color: #555; margin-bottom: 2rem; }
    .step-header  { background: #f0f4ff; padding: 0.5rem 1rem;
                    border-left: 4px solid #4a6cf7; border-radius: 4px;
                    font-weight: 600; margin: 1rem 0 0.5rem 0; }
    .metric-box   { background: #f8f9fa; padding: 1rem; border-radius: 8px;
                    text-align: center; border: 1px solid #dee2e6; }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #4a6cf7; }
    .metric-label { font-size: 0.85rem; color: #666; }
    .success-box  { background: #d4edda; color: #155724; padding: 0.75rem 1rem;
                    border-radius: 6px; border: 1px solid #c3e6cb; }
    .error-box    { background: #f8d7da; color: #721c24; padding: 0.75rem 1rem;
                    border-radius: 6px; border: 1px solid #f5c6cb; }
    .info-box     { background: #d1ecf1; color: #0c5460; padding: 0.75rem 1rem;
                    border-radius: 6px; border: 1px solid #bee5eb; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    sd_steps = st.slider("SD Inference Steps", 10, 50, 20,
                         help="More steps = better quality but slower")
    occlusion_ratio = st.slider("Occlusion Ratio", 0.2, 0.8, 0.4, step=0.1,
                                help="How much of the weapon to occlude (0.4 = 40%)")

    st.markdown("---")
    st.markdown("""
**Pipeline:**
1. Upload clean weapon image
2. RT-DETR auto-detects class + bbox
3. App occludes only the weapon region
4. SD Inpainting reconstructs hidden part
5. Canny generates forensic sketch
6. Metrics vs original are computed

*Capstone Project — CSE*
""")


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🔍 Forensic Weapon Reconstruction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Class-Conditioned Diffusion-Based Reconstruction of Occluded Weapons</div>', unsafe_allow_html=True)

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a clean weapon image — RT-DETR will auto-detect the class",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
)

if uploaded_file is not None:

    # Save to permanent output folder named after the image
    project_root = Path(__file__).parent.parent
    stem = Path(uploaded_file.name).stem
    output_dir = project_root / "output" / "results" / stem
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = output_dir / uploaded_file.name
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Show uploaded image
    st.markdown('<div class="step-header">📥 Uploaded Image</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(str(input_path), caption=uploaded_file.name, use_container_width=True)

    st.markdown(
        f'<div class="info-box">ℹ️ RT-DETR will auto-detect the weapon class. '
        f'Will occlude <b>{int(occlusion_ratio*100)}%</b> of the weapon region.</div>',
        unsafe_allow_html=True
    )

    # ── Run ───────────────────────────────────────────────────────────────────
    if st.button("🚀 Run Reconstruction & Evaluation", type="primary", use_container_width=True):

        t_start = time.time()
        scores = None

        with st.spinner("Running... SD inpainting takes ~10 min on M1"):
            try:
                eval_path = str(project_root / "evaluation")
                if eval_path not in sys.path:
                    sys.path.insert(0, eval_path)
                from test_clean_image import run_test

                scores = run_test(
                    image_path=str(input_path),
                    class_name="auto",
                    output_dir=str(output_dir),
                    steps=sd_steps,
                    occlusion_ratio=occlusion_ratio,
                )
                run_ok = True
            except Exception as e:
                import traceback
                run_ok = False
                err_msg = traceback.format_exc()

        elapsed = time.time() - t_start

        if not run_ok:
            st.markdown(f'<div class="error-box">❌ Pipeline failed:<br><pre>{err_msg}</pre></div>',
                        unsafe_allow_html=True)
            st.stop()

        final_comparison = output_dir / "7_final_comparison.jpg"
        if not final_comparison.exists():
            st.markdown('<div class="error-box">❌ No output generated — weapon may not have been detected.</div>',
                        unsafe_allow_html=True)
            st.stop()

        detected_class = scores.get("class_name", "unknown") if scores else "unknown"
        st.markdown(
            f'<div class="success-box">✅ Done in {elapsed:.1f}s — Detected class: <b>{detected_class}</b></div>',
            unsafe_allow_html=True
        )

        # ── Step by step ──────────────────────────────────────────────────────
        st.markdown('<div class="step-header">🖼️ Step by Step Results</div>', unsafe_allow_html=True)

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.markdown("**Original → Occluded**")
            p1 = output_dir / "1_original.jpg"
            p2 = output_dir / "2_occluded.jpg"
            if p1.exists(): st.image(str(p1), caption="1. Original (Ground Truth)", use_container_width=True)
            if p2.exists(): st.image(str(p2), caption="2. Occluded Input",          use_container_width=True)

        with col_b:
            st.markdown("**Reconstruction**")
            p4 = output_dir / "4_reconstructed.jpg"
            p3 = output_dir / "3_occlusion_mask.png"
            if p4.exists(): st.image(str(p4), caption="3. SD Reconstructed Weapon", use_container_width=True)
            if p3.exists(): st.image(str(p3), caption="Occlusion Mask",              use_container_width=True)

        with col_c:
            st.markdown("**Sketch & IoU**")
            p5 = output_dir / "5_sketch.png"
            p6 = output_dir / "6_iou_overlap.jpg"
            if p5.exists(): st.image(str(p5), caption="4. Forensic Sketch",                                    use_container_width=True)
            if p6.exists(): st.image(str(p6), caption="5. IoU Overlap (Green=Match, Red=Extra, Blue=Missing)", use_container_width=True)

        # ── Final comparison ──────────────────────────────────────────────────
        st.markdown('<div class="step-header">📊 Final Comparison</div>', unsafe_allow_html=True)
        st.image(str(final_comparison),
                 caption="Original → Occluded → Reconstruction → Sketch → IoU Overlap | Metrics bar at bottom",
                 use_container_width=True)

        # ── Metrics ───────────────────────────────────────────────────────────
        if scores:
            st.markdown('<div class="step-header">📈 Evaluation Metrics</div>', unsafe_allow_html=True)
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.markdown(f'<div class="metric-box"><div class="metric-value">{detected_class}</div><div class="metric-label">Detected Class</div></div>', unsafe_allow_html=True)
            m2.markdown(f'<div class="metric-box"><div class="metric-value">{scores["ssim"]:.3f}</div><div class="metric-label">SSIM<br>(1.0 = perfect)</div></div>', unsafe_allow_html=True)
            m3.markdown(f'<div class="metric-box"><div class="metric-value">{scores["psnr"]:.1f}</div><div class="metric-label">PSNR dB<br>(&gt;20 = good)</div></div>', unsafe_allow_html=True)
            m4.markdown(f'<div class="metric-box"><div class="metric-value">{scores["region_iou"]:.3f}</div><div class="metric-label">Region IoU<br>(occluded area)</div></div>', unsafe_allow_html=True)
            m5.markdown(f'<div class="metric-box"><div class="metric-value">{scores["edge_iou"]:.3f}</div><div class="metric-label">Edge IoU<br>(structure)</div></div>', unsafe_allow_html=True)

        # ── Downloads ─────────────────────────────────────────────────────────
        st.markdown('<div class="step-header">💾 Downloads</div>', unsafe_allow_html=True)
        dcols = st.columns(3)
        for i, (label, path, mime) in enumerate([
            ("Download Reconstruction", output_dir / "4_reconstructed.jpg", "image/jpeg"),
            ("Download Sketch",         output_dir / "5_sketch.png",        "image/png"),
            ("Download Comparison",     final_comparison,                   "image/jpeg"),
        ]):
            if path.exists():
                with open(path, "rb") as f:
                    dcols[i].download_button(label=label, data=f, file_name=path.name, mime=mime)

else:
    st.info("👆 Upload a clean weapon image. RT-DETR will auto-detect the class and the pipeline will reconstruct the occluded region.")
    st.markdown("---")
    st.markdown("### Pipeline Overview")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown("**1. Input**\nClean weapon image")
    c2.markdown("**2. Detect**\nRT-DETR finds weapon + class")
    c3.markdown("**3. Reconstruct**\nSD fills the hidden region")
    c4.markdown("**4. Sketch**\nCanny forensic sketch")
    c5.markdown("**5. Evaluate**\nSSIM, PSNR, Region IoU, Edge IoU")