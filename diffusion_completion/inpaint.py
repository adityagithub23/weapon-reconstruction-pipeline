"""
inpaint.py
----------
Uses Stable Diffusion Inpainting to reconstruct the occluded/missing
portion of a weapon based on its visible region + class label.

Model: runwayml/stable-diffusion-inpainting (no fine-tuning needed)
Device: MPS (Apple M1) / CUDA / CPU fallback

Setup:
    pip install diffusers transformers accelerate Pillow torch torchvision

Usage (standalone test):
    python diffusion_completion/inpaint.py \
        --image output/segmentation/img001_crop512.jpg \
        --mask  output/segmentation/img001_mask_missing.png \
        --class_name pistol \
        --output output/reconstructed/img001_reconstructed.jpg

Pipeline:
    512×512 weapon crop + 512×512 missing mask + text prompt
    → Stable Diffusion Inpainting
    → 512×512 reconstructed full weapon image
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from PIL import Image


# ── Class-conditioned text prompts ────────────────────────────────────────────
SD_PROMPTS = {
    "pistol":   ("a complete realistic pistol handgun, full weapon visible, "
                 "clean neutral background, forensic technical illustration, "
                 "sharp detail, professional quality, no blur"),
    "firearm":  ("a complete realistic firearm, rifle or assault weapon, full weapon visible, "
                 "clean background, forensic technical illustration, sharp detail"),
    "knife":    ("a complete realistic knife with full blade visible, sharp edge, "
                 "clean neutral background, forensic technical illustration, high detail"),
    "grenade":  ("a complete realistic hand grenade, full object visible, "
                 "clean background, forensic illustration, high detail"),
    "rocket":   ("a complete realistic rocket launcher or missile, full weapon visible, "
                 "clean background, forensic technical illustration, high detail"),
}

NEGATIVE_PROMPT = (
    "blurry, low quality, distorted, duplicate, extra parts, watermark, "
    "text, signature, deformed, unrealistic, cartoon, anime, sketch"
)


# ── Device selection ──────────────────────────────────────────────────────────
def get_device():
    """Select best available device. MPS for M1 Mac, CUDA for NVIDIA, else CPU."""
    if torch.backends.mps.is_available():
        print("[Inpaint] Using Apple MPS (M1/M2)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("[Inpaint] Using CUDA GPU")
        return torch.device("cuda")
    else:
        print("[Inpaint] Using CPU (slow, consider patience)")
        return torch.device("cpu")


# ── Load pipeline ─────────────────────────────────────────────────────────────
_pipeline_cache = None

def load_inpainting_pipeline(model_id: str = "runwayml/stable-diffusion-inpainting"):
    """
    Load SD Inpainting pipeline. Cached after first load.
    First run will download ~4GB — subsequent runs are instant.
    """
    global _pipeline_cache
    if _pipeline_cache is not None:
        return _pipeline_cache

    from diffusers import StableDiffusionInpaintPipeline

    device = get_device()

    print(f"[Inpaint] Loading pipeline '{model_id}' ...")
    print("[Inpaint] First run downloads ~4GB — please wait...")

    # Use float32 for MPS (float16 has issues on some M1 builds)
    dtype = torch.float32 if device.type == "mps" else torch.float16

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,      # disable NSFW filter (not relevant for forensics)
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)

    # Memory optimization for M1
    if device.type == "mps":
        pipe.enable_attention_slicing()
    elif device.type == "cuda":
        pipe.enable_xformers_memory_efficient_attention()

    print("[Inpaint] Pipeline ready.")
    _pipeline_cache = pipe
    return pipe


# ── Core inpainting function ──────────────────────────────────────────────────
def inpaint_weapon(
    crop_image: np.ndarray | Image.Image,
    missing_mask: np.ndarray | Image.Image,
    class_name: str,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    num_images: int = 1,
    seed: int = 42,
) -> list[Image.Image]:
    """
    Reconstruct the missing portion of a weapon image using SD Inpainting.

    Args:
        crop_image:            512×512 BGR numpy array or PIL Image of weapon crop
        missing_mask:          512×512 grayscale numpy array or PIL Image
                               (255 = region to inpaint, 0 = region to keep)
        class_name:            Weapon class string e.g. "pistol", "knife"
        num_inference_steps:   SD denoising steps (20-50, higher=better quality/slower)
        guidance_scale:        How closely to follow the text prompt (5-15)
        num_images:            Number of variants to generate
        seed:                  Random seed for reproducibility

    Returns:
        List of PIL Images (reconstructed full weapon)
    """
    pipe = load_inpainting_pipeline()

    # Convert numpy arrays to PIL if needed
    if isinstance(crop_image, np.ndarray):
        import cv2
        crop_rgb = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(crop_rgb).resize((512, 512))
    else:
        pil_image = crop_image.resize((512, 512))

    if isinstance(missing_mask, np.ndarray):
        pil_mask = Image.fromarray(missing_mask).resize((512, 512))
        # Ensure it's single-channel
        if pil_mask.mode != "L":
            pil_mask = pil_mask.convert("L")
    else:
        pil_mask = missing_mask.resize((512, 512)).convert("L")

    # Get class-conditioned prompt
    prompt = SD_PROMPTS.get(class_name.lower(), SD_PROMPTS["firearm"])
    print(f"[Inpaint] Class: {class_name}")
    print(f"[Inpaint] Prompt: {prompt[:80]}...")

    # Set reproducible seed
    device = get_device()
    generator = torch.Generator(device=device.type).manual_seed(seed)

    # Run inpainting
    print(f"[Inpaint] Running {num_inference_steps} denoising steps...")
    result = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        image=pil_image,
        mask_image=pil_mask,
        height=512,
        width=512,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images,
        generator=generator,
    )

    images = result.images
    print(f"[Inpaint] Generated {len(images)} image(s).")
    return images


def inpaint_from_paths(
    image_path: str,
    mask_path: str,
    class_name: str,
    output_path: str,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    seed: int = 42,
) -> str:
    """
    Convenience wrapper: load from disk, inpaint, save to disk.

    Returns path to saved reconstructed image.
    """
    import cv2

    crop_image = cv2.imread(str(image_path))
    if crop_image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    missing_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if missing_mask is None:
        raise ValueError(f"Cannot read mask: {mask_path}")

    reconstructed = inpaint_weapon(
        crop_image=crop_image,
        missing_mask=missing_mask,
        class_name=class_name,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
    )

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    reconstructed[0].save(str(out_path))
    print(f"[Inpaint] Saved reconstruction to {out_path}")
    return str(out_path)


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SD Inpainting weapon reconstruction.")
    parser.add_argument("--image",    required=True,  help="512×512 weapon crop (JPG/PNG)")
    parser.add_argument("--mask",     required=True,  help="512×512 missing region mask (PNG, grayscale)")
    parser.add_argument("--class_name", required=True,
                        choices=["pistol", "firearm", "knife", "grenade", "rocket"],
                        help="Weapon class for conditioned prompt")
    parser.add_argument("--output",   default="output/reconstructed/result.jpg",
                        help="Output path for reconstructed image")
    parser.add_argument("--steps",    type=int,   default=30,
                        help="SD denoising steps (20-50)")
    parser.add_argument("--guidance", type=float, default=7.5,
                        help="Classifier-free guidance scale (5-15)")
    parser.add_argument("--seed",     type=int,   default=42,
                        help="Random seed")
    args = parser.parse_args()

    inpaint_from_paths(
        image_path=args.image,
        mask_path=args.mask,
        class_name=args.class_name,
        output_path=args.output,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed,
    )

def run_inpainting(image, mask, class_name, steps=30):
    """
    Wrapper used by Streamlit demo.
    Returns a reconstructed numpy image.
    """

    results = inpaint_weapon(
        crop_image=image,
        missing_mask=mask,
        class_name=class_name,
        num_inference_steps=steps
    )

    # Convert PIL → numpy for Streamlit display
    import numpy as np
    recon = np.array(results[0])

    return recon