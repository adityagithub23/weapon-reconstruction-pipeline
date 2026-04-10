"""
inpaint.py
----------
Stable Diffusion Inpainting for weapon reconstruction.
Uses runwayml/stable-diffusion-inpainting.
Preserves aspect ratio, caps at 512px for MPS memory safety,
then scales result back to original size.
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from PIL import Image


# ── Prompts ─────────────────────────────────────────
SD_PROMPTS = {
    "pistol": (
        "a complete realistic pistol handgun, full weapon visible, "
        "clean neutral background, forensic technical illustration"
    ),
    "firearm": (
        "a complete realistic firearm rifle weapon, full weapon visible, "
        "clean background, forensic technical illustration"
    ),
    "knife": (
        "a complete realistic knife with full blade visible, "
        "clean neutral background, forensic illustration"
    ),
    "grenade": (
        "a complete realistic hand grenade, full object visible"
    ),
    "rocket": (
        "a complete realistic rocket launcher weapon"
    ),
}

NEGATIVE_PROMPT = (
    "blurry, distorted, duplicate, extra parts, watermark, text"
)


# ── Device selection ───────────────────────────────
def get_device():
    if torch.backends.mps.is_available():
        print("[Inpaint] Using Apple MPS")
        return torch.device("mps")
    if torch.cuda.is_available():
        print("[Inpaint] Using CUDA")
        return torch.device("cuda")
    print("[Inpaint] Using CPU")
    return torch.device("cpu")


# ── Pipeline loader ───────────────────────────────
_pipeline_cache = None


def load_inpainting_pipeline(model_id="runwayml/stable-diffusion-inpainting"):
    global _pipeline_cache

    if _pipeline_cache is not None:
        return _pipeline_cache

    from diffusers import StableDiffusionInpaintPipeline

    device = get_device()
    dtype = torch.float32 if device.type == "mps" else torch.float16

    print("[Inpaint] Loading Stable Diffusion pipeline...")

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )

    pipe = pipe.to(device)

    if device.type == "mps":
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()

    if device.type == "cuda":
        pipe.enable_xformers_memory_efficient_attention()

    print("[Inpaint] Pipeline ready.")
    _pipeline_cache = pipe
    return pipe


# ── Core reconstruction function ───────────────────
def inpaint_weapon(
    crop_image,
    missing_mask,
    class_name,
    num_inference_steps=30,
    guidance_scale=7.5,
    num_images=1,
    seed=42,
):
    pipe = load_inpainting_pipeline()

    import cv2

    # Convert crop to PIL
    if isinstance(crop_image, np.ndarray):
        crop_rgb = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(crop_rgb)
    else:
        pil_image = crop_image

    # Convert mask to PIL
    if isinstance(missing_mask, np.ndarray):
        pil_mask = Image.fromarray(missing_mask)
        if pil_mask.mode != "L":
            pil_mask = pil_mask.convert("L")
    else:
        pil_mask = missing_mask.convert("L")

    # ── Aspect-ratio preserving resize ──────────────
    # Keep longest side at 512, stay within MPS memory
    # Must be divisible by 8 for SD
    MAX_SIZE = 512
    orig_width, orig_height = pil_image.width, pil_image.height
    scale = min(MAX_SIZE / orig_width, MAX_SIZE / orig_height, 1.0)
    width  = (int(orig_width  * scale) // 8) * 8
    height = (int(orig_height * scale) // 8) * 8

    pil_image = pil_image.resize((width, height), Image.LANCZOS)
    pil_mask  = pil_mask.resize((width, height), Image.NEAREST)
    # ────────────────────────────────────────────────

    prompt = SD_PROMPTS.get(class_name.lower(), SD_PROMPTS["firearm"])

    print(f"[Inpaint] Class: {class_name}")
    print(f"[Inpaint] Original resolution: {orig_width}x{orig_height}")
    print(f"[Inpaint] SD resolution: {width}x{height}")

    device = get_device()
    generator = torch.Generator(device=device.type).manual_seed(seed)

    print(f"[Inpaint] Running {num_inference_steps} steps...")

    result = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        image=pil_image,
        mask_image=pil_mask,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images,
        generator=generator,
    )

    # Scale results back to original size
    images = [
        img.resize((orig_width, orig_height), Image.LANCZOS)
        for img in result.images
    ]

    print(f"[Inpaint] Generated {len(images)} image(s) at {orig_width}x{orig_height}.")
    return images


# ── Standalone test ───────────────────────────────
def inpaint_from_paths(
    image_path,
    mask_path,
    class_name,
    output_path,
    num_inference_steps=30,
    guidance_scale=7.5,
    seed=42,
):
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


# ── CLI ───────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",      required=True)
    parser.add_argument("--mask",       required=True)
    parser.add_argument("--class_name", required=True)
    parser.add_argument("--output",     default="output/reconstructed/result.jpg")
    parser.add_argument("--steps",      type=int,   default=30)
    parser.add_argument("--guidance",   type=float, default=7.5)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    inpaint_from_paths(
        args.image,
        args.mask,
        args.class_name,
        args.output,
        args.steps,
        args.guidance,
        args.seed,
    )