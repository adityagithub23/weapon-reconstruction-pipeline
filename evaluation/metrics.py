"""
metrics.py
----------
Evaluation metrics for reconstruction quality.
Uses only OpenCV + NumPy (no new dependencies).
"""

import cv2
import numpy as np


def compute_iou(mask1, mask2):

    mask1 = mask1 > 0
    mask2 = mask2 > 0

    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 0.0

    return intersection / union


def compute_mse(img1, img2):

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    mse = np.mean((img1 - img2) ** 2)

    return mse


def compute_psnr(img1, img2):

    mse = compute_mse(img1, img2)

    if mse == 0:
        return 100

    PIXEL_MAX = 255.0

    psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

    return psnr


def compute_ssim(img1, img2):

    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    gray1 = gray1.astype(np.float32)
    gray2 = gray2.astype(np.float32)

    C1 = 6.5025
    C2 = 58.5225

    mu1 = cv2.GaussianBlur(gray1, (11,11), 1.5)
    mu2 = cv2.GaussianBlur(gray2, (11,11), 1.5)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(gray1 * gray1, (11,11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(gray2 * gray2, (11,11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(gray1 * gray2, (11,11), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()


def evaluate_reconstruction(original, reconstructed):

    results = {}

    results["MSE"] = compute_mse(original, reconstructed)
    results["PSNR"] = compute_psnr(original, reconstructed)
    results["SSIM"] = compute_ssim(original, reconstructed)

    return results


def evaluate_masks(gt_mask, predicted_mask):

    return {
        "IoU": compute_iou(gt_mask, predicted_mask)
    }