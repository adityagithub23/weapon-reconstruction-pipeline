import cv2
import numpy as np

def compute_adaptive_crop(image, bbox, expansion_factor=2.2, target_size=512):
    """
    Compute adaptive crop region so reconstructed weapon can fully fit.

    Args:
        image : original image
        bbox  : [x1,y1,x2,y2] detection bbox
        expansion_factor : how much larger the canvas should be
        target_size : final diffusion input size

    Returns:
        crop512      : centered crop resized to target_size
        visible_mask : mask of visible object
        missing_mask : region diffusion should generate
    """

    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox

    box_w = x2 - x1
    box_h = y2 - y1

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    # Estimate reconstruction canvas
    new_w = int(box_w * expansion_factor)
    new_h = int(box_h * expansion_factor)

    crop_x1 = max(0, cx - new_w // 2)
    crop_y1 = max(0, cy - new_h // 2)
    crop_x2 = min(w, cx + new_w // 2)
    crop_y2 = min(h, cy + new_h // 2)

    crop = image[crop_y1:crop_y2, crop_x1:crop_x2]

    # Resize to diffusion resolution
    crop512 = cv2.resize(crop, (target_size, target_size))

    # Create visible mask from original bbox
    mask = np.zeros((target_size, target_size), dtype=np.uint8)

    # Scale bbox to resized coordinates
    scale_x = target_size / (crop_x2 - crop_x1)
    scale_y = target_size / (crop_y2 - crop_y1)

    bx1 = int((x1 - crop_x1) * scale_x)
    by1 = int((y1 - crop_y1) * scale_y)
    bx2 = int((x2 - crop_x1) * scale_x)
    by2 = int((y2 - crop_y1) * scale_y)

    mask[by1:by2, bx1:bx2] = 255

    visible_mask = mask
    missing_mask = 255 - mask

    return crop512, visible_mask, missing_mask