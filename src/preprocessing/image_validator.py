import numpy as np
from PIL import Image
from typing import Union, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

def validate_chest_xray(
    image_input: Union[str, Image.Image, np.ndarray],
    saturation_threshold: float = 0.07,
    channel_diff_threshold: float = 12.0
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Validates whether the provided image is a Chest X-Ray image.
    Performs multi-tier radiological inspection:
    1. Color Saturation & RGB Channel Variance Check (X-Rays are strictly grayscale/monochrome).
    2. Luminance & Dynamic Range Check.
    3. Contrast & Texture Profile Check.

    Returns:
        Tuple[is_valid (bool), error_message (str), metrics (dict)]
    """
    try:
        # Step 1: Normalize input into PIL Image and NumPy Array
        if isinstance(image_input, str):
            try:
                pil_img = Image.open(image_input)
            except Exception as e:
                return False, f"Unable to open image file: {e}", {}
        elif isinstance(image_input, np.ndarray):
            pil_img = Image.fromarray(image_input)
        elif isinstance(image_input, Image.Image):
            pil_img = image_input
        else:
            return False, "Unsupported image type.", {}

        # Ensure RGB for color comparison
        rgb_img = pil_img.convert("RGB")
        np_rgb = np.array(rgb_img, dtype=np.float32)

        # Step 2: Color Saturation & Channel Variance Analysis
        # Extract R, G, B channels
        r, g, b = np_rgb[:, :, 0], np_rgb[:, :, 1], np_rgb[:, :, 2]
        
        # Calculate mean absolute channel differences (|R-G| + |G-B| + |B-R|)
        diff_rg = np.abs(r - g)
        diff_gb = np.abs(g - b)
        diff_br = np.abs(b - r)
        mean_channel_diff = float(np.mean(diff_rg + diff_gb + diff_br))

        # HSV Saturation calculation
        hsv_img = pil_img.convert("HSV")
        np_hsv = np.array(hsv_img, dtype=np.float32)
        s_channel = np_hsv[:, :, 1] / 255.0  # Normalize to 0-1
        mean_saturation = float(np.mean(s_channel))
        max_saturation = float(np.max(s_channel))

        # Step 3: Grayscale Intensity & Contrast Profile
        gray_img = pil_img.convert("L")
        np_gray = np.array(gray_img, dtype=np.float32)
        mean_intensity = float(np.mean(np_gray))
        std_intensity = float(np.std(np_gray))

        metrics = {
            "mean_channel_diff": round(mean_channel_diff, 3),
            "mean_saturation": round(mean_saturation, 4),
            "max_saturation": round(max_saturation, 4),
            "mean_intensity": round(mean_intensity, 2),
            "std_intensity": round(std_intensity, 2)
        }

        # Step 4: Evaluate Validation Rules
        # Rule A: High color saturation indicates a non-grayscale standard color photo (dog, car, selfie, etc.)
        if mean_channel_diff > channel_diff_threshold or mean_saturation > saturation_threshold:
            logger.warning(f"Image rejected as non-X-ray (high color content): channel_diff={mean_channel_diff:.2f}, saturation={mean_saturation:.4f}")
            return False, "Invalid image selected. Please select only Chest X-Ray images.", metrics

        # Rule B: Blank / Extreme luminance check (pure white or pure black)
        if mean_intensity < 10.0 or mean_intensity > 245.0:
            logger.warning(f"Image rejected as non-X-ray (extreme luminance): mean={mean_intensity:.2f}")
            return False, "Invalid image selected. Please select only Chest X-Ray images.", metrics

        # Rule C: Low contrast check (flat uniform image)
        if std_intensity < 10.0:
            logger.warning(f"Image rejected as non-X-ray (low contrast): std={std_intensity:.2f}")
            return False, "Invalid image selected. Please select only Chest X-Ray images.", metrics

        logger.info(f"Image passed Chest X-Ray validation: {metrics}")
        return True, "", metrics

    except Exception as e:
        logger.error(f"Error during X-ray validation: {e}")
        return False, f"Validation error: {e}", {}
