import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageEnhance, ImageOps
from typing import Dict, Any, Tuple

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

def apply_clahe(pil_img: Image.Image) -> Image.Image:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance chest X-ray contrast.
    Falls back to PIL equalized image if OpenCV is not installed.
    """
    if HAS_OPENCV:
        np_img = np.array(pil_img)
        if len(np_img.shape) == 3 and np_img.shape[2] == 3:
            yuv = cv2.cvtColor(np_img, cv2.COLOR_RGB2YUV)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
            enhanced_img = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_img = clahe.apply(np_img)
        return Image.fromarray(enhanced_img)
    else:
        # Fallback to PIL ImageOps Equalize
        return ImageOps.equalize(pil_img)


def get_transforms(image_size: int = 224, is_train: bool = True, use_clahe: bool = False) -> transforms.Compose:
    """
    Returns torchvision transforms for training (with mild augmentation) or evaluation.
    """
    transform_list = []

    if use_clahe:
        transform_list.append(transforms.Lambda(apply_clahe))

    transform_list.append(transforms.Resize((image_size, image_size)))

    if is_train:
        # Medically reasonable augmentations only (no extreme warping or flipping that changes lateral findings)
        transform_list.extend([
            transforms.RandomRotation(degrees=7),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.RandomAffine(degrees=0, translate=(0.04, 0.04), scale=(0.96, 1.04))
        ])

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return transforms.Compose(transform_list)

def preprocess_single_image(image_input: Any, image_size: int = 224, use_clahe: bool = False) -> torch.Tensor:
    """
    Preprocesses a single image (from file path, PIL Image, or NumPy array) into a PyTorch tensor (1, C, H, W).
    """
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, np.ndarray):
        image = Image.fromarray(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    eval_transform = get_transforms(image_size=image_size, is_train=False, use_clahe=use_clahe)
    tensor = eval_transform(image)
    return tensor.unsqueeze(0)  # Batch dimension (1, 3, H, W)
