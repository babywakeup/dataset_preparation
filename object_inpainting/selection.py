import numpy as np
from PIL import Image

from .config import PipelineConfig


def select_object(mask: Image.Image, config: PipelineConfig) -> tuple[int, int, int, int]:
    bbox = mask.getbbox()
    if bbox is None:
        raise ValueError("Instance mask is empty.")
    fraction = float((np.asarray(mask) > 127).mean())
    if not config.min_object_fraction <= fraction <= config.max_object_fraction:
        raise ValueError(f"Object area fraction {fraction:.4f} is outside the configured range.")
    return bbox
