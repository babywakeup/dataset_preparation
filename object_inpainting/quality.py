from dataclasses import dataclass

import numpy as np
from PIL import Image

from .config import PipelineConfig


@dataclass(frozen=True)
class PreparedSample:
    target: Image.Image
    source: Image.Image
    mask: Image.Image
    reference: Image.Image
    prompt: str


def validate_sample(sample: PreparedSample, object_mask: Image.Image, config: PipelineConfig) -> None:
    size = (config.resolution, config.resolution)
    if sample.target.size != size or sample.source.size != size or sample.mask.size != size:
        raise ValueError("Source, target and mask must have the configured square dimensions.")
    if sample.reference.size != (config.reference_resolution, config.reference_resolution):
        raise ValueError("Reference must have the configured square dimensions.")
    mask_values = np.asarray(sample.mask)
    if not np.isin(mask_values, [0, 255]).all():
        raise ValueError("Edit mask must be binary.")
    editable = mask_values > 127
    if not editable.any() or editable.all():
        raise ValueError("Edit mask must have both editable and preserved pixels.")
    if not (np.asarray(object_mask) > 127).any():
        raise ValueError("Object vanished after resizing.")
    if ((np.asarray(object_mask) > 127) & ~editable).any():
        raise ValueError("Edit mask does not cover the object.")
    source = np.asarray(sample.source)
    target = np.asarray(sample.target)
    if not np.array_equal(source[~editable], target[~editable]):
        raise ValueError("Source differs from target outside the edit mask.")
    if not (source[editable] == 128).all():
        raise ValueError("Editable source pixels must be gray (128, 128, 128).")
    if not sample.prompt.strip():
        raise ValueError("Prompt is empty.")
