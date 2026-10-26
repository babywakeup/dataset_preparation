from ..reference import extract_reference
from .base import Node


class ExtractReferenceNode(Node):
    INPUT_TYPES = {"target": "RGB_IMAGE", "object_mask": "MASK", "bbox": "BBOX", "config": "CONFIG"}
    OUTPUT_TYPES = {"reference": "RGB_IMAGE"}

    def execute(self, target, object_mask, bbox, config):
        return {
            "reference": extract_reference(
                target, object_mask, bbox, config.reference_resolution, config.reference_padding
            )
        }
