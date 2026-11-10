from ..config import PipelineConfig
from ..node_contracts import ExtractReferenceInputs, ExtractReferenceOutputs
from ..port_types import BoundingBox, MaskImage, RGBImage
from ..reference import extract_reference
from .base import Node


class ExtractReferenceNode(Node):
    INPUT_SCHEMA = ExtractReferenceInputs
    OUTPUT_SCHEMA = ExtractReferenceOutputs

    def execute(
        self, target: RGBImage, object_mask: MaskImage, bbox: BoundingBox, config: PipelineConfig
    ) -> ExtractReferenceOutputs:
        return {
            "reference": extract_reference(
                target, object_mask, bbox, config.reference_resolution, config.reference_padding
            )
        }
