from ..config import PipelineConfig
from ..masking import expand_mask
from ..node_contracts import ExpandMaskInputs, ExpandMaskOutputs
from ..port_types import MaskImage
from .base import Node


class ExpandMaskNode(Node):
    INPUT_SCHEMA = ExpandMaskInputs
    OUTPUT_SCHEMA = ExpandMaskOutputs

    def execute(self, object_mask: MaskImage, config: PipelineConfig) -> ExpandMaskOutputs:
        return {"edit_mask": expand_mask(object_mask, config.dilation_radius)}
