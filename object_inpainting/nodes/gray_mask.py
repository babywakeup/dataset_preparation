from ..masking import make_source
from ..node_contracts import GrayMaskInputs, GrayMaskOutputs
from ..port_types import MaskImage, RGBImage
from .base import Node


class GrayMaskNode(Node):
    INPUT_SCHEMA = GrayMaskInputs
    OUTPUT_SCHEMA = GrayMaskOutputs

    def execute(self, target: RGBImage, edit_mask: MaskImage) -> GrayMaskOutputs:
        return {"source": make_source(target, edit_mask)}
