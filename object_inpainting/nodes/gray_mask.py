from ..masking import make_source
from .base import Node


class GrayMaskNode(Node):
    INPUT_TYPES = {"target": "RGB_IMAGE", "edit_mask": "MASK"}
    OUTPUT_TYPES = {"source": "RGB_IMAGE"}

    def execute(self, target, edit_mask):
        return {"source": make_source(target, edit_mask)}
