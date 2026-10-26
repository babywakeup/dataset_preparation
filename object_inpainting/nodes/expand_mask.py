from ..masking import expand_mask
from .base import Node


class ExpandMaskNode(Node):
    INPUT_TYPES = {"object_mask": "MASK", "config": "CONFIG"}
    OUTPUT_TYPES = {"edit_mask": "MASK"}

    def execute(self, object_mask, config):
        return {"edit_mask": expand_mask(object_mask, config.dilation_radius)}
