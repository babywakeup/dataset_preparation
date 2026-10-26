from ..geometry import align_square
from .base import Node


class AlignSquareNode(Node):
    INPUT_TYPES = {"target": "RGB_IMAGE", "object_mask": "MASK", "edit_mask": "MASK", "config": "CONFIG"}
    OUTPUT_TYPES = {"target": "RGB_IMAGE", "object_mask": "MASK", "edit_mask": "MASK"}

    def execute(self, target, object_mask, edit_mask, config):
        target, object_mask, edit_mask = align_square(target, object_mask, edit_mask, config.resolution)
        return {"target": target, "object_mask": object_mask, "edit_mask": edit_mask}
