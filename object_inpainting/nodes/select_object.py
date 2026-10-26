from ..selection import select_object
from .base import Node


class SelectObjectNode(Node):
    INPUT_TYPES = {"object_mask": "MASK", "config": "CONFIG"}
    OUTPUT_TYPES = {"object_mask": "MASK", "bbox": "BBOX"}

    def execute(self, object_mask, config):
        return {"object_mask": object_mask, "bbox": select_object(object_mask, config)}
