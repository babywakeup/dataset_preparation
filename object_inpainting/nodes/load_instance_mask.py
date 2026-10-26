from ..records import load_instance_mask
from .base import Node


class LoadInstanceMaskNode(Node):
    INPUT_TYPES = {"path": "PATH", "target": "RGB_IMAGE"}
    OUTPUT_TYPES = {"object_mask": "MASK"}

    def execute(self, path, target):
        return {"object_mask": load_instance_mask(path, target)}
