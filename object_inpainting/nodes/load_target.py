from ..records import load_target
from .base import Node


class LoadTargetNode(Node):
    INPUT_TYPES = {"path": "PATH"}
    OUTPUT_TYPES = {"target": "RGB_IMAGE"}

    def execute(self, path):
        return {"target": load_target(path)}
