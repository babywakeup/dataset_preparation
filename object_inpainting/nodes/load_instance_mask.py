from ..node_contracts import LoadInstanceMaskInputs, LoadInstanceMaskOutputs
from ..port_types import FilePath, RGBImage
from ..records import load_instance_mask
from .base import Node


class LoadInstanceMaskNode(Node):
    INPUT_SCHEMA = LoadInstanceMaskInputs
    OUTPUT_SCHEMA = LoadInstanceMaskOutputs

    def execute(self, path: FilePath, target: RGBImage) -> LoadInstanceMaskOutputs:
        return {"object_mask": load_instance_mask(path, target)}
