from ..node_contracts import LoadTargetInputs, LoadTargetOutputs
from ..port_types import FilePath
from ..records import load_target
from .base import Node


class LoadTargetNode(Node):
    INPUT_SCHEMA = LoadTargetInputs
    OUTPUT_SCHEMA = LoadTargetOutputs

    def execute(self, path: FilePath) -> LoadTargetOutputs:
        return {"target": load_target(path)}
