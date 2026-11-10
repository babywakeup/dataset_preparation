from ..config import PipelineConfig
from ..node_contracts import SelectObjectInputs, SelectObjectOutputs
from ..port_types import MaskImage
from ..selection import select_object
from .base import Node


class SelectObjectNode(Node):
    INPUT_SCHEMA = SelectObjectInputs
    OUTPUT_SCHEMA = SelectObjectOutputs

    def execute(self, object_mask: MaskImage, config: PipelineConfig) -> SelectObjectOutputs:
        return {"object_mask": object_mask, "bbox": select_object(object_mask, config)}
