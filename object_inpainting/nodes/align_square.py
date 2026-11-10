from ..config import PipelineConfig
from ..geometry import align_square
from ..node_contracts import AlignSquareInputs, AlignSquareOutputs
from ..port_types import MaskImage, RGBImage
from .base import Node


class AlignSquareNode(Node):
    INPUT_SCHEMA = AlignSquareInputs
    OUTPUT_SCHEMA = AlignSquareOutputs

    def execute(
        self, target: RGBImage, object_mask: MaskImage, edit_mask: MaskImage, config: PipelineConfig
    ) -> AlignSquareOutputs:
        target, object_mask, edit_mask = align_square(target, object_mask, edit_mask, config.resolution)
        return {"target": target, "object_mask": object_mask, "edit_mask": edit_mask}
