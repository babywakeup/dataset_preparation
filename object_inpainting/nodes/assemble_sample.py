from ..node_contracts import AssembleSampleInputs, SampleOutputs
from ..port_types import MaskedRGBImage, MaskImage, ReferenceRGBImage, RGBImage
from ..quality import PreparedSample
from .base import Node


class AssembleSampleNode(Node):
    INPUT_SCHEMA = AssembleSampleInputs
    OUTPUT_SCHEMA = SampleOutputs

    def execute(
        self,
        target: RGBImage,
        source: MaskedRGBImage,
        edit_mask: MaskImage,
        reference: ReferenceRGBImage,
        prompt: str,
    ) -> SampleOutputs:
        return {"sample": PreparedSample(target, source, edit_mask, reference, prompt)}
