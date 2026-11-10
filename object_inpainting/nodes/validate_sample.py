from ..config import PipelineConfig
from ..node_contracts import SampleOutputs, ValidateSampleInputs
from ..port_types import MaskImage
from ..quality import PreparedSample, validate_sample
from .base import Node


class ValidateSampleNode(Node):
    INPUT_SCHEMA = ValidateSampleInputs
    OUTPUT_SCHEMA = SampleOutputs

    def execute(self, sample: PreparedSample, object_mask: MaskImage, config: PipelineConfig) -> SampleOutputs:
        validate_sample(sample, object_mask, config)
        return {"sample": sample}
