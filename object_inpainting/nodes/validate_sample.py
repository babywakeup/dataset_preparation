from ..quality import validate_sample
from .base import Node


class ValidateSampleNode(Node):
    INPUT_TYPES = {"sample": "PREPARED_SAMPLE", "object_mask": "MASK", "config": "CONFIG"}
    OUTPUT_TYPES = {"sample": "PREPARED_SAMPLE"}

    def execute(self, sample, object_mask, config):
        validate_sample(sample, object_mask, config)
        return {"sample": sample}
