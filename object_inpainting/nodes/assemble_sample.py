from ..quality import PreparedSample
from .base import Node


class AssembleSampleNode(Node):
    INPUT_TYPES = {
        "target": "RGB_IMAGE",
        "source": "RGB_IMAGE",
        "edit_mask": "MASK",
        "reference": "RGB_IMAGE",
        "prompt": "STRING",
    }
    OUTPUT_TYPES = {"sample": "PREPARED_SAMPLE"}

    def execute(self, target, source, edit_mask, reference, prompt):
        return {"sample": PreparedSample(target, source, edit_mask, reference, prompt)}
