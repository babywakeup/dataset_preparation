from ..node_contracts import RecordInputInputs, RecordInputOutputs
from ..records import ObjectRecord
from .base import Node


class RecordInputNode(Node):
    INPUT_SCHEMA = RecordInputInputs
    OUTPUT_SCHEMA = RecordInputOutputs

    def execute(self, record: ObjectRecord) -> RecordInputOutputs:
        return {
            "target_path": record.target_path,
            "mask_path": record.mask_path,
            "object_name": record.object_name,
        }
