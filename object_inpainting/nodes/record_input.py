from .base import Node


class RecordInputNode(Node):
    INPUT_TYPES = {"record": "OBJECT_RECORD"}
    OUTPUT_TYPES = {"target_path": "PATH", "mask_path": "PATH", "object_name": "STRING"}

    def execute(self, record):
        return {
            "target_path": record.target_path,
            "mask_path": record.mask_path,
            "object_name": record.object_name,
        }
