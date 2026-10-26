from pathlib import Path

from ..exporting import choose_split, export_sample
from .base import Node


class SaveSampleNode(Node):
    SIDE_EFFECTS = True
    INPUT_TYPES = {"sample": "PREPARED_SAMPLE", "record": "OBJECT_RECORD", "config": "CONFIG", "output": "PATH"}
    OUTPUT_TYPES = {"entry": "JSON_RECORD", "manifest_path": "PATH", "split": "STRING"}

    def execute(self, sample, record, config, output):
        output = Path(output)
        split = choose_split(record.target_path, config.validation_fraction, config.seed)
        return {
            "entry": export_sample(sample, record, output, split),
            "manifest_path": output / f"{split}.jsonl",
            "split": split,
        }
