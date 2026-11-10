from pathlib import Path

from ..config import PipelineConfig
from ..exporting import choose_split, export_sample
from ..node_contracts import SaveSampleInputs, SaveSampleOutputs
from ..port_types import FilePath
from ..quality import PreparedSample
from ..records import ObjectRecord
from .base import Node


class SaveSampleNode(Node):
    SIDE_EFFECTS = True
    INPUT_SCHEMA = SaveSampleInputs
    OUTPUT_SCHEMA = SaveSampleOutputs

    def execute(
        self, sample: PreparedSample, record: ObjectRecord, config: PipelineConfig, output: FilePath,
    ) -> SaveSampleOutputs:
        output = Path(output)
        split = choose_split(record.target_path, config.validation_fraction, config.seed)
        return {
            "entry": export_sample(sample, record, output, split),
            "manifest_path": output / f"{split}.jsonl",
            "split": split,
        }
