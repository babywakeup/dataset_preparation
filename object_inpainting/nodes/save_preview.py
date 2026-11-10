from pathlib import Path

from ..exporting import export_preview
from ..node_contracts import SavePreviewInputs, SavePreviewOutputs
from ..port_types import FilePath
from ..quality import PreparedSample
from .base import Node


class SavePreviewNode(Node):
    SIDE_EFFECTS = True
    INPUT_SCHEMA = SavePreviewInputs
    OUTPUT_SCHEMA = SavePreviewOutputs

    def execute(self, sample: PreparedSample, path: FilePath) -> SavePreviewOutputs:
        path = Path(path)
        export_preview(sample, path)
        return {"path": path}
