from pathlib import Path

from ..exporting import export_preview
from .base import Node


class SavePreviewNode(Node):
    SIDE_EFFECTS = True
    INPUT_TYPES = {"sample": "PREPARED_SAMPLE", "path": "PATH"}
    OUTPUT_TYPES = {"path": "PATH"}

    def execute(self, sample, path):
        path = Path(path)
        export_preview(sample, path)
        return {"path": path}
