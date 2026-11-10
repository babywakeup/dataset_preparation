import json
from pathlib import Path

from ..node_contracts import WriteJsonlInputs, WriteJsonlOutputs
from ..port_types import FilePath, JsonRecord
from .base import Node


class WriteJsonlNode(Node):
    SIDE_EFFECTS = True
    INPUT_SCHEMA = WriteJsonlInputs
    OUTPUT_SCHEMA = WriteJsonlOutputs

    def execute(self, entry: JsonRecord, path: FilePath) -> WriteJsonlOutputs:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return {"entry": entry}
