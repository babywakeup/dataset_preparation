import json
from pathlib import Path

from .base import Node


class WriteJsonlNode(Node):
    SIDE_EFFECTS = True
    INPUT_TYPES = {"entry": "JSON_RECORD", "path": "PATH"}
    OUTPUT_TYPES = {"entry": "JSON_RECORD"}

    def execute(self, entry, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return {"entry": entry}
