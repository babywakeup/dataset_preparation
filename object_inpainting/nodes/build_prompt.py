from ..node_contracts import BuildPromptInputs, BuildPromptOutputs
from ..prompting import build_prompt
from .base import Node


class BuildPromptNode(Node):
    INPUT_SCHEMA = BuildPromptInputs
    OUTPUT_SCHEMA = BuildPromptOutputs

    def execute(self, object_name: str) -> BuildPromptOutputs:
        return {"prompt": build_prompt(object_name)}
