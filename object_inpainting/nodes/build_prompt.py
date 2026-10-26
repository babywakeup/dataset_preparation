from ..prompting import build_prompt
from .base import Node


class BuildPromptNode(Node):
    INPUT_TYPES = {"object_name": "STRING"}
    OUTPUT_TYPES = {"prompt": "STRING"}

    def execute(self, object_name):
        return {"prompt": build_prompt(object_name)}
