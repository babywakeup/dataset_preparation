import json
import logging
from copy import deepcopy
from graphlib import CycleError, TopologicalSorter
from pathlib import Path

from PIL import Image

from .color_regions import ColorRegionSet
from .config import PipelineConfig
from .nodes import NODE_CLASS_MAPPINGS
from .quality import PreparedSample
from .records import ObjectRecord
from .workflow_migrations import migrate_sam3_dilation


LOGGER = logging.getLogger(__name__)


class GraphExecutionCancelled(Exception):
    """Control-flow signal; cancellation must not be reported as a node failure."""


class NodeExecutionError(ValueError):
    """Identify the original failing node without attaching input data."""

    def __init__(self, node_id: str, class_type: str, error: Exception):
        self.node_id = node_id
        self.class_type = class_type
        self.error_type = type(error).__name__
        self.message = str(error) or self.error_type
        super().__init__(f"节点 {node_id} ({class_type}) 失败：{self.error_type}: {self.message}")

    def as_dict(self) -> dict[str, str]:
        return {
            "node_id": self.node_id,
            "class_type": self.class_type,
            "error_type": self.error_type,
            "message": self.message,
        }


def is_link(binding) -> bool:
    return isinstance(binding, dict) and set(binding) == {"node", "output"}


class NodeGraph:
    def __init__(self, definition: dict, registry=None):
        if not isinstance(definition, dict) or not isinstance(definition.get("nodes"), dict):
            raise ValueError("Workflow must contain a nodes object.")
        if not isinstance(definition.get("outputs"), dict):
            raise ValueError("Workflow must contain an outputs object.")
        self.registry = dict(NODE_CLASS_MAPPINGS if registry is None else registry)
        self.definition = (
            migrate_sam3_dilation(definition)
            if {"SAM3TextMaskNode", "MaskDilationNode"}.issubset(self.registry)
            else deepcopy(definition)
        )
        self.nodes = self.definition["nodes"]
        self.outputs = self.definition["outputs"]
        self.input_types: dict[str, dict[str, str]] = {}
        self.output_types: dict[str, dict[str, str]] = {}
        dependencies = {}
        for node_id, spec in self.nodes.items():
            if not isinstance(node_id, str) or not node_id:
                raise ValueError("Node IDs must be non-empty strings.")
            if not isinstance(spec, dict) or not isinstance(spec.get("class_type"), str):
                raise ValueError(f"Node {node_id}: class_type must be a string.")
            if not isinstance(spec.get("inputs"), dict):
                raise ValueError(f"Node {node_id}: inputs must be an object.")
        for node_id, spec in self.nodes.items():
            node_type = spec["class_type"]
            if node_type not in self.registry:
                raise ValueError(f"Node {node_id}: unknown class_type {node_type}.")
            node_class = self.registry[node_type]
            node_class.prepare_spec(spec)
            input_types = self.input_types[node_id] = node_class.input_types(spec)
            self.output_types[node_id] = node_class.output_types(spec)
            for port, value in getattr(node_class, "INPUT_DEFAULTS", {}).items():
                if port in input_types:
                    spec["inputs"].setdefault(port, deepcopy(value))
            for port in input_types:
                if node_class.optional_input(port):
                    spec["inputs"].setdefault(port, None)
            if set(spec["inputs"]) != set(input_types):
                raise ValueError(f"Node {node_id}: inputs must match {list(input_types)}.")
        # Normalize every instance before checking links, including links to a
        # dynamic-output node appearing later in the serialized nodes object.
        for node_id, spec in self.nodes.items():
            input_types = self.input_types[node_id]
            dependencies[node_id] = []
            for port, binding in spec["inputs"].items():
                if isinstance(binding, dict) and "context" in binding:
                    if (
                        set(binding) != {"context"}
                        or not isinstance(binding["context"], str)
                        or not binding["context"]
                    ):
                        raise ValueError(f"Node {node_id}.{port}: context name must be a non-empty string.")
                if is_link(binding):
                    source_type = self._validate_link(binding)
                    expected_type = input_types[port]
                    if source_type != expected_type and not (
                        expected_type == "IMAGE" and source_type in {"RGB_IMAGE", "RGBA_IMAGE", "MASK", "LABEL_MAP"}
                    ):
                        raise ValueError(f"Node {node_id}.{port}: expected {expected_type}, got {source_type}.")
                    dependencies[node_id].append(binding["node"])
        for binding in self.outputs.values():
            if not is_link(binding):
                raise ValueError("Graph outputs must be node output links.")
            self._validate_link(binding)
        try:
            self.order = tuple(TopologicalSorter(dependencies).static_order())
        except CycleError as error:
            raise ValueError("Workflow contains a cycle.") from error

    def _validate_link(self, binding) -> str:
        node_id, port = binding["node"], binding["output"]
        if not isinstance(node_id, str) or not isinstance(port, str):
            raise ValueError("Link node and output names must be strings.")
        if node_id not in self.nodes:
            raise ValueError(f"Unknown linked node: {node_id}.")
        node_type = self.nodes[node_id]["class_type"]
        if node_type not in self.registry:
            raise ValueError(f"Node {node_id}: unknown class_type {node_type}.")
        ports = self.output_types[node_id]
        if port not in ports:
            raise ValueError(f"Unknown output port: {node_id}.{port}.")
        return ports[port]

    @classmethod
    def from_file(cls, path: Path, registry=None):
        with Path(path).open(encoding="utf-8") as handle:
            saved = json.load(handle)
        return cls(saved.get("workflow", saved) if isinstance(saved, dict) else saved, registry)

    def require_outputs(self, required: dict[str, str]) -> None:
        for name, expected in required.items():
            if name not in self.outputs:
                raise ValueError(f"Workflow must expose output: {name}.")
            actual = self._validate_link(self.outputs[name])
            if actual != expected:
                raise ValueError(f"Workflow output {name}: expected {expected}, got {actual}.")

    def save(self, path: Path) -> None:
        with Path(path).open("w", encoding="utf-8") as handle:
            json.dump(self.definition, handle, ensure_ascii=False, indent=2)

    def start(self, context: dict, allow_side_effects: bool = True):
        if self.definition.get("conditional_loop") is not None:
            raise ValueError("Conditional loops require the Studio conditional task executor; ordinary DAG execution cannot bypass acceptance.")
        return GraphExecution(self, context, allow_side_effects)

    def run(self, context: dict, outputs=None, allow_side_effects: bool = True) -> dict:
        execution = self.start(context, allow_side_effects)
        return {name: execution.get(name) for name in (self.outputs if outputs is None else outputs)}


class GraphExecution:
    def __init__(self, graph: NodeGraph, context: dict, allow_side_effects: bool = True):
        self.graph = graph
        self.context = dict(context)
        self.cache = {}
        self.allow_side_effects = allow_side_effects

    def get(self, name: str):
        if name not in self.graph.outputs:
            raise ValueError(f"Unknown graph output: {name}.")
        return self._resolve(self.graph.outputs[name])

    def get_node(self, node_id: str) -> dict:
        if not isinstance(node_id, str) or node_id not in self.graph.nodes:
            raise ValueError(f"Unknown node: {node_id}.")
        return self._execute(node_id)

    def _resolve(self, binding):
        if is_link(binding):
            return self._execute(binding["node"])[binding["output"]]
        if isinstance(binding, dict) and set(binding) == {"context"}:
            name = binding["context"]
            if name not in self.context:
                raise ValueError(f"Missing workflow context: {name}.")
            return self.context[name]
        return binding

    def _execute(self, node_id: str) -> dict:
        try:
            return self._execute_checked(node_id)
        except (NodeExecutionError, GraphExecutionCancelled):
            # Preserve the upstream origin and the batch cancellation signal.
            raise
        except Exception as error:
            failure = NodeExecutionError(node_id, self.graph.nodes[node_id]["class_type"], error)
            LOGGER.exception("%s", failure)
            raise failure from error

    def _execute_checked(self, node_id: str) -> dict:
        if node_id not in self.cache:
            spec = self.graph.nodes[node_id]
            node_class = self.graph.registry[spec["class_type"]]
            if not self.allow_side_effects and getattr(node_class, "SIDE_EFFECTS", False):
                raise ValueError(f"Node {node_id}: file-writing nodes are not allowed during preview.")
            inputs = {port: self._resolve(binding) for port, binding in spec["inputs"].items()}
            types = {
                "STRING": str,
                "PATH": (str, Path),
                "CONFIG": PipelineConfig,
                "OBJECT_RECORD": ObjectRecord,
                "RGB_IMAGE": Image.Image,
                "IMAGE": Image.Image,
                "MASK": Image.Image,
                "LABEL_MAP": Image.Image,
                "COLOR_REGIONS": ColorRegionSet,
                "BBOX": (tuple, list),
                "INT": int,
                "INT_LIST": list,
                "IMAGE_LIST": list,
                "FLOAT": (int, float),
                "PREPARED_SAMPLE": PreparedSample,
                "JSON_RECORD": dict,
            }
            for port, value in inputs.items():
                expected = self.graph.input_types[node_id][port]
                optional = node_class.optional_input(port)
                if optional and value is None:
                    continue
                if optional and isinstance(value, str) and not value.strip():
                    continue
                if expected in types and not isinstance(value, types[expected]):
                    raise ValueError(f"Node {node_id}.{port}: invalid runtime value for {expected}.")
            result = node_class().execute_with_spec(inputs, spec)
            output_types = self.graph.output_types[node_id]
            if not isinstance(result, dict) or set(result) != set(output_types):
                raise ValueError(f"Node {node_id}: returned outputs must match {list(output_types)}.")
            self.cache[node_id] = result
        return self.cache[node_id]
