import json
import tempfile
import unittest
from pathlib import Path

import test_object_inpainting_dataset as dataset_tests
from object_inpainting.graph import NodeGraph
from object_inpainting.nodes import NODE_CLASS_MAPPINGS, BuildPromptNode, Node
from object_inpainting.pipeline import DEFAULT_WORKFLOW, build_dataset
from object_inpainting.records import read_records


class EchoNode(Node):
    INPUT_TYPES = {"value": "STRING"}
    OUTPUT_TYPES = {"value": "STRING"}
    executions = 0

    def execute(self, value):
        type(self).executions += 1
        return {"value": value}


class NodeGraphTests(unittest.TestCase):
    def setUp(self):
        EchoNode.executions = 0
        self.registry = {"EchoNode": EchoNode}
        self.definition = {
            "nodes": {
                "second": {"class_type": "EchoNode", "inputs": {"value": {"node": "first", "output": "value"}}},
                "first": {"class_type": "EchoNode", "inputs": {"value": {"context": "value"}}},
            },
            "outputs": {"result": {"node": "second", "output": "value"}},
        }

    def test_independent_node_can_run_without_graph(self):
        result = BuildPromptNode().execute(object_name="cup")
        self.assertIn("cup", result["prompt"])
        self.assertEqual(BuildPromptNode.INPUT_TYPES, {"object_name": "STRING"})

    def test_graph_resolves_dependencies_not_json_order(self):
        graph = NodeGraph(self.definition, self.registry)
        self.assertEqual(graph.order, ("first", "second"))
        self.assertEqual(graph.run({"value": "cup"}), {"result": "cup"})
        self.assertEqual(EchoNode.executions, 2)

    def test_execution_caches_nodes_but_not_between_samples(self):
        graph = NodeGraph(self.definition, self.registry)
        execution = graph.start({"value": "cup"})
        self.assertEqual(execution.get("result"), "cup")
        self.assertEqual(execution.get("result"), "cup")
        self.assertEqual(EchoNode.executions, 2)
        self.assertEqual(graph.run({"value": "bag"}), {"result": "bag"})
        self.assertEqual(EchoNode.executions, 4)

    def test_graph_rejects_cycles(self):
        self.definition["nodes"]["first"]["inputs"]["value"] = {"node": "second", "output": "value"}
        with self.assertRaisesRegex(ValueError, "cycle"):
            NodeGraph(self.definition, self.registry)

    def test_graph_rejects_unknown_nodes_and_ports(self):
        for binding in ({"node": "missing", "output": "value"}, {"node": "first", "output": "missing"}):
            with self.subTest(binding=binding):
                self.definition["outputs"]["result"] = binding
                with self.assertRaises(ValueError):
                    NodeGraph(self.definition, self.registry)

    def test_graph_rejects_unknown_classes_and_incorrect_inputs(self):
        self.definition["nodes"]["first"]["class_type"] = "UnknownNode"
        with self.assertRaisesRegex(ValueError, "unknown class_type"):
            NodeGraph(self.definition, self.registry)
        self.definition["nodes"]["first"]["class_type"] = "EchoNode"
        self.definition["nodes"]["first"]["inputs"] = {}
        with self.assertRaisesRegex(ValueError, "inputs must match"):
            NodeGraph(self.definition, self.registry)

    def test_graph_rejects_type_mismatched_links(self):
        definition = {
            "nodes": {
                "prompt": {"class_type": "BuildPromptNode", "inputs": {"object_name": "cup"}},
                "source": {
                    "class_type": "GrayMaskNode",
                    "inputs": {
                        "target": {"node": "prompt", "output": "prompt"},
                        "edit_mask": {"context": "mask"},
                    },
                },
            },
            "outputs": {"source": {"node": "source", "output": "source"}},
        }
        with self.assertRaisesRegex(ValueError, "expected RGB_IMAGE, got STRING"):
            NodeGraph(definition)

    def test_graph_reports_missing_context(self):
        graph = NodeGraph(self.definition, self.registry)
        with self.assertRaisesRegex(ValueError, "Missing workflow context"):
            graph.run({})

    def test_graph_save_load_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "workflow.json"
            NodeGraph(self.definition, self.registry).save(path)
            graph = NodeGraph.from_file(path, self.registry)
            self.assertEqual(graph.run({"value": "cup"}), {"result": "cup"})

    def test_only_requested_branch_runs(self):
        self.definition["nodes"]["unused"] = {
            "class_type": "EchoNode",
            "inputs": {"value": {"context": "missing"}},
        }
        self.definition["outputs"]["unused"] = {"node": "unused", "output": "value"}
        graph = NodeGraph(self.definition, self.registry)
        self.assertEqual(graph.run({"value": "cup"}, outputs=["result"]), {"result": "cup"})

    def test_custom_prompt_node_can_replace_default(self):
        class CustomPromptNode(BuildPromptNode):
            def execute(self, object_name):
                return {"prompt": f"Custom restore: {object_name}"}

        fixture = dataset_tests.ObjectInpaintingDatasetTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        with DEFAULT_WORKFLOW.open() as handle:
            definition = json.load(handle)
        definition["nodes"]["prompt"]["class_type"] = "CustomPromptNode"
        registry = {**NODE_CLASS_MAPPINGS, "CustomPromptNode": CustomPromptNode}
        graph = NodeGraph(definition, registry)
        record = next(read_records(fixture.manifest))
        execution = graph.start({"record": record, "config": fixture.config, "output": fixture.root / "dataset"})
        sample = execution.get("sample")
        self.assertEqual(sample.prompt, "Custom restore: cup")
        self.assertFalse((fixture.root / "dataset").exists())
        entry = execution.get("entry")
        self.assertEqual(entry["prompt"], sample.prompt)
        execution.get("entry")
        rows = (fixture.root / "dataset" / "train.jsonl").read_text().splitlines()
        self.assertEqual(len(rows), 1)

    def test_batch_builder_uses_supplied_workflow(self):
        fixture = dataset_tests.ObjectInpaintingDatasetTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        with DEFAULT_WORKFLOW.open() as handle:
            definition = json.load(handle)
        definition["nodes"]["prompt"]["inputs"]["object_name"] = "red cup"
        workflow = fixture.root / "custom_workflow.json"
        workflow.write_text(json.dumps(definition))
        output = fixture.root / "dataset"
        build_dataset(fixture.manifest, output, fixture.config, workflow=workflow)
        entry = json.loads((output / "train.jsonl").read_text())
        self.assertIn("red cup", entry["prompt"])
        self.assertTrue((output / "workflow.json").is_file())


if __name__ == "__main__":
    unittest.main()
