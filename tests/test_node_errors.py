import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from object_inpainting.graph import NodeExecutionError, NodeGraph
from object_inpainting.nodes import NODE_CLASS_MAPPINGS, BuildPromptNode, Node
from object_inpainting_studio.batch import BatchCancelled, BatchJobs
from object_inpainting_studio.server import app
from PIL import Image


class EchoNode(Node):
    INPUT_TYPES = {"value": "STRING"}
    OUTPUT_TYPES = {"value": "STRING"}

    def execute(self, value):
        return {"value": value}


class FailingImageNode(Node):
    INPUT_TYPES = {"image": "RGB_IMAGE"}
    OUTPUT_TYPES = {"image": "RGB_IMAGE"}

    def execute(self, image):
        raise RuntimeError("test inference failed")


class NodeErrorTests(unittest.TestCase):
    def graph(self):
        return NodeGraph(
            {
                "nodes": {
                    "origin": {"class_type": "EchoNode", "inputs": {"value": {"context": "value"}}},
                    "downstream": {
                        "class_type": "EchoNode",
                        "inputs": {"value": {"node": "origin", "output": "value"}},
                    },
                },
                "outputs": {"result": {"node": "downstream", "output": "value"}},
            },
            {"EchoNode": EchoNode},
        )

    def test_errors_retain_original_node_type_cause_and_single_log(self):
        for error in (ValueError("bad input"), RuntimeError("CUDA error"), ImportError("missing dependency")):
            with self.subTest(error=type(error).__name__):
                graph = self.graph()
                with patch.object(EchoNode, "execute", side_effect=error), self.assertLogs(
                    "object_inpainting.graph", level="ERROR"
                ) as logs:
                    with self.assertRaises(NodeExecutionError) as caught:
                        graph.run({"value": "test"})
                failure = caught.exception
                self.assertEqual(failure.node_id, "origin")
                self.assertEqual(failure.class_type, "EchoNode")
                self.assertEqual(failure.error_type, type(error).__name__)
                self.assertIs(failure.__cause__, error)
                self.assertIn(str(error), str(failure))
                self.assertNotIn("downstream", str(failure))
                self.assertEqual(len(logs.records), 1)
                self.assertIsNotNone(logs.records[0].exc_info)

    def test_missing_context_and_runtime_type_identify_node(self):
        for context, message in (({}, "Missing workflow context"), ({"value": 4}, "invalid runtime value")):
            with self.subTest(context=context), self.assertLogs("object_inpainting.graph", level="ERROR"):
                with self.assertRaisesRegex(NodeExecutionError, message) as caught:
                    self.graph().run(context)
                self.assertEqual(caught.exception.node_id, "origin")

    def test_bad_output_identifies_node(self):
        with patch.object(EchoNode, "execute", return_value={}), self.assertLogs(
            "object_inpainting.graph", level="ERROR"
        ):
            with self.assertRaisesRegex(NodeExecutionError, "returned outputs") as caught:
                self.graph().run({"value": "test"})
        self.assertEqual(caught.exception.node_id, "origin")

    def test_cancellation_and_keyboard_interrupt_are_not_node_failures(self):
        for error in (BatchCancelled(), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__), patch.object(EchoNode, "execute", side_effect=error):
                with self.assertRaises(type(error)):
                    self.graph().run({"value": "test"})

    def test_api_exposes_upstream_error_and_logs_traceback(self):
        workflow = {
            "nodes": {
                "origin": {"class_type": "BuildPromptNode", "inputs": {"object_name": "cup"}},
                "selected": {
                    "class_type": "BuildPromptNode",
                    "inputs": {"object_name": {"node": "origin", "output": "prompt"}},
                },
            },
            "outputs": {},
        }
        with TestClient(app) as client, patch.object(
            BuildPromptNode, "execute", side_effect=RuntimeError("test inference failed")
        ), self.assertLogs(level="ERROR") as logs:
            response = client.post("/api/node/preview", json={"workflow": workflow, "node_id": "selected"})
        self.assertEqual(response.status_code, 422, response.text)
        data = response.json()
        self.assertIsInstance(data["detail"], str)
        self.assertIn("test inference failed", data["detail"])
        self.assertEqual(
            data["node_error"],
            {
                "node_id": "origin",
                "class_type": "BuildPromptNode",
                "error_type": "RuntimeError",
                "message": "test inference failed",
            },
        )
        self.assertTrue(any("/api/node/preview" in record.getMessage() for record in logs.records))

    def test_image_loading_error_identifies_node_before_execution(self):
        with tempfile.TemporaryDirectory() as directory, TestClient(app) as client:
            workflow = {
                "nodes": {
                    "colors": {
                        "class_type": "AutoColorRegionsNode",
                        "inputs": {"image": str(Path(directory) / "missing.png"), "mask": "unused.png"},
                    }
                },
                "outputs": {},
            }
            with self.assertLogs("object_inpainting_studio.server", level="ERROR"):
                response = client.post("/api/node/preview", json={"workflow": workflow, "node_id": "colors"})
        self.assertEqual(response.status_code, 422, response.text)
        self.assertEqual(response.json()["node_error"]["node_id"], "colors")

    def test_batch_report_exposes_node_for_inference_and_source_errors(self):
        for corrupt_source in (False, True):
            with self.subTest(corrupt_source=corrupt_source), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "input.png"
                if corrupt_source:
                    source.write_bytes(b"not an image")
                else:
                    Image.new("RGB", (16, 16), "red").save(source)
                graph = NodeGraph(
                    {
                        "nodes": {
                            "folder": {"class_type": "FolderImagesNode", "inputs": {"dir": directory}},
                            "inference": {
                                "class_type": "FailingImageNode",
                                "inputs": {"image": {"node": "folder", "output": "image"}},
                            },
                            "save": {
                                "class_type": "SaveImagesNode",
                                "inputs": {"img1": {"node": "inference", "output": "image"}, "dir": str(root / "out")},
                            },
                        },
                        "outputs": {},
                    },
                    {**NODE_CLASS_MAPPINGS, "FailingImageNode": FailingImageNode},
                )
                manager = BatchJobs()
                with self.assertLogs(level="ERROR"):
                    started = manager.start(graph, {}, "folder", "save", [source], 0, False)
                    manager.threads[started["id"]].join(10)
                self.assertFalse(manager.threads[started["id"]].is_alive())
                job = manager.get(started["id"])
                self.assertEqual(job["status"], "failed")
                expected = "folder" if corrupt_source else "inference"
                self.assertEqual(job["node_error"]["node_id"], expected)
                self.assertEqual(job["results"][0]["node_error"], job["node_error"])
                self.assertIn(expected, job["error"])


if __name__ == "__main__":
    unittest.main()