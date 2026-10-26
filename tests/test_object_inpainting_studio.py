import json
import tempfile
import unittest
from pathlib import Path

import test_object_inpainting_dataset as dataset_tests
from fastapi.testclient import TestClient
from object_inpainting.graph import NodeGraph
from object_inpainting.nodes import NODE_CLASS_MAPPINGS
from object_inpainting_studio.server import app, serialize_node_output
from PIL import Image


class ObjectInpaintingStudioTests(unittest.TestCase):
    def setUp(self):
        fixture = dataset_tests.ObjectInpaintingDatasetTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.client = TestClient(app)

    def test_serves_browser_application_and_node_catalog(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Object Inpainting Dataset Studio", response.text)
        catalog = self.client.get("/api/node-types").json()
        self.assertIn("RecordInputNode", catalog)
        self.assertEqual(catalog["GrayMaskNode"]["inputs"], {"target": "RGB_IMAGE", "edit_mask": "MASK"})

    def test_default_workflow_and_config_are_exposed(self):
        workflow = self.client.get("/api/workflow/default").json()
        response = self.client.post("/api/workflow/validate", json={"workflow": workflow})
        self.assertEqual(response.status_code, 200)
        self.assertIn("validate", response.json()["execution_order"])
        config = self.client.get("/api/default-config").json()
        self.assertEqual(config["resolution"], 1024)
        self.assertEqual(config["reference_resolution"], 512)

    def test_preview_returns_images_and_prompt(self):
        response = self.client.post(
            "/api/preview",
            json={
                "record": {
                    "target_path": str(self.fixture.root / "target.png"),
                    "mask_path": str(self.fixture.root / "mask.png"),
                    "object_name": "cup",
                },
                "config": {"resolution": 64, "reference_resolution": 32},
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("cup", payload["prompt"])
        for name in ("target", "source", "mask", "reference"):
            self.assertTrue(payload[name].startswith("data:image/png;base64,"))

    def test_preview_rejects_invalid_graph_and_missing_files(self):
        invalid = self.client.post(
            "/api/workflow/validate",
            json={"workflow": {"nodes": {"bad": {"class_type": "MissingNode", "inputs": {}}}, "outputs": {}}},
        )
        self.assertEqual(invalid.status_code, 422)
        missing = self.client.post(
            "/api/preview",
            json={
                "record": {"target_path": "/missing.png", "mask_path": "/missing_mask.png", "object_name": "cup"},
                "config": {"resolution": 64, "reference_resolution": 32},
            },
        )
        self.assertEqual(missing.status_code, 422)
        self.assertIn("No such file", missing.json()["detail"])

    def test_export_executes_browser_workflow(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            response = self.client.post(
                "/api/export",
                json={
                    "manifest_path": str(self.fixture.manifest),
                    "output_dir": str(output),
                    "preview_count": 1,
                    "config": {"resolution": 64, "reference_resolution": 32, "validation_fraction": 0},
                },
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["counts"], {"train": 1, "validation": 0, "rejected": 0})
            entry = json.loads((output / "train.jsonl").read_text())
            self.assertEqual(set(entry), {"target", "source", "mask", "references", "prompt"})
            self.assertTrue((output / "workflow.json").is_file())

    def test_browser_saved_workflow_can_be_loaded_by_backend_and_cli(self):
        workflow = self.client.get("/api/workflow/default").json()
        saved = {
            "version": 1,
            "workflow": workflow,
            "layout": {"target": {"x": 12, "y": 24}},
            "config": {"resolution": 64},
        }
        path = self.fixture.root / "saved.json"
        path.write_text(json.dumps(saved))
        self.assertEqual(NodeGraph.from_file(path).definition, workflow)
        response = self.client.post("/api/workflow/validate", json={"workflow": saved})
        self.assertEqual(response.status_code, 200)

    def test_malformed_workflows_and_config_return_json_errors(self):
        for workflow in ([], "invalid", {"nodes": [], "outputs": {}}, {"nodes": {"broken": None}, "outputs": {}}):
            with self.subTest(workflow=workflow):
                response = self.client.post("/api/workflow/validate", json={"workflow": workflow})
                self.assertEqual(response.status_code, 422)
                self.assertIsInstance(response.json()["detail"], str)
        for config in ([1], {"resolution": "64"}, {"resolution": 64.5}, {"resolution": True}):
            with self.subTest(config=config):
                response = self.client.post("/api/workflow/validate", json={"config": config})
                self.assertEqual(response.status_code, 422)

    def test_runtime_wrong_binding_is_reported_without_server_crash(self):
        workflow = self.client.get("/api/workflow/default").json()
        workflow["nodes"]["selection"]["inputs"]["config"] = "not a config"
        response = self.client.post(
            "/api/preview",
            json={
                "workflow": workflow,
                "record": {
                    "target_path": str(self.fixture.root / "target.png"),
                    "mask_path": str(self.fixture.root / "mask.png"),
                    "object_name": "cup",
                },
                "config": {"resolution": 64, "reference_resolution": 32},
            },
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("invalid runtime value", response.json()["detail"])

    def test_preview_blocks_writer_dependencies(self):
        workflow = self.client.get("/api/workflow/default").json()
        output = self.fixture.root / "must_not_exist"
        workflow["nodes"]["save"]["inputs"]["output"] = str(output)
        workflow["nodes"]["written_target"] = {
            "class_type": "LoadTargetNode",
            "inputs": {"path": {"node": "save", "output": "manifest_path"}},
        }
        original_inputs = dict(workflow["nodes"]["assemble"]["inputs"])
        original_inputs["target"] = {"node": "written_target", "output": "target"}
        workflow["nodes"]["unsafe_sample"] = {"class_type": "AssembleSampleNode", "inputs": original_inputs}
        workflow["outputs"]["sample"] = {"node": "unsafe_sample", "output": "sample"}
        response = self.client.post(
            "/api/preview",
            json={
                "workflow": workflow,
                "record": {
                    "target_path": str(self.fixture.root / "target.png"),
                    "mask_path": str(self.fixture.root / "mask.png"),
                    "object_name": "cup",
                },
                "config": {"resolution": 64, "reference_resolution": 32},
            },
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("not allowed during preview", response.json()["detail"])
        self.assertFalse(output.exists())

    def test_export_failure_does_not_publish_partial_dataset(self):
        workflow = self.client.get("/api/workflow/default").json()
        workflow["nodes"]["jsonl"]["inputs"]["path"] = None
        output = self.fixture.root / "failed_export"
        response = self.client.post(
            "/api/export",
            json={
                "workflow": workflow,
                "manifest_path": str(self.fixture.manifest),
                "output_dir": str(output),
                "config": {"resolution": 64, "reference_resolution": 32},
            },
        )
        self.assertEqual(response.status_code, 422)
        self.assertFalse(output.exists())
        self.assertEqual(list(self.fixture.root.glob(".object-inpainting-*")), [])

    def test_upload_image_and_reject_non_image(self):
        response = self.client.post(
            "/api/upload", files={"file": ("target.png", (self.fixture.root / "target.png").read_bytes(), "image/png")}
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(Path(response.json()["path"]).is_file())
        bad = self.client.post("/api/upload", files={"file": ("bad.png", b"not an image", "image/png")})
        self.assertEqual(bad.status_code, 422)

    def test_each_node_can_preview_images_masks_and_text(self):
        request = {
            "record": {
                "target_path": str(self.fixture.root / "target.png"),
                "mask_path": str(self.fixture.root / "mask.png"),
                "object_name": "cup",
            },
            "config": {"resolution": 64, "reference_resolution": 32},
        }
        for node_id, port, kind in (
            ("target", "target", "image"),
            ("expand", "edit_mask", "image"),
            ("selection", "bbox", "text"),
            ("prompt", "prompt", "text"),
            ("validate", "sample", "sample"),
        ):
            with self.subTest(node_id=node_id):
                response = self.client.post("/api/node/preview", json={**request, "node_id": node_id})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["outputs"][port]["kind"], kind)
        blocked = self.client.post("/api/node/preview", json={**request, "node_id": "save"})
        self.assertEqual(blocked.status_code, 422)
        self.assertIn("not allowed during preview", blocked.json()["detail"])

    def test_all_nodes_explain_their_ports_and_test_steps(self):
        catalog = self.client.get("/api/node-types").json()
        self.assertEqual(set(catalog), set(NODE_CLASS_MAPPINGS))
        for name, info in catalog.items():
            with self.subTest(name=name):
                self.assertTrue(info["help"]["title"])
                self.assertTrue(info["help"]["summary"])
                self.assertTrue(info["help"]["test"])
                self.assertEqual(set(info["help"]["inputs"]), set(info["inputs"]))
                self.assertEqual(set(info["help"]["outputs"]), set(info["outputs"]))

    def test_single_target_and_prompt_need_no_sample_record(self):
        for class_type, inputs, port, kind in (
            ("LoadTargetNode", {"path": str(self.fixture.root / "target.png")}, "target", "image"),
            ("BuildPromptNode", {"object_name": "cup"}, "prompt", "text"),
        ):
            with self.subTest(class_type=class_type):
                workflow = {"nodes": {"one": {"class_type": class_type, "inputs": inputs}}, "outputs": {}}
                response = self.client.post("/api/node/preview", json={"workflow": workflow, "node_id": "one"})
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()["outputs"][port]["kind"], kind)
                self.assertEqual(response.json()["executed_nodes"], ["one"])

    def test_single_mask_and_gray_nodes_accept_direct_image_paths(self):
        nodes = {
            "expand": {
                "class_type": "ExpandMaskNode",
                "inputs": {"object_mask": str(self.fixture.root / "mask.png"), "config": {"context": "config"}},
            },
            "gray": {
                "class_type": "GrayMaskNode",
                "inputs": {
                    "target": str(self.fixture.root / "target.png"),
                    "edit_mask": {"node": "expand", "output": "edit_mask"},
                },
            },
        }
        response = self.client.post(
            "/api/node/preview",
            json={"workflow": {"nodes": nodes, "outputs": {}}, "node_id": "gray", "config": {"dilation_radius": 0}},
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["executed_nodes"], ["expand", "gray"])
        self.assertEqual(response.json()["outputs"]["source"]["kind"], "image")
        self.assertEqual(nodes["expand"]["inputs"]["object_mask"], str(self.fixture.root / "mask.png"))

    def test_node_preview_ignores_unrelated_broken_branches_but_rejects_its_own_cycle(self):
        nodes = {
            "prompt": {"class_type": "BuildPromptNode", "inputs": {"object_name": "cup"}},
            "unrelated": {"class_type": "MissingNode", "inputs": {}},
        }
        request = {"workflow": {"nodes": nodes, "outputs": {"bad": None}}, "node_id": "prompt"}
        response = self.client.post("/api/node/preview", json=request)
        self.assertEqual(response.status_code, 200, response.text)
        nodes["prompt"]["inputs"]["object_name"] = {"node": "prompt", "output": "prompt"}
        response = self.client.post("/api/node/preview", json=request)
        self.assertEqual(response.status_code, 422)
        self.assertIn("cycle", response.json()["detail"])

    def test_practice_workflow_can_validate_without_weakening_dataset_contract(self):
        workflow = {
            "nodes": {"prompt": {"class_type": "BuildPromptNode", "inputs": {"object_name": "cup"}}},
            "outputs": {},
        }
        response = self.client.post("/api/workflow/validate", json={"workflow": workflow, "require_dataset": False})
        self.assertEqual(response.status_code, 200)
        response = self.client.post("/api/workflow/validate", json={"workflow": workflow})
        self.assertEqual(response.status_code, 422)
        self.assertIn("must expose output", response.json()["detail"])

    def test_output_detection_uses_values_and_handles_mixed_objects(self):
        path = self.fixture.root / "no_extension"
        Image.new("RGB", (8, 6), "red").save(path, format="PNG")
        result = serialize_node_output({"picture": path, "text": "hello", "mask": Image.new("L", (8, 6))})
        self.assertEqual(result["kind"], "object")
        self.assertEqual(result["items"]["picture"]["kind"], "image")
        self.assertEqual(result["items"]["picture"]["size"], (8, 6))
        self.assertEqual(result["items"]["mask"]["mode"], "L")
        self.assertEqual(result["items"]["text"], {"kind": "text", "value": "hello"})
        self.assertEqual(serialize_node_output("<img src=x onerror=alert(1)> > 127")["kind"], "text")


if __name__ == "__main__":
    unittest.main()
