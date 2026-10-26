import json
import tempfile
import unittest
from pathlib import Path
from threading import Event
from unittest.mock import patch

from fastapi.testclient import TestClient
from object_inpainting.graph import NodeGraph
from object_inpainting.nodes import NODE_CLASS_MAPPINGS, Node
from object_inpainting.nodes.folder_images import FolderImagesNode, scan_folder
from object_inpainting_studio import server
from object_inpainting_studio.batch import BatchJobs
from PIL import Image


class FolderBatchTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.input = self.root / "input"
        self.input.mkdir()
        self.output = self.root / "output"
        self.manager = BatchJobs()
        original = server.BATCH_JOBS
        server.BATCH_JOBS = self.manager
        self.addCleanup(setattr, server, "BATCH_JOBS", original)
        self.client = TestClient(server.app)
        self.addCleanup(self.client.close)

    def image(self, name, color):
        path = self.input / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 24), color).save(path)
        return path

    def workflow(self):
        return {
            "nodes": {
                "folder": {"class_type": "FolderImagesNode", "inputs": {"dir": str(self.input)}},
                "save": {
                    "class_type": "SaveImagesNode",
                    "inputs": {
                        "img1": {"node": "folder", "output": "image"},
                        "img3": {"node": "folder", "output": "image"},
                        "dir": str(self.output),
                    },
                },
            },
            "outputs": {},
        }

    def start(self, workflow=None, **kwargs):
        response = self.client.post(
            "/api/batch", json={"workflow": workflow or self.workflow(), "node_id": "save", **kwargs}
        )
        self.assertEqual(response.status_code, 200, response.text)
        job = response.json()
        self.manager.threads[job["id"]].join(10)
        self.assertFalse(self.manager.threads[job["id"]].is_alive())
        return self.client.get(f"/api/batch/{job['id']}").json()

    def test_folder_sorted_recursive_preview_and_validation(self):
        self.image("b.PNG", "blue")
        self.image("A.png", "red")
        self.image("nested/c.png", "green")
        (self.input / "ignore.txt").write_text("not an image")
        self.assertEqual([p.name for p in scan_folder(self.input)], ["A.png", "b.PNG"])
        self.assertEqual(len(scan_folder(self.input, 1)), 3)
        result = FolderImagesNode().execute(self.input, index=1)
        self.assertEqual(result["filename"], "b.PNG")
        self.assertEqual(result["image"].getpixel((0, 0)), (0, 0, 255))
        for kwargs in ({"index": -1}, {"index": 2}, {"recursive": 2}, {"index": True}):
            with self.assertRaises(ValueError):
                FolderImagesNode().execute(self.input, **kwargs)
        response = self.client.post("/api/node/preview", json={"workflow": self.workflow(), "node_id": "folder"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["outputs"]["filename"]["value"], "A.png")
        self.assertFalse(self.output.exists())

    def test_batch_saves_each_source_and_correlated_branches(self):
        self.image("a.png", "red")
        self.image("b.png", "blue")
        job = self.start()
        self.assertEqual((job["status"], job["total"], job["processed"], job["succeeded"]), ("completed", 2, 2, 2))
        self.assertEqual(job["failed"], 0)
        for item, color in zip(job["results"], ((255, 0, 0), (0, 0, 255))):
            self.assertEqual(set(item["paths"]), {"img1", "img3"})
            self.assertEqual(Path(item["paths"]["img1"]).name, Path(item["paths"]["img3"]).name)
            for path in item["paths"].values():
                with Image.open(path) as image:
                    self.assertEqual(image.getpixel((0, 0)), color)
        self.assertNotEqual(job["results"][0]["paths"], job["results"][1]["paths"])
        self.assertEqual(self.client.get("/api/batch/latest").json()["job"], job)
        self.assertNotIn("data:image", json.dumps(job))

    def test_loop_plan_counts_without_executing_nodes_or_writing(self):
        self.image("a.png", "red")
        (self.input / "b.png").write_bytes(b"broken image")
        with patch.object(FolderImagesNode, "execute", side_effect=AssertionError("must not execute")):
            response = self.client.post("/api/batch/plan", json={"workflow": self.workflow(), "node_id": "save"})
        self.assertEqual(response.status_code, 200, response.text)
        plan = response.json()
        self.assertEqual((plan["source_id"], plan["target_id"], plan["total"]), ("folder", "save", 2))
        self.assertEqual(plan["executed_nodes"], ["folder", "save"])
        self.assertEqual(plan["current_path"], str(self.input / "a.png"))
        self.assertFalse(self.output.exists())
        self.assertIsNone(self.manager.latest())

    def test_loop_trial_uses_same_filtered_index_as_batch(self):
        self.image("a-generated/img1/old.png", "green")
        self.image("b.png", "red")
        self.image("c.png", "blue")
        self.output = self.input / "a-generated"
        workflow = self.workflow()
        workflow["nodes"]["folder"]["inputs"].update(recursive=1, index=1)
        payload = {"workflow": workflow, "node_id": "save"}
        plan = self.client.post("/api/batch/plan", json=payload).json()
        self.assertEqual((plan["total"], plan["current_path"]), (1, str(self.input / "c.png")))
        response = self.client.post("/api/batch/preview", json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["executed_nodes"], ["folder", "save"])
        written = [path for path in (self.output / "img1").glob("*.png") if path.name != "old.png"]
        self.assertEqual(len(written), 1)
        with Image.open(written[0]) as image:
            self.assertEqual(image.getpixel((0, 0)), (0, 0, 255))
        self.assertIsNone(self.manager.latest())
        self.assertEqual(self.start(workflow)["results"][0]["source"], plan["current_path"])

    def test_loop_plan_and_trial_reject_ambiguous_or_unsafe_graphs(self):
        self.image("a.png", "red")
        for endpoint in ("plan", "preview"):
            workflow = self.workflow()
            workflow["nodes"]["second"] = workflow["nodes"]["folder"].copy()
            workflow["nodes"]["save"]["inputs"]["img3"] = {"node": "second", "output": "image"}
            response = self.client.post(f"/api/batch/{endpoint}", json={"workflow": workflow, "node_id": "save"})
            self.assertEqual(response.status_code, 422, response.text)
            self.assertIn("只能连接一个", response.json()["detail"])
            workflow = self.workflow()
            workflow["nodes"]["save"]["inputs"]["dir"] = str(self.input)
            response = self.client.post(f"/api/batch/{endpoint}", json={"workflow": workflow, "node_id": "save"})
            self.assertEqual(response.status_code, 422, response.text)
        self.assertFalse(self.output.exists())

    def test_corrupt_image_continue_and_stop_on_error(self):
        self.image("a.png", "red")
        (self.input / "b.png").write_bytes(b"bad png")
        self.image("c.png", "blue")
        job = self.start()
        self.assertEqual((job["status"], job["succeeded"], job["failed"]), ("completed", 2, 1))
        self.assertIn("error", job["results"][1])
        job = self.start(continue_on_error=False)
        self.assertEqual((job["status"], job["processed"], job["succeeded"], job["failed"]), ("failed", 2, 1, 1))
        self.assertTrue(job["error"])

    def test_start_index_and_nested_output_exclusion(self):
        self.image("a.png", "red")
        self.image("b.png", "blue")
        self.image("generated/img1/old.png", "green")
        self.output = self.input / "generated"
        workflow = self.workflow()
        workflow["nodes"]["folder"]["inputs"].update(recursive=1, index=1)
        job = self.start(workflow)
        self.assertEqual((job["total"], job["processed"]), (1, 1))
        self.assertEqual(Path(job["results"][0]["source"]).name, "b.png")
        job = self.start(workflow)
        self.assertEqual(job["total"], 1)

    def test_invalid_batch_rejected_without_writes(self):
        self.image("a.png", "red")
        cases = []
        workflow = self.workflow()
        workflow["nodes"]["save"]["inputs"]["dir"] = str(self.input)
        cases.append(workflow)
        workflow = self.workflow()
        workflow["nodes"]["folder"]["inputs"]["index"] = 99
        cases.append(workflow)
        workflow = self.workflow()
        workflow["nodes"]["save"]["inputs"]["img1"] = str(self.input / "a.png")
        workflow["nodes"]["save"]["inputs"]["img3"] = None
        cases.append(workflow)
        workflow = self.workflow()
        workflow["nodes"]["second"] = {"class_type": "FolderImagesNode", "inputs": {"dir": str(self.input)}}
        workflow["nodes"]["save"]["inputs"]["img3"] = {"node": "second", "output": "image"}
        cases.append(workflow)
        for workflow in cases:
            response = self.client.post("/api/batch", json={"workflow": workflow, "node_id": "save"})
            self.assertEqual(response.status_code, 422, response.text)
        self.assertFalse(self.output.exists())
        self.assertEqual(self.client.get("/api/batch/missing").status_code, 404)
        self.assertEqual(self.client.post("/api/batch/missing/cancel").status_code, 404)

    def test_shared_branch_runs_once_per_image_and_cancel_preserves_completed_files(self):
        self.image("a.png", "red")
        self.image("b.png", "blue")
        entered, release = Event(), Event()
        calls = []

        class CountNode(Node):
            INPUT_TYPES = {"image": "RGB_IMAGE"}
            OUTPUT_TYPES = {"image": "RGB_IMAGE"}
            CATEGORY = "test"

            def execute(self, image):
                calls.append(image.getpixel((0, 0)))
                if len(calls) == 2:
                    entered.set()
                    if not release.wait(5):
                        raise ValueError("Timed out")
                return {"image": image}

        definition = self.workflow()
        definition["nodes"]["count"] = {
            "class_type": "CountNode",
            "inputs": {"image": {"node": "folder", "output": "image"}},
        }
        for port in ("img1", "img3"):
            definition["nodes"]["save"]["inputs"][port] = {"node": "count", "output": "image"}
        graph = NodeGraph(definition, registry={**NODE_CLASS_MAPPINGS, "CountNode": CountNode})
        job = self.manager.start(graph, {}, "folder", "save", scan_folder(self.input), 0, True)
        try:
            self.assertTrue(entered.wait(5))
            self.manager.cancel(job["id"])
        finally:
            release.set()
            self.manager.threads[job["id"]].join(10)
        final = self.manager.get(job["id"])
        self.assertEqual(calls, [(255, 0, 0), (0, 0, 255)])
        self.assertEqual((final["status"], final["processed"], final["succeeded"]), ("cancelled", 1, 1))
        for path in final["results"][0]["paths"].values():
            self.assertTrue(Path(path).is_file())
        self.assertEqual(len(list((self.output / "img1").glob("*.png"))), 1)

    def test_cancel_waits_current_node_and_blocks_duplicate_start(self):
        self.image("a.png", "red")
        self.image("b.png", "blue")
        entered, release = Event(), Event()

        class WaitNode(Node):
            INPUT_TYPES = {"image": "RGB_IMAGE"}
            OUTPUT_TYPES = {"image": "RGB_IMAGE"}
            CATEGORY = "test"

            def execute(self, image):
                entered.set()
                if not release.wait(5):
                    raise ValueError("Timed out")
                return {"image": image}

        definition = self.workflow()
        definition["nodes"]["wait"] = {
            "class_type": "WaitNode",
            "inputs": {"image": {"node": "folder", "output": "image"}},
        }
        definition["nodes"]["save"]["inputs"]["img1"] = {"node": "wait", "output": "image"}
        graph = NodeGraph(definition, registry={**NODE_CLASS_MAPPINGS, "WaitNode": WaitNode})
        args = (graph, {}, "folder", "save", scan_folder(self.input), 0, True)
        job = self.manager.start(*args)
        try:
            self.assertTrue(entered.wait(5))
            with self.assertRaisesRegex(ValueError, "已有批量任务"):
                self.manager.start(*args)
            response = self.client.post(f"/api/batch/{job['id']}/cancel")
            self.assertEqual(response.json()["status"], "cancelling")
        finally:
            release.set()
            self.manager.threads[job["id"]].join(10)
        final = self.manager.get(job["id"])
        self.assertEqual((final["status"], final["processed"]), ("cancelled", 0))
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
