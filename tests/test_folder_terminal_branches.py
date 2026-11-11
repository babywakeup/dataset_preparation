"""Folder runs include all connected result branches, sharing accepted inputs."""

import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from object_inpainting.nodes import LoopDemoImageNode, LoopDemoScoreNode
from object_inpainting_studio import server
from object_inpainting_studio.batch import BatchJobs


def link(node, output):
    return {"node": node, "output": output}


class FolderTerminalTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.input = self.root / "input"
        self.input.mkdir()
        for name, color in (("a.png", "red"), ("b.png", "blue")):
            Image.new("RGB", (80, 60), color).save(self.input / name)
        self.manager = BatchJobs()
        for fixture in (patch.object(server, "BATCH_JOBS", self.manager),
                        patch("object_inpainting.nodes.loop_demo.time.sleep")):
            fixture.start()
            self.addCleanup(fixture.stop)
        self.client = TestClient(server.app)
        self.addCleanup(self.client.close)

    def workflow(self, conditional=False):
        nodes = {
            "folder": {"class_type": "FolderImagesNode", "inputs": {"dir": str(self.input)}},
            "demo": {"class_type": "LoopDemoImageNode", "inputs": {"image": link("folder", "image"), "seed": 0}},
            "score": {"class_type": "LoopDemoScoreNode", "inputs": {"image": link("demo", "image")}},
            "save": {"class_type": "SaveImagesNode", "inputs": {
                "img1": link("demo", "image"), "filename": link("folder", "filename"), "dir": str(self.root / "out")}},
        }
        workflow = {"nodes": nodes, "outputs": {}}
        if conditional:
            nodes["condition"] = deepcopy(nodes["score"])
            workflow["conditional_loop"] = {"loops": [{
                "id": "loop", "start_nodes": ["demo"], "end_nodes": ["condition"],
                "seed_nodes": ["demo"], "max_iterations": 3,
                "conditions": [{"node": "condition", "output": "score", "operator": ">=", "value": 255}],
            }]}
        return workflow

    def run_job(self, workflow, endpoint="/api/batch"):
        response = self.client.post(endpoint, json={"workflow": workflow, "node_id": "save"})
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        if "id" in result:
            self.manager.threads[result["id"]].join(5)
            self.assertFalse(self.manager.threads[result["id"]].is_alive())
            result = self.manager.get(result["id"])
            self.assertEqual(result["status"], "completed", result)
        return result

    def test_plain_all_terminals_score_and_multiple_saves_share_generation(self):
        workflow = self.workflow()
        workflow["nodes"]["save2"] = deepcopy(workflow["nodes"]["save"])
        workflow["nodes"]["save2"]["inputs"]["dir"] = str(self.root / "out2")
        workflow["nodes"]["disconnected"] = {"class_type": "LoadTargetNode", "inputs": {"path": "missing.png"}}
        with patch.object(LoopDemoImageNode, "execute", autospec=True, side_effect=LoopDemoImageNode.execute) as generate, \
             patch.object(LoopDemoScoreNode, "execute", autospec=True, side_effect=LoopDemoScoreNode.execute) as score:
            result = self.run_job(workflow)
        self.assertEqual((generate.call_count, score.call_count), (2, 2))
        self.assertEqual(len(list(self.root.rglob("out*/*/*.png"))), 4)
        self.assertEqual(set(result["results"][-1]["preview"]), {"score", "save", "save2"})
        self.assertEqual(set(result["results"][-1]["paths_by_node"]), {"save", "save2"})

    def test_trial_also_runs_scoring_terminal(self):
        with patch.object(LoopDemoScoreNode, "execute", autospec=True, side_effect=LoopDemoScoreNode.execute) as score:
            result = self.run_job(self.workflow(), "/api/batch/preview")
        self.assertEqual(score.call_count, 1)
        self.assertIn("score", result["executed_nodes"])
        self.assertEqual(result["terminal_outputs"]["score"]["score"]["value"], 0)

    def test_empty_filename_defaults_to_each_original_name_in_plain_and_conditional_batch(self):
        for conditional in (False, True):
            with self.subTest(conditional=conditional):
                workflow = self.workflow(conditional=conditional)
                workflow["nodes"]["save"]["inputs"].update(filename="", dir=str(self.root / f"named-{conditional}"))
                result = self.run_job(workflow)
                self.assertEqual(result["succeeded"], 2)
                self.assertEqual([Path(entry["paths"]["img1"]).name for entry in result["results"]], ["a.png", "b.png"])
                self.assertEqual(workflow["nodes"]["save"]["inputs"]["filename"], "")
                repeated = self.run_job(workflow)
                self.assertEqual((repeated["succeeded"], repeated["failed"]), (0, 2))

    def test_single_save_infers_name_from_load_target_and_keeps_explicit_name(self):
        workflow = {"nodes": {
            "load": {"class_type": "LoadTargetNode", "inputs": {"path": str(self.input / "a.png")}},
            "save": {"class_type": "SaveImagesNode", "inputs": {
                "img1": link("load", "target"), "dir": str(self.root / "single")}},
        }, "outputs": {}}
        response = self.client.post("/api/node/save-images", json={"workflow": workflow, "node_id": "save"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue((self.root / "single" / "img1" / "a.png").exists())
        workflow["nodes"]["save"]["inputs"]["filename"] = "custom-name.jpg"
        response = self.client.post("/api/node/save-images", json={"workflow": workflow, "node_id": "save"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue((self.root / "single" / "img1" / "custom-name.png").exists())

    def test_single_folder_save_uses_current_original_filename(self):
        workflow = self.workflow()
        workflow["nodes"]["folder"]["inputs"]["index"] = 1
        workflow["nodes"]["save"]["inputs"]["filename"] = ""
        with patch("object_inpainting.nodes.loop_demo.time.sleep"):
            response = self.client.post("/api/node/save-images", json={"workflow": workflow, "node_id": "save"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue((self.root / "out" / "img1" / "b.png").exists())

    def test_original_filename_preserves_unicode_and_multiple_dots(self):
        original = self.input / "服装.正面.JPG"
        Image.new("RGB", (12, 10), "green").save(original)
        workflow = {"nodes": {
            "load": {"class_type": "LoadTargetNode", "inputs": {"path": str(original)}},
            "save": {"class_type": "SaveImagesNode", "inputs": {"img1": link("load", "target"), "dir": str(self.root / "unicode")}},
        }, "outputs": {}}
        response = self.client.post("/api/node/save-images", json={"workflow": workflow, "node_id": "save"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue((self.root / "unicode" / "img1" / "服装.正面.png").exists())

    def test_selected_score_runs_without_saving_or_validating_unused_branches(self):
        workflow = self.workflow()
        workflow["batch_execution"] = {"mode": "selected_results", "result_nodes": ["score"], "disabled_nodes": []}
        workflow["nodes"]["unfinished"] = {"class_type": "UnknownType", "inputs": {"image": link("demo", "image")}}
        workflow["nodes"]["save"]["inputs"]["dir"] = ""
        result = self.run_job(workflow)
        self.assertEqual(result["succeeded"], 2)
        self.assertEqual(set(result["results"][-1]["preview"]), {"score"})
        self.assertEqual(result["results"][-1]["paths"], {})
        self.assertFalse((self.root / "out").exists())

    def test_disabled_unfinished_branch_is_not_validated(self):
        workflow = self.workflow()
        workflow["nodes"]["unfinished"] = {"class_type": "UnknownType", "inputs": {"image": link("demo", "image")}}
        workflow["batch_execution"] = {"disabled_nodes": ["unfinished"]}
        result = self.run_job(workflow)
        self.assertEqual(result["succeeded"], 2)

    def test_selected_result_cannot_depend_on_disabled_node(self):
        workflow = self.workflow()
        workflow["batch_execution"] = {"mode": "selected_results", "result_nodes": ["score"], "disabled_nodes": ["demo"]}
        response = self.client.post("/api/batch/plan", json={"workflow": workflow, "node_id": "save"})
        self.assertEqual(response.status_code, 422)
        self.assertIn("禁用", response.json()["detail"])

    def test_plan_exposes_authoritative_roles_and_per_item_retry_stages(self):
        workflow = self.workflow(conditional=True)
        response = self.client.post("/api/batch/plan", json={"workflow": workflow, "node_id": "save"})
        self.assertEqual(response.status_code, 200, response.text)
        plan = response.json()
        self.assertEqual(plan["node_roles"]["folder"], "source")
        self.assertEqual(plan["node_roles"]["demo"], "retry")
        self.assertEqual(plan["node_roles"]["save"], "result")
        self.assertEqual([stage["id"] for stage in plan["stages"]], ["input", "retry", "results"])
        result = self.run_job(workflow)
        self.assertEqual(result["current_item"]["status"], "succeeded")
        self.assertEqual([entry["item_index"] for entry in result["results"]], [0, 1])

    def test_selected_condition_result_accepts_without_any_save_node(self):
        workflow = self.workflow(conditional=True)
        del workflow["nodes"]["save"]
        workflow["batch_execution"] = {"mode": "selected_results", "result_nodes": ["condition"]}
        with patch.object(LoopDemoImageNode, "execute", autospec=True, side_effect=LoopDemoImageNode.execute) as generate:
            response = self.client.post("/api/batch", json={"workflow": workflow, "node_id": "folder", "source_id": "folder"})
            self.assertEqual(response.status_code, 200, response.text)
            job_id = response.json()["id"]
            self.manager.threads[job_id].join(5)
            result = self.manager.get(job_id)
        self.assertEqual(result["succeeded"], 2, result)
        self.assertEqual(generate.call_count, 6)
        self.assertEqual(result["results"][-1]["preview"]["condition"]["score"]["value"], 255)

    def test_selected_intermediate_uses_accepted_value_without_rerun(self):
        workflow = self.workflow(conditional=True)
        workflow["batch_execution"] = {"mode": "selected_results", "result_nodes": ["demo"]}
        with patch.object(LoopDemoImageNode, "execute", autospec=True, side_effect=LoopDemoImageNode.execute) as generate:
            result = self.run_job(workflow)
        self.assertEqual(generate.call_count, 6)
        self.assertEqual(result["results"][-1]["preview"]["demo"]["value"]["value"], 2)

    def test_selected_results_ignore_unrelated_unfinished_loop(self):
        workflow = self.workflow()
        workflow["batch_execution"] = {"mode": "selected_results", "result_nodes": ["score"]}
        workflow["conditional_loop"] = {"loops": [{"id": "unfinished", "start_nodes": [], "end_nodes": [], "conditions": []}]}
        result = self.run_job(workflow)
        self.assertEqual(result["succeeded"], 2)
        self.assertEqual(result["results"][-1]["preview"]["score"]["score"]["value"], 0)

    def test_legacy_selected_intermediate_preserves_accepted_result(self):
        workflow = self.workflow(conditional=True)
        settings = workflow["conditional_loop"]["loops"][0]
        workflow["conditional_loop"] = {key: value for key, value in settings.items() if key not in {"id", "end_nodes"}}
        workflow["batch_execution"] = {"mode": "selected_results", "result_nodes": ["demo"]}
        with patch.object(LoopDemoImageNode, "execute", autospec=True, side_effect=LoopDemoImageNode.execute) as generate:
            result = self.run_job(workflow)
        self.assertEqual(generate.call_count, 6)
        self.assertEqual(result["results"][-1]["preview"]["demo"]["value"]["value"], 2)

    def test_disabled_condition_dependency_is_rejected_not_bypassed(self):
        workflow = self.workflow(conditional=True)
        workflow["batch_execution"] = {"disabled_nodes": ["condition"]}
        response = self.client.post("/api/batch/plan", json={"workflow": workflow, "node_id": "save"})
        self.assertEqual(response.status_code, 422)
        self.assertIn("禁用", response.json()["detail"])

    def test_result_metadata_survives_workflow_roundtrip(self):
        workflow = self.workflow()
        workflow["batch_execution"] = {"mode": "selected_results", "result_nodes": ["score"], "disabled_nodes": ["save"]}
        response = self.client.post("/api/workflow/validate", json={"workflow": workflow, "require_dataset": False})
        self.assertEqual(response.status_code, 200, response.text)
        restored = response.json()["workflow"]
        self.assertEqual(restored["batch_execution"], workflow["batch_execution"])
        result = self.run_job(restored)
        self.assertEqual(set(result["results"][-1]["preview"]), {"score"})

    def test_conditional_terminals_use_accepted_cache_and_all_saves_wait(self):
        workflow = self.workflow(conditional=True)
        workflow["nodes"]["save2"] = deepcopy(workflow["nodes"]["save"])
        workflow["nodes"]["save2"]["inputs"]["dir"] = str(self.root / "out2")
        with patch.object(LoopDemoImageNode, "execute", autospec=True, side_effect=LoopDemoImageNode.execute) as generate:
            result = self.run_job(workflow)
        self.assertEqual(generate.call_count, 6)
        self.assertEqual(set(result["results"][-1]["preview"]), {"score", "condition", "save", "save2"})
        for path in self.root.rglob("out*/*/*.png"):
            with Image.open(path) as image:
                self.assertEqual(image.getpixel((0, 0)), (0, 0, 255))
        self.assertEqual(len(list(self.root.rglob("out*/*/*.png"))), 4)

    def test_exhausted_loop_does_not_run_terminal_branches_or_save(self):
        workflow = self.workflow(conditional=True)
        workflow["conditional_loop"]["loops"][0]["conditions"][0]["value"] = 256
        result = self.run_job(workflow)
        self.assertEqual(result["failed"], 2)
        self.assertFalse((self.root / "out").exists())

    def test_score_only_conditional_branch_is_associated_with_folder_run(self):
        workflow = self.workflow(conditional=True)
        workflow["nodes"]["save"]["inputs"]["img1"] = link("folder", "image")
        result = self.run_job(workflow)
        self.assertEqual(result["succeeded"], 2)
        self.assertEqual(result["results"][-1]["preview"]["score"]["score"]["value"], 255)

    def test_every_save_directory_is_validated_and_excluded_from_scan(self):
        workflow = self.workflow()
        workflow["nodes"]["save2"] = deepcopy(workflow["nodes"]["save"])
        workflow["nodes"]["save2"]["inputs"]["dir"] = str(self.input)
        response = self.client.post("/api/batch/plan", json={"workflow": workflow, "node_id": "save"})
        self.assertEqual(response.status_code, 422)
        self.assertFalse((self.root / "out").exists())
        nested = self.input / "generated"
        nested.mkdir()
        Image.new("RGB", (5, 5)).save(nested / "old.png")
        workflow["nodes"]["folder"]["inputs"]["recursive"] = 1
        workflow["nodes"]["save2"]["inputs"]["dir"] = str(nested)
        response = self.client.post("/api/batch/plan", json={"workflow": workflow, "node_id": "save"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["total"], 2)

    def test_invalid_terminal_rejects_plan_before_writes(self):
        workflow = self.workflow()
        workflow["nodes"]["score"]["inputs"]["image"] = link("folder", "filename")
        response = self.client.post("/api/batch/plan", json={"workflow": workflow, "node_id": "save"})
        self.assertEqual(response.status_code, 422, response.text)
        self.assertFalse((self.root / "out").exists())

    def test_disabling_visualization_still_executes_all_terminals(self):
        with patch.object(LoopDemoScoreNode, "execute", autospec=True, side_effect=LoopDemoScoreNode.execute) as score, \
             patch.object(server, "serialize_node_output", side_effect=AssertionError("no serialization")):
            response = self.client.post("/api/batch", json={
                "workflow": self.workflow(), "node_id": "save", "visualize": False})
            self.assertEqual(response.status_code, 200, response.text)
            job_id = response.json()["id"]
            self.manager.threads[job_id].join(5)
            result = self.manager.get(job_id)
        self.assertEqual(result["succeeded"], 2)
        self.assertEqual(score.call_count, 2)
        self.assertEqual(result["results"][-1]["preview"], {})

    def test_legacy_flat_loop_also_runs_multiple_terminal_saves(self):
        workflow = self.workflow(conditional=True)
        settings = workflow["conditional_loop"]["loops"][0]
        workflow["conditional_loop"] = {key: value for key, value in settings.items() if key not in {"id", "end_nodes"}}
        workflow["nodes"]["save2"] = deepcopy(workflow["nodes"]["save"])
        workflow["nodes"]["save2"]["inputs"]["dir"] = str(self.root / "out2")
        result = self.run_job(workflow)
        self.assertEqual(result["succeeded"], 2)
        self.assertEqual(len(list(self.root.rglob("out*/*/*.png"))), 4)

    def test_terminal_computation_failure_precedes_any_save(self):
        with patch.object(LoopDemoScoreNode, "execute", side_effect=ValueError("terminal failed")):
            result = self.run_job(self.workflow())
        self.assertEqual(result["failed"], 2)
        self.assertFalse((self.root / "out").exists())

    def test_multiple_save_failure_removes_only_files_created_for_current_input(self):
        existing = self.root / "out" / "img1" / "existing.png"
        existing.parent.mkdir(parents=True)
        existing.write_bytes(b"preexisting file must survive")
        for conditional in (False, True):
            with self.subTest(conditional=conditional):
                workflow = self.workflow(conditional=conditional)
                workflow["nodes"]["save2"] = deepcopy(workflow["nodes"]["save"])
                result = self.run_job(workflow)
                self.assertEqual(result["failed"], 2)
                self.assertEqual(list((self.root / "out").rglob("*.png")), [existing])
                self.assertEqual(existing.read_bytes(), b"preexisting file must survive")

    def test_condition_cannot_depend_on_an_early_writer(self):
        for flat in (False, True):
            workflow = self.workflow(conditional=True)
            workflow["nodes"].update({
                "gray": {"class_type": "GrayMaskNode", "inputs": {
                    "target": link("folder", "image"), "edit_mask": str(self.input / "a.png")}},
                "sample": {"class_type": "AssembleSampleNode", "inputs": {
                    "target": link("folder", "image"), "source": link("gray", "source"),
                    "edit_mask": str(self.input / "a.png"), "reference": link("folder", "image"), "prompt": ""}},
                "early": {"class_type": "SavePreviewNode", "inputs": {
                    "sample": link("sample", "sample"), "path": str(self.root / "premature.png")}},
                "readback": {"class_type": "LoadTargetNode", "inputs": {"path": link("early", "path")}},
            })
            workflow["nodes"]["demo"]["inputs"]["image"] = link("readback", "target")
            if flat:
                settings = workflow["conditional_loop"]["loops"][0]
                workflow["conditional_loop"] = {key: value for key, value in settings.items() if key not in {"id", "end_nodes"}}
            response = self.client.post("/api/batch/plan", json={"workflow": workflow, "node_id": "save"})
            self.assertEqual(response.status_code, 422, response.text)
            self.assertIn("上游不能包含写盘", response.json()["detail"])
        self.assertFalse((self.root / "premature.png").exists())


if __name__ == "__main__":
    unittest.main()
