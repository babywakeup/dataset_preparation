"""CPU-only conditional task regressions; no model, sleeps, or status polling."""

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from threading import Event
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from object_inpainting.graph import GraphExecutionCancelled, NodeExecutionError, NodeGraph
from object_inpainting.nodes import NODE_CLASS_MAPPINGS, Node
from object_inpainting.nodes.save_images import SaveImagesNode
from object_inpainting_studio import server
from object_inpainting_studio.batch import BatchJobs
from object_inpainting_studio.conditional_loop import (
    ConditionalPlan,
    condition_nodes,
    loop_settings,
    run_conditional,
)


def link(node, output):
    return {"node": node, "output": output}


class ConditionalLoopTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "output"
        self.input = self.root / "input"
        self.input.mkdir()
        self.calls = {"fixed": [], "generate": [], "score": [], "save": []}
        self.values = lambda seed: (float(seed >= 8), 0.0, seed >= 8)
        self.mutate_input = False
        self.hook = lambda: None
        self.fixed_hook = lambda: None
        self.clock = 0.0
        owner = self

        class FixedNode(Node):
            INPUT_TYPES = {"image": "RGB_IMAGE"}
            OUTPUT_TYPES = {"image": "RGB_IMAGE"}

            def execute(self, image):
                owner.calls["fixed"].append(image.getpixel((0, 0)))
                owner.fixed_hook()
                return {"image": image.copy()}

        class GenerateNode(Node):
            INPUT_TYPES = {"image": "RGB_IMAGE", "seed": "INT"}
            OUTPUT_TYPES = {"image": "RGB_IMAGE", "a": "FLOAT", "b": "FLOAT", "flag": "BOOL", "seed": "INT"}

            def execute(self, image, seed):
                owner.calls["generate"].append(seed)
                owner.hook()
                a, b, flag = owner.values(seed)
                pixel = image.getpixel((0, 0))
                if owner.mutate_input:
                    image.putpixel((0, 0), (250, 250, 250))
                return {"image": Image.new("RGB", (3, 2), (pixel[0], seed % 256, pixel[2])),
                        "a": a, "b": b, "flag": flag, "seed": seed}

        class ScoreNode(Node):
            INPUT_TYPES = {"value": "FLOAT"}
            OUTPUT_TYPES = {"score": "FLOAT"}

            def execute(self, value):
                owner.calls["score"].append(value)
                return {"score": value}

        class ObservedSaveNode(SaveImagesNode):
            def execute_with_spec(self, inputs, spec):
                owner.calls["save"].append(inputs["img1"].getpixel((0, 0)))
                return super().execute_with_spec(inputs, spec)

        class SideEffectNode(FixedNode):
            SIDE_EFFECTS = True

        self.registry = {"TestFixedNode": FixedNode, "TestGenerateNode": GenerateNode,
                         "TestScoreNode": ScoreNode, "TestSideEffectNode": SideEffectNode,
                         "SaveImagesNode": ObservedSaveNode}
        registry_patch = patch.dict(NODE_CLASS_MAPPINGS, self.registry)
        registry_patch.start()
        self.addCleanup(registry_patch.stop)
        self.manager = BatchJobs()
        manager_patch = patch.object(server, "BATCH_JOBS", self.manager)
        manager_patch.start()
        self.addCleanup(manager_patch.stop)
        self.client = TestClient(server.app)
        self.addCleanup(self.client.close)
        # Registered last: workers must stop before mocks and temporary paths go away.
        self.addCleanup(self.finish_workers)

    def finish_workers(self):
        for job_id, thread in tuple(self.manager.threads.items()):
            self.manager.cancel(job_id)
            thread.join(10)
            self.assertFalse(thread.is_alive(), "conditional worker did not terminate")

    def workflow(self, folder=False):
        nodes = {
            "fixed": {"class_type": "TestFixedNode", "inputs": {"image": {"context": "image"}}},
            "generate": {"class_type": "TestGenerateNode", "inputs": {"image": link("fixed", "image"), "seed": 7}},
            "score_a": {"class_type": "TestScoreNode", "inputs": {"value": link("generate", "a")}},
            "save": {"class_type": "SaveImagesNode", "save_items": [{"id": "img1", "name": "accepted"}],
                     "inputs": {"img1": link("generate", "image"), "dir": str(self.output), "filename": "accepted"}},
        }
        if folder:
            nodes["folder"] = {"class_type": "FolderImagesNode", "inputs": {"dir": str(self.input)}}
            nodes["fixed"]["inputs"]["image"] = link("folder", "image")
            nodes["save"]["inputs"]["filename"] = link("folder", "filename")
        return {"nodes": nodes, "outputs": {}, "conditional_loop": {
            "start_nodes": ["generate"], "seed_nodes": ["generate"], "max_iterations": 3,
            "conditions": [{"node": "score_a", "output": "score", "operator": ">=", "value": 1.0}],
        }}

    def plan(self, workflow=None):
        workflow = self.workflow() if workflow is None else workflow
        return server.prepare_conditional({"workflow": workflow, "node_id": "save"})[1]

    def run_loop(self, workflow=None, **kwargs):
        return run_conditional(self.plan(workflow), {"image": Image.new("RGB", (3, 2), (20, 0, 40))},
                               Event(), **kwargs)

    def api_payload(self, workflow=None):
        workflow = deepcopy(self.workflow() if workflow is None else workflow)
        if "folder" not in workflow["nodes"]:
            path = self.input / "single.png"
            Image.new("RGB", (3, 2), (20, 0, 40)).save(path)
            workflow["nodes"]["fixed"]["inputs"]["image"] = str(path)
        return {"workflow": workflow, "node_id": "save"}

    def join_job(self, job):
        thread = self.manager.threads[job["id"]]
        thread.join(10)
        self.assertFalse(thread.is_alive(), "job did not terminate")
        response = self.client.get(f"/api/batch/{job['id']}")
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def post_job(self, endpoint, payload):
        response = self.client.post(endpoint, json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        return self.join_job(response.json())

    def assert_no_writes(self):
        self.assertEqual(self.calls["save"], [])
        self.assertFalse(self.output.exists())

    def assert_saved(self, result, pixel):
        self.assertEqual(set(result["paths"]), {"img1"})
        path = Path(result["paths"]["img1"])
        self.assertTrue(path.is_file())
        with Image.open(path) as image:
            self.assertEqual(image.getpixel((0, 0)), pixel)
            self.assertEqual(image.size, (3, 2))

    def test_retry_reuses_fixed_once_and_saves_exact_accepted_image_score_seed(self):
        workflow = self.workflow()
        original = deepcopy(workflow)
        updates = []
        result = self.run_loop(workflow, progress=lambda update: updates.append(deepcopy(update)))
        self.assertTrue(result["accepted"])
        self.assertEqual((result["iterations"], result["stop_reason"]), (2, "accepted"))
        self.assertEqual(self.calls["fixed"], [(20, 0, 40)])
        self.assertEqual(self.calls["generate"], [7, 8])
        self.assertEqual(self.calls["score"], [0.0, 1.0])
        self.assertEqual(self.calls["save"], [(20, 8, 40)])
        self.assert_saved(result, (20, 8, 40))
        self.assertEqual([entry["seeds"] for entry in result["history"]], [{"generate": 7}, {"generate": 8}])
        self.assertEqual([entry["scores"][0]["value"] for entry in result["history"]], [0.0, 1.0])
        self.assertEqual([entry["accepted"] for entry in result["history"]], [False, True])
        self.assertEqual(updates[-1], {"stop_reason": "accepted"})
        self.assertEqual(workflow, original)
        self.assertEqual(len(list(self.output.rglob("*.*"))), 1)

    def test_and_requires_same_iteration_not_mixed_successes(self):
        self.values = lambda seed: (float(seed == 7), float(seed == 8), False)
        workflow = self.workflow()
        workflow["conditional_loop"].update(max_iterations=2, on_exhausted="skip")
        workflow["nodes"]["score_b"] = {"class_type": "TestScoreNode", "inputs": {"value": link("generate", "b")}}
        workflow["conditional_loop"]["conditions"].append(
            {"node": "score_b", "output": "score", "operator": ">=", "value": 1})
        result = self.run_loop(workflow)
        self.assertEqual((result["accepted"], result["skipped"], result["iterations"]), (False, True, 2))
        self.assertEqual([[score["passed"] for score in entry["scores"]] for entry in result["history"]],
                         [[True, False], [False, True]])
        self.assertEqual(self.calls["score"], [1.0, 0.0, 0.0, 1.0])
        self.assert_no_writes()

    def test_fixed_cache_is_immutable_when_retry_body_mutates_pil_input(self):
        self.mutate_input = True
        result = self.run_loop()
        self.assertEqual(self.calls["fixed"], [(20, 0, 40)])
        self.assertEqual(self.calls["generate"], [7, 8])
        # A failed attempt must not poison the next attempt's fixed image.
        self.assert_saved(result, (20, 8, 40))

    def test_numeric_comparison_operators_accept_matching_current_scores(self):
        self.values = lambda seed: (1.0, 0.0, True)
        for index, (operator, value) in enumerate((("<=", 1), (">=", 1), ("<", 2),
                                                    (">", 0), ("==", 1), ("!=", 0))):
            with self.subTest(operator=operator):
                workflow = self.workflow()
                workflow["nodes"]["save"]["inputs"]["dir"] = str(self.root / f"operator-{index}")
                workflow["conditional_loop"]["max_iterations"] = 1
                workflow["conditional_loop"]["conditions"][0].update(operator=operator, value=value)
                result = self.run_loop(workflow)
                self.assertEqual((result["accepted"], result["iterations"]), (True, 1))
                self.assert_saved(result, (20, 7, 40))

    def test_or_evaluates_every_condition_including_off_save_ancestor_branch(self):
        self.values = lambda seed: (1.0, 0.0, True)
        workflow = self.workflow()
        workflow["conditional_loop"]["mode"] = "any"
        workflow["nodes"]["score_b"] = {"class_type": "TestScoreNode", "inputs": {"value": link("generate", "b")}}
        workflow["conditional_loop"]["conditions"].append(
            {"node": "score_b", "output": "score", "operator": "==", "value": 1})
        plan = self.plan(workflow)
        self.assertEqual(set(plan.describe()["body_nodes"]), {"generate", "score_a", "score_b"})
        self.assertEqual(plan.describe()["fixed_nodes"], ["fixed"])
        result = self.run_loop(workflow)
        self.assertEqual(self.calls["score"], [1.0, 0.0])
        self.assertEqual([s["passed"] for s in result["history"][0]["scores"]], [True, False])
        self.assert_saved(result, (20, 7, 40))

    def test_max_exhaust_fail_and_skip_never_write(self):
        self.values = lambda seed: (0.0, 0.0, False)
        for policy in ("fail", "skip"):
            with self.subTest(policy=policy):
                workflow = self.workflow()
                workflow["conditional_loop"]["on_exhausted"] = policy
                result = self.run_loop(workflow)
                self.assertEqual((result["accepted"], result["iterations"], result["stop_reason"]),
                                 (False, 3, "max_iterations"))
                self.assertEqual("error" in result, policy == "fail")
                self.assertEqual(result.get("skipped", False), policy == "skip")
                self.assert_no_writes()

    def test_boolean_condition_and_seed_increments(self):
        workflow = self.workflow()
        del workflow["nodes"]["score_a"]
        workflow["conditional_loop"]["conditions"] = [
            {"node": "generate", "output": "flag", "operator": "==", "value": True}]
        result = self.run_loop(workflow)
        self.assertEqual(self.calls["generate"], [7, 8])
        self.assertIs(result["history"][0]["scores"][0]["value"], False)
        self.assertIs(result["history"][1]["scores"][0]["value"], True)
        self.assert_saved(result, (20, 8, 40))

    def test_runtime_invalid_boolean_numeric_and_nonfinite_scores_do_not_retry(self):
        for output, value, expected in (("a", True, 1), ("a", "1", 1), ("a", None, 1),
                                        ("a", float("nan"), 1), ("a", float("inf"), 1),
                                        ("flag", 1, True), ("flag", "true", True)):
            with self.subTest(output=output, value=value):
                self.calls["generate"].clear()
                self.values = lambda seed, value=value, output=output: (
                    value if output == "a" else 0.0, 0.0, value if output == "flag" else False)
                workflow = self.workflow()
                del workflow["nodes"]["score_a"]
                workflow["conditional_loop"]["conditions"] = [
                    {"node": "generate", "output": output, "operator": "==", "value": expected}]
                with self.assertRaisesRegex(ValueError, "类型|有限"):
                    self.run_loop(workflow)
                self.assertEqual(self.calls["generate"], [7])
                self.assert_no_writes()

    def test_invalid_settings_and_type_limits_rejected_without_execution(self):
        cases = [("max_iterations", value) for value in (True, 0, -1, 1001, 1.5, "2")]
        cases += [("timeout_seconds", value) for value in (True, -1, 86401, float("nan"), float("inf"), "1")]
        cases += [("mode", "AND"), ("on_exhausted", "save"), ("start_nodes", []),
                  ("start_nodes", ["missing"]), ("start_nodes", ["save"]),
                  ("start_nodes", ["generate", "generate"]), ("start_nodes", "generate"),
                  ("seed_nodes", ["generate", "generate"]), ("seed_nodes", [3]),
                  ("conditions", []), ("conditions", [{}]), ("conditions", [None]),
                  ("conditions", [{"node": "score_a"}] * 65)]
        for key, value in cases:
            with self.subTest(key=key, value=value):
                workflow = self.workflow()
                workflow["conditional_loop"][key] = value
                with self.assertRaises((ValueError, KeyError, TypeError)):
                    self.plan(workflow)
        self.assertEqual(self.calls["generate"], [])
        self.assert_no_writes()

    def test_invalid_condition_ports_types_operators_and_thresholds(self):
        cases = [("fixed", "image", "==", 1), ("generate", "image", "==", 1),
                 ("generate", "missing", "==", 1), ("generate", "a", "~=", 1),
                 ("generate", "a", "==", True), ("generate", "a", "==", float("nan")),
                 ("generate", "a", "==", float("inf")), ("generate", "a", "==", "1"),
                 ("generate", "flag", ">=", True), ("generate", "flag", "==", 1)]
        for node, output, operator, value in cases:
            with self.subTest(node=node, output=output, operator=operator, value=value):
                workflow = self.workflow()
                del workflow["nodes"]["score_a"]
                workflow["conditional_loop"]["conditions"] = [
                    {"node": node, "output": output, "operator": operator, "value": value}]
                with self.assertRaises(ValueError):
                    self.plan(workflow)
        self.assert_no_writes()

    def test_invalid_seeds_and_seed_nodes_and_valid_upper_limit(self):
        for seed in (True, -1, 1.5, "7", 2**63 - 2, link("generate", "seed"), {"context": "seed"}):
            with self.subTest(seed=seed):
                workflow = self.workflow()
                workflow["nodes"]["generate"]["inputs"]["seed"] = seed
                with self.assertRaises(ValueError):
                    self.plan(workflow)
        for node in ("fixed", "score_a", "missing", "save"):
            with self.subTest(seed_node=node):
                workflow = self.workflow()
                workflow["conditional_loop"]["seed_nodes"] = [node]
                with self.assertRaises(ValueError):
                    self.plan(workflow)
        workflow = self.workflow()
        workflow["nodes"]["generate"]["inputs"]["seed"] = 2**63 - 3
        self.assertEqual(self.plan(workflow).seeds, {"generate": 2**63 - 3})
        self.assert_no_writes()

    def test_boundary_escapes_other_side_effects_and_folder_body_rejected(self):
        workflow = self.workflow()
        workflow["nodes"]["escape"] = {"class_type": "TestFixedNode", "inputs": {"image": link("generate", "image")}}
        with self.assertRaisesRegex(ValueError, "逃逸"):
            self.plan(workflow)
        workflow = self.workflow()
        workflow["outputs"]["leak"] = link("generate", "image")
        with self.assertRaisesRegex(ValueError, "直接暴露"):
            self.plan(workflow)
        workflow = self.workflow()
        workflow["nodes"]["fixed"]["class_type"] = "TestSideEffectNode"
        with self.assertRaisesRegex(ValueError, "写盘"):
            self.plan(workflow)
        workflow = self.workflow(folder=True)
        workflow["conditional_loop"]["start_nodes"] = ["folder"]
        with self.assertRaisesRegex(ValueError, "文件夹输入"):
            self.plan(workflow)
        workflow = self.workflow()
        workflow["nodes"]["save"]["inputs"]["img1"] = link("fixed", "image")
        with self.assertRaisesRegex(ValueError, "必须连接"):
            self.plan(workflow)
        graph = NodeGraph(self.workflow())
        with self.assertRaisesRegex(ValueError, "SaveImagesNode"):
            ConditionalPlan(graph, "generate", self.workflow()["conditional_loop"], self.workflow())
        self.assert_no_writes()

    def test_settings_wrapper_null_and_ordinary_graph_bypass_guard(self):
        workflow = self.workflow()
        self.assertEqual(loop_settings({"workflow": workflow}), workflow["conditional_loop"])
        self.assertIsNone(loop_settings({"conditional_loop": None}))
        with self.assertRaises(ValueError):
            loop_settings({"conditional_loop": []})
        with self.assertRaises(ValueError):
            condition_nodes({"conditions": "bad"})
        with self.assertRaisesRegex(ValueError, "cannot bypass"):
            NodeGraph(workflow).start({})
        self.assert_no_writes()

    def test_cancel_during_long_generation_waits_cooperatively_and_never_saves(self):
        entered, release = Event(), Event()

        def wait():
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test release was not signalled")

        self.hook = wait
        response = self.client.post("/api/conditional", json=self.api_payload())
        self.assertEqual(response.status_code, 200, response.text)
        job = response.json()
        try:
            self.assertTrue(entered.wait(5))
            cancelled = self.client.post(f"/api/batch/{job['id']}/cancel")
            self.assertEqual(cancelled.json()["status"], "cancelling")
            self.assertTrue(self.manager.threads[job["id"]].is_alive())
            self.assertEqual(self.manager.get(job["id"])["processed"], 0)
        finally:
            release.set()
        final = self.join_job(job)
        self.assertEqual((final["status"], final["stop_reason"], final["processed"]), ("cancelled", "cancelled", 0))
        self.assertEqual(final["results"], [])
        self.assertEqual(self.calls["generate"], [7])
        self.assert_no_writes()

    def test_pre_cancelled_execution_never_enters_nodes(self):
        cancel = Event()
        cancel.set()
        with self.assertRaises(GraphExecutionCancelled):
            run_conditional(self.plan(), {}, cancel)
        self.assertEqual(self.calls["fixed"], [])
        self.assertEqual(self.calls["generate"], [])
        self.assert_no_writes()

    def test_generation_exception_is_not_retried_and_has_node_error_in_job(self):
        def explode():
            raise RuntimeError("synthetic generation failure")

        self.hook = explode
        with self.assertLogs("object_inpainting", level="ERROR"):
            with self.assertRaises(NodeExecutionError) as caught:
                self.run_loop()
        self.assertEqual(caught.exception.as_dict()["node_id"], "generate")
        self.calls["generate"].clear()
        with self.assertLogs(level="ERROR"):
            final = self.post_job("/api/conditional", self.api_payload())
        self.assertEqual((final["status"], final["failed"], final["processed"]), ("failed", 1, 1))
        self.assertEqual(final["stop_reason"], "error")
        self.assertEqual(final["node_error"], {"node_id": "generate", "class_type": "TestGenerateNode",
                                             "error_type": "RuntimeError", "message": "synthetic generation failure"})
        self.assertEqual(self.calls["generate"], [7])
        self.assert_no_writes()

    def test_timeout_during_generation_uses_mock_clock_no_retry_or_save(self):
        workflow = self.workflow()
        workflow["conditional_loop"].update(timeout_seconds=5, on_exhausted="skip")
        self.hook = lambda: setattr(self, "clock", 6.0)
        with patch("object_inpainting_studio.conditional_loop.monotonic", side_effect=lambda: self.clock):
            result = self.run_loop(workflow)
        self.assertEqual((result["stop_reason"], result["iterations"], result["skipped"]), ("timeout", 1, True))
        self.assertEqual(result["history"], [])
        self.assertEqual(self.calls["generate"], [7])
        self.assert_no_writes()

    def test_timeout_after_acceptance_before_save_dependencies_finish_blocks_write(self):
        workflow = self.workflow()
        workflow["conditional_loop"]["timeout_seconds"] = 5
        workflow["nodes"]["late"] = {"class_type": "TestFixedNode", "inputs": {"image": {"context": "late_image"}}}
        workflow["nodes"]["save"]["save_items"].append({"id": "img2", "name": "late"})
        workflow["nodes"]["save"]["inputs"]["img2"] = link("late", "image")
        self.values = lambda seed: (1.0, 0.0, True)

        def advance_late():
            if len(self.calls["fixed"]) == 2:
                self.clock = 6.0

        self.fixed_hook = advance_late
        context = {"image": Image.new("RGB", (3, 2), (20, 0, 40)),
                   "late_image": Image.new("RGB", (3, 2), "blue")}
        with patch("object_inpainting_studio.conditional_loop.monotonic", side_effect=lambda: self.clock):
            result = run_conditional(self.plan(workflow), context, Event())
        self.assertEqual(result["stop_reason"], "timeout")
        self.assertIn("error", result)
        self.assertTrue(result["history"][0]["accepted"])
        self.assertEqual(len(self.calls["fixed"]), 2)
        self.assert_no_writes()

    def test_plan_api_includes_score_branch_but_never_infers_writes_or_starts_job(self):
        payload = self.api_payload()
        original = deepcopy(payload)
        with patch.object(self.registry["TestGenerateNode"], "execute", side_effect=AssertionError("inference forbidden")), \
             patch.object(self.registry["SaveImagesNode"], "execute_with_spec", side_effect=AssertionError("write forbidden")):
            response = self.client.post("/api/conditional/plan", json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["body_nodes"], ["generate", "score_a"])
        self.assertEqual(response.json()["fixed_nodes"], ["fixed"])
        self.assertEqual(payload, original)
        self.assertIsNone(self.manager.latest())
        self.assertEqual(self.calls["fixed"], [])
        self.assert_no_writes()

    def test_single_task_api_start_join_status_latest_and_save_bypass_rejection(self):
        payload = self.api_payload()
        for workflow in (payload["workflow"], {"workflow": payload["workflow"]}):
            response = self.client.post("/api/node/save-images", json={"workflow": workflow, "node_id": "save"})
            self.assertEqual(response.status_code, 422, response.text)
        self.assert_no_writes()
        final = self.post_job("/api/conditional", payload)
        self.assertEqual((final["status"], final["total"], final["processed"], final["succeeded"], final["failed"]),
                         ("completed", 1, 1, 1, 0))
        self.assertEqual((final["current_iteration"], final["stop_reason"], final["current_node"]), (2, "accepted", None))
        self.assert_saved(final["results"][0], (20, 8, 40))
        self.assertEqual(self.client.get("/api/batch/latest").json()["job"], final)
        self.assertNotIn("data:image", json.dumps(final))
        snapshot = self.manager.get(final["id"])
        snapshot["iteration_history"].clear()
        self.assertEqual(len(self.manager.get(final["id"])["iteration_history"]), 2)

    def test_invalid_plan_api_rejects_missing_settings_bad_target_and_blank_dir(self):
        for case in ("settings", "target", "dir", "conditions"):
            with self.subTest(case=case):
                payload = self.api_payload()
                if case == "settings":
                    del payload["workflow"]["conditional_loop"]
                elif case == "target":
                    payload["node_id"] = "generate"
                elif case == "dir":
                    payload["workflow"]["nodes"]["save"]["inputs"]["dir"] = " "
                else:
                    payload["workflow"]["conditional_loop"]["conditions"] = []
                for endpoint in ("/api/conditional/plan", "/api/conditional"):
                    response = self.client.post(endpoint, json=payload)
                    self.assertEqual(response.status_code, 422, response.text)
        self.assertIsNone(self.manager.latest())
        self.assert_no_writes()

    def test_folder_batch_two_images_have_independent_fixed_caches_and_seeds(self):
        for name, pixel in (("a.png", (20, 0, 40)), ("b.png", (60, 0, 80))):
            Image.new("RGB", (3, 2), pixel).save(self.input / name)
        payload = self.api_payload(self.workflow(folder=True))
        plan = self.client.post("/api/batch/plan", json=payload)
        self.assertEqual(plan.status_code, 200, plan.text)
        self.assertEqual(plan.json()["total"], 2)
        self.assertEqual(self.calls["generate"], [])
        final = self.post_job("/api/batch", payload)
        self.assertEqual((final["status"], final["succeeded"], final["failed"]), ("completed", 2, 0))
        self.assertEqual(self.calls["fixed"], [(20, 0, 40), (60, 0, 80)])
        self.assertEqual(self.calls["generate"], [7, 8, 7, 8])
        for result, name, pixel in zip(final["results"], ("a.png", "b.png"), ((20, 8, 40), (60, 8, 80))):
            self.assertEqual(Path(result["source"]).name, name)
            self.assertEqual([e["seeds"] for e in result["history"]], [{"generate": 7}, {"generate": 8}])
            self.assert_saved(result, pixel)
            self.assertEqual(Path(result["paths"]["img1"]).name, name)
        self.assertEqual(len(list(self.output.rglob("*.png"))), 2)

    def test_folder_exhaustion_job_counts_fail_stop_and_skip_continue_without_files(self):
        self.values = lambda seed: (0.0, 0.0, False)
        for name in ("a.png", "b.png"):
            Image.new("RGB", (3, 2), (20, 0, 40)).save(self.input / name)
        for policy, status, processed, failed, skipped in (("fail", "failed", 1, 1, 0),
                                                          ("skip", "completed", 2, 0, 2)):
            with self.subTest(policy=policy):
                workflow = self.workflow(folder=True)
                workflow["conditional_loop"].update(max_iterations=2, on_exhausted=policy)
                payload = self.api_payload(workflow)
                payload["continue_on_error"] = False
                final = self.post_job("/api/batch", payload)
                self.assertEqual((final["status"], final["processed"], final["failed"], final["skipped"]),
                                 (status, processed, failed, skipped))
                self.assertEqual(final["succeeded"], 0)
                self.assertTrue(all(result["stop_reason"] == "max_iterations" for result in final["results"]))
                self.assertTrue(all(result["iterations"] == 2 for result in final["results"]))
                self.assert_no_writes()

    def test_folder_condition_plan_uses_decoded_literal_image_inputs(self):
        Image.new("RGB", (3, 2), (20, 0, 40)).save(self.input / "a.png")
        literal = self.root / "reference.png"
        Image.new("RGB", (3, 2), (50, 0, 60)).save(literal)
        for endpoint in ("/api/batch", "/api/batch/preview"):
            with self.subTest(endpoint=endpoint):
                workflow = self.workflow(folder=True)
                workflow["nodes"]["literal"] = {"class_type": "TestFixedNode", "inputs": {"image": str(literal)}}
                workflow["nodes"]["save"]["save_items"].append({"id": "img2", "name": "reference"})
                workflow["nodes"]["save"]["inputs"]["img2"] = link("literal", "image")
                workflow["nodes"]["save"]["inputs"]["dir"] = str(self.root / endpoint.rsplit("/", 1)[-1])
                final = self.post_job(endpoint, self.api_payload(workflow))
                self.assertEqual((final["status"], final["succeeded"]), ("completed", 1))
                with Image.open(final["results"][0]["paths"]["img2"]) as image:
                    self.assertEqual(image.getpixel((0, 0)), (50, 0, 60))

    def test_filtered_current_trial_is_async_and_writes_only_selected_image(self):
        self.output = self.input / "a-generated"
        old = self.output / "accepted" / "old.png"
        old.parent.mkdir(parents=True)
        Image.new("RGB", (3, 2), "green").save(old)
        old_bytes = old.read_bytes()
        for name, color in (("b.png", (20, 0, 40)), ("c.png", (60, 0, 80)), ("d.png", (90, 0, 100))):
            Image.new("RGB", (3, 2), color).save(self.input / name)
        workflow = self.workflow(folder=True)
        workflow["nodes"]["folder"]["inputs"].update(recursive=1, index=1)
        payload = self.api_payload(workflow)
        plan = self.client.post("/api/batch/plan", json=payload).json()
        self.assertEqual((plan["total"], Path(plan["current_path"]).name), (2, "c.png"))
        entered, release = Event(), Event()

        def wait():
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test release was not signalled")

        self.hook = wait
        response = self.client.post("/api/batch/preview", json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        job = response.json()
        try:
            self.assertEqual((job["status"], job["total"]), ("running", 1))
            self.assertTrue(entered.wait(5))
            self.assertEqual(self.manager.get(job["id"])["processed"], 0)
        finally:
            release.set()
        final = self.join_job(job)
        self.assertEqual((final["status"], final["processed"], final["succeeded"]), ("completed", 1, 1))
        self.assertEqual(final["results"][0]["source"], plan["current_path"])
        self.assert_saved(final["results"][0], (60, 8, 80))
        self.assertEqual(self.calls["generate"], [7, 8])
        self.assertEqual(old.read_bytes(), old_bytes)
        self.assertEqual({p.relative_to(self.output).as_posix() for p in self.output.rglob("*") if p.is_file()},
                         {"accepted/old.png", "accepted/c.png"})

    def test_shared_manager_rejects_overlapping_normal_and_conditional_jobs_both_directions(self):
        Image.new("RGB", (3, 2), (20, 0, 40)).save(self.input / "a.png")
        for first in ("normal", "conditional"):
            with self.subTest(first=first):
                entered, release = Event(), Event()

                def wait():
                    entered.set()
                    if not release.wait(5):
                        raise RuntimeError("test release was not signalled")

                self.hook = wait
                normal = self.workflow(folder=True)
                del normal["conditional_loop"]
                conditional = self.workflow(folder=True)
                payload = self.api_payload(normal if first == "normal" else conditional)
                response = self.client.post("/api/batch", json=payload)
                self.assertEqual(response.status_code, 200, response.text)
                job = response.json()
                try:
                    self.assertTrue(entered.wait(5))
                    for endpoint, workflow in (("/api/batch", normal), ("/api/batch", conditional),
                                               ("/api/conditional", conditional), ("/api/batch/preview", conditional)):
                        rejected = self.client.post(endpoint, json=self.api_payload(workflow))
                        self.assertEqual(rejected.status_code, 422, rejected.text)
                        self.assertIn("已有批量任务", rejected.json()["detail"])
                    self.assertEqual(len(self.manager.jobs), 1 if first == "normal" else 2)
                    self.client.post(f"/api/batch/{job['id']}/cancel")
                finally:
                    release.set()
                self.assertEqual(self.join_job(job)["status"], "cancelled")
        self.assert_no_writes()


if __name__ == "__main__":
    unittest.main()
