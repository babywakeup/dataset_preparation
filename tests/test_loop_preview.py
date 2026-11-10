"""Computation-only loop previews; CPU fixtures, no models or polling."""

import base64
import io
import json
import unittest
from copy import deepcopy
from threading import Event
from unittest.mock import patch

from PIL import Image

import test_conditional_loop as legacy
import test_multiple_conditional_loops as multiple
from object_inpainting_studio import server
from object_inpainting_studio.conditional_loop import ConditionalPlan, run_conditional


link = legacy.link


class LoopPreviewTests(unittest.TestCase):
    setUp = legacy.ConditionalLoopTests.setUp
    finish_workers = legacy.ConditionalLoopTests.finish_workers
    api_payload = legacy.ConditionalLoopTests.api_payload
    join_job = legacy.ConditionalLoopTests.join_job
    post_job = legacy.ConditionalLoopTests.post_job
    assert_no_writes = legacy.ConditionalLoopTests.assert_no_writes
    workflow = multiple.MultipleConditionalLoopTests.workflow
    loops = multiple.MultipleConditionalLoopTests.loops

    def preview_payload(self, workflow=None, loop_id="loop1"):
        payload = self.api_payload(self.workflow() if workflow is None else workflow)
        del payload["node_id"]
        payload["loop_id"] = loop_id
        return payload

    def preview(self, workflow=None, loop_id="loop1"):
        return self.post_job("/api/conditional/preview", self.preview_payload(workflow, loop_id))

    def assert_pixel(self, serialized, pixel):
        self.assertEqual(serialized["kind"], "image")
        data = base64.b64decode(serialized["image"].split(",", 1)[1])
        with Image.open(io.BytesIO(data)) as image:
            self.assertEqual(image.getpixel((0, 0)), pixel)

    def test_independent_loop_runs_alone_and_returns_json_last_outputs(self):
        job = self.preview()
        self.assertEqual((job["status"], job["processed"], job["succeeded"]), ("completed", 1, 1))
        self.assertIsNone(job["target_id"])
        self.assertEqual(job["preview_loop_id"], "loop1")
        self.assertEqual(job["required_loop_ids"], ["loop1"])
        result = job["results"][-1]
        self.assertEqual(result["preview_loop_id"], "loop1")
        self.assertEqual(result["required_loop_ids"], ["loop1"])
        self.assertEqual(set(result["preview"]), {"generate", "score_a"})
        self.assert_pixel(result["preview"]["generate"]["image"], (20, 8, 40))
        self.assertEqual(result["preview"]["score_a"]["score"], {"kind": "text", "value": 1.0})
        self.assertEqual(self.calls["generate"], [7, 8])
        self.assertEqual(result["paths"], {})
        self.assertEqual([entry["loop_id"] for entry in result["history"]], ["loop1", "loop1"])
        json.dumps(self.manager.get(job["id"]))
        self.assert_no_writes()

    def test_second_independent_loop_does_not_run_first(self):
        job = self.preview(loop_id="loop2")
        self.assertEqual(job["required_loop_ids"], ["loop2"])
        self.assertEqual(self.calls["generate"], [17, 18])
        self.assertEqual(set(job["results"][-1]["preview"]), {"generate2", "score2"})
        self.assert_no_writes()

    def test_incomplete_or_broken_unrelated_loop_does_not_block_preview(self):
        for variant in ("incomplete", "missing", "invalid"):
            with self.subTest(variant=variant):
                workflow = self.workflow()
                other = self.loops(workflow)[1]
                if variant == "incomplete":
                    other.update(start_nodes=[], end_nodes=[], conditions=[])
                elif variant == "missing":
                    other.update(start_nodes=["missing"], end_nodes=["missing"],
                                 conditions=[{"node": "missing", "output": "score", "operator": ">=", "value": 0}])
                else:
                    workflow["nodes"]["generate2"]["class_type"] = "MissingNodeClass"
                job = self.preview(workflow)
                self.assertEqual(job["status"], "completed")
                self.assertEqual(job["required_loop_ids"], ["loop1"])
                self.assertEqual(self.calls["generate"], [7, 8])
                self.assert_no_writes()
                for values in self.calls.values():
                    values.clear()

    def test_scalar_end_still_previews_computed_body_image(self):
        workflow = self.workflow()
        self.loops(workflow)[0]["end_nodes"] = ["score_a"]
        job = self.preview(workflow)
        self.assertEqual(job["status"], "completed")
        self.assert_pixel(job["results"][-1]["preview"]["generate"]["image"], (20, 8, 40))
        self.assertEqual(self.calls["generate"], [7, 8])
        self.assert_no_writes()

    def test_serial_dependencies_accept_first_and_reuse_bridge_cache(self):
        workflow = self.workflow(serial=True, bridge=True, reverse=True)
        self.mutate_input = True
        job = self.preview(workflow, "loop2")
        self.assertEqual(job["required_loop_ids"], ["loop1", "loop2"])
        self.assertEqual([item["iterations"] for item in job["results"][-1]["loops"]], [2, 2])
        self.assertEqual(self.calls["generate"], [7, 8, 17, 18])
        self.assertEqual(self.calls["fixed"], [(20, 0, 40), (20, 8, 40)])
        self.assert_pixel(job["results"][-1]["preview"]["generate2"]["image"], (20, 18, 40))
        self.assert_no_writes()

    def test_upstream_sibling_end_and_condition_are_not_skipped(self):
        workflow = self.workflow(serial=True)
        nodes = workflow["nodes"]
        nodes["end"] = {"class_type": "TestFixedNode", "inputs": {"image": link("generate", "image")}}
        self.loops(workflow)[0]["end_nodes"].append("end")
        self.loops(workflow)[0]["conditions"].append(
            {"node": "generate", "output": "flag", "operator": "==", "value": True})
        job = self.preview(workflow, "loop2")
        self.assertEqual(job["status"], "completed")
        self.assertEqual(self.calls["fixed"], [(20, 0, 40), (20, 8, 40)])
        self.assertEqual(self.calls["generate"], [7, 8, 17, 18])
        self.assert_no_writes()

    def test_failed_upstream_blocks_selected_loop_without_unaccepted_preview(self):
        workflow = self.workflow(serial=True)
        self.loops(workflow)[0]["max_iterations"] = 1
        job = self.preview(workflow, "loop2")
        self.assertEqual(job["status"], "failed")
        self.assertEqual(self.calls["generate"], [7])
        result = job["results"][-1]
        self.assertFalse(result["accepted"])
        self.assertEqual(result["preview"], {})
        self.assertEqual([item["id"] for item in result["loops"]], ["loop1"])
        self.assert_no_writes()

    def test_preview_upstream_does_not_execute_downstream_loop_or_side_effect(self):
        workflow = self.workflow(serial=True)
        workflow["nodes"]["downstream_write"] = {
            "class_type": "TestSideEffectNode", "inputs": {"image": link("generate2", "image")}}
        job = self.preview(workflow)
        self.assertEqual(job["required_loop_ids"], ["loop1"])
        self.assertEqual(self.calls["generate"], [7, 8])
        self.assertEqual(self.calls["fixed"], [(20, 0, 40)])
        self.assert_no_writes()

    def test_computation_graph_without_save_or_output_directory(self):
        workflow = self.workflow()
        del workflow["nodes"]["save"]
        workflow["outputs"] = {"image": link("generate", "image")}
        payload = self.preview_payload(workflow)
        original = deepcopy(payload)
        graph, plan = server.prepare_conditional_preview(payload)
        self.assertIs(graph, plan.graph)
        self.assertNotIn("generate2", graph.nodes)
        self.assertFalse(any(getattr(graph.registry[s["class_type"]], "SIDE_EFFECTS", False)
                             for s in graph.nodes.values()))
        job = self.post_job("/api/conditional/preview", payload)
        self.assertEqual(job["status"], "completed")
        self.assertEqual(payload, original)
        self.assert_no_writes()

    def test_empty_save_directory_is_irrelevant_and_original_config_unchanged(self):
        workflow = self.workflow(serial=True, reverse=True)
        workflow["nodes"]["save"]["inputs"]["dir"] = ""
        payload = self.preview_payload(workflow, "loop2")
        payload["config"] = {"seed": 1}
        fixed_class = self.registry["TestFixedNode"]

        class ConfiguredFixedNode(fixed_class):
            INPUT_TYPES = {"image": "RGB_IMAGE", "config": "CONFIG"}

            def execute(self, image, config):
                if config.seed != 1:
                    raise AssertionError("config was not decoded")
                return super().execute(image)

        payload["workflow"]["nodes"]["fixed"]["class_type"] = "TestConfiguredFixedNode"
        payload["workflow"]["nodes"]["fixed"]["inputs"]["config"] = {"context": "config"}
        original = deepcopy(payload)
        with patch.dict(server.NODE_CLASS_MAPPINGS, {"TestConfiguredFixedNode": ConfiguredFixedNode}):
            graph, plan = server.prepare_conditional_preview(payload)
            context = server.node_preview_context(graph, payload)
            result = run_conditional(plan, context, Event())
        self.assertTrue(result["accepted"])
        self.assertEqual(payload, original)
        self.assertEqual(workflow["nodes"]["generate"]["inputs"]["seed"], 7)
        self.assertEqual(workflow["nodes"]["generate2"]["inputs"]["seed"], 17)
        self.assert_no_writes()

    def test_exhaustion_fail_and_skip_return_last_round_including_sibling_end(self):
        for policy in ("fail", "skip"):
            with self.subTest(policy=policy):
                workflow = self.workflow()
                workflow["nodes"]["end"] = {
                    "class_type": "TestFixedNode", "inputs": {"image": link("generate", "image")}}
                self.loops(workflow)[0].update(end_nodes=["generate", "end"], on_exhausted=policy)
                self.loops(workflow)[0]["conditions"][0]["value"] = 2
                job = self.preview(workflow)
                self.assertEqual(job["status"], "failed" if policy == "fail" else "completed")
                result = job["results"][-1]
                self.assertEqual((result["accepted"], result["stop_reason"], result["iterations"]),
                                 (False, "max_iterations", 3))
                self.assertEqual(set(result["preview"]), {"generate", "end", "score_a"})
                self.assert_pixel(result["preview"]["generate"]["image"], (20, 9, 40))
                self.assert_pixel(result["preview"]["end"]["image"], (20, 9, 40))
                self.assertEqual(self.calls["generate"], [7, 8, 9])
                # History carries scalars/seeds only, not image snapshots.
                self.assertTrue(all("preview" not in entry for entry in result["history"]))
                json.dumps(self.manager.get(job["id"]))
                self.assert_no_writes()
                for values in self.calls.values():
                    values.clear()

    def test_timeout_returns_only_available_last_round_outputs_without_late_execution(self):
        workflow = self.workflow()
        self.loops(workflow)[0]["timeout_seconds"] = 5
        self.hook = lambda: setattr(self, "clock", 6)
        with patch("object_inpainting_studio.conditional_loop.monotonic", side_effect=lambda: self.clock):
            job = self.preview(workflow)
        result = job["results"][-1]
        self.assertEqual(result["stop_reason"], "timeout")
        self.assertEqual(set(result["preview"]), {"generate"})
        self.assert_pixel(result["preview"]["generate"]["image"], (20, 7, 40))
        self.assertEqual(self.calls["score"], [])
        self.assert_no_writes()

    def test_side_effect_ancestors_are_rejected_before_task_start(self):
        for upstream in (False, True):
            with self.subTest(upstream=upstream):
                workflow = self.workflow(serial=upstream)
                workflow["nodes"]["bad"] = {
                    "class_type": "TestSideEffectNode", "inputs": {"image": link("fixed", "image")}}
                workflow["nodes"]["generate"]["inputs"]["image"] = link("bad", "image")
                with self.assertLogs("object_inpainting_studio.server", level="ERROR"):
                    response = self.client.post("/api/conditional/preview",
                                                json=self.preview_payload(workflow, "loop2" if upstream else "loop1"))
                self.assertEqual(response.status_code, 422, response.text)
                self.assertIsNone(self.manager.latest())
                self.assertTrue(all(not values for values in self.calls.values()))
                self.assert_no_writes()

    def test_unknown_missing_or_invalid_loop_id_rejected(self):
        for loop_id in ("unknown", "", None, [], 1):
            with self.subTest(loop_id=loop_id):
                payload = self.preview_payload()
                payload["loop_id"] = loop_id
                with self.assertLogs("object_inpainting_studio.server", level="ERROR"):
                    response = self.client.post("/api/conditional/preview", json=payload)
                self.assertEqual(response.status_code, 422, response.text)
        payload.pop("loop_id")
        with self.assertLogs("object_inpainting_studio.server", level="ERROR"):
            self.assertEqual(self.client.post("/api/conditional/preview", json=payload).status_code, 422)
        self.assertIsNone(self.manager.latest())
        self.assert_no_writes()

    def test_normal_preview_cache_is_neither_read_nor_populated(self):
        payload = self.preview_payload()
        payload["preview_session"] = "unaccepted-session"
        before = deepcopy(server.PREVIEW_CACHE.sessions)
        with patch.object(server.PREVIEW_CACHE, "run", side_effect=AssertionError("no preview cache")):
            first = self.post_job("/api/conditional/preview", payload)
            second = self.post_job("/api/conditional/preview", payload)
        self.assertEqual((first["status"], second["status"]), ("completed", "completed"))
        self.assertEqual(self.calls["generate"], [7, 8, 7, 8])
        self.assertEqual(server.PREVIEW_CACHE.sessions, before)
        self.assert_no_writes()

    def test_initial_cache_cannot_skip_upstream_acceptance_or_freeze_bridge(self):
        workflow = self.workflow(serial=True, bridge=True)
        graph, plan = server.prepare_conditional_preview({"workflow": workflow, "loop_id": "loop2"})
        result = run_conditional(plan, {"image": Image.new("RGB", (3, 2), (20, 0, 40))}, Event(),
                                 initial_cache={"generate": {"bad": True}, "bridge": {"bad": True}})
        self.assertTrue(result["accepted"])
        self.assertEqual(self.calls["generate"], [7, 8, 17, 18])
        self.assertEqual(self.calls["fixed"], [(20, 0, 40), (20, 8, 40)])
        self.assertIs(graph, plan.graph)
        self.assert_no_writes()

    def test_shared_job_lock_and_cancel_endpoint(self):
        entered, release = Event(), Event()

        def block():
            entered.set()
            if not release.wait(10):
                raise AssertionError("worker release timed out")

        self.hook = block
        response = self.client.post("/api/conditional/preview", json=self.preview_payload())
        self.assertEqual(response.status_code, 200, response.text)
        job = response.json()
        try:
            self.assertTrue(entered.wait(10))
            for endpoint, payload in (("/api/conditional/preview", self.preview_payload()),
                                      ("/api/conditional", self.api_payload(self.workflow()))):
                with self.assertLogs("object_inpainting_studio.server", level="ERROR"):
                    blocked = self.client.post(endpoint, json=payload)
                self.assertEqual(blocked.status_code, 422, blocked.text)
            cancelled = self.client.post(f"/api/batch/{job['id']}/cancel")
            self.assertEqual(cancelled.json()["status"], "cancelling")
        finally:
            release.set()
        final = self.join_job(job)
        self.assertEqual(final["status"], "cancelled")
        self.assertEqual(final["results"], [])
        self.assert_no_writes()

    def test_serializer_callback_runs_per_round_and_final_without_rerunning_nodes(self):
        serialized = []
        original = server.serialize_node_output

        def serialize(value):
            serialized.append(value)
            return original(value)

        with patch.object(server, "serialize_node_output", side_effect=serialize):
            job = self.preview()
        self.assertEqual(job["status"], "completed")
        self.assertEqual(len(serialized), 18)  # Six ports, two snapshots + final preview.
        self.assertEqual(sum(isinstance(value, Image.Image) for value in serialized), 3)
        self.assertEqual(job["visualization_revision"], 2)
        self.assertEqual(set(job["loop_previews"]), {"loop1"})
        self.assertEqual(self.calls["generate"], [7, 8])
        self.assert_no_writes()

    def test_normal_multi_save_keeps_full_pipeline_and_no_preview_field(self):
        job = self.post_job("/api/conditional", self.api_payload(self.workflow(serial=True, bridge=True)))
        self.assertEqual(job["status"], "completed")
        self.assertEqual(self.calls["generate"], [7, 8, 17, 18])
        self.assertEqual(len(self.calls["save"]), 1)
        result = job["results"][-1]
        self.assertEqual(len(result["paths"]), 2)
        self.assertNotIn("preview", result)
        self.assertNotIn("preview_loop_id", job)
        self.assertEqual(len(list(self.output.rglob("*.png"))), 2)

    def test_flat_save_api_still_works(self):
        workflow = legacy.ConditionalLoopTests.workflow(self)
        _, plan = server.prepare_conditional({"workflow": workflow, "node_id": "save"})
        self.assertIsInstance(plan, ConditionalPlan)
        job = self.post_job("/api/conditional", self.api_payload(workflow))
        self.assertEqual(job["status"], "completed")
        self.assertEqual(self.calls["generate"], [7, 8])
        self.assertEqual(len(self.calls["save"]), 1)
        self.assertNotIn("preview", job["results"][-1])


if __name__ == "__main__":
    unittest.main()
