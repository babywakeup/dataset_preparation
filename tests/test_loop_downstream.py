"""CPU regressions for accepted-cache downstream tests and live job snapshots."""

import json
import unittest
from copy import deepcopy
from threading import Event
from unittest.mock import patch

from PIL import Image

import test_conditional_loop as legacy
import test_multiple_conditional_loops as multiple
import test_loop_preview as previews
from object_inpainting.nodes import Node
from object_inpainting_studio import server
from object_inpainting_studio.conditional_loop import run_conditional


link = legacy.link


class LoopDownstreamTests(unittest.TestCase):
    setUp = legacy.ConditionalLoopTests.setUp
    finish_workers = legacy.ConditionalLoopTests.finish_workers
    api_payload = legacy.ConditionalLoopTests.api_payload
    join_job = legacy.ConditionalLoopTests.join_job
    post_job = legacy.ConditionalLoopTests.post_job
    assert_no_writes = legacy.ConditionalLoopTests.assert_no_writes
    workflow = multiple.MultipleConditionalLoopTests.workflow
    loops = multiple.MultipleConditionalLoopTests.loops
    assert_pixel = previews.LoopPreviewTests.assert_pixel

    def payload(self, workflow=None, node="downstream", **options):
        workflow = self.workflow() if workflow is None else workflow
        if node == "downstream" and node not in workflow["nodes"]:
            workflow["nodes"][node] = {
                "class_type": "TestFixedNode", "inputs": {"image": link("generate", "image")}}
        payload = self.api_payload(workflow)
        payload.update(node_id=node, **options)
        return payload

    def run_node(self, workflow=None, node="downstream", **options):
        return self.post_job("/api/node/preview", self.payload(workflow, node, **options))

    def add_join(self, workflow):
        owner = self
        self.join_calls = []

        class Join(Node):
            INPUT_TYPES = {"first": "RGB_IMAGE", "second": "RGB_IMAGE"}
            OUTPUT_TYPES = {"image": "RGB_IMAGE"}

            def execute(self, first, second):
                owner.join_calls.append((first.getpixel((0, 0)), second.getpixel((0, 0))))
                return {"image": second.copy()}

        registry_patch = patch.dict(server.NODE_CLASS_MAPPINGS, {"TestJoin": Join})
        registry_patch.start()
        self.addCleanup(registry_patch.stop)
        workflow["nodes"]["downstream"] = {
            "class_type": "TestJoin", "inputs": {
                "first": link("generate", "image"), "second": link("generate2", "image")}}

    def test_downstream_reads_accepted_internal_not_declared_end(self):
        workflow = self.workflow()
        self.loops(workflow)[0]["end_nodes"] = ["score_a"]
        payload = self.payload(workflow)
        original = deepcopy(payload)
        with patch.object(server.PREVIEW_CACHE, "run", side_effect=AssertionError("unaccepted cache")):
            job = self.post_job("/api/node/preview", payload)
        self.assertEqual(job["status"], "completed")
        self.assertEqual(job["preview_node_id"], "downstream")
        self.assertEqual(job["required_loop_ids"], ["loop1"])
        self.assertEqual(job["conditional_loop"]["preview_node_id"], "downstream")
        self.assertEqual(self.calls["generate"], [7, 8])
        self.assertEqual(self.calls["fixed"], [(20, 0, 40), (20, 8, 40)])
        self.assert_pixel(job["results"][0]["preview"]["downstream"]["image"], (20, 8, 40))
        self.assertEqual(payload, original)
        self.assert_no_writes()

    def test_any_internal_node_test_runs_full_owning_loop(self):
        for node in ("generate", "score_a"):
            with self.subTest(node=node):
                workflow = self.workflow()
                workflow["nodes"]["sibling"] = {
                    "class_type": "TestFixedNode", "inputs": {"image": link("generate", "image")}}
                self.loops(workflow)[0]["end_nodes"] = ["sibling"]
                self.loops(workflow)[0]["conditions"].append(
                    {"node": "generate", "output": "flag", "operator": "==", "value": True})
                job = self.run_node(workflow, node)
                self.assertEqual(job["required_loop_ids"], ["loop1"])
                self.assertEqual(self.calls["generate"], [7, 8])
                self.assertEqual(self.calls["fixed"], [(20, 0, 40), (20, 8, 40)])
                self.assertEqual(len(job["results"][0]["history"][-1]["scores"]), 2)
                self.assertEqual(set(job["results"][0]["preview"]), {node})
                self.assert_no_writes()
                for values in self.calls.values():
                    values.clear()

    def test_multistart_internal_test_still_requires_other_start_and_end(self):
        workflow = self.workflow()
        workflow["nodes"]["alternate"] = deepcopy(workflow["nodes"]["generate"])
        workflow["nodes"]["alternate"]["inputs"]["seed"] = 37
        first = self.loops(workflow)[0]
        first["start_nodes"].append("alternate")
        first["seed_nodes"].append("alternate")
        first["end_nodes"].append("alternate")
        job = self.run_node(workflow, "alternate")
        self.assertEqual(self.calls["generate"], [7, 8, 38])
        self.assert_pixel(job["results"][0]["preview"]["alternate"]["image"], (20, 38, 40))
        self.assert_no_writes()

    def test_independent_join_requires_both_regions_once(self):
        workflow = self.workflow()
        self.add_join(workflow)
        job = self.run_node(workflow)
        self.assertEqual(job["required_loop_ids"], ["loop1", "loop2"])
        self.assertEqual(self.calls["generate"], [7, 8, 17, 18])
        self.assertEqual(self.join_calls, [((20, 8, 40), (20, 18, 40))])
        self.assertEqual(job["visualization_revision"], 4)
        self.assertEqual(set(job["loop_previews"]), {"loop1", "loop2"})
        self.assert_no_writes()

    def test_serial_reverse_join_uses_accepted_bridge_and_regions(self):
        workflow = self.workflow(serial=True, bridge=True, reverse=True)
        self.loops(workflow)[1]["end_nodes"] = ["score_a"]
        self.add_join(workflow)
        self.mutate_input = True
        job = self.run_node(workflow)
        self.assertEqual(job["required_loop_ids"], ["loop1", "loop2"])
        self.assertEqual(self.calls["generate"], [7, 8, 17, 18])
        self.assertEqual(self.calls["fixed"], [(20, 0, 40), (20, 8, 40)])
        self.assertEqual(self.join_calls, [((20, 8, 40), (20, 18, 40))])
        self.assert_no_writes()

    def test_unrelated_incomplete_or_invalid_loop_does_not_block(self):
        for variant in ("incomplete", "missing", "invalid"):
            with self.subTest(variant=variant):
                workflow = self.workflow()
                if variant == "incomplete":
                    self.loops(workflow)[1].update(start_nodes=[], end_nodes=[], conditions=[])
                elif variant == "missing":
                    self.loops(workflow)[1].update(start_nodes=["missing"], end_nodes=["missing"])
                else:
                    workflow["nodes"]["generate2"]["class_type"] = "MissingClass"
                job = self.run_node(workflow)
                self.assertEqual(job["status"], "completed")
                self.assertEqual(job["required_loop_ids"], ["loop1"])
                self.assertEqual(self.calls["generate"], [7, 8])
                for values in self.calls.values():
                    values.clear()
        self.assert_no_writes()

    def test_needed_incomplete_loop_rejected_not_silently_skipped(self):
        for update in ({"end_nodes": [], "conditions": []}, {"start_nodes": []},
                       {"start_nodes": [], "end_nodes": []}):
            with self.subTest(update=update):
                workflow = self.workflow()
                self.loops(workflow)[0].update(update)
                with self.assertLogs("object_inpainting_studio.server", level="ERROR"):
                    response = self.client.post("/api/node/preview", json=self.payload(workflow))
                self.assertEqual(response.status_code, 422)
                self.assertIsNone(self.manager.latest())
        self.assert_no_writes()

    def test_failed_or_skipped_loop_never_executes_downstream(self):
        for policy in ("fail", "skip"):
            with self.subTest(policy=policy):
                workflow = self.workflow()
                self.add_join(workflow)
                self.loops(workflow)[1].update(max_iterations=1, on_exhausted=policy)
                job = self.run_node(workflow)
                self.assertFalse(job["results"][0]["accepted"])
                self.assertEqual(job["results"][0]["preview"], {})
                self.assertEqual(self.join_calls, [])
                self.assertEqual(self.calls["generate"], [7, 8, 17])
                self.assertFalse(job["loop_previews"]["loop2"]["accepted"])
                self.assert_pixel(job["loop_previews"]["loop2"]["outputs"]["generate2"]["image"], (20, 17, 40))
                for values in self.calls.values():
                    values.clear()
        self.assert_no_writes()

    def test_timeout_does_not_execute_downstream_and_keeps_partial_snapshot(self):
        workflow = self.workflow()
        self.loops(workflow)[0]["timeout_seconds"] = 5
        self.hook = lambda: setattr(self, "clock", 6)
        with patch("object_inpainting_studio.conditional_loop.monotonic", side_effect=lambda: self.clock):
            job = self.run_node(workflow)
        self.assertEqual(job["results"][0]["stop_reason"], "timeout")
        self.assertEqual(self.calls["fixed"], [(20, 0, 40)])
        self.assertEqual(self.calls["score"], [])
        self.assertEqual(job["results"][0]["preview"], {})
        self.assert_pixel(job["loop_previews"]["loop1"]["outputs"]["generate"]["image"], (20, 7, 40))
        self.assert_no_writes()

    def test_second_loop_internal_test_requires_upstream_acceptance(self):
        workflow = self.workflow(serial=True, bridge=True, reverse=True)
        job = self.run_node(workflow, "generate2")
        self.assertEqual(job["required_loop_ids"], ["loop1", "loop2"])
        self.assertEqual(self.calls["generate"], [7, 8, 17, 18])
        self.assert_pixel(job["results"][0]["preview"]["generate2"]["image"], (20, 18, 40))
        self.assert_no_writes()

    def test_unrelated_node_keeps_synchronous_preview_cache_api(self):
        payload = self.payload(node="fixed")
        first = self.client.post("/api/node/preview", json=payload).json()
        self.assertIn("preview_session", first)
        self.assertNotIn("id", first)
        self.assertEqual(first["executed_nodes"], ["fixed"])
        payload["preview_session"] = first["preview_session"]
        second = self.client.post("/api/node/preview", json=payload).json()
        self.assertIn("outputs", second)
        self.assertEqual(self.calls["generate"], [])
        self.assertIsNone(self.manager.latest())
        self.assert_no_writes()

    def test_repeated_async_tests_do_not_read_or_write_normal_cache(self):
        payload = self.payload(preview_session="unaccepted")
        before = deepcopy(server.PREVIEW_CACHE.sessions)
        with patch.object(server.PREVIEW_CACHE, "run", side_effect=AssertionError("cache bypass")):
            self.post_job("/api/node/preview", payload)
            self.post_job("/api/node/preview", payload)
        self.assertEqual(self.calls["generate"], [7, 8, 7, 8])
        self.assertEqual(server.PREVIEW_CACHE.sessions, before)
        self.assert_no_writes()

    def test_flat_node_preview_explicitly_rejected_before_computation(self):
        workflow = legacy.ConditionalLoopTests.workflow(self)
        with self.assertLogs("object_inpainting_studio.server", level="ERROR"):
            response = self.client.post("/api/node/preview", json=self.payload(workflow, "generate"))
        self.assertEqual(response.status_code, 422)
        self.assertIn("flat", response.json()["detail"])
        self.assertEqual(self.calls["generate"], [])
        self.assert_no_writes()

    def test_side_effect_target_or_ancestor_rejected_before_start(self):
        for target in ("bad", "downstream", "save"):
            with self.subTest(target=target):
                workflow = self.workflow()
                workflow["nodes"]["bad"] = {
                    "class_type": "TestSideEffectNode", "inputs": {"image": link("generate", "image")}}
                workflow["nodes"]["downstream"] = {
                    "class_type": "TestFixedNode", "inputs": {"image": link("bad", "image")}}
                with self.assertLogs("object_inpainting_studio.server", level="ERROR"):
                    response = self.client.post("/api/node/preview", json=self.payload(workflow, target))
                self.assertEqual(response.status_code, 422)
                self.assertIsNone(self.manager.latest())
        self.assert_no_writes()

    def test_nested_workflow_wrapper_and_original_payload_unchanged(self):
        payload = self.payload()
        payload["workflow"] = {"workflow": payload["workflow"]}
        original = deepcopy(payload)
        job = self.post_job("/api/node/preview", payload)
        self.assertEqual(job["status"], "completed")
        self.assertEqual(payload, original)
        self.assert_no_writes()

    def test_visualize_false_disables_all_serialization_but_not_acceptance(self):
        payload = self.payload(visualize=False)
        with patch.object(server, "serialize_node_output", side_effect=AssertionError("must not serialize")):
            job = self.post_job("/api/node/preview", payload)
        self.assertFalse(job["visualize"])
        self.assertEqual(job["loop_previews"], {})
        self.assertEqual(job["visualization_revision"], 0)
        self.assertEqual(job["results"][0]["preview"], {})
        self.assertEqual([e["scores"][0]["value"] for e in job["results"][0]["history"]], [0.0, 1.0])
        self.assertEqual(self.calls["generate"], [7, 8])
        self.assertEqual(self.calls["fixed"], [(20, 0, 40), (20, 8, 40)])
        self.assertNotIn("data:image", json.dumps(job))
        self.assert_no_writes()

    def test_visualize_false_independent_preview(self):
        payload = self.api_payload(self.workflow())
        payload.update(loop_id="loop1", visualize=False)
        with patch.object(server, "serialize_node_output", side_effect=AssertionError("must not serialize")):
            job = self.post_job("/api/conditional/preview", payload)
        self.assertEqual(job["results"][0]["preview"], {})
        self.assertEqual(job["loop_previews"], {})
        self.assertEqual(self.calls["generate"], [7, 8])
        self.assert_no_writes()

    def test_visualize_rejects_non_bool_without_starting_job(self):
        for value in (None, 0, 1, "false", []):
            with self.subTest(value=value), self.assertLogs("object_inpainting_studio.server", level="ERROR"):
                response = self.client.post("/api/node/preview", json=self.payload(visualize=value))
                self.assertEqual(response.status_code, 422)
        self.assertIsNone(self.manager.latest())
        self.assert_no_writes()

    def test_live_snapshot_corresponds_to_failed_round_before_next_finishes(self):
        entered, release = Event(), Event()

        def block_second():
            if self.calls["generate"][-1] == 8:
                entered.set()
                if not release.wait(10):
                    raise AssertionError("release timed out")

        self.hook = block_second
        response = self.client.post("/api/node/preview", json=self.payload())
        self.assertEqual(response.status_code, 200, response.text)
        initial = response.json()
        self.assertTrue(initial["visualize"])
        self.assertEqual(initial["loop_previews"], {})
        self.assertEqual(initial["visualization_revision"], 0)
        try:
            self.assertTrue(entered.wait(10))
            live = self.manager.get(initial["id"])
            self.assertEqual(live["visualization_revision"], 1)
            snapshot = live["loop_previews"]["loop1"]
            self.assertEqual((snapshot["iteration"], snapshot["accepted"]), (1, False))
            self.assertEqual(live["iteration_history"][-1]["scores"][0]["value"], 0.0)
            self.assert_pixel(snapshot["outputs"]["generate"]["image"], (20, 7, 40))
            json.dumps(live)
        finally:
            release.set()
        final = self.join_job(initial)
        self.assertEqual(final["visualization_revision"], 2)
        self.assertEqual(final["loop_previews"]["loop1"]["iteration"], 2)
        self.assertEqual(set(final["loop_previews"]), {"loop1"})
        self.assertTrue(all("preview" not in e for e in final["iteration_history"]))
        self.assert_no_writes()

    def test_downstream_job_shares_lock_and_cancels_at_node_boundary(self):
        entered, release = Event(), Event()

        def block():
            entered.set()
            if not release.wait(10):
                raise AssertionError("release timed out")

        self.hook = block
        response = self.client.post("/api/node/preview", json=self.payload())
        initial = response.json()
        try:
            self.assertTrue(entered.wait(10))
            with self.assertLogs("object_inpainting_studio.server", level="ERROR"):
                conflict = self.client.post("/api/node/preview", json=self.payload())
            self.assertEqual(conflict.status_code, 422)
            self.assertEqual(self.manager.cancel(initial["id"])["status"], "cancelling")
        finally:
            release.set()
        job = self.join_job(initial)
        self.assertEqual(job["status"], "cancelled")
        self.assertEqual(job["results"], [])
        self.assertEqual(self.calls["fixed"], [(20, 0, 40)])
        self.assert_no_writes()

    def test_all_conditions_and_sibling_end_precede_downstream(self):
        workflow = self.workflow()
        workflow["nodes"]["sibling"] = {
            "class_type": "TestFixedNode", "inputs": {"image": link("generate", "image")}}
        self.loops(workflow)[0]["end_nodes"].append("sibling")
        self.loops(workflow)[0]["conditions"].append(
            {"node": "generate", "output": "b", "operator": ">=", "value": 1})
        job = self.run_node(workflow)
        self.assertFalse(job["results"][0]["accepted"])
        self.assertEqual(self.calls["fixed"], [(20, 0, 40)])
        self.assert_no_writes()

    def test_formal_multi_save_publishes_all_region_snapshots(self):
        job = self.post_job("/api/conditional", self.api_payload(self.workflow()))
        self.assertEqual(job["status"], "completed")
        self.assertEqual(job["visualization_revision"], 4)
        self.assertEqual(set(job["loop_previews"]), {"loop1", "loop2"})
        for loop_id, generator, seed in (("loop1", "generate", 8), ("loop2", "generate2", 18)):
            snapshot = job["loop_previews"][loop_id]
            self.assertEqual((snapshot["iteration"], snapshot["accepted"]), (2, True))
            self.assert_pixel(snapshot["outputs"][generator]["image"], (20, seed, 40))
        self.assertEqual(len(self.calls["save"]), 1)

    def test_formal_flat_save_publishes_current_round(self):
        job = self.post_job("/api/conditional", self.api_payload(legacy.ConditionalLoopTests.workflow(self)))
        self.assertEqual(job["visualization_revision"], 2)
        self.assertEqual(set(job["loop_previews"]), {"flat"})
        self.assert_pixel(job["loop_previews"]["flat"]["outputs"]["generate"]["image"], (20, 8, 40))
        self.assertEqual(len(self.calls["save"]), 1)

    def test_formal_save_visualize_false_keeps_save_and_scores(self):
        for flat in (False, True):
            with self.subTest(flat=flat):
                workflow = legacy.ConditionalLoopTests.workflow(self) if flat else self.workflow()
                workflow["nodes"]["save"]["inputs"]["dir"] = str(self.root / f"save-{flat}")
                payload = self.api_payload(workflow)
                payload["visualize"] = False
                with patch.object(server, "serialize_node_output", side_effect=AssertionError("no images")):
                    job = self.post_job("/api/conditional", payload)
                self.assertTrue(job["results"][0]["accepted"])
                self.assertEqual(job["loop_previews"], {})
                self.assertEqual(job["visualization_revision"], 0)
                self.assertNotIn("data:image", json.dumps(job))
        self.assertEqual(len(self.calls["save"]), 2)

    def test_folder_conditional_trial_and_run_publish_snapshots(self):
        for endpoint in ("/api/batch/preview", "/api/batch"):
            with self.subTest(endpoint=endpoint):
                Image.new("RGB", (3, 2), (20, 0, 40)).save(self.input / "one.png")
                workflow = self.workflow(folder=True)
                workflow["nodes"]["save"]["inputs"]["dir"] = str(self.root / endpoint.rsplit("/", 1)[-1])
                job = self.post_job(endpoint, self.api_payload(workflow))
                self.assertEqual(job["visualization_revision"], 5)
                self.assertEqual(set(job["loop_previews"]), {"loop1", "loop2", "batch"})
                self.assertEqual(job["status"], "completed")

    def test_folder_images_do_not_accumulate_visual_history(self):
        for index in range(2):
            Image.new("RGB", (3, 2), (20, 0, 40)).save(self.input / f"{index}.png")
        job = self.post_job("/api/batch", self.api_payload(self.workflow(folder=True)))
        self.assertEqual(job["processed"], 2)
        self.assertEqual(job["visualization_revision"], 10)
        self.assertEqual(set(job["loop_previews"]), {"loop1", "loop2", "batch"})
        self.assertTrue(all("preview" not in entry for result in job["results"] for entry in result["history"]))

    def test_nonconditional_folder_batch_publishes_visual_outputs(self):
        Image.new("RGB", (3, 2), (20, 0, 40)).save(self.input / "one.png")
        workflow = legacy.ConditionalLoopTests.workflow(self, folder=True)
        del workflow["conditional_loop"]
        job = self.post_job("/api/batch", self.api_payload(workflow))
        self.assertEqual(job["status"], "completed")
        self.assertGreater(job["visualization_revision"], 0)
        self.assertIn("batch", job["loop_previews"])
        self.assert_pixel(job["results"][0]["preview"]["save"]["paths"]["items"]["img1"], (20, 7, 40))

    def test_serializer_callback_is_optional_for_direct_jobs(self):
        payload = self.api_payload(self.workflow())
        payload["loop_id"] = "loop1"
        graph, plan = server.prepare_conditional_preview(payload)
        context = server.node_preview_context(graph, payload)
        job = self.manager.start(graph, context, None, None, [None], 0, False, conditional=plan)
        final = self.join_job(job)
        self.assertTrue(final["results"][0]["accepted"])
        self.assertEqual(final["results"][0]["preview"], {})
        self.assertEqual(final["loop_previews"], {})
        json.dumps(final)
        self.assert_no_writes()

    def test_raw_progress_publishes_failed_and_accepted_images_with_scores(self):
        payload = self.api_payload(self.workflow())
        payload["loop_id"] = "loop1"
        graph, plan = server.prepare_conditional_preview(payload)
        context = server.node_preview_context(graph, payload)
        updates = []
        run_conditional(plan, context, Event(), lambda update: updates.append(deepcopy(update)))
        rounds = [u for u in updates if "loop_preview" in u]
        self.assertEqual(len(rounds), 2)
        for update, seed, accepted in zip(rounds, (7, 8), (False, True)):
            snapshot = update["loop_preview"]
            self.assertEqual(snapshot["outputs"]["generate"]["image"].getpixel((0, 0)), (20, seed, 40))
            self.assertEqual(snapshot["accepted"], accepted)
            self.assertEqual(update["iteration_history"][-1]["accepted"], accepted)
        self.assert_no_writes()

    def test_new_escape_policy_applies_to_formal_plans_too(self):
        workflow = self.workflow()
        self.loops(workflow)[0]["end_nodes"] = ["score_a"]
        workflow["nodes"]["downstream"] = {
            "class_type": "TestFixedNode", "inputs": {"image": link("generate", "image")}}
        workflow["nodes"]["save"]["inputs"]["img1"] = link("downstream", "image")
        job = self.post_job("/api/conditional", self.api_payload(workflow))
        self.assertTrue(job["results"][0]["accepted"])
        self.assertEqual(self.calls["generate"], [7, 8, 17, 18])
        self.assertEqual(self.calls["save"], [(20, 8, 40)])

    def test_downstream_is_not_executed_when_upstream_condition_raises(self):
        self.hook = lambda: (_ for _ in ()).throw(ValueError("generated failure"))
        job = self.run_node()
        self.assertEqual(job["status"], "failed")
        self.assertEqual(job["node_error"]["node_id"], "generate")
        self.assertEqual(job["results"][0]["stop_reason"], "error")
        self.assertEqual(self.calls["fixed"], [(20, 0, 40)])
        self.assertFalse(job["loop_previews"]["loop1"]["accepted"])
        self.assert_no_writes()

    def test_downstream_timeout_after_all_loops_never_returns_accepted_preview(self):
        workflow = self.workflow()
        self.loops(workflow)[0]["timeout_seconds"] = 5

        def late_downstream():
            if len(self.calls["fixed"]) == 2:
                self.clock = 6

        self.fixed_hook = late_downstream
        with patch("object_inpainting_studio.conditional_loop.monotonic", side_effect=lambda: self.clock):
            job = self.run_node(workflow)
        self.assertEqual(job["results"][0]["stop_reason"], "timeout")
        self.assertFalse(job["results"][0]["accepted"])
        self.assertEqual(job["results"][0]["preview"], {})
        self.assertEqual(self.calls["generate"], [7, 8])
        self.assert_no_writes()

    def test_required_sibling_end_can_add_another_upstream_loop(self):
        workflow = self.workflow(reverse=True)
        self.add_join(workflow)
        self.loops(workflow)[1]["end_nodes"].append("downstream")
        # loop1's second end consumes loop2 even though the tested node does
        # not. Its full acceptance must therefore discover and run loop2 first.
        job = self.run_node(workflow, "generate")
        self.assertEqual(job["required_loop_ids"], ["loop2", "loop1"])
        self.assertEqual(self.calls["generate"], [17, 18, 7, 8])
        self.assertEqual(self.join_calls, [((20, 8, 40), (20, 18, 40))])
        self.assert_no_writes()

    def test_visualize_false_folder_trial_and_batch_keep_computation(self):
        Image.new("RGB", (3, 2), (20, 0, 40)).save(self.input / "one.png")
        for index, endpoint in enumerate(("/api/batch/preview", "/api/batch")):
            with self.subTest(endpoint=endpoint):
                workflow = self.workflow(folder=True)
                workflow["nodes"]["save"]["inputs"]["dir"] = str(self.root / f"disabled-{index}")
                payload = self.api_payload(workflow)
                payload["visualize"] = False
                with patch.object(server, "serialize_node_output", side_effect=AssertionError("no images")):
                    job = self.post_job(endpoint, payload)
                self.assertEqual(job["status"], "completed")
                self.assertTrue(job["results"][0]["accepted"])
                self.assertEqual(job["loop_previews"], {})
                self.assertEqual(job["visualization_revision"], 0)
        self.assertEqual(self.calls["generate"], [7, 8, 17, 18] * 2)

    def test_no_loop_preview_visualize_false_keeps_cache_execution(self):
        payload = self.payload(node="fixed", visualize=False)
        with patch.object(server, "serialize_node_output", side_effect=AssertionError("no images")):
            response = self.client.post("/api/node/preview", json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["outputs"], {})
        self.assertIn("preview_session", response.json())
        self.assertEqual(self.calls["fixed"], [(20, 0, 40)])
        self.assertIsNone(self.manager.latest())
        self.assert_no_writes()

    def test_flat_failure_still_publishes_last_scored_image(self):
        workflow = legacy.ConditionalLoopTests.workflow(self)
        workflow["conditional_loop"]["max_iterations"] = 1
        job = self.post_job("/api/conditional", self.api_payload(workflow))
        self.assertEqual(job["status"], "failed")
        self.assertEqual(job["visualization_revision"], 1)
        self.assertFalse(job["loop_previews"]["flat"]["accepted"])
        self.assertEqual(job["iteration_history"][-1]["scores"][0]["value"], 0.0)
        self.assert_pixel(job["loop_previews"]["flat"]["outputs"]["generate"]["image"], (20, 7, 40))
        self.assert_no_writes()

    def test_formal_workflow_outputs_still_cannot_bypass_loop(self):
        workflow = self.workflow()
        workflow["outputs"] = {"leak": link("generate", "image")}
        with self.assertLogs("object_inpainting_studio.server", level="ERROR"):
            response = self.client.post("/api/conditional", json=self.api_payload(workflow))
        self.assertEqual(response.status_code, 422)
        self.assertIsNone(self.manager.latest())
        self.assert_no_writes()


if __name__ == "__main__":
    unittest.main()
