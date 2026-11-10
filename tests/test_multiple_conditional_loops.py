"""Multi-region CPU/PIL regressions, sharing only the existing test fixture."""

import json
import unittest
from copy import deepcopy
from threading import Event
from unittest.mock import patch

from PIL import Image

import test_conditional_loop as legacy
from object_inpainting.graph import GraphExecutionCancelled, NodeExecutionError, NodeGraph
from object_inpainting.nodes.folder_images import folder_image
from object_inpainting_studio import server
from object_inpainting_studio.conditional_loop import (
    ConditionalPlan, MultiConditionalPlan, build_conditional_plan,
    condition_nodes, loop_settings, run_conditional,
)


link = legacy.link


class MultipleConditionalLoopTests(unittest.TestCase):
    setUp = legacy.ConditionalLoopTests.setUp
    finish_workers = legacy.ConditionalLoopTests.finish_workers
    plan = legacy.ConditionalLoopTests.plan
    run_loop = legacy.ConditionalLoopTests.run_loop
    api_payload = legacy.ConditionalLoopTests.api_payload
    join_job = legacy.ConditionalLoopTests.join_job
    post_job = legacy.ConditionalLoopTests.post_job
    assert_no_writes = legacy.ConditionalLoopTests.assert_no_writes

    def workflow(self, folder=False, serial=False, bridge=False, reverse=False):
        workflow = legacy.ConditionalLoopTests.workflow(self, folder)
        first = workflow.pop("conditional_loop")
        first.update(id="loop1", name="循环1", end_nodes=["generate"])
        second = deepcopy(first)
        second.update(id="loop2", name="循环2", start_nodes=["generate2"],
                      end_nodes=["generate2"], seed_nodes=["generate2"],
                      conditions=[{"node": "score2", "output": "score", "operator": ">=", "value": 1}])
        nodes = workflow["nodes"]
        nodes["generate2"] = deepcopy(nodes["generate"])
        nodes["generate2"]["inputs"]["seed"] = 17
        if serial:
            source = "generate"
            if bridge:
                nodes["bridge"] = {"class_type": "TestFixedNode", "inputs": {"image": link(source, "image")}}
                source = "bridge"
            nodes["generate2"]["inputs"]["image"] = link(source, "image")
        nodes["score2"] = {"class_type": "TestScoreNode", "inputs": {"value": link("generate2", "a")}}
        nodes["save"]["save_items"].append({"id": "img2", "name": "second"})
        nodes["save"]["inputs"]["img2"] = link("generate2", "image")
        workflow["conditional_loop"] = {"loops": [second, first] if reverse else [first, second]}
        self.values = lambda seed: (float(seed % 10 >= 8), 0.0, seed % 10 >= 8)
        return workflow

    def loops(self, workflow):
        return workflow["conditional_loop"]["loops"]

    def workflow_with_save_choices(self):
        workflow = self.workflow()
        nodes = workflow["nodes"]
        nodes["save2"] = deepcopy(nodes["save"])
        nodes["save2"]["inputs"]["dir"] = str(self.root / "alternative")
        for target in ("save", "save2"):
            branch = f"{target}_input"
            nodes[branch] = {"class_type": "TestFixedNode", "inputs": {"image": link("generate", "image")}}
            nodes[target]["inputs"]["img1"] = link(branch, "image")
        return workflow

    def test_independent_regions_each_retry_save_once_real_pixels(self):
        workflow = self.workflow()
        original, updates = deepcopy(workflow), []
        result = self.run_loop(workflow, progress=lambda u: updates.append(deepcopy(u)))
        self.assertTrue(result["accepted"])
        self.assertEqual(result["iterations"], 4)
        self.assertEqual(self.calls["generate"], [7, 8, 17, 18])
        self.assertEqual(self.calls["fixed"], [(20, 0, 40)])
        self.assertEqual(self.calls["save"], [(20, 8, 40)])
        self.assertEqual([r["iterations"] for r in result["loops"]], [2, 2])
        self.assertEqual([e["loop_id"] for e in result["history"]], ["loop1", "loop1", "loop2", "loop2"])
        self.assertEqual([e["loop_name"] for e in result["history"]], ["循环1", "循环1", "循环2", "循环2"])
        for port, pixel in (("img1", (20, 8, 40)), ("img2", (20, 18, 40))):
            with Image.open(result["paths"][port]) as image:
                self.assertEqual(image.getpixel((0, 0)), pixel)
        self.assertEqual(len(list(self.output.rglob("*.png"))), 2)
        self.assertEqual(workflow, original)
        self.assertEqual([(u["current_loop_id"], u["current_loop_index"], u["current_loop_count"])
                          for u in updates if "current_loop_id" in u], [("loop1", 1, 2), ("loop2", 2, 2)])
        json.dumps(result)  # No PIL cache in the report.

    def test_serial_reverse_list_sorted_and_accepted_cache_reused(self):
        workflow = self.workflow(serial=True, reverse=True)
        self.assertEqual([r["id"] for r in self.plan(workflow).describe()["loops"]], ["loop1", "loop2"])
        self.assertTrue(self.run_loop(workflow)["accepted"])
        self.assertEqual(self.calls["generate"], [7, 8, 17, 18])
        self.assertEqual(len(self.calls["save"]), 1)

    def test_transitive_bridge_dependency_sorted_and_cached_once(self):
        workflow = self.workflow(serial=True, bridge=True, reverse=True)
        description = self.plan(workflow).describe()
        self.assertEqual([r["id"] for r in description["loops"]], ["loop1", "loop2"])
        self.assertIn("bridge", description["fixed_nodes"])
        self.assertNotIn("bridge", description["body_nodes"])
        self.assertTrue(self.run_loop(workflow)["accepted"])
        self.assertEqual(self.calls["generate"], [7, 8, 17, 18])
        self.assertEqual(self.calls["fixed"], [(20, 0, 40), (20, 8, 40)])

    def test_second_exhausted_fail_writes_nothing(self):
        workflow = self.workflow(serial=True)
        self.loops(workflow)[1]["conditions"][0]["value"] = 2
        result = self.run_loop(workflow)
        self.assertFalse(result["accepted"])
        self.assertIn("error", result)
        self.assertEqual(result["iterations"], 5)
        self.assertEqual([r["accepted"] for r in result["loops"]], [True, False])
        self.assertEqual(self.calls["generate"], [7, 8, 17, 18, 19])
        self.assert_no_writes()

    def test_second_exhausted_skip_writes_nothing(self):
        workflow = self.workflow()
        self.loops(workflow)[1].update(max_iterations=1, on_exhausted="skip")
        result = self.run_loop(workflow)
        self.assertTrue(result["skipped"])
        self.assertEqual(result["paths"], {})
        self.assertNotIn("error", result)
        self.assert_no_writes()

    def test_first_failure_does_not_execute_future_region(self):
        workflow = self.workflow()
        self.loops(workflow)[0]["max_iterations"] = 1
        result = self.run_loop(workflow)
        self.assertFalse(result["accepted"])
        self.assertEqual(len(result["loops"]), 1)
        self.assertEqual(self.calls["generate"], [7])
        self.assert_no_writes()

    def test_explicit_end_is_forced_before_next_region_not_deferred_to_save(self):
        workflow = self.workflow()
        workflow["nodes"]["end"] = {"class_type": "TestFixedNode", "inputs": {"image": link("generate", "image")}}
        workflow["nodes"]["save"]["inputs"]["img1"] = link("end", "image")
        self.loops(workflow)[0]["end_nodes"] = ["end"]
        observations = []
        self.hook = lambda: observations.append(len(self.calls["fixed"]))
        self.assertTrue(self.run_loop(workflow)["accepted"])
        self.assertEqual(observations, [1, 1, 2, 2])
        self.assertEqual(self.calls["fixed"], [(20, 0, 40), (20, 8, 40)])

    def test_mutable_fixed_and_accepted_images_are_isolated(self):
        workflow = self.workflow(serial=True, bridge=True)
        self.mutate_input = True
        result = self.run_loop(workflow)
        self.assertEqual(self.calls["fixed"], [(20, 0, 40), (20, 8, 40)])
        with Image.open(result["paths"]["img1"]) as image:
            self.assertEqual(image.getpixel((0, 0)), (20, 8, 40))
        with Image.open(result["paths"]["img2"]) as image:
            self.assertEqual(image.getpixel((0, 0)), (20, 18, 40))

    def test_initial_cache_cannot_bypass_acceptance_or_include_generated_nodes(self):
        workflow = self.workflow()
        self.assertTrue(self.run_loop(workflow, initial_cache={"generate": {"invalid": True},
                                                             "save": {"paths": {"bad": "bad"}}})["accepted"])
        self.assertEqual(self.calls["generate"], [7, 8, 17, 18])
        self.assertEqual(len(self.calls["save"]), 1)

    def test_settings_is_dict_and_slice_collects_all_conditions_and_ends(self):
        workflow = self.workflow()
        settings = loop_settings({"workflow": workflow})
        self.assertIsInstance(settings, dict)
        self.assertEqual(set(condition_nodes(settings)), {"generate", "score_a", "generate2", "score2"})
        self.assertEqual(settings, workflow["conditional_loop"])

    def test_schema_rejects_mixed_flat_even_unknown_root_field(self):
        for key in ("conditions", "start_nodes", "max_iterations", "unknown"):
            with self.subTest(key=key):
                workflow = self.workflow()
                workflow["conditional_loop"][key] = []
                with self.assertRaisesRegex(ValueError, "混合"):
                    loop_settings(workflow)

    def test_loop_count_limits_and_entry_types(self):
        for values in ([], {}, None, [None], ["loop"], [{}], [{"id": "x", "loops": []}],
                       [{"id": str(i)} for i in range(33)]):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    loop_settings({"conditional_loop": {"loops": values}})
        workflow = self.workflow()
        self.assertIsInstance(build_conditional_plan(self.plan(workflow).graph, "save",
                                                    workflow["conditional_loop"], workflow), MultiConditionalPlan)

    def test_duplicate_empty_and_invalid_ids_names(self):
        for key, value in (("id", "loop1"), ("id", ""), ("id", " "), ("id", 2),
                           ("name", ""), ("name", None)):
            with self.subTest(key=key, value=value):
                workflow = self.workflow()
                self.loops(workflow)[1][key] = value
                with self.assertRaises(ValueError):
                    self.plan(workflow)

    def test_explicit_nonempty_unique_end_nodes_required(self):
        for value in (None, [], "generate", [""], [2], ["generate", "generate"], ["save"], ["missing"]):
            with self.subTest(value=value):
                workflow = self.workflow()
                self.loops(workflow)[0]["end_nodes"] = value
                with self.assertRaises(ValueError):
                    self.plan(workflow)
        workflow = self.workflow()
        del self.loops(workflow)[0]["end_nodes"]
        with self.assertRaises(ValueError):
            self.plan(workflow)

    def test_overlap_or_nested_regions_rejected(self):
        for serial in (False, True):
            with self.subTest(serial=serial):
                workflow = self.workflow(serial=serial)
                self.loops(workflow)[1]["start_nodes"] = ["generate"] if serial else ["generate", "generate2"]
                self.loops(workflow)[1]["end_nodes"] = ["generate", "generate2"]
                with self.assertRaisesRegex(ValueError, "重叠|嵌套|start"):
                    self.plan(workflow)

    def test_every_start_must_reach_an_end(self):
        workflow = self.workflow()
        self.loops(workflow)[0]["start_nodes"].append("score_a")
        with self.assertRaisesRegex(ValueError, "start"):
            self.plan(workflow)

    def test_all_ends_and_conditions_must_be_inside_region(self):
        for key, value in (("end_nodes", ["generate", "fixed"]),
                           ("conditions", [{"node": "generate2", "output": "a", "operator": ">=", "value": 1}])):
            with self.subTest(key=key):
                workflow = self.workflow()
                self.loops(workflow)[0][key] = value
                with self.assertRaisesRegex(ValueError, "循环体"):
                    self.plan(workflow)

    def test_unselected_branch_cannot_consume_undeclared_intermediate(self):
        workflow = self.workflow()
        workflow["nodes"]["escape"] = {"class_type": "TestFixedNode", "inputs": {"image": link("generate", "image")}}
        self.loops(workflow)[0]["end_nodes"] = ["score_a"]
        with self.assertRaisesRegex(ValueError, "逃逸"):
            self.plan(workflow)

    def test_selected_save_reads_accepted_intermediate_without_extra_end(self):
        workflow = self.workflow()
        self.loops(workflow)[0].update(end_nodes=["score_a"], conditions=[
            {"node": "generate", "output": "a", "operator": ">=", "value": 1}])
        result = self.run_loop(workflow)
        self.assertTrue(result["accepted"])
        self.assertEqual(self.calls["generate"], [7, 8, 17, 18])
        self.assertEqual(len(self.calls["save"]), 1)
        with Image.open(result["paths"]["img1"]) as image:
            self.assertEqual(image.getpixel((0, 0)), (20, 8, 40))

    def test_declared_exit_can_feed_ordinary_downstream_or_unselected_branch(self):
        workflow = self.workflow()
        workflow["nodes"]["downstream"] = {"class_type": "TestFixedNode", "inputs": {"image": link("generate", "image")}}
        workflow["nodes"]["unselected"] = deepcopy(workflow["nodes"]["downstream"])
        workflow["nodes"]["save"]["inputs"]["img1"] = link("downstream", "image")
        self.assertTrue(self.run_loop(workflow)["accepted"])
        self.assertEqual(len(self.calls["save"]), 1)

    def test_direct_workflow_output_is_rejected_even_declared_end(self):
        workflow = self.workflow()
        workflow["outputs"]["leak"] = link("generate", "image")
        with self.assertRaisesRegex(ValueError, "直接暴露"):
            self.plan(workflow)

    def test_unrelated_region_is_not_run(self):
        workflow = self.workflow()
        workflow["nodes"]["save"]["inputs"].pop("img2")
        workflow["nodes"]["save"]["save_items"].pop()
        with self.assertRaisesRegex(ValueError, "无关"):
            self.plan(workflow)
        self.assertEqual(self.calls["generate"], [])

    def test_global_second_side_effect_and_folder_body_rejected(self):
        for kind in ("save", "side", "folder"):
            with self.subTest(kind=kind):
                workflow = self.workflow(folder=kind == "folder")
                if kind == "save":
                    workflow["nodes"]["extra"] = deepcopy(workflow["nodes"]["save"])
                elif kind == "side":
                    workflow["nodes"]["extra"] = {"class_type": "TestSideEffectNode", "inputs": {"image": link("fixed", "image")}}
                    # Keep this side effect in the selected ancestor slice.
                    workflow["nodes"]["generate"]["inputs"]["image"] = link("extra", "image")
                else:
                    self.loops(workflow)[0]["start_nodes"] = ["folder"]
                with self.assertRaisesRegex(ValueError, "写盘|文件夹"):
                    if kind == "save":
                        # A custom execution graph containing every node still
                        # rejects the second save; server-side slicing is required.
                        graph = NodeGraph(workflow)
                        build_conditional_plan(graph, "save", workflow["conditional_loop"], workflow)
                    else:
                        self.plan(workflow)

    def test_two_save_choices_plan_selected_slice_and_run_only_selected_branch(self):
        workflow = self.workflow_with_save_choices()
        original = deepcopy(workflow)
        for target, other in (("save", "save2"), ("save2", "save")):
            with self.subTest(target=target):
                graph, plan = server.prepare_conditional({"workflow": workflow, "node_id": target})
                self.assertIsInstance(plan, MultiConditionalPlan)
                self.assertIs(plan.graph, graph)
                self.assertEqual(plan.describe()["target_id"], target)
                self.assertEqual(set(graph.nodes), set(workflow["nodes"]) - {other, f"{other}_input"})
                self.assertIn(f"{target}_input", plan.fixed)
                self.assertEqual([n for n, spec in graph.nodes.items()
                                  if getattr(graph.registry[spec["class_type"]], "SIDE_EFFECTS", False)], [target])
        self.assert_no_writes()
        self.assertFalse((self.root / "alternative").exists())
        self.assertTrue(all(not values for values in self.calls.values()))
        result = self.run_loop(workflow)
        self.assertTrue(result["accepted"])
        self.assertEqual(self.calls["generate"], [7, 8, 17, 18])
        self.assertEqual(self.calls["fixed"], [(20, 0, 40), (20, 8, 40)])
        self.assertEqual(self.calls["save"], [(20, 8, 40)])
        self.assertEqual(len(list(self.output.rglob("*.png"))), 2)
        self.assertFalse((self.root / "alternative").exists())
        self.assertEqual(workflow, original)

    def test_api_two_save_choices_plan_both_and_write_only_second_directory(self):
        workflow = self.workflow_with_save_choices()
        payload = self.api_payload(workflow)
        for target in ("save", "save2"):
            payload["node_id"] = target
            response = self.client.post("/api/conditional/plan", json=payload)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["target_id"], target)
            self.assertIn(f"{target}_input", response.json()["fixed_nodes"])
            other = "save2" if target == "save" else "save"
            self.assertNotIn(f"{other}_input", response.json()["fixed_nodes"])
        self.assertTrue(all(not values for values in self.calls.values()))
        self.assert_no_writes()
        alternative = self.root / "alternative"
        self.assertFalse(alternative.exists())
        final = self.post_job("/api/conditional", payload)
        self.assertEqual((final["status"], final["succeeded"]), ("completed", 1))
        self.assertEqual(self.calls["generate"], [7, 8, 17, 18])
        self.assertEqual(self.calls["fixed"], [(20, 0, 40), (20, 8, 40)])
        self.assertEqual(self.calls["save"], [(20, 8, 40)])
        self.assertFalse(self.output.exists())
        self.assertEqual(len(list(alternative.rglob("*.png"))), 2)
        for port, pixel in (("img1", (20, 8, 40)), ("img2", (20, 18, 40))):
            path = alternative / ("accepted" if port == "img1" else "second") / "accepted.png"
            with Image.open(path) as image:
                self.assertEqual(image.getpixel((0, 0)), pixel)

    def test_unselected_save_consuming_body_intermediate_still_rejected(self):
        workflow = self.workflow()
        nodes = workflow["nodes"]
        nodes["save2"] = deepcopy(nodes["save"])
        nodes["save2"]["inputs"]["dir"] = str(self.root / "alternative")
        # The unselected save consumes only an undeclared body intermediate.
        nodes["save2"]["inputs"].pop("img2")
        nodes["save2"]["save_items"].pop()
        nodes["end"] = {"class_type": "TestFixedNode", "inputs": {"image": link("generate", "image")}}
        nodes["save"]["inputs"]["img1"] = link("end", "image")
        self.loops(workflow)[0]["end_nodes"] = ["end"]
        graph = server.node_preview_graph(workflow, "save", condition_nodes(workflow["conditional_loop"]))
        self.assertNotIn("save2", graph.nodes)
        with self.assertRaisesRegex(ValueError, "逃逸到节点 save2"):
            build_conditional_plan(graph, "save", workflow["conditional_loop"], workflow)
        with self.assertLogs("object_inpainting_studio.server", level="ERROR"):
            for endpoint in ("/api/conditional/plan", "/api/conditional"):
                response = self.client.post(endpoint, json=self.api_payload(workflow))
                self.assertEqual(response.status_code, 422, response.text)
                self.assertIn("逃逸到节点 save2", response.json()["detail"])
        self.assertIsNone(self.manager.latest())
        self.assertTrue(all(not values for values in self.calls.values()))
        self.assert_no_writes()
        self.assertFalse((self.root / "alternative").exists())

    def test_cross_future_dependencies_between_disjoint_bodies_rejected(self):
        workflow = self.workflow()
        nodes = workflow["nodes"]
        nodes["a2"] = deepcopy(nodes["generate"])
        nodes["b2"] = deepcopy(nodes["generate2"])
        nodes["generate2"]["inputs"]["image"] = link("generate", "image")
        nodes["a2"]["inputs"]["image"] = link("b2", "image")
        nodes["save"]["inputs"]["img1"] = link("a2", "image")
        self.loops(workflow)[0].update(start_nodes=["generate", "a2"], end_nodes=["generate", "a2"])
        self.loops(workflow)[1].update(start_nodes=["generate2", "b2"], end_nodes=["generate2", "b2"])
        with self.assertRaisesRegex(ValueError, "未来依赖"):
            self.plan(workflow)

    def test_region_limits_conditions_and_seed_validation(self):
        cases = [("max_iterations", True), ("max_iterations", 0), ("max_iterations", 1001),
                 ("timeout_seconds", True), ("timeout_seconds", float("inf")), ("timeout_seconds", -1),
                 ("mode", "bad"), ("on_exhausted", "save"), ("start_nodes", []),
                 ("seed_nodes", ["fixed"]), ("seed_nodes", ["generate", "generate"]),
                 ("conditions", []), ("conditions", [{"node": "generate", "output": "image", "operator": "==", "value": 1}]),
                 ("conditions", [{"node": "generate", "output": "a", "operator": "==", "value": True}]),
                 ("conditions", [{"node": "generate", "output": "flag", "operator": ">", "value": True}]),
                 ("conditions", [{"node": "generate", "output": "a", "operator": "bad", "value": 1}])]
        for key, value in cases:
            with self.subTest(key=key, value=value):
                workflow = self.workflow()
                self.loops(workflow)[0][key] = value
                with self.assertRaises(ValueError):
                    self.plan(workflow)

    def test_api_plan_and_run_persist_progress_without_inference_on_plan(self):
        payload = self.api_payload(self.workflow(serial=True, bridge=True, reverse=True))
        with patch.object(self.registry["TestGenerateNode"], "execute", side_effect=AssertionError("no inference")):
            response = self.client.post("/api/conditional/plan", json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        description = response.json()
        self.assertEqual([r["id"] for r in description["loops"]], ["loop1", "loop2"])
        self.assertEqual(description["target_id"], "save")
        self.assertIn("max_iterations", description)
        self.assertEqual(description["seed_nodes"], ["generate", "generate2"])
        self.assertEqual(len(description["conditions"]), 2)
        self.assertEqual(self.calls["save"], [])
        final = self.post_job("/api/conditional", payload)
        self.assertEqual((final["status"], final["succeeded"]), ("completed", 1))
        self.assertEqual((final["current_loop_id"], final["current_loop_name"],
                          final["current_loop_index"], final["current_loop_count"]), ("loop2", "循环2", 2, 2))
        self.assertEqual(len(final["iteration_history"]), 4)
        self.assertEqual(len(final["results"][0]["loops"]), 2)
        json.dumps(final)

    def test_api_invalid_plan_and_run_reject_without_job(self):
        workflow = self.workflow()
        self.loops(workflow)[1]["id"] = "loop1"
        with self.assertLogs("object_inpainting_studio.server", level="ERROR"):
            for endpoint in ("/api/conditional/plan", "/api/conditional"):
                response = self.client.post(endpoint, json=self.api_payload(workflow))
                self.assertEqual(response.status_code, 422, response.text)
        self.assertIsNone(self.manager.latest())
        self.assert_no_writes()

    def test_factory_and_api_legacy_flat_still_work(self):
        workflow = legacy.ConditionalLoopTests.workflow(self)
        plan = self.plan(workflow)
        self.assertIsInstance(plan, ConditionalPlan)
        self.assertNotIn("loops", plan.describe())
        final = self.post_job("/api/conditional", self.api_payload(workflow))
        self.assertEqual(final["status"], "completed")
        self.assertEqual(final["results"][0]["iterations"], 2)
        self.assertNotIn("loop_id", final["iteration_history"][0])

    def test_folder_each_file_resets_loop_fields_history_seeds_and_caches(self):
        for name, pixel in (("a.png", (20, 0, 40)), ("b.png", (60, 0, 80))):
            Image.new("RGB", (3, 2), pixel).save(self.input / name)
        payload = self.api_payload(self.workflow(folder=True, serial=True, bridge=True, reverse=True))
        resets = []

        def decode(*args):
            snapshot = self.manager.latest()
            resets.append(tuple(snapshot[k] for k in ("current_loop_id", "current_loop_name", "current_loop_index",
                                                       "current_loop_count", "current_iteration", "iteration_history")))
            return folder_image(*args)

        with patch("object_inpainting_studio.batch.folder_image", side_effect=decode):
            final = self.post_job("/api/batch", payload)
        self.assertEqual(final["status"], "completed")
        self.assertEqual((final["processed"], final["succeeded"]), (2, 2))
        self.assertEqual(resets, [(None, None, 0, 0, 0, []), (None, None, 0, 0, 0, [])])
        self.assertEqual(self.calls["generate"], [7, 8, 17, 18] * 2)
        self.assertEqual(self.calls["fixed"], [(20, 0, 40), (20, 8, 40), (60, 0, 80), (60, 8, 80)])
        self.assertEqual([len(r["history"]) for r in final["results"]], [4, 4])
        self.assertEqual(len(list(self.output.rglob("*.png"))), 4)

    def test_precancelled_and_cancel_between_regions_never_save(self):
        workflow = self.workflow()
        cancel = Event()
        cancel.set()
        with self.assertRaises(GraphExecutionCancelled):
            run_conditional(self.plan(workflow), {}, cancel)
        self.assertEqual(self.calls["generate"], [])
        cancel.clear()

        def progress(update):
            if update.get("current_loop_id") == "loop2":
                cancel.set()

        with self.assertRaises(GraphExecutionCancelled):
            run_conditional(
                self.plan(workflow), {"image": Image.new("RGB", (3, 2), (20, 0, 40))}, cancel, progress)
        self.assertEqual(self.calls["generate"], [7, 8])
        self.assert_no_writes()

    def test_cancel_after_all_accepted_before_save(self):
        workflow, cancel = self.workflow(), Event()

        def progress(update):
            history = update.get("iteration_history", [])
            if len(history) == 4 and history[-1]["accepted"]:
                cancel.set()

        with self.assertRaises(GraphExecutionCancelled):
            run_conditional(self.plan(workflow), {"image": Image.new("RGB", (3, 2))}, cancel, progress)
        self.assertEqual(self.calls["generate"], [7, 8, 17, 18])
        self.assert_no_writes()

    def test_timeout_second_region_and_late_save_dependency(self):
        for late in (False, True):
            with self.subTest(late=late):
                workflow = self.workflow()
                self.clock = 0
                self.loops(workflow)[1].update(timeout_seconds=5, on_exhausted="skip")
                if late:
                    workflow["nodes"]["late"] = {"class_type": "TestFixedNode", "inputs": {"image": {"context": "image"}}}
                    workflow["nodes"]["save"]["save_items"].append({"id": "img3", "name": "late"})
                    workflow["nodes"]["save"]["inputs"]["img3"] = link("late", "image")
                    self.fixed_hook = lambda: setattr(self, "clock", 6) if len(self.calls["fixed"]) > 1 else None
                else:
                    self.hook = lambda: setattr(self, "clock", 6) if self.calls["generate"][-1] == 17 else None
                with patch("object_inpainting_studio.conditional_loop.monotonic", side_effect=lambda: self.clock):
                    result = self.run_loop(workflow)
                self.assertEqual(result["stop_reason"], "timeout")
                self.assertTrue(result["skipped"])
                self.assert_no_writes()
                self.hook, self.fixed_hook = lambda: None, lambda: None
                for values in self.calls.values():
                    values.clear()

    def test_node_exception_is_not_retried_and_never_saves(self):
        workflow = self.workflow()

        def explode():
            if self.calls["generate"][-1] == 17:
                raise RuntimeError("second region failed")

        self.hook = explode
        with self.assertLogs("object_inpainting", level="ERROR"):
            with self.assertRaises(NodeExecutionError):
                self.run_loop(workflow)
        self.assertEqual(self.calls["generate"], [7, 8, 17])
        self.assert_no_writes()

    def test_thirty_two_independent_regions_supported(self):
        workflow = self.workflow()
        fixed, generate, save = (deepcopy(workflow["nodes"][key]) for key in ("fixed", "generate", "save"))
        workflow["nodes"] = {"fixed": fixed, "save": save}
        save["save_items"] = []
        save["inputs"] = {"dir": str(self.output), "filename": "all"}
        entries = []
        for index in range(32):
            node, port = f"gen{index}", f"img{index + 1}"
            workflow["nodes"][node] = deepcopy(generate)
            save["save_items"].append({"id": port, "name": f"item{index}"})
            save["inputs"][port] = link(node, "image")
            entries.append({"id": f"loop{index}", "start_nodes": [node], "end_nodes": [node],
                            "seed_nodes": [node], "max_iterations": 2,
                            "conditions": [{"node": node, "output": "a", "operator": ">=", "value": 1}]})
        workflow["conditional_loop"] = {"loops": entries}
        self.assertEqual(len(self.plan(workflow).describe()["loops"]), 32)
        result = self.run_loop(workflow)
        self.assertTrue(result["accepted"])
        self.assertEqual((len(result["loops"]), result["iterations"], len(result["paths"])), (32, 64, 32))
        self.assertEqual(len(self.calls["save"]), 1)

    def test_single_entry_new_format_supported(self):
        workflow = self.workflow()
        self.loops(workflow).pop()
        del workflow["nodes"]["generate2"], workflow["nodes"]["score2"]
        workflow["nodes"]["save"]["save_items"].pop()
        workflow["nodes"]["save"]["inputs"].pop("img2")
        result = self.run_loop(workflow)
        self.assertEqual((result["accepted"], result["iterations"], len(result["loops"])), (True, 2, 1))

    def test_any_mode_evaluates_all_conditions_without_cross_round_mix(self):
        workflow = self.workflow()
        self.loops(workflow)[0].update(mode="any", conditions=[
            {"node": "generate", "output": "a", "operator": ">=", "value": 1},
            {"node": "score_a", "output": "score", "operator": "==", "value": 0}])
        result = self.run_loop(workflow)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["loops"][0]["iterations"], 1)
        self.assertEqual([s["passed"] for s in result["history"][0]["scores"]], [False, True])
        self.assertEqual(self.calls["score"], [0.0, 0.0, 1.0])

    def test_runtime_nonfinite_or_wrong_scalar_is_not_retried(self):
        for value in (float("nan"), float("inf"), True, "bad"):
            with self.subTest(value=value):
                workflow = self.workflow()
                self.loops(workflow)[0]["conditions"] = [
                    {"node": "generate", "output": "a", "operator": ">=", "value": 1}]
                self.values = lambda seed, value=value: (value, 0.0, False)
                with self.assertRaisesRegex(ValueError, "类型|有限"):
                    self.run_loop(workflow)
                self.assertEqual(self.calls["generate"], [7])
                self.calls["generate"].clear()
                self.assert_no_writes()

    def test_seed_direct_literal_and_overflow_validation(self):
        for value in (True, -1, 2**63 - 2, {"context": "seed"}):
            with self.subTest(value=value):
                workflow = self.workflow()
                workflow["nodes"]["generate"]["inputs"]["seed"] = value
                with self.assertRaises(ValueError):
                    self.plan(workflow)

    def test_cancel_during_generation_or_forced_end_never_saves(self):
        for at_end in (False, True):
            with self.subTest(at_end=at_end):
                workflow, cancel = self.workflow(), Event()
                if at_end:
                    workflow["nodes"]["end"] = {"class_type": "TestFixedNode", "inputs": {"image": link("generate", "image")}}
                    workflow["nodes"]["save"]["inputs"]["img1"] = link("end", "image")
                    self.loops(workflow)[0]["end_nodes"] = ["end"]
                    self.fixed_hook = lambda: cancel.set() if len(self.calls["fixed"]) == 2 else None
                else:
                    self.hook = cancel.set
                with self.assertRaises(GraphExecutionCancelled):
                    run_conditional(self.plan(workflow), {"image": Image.new("RGB", (3, 2))}, cancel)
                self.assert_no_writes()
                self.hook, self.fixed_hook = lambda: None, lambda: None
                for values in self.calls.values():
                    values.clear()

    def test_cancel_during_completed_save_keeps_accepted_result(self):
        workflow, cancel = self.workflow(), Event()
        save_class = self.registry["SaveImagesNode"]
        original = save_class.execute_with_spec

        def finish_save(instance, inputs, spec):
            result = original(instance, inputs, spec)
            cancel.set()
            return result

        with patch.object(save_class, "execute_with_spec", finish_save):
            result = run_conditional(self.plan(workflow), {"image": Image.new("RGB", (3, 2))}, cancel)
        self.assertTrue(result["accepted"])
        self.assertEqual(len(self.calls["save"]), 1)

    def test_timeout_during_forced_end_never_enters_next_loop(self):
        workflow = self.workflow()
        self.loops(workflow)[0]["timeout_seconds"] = 5
        workflow["nodes"]["end"] = {"class_type": "TestFixedNode", "inputs": {"image": link("generate", "image")}}
        workflow["nodes"]["save"]["inputs"]["img1"] = link("end", "image")
        self.loops(workflow)[0]["end_nodes"] = ["end"]
        self.fixed_hook = lambda: setattr(self, "clock", 6) if len(self.calls["fixed"]) == 2 else None
        with patch("object_inpainting_studio.conditional_loop.monotonic", side_effect=lambda: self.clock):
            result = self.run_loop(workflow)
        self.assertEqual((result["accepted"], result["stop_reason"]), (False, "timeout"))
        self.assertEqual(self.calls["generate"], [7, 8])
        self.assert_no_writes()

    def test_end_outside_save_ancestry_is_sliced_in_and_forced(self):
        workflow = self.workflow()
        workflow["nodes"]["offsave_end"] = {
            "class_type": "TestFixedNode", "inputs": {"image": link("generate", "image")}}
        self.loops(workflow)[0]["end_nodes"].append("offsave_end")
        plan = self.plan(workflow)
        self.assertIn("offsave_end", plan.describe()["loops"][0]["body_nodes"])
        observations = []
        self.hook = lambda: observations.append(len(self.calls["fixed"]))
        self.assertTrue(self.run_loop(workflow)["accepted"])
        self.assertEqual(observations, [1, 1, 2, 2])
        self.assertEqual(self.calls["fixed"], [(20, 0, 40), (20, 8, 40)])

    def test_folder_second_region_fail_stops_and_skip_continues_no_writes(self):
        for name in ("a.png", "b.png"):
            Image.new("RGB", (3, 2), (20, 0, 40)).save(self.input / name)
        for policy, processed, status in (("fail", 1, "failed"), ("skip", 2, "completed")):
            with self.subTest(policy=policy):
                workflow = self.workflow(folder=True)
                self.loops(workflow)[1].update(max_iterations=1, on_exhausted=policy)
                payload = self.api_payload(workflow)
                payload["continue_on_error"] = False
                final = self.post_job("/api/batch", payload)
                self.assertEqual((final["status"], final["processed"], final["succeeded"]), (status, processed, 0))
                self.assertEqual(final["skipped"], 2 if policy == "skip" else 0)
                for result in final["results"]:
                    self.assertEqual([r["accepted"] for r in result["loops"]], [True, False])
                    self.assertEqual(result["iterations"], 3)
                self.assert_no_writes()

    def test_folder_current_preview_uses_multi_plan_and_only_selected_file(self):
        for name in ("a.png", "b.png"):
            Image.new("RGB", (3, 2), (20, 0, 40)).save(self.input / name)
        workflow = self.workflow(folder=True)
        workflow["nodes"]["folder"]["inputs"]["index"] = 1
        final = self.post_job("/api/batch/preview", self.api_payload(workflow))
        self.assertEqual((final["status"], final["total"], final["succeeded"]), ("completed", 1, 1))
        self.assertTrue(final["results"][0]["source"].endswith("b.png"))
        self.assertEqual(len(final["results"][0]["loops"]), 2)
        self.assertEqual(self.calls["generate"], [7, 8, 17, 18])
        self.assertEqual(len(list(self.output.rglob("*.png"))), 2)


if __name__ == "__main__":
    unittest.main()
