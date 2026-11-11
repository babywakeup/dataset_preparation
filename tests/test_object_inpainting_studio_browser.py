import base64
import io
import json
import os
import re
import socket
import unittest
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import patch

import test_object_inpainting_dataset as dataset_tests
import uvicorn
from object_inpainting.nodes import NODE_CLASS_MAPPINGS, GrayMaskNode
from object_inpainting.nodes.loop_demo import LoopDemoImageNode
from object_inpainting_studio.server import app
from PIL import Image
from playwright.sync_api import expect, sync_playwright


class StudioBrowserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = os.environ.get("STUDIO_TEST_URL")
        cls.server = None
        if not cls.base_url:
            cls.socket = socket.socket()
            cls.socket.bind(("127.0.0.1", 0))
            cls.socket.listen()
            cls.base_url = f"http://127.0.0.1:{cls.socket.getsockname()[1]}"
            cls.server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
            cls.server_thread = Thread(target=cls.server.run, kwargs={"sockets": [cls.socket]}, daemon=True)
            cls.server_thread.start()
        cls.playwright = sync_playwright().start()
        cls.browser = cls.playwright.chromium.launch(headless=True)

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.playwright.stop()
        if cls.server is not None:
            cls.server.should_exit = True
            cls.server_thread.join(timeout=10)
            cls.socket.close()

    def setUp(self):
        self.fixture = dataset_tests.ObjectInpaintingDatasetTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.context = self.browser.new_context(viewport={"width": 1600, "height": 1000}, accept_downloads=True)
        self.addCleanup(self.context.close)
        self.page = self.context.new_page()
        self.errors = []
        self.page.on("pageerror", lambda error: self.errors.append(str(error)))
        self.page.goto(self.base_url + "/")
        expect(self.page.locator("#status")).to_contain_text("Ready.")

    def test_folder_all_terminal_branches_run_and_show_score_without_condition(self):
        folder = self.fixture.root / "terminal-input"
        folder.mkdir()
        Image.new("RGB", (80, 60), (25, 50, 255)).save(folder / "one.png")
        workflow = {"nodes": {
            "folder": {"class_type": "FolderImagesNode", "inputs": {"dir": str(folder)}},
            "demo": {"class_type": "LoopDemoImageNode", "inputs": {
                "image": {"node": "folder", "output": "image"}, "seed": 0}},
            "score": {"class_type": "LoopDemoScoreNode", "inputs": {
                "image": {"node": "demo", "output": "image"}}},
            **{name: {"class_type": "SaveImagesNode", "inputs": {
                "img1": {"node": "demo", "output": "image"},
                "dir": str(self.fixture.root / name)}} for name in ("save", "save2")},
        }, "outputs": {}}
        self.page.evaluate("""workflow => {
            state.workflow = workflow; state.layout = {}; state.nodeResults = {};
            restoreLoop({sourceId: 'folder', targetId: 'save', enabled: true}); render();
        }""", workflow)
        expect(self.page.locator('[data-node-id="score"] .loop-role')).to_have_text("结果节点")
        expect(self.page.locator("#loop-run")).to_be_enabled()
        with patch("object_inpainting.nodes.loop_demo.time.sleep"):
            self.page.locator("#loop-run").click()
            self.page.wait_for_function("batch.job?.status === 'completed'")
        self.page.wait_for_function("state.nodeResults.score?.score?.value === 0")
        self.assertEqual(set(self.page.evaluate("Object.keys(batch.job.results[0].paths_by_node)")), {"save", "save2"})
        for name in ("save", "save2"):
            self.assertEqual(len(list((self.fixture.root / name).rglob("*.png"))), 1)
            self.assertTrue((self.fixture.root / name / "img1" / "one.png").exists())
        self.assertEqual(self.errors, [])

    def test_batch_result_selection_disable_and_authoritative_plan_without_save(self):
        folder = self.fixture.root / "selected-input"
        folder.mkdir()
        Image.new("RGB", (80, 60), "blue").save(folder / "one.png")
        output = self.fixture.root / "must-not-be-created"
        workflow = {"nodes": {
            "folder": {"class_type": "FolderImagesNode", "inputs": {"dir": str(folder)}},
            "demo": {"class_type": "LoopDemoImageNode", "inputs": {"image": {"node": "folder", "output": "image"}, "seed": 0}},
            "score": {"class_type": "LoopDemoScoreNode", "inputs": {"image": {"node": "demo", "output": "image"}}},
            "unfinished": {"class_type": "LoopDemoScoreNode", "inputs": {"image": {"node": "demo", "output": "value"}}},
            "save": {"class_type": "SaveImagesNode", "inputs": {"img1": {"node": "demo", "output": "image"}, "dir": str(output)}},
        }, "outputs": {}}
        self.page.evaluate("""workflow => {
            state.workflow = workflow; state.layout = {}; state.nodeResults = {};
            restoreLoop({sourceId: 'folder', targetId: 'save', enabled: true}); render();
        }""", workflow)
        self.page.locator('[data-node-id="unfinished"] .disabled_nodes').click()
        expect(self.page.locator("#loop-run")).to_be_enabled()
        expect(self.page.locator('[data-node-id="unfinished"] .loop-role')).to_have_text("不执行")
        self.page.locator('[data-node-id="score"] .result_nodes').click()
        expect(self.page.locator("#batch-result-mode")).to_have_value("selected_results")
        expect(self.page.locator("#loop-run")).to_be_enabled()
        expect(self.page.locator('[data-node-id="save"] .loop-role')).to_have_text("不执行")
        with patch("object_inpainting.nodes.loop_demo.time.sleep"):
            self.page.locator("#loop-run").click()
            self.page.wait_for_function("batch.job?.status === 'completed'")
        self.page.wait_for_function("state.nodeResults.score?.score?.value === 0")
        self.assertFalse(output.exists())
        self.assertEqual(self.page.evaluate("batch.job.current_item.status"), "succeeded")
        self.assertEqual(self.page.evaluate("Object.keys(batch.job.results[0].preview)"), ["score"])
        self.assertEqual(self.errors, [])

    def test_cpu_demo_previews_each_loop_without_save_and_keeps_failed_last_image(self) -> None:
        self.page.evaluate("""() => {
            state.workflow = {nodes: {}, outputs: {}};
            state.layout = {}; restoreLoop(); render();
        }""")
        self.page.locator("#conditional-panel > summary").click()
        self.page.locator("#conditional-demo").click()
        cards = self.page.locator(".multiple-loop-card")
        expect(cards).to_have_count(2)
        ids = self.page.evaluate("multipleLoops().map(item => item.id)")
        first, second = cards.nth(0), cards.nth(1)
        first.get_by_role("button", name="预览此循环 · 不写盘", exact=True).click()
        expect(first.locator(".multiple-preview")).to_contain_text("已通过", timeout=15000)
        expect(first.locator(".multiple-preview img")).to_have_count(1)
        self.assertEqual(self.page.evaluate("batch.job.required_loop_ids"), [ids[0]])
        self.assertEqual(self.page.evaluate("batch.job.results[0].iterations"), 3)
        self.assertEqual(self.page.evaluate("batch.job.iteration_history.map(e => e.scores[0].value)"), [0, 0, 255])
        self.assertEqual(self.page.evaluate("batch.job.results[0].paths"), {})
        first_url = first.locator(".multiple-preview img").get_attribute("src")
        second.get_by_role("button", name="预览此循环 · 不写盘", exact=True).click()
        expect(second.locator(".multiple-preview")).to_contain_text("已通过", timeout=15000)
        self.assertEqual(self.page.evaluate("batch.job.required_loop_ids"), [ids[1]])
        self.assertEqual(self.page.evaluate("batch.job.results[0].iterations"), 2)
        expect(first.locator(".multiple-preview img")).to_have_attribute("src", first_url)
        threshold = second.get_by_label("条件 1 比较值", exact=True)
        threshold.fill("256")
        threshold.press("Tab")
        second.get_by_role("button", name="预览此循环 · 不写盘", exact=True).click()
        expect(second.locator(".multiple-preview")).to_contain_text("未通过 / 已停止", timeout=15000)
        self.assertEqual(self.page.evaluate("batch.job.status"), "failed")
        self.assertEqual(self.page.evaluate("batch.job.results[0].iterations"), 3)
        self.assertEqual(self.page.evaluate("batch.job.iteration_history.map(e => e.scores[0].value)"), [0, 255, 0])
        self.assertEqual(self.page.evaluate("batch.job.iteration_history.map(e => Object.values(e.seeds)[0])"), [1, 2, 3])
        preview = self.page.evaluate("Object.values(batch.job.results[0].preview).find(v => v.image).image")
        with Image.open(io.BytesIO(base64.b64decode(preview["image"].split(",", 1)[1]))) as image:
            self.assertEqual(image.getpixel((0, 0)), (255, 0, 0))
        expect(second.locator(".multiple-preview img")).to_have_attribute("src", preview["image"])
        expect(first.locator(".multiple-preview img")).to_have_attribute("src", first_url)
        self.assertEqual(self.page.evaluate("Object.values(state.workflow.nodes).filter(n => n.class_type === 'SaveImagesNode').length"), 0)
        self.assertEqual(self.errors, [])

    def load_cpu_downstream_loop(self) -> None:
        self.page.evaluate("""() => {
            state.workflow = {nodes: {
                generator: {class_type: 'LoopDemoImageNode', inputs: {seed: 0}},
                score: {class_type: 'LoopDemoScoreNode', inputs: {image: link('generator', 'image')}},
                next: {class_type: 'LoopDemoScoreNode', inputs: {image: link('generator', 'image')}}
            }, outputs: {}, conditional_loop: {loops: [{
                id: 'rgb', name: 'RGB 验收', start_nodes: ['generator'], end_nodes: ['score'],
                seed_nodes: ['generator'], conditions: [{node: 'score', output: 'score', operator: '>=', value: 255}],
                mode: 'all', max_iterations: 3, timeout_seconds: 0, on_exhausted: 'fail'
            }]}};
            state.layout = {generator: {x: 30, y: 40}, score: {x: 390, y: 40}, next: {x: 750, y: 40}};
            restoreLoop(); render();
        }""")

    def assert_cpu_loop_image(self, color) -> None:
        source = self.page.evaluate("state.nodeResults.generator.image.image")
        with Image.open(io.BytesIO(base64.b64decode(source.split(',', 1)[1]))) as image:
            self.assertEqual((image.mode, image.size), ('RGB', (32, 32)))
            self.assertEqual(image.getcolors(), [(1024, color)])
        for selector in ('.node[data-node-id="generator"] .node-output-view img',
                         '#run-visualization img', '.multiple-preview img'):
            expect(self.page.locator(selector)).to_have_count(1)
            expect(self.page.locator(selector)).to_have_attribute("src", source)
            expect(self.page.locator(selector)).to_be_visible()

    def test_cpu_downstream_preview_runs_required_loop_and_updates_real_images(self) -> None:
        if os.environ.get("STUDIO_TEST_URL"):
            self.skipTest("execution synchronization requires the in-process server")
        self.load_cpu_downstream_loop()
        expect(self.page.locator("#visualize")).to_have_value("yes")
        release_green, release_blue = Event(), Event()
        calls = []
        original_execute = LoopDemoImageNode.execute

        def execute(node, seed=0):
            calls.append(seed)
            if seed == 1:
                self.assertTrue(release_green.wait(30), "browser did not release the green round")
            elif seed == 2:
                self.assertTrue(release_blue.wait(30), "browser did not release the blue round")
            return original_execute(node, seed)

        with patch.object(LoopDemoImageNode, "execute", new=execute):
            try:
                with self.page.expect_response(lambda response: response.url.endswith("/api/node/preview") and response.request.method == "POST") as submitted:
                    self.page.locator('.node[data-node-id="next"] .node-preview-button').click()
                self.assertTrue(submitted.value.ok)
                initial = submitted.value.json()
                self.assertIn("id", initial)
                self.assertNotIn("outputs", initial)  # Standard endpoint returned an asynchronous job.
                self.assertEqual(initial["preview_node_id"], "next")
                self.assertEqual(initial["required_loop_ids"], ["rgb"])
                self.page.wait_for_function("state.nodeResults.generator?.value?.value === 0")
                self.assertEqual(self.page.evaluate("batch.job.status"), "running")
                self.assert_cpu_loop_image((255, 0, 0))
                release_green.set()
                self.page.wait_for_function("state.nodeResults.generator?.value?.value === 1")
                self.assert_cpu_loop_image((0, 255, 0))
                release_blue.set()
                expect(self.page.locator("#batch-status")).to_contain_text("已完成", timeout=15000)
                self.assert_cpu_loop_image((0, 0, 255))
            finally:
                release_green.set()
                release_blue.set()
                self.page.wait_for_function("batch.job && !['running', 'cancelling'].includes(batch.job.status)", timeout=15000)
        self.assertEqual(calls, [0, 1, 2])
        self.assertEqual(self.page.evaluate("batch.job.required_loop_ids"), ["rgb"])
        self.assertTrue(self.page.evaluate("batch.job.visualize"))
        self.assertEqual(self.page.evaluate("batch.job.results[0].iterations"), 3)
        self.assertTrue(self.page.evaluate("batch.job.results[0].accepted"))
        self.assertEqual(self.page.evaluate("batch.job.iteration_history.map(e => e.scores[0].value)"), [0, 0, 255])
        self.assertEqual(self.page.evaluate("state.nodeResults.next.score.value"), 255)
        self.assertEqual(self.page.evaluate("batch.job.results[0].preview.next.score.value"), 255)
        self.assertEqual(self.page.evaluate("batch.job.results[0].paths"), {})
        self.assertEqual(self.page.evaluate("state.workflow.nodes.generator.inputs.seed"), 0)
        self.assertEqual(self.errors, [])

    def test_cpu_downstream_visualize_no_skips_serialization_and_yes_restores_output(self) -> None:
        if os.environ.get("STUDIO_TEST_URL"):
            self.skipTest("serialization observation requires the in-process server")
        self.load_cpu_downstream_loop()
        self.page.locator("#visualize").select_option("no")
        with patch("object_inpainting_studio.server.serialize_node_output", side_effect=AssertionError("visualize=no must not serialize outputs")) as serializer:
            with self.page.expect_response(lambda response: response.url.endswith("/api/node/preview") and response.request.method == "POST") as submitted:
                self.page.locator('.node[data-node-id="next"] .node-preview-button').click()
            self.assertTrue(submitted.value.ok)
            self.assertFalse(submitted.value.request.post_data_json["visualize"])
            expect(self.page.locator("#batch-status")).to_contain_text("已完成", timeout=15000)
            serializer.assert_not_called()
        job = self.page.evaluate("batch.job")
        self.assertFalse(job["visualize"])
        self.assertEqual(job["required_loop_ids"], ["rgb"])
        self.assertEqual(job["loop_previews"], {})
        self.assertEqual(job["visualization_revision"], 0)
        self.assertTrue(job["results"][0]["accepted"])
        self.assertEqual(job["results"][0]["iterations"], 3)
        self.assertEqual(job["results"][0]["preview"], {})
        self.assertEqual([entry["scores"][0]["value"] for entry in job["iteration_history"]], [0, 0, 255])
        self.assertNotIn("data:image", json.dumps(job))
        self.assertEqual(self.page.evaluate("state.nodeResults"), {})
        for selector in ("#run-visualization", ".multiple-preview", "#node-preview"):
            expect(self.page.locator(selector)).to_be_hidden()
        expect(self.page.locator("body")).to_have_class(re.compile(r"\bvisualization-off\b"))

        previous_id = job["id"]
        self.page.locator("#visualize").select_option("yes")
        expect(self.page.locator("body")).not_to_have_class(re.compile(r"\bvisualization-off\b"))
        with self.page.expect_response(lambda response: response.url.endswith("/api/node/preview") and response.request.method == "POST") as submitted:
            self.page.locator('.node[data-node-id="next"] .node-preview-button').click()
        self.assertTrue(submitted.value.ok)
        self.assertNotEqual(submitted.value.json()["id"], previous_id)
        expect(self.page.locator("#batch-status")).to_contain_text("已完成", timeout=15000)
        self.assertTrue(self.page.evaluate("batch.job.visualize"))
        self.assertEqual(self.page.evaluate("state.nodeResults.next.score.value"), 255)
        self.assert_cpu_loop_image((0, 0, 255))
        self.assertEqual(self.errors, [])

    def test_conditional_retry_save_auto_validates_and_saves_only_accepted_attempt(self) -> None:
        if os.environ.get("STUDIO_TEST_URL"):
            self.skipTest("model substitute requires the in-process server")
        original = self.fixture.root / "conditional-original.png"
        reference = self.fixture.root / "conditional-reference.png"
        output = self.fixture.root / "conditional-output"
        Image.new("RGB", (32, 32), "white").save(original)
        Image.new("RGB", (32, 32), "blue").save(reference)
        workflow = {"nodes": {
            "source": {"class_type": "LoadTargetNode", "inputs": {"path": str(original)}},
            "generate": {"class_type": "FluxKleinReferenceNode", "inputs": {
                "reference_1": {"node": "source", "output": "target"}, "prompt": "test", "seed": 0}},
            "score": {"class_type": "ColorSimilarityNode", "inputs": {
                "image1": {"node": "generate", "output": "image"}, "image2": str(reference)}},
            "save": {"class_type": "SaveImagesNode", "inputs": {
                "img1": {"node": "generate", "output": "image"}, "dir": str(output), "filename": "accepted"}},
        }, "outputs": {}}
        calls = []

        def generate(**inputs):
            calls.append(inputs["seed"])
            return {"image": Image.new("RGB", (32, 32), "blue" if inputs["seed"] == 1 else "red")}

        self.page.evaluate("""workflow => {
            state.workflow = workflow; restoreLoop(); render();
        }""", workflow)
        self.page.locator("#conditional-add").click()
        self.page.locator('.node[data-node-id="generate"] .node-header').click()
        self.page.locator('.node[data-node-id="score"] .node-header').click()
        self.page.locator('.node[data-node-id="score"] .node-header').click()
        self.page.get_by_label("条件 1 运算符", exact=True).select_option("<=")
        threshold = self.page.get_by_label("条件 1 比较值", exact=True)
        threshold.fill("1")
        threshold.press("Tab")
        with patch.object(NODE_CLASS_MAPPINGS["FluxKleinReferenceNode"], "execute", side_effect=generate):
            self.assertEqual(calls, [])
            self.assertFalse(output.exists())
            with self.page.expect_response(lambda response: response.url.endswith("/api/conditional") and response.request.method == "POST") as submitted:
                self.page.locator('.node[data-node-id="save"] .node-preview-button').click()
            self.assertTrue(submitted.value.ok)
            expect(self.page.locator("#batch-status")).to_contain_text("已完成", timeout=15000)
        self.assertEqual(calls, [0, 1])
        expect(self.page.locator("#batch-counts")).to_contain_text("成功 1")
        expect(self.page.locator("#batch-current")).to_contain_text("accepted")
        expect(self.page.locator("#conditional-attempts")).to_contain_text("score.score")
        written = list(output.rglob("*.png"))
        self.assertEqual(len(written), 1)
        with Image.open(written[0]) as image:
            self.assertEqual(image.getpixel((0, 0)), (0, 0, 255))
        self.assertEqual(self.errors, [])

    def test_multiple_loop_wizard_runs_each_region_and_saves_once(self) -> None:
        if os.environ.get("STUDIO_TEST_URL"):
            self.skipTest("model substitute requires the in-process server")
        original = self.fixture.root / "multiple-original.png"
        reference = self.fixture.root / "multiple-reference.png"
        output = self.fixture.root / "multiple-output"
        Image.new("RGB", (32, 32), "white").save(original)
        Image.new("RGB", (32, 32), "blue").save(reference)
        workflow = {"nodes": {
            "source": {"class_type": "LoadTargetNode", "inputs": {"path": str(original)}},
            "generate": {"class_type": "FluxKleinReferenceNode", "inputs": {
                "reference_1": {"node": "source", "output": "target"}, "prompt": "first", "seed": 0}},
            "generate_2": {"class_type": "FluxKleinReferenceNode", "inputs": {
                "reference_1": {"node": "source", "output": "target"}, "prompt": "second", "seed": 10}},
            "score": {"class_type": "ColorSimilarityNode", "inputs": {
                "image1": {"node": "generate", "output": "image"}, "image2": str(reference)}},
            "score_2": {"class_type": "ColorSimilarityNode", "inputs": {
                "image1": {"node": "generate_2", "output": "image"}, "image2": str(reference)}},
            "save": {"class_type": "SaveImagesNode", "inputs": {
                "img1": {"node": "generate", "output": "image"},
                "img2": {"node": "generate_2", "output": "image"},
                "dir": str(output), "filename": "accepted"}},
        }, "outputs": {}}
        calls = []

        def generate(**inputs):
            calls.append((inputs["prompt"], inputs["seed"]))
            return {"image": Image.new("RGB", (32, 32), "blue" if inputs["seed"] % 10 == 1 else "red")}

        self.page.evaluate("""workflow => {
            state.workflow = workflow; restoreLoop(); state.layout = {}; render();
        }""", workflow)
        expect(self.page.locator("#conditional-panel")).not_to_have_attribute("open", "")
        for generator, score in (("generate", "score"), ("generate_2", "score_2")):
            self.page.locator("#conditional-add").click()
            self.page.locator(f'.node[data-node-id="{generator}"] .node-header').click()
            self.assertEqual(self.page.evaluate("conditionalUI.picking.kind"), "end_nodes")
            self.page.locator(f'.node[data-node-id="{score}"] .node-header').click()
            self.assertEqual(self.page.evaluate("conditionalUI.picking.kind"), "condition")
            # A node with one scalar output can be selected by title alone.
            self.page.locator(f'.node[data-node-id="{score}"] .node-header').click()
            self.assertIsNone(self.page.evaluate("conditionalUI.picking"))
        expect(self.page.locator(".multiple-loop-card")).to_have_count(2)
        self.assertEqual(self.page.evaluate("state.workflow.conditional_loop.loops.map(l => [l.start_nodes, l.end_nodes, l.conditions[0].node])"), [
            [["generate"], ["score"], "score"], [["generate_2"], ["score_2"], "score_2"]
        ])
        self.assertEqual(self.page.evaluate("conditionalUI.targetId"), "save")
        second = self.page.locator(".multiple-loop-card").nth(1)
        threshold = second.get_by_label("条件 1 比较值", exact=True)
        threshold.focus()
        # A progress-only render must not replace the focused input.
        self.page.evaluate("window.focusedLoopInput = document.activeElement; renderBatch()")
        self.assertTrue(self.page.evaluate("window.focusedLoopInput === document.activeElement"))
        threshold.press("Tab")
        loop_names = self.page.evaluate("multipleLoops().map(item => item.name)")
        expect(self.page.locator("#conditional-error")).to_have_text("")
        with patch.object(NODE_CLASS_MAPPINGS["FluxKleinReferenceNode"], "execute", side_effect=generate):
            self.assertFalse(output.exists())
            self.assertEqual(calls, [])
            with self.page.expect_response(lambda response: response.url.endswith("/api/conditional") and response.request.method == "POST") as submitted:
                # The large wizard graph can place this button behind the sticky toolbar.
                # Invoke the same handler rather than bypassing it with an API request.
                self.page.evaluate("previewNode('save')")
            self.assertTrue(submitted.value.ok)
            expect(self.page.locator("#batch-status")).to_contain_text("已完成", timeout=15000)
        self.assertEqual(calls, [("first", 0), ("first", 1), ("second", 10), ("second", 11)])
        self.assertEqual(self.page.evaluate("batch.job.results[0].loops.map(l => l.iterations)"), [2, 2])
        for name in loop_names:
            expect(self.page.locator("#conditional-attempts")).to_contain_text(name)
        written = list(output.rglob("*.png"))
        self.assertEqual(len(written), 2)
        for path in written:
            with Image.open(path) as image:
                self.assertEqual(image.getpixel((0, 0)), (0, 0, 255))

        with self.page.expect_download() as downloaded:
            self.page.locator("#save-workflow").click()
        downloaded_path = self.fixture.root / "multiple-workflow.json"
        downloaded.value.save_as(downloaded_path)
        exported = json.loads(downloaded_path.read_text(encoding="utf-8"))
        self.assertEqual(len(exported["workflow"]["conditional_loop"]["loops"]), 2)
        self.page.locator("#load-workflow").set_input_files(downloaded_path)
        expect(self.page.locator(".multiple-loop-card")).to_have_count(2)
        self.assertEqual(self.page.evaluate("state.workflow.conditional_loop"), exported["workflow"]["conditional_loop"])
        self.assertEqual(self.page.evaluate("conditionalUI.targetId"), "save")
        second_id = exported["workflow"]["conditional_loop"]["loops"][1]["id"]
        self.page.locator(".multiple-loop-card").first.get_by_role("button", name="删除循环", exact=True).click()
        self.assertEqual(self.page.evaluate("state.workflow.conditional_loop.loops[0].id"), second_id)
        self.page.locator(".multiple-loop-card").first.get_by_role("button", name="删除循环", exact=True).click()
        self.assertFalse(self.page.evaluate("Object.hasOwn(state.workflow, 'conditional_loop')"))
        self.assertEqual(self.page.evaluate("state.workflow.nodes.save.inputs.img2.node"), "generate_2")
        self.assertEqual(self.errors, [])

    def test_conditional_canvas_picker_selects_exact_instances_and_ports(self) -> None:
        self.page.evaluate("""() => {
            state.workflow = {nodes: {
                original: {class_type: 'LoadTargetNode', inputs: {path: ''}},
                generate: {class_type: 'FluxKleinReferenceNode', inputs: {reference_1: link('original', 'target'), seed: 0}},
                generate_2: {class_type: 'FluxKleinReferenceNode', inputs: {reference_1: link('original', 'target'), seed: 7}},
                score: {class_type: 'ColorSimilarityNode', inputs: {image1: link('generate', 'image'), image2: link('original', 'target')}},
                score_2: {class_type: 'ColorSimilarityNode', inputs: {image1: link('generate_2', 'image'), image2: link('original', 'target')}},
                save: {class_type: 'SaveImagesNode', inputs: {img1: link('generate', 'image'), dir: '/unused'}},
                save_2: {class_type: 'SaveImagesNode', inputs: {img1: link('generate_2', 'image'), dir: '/unused2'}}
            }, outputs: {}, conditional_loop: {
                name: '原有换色循环', start_nodes: ['generate_2'], seed_nodes: ['generate_2'],
                conditions: [{node: 'score_2', output: 'score', operator: '<=', value: 10}],
                max_iterations: 12, timeout_seconds: 45, mode: 'all', on_exhausted: 'skip'
            }};
            restoreLoop({conditionalTargetId: 'save_2'}); state.layout = {}; state.selected = null;
            state.previewSession = 'historical-session';
            state.nodeResults = {generate_2: {image: {kind: 'value', value: 'historical'}}};
            render();
        }""")
        self.page.locator("#conditional-panel").evaluate("element => element.open = true")
        expect(self.page.locator("#conditional-legacy, #conditional-enabled, #conditional-fields, #conditional-check, #conditional-run, #conditional-plan, #conditional-target")).to_have_count(0)
        expect(self.page.get_by_label("循环名称", exact=True)).to_have_count(0)
        expect(self.page.locator(".multiple-loop-card details")).to_have_count(0)
        expect(self.page.locator(".multiple-loop-card")).to_have_count(1)
        self.assertEqual(self.page.evaluate("state.workflow.conditional_loop.loops[0].name"), "原有换色循环")
        self.assertEqual(self.page.evaluate("state.workflow.conditional_loop.loops[0].max_iterations"), 12)
        self.assertEqual(self.page.evaluate("state.workflow.conditional_loop.loops[0].timeout_seconds"), 45)
        self.assertEqual(self.page.evaluate("state.workflow.conditional_loop.loops[0].on_exhausted"), "skip")

        def title(node_id: str):
            return self.page.locator(f'.node[data-node-id="{node_id}"] .node-header')

        expect(self.page.get_by_role("button", name="画布选择最终保存目标", exact=True)).to_have_count(0)
        self.assertEqual(self.page.evaluate("conditionalUI.targetId"), "save_2")

        self.page.get_by_role("button", name="重选起点", exact=True).click()
        title("generate_2").click()
        self.assertEqual(self.page.evaluate("state.workflow.conditional_loop.loops[0].start_nodes"), ["generate_2"])
        self.assertEqual(self.page.evaluate("state.workflow.conditional_loop.loops[0].seed_nodes"), ["generate_2"])
        self.page.get_by_label("条件 1 运算符", exact=True).select_option("<=")
        threshold = self.page.get_by_label("条件 1 比较值", exact=True)
        threshold.fill("10")
        threshold.press("Tab")
        bindings = self.page.evaluate("JSON.stringify(Object.fromEntries(Object.entries(state.workflow.nodes).map(([id, n]) => [id, n.inputs])))")
        self.page.evaluate("state.connecting = {node: 'original', output: 'target'}")
        self.page.get_by_role("button", name="重选条件", exact=True).click()
        self.page.locator('.port.output[data-node-id="generate_2"][data-port="image"]').click()
        self.assertIsNotNone(self.page.evaluate("conditionalUI.picking"))
        self.page.locator('.port.output[data-node-id="score_2"][data-port="score"]').click()
        self.assertEqual(self.page.evaluate("state.workflow.conditional_loop.loops[0].conditions"), [
            {"node": "score_2", "output": "score", "operator": "<=", "value": 10}
        ])
        self.assertIsNone(self.page.evaluate("state.connecting"))
        self.assertEqual(self.page.evaluate("JSON.stringify(Object.fromEntries(Object.entries(state.workflow.nodes).map(([id, n]) => [id, n.inputs])))"), bindings)
        expect(title("generate_2").locator(".conditional-role")).to_contain_text("起点")
        expect(title("score_2").locator(".conditional-role")).to_contain_text("条件")
        expect(title("save_2").locator(".conditional-role")).to_contain_text("最终保存")
        expect(self.page.locator("#conditional-error")).to_have_text("")
        self.assertEqual(self.page.evaluate("[...multipleBody(state.workflow.conditional_loop.loops[0])]"), ["generate_2", "score_2"])

        self.page.get_by_role("button", name=re.compile(r"^终点 · score_2 ·")).click()
        self.assertEqual(self.page.evaluate("state.selected"), "score_2")
        expect(self.page.locator('.node[data-node-id="score_2"]')).to_have_class(re.compile(r"\bselected\b"))
        self.assertEqual(self.page.evaluate("state.previewSession"), "historical-session")
        self.assertTrue(self.page.evaluate("Boolean(state.nodeResults.generate_2)"))
        self.page.evaluate("""() => {
            const workflow = JSON.parse(JSON.stringify(state.workflow));
            const settings = JSON.parse(JSON.stringify(loopSettings()));
            state.workflow = workflow; restoreLoop(settings); render();
        }""")
        self.assertEqual(self.page.evaluate("conditionalUI.targetId"), "save_2")
        self.assertIsNone(self.page.evaluate("conditionalUI.picking"))
        self.assertEqual(self.page.evaluate("state.workflow.conditional_loop.loops[0].conditions[0].node"), "score_2")
        self.assertEqual(self.errors, [])

    def test_delete_node_button_cleans_links_outputs_and_preview(self) -> None:
        self.page.evaluate("""() => {
            state.workflow = {
                nodes: {
                    first: {class_type: 'BuildPromptNode', inputs: {object_name: 'cup'}},
                    second: {class_type: 'BuildPromptNode', inputs: {object_name: {node: 'first', output: 'prompt'}}}
                },
                outputs: {prompt: {node: 'first', output: 'prompt'}}
            };
            state.selected = 'first';
            state.connecting = {node: 'first', output: 'prompt'};
            state.nodeResults = {first: {prompt: {kind: 'value', value: 'old output'}}};
            restoreLoop(); render();
        }""")
        self.page.locator('.node[data-node-id="first"] .node-delete-button').click()
        expect(self.page.locator('.node[data-node-id="first"]')).to_have_count(0)
        expect(self.page.locator('.node[data-node-id="second"]')).to_have_count(1)
        expect(self.page.locator("#status")).to_contain_text("已删除节点 first")
        expect(self.page.locator("#status")).to_contain_text("second.object_name")
        self.assertEqual(self.page.evaluate("state.workflow.nodes.second.inputs.object_name"), None)
        self.assertEqual(self.page.evaluate("state.workflow.outputs"), {})
        self.assertEqual(self.page.evaluate("state.nodeResults"), {})
        self.assertEqual(self.page.evaluate("state.connecting"), None)
        self.assertEqual(self.errors, [])

    def test_delete_shortcut_ignores_input_and_blocks_active_tasks(self) -> None:
        self.page.locator('[data-starter="BuildPromptNode"]').click()
        node_id = self.page.locator(".node").get_attribute("data-node-id")
        field = self.page.locator('.node input[data-field="object_name"]')
        field.focus()
        self.page.keyboard.press("Delete")
        expect(self.page.locator(".node")).to_have_count(1)
        self.page.evaluate("document.activeElement.blur(); loopUI.busy = true; renderBatch();")
        expect(self.page.locator(".node .node-delete-button")).to_be_disabled()
        self.page.keyboard.press("Delete")
        expect(self.page.locator(".node")).to_have_count(1)
        self.page.evaluate("loopUI.busy = false; renderBatch();")
        self.page.keyboard.press("Delete")
        expect(self.page.locator(".node")).to_have_count(0)
        expect(self.page.locator("#status")).to_contain_text(f"已删除节点 {node_id}")
        self.assertEqual(self.errors, [])

    def test_upstream_error_marks_original_node_and_clears_after_retry(self) -> None:
        image_path = self.fixture.root / "error-colors.png"
        mask_path = self.fixture.root / "error-mask.png"
        Image.new("RGB", (32, 24), "red").save(image_path)
        Image.new("L", (32, 24), 255).save(mask_path)
        workflow = {
            "nodes": {
                "colors": {
                    "class_type": "AutoColorRegionsNode",
                    "inputs": {"image": str(image_path), "mask": str(mask_path), "detail": 3},
                },
                "select": {
                    "class_type": "SelectColorRegionNode",
                    "inputs": {"regions": {"node": "colors", "output": "regions"}},
                },
            },
            "outputs": {},
        }
        path = self.fixture.root / "error-workflow.json"
        path.write_text(json.dumps({"workflow": workflow}))
        self.page.locator("#load-workflow").set_input_files(path)
        expect(self.page.locator("#status")).to_contain_text("Imported workflow")
        self.page.locator('.node[data-node-id="select"] .node-preview-button').click()
        expect(self.page.locator("#status")).to_contain_text("请求节点 select")
        expect(self.page.locator("#status")).to_contain_text("colors (AutoColorRegionsNode)")
        expect(self.page.locator('.node[data-node-id="colors"]')).to_have_class(re.compile("node-failed"))
        expect(self.page.locator('.node[data-node-id="select"]')).not_to_have_class(re.compile("node-failed"))
        expect(self.page.locator("#node-preview .node-error")).to_contain_text("detail 必须")
        detail = self.page.locator('.node[data-node-id="colors"] input[data-field="detail"]')
        detail.fill("1")
        detail.press("Enter")
        self.page.locator('.node[data-node-id="select"] .node-preview-button').click()
        expect(self.page.locator("#status")).to_contain_text("测试成功")
        expect(self.page.locator(".node-failed")).to_have_count(0)
        expect(self.page.locator(".node-error")).to_have_count(0)
        expect(self.page.locator('#node-preview img[alt="mask"]')).to_have_count(1)
        self.assertEqual(self.errors, [])

    def test_swatch_dynamic_outputs_connect_score_save_and_shrink(self) -> None:
        root = self.fixture.root
        image_path, mask_path, reference_path = root / "dynamic-colors.png", root / "dynamic-mask.png", root / "red.png"
        output = root / "dynamic-swatches"
        image = Image.new("RGB", (16, 16), "red")
        image.paste("blue", (0, 0, 4, 16))
        image.paste("lime", (0, 0, 4, 4))
        image.save(image_path)
        Image.new("L", image.size, 255).save(mask_path)
        Image.new("RGB", (32, 32), "red").save(reference_path)
        workflow = {"nodes": {
            "regions": {"class_type": "AutoColorRegionsNode", "inputs": {"image": str(image_path), "mask": str(mask_path)}},
            "swatches": {"class_type": "ColorRegionSwatchesNode", "inputs": {
                "regions": {"node": "regions", "output": "regions"}, "indices": [0]}},
            "score": {"class_type": "ColorSimilarityNode", "inputs": {
                "image1": {"node": "swatches", "output": "img1"}, "image2": str(reference_path)}},
            "save": {"class_type": "SaveImagesNode", "inputs": {
                "img1": {"node": "swatches", "output": "img1"}, "dir": str(output)}},
        }, "outputs": {}}
        saved = root / "dynamic-swatches.json"
        saved.write_text(json.dumps({"workflow": workflow}))
        self.page.locator("#load-workflow").set_input_files(saved)
        expect(self.page.locator("#status")).to_contain_text("Imported workflow")
        card = self.page.locator('.node[data-node-id="swatches"]')
        expect(card.locator('.port.output[data-port^="img"]')).to_have_count(1)
        expect(card.locator('.port.output[data-port="images"]')).to_have_count(0)
        indices = card.locator('input[data-field="indices"]')
        indices.fill("[0,2,1]")
        indices.press("Enter")
        expect(card.locator('.port.output[data-port^="img"]')).to_have_count(3)
        card.locator('.port.output[data-port="img2"]').click()
        self.page.locator('.node[data-node-id="save"] .port.input[data-port="img2"]').click()
        self.assertEqual(self.page.evaluate("state.workflow.nodes.save.inputs.img2"), {"node": "swatches", "output": "img2"})
        card.locator(".node-preview-button").click()
        expect(self.page.locator("#status")).to_contain_text("测试成功：swatches")
        expect(self.page.locator('#node-preview img[alt^="img"]')).to_have_count(3)
        self.page.locator('.node[data-node-id="score"] .node-preview-button').click()
        expect(self.page.locator("#status")).to_contain_text("测试成功：score")
        self.assertEqual(self.page.evaluate("state.nodeResults.score.score.value"), 0.0)
        self.page.locator('.node[data-node-id="save"] .node-preview-button').click()
        expect(self.page.locator("#status")).to_contain_text("保存成功")
        self.assertEqual(len(list(output.rglob("*.png"))), 2)
        for folder, color in (("img1", (255, 0, 0)), ("img2", (0, 255, 0))):
            with Image.open(next((output / folder).glob("*.png"))) as image:
                self.assertEqual(image.getcolors(), [(1024, color)])
        session = self.page.evaluate("state.previewSession")
        indices.fill("[0]")
        indices.press("Enter")
        expect(card.locator('.port.output[data-port^="img"]')).to_have_count(1)
        self.assertIsNone(self.page.evaluate("state.workflow.nodes.save.inputs.img2"))
        self.assertEqual(self.page.evaluate("state.workflow.nodes.save.inputs.img1.output"), "img1")
        self.assertEqual(self.page.evaluate("state.previewSession"), session)
        self.assertEqual(self.page.evaluate("Object.keys(state.nodeResults.swatches).sort()"), ["colors", "img1", "img2", "img3"])
        expect(card.locator(".preview-state")).to_contain_text("历史预览")
        self.assertEqual(self.errors, [])

    def test_color_region_swatches_list_input_preview_and_separate_save(self) -> None:
        image_path = self.fixture.root / "swatch-image.png"
        mask_path = self.fixture.root / "swatch-mask.png"
        output = self.fixture.root / "swatch-output"
        image = Image.new("RGB", (16, 16), "red")
        image.paste("blue", (0, 0, 4, 16))
        image.save(image_path)
        Image.new("L", image.size, 255).save(mask_path)
        workflow = {
            "nodes": {
                "regions": {"class_type": "AutoColorRegionsNode", "inputs": {"image": str(image_path), "mask": str(mask_path)}},
                "swatches": {"class_type": "ColorRegionSwatchesNode", "inputs": {"regions": {"node": "regions", "output": "regions"}}},
                "save": {"class_type": "SaveImagesNode", "inputs": {"images": {"node": "swatches", "output": "images"}, "dir": str(output)}},
            },
            "outputs": {},
        }
        path = self.fixture.root / "swatch-workflow.json"
        path.write_text(json.dumps({"workflow": workflow}))
        self.page.locator("#load-workflow").set_input_files(path)
        expect(self.page.locator("#status")).to_contain_text("Imported workflow")
        indices = self.page.locator('.node[data-node-id="swatches"] input[data-field="indices"]')
        expect(indices).to_have_value("[0]")
        indices.fill("[0,1]")
        indices.press("Enter")
        self.assertEqual(self.page.evaluate("state.workflow.nodes.swatches.inputs.indices"), [0, 1])
        self.page.locator('.node[data-node-id="swatches"] .node-preview-button').click()
        # Legacy consumers retain their list alongside the new individual ports.
        expect(self.page.locator('#node-preview img[alt^="img"]')).to_have_count(2)
        expect(self.page.locator("#node-preview .composite-output img")).to_have_count(2)
        self.assertTrue(self.page.evaluate("state.workflow.nodes.swatches.legacy_swatch_images"))
        expect(self.page.locator("#node-preview")).to_contain_text("32 × 32 RGB")
        self.assertFalse(output.exists())
        self.page.locator('.node[data-node-id="save"] .node-preview-button').click()
        expect(self.page.locator("#status")).to_contain_text("保存成功")
        self.assertEqual(len(list(output.rglob("*.png"))), 2)
        for folder, color in (("images_0", (255, 0, 0)), ("images_1", (0, 0, 255))):
            with Image.open(next((output / folder).glob("*.png"))) as saved:
                self.assertEqual((saved.size, saved.mode), ((32, 32), "RGB"))
                self.assertEqual(saved.getcolors(), [(1024, color)])
        self.assertEqual(self.errors, [])

    def test_auto_color_regions_upload_gallery_and_invalid_detail(self) -> None:
        image = Image.new("RGB", (32, 24), "red")
        image.paste("blue", (16, 0, 32, 24))
        image.save(self.fixture.root / "colors.png")
        Image.new("L", image.size, 255).save(self.fixture.root / "colors-mask.png")
        self.page.locator('[data-starter="AutoColorRegionsNode"]').click()
        expect(self.page.locator("#node-help")).to_contain_text("不需要填写主色数量")
        self.page.locator('.node input[data-upload="image"]').set_input_files(self.fixture.root / "colors.png")
        self.page.locator('.node input[data-upload="mask"]').set_input_files(self.fixture.root / "colors-mask.png")
        detail = self.page.locator('.node input[data-field="detail"]')
        expect(detail).to_have_value("1")
        self.page.locator(".node-preview-button").click()
        gallery = self.page.locator("#node-preview .color-region-view")
        expect(gallery).to_contain_text("2 个颜色区域")
        expect(gallery.locator(".color-region-card")).to_have_count(2)
        expect(gallery.locator(".region-swatch")).to_have_count(2)
        expect(gallery.locator(".color-region-card").first).to_contain_text("50.00%")
        expect(gallery.locator("img.alpha-preview")).to_have_count(2)
        expect(self.page.locator(".node .color-region-card")).to_have_count(0)
        expect(self.page.locator("#node-preview .label-map-preview")).not_to_have_attribute("open", "")
        gallery.locator("img.alpha-preview").first.click()
        expect(self.page.locator("#image-dialog img")).to_have_class("alpha-preview")
        self.page.locator("#close-image-dialog").click()
        detail.fill("3")
        detail.press("Enter")
        expect(self.page.locator("#node-preview .color-region-view")).to_have_count(0)
        self.page.locator(".node-preview-button").click()
        expect(self.page.locator("#status")).to_contain_text("detail 必须")
        self.assertEqual(self.errors, [])

    def test_color_region_selection_and_label_map_save_connection(self) -> None:
        image = Image.new("RGB", (32, 24), "red")
        image.paste("blue", (16, 0, 32, 24))
        image.save(self.fixture.root / "colors.png")
        Image.new("L", image.size, 255).save(self.fixture.root / "colors-mask.png")
        output = self.fixture.root / "color-output"
        workflow = {
            "nodes": {
                "colors": {
                    "class_type": "AutoColorRegionsNode",
                    "inputs": {
                        "image": str(self.fixture.root / "colors.png"),
                        "mask": str(self.fixture.root / "colors-mask.png"),
                    },
                },
                "select": {
                    "class_type": "SelectColorRegionNode",
                    "inputs": {"regions": {"node": "colors", "output": "regions"}, "region_id": 2},
                },
                "save": {"class_type": "SaveImagesNode", "inputs": {"dir": str(output)}},
            },
            "outputs": {},
        }
        path = self.fixture.root / "color-workflow.json"
        path.write_text(
            json.dumps(
                {
                    "workflow": workflow,
                    "layout": {
                        "colors": {"x": 20, "y": 30},
                        "select": {"x": 360, "y": 30},
                        "save": {"x": 700, "y": 30},
                    },
                }
            )
        )
        self.page.locator("#load-workflow").set_input_files(path)
        expect(self.page.locator("#status")).to_contain_text("Imported workflow")
        self.page.locator('.node[data-node-id="select"] .node-preview-button').click()
        expect(self.page.locator("#node-preview img.alpha-preview")).to_have_count(1)
        expect(self.page.locator("#node-preview")).to_contain_text("32 × 24 RGBA")
        self.page.locator('.port.output[data-node-id="colors"][data-port="labels"]').click()
        self.page.locator('.port.input[data-node-id="save"][data-port="img3"]').click()
        self.page.locator('.node[data-node-id="save"] .node-preview-button').click()
        expect(self.page.locator("#status")).to_contain_text("保存成功")
        saved = list((output / "img3").glob("*.png"))
        self.assertEqual(len(saved), 1)
        with Image.open(saved[0]) as labels:
            self.assertEqual(labels.size, image.size)
            self.assertEqual({value for value, count in enumerate(labels.histogram()) if count}, {1, 2})
        self.assertEqual(self.errors, [])

    def test_random_recolor_starter_uploads_and_previews_source_metadata(self) -> None:
        self.page.locator('[data-starter="FluxRandomRecolorNode"]').click()
        expect(self.page.locator("#node-help")).to_contain_text("生成 source")
        expect(self.page.locator('.node input[data-field="seed"]')).to_have_value("-1")
        expect(self.page.locator('.node input[data-field="num_inference_steps"]')).to_have_value("28")
        self.page.locator('.node input[data-upload="target"]').set_input_files(self.fixture.root / "target.png")
        with (
            Image.open(self.fixture.root / "target.png") as target,
            Image.open(self.fixture.root / "mask.png") as mask,
        ):
            gray = GrayMaskNode().execute(target.convert("RGB"), mask.convert("L"))["source"]
            gray.save(self.fixture.root / "gray.png")
        self.page.locator('.node input[data-upload="mask"]').set_input_files(self.fixture.root / "gray.png")
        seed = self.page.locator('.node input[data-field="seed"]')
        seed.fill("42")
        seed.press("Enter")
        calls = []

        def generate(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(images=[Image.new("RGB", (kwargs["width"], kwargs["height"]), "blue")])

        with patch("object_inpainting.nodes.flux_klein_reference.get_flux_klein", return_value=generate):
            self.page.locator(".node-preview-button").click()
            expect(self.page.locator('#node-preview img[alt="source"]')).to_have_count(1)
            expect(self.page.locator('#node-preview img[alt="masked_source"]')).to_have_count(1)
            expect(self.page.locator('#node-preview img[alt="sampled_color"]')).to_have_count(1)
            expect(self.page.locator("#node-preview")).to_contain_text("actual_seed")
            expect(self.page.locator("#node-preview")).to_contain_text("42")
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0]["image"]), 3)
        self.assertEqual(self.errors, [])

    def test_graymask_output_connects_directly_to_random_recolor_mask(self) -> None:
        workflow = {
            "nodes": {
                "gray": {
                    "class_type": "GrayMaskNode",
                    "inputs": {
                        "target": str(self.fixture.root / "target.png"),
                        "edit_mask": str(self.fixture.root / "mask.png"),
                    },
                },
                "recolor": {
                    "class_type": "FluxRandomRecolorNode",
                    "inputs": {
                        "target": str(self.fixture.root / "target.png"),
                        "mask": {"node": "gray", "output": "source"},
                        "seed": 42,
                    },
                },
            },
            "outputs": {},
        }
        path = self.fixture.root / "gray-recolor-workflow.json"
        path.write_text(
            json.dumps({"workflow": workflow, "layout": {"gray": {"x": 20, "y": 30}, "recolor": {"x": 360, "y": 30}}})
        )
        self.page.locator("#load-workflow").set_input_files(path)
        expect(self.page.locator("#status")).to_contain_text("Imported workflow")
        self.page.locator('.port.output[data-node-id="gray"][data-port="source"]').click()
        self.page.locator('.port.input[data-node-id="recolor"][data-port="mask"]').click()
        self.page.locator("#validate-workflow").click()
        expect(self.page.locator("#status")).to_contain_text('"valid": true')

        def generate(**kwargs):
            return SimpleNamespace(images=[Image.new("RGB", (kwargs["width"], kwargs["height"]), "blue")])

        with patch("object_inpainting.nodes.flux_klein_reference.get_flux_klein", return_value=generate):
            self.page.locator('.node[data-node-id="recolor"] .node-preview-button').click()
            expect(self.page.locator('#node-preview img[alt="source"]')).to_have_count(1)
        self.assertEqual(self.errors, [])

    def configure_sample(self):
        self.page.locator("#target-path").fill(str(self.fixture.root / "target.png"))
        self.page.locator("#mask-path").fill(str(self.fixture.root / "mask.png"))
        self.page.locator("#object-name").fill("cup")
        self.page.locator("#config-resolution").fill("64")
        self.page.locator("#config-reference_resolution").fill("32")
        self.page.locator("#config-validation_fraction").fill("0")

    def test_flux_generation_prompt_controls_and_empty_references_validate(self):
        self.page.locator('[data-starter="FluxKleinReferenceNode"]').click()
        expect(self.page.locator("#node-help")).to_contain_text("实际调用")
        prompt = self.page.locator('.node input[data-field="prompt"]')
        prompt.fill("A coat on a white background")
        prompt.press("Enter")
        for name, value in (
            ("width", "1024"),
            ("height", "1024"),
            ("num_inference_steps", "28"),
            ("guidance_scale", "4"),
            ("seed", "0"),
        ):
            expect(self.page.locator(f'.node input[data-field="{name}"]')).to_have_value(value)
        for name in ("reference_1", "reference_2", "reference_3"):
            expect(self.page.locator(f'.node input[data-field="{name}"]')).to_have_value("")
            expect(self.page.locator(f'.node input[data-upload="{name}"]')).to_have_count(1)
        width = self.page.locator('.node input[data-field="width"]')
        width.fill("512")
        width.press("Enter")
        self.page.locator("#validate-workflow").click()
        expect(self.page.locator("#status")).to_contain_text('"valid": true')
        self.assertEqual(self.errors, [])

    def test_initialization_drag_connections_and_validation(self):
        expect(self.page.locator(".node")).to_have_count(14)
        expect(self.page.locator(".catalog-button")).to_have_count(len(NODE_CLASS_MAPPINGS))
        header = self.page.locator('.node[data-node-id="record"] .node-header')
        box = header.bounding_box()
        self.page.mouse.move(box["x"] + 30, box["y"] + 15)
        self.page.mouse.down()
        self.page.mouse.move(box["x"] + 110, box["y"] + 55, steps=8)
        self.page.mouse.up()
        moved = header.bounding_box()
        self.assertGreater(moved["x"], box["x"] + 50)
        self.page.locator('.port.output[data-node-id="record"][data-port="object_name"]').click()
        self.page.locator('.port.input[data-node-id="prompt"][data-port="object_name"]').click()
        self.page.locator("#validate-workflow").click()
        expect(self.page.locator("#status")).to_contain_text('"valid": true')
        self.assertEqual(self.errors, [])

    def test_save_images_button_writes_only_populated_port_folder(self):
        self.page.locator('[data-starter="SaveImagesNode"]').click()
        expect(self.page.locator("#node-help")).to_contain_text("同一组")
        directory = self.fixture.root / "saved-images"
        field = self.page.locator('.node input[data-field="dir"]')
        field.fill(str(directory))
        field.press("Enter")
        expect(self.page.locator('.node input[data-upload="dir"]')).to_have_count(0)
        self.page.locator('.node input[data-upload="img3"]').set_input_files(self.fixture.root / "target.png")
        expect(self.page.locator('.node input[data-field="img3"]')).to_have_value(re.compile(".*studio-uploads-.*"))
        self.page.locator(".node-preview-button").click()
        expect(self.page.locator("#status")).to_contain_text("保存成功")
        self.assertEqual(len(list((directory / "img3").glob("*.png"))), 1)
        self.assertFalse((directory / "img1").exists())
        self.assertEqual(self.errors, [])

    def test_dynamic_save_items_name_upload_delete_and_workflow_round_trip(self) -> None:
        self.page.locator('[data-starter="SaveImagesNode"]').click()
        add = self.page.locator(".node .save-item-add")
        expect(add).to_be_enabled()
        for _ in range(4):
            add.click()
        expect(self.page.locator(".node input[data-save-name]")).to_have_count(8)
        name = self.page.locator('.node input[data-save-name="img8"]')
        name.fill("透明参考图")
        name.press("Enter")
        self.page.get_by_role("button", name="移除保存项 img2", exact=True).click()
        add.click()
        expect(self.page.locator('.node input[data-save-name="img2"]')).to_have_count(0)
        expect(self.page.locator('.node input[data-save-name="img9"]')).to_have_count(1)
        directory = self.fixture.root / "dynamic-save"
        field = self.page.locator('.node input[data-field="dir"]')
        field.fill(str(directory))
        field.press("Enter")
        filename = self.page.locator('.node input[data-field="filename"]')
        filename.fill("sample.jpg")
        filename.press("Enter")
        self.page.locator('.node input[data-upload="img8"]').set_input_files(self.fixture.root / "target.png")
        expect(self.page.locator('.node input[data-field="img8"]')).to_have_value(re.compile(".*studio-uploads-.*"))
        self.page.locator(".node-preview-button").click()
        expect(self.page.locator("#status")).to_contain_text("保存成功")
        self.assertTrue((directory / "透明参考图" / "sample.png").is_file())
        self.assertFalse((directory / "img1").exists())
        with self.page.expect_download() as event:
            self.page.locator("#save-workflow").click()
        saved = self.fixture.root / "dynamic-workflow.json"
        event.value.save_as(saved)
        spec = next(iter(json.loads(saved.read_text())["workflow"]["nodes"].values()))
        self.assertEqual(spec["save_next_id"], 10)
        self.assertNotIn("img2", spec["inputs"])
        self.assertEqual(next(item["name"] for item in spec["save_items"] if item["id"] == "img8"), "透明参考图")
        self.page.locator("#new-workflow").click()
        self.page.locator("#load-workflow").set_input_files(saved)
        expect(self.page.locator("#status")).to_contain_text("Imported workflow")
        expect(self.page.locator('.node input[data-save-name="img8"]')).to_have_value("透明参考图")
        expect(self.page.locator('.node input[data-save-name="img2"]')).to_have_count(0)
        self.page.locator(".node .save-item-add").click()
        expect(self.page.locator('.node input[data-save-name="img10"]')).to_have_count(1)
        self.assertEqual(self.errors, [])

    def test_folder_batch_runs_and_downloads_source_to_saved_path_report(self):
        self.page.locator('[data-starter="FolderImagesNode"]').click()
        expect(self.page.locator("#node-help")).to_contain_text("文件夹")
        expect(self.page.locator('.node input[data-upload="dir"]')).to_have_count(0)
        expect(self.page.locator('.node input[data-field="index"]')).to_have_value("0")
        expect(self.page.locator('.node input[data-field="recursive"]')).to_have_value("0")
        directory = self.fixture.root / "batch-input"
        directory.mkdir()
        for name in ("a.png", "c.png"):
            (directory / name).write_bytes((self.fixture.root / "target.png").read_bytes())
        (directory / "b.png").write_bytes(b"broken image")
        output = self.fixture.root / "batch-output"
        workflow = {
            "nodes": {
                "folder": {"class_type": "FolderImagesNode", "inputs": {"dir": str(directory)}},
                "save": {
                    "class_type": "SaveImagesNode",
                    "inputs": {"img2": {"node": "folder", "output": "image"}, "dir": str(output)},
                },
            },
            "outputs": {},
        }
        path = self.fixture.root / "batch-workflow.json"
        path.write_text(
            json.dumps({"workflow": workflow, "layout": {"folder": {"x": 30, "y": 30}, "save": {"x": 400, "y": 30}}})
        )
        self.page.locator("#load-workflow").set_input_files(path)
        expect(self.page.locator("#status")).to_contain_text("Imported workflow")
        self.page.locator("#batch-continue-on-error").check()
        self.page.locator(".node-batch-button").click()
        expect(self.page.locator("#loop-summary")).to_contain_text("3 张输入")
        self.page.locator("#loop-run").click()
        expect(self.page.locator("#batch-status")).to_contain_text("已完成")
        expect(self.page.locator("#batch-counts")).to_contain_text("已处理 3 / 3 · 成功 2 · 失败 1")
        self.assertEqual(len(list((output / "img2").glob("*.png"))), 2)
        self.assertFalse((output / "img1").exists())
        with self.page.expect_download() as event:
            self.page.locator("#batch-download").click()
        report = self.fixture.root / "batch-report.json"
        event.value.save_as(report)
        job = json.loads(report.read_text())
        self.assertEqual(len(job["results"]), 3)
        self.assertEqual(job["results"][0]["source"], str(directory / "a.png"))
        self.assertTrue(Path(job["results"][0]["paths"]["img2"]).is_file())
        self.assertIn("error", job["results"][1])
        self.page.reload()
        expect(self.page.locator("#batch-status")).to_contain_text(job["id"])
        expect(self.page.locator("#batch-counts")).to_contain_text("已处理 3 / 3")
        expect(self.page.locator("#batch-cancel")).to_be_disabled()
        self.assertEqual(self.errors, [])

    def import_loop_workflow(self, extra_nodes=None, extra_inputs=None):
        directory = self.fixture.root / "loop-input"
        directory.mkdir(exist_ok=True)
        for name in ("a.png", "b.png"):
            (directory / name).write_bytes((self.fixture.root / "target.png").read_bytes())
        workflow = {
            "nodes": {
                "folder": {"class_type": "FolderImagesNode", "inputs": {"dir": str(directory)}},
                "save": {
                    "class_type": "SaveImagesNode",
                    "inputs": {
                        "img1": {"node": "folder", "output": "image"},
                        "dir": str(self.fixture.root / "loop-output"),
                        **(extra_inputs or {}),
                    },
                },
                **(extra_nodes or {}),
            },
            "outputs": {},
        }
        path = self.fixture.root / "loop-workflow.json"
        path.write_text(
            json.dumps({"workflow": workflow, "layout": {"folder": {"x": 50, "y": 60}, "save": {"x": 410, "y": 60}}})
        )
        self.page.locator("#load-workflow").set_input_files(path)
        expect(self.page.locator("#status")).to_contain_text("Imported workflow")
        return workflow

    def enable_loop(self):
        self.page.locator('.node[data-node-id="folder"] [data-loop-mode="each"]').click()
        expect(self.page.locator("#loop-summary")).to_contain_text("2 张输入")
        expect(self.page.locator("#loop-run")).to_be_enabled()

    def test_loop_mode_highlights_only_repeated_path_and_fixed_inputs(self):
        self.import_loop_workflow(
            extra_nodes={
                "fixed": {"class_type": "LoadTargetNode", "inputs": {"path": str(self.fixture.root / "target.png")}},
                "unused": {"class_type": "BuildPromptNode", "inputs": {"object_name": "cup"}},
            },
            extra_inputs={"img3": {"node": "fixed", "output": "target"}},
        )
        expect(self.page.locator("#loop-run")).to_be_disabled()
        self.enable_loop()
        expect(self.page.locator(".node.loop-member")).to_have_count(2)
        expect(self.page.locator('.node[data-node-id="fixed"]')).to_have_class(re.compile("loop-shared"))
        expect(self.page.locator('.node[data-node-id="unused"]')).to_have_class(re.compile("loop-outside"))
        expect(self.page.locator(".loop-edge")).to_have_count(1)
        expect(self.page.locator(".shared-edge")).to_have_count(1)
        expect(self.page.locator("#loop-region")).to_be_visible()
        expect(self.page.locator("#loop-steps li")).to_have_count(2)
        self.page.locator("#loop-arrange").click()
        self.assertGreater(
            self.page.locator('.node[data-node-id="save"]').evaluate("node => node.offsetLeft"),
            self.page.locator('.node[data-node-id="folder"]').evaluate("node => node.offsetLeft"),
        )
        self.page.locator('.node[data-node-id="folder"] [data-loop-mode="single"]').click()
        expect(self.page.locator("#loop-run")).to_be_disabled()
        expect(self.page.locator("#loop-region")).not_to_be_visible()
        expect(self.page.locator(".node.loop-member")).to_have_count(0)
        self.assertEqual(self.errors, [])

    def test_loop_trial_saves_one_then_run_all_and_roundtrips_mode(self):
        self.import_loop_workflow()
        self.enable_loop()
        with self.page.expect_download() as event:
            self.page.locator("#save-workflow").click()
        saved_path = self.fixture.root / "loop-saved.json"
        event.value.save_as(saved_path)
        self.assertEqual(
            json.loads(saved_path.read_text())["loop"], {"sourceId": "folder", "targetId": "save", "enabled": True, "conditionalTargetId": None}
        )
        self.page.locator("#new-workflow").click()
        self.page.locator("#load-workflow").set_input_files(saved_path)
        expect(self.page.locator("#loop-summary")).to_contain_text("2 张输入")
        self.page.locator("#loop-trial").click()
        expect(self.page.locator("#status")).to_contain_text("保存成功")
        output = self.fixture.root / "loop-output" / "img1"
        self.assertEqual(len(list(output.glob("*.png"))), 1)
        self.page.locator("#loop-run").click()
        expect(self.page.locator("#batch-status")).to_contain_text("已完成")
        expect(self.page.locator("#batch-counts")).to_contain_text("成功 1")
        expect(self.page.locator("#batch-counts")).to_contain_text("失败 1")
        self.assertEqual(len(list(output.glob("*.png"))), 2)
        self.assertEqual(self.errors, [])

    def test_save_node_explicitly_distinguishes_single_save_from_loop_run(self):
        self.import_loop_workflow()
        self.enable_loop()
        requests = []
        self.page.on("request", lambda request: requests.append((request.method, request.url)))
        save = self.page.locator('.node[data-node-id="save"]')
        save.locator(".node-settings > summary").click()
        expect(save.locator(".node-preview-button")).to_have_text("仅保存当前张 · 不循环")
        expect(save.locator(".save-run-hint")).to_contain_text("运行全部")
        save.locator(".node-preview-button").click()
        expect(self.page.locator("#status")).to_contain_text("本次仅保存当前张，未启动循环")
        output = self.fixture.root / "loop-output" / "img1"
        self.assertEqual(len(list(output.glob("*.png"))), 1)
        self.assertTrue(any(method == "POST" and url.endswith("/api/node/save-images") for method, url in requests))
        self.assertFalse(any(method == "POST" and url.endswith("/api/batch") for method, url in requests))
        self.page.locator("#loop-run").click()
        expect(self.page.locator("#batch-counts")).to_contain_text("已处理 2 / 2")
        expect(self.page.locator("#batch-status")).to_contain_text("已完成")
        self.assertTrue(any(method == "POST" and url.endswith("/api/batch") for method, url in requests))
        self.assertEqual(len(list(output.glob("*.png"))), 2)
        self.assertEqual(self.errors, [])

    def test_loop_revalidates_edits_and_disables_unsafe_output(self):
        self.import_loop_workflow()
        self.enable_loop()
        self.page.locator('.node[data-node-id="save"] .node-settings > summary').click()
        directory = self.page.locator('.node[data-node-id="save"] input[data-field="dir"]')
        directory.fill(str(self.fixture.root / "loop-input"))
        directory.press("Enter")
        expect(self.page.locator("#loop-run")).to_be_disabled()
        expect(self.page.locator("#loop-message")).to_contain_text("不能是输入目录")
        directory.fill(str(self.fixture.root / "safe-output"))
        directory.press("Enter")
        expect(self.page.locator("#loop-run")).to_be_enabled()
        self.page.locator('.node[data-node-id="folder"] .node-settings > summary').click()
        index = self.page.locator('.node[data-node-id="folder"] input[data-field="index"]')
        index.fill("1")
        index.press("Enter")
        expect(self.page.locator("#loop-summary")).to_contain_text("1 张输入")
        expect(self.page.locator("#loop-message")).to_contain_text("b.png")
        self.assertEqual(self.errors, [])

    def test_loop_multiple_folders_are_explicitly_rejected(self):
        self.import_loop_workflow(
            extra_nodes={
                "second": {"class_type": "FolderImagesNode", "inputs": {"dir": str(self.fixture.root / "loop-input")}}
            },
            extra_inputs={"img2": {"node": "second", "output": "image"}},
        )
        self.page.locator('.node[data-node-id="folder"] [data-loop-mode="each"]').click()
        expect(self.page.locator("#loop-summary")).to_contain_text("多个文件夹")
        expect(self.page.locator("#loop-run")).to_be_disabled()
        expect(self.page.locator("#loop-trial")).to_be_disabled()
        self.assertEqual(self.errors, [])

    def test_loop_multiple_save_branches_are_included_together(self):
        self.import_loop_workflow(
            extra_nodes={
                "other_save": {
                    "class_type": "SaveImagesNode",
                    "inputs": {
                        "img1": {"node": "folder", "output": "image"},
                        "dir": str(self.fixture.root / "other-output"),
                    },
                },
            }
        )
        self.page.locator('.node[data-node-id="folder"] [data-loop-mode="each"]').click()
        expect(self.page.locator("#loop-run")).to_be_enabled()
        self.page.locator("#loop-target").select_option("save")
        expect(self.page.locator("#loop-run")).to_be_enabled()
        expect(self.page.locator('.node[data-node-id="other_save"]')).to_have_class(re.compile("loop-member"))
        self.page.locator("#loop-target").select_option("other_save")
        expect(self.page.locator("#loop-message")).to_contain_text("other-output")
        expect(self.page.locator('.node[data-node-id="save"]')).to_have_class(re.compile("loop-member"))
        self.assertEqual(self.errors, [])

    def test_loop_blocks_back_edge_and_provides_ready_connected_example(self):
        self.page.locator("#loop-example").click()
        expect(self.page.locator(".node")).to_have_count(2)
        expect(self.page.locator("#loop-mode")).to_have_text("↻ 逐张模式")
        expect(self.page.locator(".loop-edge")).to_have_count(1)
        expect(self.page.locator("#loop-run")).to_be_disabled()
        self.import_loop_workflow(
            extra_nodes={
                "second": {"class_type": "FolderImagesNode", "inputs": {"dir": {"node": "folder", "output": "path"}}},
            }
        )
        self.page.locator('.port.output[data-node-id="second"][data-port="path"]').click()
        self.page.locator('.port.input[data-node-id="folder"][data-port="dir"]').click()
        expect(self.page.locator("#status")).to_contain_text("这条连接会形成回路")
        self.page.locator('.port.output[data-node-id="folder"][data-port="path"]').click()
        self.page.locator('.port.input[data-node-id="folder"][data-port="dir"]').click()
        expect(self.page.locator("#status")).to_contain_text("cannot connect to itself")
        self.assertEqual(self.errors, [])

    def test_download_then_import_preserves_graph_layout_and_config(self):
        self.configure_sample()
        with self.page.expect_download() as event:
            self.page.locator("#save-workflow").click()
        path = self.fixture.root / "roundtrip.json"
        event.value.save_as(path)
        saved = json.loads(path.read_text())
        self.assertEqual(saved["config"]["resolution"], 64)
        self.assertEqual(len(saved["workflow"]["nodes"]), 14)
        self.page.locator("#config-resolution").fill("128")
        self.page.locator("#load-workflow").set_input_files(path)
        expect(self.page.locator("#status")).to_contain_text("Imported workflow")
        expect(self.page.locator("#config-resolution")).to_have_value("64")
        with self.page.expect_download() as event:
            self.page.locator("#save-workflow").click()
        second = self.fixture.root / "second.json"
        event.value.save_as(second)
        self.assertEqual(json.loads(second.read_text()), saved)
        self.assertEqual(self.errors, [])

    def test_bad_import_preserves_current_canvas_and_can_retry_same_file(self):
        path = self.fixture.root / "broken.json"
        for text in ("not json", json.dumps({"nodes": [], "outputs": {}})):
            path.write_text(text)
            self.page.locator("#load-workflow").set_input_files(path)
            expect(self.page.locator("#status")).to_contain_text("Import failed")
            expect(self.page.locator(".node")).to_have_count(14)
        self.assertEqual(self.errors, [])

    def test_preview_and_dataset_export_from_browser(self):
        self.configure_sample()
        self.page.locator("#preview").click()
        expect(self.page.locator("#preview-grid img")).to_have_count(4)
        expect(self.page.locator("#status")).to_contain_text("Restore the cup")
        output = self.fixture.root / "browser_export"
        self.page.locator("#manifest-path").fill(str(self.fixture.manifest))
        self.page.locator("#output-dir").fill(str(output))
        self.page.locator("#export").click()
        expect(self.page.locator("#status")).to_contain_text('"train": 1')
        entry = json.loads((output / "train.jsonl").read_text())
        for path in [entry["target"], entry["source"], entry["mask"], *entry["references"]]:
            self.assertTrue((output / path).is_file())
        self.assertEqual(self.errors, [])

    def test_inline_inputs_port_click_and_node_output_preview(self):
        record = self.page.locator('.node[data-node-id="record"]')
        record.locator('input[data-field="target_path"]').fill(str(self.fixture.root / "target.png"))
        record.locator('input[data-field="target_path"]').press("Enter")
        record.locator('input[data-field="mask_path"]').fill(str(self.fixture.root / "mask.png"))
        record.locator('input[data-field="mask_path"]').press("Enter")
        record.locator('input[data-field="object_name"]').fill("cup")
        record.locator('input[data-field="object_name"]').press("Enter")
        expect(self.page.locator("#object-name")).to_have_value("cup")
        self.page.locator('.port.input[data-node-id="prompt"][data-port="object_name"]').click()
        expect(self.page.locator("#node-fields h3")).to_have_text("prompt")
        target = self.page.locator('.node[data-node-id="target"]')
        target.locator(".node-preview-button").click()
        expect(target.locator(".node-output-view img")).to_have_count(1)
        expect(self.page.locator("#node-preview img")).to_have_count(1)
        self.assertEqual(self.errors, [])

    def test_browser_file_picker_uploads_target_and_mask(self):
        record = self.page.locator('.node[data-node-id="record"]')
        record.locator('input[data-upload="target_path"]').set_input_files(self.fixture.root / "target.png")
        expect(self.page.locator("#target-path")).to_have_value(re.compile(".*studio-uploads-.*"))
        record.locator('input[data-upload="mask_path"]').set_input_files(self.fixture.root / "mask.png")
        expect(self.page.locator("#mask-path")).to_have_value(re.compile(".*studio-uploads-.*"))
        self.page.locator('.node[data-node-id="target"] .node-preview-button').click()
        expect(self.page.locator('.node[data-node-id="target"] .node-output-view img')).to_have_count(1)
        self.assertEqual(self.errors, [])

    def test_prompt_practice_retains_historical_output_until_retest(self):
        self.page.locator('[data-starter="BuildPromptNode"]').click()
        expect(self.page.locator(".node")).to_have_count(1)
        expect(self.page.locator("#node-help")).to_contain_text("不需要任何图片")
        self.page.locator(".node-preview-button").click()
        expect(self.page.locator("#node-preview")).to_contain_text("Restore the cup")
        expect(self.page.locator("#node-preview img")).to_have_count(0)
        field = self.page.locator('.node input[data-field="object_name"]')
        field.fill("bag")
        field.press("Enter")
        expect(self.page.locator("#node-preview")).to_contain_text("Restore the cup")
        expect(self.page.locator("#node-preview .preview-state")).to_contain_text("历史预览")
        self.page.locator(".node-preview-button").click()
        expect(self.page.locator("#node-preview")).to_contain_text("Restore the bag")
        expect(self.page.locator("#status")).to_contain_text("实际执行：buildprompt")
        self.assertEqual(self.errors, [])

    def test_flux_then_edit_save_reuses_model_and_keeps_preview_until_hidden(self):
        if os.environ.get("STUDIO_TEST_URL"):
            self.skipTest("CPU model substitute requires the in-process test server")
        output = self.fixture.root / "edited-save-output"
        workflow = {"nodes": {
            "input": {"class_type": "LoadTargetNode", "inputs": {
                "path": str(self.fixture.root / "target.png"),
            }},
            "flux": {"class_type": "FluxKleinReferenceNode", "inputs": {
                "prompt": "CPU regression substitute", "reference_1": {"node": "input", "output": "target"},
            }},
            "save": {"class_type": "SaveImagesNode", "inputs": {
                "img1": {"node": "flux", "output": "image"},
                "dir": str(self.fixture.root / "before-edit"),
            }},
        }, "outputs": {}}
        with patch.object(NODE_CLASS_MAPPINGS["FluxKleinReferenceNode"], "execute", return_value={
            "image": Image.new("RGB", (16, 16), (12, 34, 56)),
        }) as model:
            self.page.evaluate("""workflow => {
                startPractice(); state.workflow = workflow;
                state.layout = {input: {x: 30, y: 30}, flux: {x: 350, y: 30}, save: {x: 680, y: 30}};
                render();
            }""", workflow)
            flux = self.page.locator('.node[data-node-id="flux"]')
            flux.locator(".node-preview-button").click()
            expect(self.page.locator("#status")).to_contain_text("测试成功：flux")
            expect(flux.locator(".preview-body img")).to_be_visible()
            save = self.page.locator('.node[data-node-id="save"]')
            save.locator('input[data-field="dir"]').fill(str(output))
            save.locator('input[data-field="dir"]').press("Enter")
            save.locator('input[data-save-name="img1"]').fill("generated")
            save.locator('input[data-save-name="img1"]').press("Enter")
            expect(flux.locator(".preview-body img")).to_be_visible()
            save.locator(".node-preview-button").click()
            expect(self.page.locator("#status")).to_contain_text("保存成功：save")
            expect(self.page.locator("#status")).to_contain_text("复用上游")
            self.assertEqual(model.call_count, 1)
            expect(flux.locator(".preview-body img")).to_be_visible()
            saved = list((output / "generated").glob("*.png"))
            self.assertEqual(len(saved), 1)
            with Image.open(saved[0]) as image:
                self.assertEqual(image.getpixel((0, 0)), (12, 34, 56))
            flux.locator(".preview-toggle").click()
            expect(flux.locator(".preview-body")).to_be_hidden()
            # A full UI re-render must not reopen an explicitly hidden preview.
            self.page.evaluate("render()")
            expect(flux.locator(".preview-body")).to_be_hidden()
            flux.locator(".preview-toggle").click()
            expect(flux.locator(".preview-body img")).to_be_visible()
            self.assertEqual(model.call_count, 1)
        self.assertEqual(self.errors, [])

    def test_target_practice_file_button_and_image_zoom(self):
        self.page.locator('[data-starter="LoadTargetNode"]').click()
        with self.page.expect_file_chooser() as event:
            self.page.locator(".node .file-picker").click()
        event.value.set_files(self.fixture.root / "target.png")
        expect(self.page.locator('.node input[data-field="path"]')).to_have_value(re.compile(".*studio-uploads-.*"))
        self.page.locator(".node-preview-button").click()
        expect(self.page.locator("#node-preview img")).to_have_count(1)
        expect(self.page.locator("#node-preview figcaption")).to_contain_text("96 × 64")
        self.page.locator(".node-output-view img").first.click()
        expect(self.page.locator("#image-dialog")).to_be_visible()
        self.page.locator("#close-image-dialog").click()
        expect(self.page.locator("#image-dialog")).not_to_be_visible()
        expect(self.page.locator("#mask-path")).to_have_value("")
        self.assertEqual(self.errors, [])

    def test_mask_practice_and_its_parameter_help(self):
        self.page.locator('[data-starter="ExpandMaskNode"]').click()
        expect(self.page.locator("#node-help")).to_contain_text("扩展半径")
        self.page.locator('.node input[data-upload="object_mask"]').set_input_files(self.fixture.root / "mask.png")
        expect(self.page.locator('.node input[data-field="object_mask"]')).to_have_value(
            re.compile(".*studio-uploads-.*")
        )
        self.page.locator(".node button", has_text="编辑相关参数").click()
        expect(self.page.locator("#config-dilation_radius")).to_be_focused()
        self.page.locator("#config-dilation_radius").fill("0")
        self.page.locator(".node-preview-button").click()
        expect(self.page.locator("#node-preview figcaption")).to_contain_text("edit_mask · 图像")
        expect(self.page.locator("#status")).to_contain_text("实际执行：expandmask")
        self.assertEqual(self.errors, [])

    def test_single_node_workflow_roundtrip_and_restore_complete_example(self):
        self.page.locator('[data-starter="BuildPromptNode"]').click()
        with self.page.expect_download() as event:
            self.page.locator("#save-workflow").click()
        path = self.fixture.root / "practice.json"
        event.value.save_as(path)
        saved = json.loads(path.read_text())
        self.assertEqual(saved["workflow"]["outputs"], {})
        self.page.locator("#new-workflow").click()
        expect(self.page.locator(".node")).to_have_count(0)
        self.page.locator("#load-workflow").set_input_files(path)
        expect(self.page.locator("#status")).to_contain_text("Imported workflow")
        expect(self.page.locator(".node")).to_have_count(1)
        self.page.locator(".node-preview-button").click()
        expect(self.page.locator("#node-preview")).to_contain_text("Restore the cup")
        self.page.locator("#default-workflow").click()
        expect(self.page.locator(".node")).to_have_count(14)
        self.assertEqual(self.errors, [])

    def test_sam2_starter_explains_hf_model_and_checks_box_before_loading(self):
        self.page.locator('[data-starter="SAM2MaskNode"]').click()
        expect(self.page.locator(".node")).to_have_count(1)
        expect(self.page.locator("#node-help")).to_contain_text("facebook/sam2-hiera-tiny")
        self.page.locator('.node input[data-upload="image"]').set_input_files(self.fixture.root / "target.png")
        expect(self.page.locator('.node input[data-field="image"]')).to_have_value(re.compile(".*studio-uploads-.*"))
        self.page.locator('.node input[data-field="bbox"]').fill("[0, 0, 0, 0]")
        self.page.locator('.node input[data-field="bbox"]').press("Enter")
        self.page.locator(".node-preview-button").click()
        expect(self.page.locator("#status")).to_contain_text("物体框必须在图片范围内")
        self.assertEqual(self.errors, [])

    def test_mask_to_rgba_starter_uploads_and_previews_transparency(self):
        self.page.locator('[data-starter="MaskToRGBANode"]').click()
        expect(self.page.locator(".node")).to_have_count(1)
        expect(self.page.locator("#node-help")).to_contain_text("Alpha")
        expect(self.page.locator("#node-help")).to_contain_text("不裁剪")
        for name, filename in (("image", "target.png"), ("mask", "mask.png")):
            self.page.locator(f'.node input[data-upload="{name}"]').set_input_files(self.fixture.root / filename)
            expect(self.page.locator(f'.node input[data-field="{name}"]')).to_have_value(
                re.compile(".*studio-uploads-.*")
            )
        self.page.locator(".node-preview-button").click()
        preview = self.page.locator("#node-preview img")
        expect(preview).to_have_count(1)
        expect(preview).to_have_class("alpha-preview")
        expect(self.page.locator("#node-preview figcaption")).to_contain_text("96 × 64 RGBA")
        self.assertIn("conic-gradient", preview.evaluate("image => getComputedStyle(image).backgroundImage"))
        preview.click()
        expect(self.page.locator("#image-dialog")).to_be_visible()
        expect(self.page.locator("#image-dialog img")).to_have_class("alpha-preview")
        self.page.locator("#close-image-dialog").click()
        self.page.locator('[data-starter="LoadTargetNode"]').click()
        self.page.locator('.node input[data-upload="path"]').set_input_files(self.fixture.root / "target.png")
        expect(self.page.locator('.node input[data-field="path"]')).to_have_value(re.compile(".*studio-uploads-.*"))
        self.page.locator(".node-preview-button").click()
        self.page.locator("#node-preview img").click()
        expect(self.page.locator("#image-dialog img")).not_to_have_class("alpha-preview")
        self.assertEqual(self.errors, [])

    def test_sam3_starter_explains_local_text_segmentation_and_validates_text(self):
        self.page.locator('[data-starter="SAM3TextMaskNode"]').click()
        expect(self.page.locator(".node")).to_have_count(1)
        expect(self.page.locator("#node-help")).to_contain_text("ModelScope")
        expect(self.page.locator("#node-help")).to_contain_text("置信度最高")
        expect(self.page.locator('.node input[data-field="text"]')).to_have_value("cup")
        expect(self.page.locator('.node input[data-field="bbox"]')).to_have_count(0)
        self.page.locator('.node input[data-upload="image"]').set_input_files(self.fixture.root / "target.png")
        expect(self.page.locator('.node input[data-field="image"]')).to_have_value(re.compile(".*studio-uploads-.*"))
        self.page.locator('.node input[data-field="text"]').fill(" ")
        self.page.locator('.node input[data-field="text"]').press("Enter")
        self.page.locator(".node-preview-button").click()
        expect(self.page.locator("#status")).to_contain_text("请输入要分割的物体文本")
        self.assertEqual(self.errors, [])


if __name__ == "__main__":
    unittest.main()
