import json
import os
import re
import socket
import unittest
from pathlib import Path
from threading import Thread
from types import SimpleNamespace
from unittest.mock import patch

import test_object_inpainting_dataset as dataset_tests
import uvicorn
from object_inpainting.nodes import NODE_CLASS_MAPPINGS, GrayMaskNode
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
            json.loads(saved_path.read_text())["loop"], {"sourceId": "folder", "targetId": "save", "enabled": True}
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
        expect(self.page.locator("#batch-counts")).to_contain_text("成功 2")
        self.assertEqual(len(list(output.glob("*.png"))), 3)
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
        self.assertEqual(len(list(output.glob("*.png"))), 3)
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

    def test_loop_multiple_save_branches_require_explicit_target(self):
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
        expect(self.page.locator("#loop-summary")).to_contain_text("多个保存分支")
        expect(self.page.locator("#loop-run")).to_be_disabled()
        self.page.locator("#loop-target").select_option("save")
        expect(self.page.locator("#loop-run")).to_be_enabled()
        expect(self.page.locator('.node[data-node-id="other_save"]')).to_have_class(re.compile("loop-outside"))
        self.page.locator("#loop-target").select_option("other_save")
        expect(self.page.locator("#loop-message")).to_contain_text("other-output")
        expect(self.page.locator('.node[data-node-id="save"]')).to_have_class(re.compile("loop-outside"))
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

    def test_prompt_practice_displays_text_and_invalidates_old_output(self):
        self.page.locator('[data-starter="BuildPromptNode"]').click()
        expect(self.page.locator(".node")).to_have_count(1)
        expect(self.page.locator("#node-help")).to_contain_text("不需要任何图片")
        self.page.locator(".node-preview-button").click()
        expect(self.page.locator("#node-preview")).to_contain_text("Restore the cup")
        expect(self.page.locator("#node-preview img")).to_have_count(0)
        field = self.page.locator('.node input[data-field="object_name"]')
        field.fill("bag")
        field.press("Enter")
        expect(self.page.locator("#node-preview pre")).to_have_count(0)
        self.page.locator(".node-preview-button").click()
        expect(self.page.locator("#node-preview")).to_contain_text("Restore the bag")
        expect(self.page.locator("#status")).to_contain_text("实际执行：buildprompt")
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
