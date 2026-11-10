"""Real CPU loop-demo nodes, PNG pixels, Studio catalog and preview coverage."""

import base64
import io
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from typing import get_type_hints, is_typeddict
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from object_inpainting.graph import NodeGraph
from object_inpainting.node_contracts import port_schema
from object_inpainting.nodes import NODE_CLASS_MAPPINGS, LoopDemoImageNode, LoopDemoScoreNode
from object_inpainting_studio.server import app, prepare_conditional
from object_inpainting_studio.conditional_loop import run_conditional


class LoopDemoTests(unittest.TestCase):
    def setUp(self):
        delay = patch("object_inpainting.nodes.loop_demo.time.sleep")
        delay.start()
        self.addCleanup(delay.stop)

    def test_folder_image_marker_preserves_source_and_scores_only_marker(self):
        source = Image.new("RGB", (80, 60), (25, 50, 255))
        before = source.tobytes()
        for seed, color, score in ((0, (255, 0, 0), 0.0), (1, (0, 255, 0), 0.0),
                                   (2, (0, 0, 255), 255.0)):
            result = LoopDemoImageNode().execute(seed=seed, image=source)
            image = result["image"]
            self.assertEqual(image.size, source.size)
            self.assertEqual(image.getpixel((31, 31)), color)
            self.assertEqual(image.getpixel((32, 0)), (25, 50, 255))
            self.assertEqual(image.getpixel((79, 59)), (25, 50, 255))
            self.assertEqual(LoopDemoScoreNode().execute(image), {"score": score})
        self.assertEqual(source.tobytes(), before)

    def test_marker_handles_small_images_and_rejects_invalid_input(self):
        result = LoopDemoImageNode().execute(image=Image.new("RGB", (7, 5)), seed=2)
        self.assertEqual(result["image"].size, (7, 5))
        self.assertEqual(LoopDemoScoreNode().execute(result["image"]), {"score": 255.0})
        for image in ("file.png", Image.new("L", (5, 5)), Image.new("RGB", (0, 5))):
            with self.assertRaisesRegex(ValueError, "image"):
                LoopDemoImageNode().execute(image=image)

    def workflow(self, seed=None):
        return {
            "nodes": {
                "color": {"class_type": "LoopDemoImageNode", "inputs": {} if seed is None else {"seed": seed}},
                "score": {"class_type": "LoopDemoScoreNode", "inputs": {
                    "image": {"node": "color", "output": "image"},
                }},
            },
            "outputs": {
                "image": {"node": "color", "output": "image"},
                "value": {"node": "color", "output": "value"},
                "score": {"node": "score", "output": "score"},
            },
        }

    def assert_rgb(self, image, color):
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (32, 32))
        self.assertEqual(image.tobytes(), bytes(color) * (32 * 32))

    def test_seed_colors_values_and_real_png_roundtrip(self):
        colors = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (122, 122, 122),
                  (255, 255, 0), (0, 255, 255), (255, 0, 255))
        for seed in (0, 1, 2, 3, 4, 5, 2**63 - 1):
            with self.subTest(seed=seed):
                result = LoopDemoImageNode().execute(seed)
                self.assertEqual(set(result), {"image", "value"})
                self.assertIs(type(result["value"]), int)
                self.assertEqual(result["value"], seed)
                self.assert_rgb(result["image"], colors[seed % len(colors)])
                buffer = io.BytesIO()
                result["image"].save(buffer, format="PNG")
                with Image.open(io.BytesIO(buffer.getvalue())) as decoded:
                    self.assertEqual(decoded.format, "PNG")
                    self.assert_rgb(decoded, colors[seed % len(colors)])

    def test_default_seed_and_independent_image_instances(self):
        first = LoopDemoImageNode().execute()
        second = LoopDemoImageNode().execute()
        self.assertEqual(first["value"], 0)
        self.assert_rgb(first["image"], (255, 0, 0))
        self.assertIsNot(first["image"], second["image"])
        first["image"].putpixel((0, 0), (0, 0, 0))
        self.assert_rgb(second["image"], (255, 0, 0))

    def test_negative_and_non_integer_seeds_are_rejected(self):
        for seed in (-2, -3, True, False, 1.5, "2", None):
            with self.subTest(seed=seed), self.assertRaisesRegex(ValueError, "seed"):
                LoopDemoImageNode().execute(seed)  # type: ignore[arg-type]

    def test_random_seed_returns_actual_seed_and_reproducible_color(self):
        with patch("object_inpainting.nodes.loop_demo.randrange", side_effect=[10, 11]) as draw:
            first = LoopDemoImageNode().execute(-1)
            second = LoopDemoImageNode().execute(-1)
        self.assertEqual(draw.call_count, 2)
        for result, seed in ((first, 10), (second, 11)):
            self.assertEqual(result["value"], seed)
            self.assertEqual(result["image"].tobytes(),
                             LoopDemoImageNode().execute(seed)["image"].tobytes())

    def test_real_blue_channel_scores_and_whole_image_average(self):
        for seed, expected in ((0, 0.0), (1, 0.0), (2, 255.0), (3, 122.0)):
            image = LoopDemoImageNode().execute(seed)["image"]
            before = image.tobytes()
            result = LoopDemoScoreNode().execute(image)
            self.assertEqual(result, {"score": expected})
            self.assertIs(type(result["score"]), float)
            self.assertEqual(image.tobytes(), before)
        image = Image.new("RGB", (2, 1), (255, 255, 0))
        image.putpixel((1, 0), (0, 0, 255))
        self.assertEqual(LoopDemoScoreNode().execute(image), {"score": 127.5})
        for intensity in (0, 64, 255):
            image = Image.new("RGB", (1, 3), (10, 20, intensity))
            self.assertEqual(LoopDemoScoreNode().execute(image), {"score": float(intensity)})

    def test_score_rejects_non_rgb_and_empty_images(self):
        for image in (None, "image.png", Image.new("L", (2, 2)),
                      Image.new("RGBA", (2, 2)), Image.new("RGB", (0, 2))):
            with self.subTest(image=image), self.assertRaisesRegex(ValueError, "image"):
                LoopDemoScoreNode().execute(image)  # type: ignore[arg-type]

    def test_registered_typeddict_contracts_generate_exact_port_schemas(self):
        for node, inputs, outputs in (
            (LoopDemoImageNode, {"seed": "INT", "image": "RGB_IMAGE"}, {"image": "RGB_IMAGE", "value": "INT"}),
            (LoopDemoScoreNode, {"image": "RGB_IMAGE"}, {"score": "FLOAT"}),
        ):
            with self.subTest(node=node.__name__):
                self.assertIs(NODE_CLASS_MAPPINGS[node.__name__], node)
                self.assertTrue(is_typeddict(node.INPUT_SCHEMA))
                self.assertTrue(is_typeddict(node.OUTPUT_SCHEMA))
                self.assertEqual(node.INPUT_TYPES, inputs)
                self.assertEqual(node.OUTPUT_TYPES, outputs)
                self.assertEqual(port_schema(node.INPUT_SCHEMA), inputs)
                self.assertEqual(port_schema(node.OUTPUT_SCHEMA), outputs)
                annotations = get_type_hints(node.execute, include_extras=True)
                self.assertIs(annotations.pop("return"), node.OUTPUT_SCHEMA)
                self.assertEqual(annotations, get_type_hints(node.INPUT_SCHEMA, include_extras=True))

    def test_graph_applies_seed_default_and_keeps_original_definition(self):
        workflow = self.workflow()
        original = deepcopy(workflow)
        graph = NodeGraph(workflow)
        result = graph.run({})
        self.assertEqual(graph.nodes["color"]["inputs"], {"seed": 0, "image": None})
        self.assertEqual((result["value"], result["score"]), (0, 0.0))
        self.assert_rgb(result["image"], (255, 0, 0))
        self.assertEqual(workflow, original)

    def test_studio_catalog_has_titles_defaults_contracts_and_port_roles(self):
        with TestClient(app) as client:
            response = client.get("/api/node-types")
        self.assertEqual(response.status_code, 200, response.text)
        catalog = response.json()
        for node, title, defaults in (
            (LoopDemoImageNode, "循环演示·色图", {"seed": 0}),
            (LoopDemoScoreNode, "循环演示·评分", {}),
        ):
            info = catalog[node.__name__]
            self.assertEqual(info["help"]["title"], title)
            self.assertEqual(info["inputs"], node.INPUT_TYPES)
            self.assertEqual(info["outputs"], node.OUTPUT_TYPES)
            self.assertEqual(info["input_defaults"], defaults)
            self.assertEqual(info["input_contract"], node.INPUT_SCHEMA.__name__)
            self.assertEqual(info["output_contract"], node.OUTPUT_SCHEMA.__name__)
            self.assertFalse(info["side_effects"])
            self.assertEqual(info["help"]["parameters"], [])
            for direction in ("inputs", "outputs"):
                for port in info[direction]:
                    self.assertIn(port, info["help"][direction])
                    self.assertTrue(info["port_labels"][direction][port]["role"])

    def test_preview_api_returns_real_png_value_and_connected_score(self):
        with TestClient(app) as client:
            for seed, color, score in ((0, (255, 0, 0), 0.0), (1, (0, 255, 0), 0.0),
                                       (2, (0, 0, 255), 255.0)):
                with self.subTest(seed=seed):
                    workflow = self.workflow(seed)
                    response = client.post("/api/node/preview", json={"workflow": workflow, "node_id": "color"})
                    self.assertEqual(response.status_code, 200, response.text)
                    outputs = response.json()["outputs"]
                    self.assertEqual(outputs["value"], {"kind": "text", "value": seed})
                    self.assertEqual(outputs["image"]["kind"], "image")
                    self.assertEqual(outputs["image"]["mode"], "RGB")
                    self.assertEqual(outputs["image"]["size"], [32, 32])
                    header, encoded = outputs["image"]["image"].split(",", 1)
                    self.assertEqual(header, "data:image/png;base64")
                    with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
                        self.assertEqual(image.format, "PNG")
                        self.assert_rgb(image, color)
                    response = client.post("/api/node/preview", json={"workflow": workflow, "node_id": "score"})
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertEqual(response.json()["outputs"]["score"], {"kind": "text", "value": score})
            with patch("object_inpainting.nodes.loop_demo.randrange", return_value=11):
                response = client.post("/api/node/preview", json={"workflow": self.workflow(-1), "node_id": "color"})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["outputs"]["value"], {"kind": "text", "value": 11})
            response = client.post("/api/node/preview", json={"workflow": self.workflow(-2), "node_id": "color"})
            self.assertEqual(response.status_code, 422, response.text)
            self.assertIn("seed", response.json()["detail"])

    def test_real_loop_passes_on_third_attempt_and_existing_save_writes_only_blue_png(self):
        for maximum in (3, 2):
            with self.subTest(maximum=maximum), TemporaryDirectory() as directory:
                output = Path(directory) / "output"
                workflow = self.workflow()  # Omitted seed must still become the loop's starting zero.
                workflow["outputs"] = {}
                workflow["nodes"]["save"] = {
                    "class_type": "SaveImagesNode",
                    "save_items": [{"id": "img1", "name": "accepted"}],
                    "inputs": {"img1": {"node": "color", "output": "image"},
                               "dir": str(output), "filename": "demo"},
                }
                workflow["conditional_loop"] = {"loops": [{
                    "id": "demo", "name": "色图评分", "start_nodes": ["color"],
                    "end_nodes": ["score"], "seed_nodes": ["color"],
                    "max_iterations": maximum, "on_exhausted": "skip",
                    "conditions": [{"node": "score", "output": "score", "operator": ">=", "value": 255}],
                }]}
                original = deepcopy(workflow)
                _, plan = prepare_conditional({"workflow": workflow, "node_id": "save"})
                self.assertFalse(output.exists(), "planning must not write files")
                result = run_conditional(plan, {}, Event())
                self.assertEqual(result["accepted"], maximum == 3)
                self.assertEqual(result["iterations"], maximum)
                self.assertEqual([entry["seeds"] for entry in result["history"]],
                                 [{"color": seed} for seed in range(maximum)])
                self.assertEqual([entry["scores"][0]["value"] for entry in result["history"]],
                                 [0.0, 0.0, 255.0][:maximum])
                self.assertEqual([entry["accepted"] for entry in result["history"]],
                                 [False, False, True][:maximum])
                self.assertEqual(workflow, original)
                if maximum == 3:
                    self.assertEqual(result["stop_reason"], "accepted")
                    self.assertEqual(set(result["paths"]), {"img1"})
                    path = Path(result["paths"]["img1"])
                    saved = list(output.rglob("*.png"))
                    self.assertEqual(len(saved), 1)
                    self.assertTrue(saved[0].samefile(path))
                    with Image.open(path) as image:
                        self.assertEqual(image.format, "PNG")
                        self.assert_rgb(image, (0, 0, 255))
                else:
                    self.assertEqual(result["stop_reason"], "max_iterations")
                    self.assertEqual(result["paths"], {})
                    self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
