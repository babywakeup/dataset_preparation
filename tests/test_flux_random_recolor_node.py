import base64
import inspect
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient
from object_inpainting.graph import NodeGraph
from object_inpainting.nodes import flux_klein_reference, flux_random_recolor
from object_inpainting.nodes.flux_random_recolor import FluxRandomRecolorNode, choose_recolor, recover_recolor_mask
from object_inpainting.nodes.gray_mask import GrayMaskNode
from object_inpainting_studio.server import app
from PIL import Image


class FluxRandomRecolorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.target = Image.new("RGB", (35, 21), (128, 32, 64))
        self.edit_mask = Image.new("L", self.target.size)
        self.edit_mask.paste(255, (5, 3, 30, 18))
        self.mask = GrayMaskNode().execute(self.target, self.edit_mask)["source"]
        self.calls = []

        def generate(**kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(images=[Image.new("RGB", (kwargs["width"], kwargs["height"]), (17, 201, 99))])

        self.loader = patch.object(flux_klein_reference, "get_flux_klein", return_value=generate)
        self.loader.start()
        self.addCleanup(self.loader.stop)

    def test_real_generation_boundary_gets_three_ordered_references_and_seed(self) -> None:
        outputs = FluxRandomRecolorNode().execute(self.target, self.mask, seed=42)
        call = self.calls[0]
        self.assertEqual((call["width"], call["height"]), (48, 32))
        self.assertEqual(call["generator"].initial_seed(), 42)
        self.assertEqual(call["num_inference_steps"], 28)
        self.assertEqual(call["guidance_scale"], 4.0)
        self.assertEqual(call["output_type"], "pil")
        self.assertEqual(len(call["image"]), 3)
        np.testing.assert_array_equal(call["image"][0].crop((0, 0, 35, 21)), self.target)
        gray_reference = call["image"][1]
        self.assertEqual(gray_reference.mode, "RGB")
        self.assertEqual(gray_reference.size, call["image"][0].size)
        self.assertEqual(gray_reference.getpixel((10, 10)), (128, 128, 128))
        self.assertEqual(gray_reference.getpixel((0, 0)), self.target.getpixel((0, 0)))
        self.assertEqual(gray_reference.getpixel((40, 25)), (128, 128, 128))
        np.testing.assert_array_equal(gray_reference.crop((0, 0, 35, 21)), self.mask)
        np.testing.assert_array_equal(outputs["mask"], self.edit_mask)
        np.testing.assert_array_equal(call["image"][2], outputs["sampled_color"])
        self.assertIn(outputs["color_name"], call["prompt"])
        self.assertIn("fabric weave, texture, folds", call["prompt"])
        self.assertIn("editable clothing region covered in uniform gray", call["prompt"])
        self.assertIn("Use image 1, not the gray fill in image 2", call["prompt"])
        self.assertNotIn("binary editing mask", call["prompt"])
        self.assertNotIn("white mask", call["prompt"])

    def test_naturally_gray_pixels_outside_mask_remain_unedited(self) -> None:
        self.target.putpixel((0, 0), (128, 128, 128))
        self.mask = GrayMaskNode().execute(self.target, self.edit_mask)["source"]
        outputs = FluxRandomRecolorNode().execute(self.target, self.mask, seed=42)
        self.assertEqual(self.calls[0]["image"][1].getpixel((0, 0)), (128, 128, 128))
        self.assertEqual(outputs["mask"].getpixel((0, 0)), 0)
        self.assertEqual(outputs["source"].getpixel((0, 0)), (128, 128, 128))
        self.assertEqual(outputs["source"].getpixel((10, 10)), (17, 201, 99))

    def test_generation_pixels_used_only_inside_mask_without_mutating_inputs(self) -> None:
        original, original_mask = np.asarray(self.target).copy(), np.asarray(self.mask).copy()
        outputs = FluxRandomRecolorNode().execute(self.target, self.mask, seed=42)
        source = np.asarray(outputs["source"])
        active = np.asarray(self.edit_mask) > 127
        np.testing.assert_array_equal(source[~active], original[~active])
        self.assertTrue(np.all(source[active] == (17, 201, 99)))
        self.assertEqual(outputs["source"].size, self.target.size)
        self.assertEqual(outputs["source"].mode, "RGB")
        masked = np.asarray(outputs["masked_source"])
        self.assertTrue(np.all(masked[active] == 128))
        np.testing.assert_array_equal(masked[~active], original[~active])
        np.testing.assert_array_equal(self.target, original)
        np.testing.assert_array_equal(self.mask, original_mask)

    def test_fixed_seed_repeats_candidate_and_random_seed_is_reported(self) -> None:
        first = FluxRandomRecolorNode().execute(self.target, self.mask, seed=7)
        second = FluxRandomRecolorNode().execute(self.target, self.mask, seed=7)
        self.assertEqual(first["color_name"], second["color_name"])
        np.testing.assert_array_equal(first["sampled_color"], second["sampled_color"])
        with patch.object(flux_random_recolor.secrets, "randbelow", side_effect=[101, 102]):
            random_first = FluxRandomRecolorNode().execute(self.target, self.mask)
            random_second = FluxRandomRecolorNode().execute(self.target, self.mask)
        self.assertEqual((random_first["actual_seed"], random_second["actual_seed"]), (101, 102))
        self.assertEqual([call["generator"].initial_seed() for call in self.calls], [7, 7, 101, 102])

    def test_candidate_differs_from_original_median_color(self) -> None:
        name, rgb = choose_recolor(self.target, self.edit_mask, 42)
        self.assertIn((name, rgb), flux_random_recolor.RECOLOR_PALETTE)
        self.assertGreaterEqual(np.linalg.norm(np.asarray(rgb) - (128, 32, 64)), 80)

    def test_black_white_masks_are_rejected_including_rgb_conversions(self) -> None:
        for image in (self.edit_mask, self.edit_mask.convert("1"), self.edit_mask.convert("RGB")):
            with self.subTest(mode=image.mode), self.assertRaisesRegex(ValueError, "mask"):
                FluxRandomRecolorNode().execute(self.target, image, seed=0)
        self.assertEqual(self.calls, [])

    def test_original_gray_inside_edit_area_is_preserved_as_unidentifiable(self) -> None:
        self.target.putpixel((10, 10), (128, 128, 128))
        self.mask = GrayMaskNode().execute(self.target, self.edit_mask)["source"]
        outputs = FluxRandomRecolorNode().execute(self.target, self.mask, seed=0)
        self.assertEqual(outputs["mask"].getpixel((10, 10)), 0)
        self.assertEqual(outputs["source"].getpixel((10, 10)), (128, 128, 128))
        self.assertEqual(outputs["source"].getpixel((11, 10)), (17, 201, 99))

    def test_unrelated_changes_and_partial_gray_fills_are_rejected(self) -> None:
        for pixel in ((0, 0, 0), (127, 128, 128), (128, 64, 96)):
            invalid = self.mask.copy()
            invalid.putpixel((0, 0), pixel)
            with self.subTest(pixel=pixel), self.assertRaisesRegex(ValueError, "变化像素"):
                recover_recolor_mask(self.target, invalid)
        self.assertEqual(self.calls, [])

    def test_invalid_inputs_never_start_inference(self) -> None:
        for values in (
            {"target": None},
            {"target": self.target.convert("RGBA")},
            {"mask": self.mask.convert("L")},
            {"mask": Image.new("RGB", (16, 16), (128, 128, 128))},
            {"mask": self.target.copy()},
            {"target": Image.new("RGB", (8, 8)), "mask": Image.new("RGB", (8, 8), (128, 128, 128))},
            {"seed": True},
            {"seed": -2},
            {"seed": 2**32},
            {"num_inference_steps": 0},
            {"guidance_scale": float("nan")},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                FluxRandomRecolorNode().execute(**{"target": self.target, "mask": self.mask, **values})
        self.assertEqual(self.calls, [])

    def test_generation_errors_are_not_replaced_with_fake_recolors(self) -> None:
        with patch.object(flux_klein_reference, "get_flux_klein", side_effect=RuntimeError("offline model missing")):
            with self.assertRaisesRegex(ValueError, "offline model missing"):
                FluxRandomRecolorNode().execute(self.target, self.mask)
        with patch.object(
            flux_klein_reference,
            "get_flux_klein",
            return_value=lambda **_: SimpleNamespace(images=[Image.new("RGB", (16, 16))]),
        ):
            with self.assertRaisesRegex(ValueError, "生成尺寸"):
                FluxRandomRecolorNode().execute(self.target, self.mask)

    def test_graph_defaults_and_same_execution_share_one_generation(self) -> None:
        graph = NodeGraph(
            {
                "nodes": {
                    "gray": {
                        "class_type": "GrayMaskNode",
                        "inputs": {"target": {"context": "target"}, "edit_mask": {"context": "edit_mask"}},
                    },
                    "recolor": {
                        "class_type": "FluxRandomRecolorNode",
                        "inputs": {"target": {"context": "target"}, "mask": {"node": "gray", "output": "source"}},
                    },
                },
                "outputs": {
                    "source": {"node": "recolor", "output": "source"},
                    "masked": {"node": "recolor", "output": "masked_source"},
                    "seed": {"node": "recolor", "output": "actual_seed"},
                },
            }
        )
        outputs = graph.run({"target": self.target, "edit_mask": self.edit_mask}, allow_side_effects=False)
        self.assertEqual(outputs["source"].size, self.target.size)
        self.assertIsInstance(outputs["seed"], int)
        self.assertEqual(len(self.calls), 1)

    def test_graph_rejects_old_mask_port_connections(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected RGB_IMAGE, got MASK"):
            NodeGraph(
                {
                    "nodes": {
                        "expand": {
                            "class_type": "ExpandMaskNode",
                            "inputs": {"object_mask": {"context": "mask"}, "config": {"context": "config"}},
                        },
                        "recolor": {
                            "class_type": "FluxRandomRecolorNode",
                            "inputs": {
                                "target": {"context": "target"},
                                "mask": {"node": "expand", "output": "edit_mask"},
                            },
                        },
                    },
                    "outputs": {},
                }
            )

    def test_api_rejects_uploaded_black_white_mask_before_inference(self) -> None:
        with TestClient(app) as client:
            paths = {}
            for name, image in (("target", self.target), ("mask", self.edit_mask)):
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                upload = client.post("/api/upload", files={"file": (f"{name}.png", buffer.getvalue(), "image/png")})
                paths[name] = upload.json()["path"]
            workflow = {
                "nodes": {"recolor": {"class_type": "FluxRandomRecolorNode", "inputs": paths}},
                "outputs": {},
            }
            response = client.post("/api/node/preview", json={"workflow": workflow, "node_id": "recolor"})
            self.assertEqual(response.status_code, 422, response.text)
            self.assertIn("涂灰原图", response.text)
        self.assertEqual(self.calls, [])

    def test_api_upload_preview_returns_source_and_metadata(self) -> None:
        with TestClient(app) as client:
            paths = {}
            for name, image in (("target", self.target), ("mask", self.mask)):
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                upload = client.post("/api/upload", files={"file": (f"{name}.png", buffer.getvalue(), "image/png")})
                self.assertEqual(upload.status_code, 200, upload.text)
                paths[name] = upload.json()["path"]
            workflow = {
                "nodes": {"recolor": {"class_type": "FluxRandomRecolorNode", "inputs": {**paths, "seed": 42}}},
                "outputs": {},
            }
            response = client.post("/api/node/preview", json={"workflow": workflow, "node_id": "recolor"})
            self.assertEqual(response.status_code, 200, response.text)
            outputs = response.json()["outputs"]
            self.assertEqual(outputs["actual_seed"]["value"], 42)
            self.assertEqual(outputs["mask"]["mode"], "L")
            with Image.open(io.BytesIO(base64.b64decode(outputs["source"]["image"].split(",", 1)[1]))) as image:
                self.assertEqual(image.size, self.target.size)
                self.assertEqual(image.getpixel((10, 10)), (17, 201, 99))
                self.assertEqual(image.getpixel((0, 0)), self.target.getpixel((0, 0)))

    def test_public_interfaces_describe_input_and_output_types(self) -> None:
        for function in (choose_recolor, recover_recolor_mask, FluxRandomRecolorNode.execute):
            signature = inspect.signature(function)
            self.assertNotEqual(signature.return_annotation, inspect.Signature.empty)
            for name, parameter in signature.parameters.items():
                if name != "self":
                    self.assertNotEqual(parameter.annotation, inspect.Parameter.empty)
            self.assertIn("Args:", function.__doc__)
            self.assertIn("Returns:", function.__doc__)


if __name__ == "__main__":
    unittest.main()
