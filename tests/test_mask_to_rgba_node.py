import base64
import io
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from fastapi.testclient import TestClient
from object_inpainting.graph import NodeGraph
from object_inpainting.nodes import NODE_CLASS_MAPPINGS
from object_inpainting.nodes.mask_to_rgba import MaskToRGBANode
from object_inpainting.nodes.rgba_to_white_background import RGBAToWhiteBackgroundNode
from object_inpainting_studio.server import app
from PIL import Image


class MaskToRGBANodeTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.fromarray(np.arange(18, dtype=np.uint8).reshape(2, 3, 3))
        self.mask = Image.fromarray(np.array([[0, 128, 255], [64, 192, 255]], dtype=np.uint8))

    def test_preserves_rgb_alpha_size_and_original_inputs(self):
        original = np.asarray(self.image).copy()
        alpha = np.asarray(self.mask).copy()
        result = MaskToRGBANode().execute(self.image, self.mask)
        self.assertEqual(set(result), {"rgba"})
        rgba = result["rgba"]
        self.assertEqual(rgba.mode, "RGBA")
        self.assertEqual(rgba.size, self.image.size)
        np.testing.assert_array_equal(np.asarray(rgba)[:, :, :3], original)
        np.testing.assert_array_equal(np.asarray(rgba)[:, :, 3], alpha)
        np.testing.assert_array_equal(np.asarray(self.image), original)
        np.testing.assert_array_equal(np.asarray(self.mask), alpha)
        self.assertEqual(self.image.mode, "RGB")

    def test_binary_and_fully_transparent_masks_are_supported(self):
        for mode, value, expected in (("1", 1, 255), ("L", 0, 0), ("L", 255, 255)):
            with self.subTest(mode=mode, value=value):
                mask = Image.new(mode, self.image.size, value)
                rgba = MaskToRGBANode().execute(self.image, mask)["rgba"]
                np.testing.assert_array_equal(np.asarray(rgba)[:, :, 3], expected)

    def test_rejects_size_mismatch_and_non_grayscale_masks(self):
        for mask, message in (
            (Image.new("L", (1, 1)), "尺寸必须一致"),
            (Image.new("RGB", self.image.size), "灰度"),
            (Image.new("F", self.image.size), "灰度"),
        ):
            with self.subTest(mode=mask.mode, size=mask.size), self.assertRaisesRegex(ValueError, message):
                MaskToRGBANode().execute(self.image, mask)

    def test_registers_and_executes_linked_graph_without_side_effects(self):
        self.assertIs(NODE_CLASS_MAPPINGS["MaskToRGBANode"], MaskToRGBANode)
        definition = {
            "nodes": {
                "load": {"class_type": "LoadTargetNode", "inputs": {"path": {"context": "image_path"}}},
                "cutout": {
                    "class_type": "MaskToRGBANode",
                    "inputs": {"image": {"node": "load", "output": "target"}, "mask": {"context": "mask"}},
                },
            },
            "outputs": {"cutout": {"node": "cutout", "output": "rgba"}},
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            self.image.save(path)
            graph = NodeGraph(definition)
            graph.require_outputs({"cutout": "RGBA_IMAGE"})
            result = graph.run({"image_path": str(path), "mask": self.mask}, allow_side_effects=False)
            self.assertEqual(list(Path(directory).iterdir()), [path])
        np.testing.assert_array_equal(np.asarray(result["cutout"])[:, :, 3], np.asarray(self.mask))

    def test_api_uploaded_grayscale_mask_keeps_partial_alpha_in_png(self):
        client = TestClient(app)
        paths = {}
        for name, image in (("image", self.image), ("mask", self.mask)):
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            response = client.post("/api/upload", files={"file": (f"{name}.png", buffer.getvalue(), "image/png")})
            self.assertEqual(response.status_code, 200, response.text)
            paths[name] = response.json()["path"]
        workflow = {
            "nodes": {"cutout": {"class_type": "MaskToRGBANode", "inputs": paths}},
            "outputs": {},
        }
        response = client.post("/api/node/preview", json={"workflow": workflow, "node_id": "cutout"})
        self.assertEqual(response.status_code, 200, response.text)
        output = response.json()["outputs"]["rgba"]
        self.assertEqual(output["kind"], "image")
        self.assertEqual(output["mode"], "RGBA")
        self.assertEqual(output["size"], [3, 2])
        with Image.open(io.BytesIO(base64.b64decode(output["image"].split(",", 1)[1]))) as rgba:
            np.testing.assert_array_equal(np.asarray(rgba)[:, :, :3], np.asarray(self.image))
            np.testing.assert_array_equal(np.asarray(rgba)[:, :, 3], np.asarray(self.mask))
        catalog = client.get("/api/node-types").json()["MaskToRGBANode"]
        self.assertEqual(catalog["outputs"], {"rgba": "RGBA_IMAGE"})
        self.assertIn("半透明", catalog["help"]["inputs"]["mask"])

    def test_other_mask_nodes_keep_existing_binary_upload_semantics(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "mask.png"
            self.mask.save(path)
            workflow = {
                "nodes": {
                    "expand": {
                        "class_type": "ExpandMaskNode",
                        "inputs": {"object_mask": str(path), "config": {"context": "config"}},
                    }
                },
                "outputs": {},
            }
            response = TestClient(app).post(
                "/api/node/preview",
                json={"workflow": workflow, "node_id": "expand", "config": {"dilation_radius": 0}},
            )
        self.assertEqual(response.status_code, 200, response.text)
        encoded = response.json()["outputs"]["edit_mask"]["image"]
        with Image.open(io.BytesIO(base64.b64decode(encoded.split(",", 1)[1]))) as mask:
            np.testing.assert_array_equal(np.asarray(mask), (np.asarray(self.mask) > 127).astype(np.uint8) * 255)


class RGBAToWhiteBackgroundNodeTests(unittest.TestCase):
    def test_composites_transparent_pixels_onto_white_background(self):
        rgba = Image.new("RGBA", (2, 2), (0, 0, 0, 0))
        rgba.putpixel((0, 0), (255, 0, 0, 255))
        rgba.putpixel((1, 1), (0, 0, 255, 128))

        result = RGBAToWhiteBackgroundNode().execute(rgba)

        self.assertEqual(set(result), {"rgb"})
        rgb = result["rgb"]
        self.assertEqual(rgb.mode, "RGB")
        self.assertEqual(rgb.size, rgba.size)
        self.assertEqual(rgb.getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(rgb.getpixel((1, 0)), (255, 255, 255))
        self.assertEqual(rgb.getpixel((1, 1)), (127, 127, 255))


if __name__ == "__main__":
    unittest.main()
