import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from object_inpainting.graph import NodeGraph
from object_inpainting.nodes.crop import CropNode
from object_inpainting.nodes.gray_mask import GrayMaskNode
from object_inpainting.nodes.mask_dilation import MaskDilationNode
from object_inpainting.nodes.sam3_text_mask import SAM3TextMaskNode
from object_inpainting.workflow_migrations import migrate_sam3_dilation
from object_inpainting_studio.server import app


class MaskDilationNodeTests(unittest.TestCase):
    def setUp(self):
        self.node = MaskDilationNode()
        self.mask = Image.new("L", (12, 8))
        self.mask.paste(255, (3, 2, 8, 6))

    def test_steps_kernel_and_clipped_bbox(self):
        for steps, kernel, bbox in ((1, 3, [2, 1, 9, 7]), (2, 3, [1, 0, 10, 8]), (1, 5, [1, 0, 10, 8]), (3, 5, [0, 0, 12, 8])):
            with self.subTest(steps=steps, kernel=kernel):
                result = self.node.execute(self.mask, steps, kernel)
                expected = np.zeros((8, 12), dtype=np.uint8)
                expected[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 255
                np.testing.assert_array_equal(np.asarray(result["mask"]), expected)
                self.assertEqual(result["mask"].mode, "L")
                self.assertEqual(result["mask"].size, self.mask.size)
                self.assertEqual(result["bbox"], bbox)
                self.assertEqual(CropNode().execute(Image.new("RGB", self.mask.size), bbox)["cropped"].size,
                                 (bbox[2] - bbox[0], bbox[3] - bbox[1]))
                self.assertIn(f"膨胀 {steps} 次", result["quality"])
                self.assertIn(f"{kernel}×{kernel}", result["quality"])

    def test_noop_is_independent_copy_and_input_never_changes(self):
        before = self.mask.tobytes()
        for steps, kernel in ((0, 11), (3, 1), (2, 3)):
            result = self.node.execute(self.mask, steps, kernel)
            self.assertIsNot(result["mask"], self.mask)
            self.assertEqual(self.mask.tobytes(), before)
            if steps == 0 or kernel == 1:
                self.assertEqual(result["mask"].tobytes(), before)
                self.assertEqual(result["bbox"], [3, 2, 8, 6])

    def test_one_bit_and_grayscale_masks_are_supported(self):
        self.assertEqual(self.node.execute(self.mask.convert("1"))["mask"].getpixel((2, 1)), 255)
        gray = Image.new("L", (5, 5))
        gray.putpixel((2, 2), 128)
        result = self.node.execute(gray)
        self.assertEqual(result["mask"].getpixel((1, 1)), 128)
        self.assertEqual(set(result["mask"].getdata()), {0, 128})

    def test_empty_mask_stays_empty_with_zero_bbox(self):
        result = self.node.execute(Image.new("L", (12, 8)), 3, 11)
        self.assertIsNone(result["mask"].getbbox())
        self.assertEqual(result["bbox"], [0, 0, 0, 0])
        self.assertIn("0 → 0", result["quality"])

    def test_invalid_parameters_are_rejected(self):
        for key, values in {
            "dilation_steps": (-1, True, 1.5, "2", None),
            "dilation_kernel": (0, -3, 2, True, 3.0, "3", None),
        }.items():
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, key):
                    self.node.execute(self.mask, **{key: value})

    def test_invalid_mask_modes_are_rejected(self):
        for mask in (None, "mask.png", Image.new("RGB", (4, 4)), Image.new("RGBA", (4, 4)), Image.new("I", (4, 4))):
            with self.subTest(mask=mask), self.assertRaisesRegex(ValueError, "mask"):
                self.node.execute(mask)

    def test_expanded_mask_can_feed_gray_mask_without_expanding_reference_branch(self):
        image = Image.new("RGB", self.mask.size, "white")
        expanded = self.node.execute(self.mask)["mask"]
        edited = GrayMaskNode().execute(image, expanded)["source"]
        self.assertEqual(edited.getpixel((2, 1)), (128, 128, 128))
        self.assertEqual(self.mask.getpixel((2, 1)), 0)

    def test_catalog_and_practice_entry_expose_independent_controls(self):
        catalog = TestClient(app).get("/api/node-types").json()
        info = catalog["MaskDilationNode"]
        self.assertEqual(info["inputs"], {"mask": "MASK", "dilation_steps": "INT", "dilation_kernel": "INT"})
        self.assertEqual(info["input_defaults"], {"dilation_steps": 1, "dilation_kernel": 3})
        self.assertEqual(info["outputs"], {"mask": "MASK", "bbox": "BBOX", "quality": "STRING"})
        self.assertEqual(info["input_contract"], "MaskDilationInputs")
        self.assertEqual(info["help"]["parameters"], [])
        for key in info["inputs"]:
            self.assertIn(key, info["help"]["inputs"])
            self.assertTrue(info["port_labels"]["inputs"][key]["role"])
        self.assertNotIn("dilation_steps", catalog["SAM3TextMaskNode"]["help"]["inputs"])
        self.assertIn('data-starter="MaskDilationNode"', TestClient(app).get("/").text)

    def test_api_preview_mask_bbox_and_invalid_kernel_error(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "mask.png"
            self.mask.save(path)
            inputs = {"mask": str(path), "dilation_steps": 2, "dilation_kernel": 3}
            workflow = {"nodes": {"dilate": {"class_type": "MaskDilationNode", "inputs": inputs}}, "outputs": {}}
            response = TestClient(app).post("/api/node/preview", json={"workflow": workflow, "node_id": "dilate"})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["outputs"]["mask"]["mode"], "L")
            self.assertEqual(response.json()["outputs"]["bbox"]["value"], [1, 0, 10, 8])
            inputs["dilation_kernel"] = 4
            response = TestClient(app).post("/api/node/preview", json={"workflow": workflow, "node_id": "dilate"})
            self.assertEqual(response.status_code, 422, response.text)
            self.assertEqual(response.json()["node_error"]["node_id"], "dilate")
            self.assertIn("dilation_kernel", response.json()["detail"])

    def legacy_workflow(self):
        return {
            "nodes": {
                "sam": {"class_type": "SAM3TextMaskNode", "inputs": {
                    "image": {"context": "image"}, "text": "bag", "dilation_steps": 2, "dilation_kernel": 3,
                }},
                "gray": {"class_type": "GrayMaskNode", "inputs": {
                    "target": {"context": "image"}, "edit_mask": {"node": "sam", "output": "object_mask"},
                }},
            },
            "outputs": {"mask": {"node": "sam", "output": "object_mask"}, "bbox": {"node": "sam", "output": "bbox"}},
        }

    def test_legacy_graph_migrates_without_mutation_and_preserves_mask_and_bbox(self):
        workflow = self.legacy_workflow()
        before = deepcopy(workflow)
        graph = NodeGraph(workflow)
        self.assertEqual(workflow, before)
        self.assertEqual(graph.nodes["gray"]["inputs"]["edit_mask"], {"node": "sam_dilation", "output": "mask"})
        self.assertNotIn("dilation_steps", graph.nodes["sam"]["inputs"])
        with patch.object(SAM3TextMaskNode, "execute", return_value={"object_mask": self.mask, "bbox": [3, 2, 8, 6], "quality": "model"}):
            result = graph.run({"image": Image.new("RGB", self.mask.size)})
        self.assertEqual(result["bbox"], [1, 0, 10, 8])
        self.assertEqual(result["mask"].tobytes(), self.node.execute(self.mask, 2, 3)["mask"].tobytes())
        self.assertEqual(migrate_sam3_dilation(graph.definition), graph.definition)

    def test_migration_avoids_id_collision_and_preserves_disabled_old_default(self):
        workflow = self.legacy_workflow()
        del workflow["nodes"]["sam"]["inputs"]["dilation_steps"]
        workflow["nodes"]["sam_dilation"] = {"class_type": "LoadTargetNode", "inputs": {"path": "unused.png"}}
        migrated = migrate_sam3_dilation(workflow)
        self.assertEqual(migrated["nodes"]["sam_dilation"], workflow["nodes"]["sam_dilation"])
        self.assertEqual(migrated["nodes"]["sam_dilation_2"]["inputs"]["dilation_steps"], 0)
        self.assertEqual(migrated["outputs"]["bbox"]["node"], "sam_dilation_2")

    def test_multiple_legacy_sam_nodes_migrate_independently(self):
        workflow = self.legacy_workflow()
        workflow["nodes"]["other"] = deepcopy(workflow["nodes"]["sam"])
        workflow["outputs"]["other_mask"] = {"node": "other", "output": "object_mask"}
        migrated = migrate_sam3_dilation(workflow)
        self.assertEqual(migrated["outputs"]["other_mask"], {"node": "other_dilation", "output": "mask"})
        self.assertEqual(migrated["nodes"]["other_dilation"]["inputs"]["mask"], {"node": "other", "output": "object_mask"})

    def test_validation_returns_visible_migrated_graph(self):
        response = TestClient(app).post("/api/workflow/validate", json={"workflow": self.legacy_workflow(), "require_dataset": False})
        self.assertEqual(response.status_code, 200, response.text)
        normalized = response.json()["workflow"]
        self.assertIn("sam_dilation", normalized["nodes"])
        self.assertNotIn("dilation_kernel", normalized["nodes"]["sam"]["inputs"])


if __name__ == "__main__":
    unittest.main()
