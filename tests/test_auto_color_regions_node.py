import base64
import inspect
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from object_inpainting.color_regions import (
    ColorRegionSet,
    cluster_colors,
    color_bins,
    nearest_centers,
    render_color_regions,
    segment_colors,
    select_color_region,
    weighted_lab,
)
from object_inpainting.graph import NodeGraph
from object_inpainting.nodes import AutoColorRegionsNode, SelectColorRegionNode
from object_inpainting_studio.server import app, serialize_node_output
from PIL import Image


class AutoColorRegionsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rgb = np.full((24, 32, 3), (30, 190, 30), dtype=np.uint8)
        self.rgb[4:20, 4:16] = (255, 0, 0)
        self.rgb[4:20, 16:28] = (0, 0, 255)
        self.mask = np.zeros((24, 32), dtype=np.uint8)
        self.mask[4:20, 4:28] = 255
        self.image = Image.fromarray(self.rgb)
        self.mask_image = Image.fromarray(self.mask)

    def test_two_colors_cover_only_foreground_without_mutating_inputs(self) -> None:
        result = AutoColorRegionsNode().execute(self.image, self.mask_image)
        regions = result["regions"]
        self.assertIsInstance(regions, ColorRegionSet)
        self.assertEqual(len(regions.regions), 2)
        self.assertEqual({region.color for region in regions.regions}, {(255, 0, 0), (0, 0, 255)})
        self.assertAlmostEqual(sum(region.fraction for region in regions.regions), 1)
        np.testing.assert_array_equal(np.asarray(regions.labels) > 0, self.mask > 127)
        np.testing.assert_array_equal(np.asarray(self.image), self.rgb)
        np.testing.assert_array_equal(np.asarray(self.mask_image), self.mask)
        for output in (result["overlay"], result["palette"]):
            self.assertEqual((output.mode, output.size), ("RGB", self.image.size))
            np.testing.assert_array_equal(np.asarray(output)[self.mask == 0], self.rgb[self.mask == 0])

    def test_background_color_cannot_influence_foreground_partition(self) -> None:
        first = segment_colors(self.image, self.mask_image)
        changed = self.rgb.copy()
        changed[self.mask == 0] = (255, 255, 255)
        second = segment_colors(Image.fromarray(changed), self.mask_image)
        self.assertEqual(first.regions, second.regions)
        np.testing.assert_array_equal(first.labels, second.labels)

    def test_uniform_and_single_pixel_inputs_return_one_region(self) -> None:
        for size in ((1, 1), (40, 30)):
            result = segment_colors(Image.new("RGB", size, "white"), Image.new("L", size, 255))
            self.assertEqual(len(result.regions), 1)
            self.assertEqual(result.regions[0].fraction, 1)
            self.assertEqual(result.regions[0].color, (255, 255, 255))

    def test_tiny_distinct_color_is_not_removed_by_area(self) -> None:
        pixels = np.full((100, 100, 3), 255, dtype=np.uint8)
        pixels[0, 0] = (255, 0, 0)
        result = segment_colors(Image.fromarray(pixels), Image.new("L", (100, 100), 255))
        self.assertEqual(len(result.regions), 2)
        self.assertEqual([region.pixels for region in result.regions], [9999, 1])
        self.assertEqual(result.regions[1].color, (255, 0, 0))

    def test_same_color_can_be_disconnected(self) -> None:
        pixels = np.full((10, 30, 3), (255, 0, 0), dtype=np.uint8)
        pixels[:, 10:20] = (0, 0, 255)
        result = segment_colors(Image.fromarray(pixels), Image.new("L", (30, 10), 255))
        self.assertEqual(len(result.regions), 2)
        self.assertEqual([region.pixels for region in result.regions], [200, 100])
        labels = np.asarray(result.labels)
        np.testing.assert_array_equal(labels[:, :10], labels[:, 20:])

    def test_soft_mask_threshold_and_binary_mode(self) -> None:
        for mask in (Image.fromarray(np.array([[127, 128]], dtype=np.uint8)), Image.new("1", (2, 1), 1)):
            result = segment_colors(Image.new("RGB", (2, 1), "red"), mask)
            self.assertEqual(sum(r.pixels for r in result.regions), int((np.asarray(mask.convert("L")) > 127).sum()))

    def test_invalid_inputs_report_clear_errors(self) -> None:
        for image, mask, detail, message in (
            (None, self.mask_image, 1, "PIL"),
            (self.image.convert("RGBA"), self.mask_image, 1, "RGB"),
            (self.image, self.mask_image.convert("RGB"), 1, "灰度"),
            (self.image, Image.new("L", (2, 2), 255), 1, "尺寸"),
            (self.image, Image.new("L", self.image.size), 1, "为空"),
            (self.image, self.mask_image, True, "detail"),
            (self.image, self.mask_image, 3, "detail"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                segment_colors(image, mask, detail)

    def test_fine_split_is_deterministic_and_never_exceeds_internal_cap(self) -> None:
        pixels = np.random.default_rng(7).integers(0, 256, size=(30, 40, 3), dtype=np.uint8)
        image, mask = Image.fromarray(pixels), Image.new("L", (40, 30), 255)
        coarse = segment_colors(image, mask, 0)
        fine = segment_colors(image, mask, 2)
        repeated = segment_colors(image, mask, 2)
        self.assertLessEqual(len(coarse.regions), len(fine.regions))
        self.assertLessEqual(len(fine.regions), 32)
        self.assertEqual(fine.regions, repeated.regions)
        np.testing.assert_array_equal(fine.labels, repeated.labels)

    def test_lab_scale_and_weighted_bins_are_explicit(self) -> None:
        rgb = np.array([[255, 255, 255], [255, 255, 255], [0, 0, 0]], dtype=np.uint8)
        features = weighted_lab(rgb)
        self.assertAlmostEqual(features[0, 0], 50, places=3)
        self.assertAlmostEqual(features[2, 0], 0, places=3)
        centers, weights = color_bins(rgb, features)
        self.assertEqual(len(centers), 2)
        self.assertEqual(sorted(weights.tolist()), [1, 2])

    def test_nearest_centers_processes_more_than_one_chunk(self) -> None:
        pixels = np.zeros((32769, 3), dtype=np.float64)
        pixels[-1] = 10
        assignments = nearest_centers(pixels, np.array([[0, 0, 0], [10, 10, 10]], dtype=np.float64))
        self.assertEqual(assignments.shape, (32769,))
        self.assertEqual(int(assignments.sum()), 1)

    def test_select_region_produces_full_size_mask_and_original_rgb(self) -> None:
        regions = segment_colors(self.image, self.mask_image)
        masks = []
        for region in regions.regions:
            outputs = SelectColorRegionNode().execute(regions, region.id)
            mask, rgba = outputs["mask"], outputs["rgba"]
            self.assertEqual((mask.mode, rgba.mode), ("L", "RGBA"))
            self.assertEqual(mask.size, self.image.size)
            np.testing.assert_array_equal(np.asarray(rgba)[:, :, 3], mask)
            np.testing.assert_array_equal(np.asarray(rgba)[:, :, :3], self.rgb)
            masks.append(np.asarray(mask) > 0)
        np.testing.assert_array_equal(np.sum(masks, axis=0), self.mask > 0)
        for invalid in (0, 3, True, "1"):
            with self.assertRaisesRegex(ValueError, "region_id"):
                select_color_region(regions, invalid)
        with self.assertRaisesRegex(ValueError, "regions"):
            select_color_region({})

    def test_region_collection_has_dedicated_preview_serialization(self) -> None:
        result = segment_colors(self.image, self.mask_image)
        output = serialize_node_output(result)
        self.assertEqual((output["kind"], output["count"]), ("color_regions", 2))
        for item in output["items"]:
            self.assertEqual(item["fraction"], 0.5)
            for name, mode in (("mask", "L"), ("rgba", "RGBA")):
                content = base64.b64decode(item[name].split(",", 1)[1])
                with Image.open(io.BytesIO(content)) as preview:
                    self.assertEqual(preview.mode, mode)
                    self.assertLessEqual(max(preview.size), 256)

    def test_graph_runs_selection_and_saves_labels_without_binarizing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            definition = {
                "nodes": {
                    "colors": {
                        "class_type": "AutoColorRegionsNode",
                        "inputs": {"image": {"context": "image"}, "mask": {"context": "mask"}},
                    },
                    "select": {
                        "class_type": "SelectColorRegionNode",
                        "inputs": {"regions": {"node": "colors", "output": "regions"}},
                    },
                    "save": {
                        "class_type": "SaveImagesNode",
                        "inputs": {
                            "img1": {"node": "select", "output": "mask"},
                            "img2": {"node": "colors", "output": "labels"},
                            "dir": directory,
                        },
                    },
                },
                "outputs": {"paths": {"node": "save", "output": "paths"}},
            }
            paths = NodeGraph(definition).run({"image": self.image, "mask": self.mask_image})["paths"]
            with Image.open(paths["img2"]) as labels:
                self.assertEqual(set(np.unique(labels)), {0, 1, 2})
            definition["nodes"]["select"]["inputs"]["regions"] = {"context": "invalid"}
            with self.assertRaisesRegex(ValueError, "COLOR_REGIONS"):
                NodeGraph(definition).run({"image": self.image, "mask": self.mask_image, "invalid": {}})

    def test_studio_upload_preview_and_selection_need_no_sample_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory, TestClient(app) as client:
            root = Path(directory)
            self.image.save(root / "image.png")
            self.mask_image.save(root / "mask.png")
            workflow = {
                "nodes": {
                    "colors": {
                        "class_type": "AutoColorRegionsNode",
                        "inputs": {"image": str(root / "image.png"), "mask": str(root / "mask.png")},
                    },
                    "select": {
                        "class_type": "SelectColorRegionNode",
                        "inputs": {"regions": {"node": "colors", "output": "regions"}, "region_id": 2},
                    },
                },
                "outputs": {},
            }
            response = client.post("/api/node/preview", json={"workflow": workflow, "node_id": "colors"})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["outputs"]["regions"]["count"], 2)
            selected = client.post("/api/node/preview", json={"workflow": workflow, "node_id": "select"})
            self.assertEqual(selected.status_code, 200, selected.text)
            self.assertEqual(selected.json()["outputs"]["mask"]["mode"], "L")
            self.assertEqual(selected.json()["outputs"]["rgba"]["mode"], "RGBA")
            self.assertEqual(sorted(path.name for path in root.iterdir()), ["image.png", "mask.png"])

    def test_public_functions_have_input_output_annotations_and_descriptions(self) -> None:
        functions = (
            weighted_lab,
            color_bins,
            nearest_centers,
            cluster_colors,
            segment_colors,
            render_color_regions,
            select_color_region,
            AutoColorRegionsNode.execute,
            SelectColorRegionNode.execute,
        )
        for function in functions:
            with self.subTest(function=function.__name__):
                signature = inspect.signature(function)
                self.assertNotEqual(signature.return_annotation, inspect.Signature.empty)
                for name, parameter in signature.parameters.items():
                    if name != "self":
                        self.assertNotEqual(parameter.annotation, inspect.Parameter.empty)
                self.assertIn("Args:", function.__doc__)
                self.assertIn("Returns:", function.__doc__)


if __name__ == "__main__":
    unittest.main()
