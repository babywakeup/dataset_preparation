import base64
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from object_inpainting.color_regions import segment_colors
from object_inpainting.graph import NodeGraph
from object_inpainting.nodes import ColorRegionSwatchesNode
from object_inpainting.nodes.save_images import SaveImagesNode
from object_inpainting_studio import server
from object_inpainting_studio.batch import BatchJobs
from object_inpainting_studio.server import app, serialize_node_output
from PIL import Image


class ColorRegionSwatchesTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (16, 16), "red")
        self.image.paste("blue", (0, 0, 4, 16))
        self.image.paste("lime", (0, 0, 4, 4))
        self.mask = Image.new("L", self.image.size, 255)
        self.regions = segment_colors(self.image, self.mask)
        self.node = ColorRegionSwatchesNode()

    def test_zero_based_top_two_are_independent_solid_rgb_32_square_images(self):
        result = self.node.execute(self.regions, [0, 1])
        self.assertEqual(set(result), {"img1", "img2", "colors"})
        for name, color in zip(("img1", "img2"), ((255, 0, 0), (0, 0, 255))):
            image = result[name]
            self.assertEqual((image.mode, image.size), ("RGB", (32, 32)))
            self.assertEqual(image.getcolors(), [(1024, color)])
        self.assertEqual(result["colors"]["indices"], [0, 1])
        self.assertEqual(result["colors"]["region_ids"], [1, 2])
        self.assertEqual(result["colors"]["rgb"], [(255, 0, 0), (0, 0, 255)])
        self.assertEqual(result["colors"]["fractions"], [192 / 256, 48 / 256])

    def test_input_order_duplicates_and_region_source_are_preserved(self):
        indices = [2, 0, 2]
        original = self.regions.image.tobytes(), self.regions.labels.tobytes()
        result = self.node.execute(self.regions, indices)
        self.assertEqual(result["colors"]["rgb"], [(0, 255, 0), (255, 0, 0), (0, 255, 0)])
        images = [result[f"img{position}"] for position in range(1, 4)]
        self.assertIsNot(images[0], images[2])
        images[0].putpixel((0, 0), (0, 0, 0))
        self.assertEqual(images[2].getpixel((0, 0)), (0, 255, 0))
        self.assertEqual((self.regions.image.tobytes(), self.regions.labels.tobytes()), original)
        self.assertEqual(indices, [2, 0, 2])

    def test_default_selects_only_largest_color(self):
        regions = segment_colors(Image.new("RGB", (16, 16), (12, 34, 56)), self.mask)
        result = self.node.execute(regions)
        self.assertEqual(result["colors"]["indices"], [0])
        self.assertEqual(set(result), {"img1", "colors"})
        self.assertEqual(result["img1"].getcolors(), [(1024, (12, 34, 56))])

    def test_invalid_lists_and_out_of_range_indices_report_clear_errors(self):
        for invalid in ([], "[0,1]", 0, True, (0, 1), [True], [0.0], ["0"]):
            with self.subTest(indices=invalid), self.assertRaisesRegex(ValueError, "indices"):
                self.node.execute(self.regions, invalid)
        for invalid in ([-1], [3], [0, 9]):
            with self.subTest(indices=invalid), self.assertRaisesRegex(ValueError, "越界"):
                self.node.execute(self.regions, invalid)
        with self.assertRaisesRegex(ValueError, "regions"):
            self.node.execute({"regions": []}, [0])

    def test_preview_serializes_each_image_not_pil_repr(self):
        result = self.node.execute(self.regions, [0, 1])
        for name, color in zip(("img1", "img2"), ((255, 0, 0), (0, 0, 255))):
            item = serialize_node_output(result[name])
            self.assertEqual(item["kind"], "image")
            with Image.open(io.BytesIO(base64.b64decode(item["image"].split(",", 1)[1]))) as image:
                self.assertEqual((image.mode, image.size), ("RGB", (32, 32)))
                self.assertEqual(image.getpixel((0, 0)), color)

    def test_graph_saves_separate_swatches_alongside_existing_single_image_ports(self):
        with tempfile.TemporaryDirectory() as directory:
            graph = NodeGraph({
                "nodes": {
                    "regions": {"class_type": "AutoColorRegionsNode", "inputs": {"image": {"context": "image"}, "mask": {"context": "mask"}}},
                    "swatches": {"class_type": "ColorRegionSwatchesNode", "inputs": {"regions": {"node": "regions", "output": "regions"}, "indices": [0, 1]}},
                    "save": {"class_type": "SaveImagesNode", "inputs": {"img1": {"node": "swatches", "output": "img1"}, "img2": {"node": "swatches", "output": "img2"}, "img4": {"context": "image"}, "dir": directory}},
                },
                "outputs": {"paths": {"node": "save", "output": "paths"}},
            })
            paths = graph.run({"image": self.image, "mask": self.mask})["paths"]
            self.assertEqual(set(paths), {"img4", "img1", "img2"})
            self.assertEqual(len({Path(path).name for path in paths.values()}), 1)
            for name, color in (("img1", (255, 0, 0)), ("img2", (0, 0, 255))):
                with Image.open(paths[name]) as saved:
                    self.assertEqual((saved.mode, saved.size), ("RGB", (32, 32)))
                    self.assertEqual(saved.getcolors(), [(1024, color)])

    def test_bad_image_list_does_not_create_output_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "out"
            for invalid in ("images", {"image": self.image}, [self.image, None], [self.image, "bad"]):
                with self.subTest(images=invalid), self.assertRaisesRegex(ValueError, "images"):
                    SaveImagesNode().execute(root, img1=self.image, images=invalid)
                self.assertFalse(root.exists())
            with self.assertRaises(ValueError):
                SaveImagesNode().execute(root, images=[])
            self.assertFalse(root.exists())

    def test_image_list_preserves_modes_and_does_not_overwrite_prior_files(self):
        images = [Image.new("RGBA", (4, 6), (12, 34, 56, 78)), Image.new("L", (3, 5), 127)]
        with tempfile.TemporaryDirectory() as directory:
            first = SaveImagesNode().execute(directory, images=images)["paths"]
            second = SaveImagesNode().execute(directory, images=images)["paths"]
            self.assertNotEqual(first, second)
            self.assertEqual(len(list(Path(directory).rglob("*.png"))), 4)
            for index, source in enumerate(images):
                path = Path(first[f"images_{index}"])
                self.assertEqual(path.parent.name, f"images_{index}")
                with Image.open(path) as saved:
                    self.assertEqual((saved.mode, saved.size, saved.tobytes()), (source.mode, source.size, source.tobytes()))

    def test_api_catalog_preview_save_and_error_origin(self):
        with tempfile.TemporaryDirectory() as directory, TestClient(app) as client:
            root = Path(directory)
            self.image.save(root / "image.png")
            self.mask.save(root / "mask.png")
            catalog = client.get("/api/node-types").json()
            self.assertEqual(catalog["ColorRegionSwatchesNode"]["input_defaults"]["indices"], [0])
            self.assertEqual(catalog["SaveImagesNode"]["inputs"]["images"], "IMAGE_LIST")
            workflow = {
                "nodes": {
                    "regions": {"class_type": "AutoColorRegionsNode", "inputs": {"image": str(root / "image.png"), "mask": str(root / "mask.png")}},
                    "swatches": {"class_type": "ColorRegionSwatchesNode", "inputs": {"regions": {"node": "regions", "output": "regions"}, "indices": [0, 1]}},
                    "save": {"class_type": "SaveImagesNode", "inputs": {"img1": {"node": "swatches", "output": "img1"}, "img2": {"node": "swatches", "output": "img2"}, "dir": str(root / "out")}},
                },
                "outputs": {},
            }
            response = client.post("/api/node/preview", json={"workflow": workflow, "node_id": "swatches"})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(set(response.json()["outputs"]), {"img1", "img2", "colors"})
            self.assertFalse((root / "out").exists())
            response = client.post("/api/node/save-images", json={"workflow": workflow, "node_id": "save"})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(len(list((root / "out").rglob("*.png"))), 2)
            workflow["nodes"]["swatches"]["inputs"]["indices"] = [99]
            with self.assertLogs(level="ERROR"):
                response = client.post("/api/node/preview", json={"workflow": workflow, "node_id": "swatches"})
            self.assertEqual(response.status_code, 422, response.text)
            self.assertEqual(response.json()["node_error"]["node_id"], "swatches")
            self.assertIn("越界", response.json()["detail"])

    def test_folder_loop_saves_swatches_and_reports_missing_rank_per_image(self):
        manager = BatchJobs()
        with tempfile.TemporaryDirectory() as directory, patch.object(server, "BATCH_JOBS", manager), TestClient(app) as client:
            root = Path(directory)
            source = root / "input"
            source.mkdir()
            self.image.save(source / "a.png")
            Image.new("RGB", self.image.size, "red").save(source / "b.png")
            self.image.save(source / "c.png")
            self.mask.save(root / "mask.png")
            workflow = {
                "nodes": {
                    "folder": {"class_type": "FolderImagesNode", "inputs": {"dir": str(source)}},
                    "regions": {"class_type": "AutoColorRegionsNode", "inputs": {"image": {"node": "folder", "output": "image"}, "mask": str(root / "mask.png")}},
                    "swatches": {"class_type": "ColorRegionSwatchesNode", "inputs": {"regions": {"node": "regions", "output": "regions"}, "indices": [0, 1]}},
                    "save": {"class_type": "SaveImagesNode", "inputs": {"img1": {"node": "swatches", "output": "img1"}, "img2": {"node": "swatches", "output": "img2"}, "dir": str(root / "out")}},
                },
                "outputs": {},
            }
            with self.assertLogs(level="ERROR"):
                response = client.post("/api/batch", json={"workflow": workflow, "node_id": "save", "continue_on_error": True})
                self.assertEqual(response.status_code, 200, response.text)
                job_id = response.json()["id"]
                manager.threads[job_id].join(20)
                self.assertFalse(manager.threads[job_id].is_alive())
            job = manager.get(job_id)
            self.assertEqual((job["status"], job["succeeded"], job["failed"]), ("completed", 2, 1))
            self.assertEqual(job["results"][1]["node_error"]["node_id"], "swatches")
            self.assertEqual(len(list((root / "out").rglob("*.png"))), 4)
            self.assertNotEqual(job["results"][0]["paths"], job["results"][2]["paths"])
            for item in (job["results"][0], job["results"][2]):
                self.assertEqual(set(item["paths"]), {"img1", "img2"})


if __name__ == "__main__":
    unittest.main()
