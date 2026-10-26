import io
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient
from object_inpainting.graph import NodeGraph
from object_inpainting.nodes.save_images import SaveImagesNode
from object_inpainting_studio.server import app
from PIL import Image


class SaveImagesTests(unittest.TestCase):
    def test_four_images_preserve_pixels_modes_and_share_filename(self):
        images = {
            "img1": Image.new("RGB", (5, 7), (10, 20, 30)),
            "img2": Image.new("RGBA", (6, 8), (40, 50, 60, 128)),
            "img3": Image.new("L", (4, 9), 127),
            "img4": Image.new("RGB", (3, 2), "blue"),
        }
        with TemporaryDirectory() as directory:
            paths = SaveImagesNode().execute(directory, **images)["paths"]
            self.assertEqual(set(paths), set(images))
            self.assertEqual(len({Path(path).name for path in paths.values()}), 1)
            for name, source in images.items():
                path = Path(paths[name])
                self.assertEqual(path.parent, Path(directory) / name)
                with Image.open(path) as saved:
                    self.assertEqual(
                        (saved.mode, saved.size, saved.tobytes()), (source.mode, source.size, source.tobytes())
                    )

    def test_skips_empty_inputs_without_renumbering_and_never_overwrites(self):
        with TemporaryDirectory() as directory:
            node = SaveImagesNode()
            image = Image.new("RGB", (2, 2), "red")
            first = node.execute(directory, img1=" ", img3=image)["paths"]
            second = node.execute(directory, img3=image)["paths"]
            self.assertEqual(set(first), {"img3"})
            self.assertNotEqual(first, second)
            self.assertEqual([path.name for path in Path(directory).iterdir()], ["img3"])
            self.assertEqual(len(list((Path(directory) / "img3").glob("*.png"))), 2)

    def test_invalid_inputs_do_not_create_output(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "output"
            for inputs in ({}, {"img2": 123}, {"img1": Image.new("RGB", (2, 2)), "img4": "not-an-image"}):
                with self.subTest(inputs=inputs), self.assertRaises(ValueError):
                    SaveImagesNode().execute(root, **inputs)
                self.assertFalse(root.exists())
            with self.assertRaises(ValueError):
                SaveImagesNode().execute(" ", img1=Image.new("RGB", (2, 2)))

    def test_api_save_preserves_upload_alpha_and_preview_cannot_write(self):
        client = TestClient(app)
        buffer = io.BytesIO()
        Image.new("RGBA", (4, 4), (12, 34, 56, 78)).save(buffer, format="PNG")
        upload = client.post("/api/upload", files={"file": ("cutout.png", buffer.getvalue(), "image/png")})
        self.assertEqual(upload.status_code, 200)
        with TemporaryDirectory() as directory:
            root = Path(directory) / "output"
            payload = {
                "node_id": "save",
                "workflow": {
                    "nodes": {
                        "save": {
                            "class_type": "SaveImagesNode",
                            "inputs": {"dir": str(root), "img2": upload.json()["path"]},
                        }
                    },
                    "outputs": {},
                },
            }
            preview = client.post("/api/node/preview", json=payload)
            self.assertEqual(preview.status_code, 422)
            self.assertFalse(root.exists())
            response = client.post("/api/node/save-images", json=payload)
            self.assertEqual(response.status_code, 200, response.text)
            files = list((root / "img2").glob("*.png"))
            self.assertEqual(len(files), 1)
            with Image.open(files[0]) as saved:
                self.assertEqual(saved.mode, "RGBA")
                self.assertEqual(saved.getpixel((0, 0)), (12, 34, 56, 78))
            self.assertFalse((root / "img1").exists())

    def test_graph_accepts_rgb_rgba_mask_links_and_blocks_side_effects(self):
        for source_type in ("RGB_IMAGE", "RGBA_IMAGE", "MASK"):
            from object_inpainting.nodes.base import Node

            class ImageNode(Node):
                OUTPUT_TYPES = {"image": source_type}

                def execute(self):
                    return {"image": Image.new("L", (3, 3), 128)}

            with self.subTest(source_type=source_type), TemporaryDirectory() as directory:
                graph = NodeGraph(
                    {
                        "nodes": {
                            "source": {"class_type": "ImageNode", "inputs": {}},
                            "save": {
                                "class_type": "SaveImagesNode",
                                "inputs": {"dir": directory, "img4": {"node": "source", "output": "image"}},
                            },
                        },
                        "outputs": {"paths": {"node": "save", "output": "paths"}},
                    },
                    registry={"ImageNode": ImageNode, "SaveImagesNode": SaveImagesNode},
                )
                with self.assertRaisesRegex(ValueError, "file-writing"):
                    graph.run({}, allow_side_effects=False)
                paths = graph.run({})["paths"]
                self.assertTrue(Path(paths["img4"]).is_file())

    def test_save_endpoint_rejects_other_writer_nodes(self):
        payload = {
            "node_id": "write",
            "workflow": {
                "nodes": {
                    "write": {"class_type": "SavePreviewNode", "inputs": {"sample": None, "path": "/unused.png"}}
                },
                "outputs": {},
            },
        }
        response = TestClient(app).post("/api/node/save-images", json=payload)
        self.assertEqual(response.status_code, 422)
        self.assertIn("仅允许", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
