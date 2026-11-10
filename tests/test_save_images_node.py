import io
import json
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi.testclient import TestClient
from object_inpainting.graph import NodeGraph
from object_inpainting.nodes.save_images import SaveImagesNode
from object_inpainting_studio.server import app
from PIL import Image


class SaveImagesTests(unittest.TestCase):
    def dynamic_workflow(self, directory, count=8):
        return {
            "nodes": {"save": {
                "class_type": "SaveImagesNode",
                "save_items": [{"id": f"img{i}", "name": f"view_{i}"} for i in range(1, count + 1)],
                "inputs": {"dir": str(directory), **{f"img{i}": {"context": f"img{i}"} for i in range(1, count + 1)}},
            }},
            "outputs": {"paths": {"node": "save", "output": "paths"}},
        }

    def test_dynamic_twelve_items_preserve_modes_names_and_shared_filename(self):
        with TemporaryDirectory() as directory:
            images = {f"img{i}": Image.new(("RGB", "RGBA", "L")[(i - 1) % 3], (i + 1, 3), i) for i in range(1, 13)}
            graph = NodeGraph(self.dynamic_workflow(directory, 12))
            with self.assertRaisesRegex(ValueError, "file-writing"):
                graph.run(images, allow_side_effects=False)
            self.assertFalse(list(Path(directory).iterdir()))
            paths = graph.run(images)["paths"]
            self.assertEqual(set(paths), set(images))
            self.assertEqual(len({Path(path).name for path in paths.values()}), 1)
            for key, path in paths.items():
                self.assertEqual(Path(path).parent.name, f"view_{key[3:]}")
                with Image.open(path) as saved:
                    self.assertEqual((saved.mode, saved.size, saved.tobytes()), (images[key].mode, images[key].size, images[key].tobytes()))

    def test_legacy_normalization_is_deep_copied_idempotent_and_round_trips(self):
        definition = {"nodes": {"save": {"class_type": "SaveImagesNode", "inputs": {"dir": "out"}}}, "outputs": {}}
        original = deepcopy(definition)
        graph = NodeGraph(definition)
        self.assertEqual(definition, original)
        spec = graph.nodes["save"]
        self.assertEqual([item["id"] for item in spec["save_items"]], ["img1", "img2", "img3", "img4"])
        self.assertEqual(spec["inputs"]["filename"], "")
        self.assertEqual(spec["save_next_id"], 5)
        self.assertEqual(NodeGraph(graph.definition).definition, graph.definition)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "workflow.json"
            graph.save(path)
            self.assertEqual(NodeGraph.from_file(path).definition, graph.definition)

    def test_explicit_items_do_not_resurrect_deleted_slots_and_keep_stable_keys(self):
        with TemporaryDirectory() as directory:
            definition = self.dynamic_workflow(directory)
            spec = definition["nodes"]["save"]
            spec["save_items"] = [{"id": "img3", "name": "目标图"}, {"id": "img8", "name": ""}]
            spec["save_next_id"] = 10
            spec["inputs"] = {"dir": directory, "img3": {"context": "image"}, "img8": None, "filename": "sample.jpg"}
            graph = NodeGraph(definition)
            self.assertNotIn("img1", graph.nodes["save"]["inputs"])
            first = graph.run({"image": Image.new("RGBA", (2, 3))})["paths"]
            self.assertEqual(set(first), {"img3"})
            self.assertEqual(Path(first["img3"]).resolve().relative_to(Path(directory).resolve()).as_posix(), "目标图/sample.png")
            normalized = deepcopy(graph.definition)
            normalized["nodes"]["save"]["save_items"][0]["name"] = "新名称"
            second = NodeGraph(normalized).run({"image": Image.new("RGBA", (2, 3))})["paths"]
            self.assertEqual(set(second), {"img3"})
            self.assertEqual(Path(second["img3"]).parent.name, "新名称")
            self.assertTrue(Path(first["img3"]).exists())

    def test_invalid_metadata_and_undeclared_inputs_are_rejected(self):
        bad_items = [None, {}, [{"id": "img0", "name": "target"}], [{"id": "img8", "name": 1}],
                     [{"id": "img8", "name": "a"}, {"id": "img8", "name": "b"}],
                     [{"id": "img8", "name": "Mask"}, {"id": "img9", "name": "mask"}],
                     [{"id": "img8", "name": "../escape"}], [{"id": "img8", "name": "a\\b"}],
                     [{"id": "img8", "name": "a\x00b"}]]
        for items in bad_items:
            with self.subTest(items=items), self.assertRaises(ValueError):
                NodeGraph({"nodes": {"save": {"class_type": "SaveImagesNode", "save_items": items, "inputs": {"dir": "out"}}}, "outputs": {}})
        definition = self.dynamic_workflow("out", 1)
        for key in ("img2", "anything"):
            invalid = deepcopy(definition)
            invalid["nodes"]["save"]["inputs"][key] = None
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "inputs must match"):
                NodeGraph(invalid)
        definition["nodes"]["save"]["save_next_id"] = 1
        with self.assertRaisesRegex(ValueError, "save_next_id"):
            NodeGraph(definition)

    def test_dynamic_ports_reject_incompatible_links_and_runtime_values(self):
        definition = self.dynamic_workflow("out", 8)
        definition["nodes"]["save"]["inputs"]["img8"] = {"node": "prompt", "output": "prompt"}
        definition["nodes"]["prompt"] = {"class_type": "BuildPromptNode", "inputs": {"object_name": "cup"}}
        with self.assertRaisesRegex(ValueError, "img8: expected IMAGE, got STRING"):
            NodeGraph(definition)
        with TemporaryDirectory() as directory:
            root = Path(directory) / "output"
            definition = self.dynamic_workflow(root, 1)
            with self.assertRaisesRegex(ValueError, "invalid runtime value for IMAGE"):
                NodeGraph(definition).run({"img1": 123})
            self.assertFalse(root.exists())

    def test_custom_filename_conflict_preflight_does_not_partially_write_or_overwrite(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "img8").mkdir()
            existing = root / "img8" / "sample.png"
            existing.write_bytes(b"original")
            node = SaveImagesNode()
            with self.assertRaisesRegex(ValueError, "已存在"):
                node.execute(root, img1=Image.new("RGB", (2, 2)), img8=Image.new("L", (2, 2)), filename="sample.jpg")
            self.assertEqual(existing.read_bytes(), b"original")
            self.assertFalse((root / "img1").exists())

    def test_filename_validation_and_direct_extra_slots(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "output"
            image = Image.new("RGB", (2, 2))
            for filename in ("../escape.png", "a\\b.png", "/tmp/a.png", ".", "..", "a\x00b", "bad:"):
                with self.subTest(filename=filename), self.assertRaises(ValueError):
                    SaveImagesNode().execute(root, img8=image, filename=filename)
                self.assertFalse(root.exists())
            paths = SaveImagesNode().execute(root, img25=image, filename="中文名.JPG")["paths"]
            self.assertEqual(Path(paths["img25"]).name, "中文名.png")
            with self.assertRaisesRegex(ValueError, "imgN"):
                SaveImagesNode().execute(root, wrong=image)

    def test_write_failure_rolls_back_only_new_files(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            original_open = Path.open
            def failing_open(path, *args, **kwargs):
                if path.parent.name == "img8" and args == ("xb",):
                    raise OSError("simulated disk failure")
                return original_open(path, *args, **kwargs)
            with patch.object(Path, "open", failing_open), self.assertRaisesRegex(OSError, "simulated"):
                SaveImagesNode().execute(root, img1=Image.new("RGB", (2, 2)), img8=Image.new("L", (2, 2)))
            self.assertFalse(list(root.rglob("*.png")))

    def test_symlink_escape_and_list_name_collision_fail_before_writing(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "output"
            outside = Path(directory) / "outside"
            root.mkdir()
            outside.mkdir()
            (root / "img8").symlink_to(outside, target_is_directory=True)
            image = Image.new("RGB", (2, 2))
            with self.assertRaisesRegex(ValueError, "符号链接"):
                SaveImagesNode().execute(root, img1=image, img8=image)
            self.assertFalse(list(outside.iterdir()))
            self.assertFalse((root / "img1").exists())
            spec = {"class_type": "SaveImagesNode", "save_items": [{"id": "img1", "name": "images_0"}],
                    "inputs": {"dir": str(root), "img1": {"context": "image"}, "images": {"context": "images"}}}
            graph = NodeGraph({"nodes": {"save": spec}, "outputs": {"paths": {"node": "save", "output": "paths"}}})
            with self.assertRaisesRegex(ValueError, "名称重复"):
                graph.run({"image": image, "images": [image]})
            self.assertFalse((root / "images_0").exists())

    def test_api_dynamic_upload_validate_and_named_save(self):
        client = TestClient(app)
        buffer = io.BytesIO()
        Image.new("RGBA", (3, 4), (12, 34, 56, 78)).save(buffer, format="PNG")
        upload = client.post("/api/upload", files={"file": ("cutout.png", buffer.getvalue(), "image/png")})
        self.assertEqual(upload.status_code, 200)
        with TemporaryDirectory() as directory:
            root = Path(directory) / "output"
            workflow = {"nodes": {"save": {"class_type": "SaveImagesNode",
                "save_items": [{"id": "img8", "name": "透明参考图"}], "save_next_id": 12,
                "inputs": {"dir": str(root), "img8": upload.json()["path"], "filename": "test.jpg"}}}, "outputs": {}}
            validation = client.post("/api/workflow/validate", json={"workflow": workflow, "require_dataset": False})
            self.assertEqual(validation.status_code, 200, validation.text)
            normalized = validation.json()["workflow"]
            self.assertNotIn("img1", normalized["nodes"]["save"]["inputs"])
            self.assertEqual(normalized["nodes"]["save"]["save_next_id"], 12)
            payload = {"workflow": json.loads(json.dumps(normalized)), "node_id": "save"}
            self.assertEqual(client.post("/api/node/preview", json=payload).status_code, 422)
            self.assertFalse(root.exists())
            response = client.post("/api/node/save-images", json=payload)
            self.assertEqual(response.status_code, 200, response.text)
            with Image.open(root / "透明参考图" / "test.png") as saved:
                self.assertEqual(saved.getpixel((0, 0)), (12, 34, 56, 78))
            self.assertIn("img8", response.json()["outputs"]["paths"]["items"])
            self.assertEqual(client.post("/api/node/save-images", json=payload).status_code, 422)

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
                self.assertEqual(path.parent.resolve(), (Path(directory) / name).resolve())
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
