import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from object_inpainting.nodes import NODE_CLASS_MAPPINGS, Node
from object_inpainting_studio import server
from object_inpainting_studio.preview_cache import PreviewCache


class CountingImageNode(Node):
    INPUT_TYPES = {"value": "INT"}
    OUTPUT_TYPES = {"image": "RGB_IMAGE"}
    calls = 0

    def execute(self, value):
        type(self).calls += 1
        return {"image": Image.new("RGB", (2, 2), (value, 0, 0))}


class ReadPixelNode(Node):
    INPUT_TYPES = {"image": "RGB_IMAGE"}
    OUTPUT_TYPES = {"value": "INT"}
    calls = 0

    def execute(self, image):
        type(self).calls += 1
        value = image.getpixel((0, 0))[0]
        if value == 99:
            raise ValueError("test failure")
        # A downstream implementation must not mutate the retained upstream result.
        image.putpixel((0, 0), (0, 0, 0))
        return {"value": value}


class PreviewCacheTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(server.app)
        self.cache = PreviewCache()
        self.addCleanup(patch.stopall)
        patch.object(server, "PREVIEW_CACHE", self.cache).start()
        patch.dict(NODE_CLASS_MAPPINGS, {
            "CountingImageNode": CountingImageNode, "ReadPixelNode": ReadPixelNode,
        }).start()
        CountingImageNode.calls = ReadPixelNode.calls = 0
        self.workflow = {"nodes": {
            "up": {"class_type": "CountingImageNode", "inputs": {"value": 12}},
            "down": {"class_type": "ReadPixelNode", "inputs": {
                "image": {"node": "up", "output": "image"},
            }},
        }, "outputs": {}}
        self.session = None

    def preview(self, node_id, expected=200, **extra):
        response = self.client.post("/api/node/preview", json={
            "workflow": self.workflow, "node_id": node_id,
            "preview_session": self.session, **extra,
        })
        self.assertEqual(response.status_code, expected, response.text)
        result = response.json()
        if response.status_code == 200:
            self.session = result["preview_session"]
        return result

    def test_downstream_reuses_real_upstream_result_and_preserves_cached_image(self):
        self.preview("up")
        for _ in range(2):
            result = self.preview("down")
            self.assertEqual(result["outputs"]["value"]["value"], 12)
            self.assertEqual(result["executed_nodes"], ["down"])
            self.assertEqual(result["reused_nodes"], ["up"])
        self.assertEqual(CountingImageNode.calls, 1)
        self.assertEqual(ReadPixelNode.calls, 2)

    def test_explicit_upstream_retest_reruns_and_invalidates_descendants(self):
        self.preview("down")
        self.preview("up")
        self.preview("down")
        self.assertEqual(CountingImageNode.calls, 2)
        self.assertEqual(ReadPixelNode.calls, 2)

    def test_upstream_retest_invalidates_previously_cached_intermediate_nodes(self):
        self.workflow["nodes"]["middle"] = self.workflow["nodes"].pop("down")
        self.workflow["nodes"]["regenerate"] = {
            "class_type": "CountingImageNode", "inputs": {
                "value": {"node": "middle", "output": "value"},
            },
        }
        self.workflow["nodes"]["down"] = {
            "class_type": "ReadPixelNode", "inputs": {
                "image": {"node": "regenerate", "output": "image"},
            },
        }
        self.preview("regenerate")
        self.assertEqual(self.preview("down")["reused_nodes"], ["up", "middle", "regenerate"])
        self.preview("up")
        result = self.preview("down")
        self.assertEqual(result["reused_nodes"], ["up"])
        self.assertEqual(result["executed_nodes"], ["middle", "regenerate", "down"])

    def test_folder_changes_invalidate_cached_source_without_changing_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "one.png"
            Image.new("RGB", (2, 2), (10, 0, 0)).save(path)
            self.workflow["nodes"]["up"] = {
                "class_type": "FolderImagesNode", "inputs": {"dir": directory},
            }
            self.preview("up")
            self.assertEqual(self.preview("down")["reused_nodes"], ["up"])
            Image.new("RGB", (3, 2), (30, 0, 0)).save(path)
            result = self.preview("down")
            self.assertEqual(result["executed_nodes"], ["up", "down"])
            self.assertEqual(result["outputs"]["value"]["value"], 30)

    def test_changed_input_or_connection_cannot_reuse_stale_result(self):
        self.preview("up")
        self.workflow["nodes"]["up"]["inputs"]["value"] = 42
        result = self.preview("down")
        self.assertEqual(result["executed_nodes"], ["up", "down"])
        self.assertEqual(result["outputs"]["value"]["value"], 42)
        self.workflow["nodes"]["other"] = deepcopy(self.workflow["nodes"]["up"])
        self.workflow["nodes"]["down"]["inputs"]["image"]["node"] = "other"
        self.preview("down")
        self.assertEqual(CountingImageNode.calls, 3)

    def test_sessions_are_isolated_and_unknown_or_expired_session_is_safe(self):
        self.preview("up")
        old = self.session
        self.session = "unknown-session"
        self.preview("down")
        self.assertNotEqual(self.session, old)
        self.assertEqual(CountingImageNode.calls, 2)
        self.cache.ttl = -1
        self.preview("down")
        self.assertEqual(CountingImageNode.calls, 3)

    def test_long_inference_does_not_expire_session_before_next_save(self):
        clock = [0]
        original = CountingImageNode.execute

        def slow_execute(node, value):
            clock[0] += 1000  # Longer than the 900-second idle TTL.
            return original(node, value)

        with patch("object_inpainting_studio.preview_cache.monotonic", side_effect=lambda: clock[0]), patch.object(
            CountingImageNode, "execute", autospec=True, side_effect=slow_execute,
        ):
            self.preview("up")
            result = self.preview("down")
        self.assertEqual(result["reused_nodes"], ["up"])
        self.assertEqual(CountingImageNode.calls, 1)

    def test_failed_downstream_keeps_good_ancestors_but_never_caches_failure(self):
        self.workflow["nodes"]["up"]["inputs"]["value"] = 99
        self.preview("up")
        self.preview("down", expected=422)
        self.preview("down", expected=422)
        self.assertEqual(CountingImageNode.calls, 1)
        self.assertEqual(ReadPixelNode.calls, 2)
        self.assertNotIn("down", self.cache.sessions[self.session][1])

    def test_external_image_change_invalidates_same_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            Image.new("RGB", (2, 2), (10, 0, 0)).save(path)
            self.workflow["nodes"]["up"] = {
                "class_type": "LoadTargetNode", "inputs": {"path": str(path)},
            }
            self.workflow["nodes"]["down"]["inputs"]["image"]["output"] = "target"
            self.preview("up")
            self.assertEqual(self.preview("down")["reused_nodes"], ["up"])
            Image.new("RGB", (3, 2), (30, 0, 0)).save(path)
            result = self.preview("down")
            self.assertEqual(result["executed_nodes"], ["up", "down"])
            self.assertEqual(result["outputs"]["value"]["value"], 30)

    def test_direct_upload_binding_and_config_change_invalidate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mask.png"
            Image.new("L", (3, 3), 255).save(path)
            self.workflow = {"nodes": {
                "up": {"class_type": "ExpandMaskNode", "inputs": {
                    "object_mask": str(path), "config": {"context": "config"},
                }},
                "down": {"class_type": "MaskDilationNode", "inputs": {
                    "mask": {"node": "up", "output": "edit_mask"},
                }},
            }, "outputs": {}}
            self.preview("up", config={"dilation_radius": 0})
            self.assertEqual(self.preview("down", config={"dilation_radius": 0})["reused_nodes"], ["up"])
            self.assertEqual(self.preview("down", config={"dilation_radius": 1})["reused_nodes"], [])
            Image.new("L", (4, 4), 255).save(path)
            self.assertEqual(self.preview("down", config={"dilation_radius": 1})["reused_nodes"], [])

    def test_save_reuses_ancestors_but_performs_each_write(self):
        self.preview("up")
        with tempfile.TemporaryDirectory() as directory:
            self.workflow["nodes"]["save"] = {"class_type": "SaveImagesNode", "inputs": {
                "img1": {"node": "up", "output": "image"}, "dir": directory,
            }}
            for _ in range(2):
                response = self.client.post("/api/node/save-images", json={
                    "workflow": self.workflow, "node_id": "save", "preview_session": self.session,
                })
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()["executed_nodes"], ["save"])
                self.assertEqual(response.json()["reused_nodes"], ["up"])
            self.assertEqual(len(list(Path(directory).rglob("*.png"))), 2)
            self.assertEqual(CountingImageNode.calls, 1)

    def test_cache_is_bounded_and_eviction_recomputes(self):
        self.cache.max_sessions = self.cache.max_nodes = 1
        self.preview("down")
        self.assertEqual(len(self.cache.sessions[self.session][1]), 1)
        self.preview("down")
        self.assertEqual(CountingImageNode.calls, 2)
        self.session = None
        self.preview("up")
        self.assertEqual(len(self.cache.sessions), 1)


if __name__ == "__main__":
    unittest.main()
