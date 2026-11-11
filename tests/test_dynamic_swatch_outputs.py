"""CPU regressions for per-instance swatch ports and rejection of removed list ports."""

import base64
import io
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from object_inpainting.color_regions import segment_colors
from object_inpainting.graph import NodeExecutionError, NodeGraph
from object_inpainting.nodes import NODE_CLASS_MAPPINGS, ColorRegionSwatchesNode
from object_inpainting.nodes.auto_color_regions import AutoColorRegionsNode
from object_inpainting_studio import server
from object_inpainting_studio.batch import BatchJobs
from object_inpainting_studio.preview_cache import PreviewCache


def link(node, output):
    return {"node": node, "output": output}


def swatch_spec(indices=None):
    inputs = {"regions": {"context": "regions"}}
    if indices is not None:
        inputs["indices"] = indices
    return {"class_type": "ColorRegionSwatchesNode", "inputs": inputs}


class DynamicSwatchOutputsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.image = Image.new("RGB", (16, 16), "red")
        cls.image.paste("blue", (0, 0, 4, 16))
        cls.image.paste("lime", (0, 0, 4, 4))
        cls.mask = Image.new("L", cls.image.size, 255)
        cls.regions = segment_colors(cls.image, cls.mask)
        cls.colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0)]

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.image.save(self.root / "image.png")
        self.mask.save(self.root / "mask.png")
        self.cache_patch = patch.object(server, "PREVIEW_CACHE", PreviewCache())
        self.cache_patch.start()
        self.addCleanup(self.cache_patch.stop)
        self.client = TestClient(server.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def workflow(self, indices=None):
        return {"nodes": {"swatches": swatch_spec(indices)}, "outputs": {}}

    def api_workflow(self, indices=None):
        workflow = self.workflow(indices)
        workflow["nodes"]["regions"] = {
            "class_type": "AutoColorRegionsNode", "inputs": {
                "image": str(self.root / "image.png"), "mask": str(self.root / "mask.png"),
            },
        }
        workflow["nodes"]["swatches"]["inputs"]["regions"] = link("regions", "regions")
        return workflow

    def preview(self, workflow, node_id="swatches", session=None):
        response = self.client.post("/api/node/preview", json={
            "workflow": workflow, "node_id": node_id, "preview_session": session,
        })
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def assert_preview_image(self, item, color):
        self.assertEqual(item["kind"], "image")
        data = base64.b64decode(item["image"].split(",", 1)[1])
        with Image.open(io.BytesIO(data)) as image:
            self.assertEqual((image.mode, image.size), ("RGB", (32, 32)))
            self.assertEqual(image.getcolors(), [(1024, color)])

    def assert_saved_image(self, path, color):
        with Image.open(path) as image:
            self.assertEqual((image.mode, image.size), ("RGB", (32, 32)))
            self.assertEqual(image.getcolors(), [(1024, color)])

    def test_literal_output_hook_defaults_order_and_arbitrary_cardinality(self):
        for indices in (None, [0, 2, 1], [0] * 11):
            with self.subTest(indices=indices):
                graph = NodeGraph(self.workflow(indices))
                selected = [0] if indices is None else indices
                expected = {f"img{i}": "RGB_IMAGE" for i in range(1, len(selected) + 1)}
                expected["colors"] = "JSON_RECORD"
                spec = graph.nodes["swatches"]
                self.assertEqual(spec["inputs"]["indices"], selected)
                self.assertEqual(ColorRegionSwatchesNode.output_types(spec), expected)
                self.assertEqual(graph.output_types["swatches"], expected)
                result = graph.start({"regions": self.regions}).get_node("swatches")
                self.assertEqual(set(result), set(expected))
                self.assertNotIn("images", result)
                for position, index in enumerate(selected, 1):
                    self.assertEqual(result[f"img{position}"].getcolors(), [(1024, self.colors[index])])

    def test_direct_execute_preserves_order_duplicates_and_independent_pil_objects(self):
        indices = [2, 0, 2, 1, 0, 2]
        before = self.regions.image.tobytes(), self.regions.labels.tobytes()
        result = ColorRegionSwatchesNode().execute(self.regions, indices)
        self.assertEqual(set(result), {"colors", *(f"img{i}" for i in range(1, 7))})
        self.assertEqual(result["colors"]["indices"], indices)
        self.assertEqual(result["colors"]["region_ids"], [3, 1, 3, 2, 1, 3])
        self.assertEqual(result["colors"]["rgb"], [self.colors[i] for i in indices])
        self.assertEqual(result["colors"]["fractions"], [self.regions.regions[i].fraction for i in indices])
        images = [result[f"img{i}"] for i in range(1, 7)]
        self.assertEqual(len({id(image) for image in images}), 6)
        for image, index in zip(images, indices):
            self.assertEqual((image.mode, image.size), ("RGB", (32, 32)))
            self.assertEqual(image.getcolors(), [(1024, self.colors[index])])
        images[0].putpixel((0, 0), (1, 2, 3))
        self.assertEqual(images[2].getpixel((0, 0)), self.colors[2])
        self.assertEqual(images[5].getpixel((0, 0)), self.colors[2])
        result["colors"]["indices"].append(99)
        self.assertEqual(indices, [2, 0, 2, 1, 0, 2])
        self.assertEqual((self.regions.image.tobytes(), self.regions.labels.tobytes()), before)

    def test_instances_have_distinct_output_cardinalities_without_class_state_leaks(self):
        workflow = {"nodes": {"one": swatch_spec(), "many": swatch_spec([0] * 7)}, "outputs": {}}
        graph = NodeGraph(workflow)
        self.assertEqual(graph.output_types["one"], {"img1": "RGB_IMAGE", "colors": "JSON_RECORD"})
        self.assertEqual(len(graph.output_types["many"]), 8)
        self.assertIsNot(graph.output_types["one"], graph.output_types["many"])
        execution = graph.start({"regions": self.regions})
        self.assertEqual(len(execution.get_node("one")), 2)
        self.assertEqual(len(execution.get_node("many")), 8)
        again = NodeGraph(self.workflow())
        self.assertEqual(again.output_types["swatches"], graph.output_types["one"])
        self.assertNotIn("indices", workflow["nodes"]["one"]["inputs"])

    def test_consumer_serialized_before_dynamic_producer_resolves_last_port(self):
        workflow = {"nodes": {
            "score": {"class_type": "ColorSimilarityNode", "inputs": {
                "image1": link("swatches", "img8"), "image2": link("swatches", "img1"),
            }},
            "swatches": swatch_spec([0] * 8),
        }, "outputs": {"score": link("score", "score")}}
        graph = NodeGraph(workflow)
        self.assertLess(graph.order.index("swatches"), graph.order.index("score"))
        self.assertEqual(graph.run({"regions": self.regions}), {"score": 0.0})

    def test_separate_rgb_links_score_and_save_real_files(self):
        workflow = self.workflow([0, 2, 1])
        workflow["nodes"].update({
            "score": {"class_type": "ColorSimilarityNode", "inputs": {
                "image1": link("swatches", "img1"), "image2": link("swatches", "img2"),
            }},
            "save": {"class_type": "SaveImagesNode", "inputs": {
                **{f"img{i}": link("swatches", f"img{i}") for i in range(1, 4)},
                "dir": str(self.root / "new"), "filename": "sample.jpg",
            }},
        })
        workflow["outputs"] = {"score": link("score", "score"), "paths": link("save", "paths")}
        graph = NodeGraph(workflow)
        graph.require_outputs({"score": "FLOAT", "paths": "JSON_RECORD"})
        result = graph.run({"regions": self.regions})
        self.assertGreater(result["score"], 0)
        self.assertEqual(set(result["paths"]), {"img1", "img2", "img3"})
        for position, index in enumerate([0, 2, 1], 1):
            path = Path(result["paths"][f"img{position}"])
            self.assertTrue(path.samefile(self.root / "new" / f"img{position}" / "sample.png"))
            self.assert_saved_image(path, self.colors[index])
        self.assertEqual(len(list((self.root / "new").rglob("*.png"))), 3)

    def test_unknown_dynamic_output_links_rejected_in_consumers_and_graph_outputs(self):
        for output in ("img0", "img4", "img99", "img01", "missing"):
            for consumer in (False, True):
                with self.subTest(output=output, consumer=consumer):
                    workflow = self.workflow([0, 2, 1])
                    if consumer:
                        workflow["nodes"]["score"] = {"class_type": "ColorSimilarityNode", "inputs": {
                            "image1": link("swatches", output), "image2": link("swatches", "img1"),
                        }}
                    else:
                        workflow["outputs"]["bad"] = link("swatches", output)
                    with self.assertRaisesRegex(ValueError, "Unknown output port.*swatches"):
                        NodeGraph(workflow)

    def test_link_types_and_required_graph_output_types_are_checked(self):
        for output, destination, expected in (
            ("colors", "image1", "expected RGB_IMAGE, got JSON_RECORD"),
            ("img1", "images", "expected IMAGE_LIST, got RGB_IMAGE"),
        ):
            with self.subTest(output=output, destination=destination):
                workflow = self.workflow([0, 2, 1])
                if destination == "images":
                    spec = {"class_type": "SaveImagesNode", "inputs": {"dir": str(self.root / "bad")}}
                else:
                    spec = {"class_type": "ColorSimilarityNode", "inputs": {"image2": link("swatches", "img1")}}
                spec["inputs"][destination] = link("swatches", output)
                workflow["nodes"]["consumer"] = spec
                with self.assertRaisesRegex(ValueError, expected):
                    NodeGraph(workflow)
        workflow = self.workflow()
        workflow["outputs"]["image"] = link("swatches", "img1")
        graph = NodeGraph(workflow)
        graph.require_outputs({"image": "RGB_IMAGE"})
        with self.assertRaisesRegex(ValueError, "expected IMAGE_LIST, got RGB_IMAGE"):
            graph.require_outputs({"image": "IMAGE_LIST"})
        self.assertFalse((self.root / "bad").exists())

    def test_subclass_actual_return_fields_must_match_instance_output_contract(self):
        for bad_field in ("missing", "extra", "old_list"):
            with self.subTest(bad_field=bad_field):
                class WrongReturnSwatches(ColorRegionSwatchesNode):
                    def execute_with_spec(self, inputs, spec):
                        result = dict(super().execute_with_spec(inputs, spec))
                        if bad_field == "missing":
                            result.pop("img3")
                        elif bad_field == "extra":
                            result["img4"] = result["img1"]
                        else:
                            result = {"images": [result["img1"]], "colors": result["colors"]}
                        return result

                registry = {**NODE_CLASS_MAPPINGS, "ColorRegionSwatchesNode": WrongReturnSwatches}
                graph = NodeGraph(self.workflow([0, 2, 1]), registry=registry)
                with self.assertLogs("object_inpainting.graph", level="ERROR"):
                    with self.assertRaisesRegex(NodeExecutionError, "returned outputs must match") as caught:
                        graph.start({"regions": self.regions}).get_node("swatches")
                self.assertEqual(caught.exception.node_id, "swatches")
                self.assertEqual(caught.exception.error_type, "ValueError")

    def test_removed_list_port_is_not_enabled_by_old_metadata(self):
        workflow = self.workflow([0, 1])
        workflow["nodes"]["swatches"]["legacy_swatch_images"] = True
        graph = NodeGraph(workflow)
        self.assertNotIn("images", graph.output_types["swatches"])
        workflow["outputs"]["old_list"] = link("swatches", "images")
        with self.assertRaisesRegex(ValueError, "images"):
            NodeGraph(workflow)

    def test_normalization_is_deepcopied_and_disk_roundtrips(self):
        workflow = self.workflow([0, 2, 1])
        workflow["nodes"]["save"] = {"class_type": "SaveImagesNode", "inputs": {
            "img1": link("swatches", "img1"), "img2": link("swatches", "img2"),
            "dir": str(self.root / "out"), "filename": "sample.jpg",
        }}
        before = deepcopy(workflow)
        graph = NodeGraph(workflow)
        self.assertEqual(NodeGraph(graph.definition).definition, graph.definition)
        self.assertEqual(workflow, before)
        response = self.client.post("/api/workflow/validate", json={"workflow": workflow, "require_dataset": False})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["workflow"], graph.definition)
        path = self.root / "workflow.json"
        graph.save(path)
        loaded = NodeGraph.from_file(path)
        self.assertEqual(loaded.definition, graph.definition)
        self.assertEqual(loaded.output_types, graph.output_types)
        self.assertFalse((self.root / "out").exists())

    def test_old_list_consumers_and_graph_outputs_are_rejected(self):
        for graph_output in (False, True):
            workflow = self.workflow([0, 2, 1])
            if graph_output:
                workflow["outputs"] = {"old_list": link("swatches", "images")}
            else:
                workflow["nodes"]["save"] = {"class_type": "SaveImagesNode", "inputs": {
                    "images": link("swatches", "images"), "dir": str(self.root / "old"),
                }}
            with self.assertRaisesRegex(ValueError, "images"):
                NodeGraph(workflow)
            response = self.client.post("/api/workflow/validate", json={"workflow": workflow, "require_dataset": False})
            self.assertEqual(response.status_code, 422, response.text)
        self.assertFalse((self.root / "old").exists())

    def test_standalone_api_preview_exposes_only_new_imgn_images(self):
        result = self.preview(self.api_workflow([0, 2, 1]))
        self.assertEqual(set(result["outputs"]), {"img1", "img2", "img3", "colors"})
        for position, index in enumerate([0, 2, 1], 1):
            self.assert_preview_image(result["outputs"][f"img{position}"], self.colors[index])
        self.assertNotIn("images", result["outputs"])
        self.assertEqual(set(result["executed_nodes"]), {"regions", "swatches"})
        self.assertEqual(sorted(path.name for path in self.root.iterdir()), ["image.png", "mask.png"])

    def test_preview_does_not_restore_removed_port_from_unselected_consumers(self):
        workflow = self.api_workflow([0, 2, 1])
        workflow["nodes"]["save"] = {"class_type": "SaveImagesNode", "inputs": {
            "images": link("swatches", "images"), "dir": str(self.root / "never"),
        }}
        before = deepcopy(workflow)
        result = self.preview(workflow)
        self.assertEqual(set(result["outputs"]), {"img1", "img2", "img3", "colors"})
        response = self.client.post("/api/node/save-images", json={"workflow": workflow, "node_id": "save"})
        self.assertEqual(response.status_code, 422, response.text)
        self.assertEqual(workflow, before)
        self.assertFalse((self.root / "never").exists())

    def test_indices_change_invalidates_cached_swatch_but_reuses_untouched_ancestors(self):
        workflow = self.api_workflow([0, 2, 1])
        workflow["nodes"]["reference"] = swatch_spec([0])
        workflow["nodes"]["reference"]["inputs"]["regions"] = link("regions", "regions")
        workflow["nodes"]["score"] = {"class_type": "ColorSimilarityNode", "inputs": {
            "image1": link("swatches", "img1"), "image2": link("reference", "img1"),
        }}
        with patch.object(AutoColorRegionsNode, "execute", autospec=True, side_effect=AutoColorRegionsNode.execute) as segmentation, patch.object(
            ColorRegionSwatchesNode, "execute", autospec=True, side_effect=ColorRegionSwatchesNode.execute,
        ) as swatches:
            first = self.preview(workflow, "score")
            session = first["preview_session"]
            self.assertEqual(first["outputs"]["score"]["value"], 0.0)
            self.assertEqual((segmentation.call_count, swatches.call_count), (1, 2))
            workflow["nodes"]["swatches"]["inputs"]["indices"] = [2, 0]
            changed = self.preview(workflow, "score", session)
            self.assertEqual(changed["preview_session"], session)
            self.assertEqual(set(changed["executed_nodes"]), {"swatches", "score"})
            self.assertEqual(set(changed["reused_nodes"]), {"regions", "reference"})
            self.assertGreater(changed["outputs"]["score"]["value"], 0)
            self.assertEqual((segmentation.call_count, swatches.call_count), (1, 3))
            repeated = self.preview(workflow, "score", session)
            self.assertEqual(repeated["preview_session"], session)
            self.assertEqual(repeated["executed_nodes"], ["score"])
            self.assertEqual(set(repeated["reused_nodes"]), {"regions", "reference", "swatches"})
            self.assertEqual(repeated["outputs"], changed["outputs"])
            self.assertEqual((segmentation.call_count, swatches.call_count), (1, 3))
        cached = server.PREVIEW_CACHE.sessions[session][1]["swatches"][2]
        self.assertEqual(set(cached), {"img1", "img2", "colors"})
        self.assertEqual(cached["img1"].getpixel((0, 0)), self.colors[2])

    def test_nonliteral_or_invalid_indices_are_rejected_before_any_execution(self):
        invalid_values = ([], "[0,2,1]", 0, True, (0, 1), [True], [0.0], ["0"], [-1],
                          {"context": "indices"}, link("swatches", "colors"))
        with patch.object(ColorRegionSwatchesNode, "execute") as swatches, patch.object(AutoColorRegionsNode, "execute") as segmentation:
            for indices in invalid_values:
                with self.subTest(indices=indices):
                    with self.assertRaisesRegex(ValueError, "indices"):
                        NodeGraph(self.workflow(indices))
                    # JSON encodes tuples as arrays: only the direct Python API
                    # can distinguish an invalid tuple from a valid list.
                    if isinstance(indices, tuple):
                        continue
                    with self.assertLogs(level="ERROR"):
                        response = self.client.post("/api/node/preview", json={
                            "workflow": self.api_workflow(indices), "node_id": "swatches", "indices": [0],
                        })
                    self.assertEqual(response.status_code, 422, response.text)
                    self.assertIn("indices", response.json()["detail"])
            swatches.assert_not_called()
            segmentation.assert_not_called()
        self.assertEqual(sorted(path.name for path in self.root.iterdir()), ["image.png", "mask.png"])

    def test_folder_batch_saves_independent_imgn_and_missing_region_writes_nothing(self):
        source = self.root / "input"
        source.mkdir()
        self.image.save(source / "a.png")
        Image.new("RGB", self.image.size, "red").save(source / "b.png")
        self.image.save(source / "c.png")
        workflow = self.api_workflow([0, 2, 1])
        workflow["nodes"]["folder"] = {"class_type": "FolderImagesNode", "inputs": {"dir": str(source)}}
        workflow["nodes"]["regions"]["inputs"]["image"] = link("folder", "image")
        workflow["nodes"]["save"] = {"class_type": "SaveImagesNode", "inputs": {
            **{f"img{i}": link("swatches", f"img{i}") for i in range(1, 4)},
            "filename": link("folder", "filename"), "dir": str(self.root / "out"),
        }}
        manager = BatchJobs()
        with patch.object(server, "BATCH_JOBS", manager), self.assertLogs(level="ERROR"):
            response = self.client.post("/api/batch", json={
                "workflow": workflow, "node_id": "save", "continue_on_error": True,
            })
            self.assertEqual(response.status_code, 200, response.text)
            job_id = response.json()["id"]
            manager.threads[job_id].join(20)
            self.assertFalse(manager.threads[job_id].is_alive(), "CPU batch did not finish")
        job = manager.get(job_id)
        self.assertEqual((job["status"], job["succeeded"], job["failed"]), ("completed", 2, 1))
        failure = job["results"][1]
        self.assertEqual(failure["node_error"]["node_id"], "swatches")
        self.assertIn("越界", failure["node_error"]["message"])
        self.assertFalse(failure.get("paths"))
        for item, filename in ((job["results"][0], "a.png"), (job["results"][2], "c.png")):
            self.assertEqual(set(item["paths"]), {"img1", "img2", "img3"})
            for position, index in enumerate([0, 2, 1], 1):
                path = self.root / "out" / f"img{position}" / filename
                self.assertTrue(Path(item["paths"][f"img{position}"]).samefile(path))
                self.assert_saved_image(path, self.colors[index])
        self.assertEqual(len(list((self.root / "out").rglob("*.png"))), 6)
        self.assertEqual(list((self.root / "out").rglob("b.png")), [])


if __name__ == "__main__":
    unittest.main()
