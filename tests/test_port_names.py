import copy
import unittest

from fastapi.testclient import TestClient
from object_inpainting.nodes import NODE_CLASS_MAPPINGS
from object_inpainting_studio.port_names import port_labels
from object_inpainting_studio.server import app, node_catalog


class PortNamesTests(unittest.TestCase):
    def setUp(self):
        self.catalog = node_catalog()

    def label(self, node, direction, key):
        return self.catalog[node]["port_labels"][direction][key]

    def test_all_ports_have_metadata_without_schema_changes(self):
        for node, cls in NODE_CLASS_MAPPINGS.items():
            for direction, schema in (("inputs", cls.INPUT_TYPES), ("outputs", cls.OUTPUT_TYPES)):
                with self.subTest(node=node, direction=direction):
                    self.assertEqual(self.catalog[node][direction], schema)
                    labels = self.catalog[node]["port_labels"][direction]
                    self.assertEqual(set(labels), set(schema))
                    for label in labels.values():
                        self.assertEqual(set(label), {"name", "role", "kind"})
                        self.assertTrue(label["name"])
                        self.assertNotIn(" ", label["name"])

    def test_rgb_image_ports_share_format_name(self):
        ports = (
            ("LoadTargetNode", "outputs", "target"),
            ("FolderImagesNode", "outputs", "image"),
            ("CropNode", "outputs", "cropped"),
            ("SAM3TextMaskNode", "inputs", "image"),
            ("AutoColorRegionsNode", "inputs", "image"),
            ("GrayMaskNode", "inputs", "target"),
            ("FluxRandomRecolorNode", "outputs", "source"),
            ("FluxKleinReferenceNode", "outputs", "image"),
        )
        for node, direction, key in ports:
            with self.subTest(node=node, key=key):
                self.assertEqual(self.label(node, direction, key)["name"], "rgb_image")

    def test_gray_mask_chain_has_same_name_and_semantics(self):
        ports = (
            ("GrayMaskNode", "outputs", "source"),
            ("FluxRandomRecolorNode", "inputs", "mask"),
            ("FluxRandomRecolorNode", "outputs", "masked_source"),
            ("AssembleSampleNode", "inputs", "source"),
        )
        for node, direction, key in ports:
            label = self.label(node, direction, key)
            self.assertEqual(label["name"], "masked_rgb_image")
            self.assertEqual(label["kind"], "masked_rgb_image")
        self.assertIn("不是黑白", self.label("FluxRandomRecolorNode", "inputs", "mask")["role"])
        self.assertEqual(self.label("FluxRandomRecolorNode", "outputs", "mask")["name"], "mask")
        self.assertEqual(self.catalog["FluxRandomRecolorNode"]["inputs"]["mask"], "RGB_IMAGE")
        self.assertEqual(self.catalog["FluxRandomRecolorNode"]["outputs"]["mask"], "MASK")

    def test_masks_have_shared_format_and_distinct_roles(self):
        object_mask = self.label("AlignSquareNode", "inputs", "object_mask")
        edit_mask = self.label("AlignSquareNode", "inputs", "edit_mask")
        self.assertEqual(object_mask["name"], "mask")
        self.assertEqual(edit_mask["name"], "mask")
        self.assertNotEqual(object_mask["role"], edit_mask["role"])
        self.assertNotEqual(object_mask["kind"], edit_mask["kind"])
        self.assertEqual(self.label("AutoColorRegionsNode", "outputs", "labels")["name"], "label_map")

    def test_rgba_reference_and_image_list_chains_match(self):
        self.assertEqual(self.label("MaskToRGBANode", "outputs", "rgba")["name"], "rgba_image")
        self.assertEqual(self.label("RGBAToWhiteBackgroundNode", "inputs", "rgba")["name"], "rgba_image")
        for node, direction, key in (("ExtractReferenceNode", "outputs", "reference"), ("RGBAToWhiteBackgroundNode", "outputs", "rgb"), ("AssembleSampleNode", "inputs", "reference")):
            self.assertEqual(self.label(node, direction, key)["name"], "reference_rgb_image")
        for index in range(1, 4):
            label = self.label("FluxKleinReferenceNode", "inputs", f"reference_{index}")
            self.assertEqual(label["name"], f"reference_rgb_image_{index}")
            self.assertEqual(label["kind"], "reference_rgb_image")
        self.assertEqual(self.label("SaveImagesNode", "inputs", "images")["name"], "image_list")
        self.assertEqual(self.label("ColorRegionSwatchesNode", "outputs", "img1")["name"], "img1")
        self.assertEqual(self.label("ColorRegionSwatchesNode", "outputs", "img1")["kind"], "rgb_image")

    def test_special_rgb_visualizations_are_not_named_as_original_or_mask(self):
        for node, key, name in (("AutoColorRegionsNode", "overlay", "overlay_rgb_image"), ("AutoColorRegionsNode", "palette", "palette_rgb_image"), ("FluxRandomRecolorNode", "sampled_color", "color_rgb_image")):
            self.assertEqual(self.label(node, "outputs", key)["name"], name)

    def test_index_bases_and_file_roles_remain_explicit(self):
        self.assertIn("0-based", self.label("ColorRegionSwatchesNode", "inputs", "indices")["role"])
        self.assertIn("1-based", self.label("SelectColorRegionNode", "inputs", "region_id")["role"])
        self.assertEqual(self.label("SaveImagesNode", "inputs", "dir")["name"], "output_dir")
        self.assertEqual(self.label("FolderImagesNode", "inputs", "dir")["name"], "input_dir")

    def test_labels_do_not_mutate_schemas_and_unknown_ports_fall_back(self):
        inputs = {"image": "RGB_IMAGE", "new_parameter": "STRING"}
        outputs = {"result": "NEW_TYPE"}
        before = copy.deepcopy((inputs, outputs))
        result = port_labels("NewNode", inputs, outputs)
        result["inputs"]["image"]["name"] = "changed"
        self.assertEqual((inputs, outputs), before)
        self.assertEqual(result["outputs"]["result"]["name"], "result")
        self.assertEqual(result["inputs"]["new_parameter"]["name"], "new_parameter")

    def test_api_exposes_labels_and_keeps_legacy_workflow_ports_valid(self):
        with TestClient(app) as client:
            response = client.get("/api/node-types")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), self.catalog)
            definition = {
                "nodes": {
                    "gray": {"class_type": "GrayMaskNode", "inputs": {"target": None, "edit_mask": None}},
                    "recolor": {"class_type": "FluxRandomRecolorNode", "inputs": {"target": None, "mask": {"node": "gray", "output": "source"}}},
                },
                "outputs": {"source": {"node": "recolor", "output": "source"}},
            }
            from object_inpainting.graph import NodeGraph

            graph = NodeGraph(definition)
            self.assertEqual(graph.nodes["recolor"]["inputs"]["mask"], {"node": "gray", "output": "source"})
            self.assertEqual(graph.outputs, definition["outputs"])


if __name__ == "__main__":
    unittest.main()
