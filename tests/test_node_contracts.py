import inspect
import unittest
from pathlib import Path
from typing import Annotated, TypedDict, get_args, get_origin, get_type_hints, is_typeddict

from fastapi.testclient import TestClient
from object_inpainting import node_contracts as contracts
from object_inpainting.node_contracts import port_schema
from object_inpainting.nodes import NODE_CLASS_MAPPINGS
from object_inpainting.nodes.base import Node
from object_inpainting.port_types import ImageRole, MaskedRGBImage, MaskImage, OptionalImage, PortType, RGBImage
from object_inpainting_studio.server import app
from PIL import Image


class NodeContractTests(unittest.TestCase):
    def test_all_registered_nodes_have_input_and_output_classes(self):
        self.assertEqual(len(NODE_CLASS_MAPPINGS), 28)
        for name, node in NODE_CLASS_MAPPINGS.items():
            with self.subTest(node=name):
                self.assertTrue(is_typeddict(node.INPUT_SCHEMA))
                self.assertTrue(is_typeddict(node.OUTPUT_SCHEMA))
                self.assertEqual(node.INPUT_TYPES, port_schema(node.INPUT_SCHEMA))
                self.assertEqual(node.OUTPUT_TYPES, port_schema(node.OUTPUT_SCHEMA))

    def test_execute_parameters_exactly_match_input_contract_types(self):
        for name, node in NODE_CLASS_MAPPINGS.items():
            with self.subTest(node=name):
                parameters = inspect.signature(node.execute).parameters
                expected = get_type_hints(node.INPUT_SCHEMA, include_extras=True)
                actual = get_type_hints(node.execute, include_extras=True)
                dynamic = {key for key, parameter in parameters.items() if parameter.kind == inspect.Parameter.VAR_KEYWORD}
                if dynamic:
                    self.assertTrue(getattr(node, "DYNAMIC_IMAGE_INPUTS", False))
                    self.assertEqual(dynamic, {"extra_images"})
                    self.assertEqual(actual["extra_images"], OptionalImage)
                    self.assertTrue(node.optional_input("img8"))
                    self.assertFalse(node.optional_input("invalid"))
                self.assertEqual(set(parameters) - {"self"} - dynamic, set(expected))
                self.assertEqual({key: value for key, value in actual.items() if key != "return" and key not in dynamic}, expected)
                self.assertIs(actual["return"], node.OUTPUT_SCHEMA)

    def test_semantic_image_aliases_do_not_wrap_or_copy_pil_images(self):
        for alias, port_type in ((RGBImage, PortType.RGB_IMAGE), (MaskedRGBImage, PortType.RGB_IMAGE), (MaskImage, PortType.MASK)):
            self.assertIs(get_origin(alias), Annotated)
            self.assertIs(get_args(alias)[0], Image.Image)
            self.assertEqual(get_args(alias)[1], port_type)
        self.assertNotEqual(get_args(RGBImage)[2], get_args(MaskedRGBImage)[2])
        self.assertEqual(get_args(MaskedRGBImage)[2], ImageRole.MASKED_RGB)

    def test_gray_mask_and_recolor_contracts_do_not_confuse_rgb_with_mask(self):
        gray = get_type_hints(contracts.GrayMaskOutputs, include_extras=True)
        recolor_inputs = get_type_hints(contracts.FluxRandomRecolorInputs, include_extras=True)
        recolor_outputs = get_type_hints(contracts.RandomRecolorOutputs, include_extras=True)
        self.assertEqual(gray["source"], MaskedRGBImage)
        self.assertEqual(recolor_inputs["mask"], MaskedRGBImage)
        self.assertEqual(recolor_outputs["source"], RGBImage)
        self.assertEqual(recolor_outputs["mask"], MaskImage)
        image = Image.new("RGB", (2, 2), "red")
        result = NODE_CLASS_MAPPINGS["GrayMaskNode"]().execute(image, Image.new("L", image.size, 255))
        self.assertIsInstance(result, dict)
        self.assertIsInstance(result["source"], Image.Image)
        self.assertEqual(result["source"].getpixel((0, 0)), (128, 128, 128))

    def test_optional_defaults_remain_in_signature_and_graph_port_categories(self):
        for name, node in NODE_CLASS_MAPPINGS.items():
            for key, value in getattr(node, "INPUT_DEFAULTS", {}).items():
                parameter = inspect.signature(node.execute).parameters[key]
                # Graph defaults can supply otherwise-required standalone args,
                # e.g. FolderImagesNode.dir; annotations must not change that API.
                if parameter.default is inspect.Parameter.empty:
                    self.assertIn(key, node.INPUT_TYPES)
                    continue
                if name == "ColorRegionSwatchesNode" and key == "indices":
                    self.assertIsNone(parameter.default)
                else:
                    self.assertEqual(parameter.default, value, (name, key))
            for key in getattr(node, "OPTIONAL_INPUTS", ()):
                self.assertIsNone(inspect.signature(node.execute).parameters[key].default)
        self.assertEqual(NODE_CLASS_MAPPINGS["SaveImagesNode"].INPUT_TYPES["images"], "IMAGE_LIST")
        self.assertEqual(NODE_CLASS_MAPPINGS["FluxKleinReferenceNode"].INPUT_TYPES["reference_1"], "RGB_IMAGE")

    def test_legacy_output_class_imports_still_resolve_to_one_shared_contract(self):
        from object_inpainting.nodes.color_region_swatches import ColorSwatchOutputs, SwatchColors
        from object_inpainting.nodes.flux_random_recolor import RandomRecolorOutputs

        self.assertIs(ColorSwatchOutputs, contracts.ColorSwatchOutputs)
        self.assertIs(SwatchColors, contracts.SwatchColors)
        self.assertIs(RandomRecolorOutputs, contracts.RandomRecolorOutputs)

    def test_unknown_contract_types_fail_instead_of_silently_using_wrong_category(self):
        class BadContract(TypedDict):
            field: bytes

        with self.assertRaisesRegex(TypeError, "BadContract.field"):
            port_schema(BadContract)
        with self.assertRaisesRegex(TypeError, "TypedDict"):
            port_schema(dict)

    def test_legacy_custom_nodes_can_keep_explicit_string_port_tables(self):
        class LegacyNode(Node):
            INPUT_TYPES = {"value": "STRING"}
            OUTPUT_TYPES = {"value": "STRING"}

            def execute(self, value: str) -> dict[str, str]:
                return {"value": value}

        self.assertEqual(LegacyNode.INPUT_TYPES, {"value": "STRING"})
        self.assertEqual(LegacyNode().execute("ok"), {"value": "ok"})

    def test_studio_reports_contract_class_names_and_preserves_wire_types(self):
        with TestClient(app) as client:
            response = client.get("/api/node-types")
            self.assertEqual(response.status_code, 200)
            catalog = response.json()
            for name, node in NODE_CLASS_MAPPINGS.items():
                self.assertEqual(catalog[name]["input_contract"], node.INPUT_SCHEMA.__name__)
                self.assertEqual(catalog[name]["output_contract"], node.OUTPUT_SCHEMA.__name__)
                self.assertEqual(catalog[name]["inputs"], node.INPUT_TYPES)
                self.assertEqual(catalog[name]["outputs"], node.OUTPUT_TYPES)
        self.assertEqual(port_schema(contracts.FolderImagesOutputs)["path"], "PATH")
        self.assertIs(get_type_hints(contracts.SaveSampleOutputs)["manifest_path"], Path)


if __name__ == "__main__":
    unittest.main()
