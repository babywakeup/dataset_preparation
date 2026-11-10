import unittest

from PIL import Image

from object_inpainting.graph import NodeGraph
from object_inpainting.nodes import ColorSimilarityNode, NODE_CLASS_MAPPINGS


class ColorSimilarityNodeTests(unittest.TestCase):
    def setUp(self):
        self.node = ColorSimilarityNode()

    def test_identical_colors_yield_zero_score(self):
        red1 = Image.new("RGB", (32, 32), (255, 0, 0))
        red2 = Image.new("RGB", (64, 64), (255, 0, 0))
        result = self.node.execute(red1, red2)
        self.assertIsInstance(result, dict)
        self.assertIn("score", result)
        self.assertAlmostEqual(result["score"], 0.0, places=5)

    def test_black_and_white_yields_score_100(self):
        black = Image.new("RGB", (16, 16), (0, 0, 0))
        white = Image.new("RGB", (16, 16), (255, 255, 255))
        result = self.node.execute(black, white)
        self.assertAlmostEqual(result["score"], 100.0, places=4)

    def test_different_sizes_and_modes_handled_properly(self):
        # 32x32 vs 1x1, RGBA mode converted to RGB
        img1 = Image.new("RGBA", (32, 32), (100, 150, 200, 255))
        img2 = Image.new("RGB", (1, 1), (100, 150, 200))
        result = self.node.execute(img1, img2)
        self.assertAlmostEqual(result["score"], 0.0, places=4)

    def test_score_is_non_negative_and_bounded_in_srgb(self):
        green = Image.new("RGB", (10, 10), (0, 255, 0))
        magenta = Image.new("RGB", (10, 10), (255, 0, 255))
        result = self.node.execute(green, magenta)
        self.assertGreaterEqual(result["score"], 0.0)
        # CIEDE2000 in sRGB space has a maximum of ~119.03
        self.assertLess(result["score"], 120.0)
        self.assertGreater(result["score"], 110.0)

    def test_invalid_inputs_raise_value_error(self):
        valid = Image.new("RGB", (10, 10), (255, 255, 255))
        with self.assertRaises(ValueError):
            self.node.execute("not_an_image", valid)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            self.node.execute(valid, None)  # type: ignore[arg-type]

    def test_graph_executes_color_similarity_node(self):
        definition = {
            "nodes": {
                "similarity": {
                    "class_type": "ColorSimilarityNode",
                    "inputs": {
                        "image1": {"context": "image1"},
                        "image2": {"context": "image2"},
                    },
                }
            },
            "outputs": {
                "score": {"node": "similarity", "output": "score"},
            },
        }
        graph = NodeGraph(definition, registry=NODE_CLASS_MAPPINGS)
        color_a = Image.new("RGB", (16, 16), (255, 128, 0))
        color_b = Image.new("RGB", (16, 16), (255, 128, 0))
        outputs = graph.run({"image1": color_a, "image2": color_b})
        self.assertAlmostEqual(outputs["score"], 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
