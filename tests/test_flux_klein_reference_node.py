import base64
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient
from object_inpainting.graph import NodeGraph
from object_inpainting.nodes import flux_klein_reference
from object_inpainting_studio.server import app
from PIL import Image


class FluxKleinGenerationTests(unittest.TestCase):
    def setUp(self):
        self.calls = []

        def generate(**kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(images=[Image.new("RGB", (64, 64), "red")])

        self.loader = patch.object(flux_klein_reference, "get_flux_klein", return_value=generate)
        self.loader.start()
        self.addCleanup(self.loader.stop)

    def test_returns_generated_image_with_ordered_nonempty_references(self):
        first = Image.new("RGB", (32, 32), "blue")
        third = Image.new("RGB", (32, 32), "green")
        result = flux_klein_reference.FluxKleinReferenceNode().execute(
            "  a product photo  ", first, " ", third, width=64, height=64, seed=42
        )
        self.assertEqual(set(result), {"image"})
        self.assertEqual(result["image"].getpixel((0, 0)), (255, 0, 0))
        call = self.calls[0]
        self.assertEqual(call["prompt"], "a product photo")
        self.assertEqual([img.getpixel((0, 0)) for img in call["image"]], [(0, 0, 255), (0, 128, 0)])
        self.assertEqual(call["generator"].initial_seed(), 42)
        self.assertEqual((call["width"], call["height"]), (64, 64))

    def test_graph_accepts_omitted_references_and_uses_text_to_image(self):
        graph = NodeGraph(
            {
                "nodes": {"generate": {"class_type": "FluxKleinReferenceNode", "inputs": {"prompt": "a coat"}}},
                "outputs": {"image": {"node": "generate", "output": "image"}},
            }
        )
        result = graph.run({}, allow_side_effects=False)
        self.assertIsInstance(result["image"], Image.Image)
        self.assertIsNone(self.calls[0]["image"])
        self.assertEqual(self.calls[0]["num_inference_steps"], 28)
        self.assertEqual(self.calls[0]["guidance_scale"], 4.0)

    def test_invalid_inputs_do_not_start_inference(self):
        for inputs in (
            {"prompt": " "},
            {"prompt": "coat", "width": 63},
            {"prompt": "coat", "num_inference_steps": 0},
            {"prompt": "coat", "reference_1": 123},
            {"prompt": "coat", "guidance_scale": float("nan")},
        ):
            with self.subTest(inputs=inputs), self.assertRaises(ValueError):
                flux_klein_reference.FluxKleinReferenceNode().execute(**inputs)
        self.assertEqual(self.calls, [])

    def test_api_upload_and_empty_references_return_image_preview(self):
        client = TestClient(app)
        buffer = io.BytesIO()
        Image.new("RGB", (32, 32), "blue").save(buffer, format="PNG")
        uploaded = client.post("/api/upload", files={"file": ("reference.png", buffer.getvalue(), "image/png")})
        self.assertEqual(uploaded.status_code, 200, uploaded.text)
        workflow = {
            "nodes": {
                "generate": {
                    "class_type": "FluxKleinReferenceNode",
                    "inputs": {
                        "prompt": "a product photo",
                        "reference_1": "",
                        "reference_2": uploaded.json()["path"],
                        "reference_3": None,
                        "width": 64,
                        "height": 64,
                    },
                }
            },
            "outputs": {},
        }
        response = client.post("/api/node/preview", json={"workflow": workflow, "node_id": "generate"})
        self.assertEqual(response.status_code, 200, response.text)
        output = response.json()["outputs"]["image"]
        self.assertEqual(output["kind"], "image")
        with Image.open(io.BytesIO(base64.b64decode(output["image"].split(",", 1)[1]))) as image:
            self.assertEqual(image.getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(len(self.calls[0]["image"]), 1)


if __name__ == "__main__":
    unittest.main()
