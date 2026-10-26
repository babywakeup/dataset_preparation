import os
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from fastapi.testclient import TestClient
from object_inpainting.nodes import sam2_mask
from object_inpainting_studio.server import app
from PIL import Image
from transformers import BatchFeature, Sam2Model, Sam2Processor


class SAM2MaskNodeTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (12, 8), "white")
        environment = patch.dict(os.environ, {"OBJECT_INPAINTING_SAM2_DEVICE": "cpu"})
        environment.start()
        self.addCleanup(environment.stop)
        self.mask = torch.zeros((1, 1, 8, 12), dtype=torch.bool)
        self.mask[0, 0, 2:6, 3:8] = True
        self.model = MagicMock()
        self.model.return_value.pred_masks = torch.zeros((1, 1, 1, 2, 2))
        self.model.return_value.iou_scores = torch.tensor([[[0.92]]])
        self.processor = MagicMock()
        self.processor.side_effect = lambda **kwargs: BatchFeature(
            {
                "pixel_values": torch.zeros((1, 3, 16, 16)),
                "input_boxes": torch.tensor(kwargs["input_boxes"]),
                "original_sizes": torch.tensor([[8, 12]]),
            }
        )
        self.processor.post_process_masks.return_value = [self.mask]

    def test_both_hf_loaders_receive_explicit_cache_and_model_is_reused(self):
        sam2_mask.get_sam2.cache_clear()
        self.addCleanup(sam2_mask.get_sam2.cache_clear)
        self.model.to.return_value = self.model
        self.model.eval.return_value = self.model
        with (
            patch.object(Sam2Model, "from_pretrained", return_value=self.model) as load_model,
            patch.object(Sam2Processor, "from_pretrained", return_value=self.processor) as load_processor,
        ):
            self.assertEqual(sam2_mask.get_sam2("cpu"), (self.model, self.processor))
            sam2_mask.get_sam2("cpu")
        load_model.assert_called_once_with(
            "facebook/sam2-hiera-tiny", cache_dir=str(sam2_mask.CACHE_DIR), use_safetensors=True
        )
        load_processor.assert_called_once_with("facebook/sam2-hiera-tiny", cache_dir=str(sam2_mask.CACHE_DIR))
        self.assertEqual(sam2_mask.CACHE_DIR.parent.name, "hf_cache")
        self.model.eval.assert_called_once()

    def test_invalid_boxes_do_not_load_model(self):
        with patch.object(sam2_mask, "get_sam2") as loader:
            for bbox in (
                None,
                [1, 2],
                [True, 0, 8, 6],
                ["1", 0, 8, 6],
                [0, 0, float("nan"), 6],
                [-1, 0, 8, 6],
                [0, 0, 13, 6],
                [8, 0, 2, 6],
            ):
                with self.subTest(bbox=bbox), self.assertRaises(ValueError):
                    sam2_mask.SAM2MaskNode().execute(self.image, bbox)
            loader.assert_not_called()

    def test_preserves_size_binary_pixels_and_uses_box_prompt(self):
        with patch.object(sam2_mask, "get_sam2", return_value=(self.model, self.processor)):
            result = sam2_mask.SAM2MaskNode().execute(self.image, [3, 2, 8, 6])
        self.assertEqual(result["object_mask"].size, self.image.size)
        self.assertEqual(result["object_mask"].mode, "L")
        np.testing.assert_array_equal(
            np.asarray(result["object_mask"]), self.mask[0, 0].numpy().astype(np.uint8) * 255
        )
        self.assertIn("0.9200", result["quality"])
        self.assertEqual(self.processor.call_args.kwargs["input_boxes"], [[[3.0, 2.0, 8.0, 6.0]]])
        self.assertFalse(self.model.call_args.kwargs["multimask_output"])
        self.assertNotIn("original_sizes", self.model.call_args.kwargs)

    def test_invalid_outputs_raise_clear_errors(self):
        for mask, score in (
            (torch.zeros_like(self.mask), 0.9),
            (torch.ones((1, 1, 4, 4)), 0.9),
            (self.mask, float("nan")),
        ):
            with (
                self.subTest(shape=mask.shape, score=score),
                patch.object(sam2_mask, "get_sam2", return_value=(self.model, self.processor)),
                self.assertRaises(ValueError),
            ):
                self.processor.post_process_masks.return_value = [mask]
                self.model.return_value.iou_scores = torch.tensor([[[score]]])
                sam2_mask.SAM2MaskNode().execute(self.image, [3, 2, 8, 6])

    def test_hf_download_errors_become_actionable_node_errors(self):
        with (
            patch.object(sam2_mask, "get_sam2", side_effect=OSError("offline")),
            self.assertRaisesRegex(ValueError, "HF SAM2"),
        ):
            sam2_mask.SAM2MaskNode().execute(self.image, [3, 2, 8, 6])

    def test_invalid_logical_cuda_device_is_rejected_before_loading(self):
        with (
            patch.dict(os.environ, {"OBJECT_INPAINTING_SAM2_DEVICE": "cuda:7", "CUDA_VISIBLE_DEVICES": "7"}),
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "device_count", return_value=1),
            patch.object(sam2_mask, "get_sam2") as loader,
            self.assertRaisesRegex(ValueError, "cuda:0"),
        ):
            sam2_mask.SAM2MaskNode().execute(self.image, [3, 2, 8, 6])
        loader.assert_not_called()

    def test_api_preview_renders_mask_and_quality(self):
        path = sam2_mask.CACHE_DIR.parent / "unused_image.png"
        workflow = {
            "nodes": {
                "sam": {
                    "class_type": "SAM2MaskNode",
                    "inputs": {"image": {"node": "load", "output": "target"}, "bbox": [3, 2, 8, 6]},
                },
                "load": {"class_type": "LoadTargetNode", "inputs": {"path": str(path)}},
            },
            "outputs": {},
        }
        with (
            patch("object_inpainting.nodes.load_target.load_target", return_value=self.image),
            patch.object(sam2_mask, "get_sam2", return_value=(self.model, self.processor)),
        ):
            response = TestClient(app).post("/api/node/preview", json={"workflow": workflow, "node_id": "sam"})
        self.assertEqual(response.status_code, 200, response.text)
        outputs = response.json()["outputs"]
        self.assertEqual(outputs["object_mask"]["kind"], "image")
        self.assertEqual(outputs["quality"]["kind"], "text")
        self.assertEqual(response.json()["executed_nodes"], ["load", "sam"])


if __name__ == "__main__":
    unittest.main()
