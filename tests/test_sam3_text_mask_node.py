import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from fastapi.testclient import TestClient
from object_inpainting.nodes import sam3_text_mask
from object_inpainting.nodes.crop import CropNode
from object_inpainting_studio.server import app
from PIL import Image
from transformers import BatchFeature, Sam3ImageProcessor, Sam3Model, Sam3Processor
from transformers.models.sam3.modeling_sam3 import Sam3ImageSegmentationOutput


class SAM3TextMaskNodeTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.model_dir = str(Path(self.directory.name).resolve())
        environment = patch.dict(
            os.environ,
            {"OBJECT_INPAINTING_SAM3_DEVICE": "cpu", "OBJECT_INPAINTING_SAM3_MODEL_DIR": self.model_dir},
        )
        environment.start()
        self.addCleanup(environment.stop)
        self.image = Image.new("RGB", (12, 8), "white")
        self.masks = torch.full((1, 3, 8, 12), -10.0)
        self.masks[0, 0, :2, :2] = 10
        self.masks[0, 1, 2:6, 3:8] = 10
        self.model = MagicMock()
        self.model.return_value = Sam3ImageSegmentationOutput(
            pred_logits=torch.tensor([[1.5, 3.0, -4.0]]),
            pred_boxes=torch.zeros((1, 3, 4)),
            pred_masks=self.masks,
        )
        self.processor = MagicMock(spec=Sam3Processor)
        self.processor.side_effect = lambda **kwargs: BatchFeature(
            {"pixel_values": torch.zeros((1, 3, 16, 16)), "input_ids": torch.tensor([[1, 2]])}
        )
        self.processor.post_process_instance_segmentation.side_effect = (
            Sam3ImageProcessor().post_process_instance_segmentation
        )

    def test_local_loaders_never_download_and_model_is_reused(self):
        sam3_text_mask.get_sam3.cache_clear()
        self.addCleanup(sam3_text_mask.get_sam3.cache_clear)
        self.model.to.return_value = self.model
        self.model.eval.return_value = self.model
        with (
            patch.object(Sam3Model, "from_pretrained", return_value=self.model) as load_model,
            patch.object(Sam3Processor, "from_pretrained", return_value=self.processor) as load_processor,
        ):
            self.assertEqual(sam3_text_mask.get_sam3(self.model_dir, "cpu"), (self.model, self.processor))
            sam3_text_mask.get_sam3(self.model_dir, "cpu")
        load_model.assert_called_once_with(self.model_dir, local_files_only=True, use_safetensors=True)
        load_processor.assert_called_once_with(self.model_dir, local_files_only=True)
        self.assertEqual(sam3_text_mask.MODEL_DIR.parent.name, "hf_cache")
        self.model.eval.assert_called_once()

    def test_missing_local_directory_does_not_call_remote_loaders(self):
        with (
            patch.object(Sam3Model, "from_pretrained") as load_model,
            self.assertRaisesRegex(ValueError, "本地模型目录不存在"),
        ):
            sam3_text_mask.get_sam3(str(Path(self.model_dir) / "missing"), "cpu")
        load_model.assert_not_called()

    def test_empty_text_is_rejected_before_loading(self):
        with patch.object(sam3_text_mask, "get_sam3") as loader:
            for text in (None, "", "   ", 1):
                with self.subTest(text=text), self.assertRaisesRegex(ValueError, "物体文本"):
                    sam3_text_mask.SAM3TextMaskNode().execute(self.image, text)
            loader.assert_not_called()

    def test_selects_best_instance_without_merging_and_bbox_can_feed_crop(self):
        with patch.object(sam3_text_mask, "get_sam3", return_value=(self.model, self.processor)) as loader:
            result = sam3_text_mask.SAM3TextMaskNode().execute(self.image, "  red bag  ")
        loader.assert_called_once_with(self.model_dir, "cpu")
        self.assertEqual(result["object_mask"].mode, "L")
        self.assertEqual(result["object_mask"].size, self.image.size)
        np.testing.assert_array_equal(
            np.asarray(result["object_mask"]), (self.masks[0, 1].numpy() > 0).astype(np.uint8) * 255
        )
        self.assertEqual(result["bbox"], [3, 2, 8, 6])
        self.assertEqual(CropNode().execute(self.image, result["bbox"])["cropped"].size, (5, 4))
        self.assertIn("匹配 2 个物体", result["quality"])
        self.assertIn("0.9526", result["quality"])
        self.assertIn("未合并", result["quality"])
        self.assertEqual(self.processor.call_args.kwargs["text"], "red bag")
        self.assertEqual(self.processor.call_args.kwargs["images"].mode, "RGB")
        self.assertEqual(
            self.processor.post_process_instance_segmentation.call_args.kwargs,
            {"threshold": 0.5, "mask_threshold": 0.5, "target_sizes": [(8, 12)]},
        )

    def test_no_matching_object_has_actionable_error(self):
        self.model.return_value.pred_logits = torch.full((1, 3), -10.0)
        with (
            patch.object(sam3_text_mask, "get_sam3", return_value=(self.model, self.processor)),
            self.assertRaisesRegex(ValueError, "未找到.*0.5"),
        ):
            sam3_text_mask.SAM3TextMaskNode().execute(self.image, "cup")

    def test_empty_selected_mask_is_not_accepted(self):
        self.masks.fill_(-10)
        with (
            patch.object(sam3_text_mask, "get_sam3", return_value=(self.model, self.processor)),
            self.assertRaisesRegex(ValueError, "Mask 为空"),
        ):
            sam3_text_mask.SAM3TextMaskNode().execute(self.image, "cup")

    def test_invalid_result_size_and_scores_are_rejected(self):
        for masks, scores, message in (
            (torch.ones((1, 4, 4)), torch.tensor([0.9]), "尺寸"),
            (torch.ones((1, 8, 12)), torch.tensor([float("nan")]), "无效置信度"),
        ):
            with (
                self.subTest(message=message),
                patch.object(sam3_text_mask, "get_sam3", return_value=(self.model, self.processor)),
                self.assertRaisesRegex(ValueError, message),
            ):
                self.processor.post_process_instance_segmentation.side_effect = None
                self.processor.post_process_instance_segmentation.return_value = [{"masks": masks, "scores": scores}]
                sam3_text_mask.SAM3TextMaskNode().execute(self.image, "cup")

    def test_local_loading_error_includes_directory(self):
        with (
            patch.object(sam3_text_mask, "get_sam3", side_effect=OSError("missing weights")),
            self.assertRaisesRegex(ValueError, "SAM3 本地加载") as error,
        ):
            sam3_text_mask.SAM3TextMaskNode().execute(self.image, "cup")
        self.assertIn(self.model_dir, str(error.exception))

    def test_invalid_device_is_rejected_before_loading(self):
        for device in ("invalid", "mps", "cuda:7"):
            with (
                self.subTest(device=device),
                patch.dict(os.environ, {"OBJECT_INPAINTING_SAM3_DEVICE": device}),
                patch.object(torch.cuda, "is_available", return_value=True),
                patch.object(torch.cuda, "device_count", return_value=1),
                patch.object(sam3_text_mask, "get_sam3") as loader,
                self.assertRaises(ValueError),
            ):
                sam3_text_mask.SAM3TextMaskNode().execute(self.image, "cup")
            loader.assert_not_called()

    def test_unavailable_cuda_is_rejected_before_loading(self):
        with (
            patch.dict(os.environ, {"OBJECT_INPAINTING_SAM3_DEVICE": "cuda:0"}),
            patch.object(torch.cuda, "is_available", return_value=False),
            patch.object(sam3_text_mask, "get_sam3") as loader,
            self.assertRaisesRegex(ValueError, "没有可用的 CUDA"),
        ):
            sam3_text_mask.SAM3TextMaskNode().execute(self.image, "cup")
        loader.assert_not_called()

    def test_api_preview_displays_image_bbox_and_confidence(self):
        path = Path(self.model_dir) / "image.png"
        self.image.save(path)
        workflow = {
            "nodes": {"sam": {"class_type": "SAM3TextMaskNode", "inputs": {"image": str(path), "text": "bag"}}},
            "outputs": {},
        }
        with patch.object(sam3_text_mask, "get_sam3", return_value=(self.model, self.processor)):
            response = TestClient(app).post("/api/node/preview", json={"workflow": workflow, "node_id": "sam"})
        self.assertEqual(response.status_code, 200, response.text)
        outputs = response.json()["outputs"]
        self.assertEqual(outputs["object_mask"]["kind"], "image")
        self.assertEqual(outputs["bbox"]["kind"], "text")
        self.assertEqual(outputs["quality"]["kind"], "text")
        self.assertEqual(response.json()["executed_nodes"], ["sam"])
        workflow["nodes"]["sam"]["inputs"]["text"] = " "
        with patch.object(sam3_text_mask, "get_sam3") as loader:
            response = TestClient(app).post("/api/node/preview", json={"workflow": workflow, "node_id": "sam"})
        self.assertEqual(response.status_code, 422)
        loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
