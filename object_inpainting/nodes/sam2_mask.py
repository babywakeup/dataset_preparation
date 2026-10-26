import os
from contextlib import nullcontext
from functools import lru_cache
from numbers import Real
from pathlib import Path
from threading import Lock

import numpy as np
from PIL import Image

from .base import Node


MODEL_ID = "facebook/sam2-hiera-tiny"
CACHE_DIR = Path(__file__).resolve().parents[4].parent / "hf_cache" / "hub"
INFERENCE_LOCK = Lock()


@lru_cache(maxsize=1)
def get_sam2(device: str):
    from transformers import Sam2Model, Sam2Processor

    model = Sam2Model.from_pretrained(MODEL_ID, cache_dir=str(CACHE_DIR), use_safetensors=True).to(device).eval()
    processor = Sam2Processor.from_pretrained(MODEL_ID, cache_dir=str(CACHE_DIR))
    return model, processor


class SAM2MaskNode(Node):
    INPUT_TYPES = {"image": "RGB_IMAGE", "bbox": "BBOX"}
    OUTPUT_TYPES = {"object_mask": "MASK", "quality": "STRING"}
    CATEGORY = "dataset"

    def execute(self, image: Image.Image, bbox) -> dict:
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError("物体框必须是 [左, 上, 右, 下]。")
        if any(isinstance(value, bool) or not isinstance(value, Real) for value in bbox):
            raise ValueError("物体框坐标必须是数字。")
        box = np.asarray(bbox, dtype=np.float64)
        if not np.isfinite(box).all():
            raise ValueError("物体框坐标必须是有限数值。")
        left, top, right, bottom = box
        width, height = image.size
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            raise ValueError("物体框必须在图片范围内，并具有正的宽高。")

        try:
            import torch
        except ImportError as error:
            raise ValueError("SAM2 节点需要安装 torch 和支持 SAM2 的 transformers。") from error

        requested = os.environ.get("OBJECT_INPAINTING_SAM2_DEVICE", "auto")
        device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if requested == "auto"
            else torch.device(requested)
        )
        if device.type not in {"cpu", "cuda"}:
            raise ValueError("SAM2 当前支持 cpu 或 cuda 设备。")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("指定了 CUDA，但当前环境没有可用的 CUDA 设备。")
        if device.type == "cuda" and device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(
                f"{device} 不可用：当前只有 {torch.cuda.device_count()} 个可见 CUDA 设备。"
                "设备编号相对于 CUDA_VISIBLE_DEVICES；仅暴露一张 GPU 时请使用 cuda:0。"
            )

        try:
            with INFERENCE_LOCK:
                model, processor = get_sam2(str(device))
                inputs = processor(images=image.convert("RGB"), input_boxes=[[box.tolist()]], return_tensors="pt").to(
                    device
                )
                original_sizes = inputs.pop("original_sizes")
                use_bfloat16 = device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8
                autocast = torch.autocast("cuda", dtype=torch.bfloat16) if use_bfloat16 else nullcontext()
                with torch.inference_mode(), autocast:
                    outputs = model(**inputs, multimask_output=False)
                masks = processor.post_process_masks(
                    outputs.pred_masks.float().cpu(), original_sizes.cpu(), binarize=True
                )
                mask = masks[0][0, 0].detach().cpu().numpy().astype(bool)
                quality = float(outputs.iou_scores[0, 0, 0].float().cpu())
        except (ImportError, OSError, RuntimeError) as error:
            raise ValueError(f"HF SAM2 加载或推理失败：{error}") from error

        if mask.shape != (height, width):
            raise ValueError("SAM2 输出 Mask 尺寸与输入图片不一致。")
        if not mask.any():
            raise ValueError("SAM2 未分割出物体，请调整物体框。")
        if not np.isfinite(quality):
            raise ValueError("SAM2 返回了无效质量分数。")
        return {
            "object_mask": Image.fromarray(mask.astype(np.uint8) * 255),
            "quality": f"SAM2 预测质量分数：{quality:.4f}（非实测准确率）",
        }
