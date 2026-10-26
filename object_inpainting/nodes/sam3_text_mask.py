import os
from contextlib import nullcontext
from functools import lru_cache
from pathlib import Path
from threading import Lock

import numpy as np
from PIL import Image

from .base import Node


MODEL_DIR = Path(__file__).resolve().parents[4].parent / "hf_cache" / "sam3"
INFERENCE_LOCK = Lock()


@lru_cache(maxsize=1)
def get_sam3(model_dir: str, device: str):
    from transformers import Sam3Model, Sam3Processor

    if not Path(model_dir).is_dir():
        raise ValueError(f"SAM3 本地模型目录不存在：{model_dir}。请先通过 ModelScope 下载完整模型。")
    model = Sam3Model.from_pretrained(model_dir, local_files_only=True, use_safetensors=True).to(device).eval()
    processor = Sam3Processor.from_pretrained(model_dir, local_files_only=True)
    return model, processor


class SAM3TextMaskNode(Node):
    INPUT_TYPES = {"image": "RGB_IMAGE", "text": "STRING"}
    OUTPUT_TYPES = {"object_mask": "MASK", "bbox": "BBOX", "quality": "STRING"}
    CATEGORY = "dataset"

    def execute(self, image: Image.Image, text: str) -> dict:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("请输入要分割的物体文本，例如 cup、red bag 或 truck。")
        text = text.strip()

        try:
            import torch
        except ImportError as error:
            raise ValueError("SAM3 节点需要 torch 和支持 Sam3Model、Sam3Processor 的 transformers。") from error

        requested = os.environ.get("OBJECT_INPAINTING_SAM3_DEVICE", "auto")
        try:
            device = (
                torch.device("cuda" if torch.cuda.is_available() else "cpu")
                if requested == "auto"
                else torch.device(requested)
            )
        except (RuntimeError, ValueError) as error:
            raise ValueError("SAM3 设备应填写 cpu、cuda 或 cuda:0 等逻辑设备编号。") from error
        if device.type not in {"cpu", "cuda"}:
            raise ValueError("SAM3 当前支持 cpu 或 cuda 设备。")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("指定了 CUDA，但当前环境没有可用的 CUDA 设备。")
        if device.type == "cuda" and device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(
                f"{device} 不可用：当前只有 {torch.cuda.device_count()} 个可见 CUDA 设备。"
                "设备编号相对于 CUDA_VISIBLE_DEVICES；仅暴露一张 GPU 时请使用 cuda:0。"
            )

        model_dir = str(
            Path(os.environ.get("OBJECT_INPAINTING_SAM3_MODEL_DIR", str(MODEL_DIR))).expanduser().resolve()
        )
        width, height = image.size
        try:
            with INFERENCE_LOCK:
                model, processor = get_sam3(model_dir, str(device))
                inputs = processor(images=image.convert("RGB"), text=text, return_tensors="pt").to(device)
                use_bfloat16 = device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8
                autocast = torch.autocast("cuda", dtype=torch.bfloat16) if use_bfloat16 else nullcontext()
                with torch.inference_mode(), autocast:
                    outputs = model(**inputs)
                    result = processor.post_process_instance_segmentation(
                        outputs, threshold=0.5, mask_threshold=0.5, target_sizes=[(height, width)]
                    )[0]
                scores = result["scores"].detach().float().cpu().numpy()
                if not len(scores):
                    raise ValueError("SAM3 未找到置信度超过 0.5 的匹配物体，请换用更具体的简短英文描述。")
                if not np.isfinite(scores).all():
                    raise ValueError("SAM3 返回了无效置信度。")
                best = int(scores.argmax())
                mask = result["masks"][best].detach().cpu().numpy().astype(bool)
                quality = float(scores[best])
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            raise ValueError(f"SAM3 本地加载或分割失败（{model_dir}）：{error}") from error

        if mask.shape != (height, width):
            raise ValueError("SAM3 输出 Mask 尺寸与输入图片不一致。")
        if not mask.any():
            raise ValueError("SAM3 选中的物体 Mask 为空，请调整文本描述。")
        ys, xs = np.nonzero(mask)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        return {
            "object_mask": Image.fromarray(mask.astype(np.uint8) * 255),
            "bbox": bbox,
            "quality": (
                f"SAM3 文本：{text}；匹配 {len(scores)} 个物体，已选择置信度最高的单个实例；"
                f"预测置信度：{quality:.4f}（非实测准确率），未合并多个物体。"
            ),
        }
