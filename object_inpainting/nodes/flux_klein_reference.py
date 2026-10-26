import math
import os
from functools import lru_cache
from pathlib import Path
from threading import Lock

from PIL import Image, ImageOps

from .base import Node


MODEL_ID = "black-forest-labs/FLUX.2-klein-base-9B"
CACHE_DIR = Path(__file__).resolve().parents[4].parent / "hf_cache"
INFERENCE_LOCK = Lock()


@lru_cache(maxsize=1)
def get_flux_klein(model: str, device: str):
    import torch

    from diffusers import Flux2KleinPipeline

    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    pipeline = Flux2KleinPipeline.from_pretrained(
        model,
        dtype=dtype,
        cache_dir=str(CACHE_DIR),
        local_files_only=True,
    )
    return pipeline.to(device)


class FluxKleinReferenceNode(Node):
    INPUT_TYPES = {
        "prompt": "STRING",
        "reference_1": "RGB_IMAGE",
        "reference_2": "RGB_IMAGE",
        "reference_3": "RGB_IMAGE",
        "width": "INT",
        "height": "INT",
        "num_inference_steps": "INT",
        "guidance_scale": "FLOAT",
        "seed": "INT",
    }
    OUTPUT_TYPES = {"image": "RGB_IMAGE"}
    CATEGORY = "dataset"
    OPTIONAL_INPUTS = {"reference_1", "reference_2", "reference_3"}
    INPUT_DEFAULTS = {"width": 1024, "height": 1024, "num_inference_steps": 28, "guidance_scale": 4.0, "seed": 0}

    def execute(
        self,
        prompt: str,
        reference_1=None,
        reference_2=None,
        reference_3=None,
        width=1024,
        height=1024,
        num_inference_steps=28,
        guidance_scale=4.0,
        seed=0,
    ) -> dict:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt 不能为空。")
        prompt = prompt.strip()
        for name, value in (("width", width), ("height", height)):
            if type(value) is not int or value < 16 or value % 16:
                raise ValueError(f"{name} 必须是至少 16 的 16 倍数整数。")
        if type(num_inference_steps) is not int or num_inference_steps < 1:
            raise ValueError("num_inference_steps 必须是正整数。")
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("seed 必须是 0 到 2**63-1 之间的整数。")
        if isinstance(guidance_scale, bool) or not isinstance(guidance_scale, (int, float)):
            raise ValueError("guidance_scale 必须是有限的非负数。")
        if not math.isfinite(guidance_scale) or guidance_scale < 0:
            raise ValueError("guidance_scale 必须是有限的非负数。")

        references = []
        for name, value in (("reference_1", reference_1), ("reference_2", reference_2), ("reference_3", reference_3)):
            if value is None:
                continue
            if isinstance(value, (str, Path)):
                if isinstance(value, str) and not value.strip():
                    continue
                try:
                    with Image.open(Path(value).expanduser()) as opened:
                        value = ImageOps.exif_transpose(opened).copy()
                except (OSError, ValueError) as error:
                    raise ValueError(f"{name} 不是有效的图片路径：{value}") from error
            if isinstance(value, Image.Image):
                image = ImageOps.exif_transpose(value).convert("RGBA")
                if min(image.size) < 16:
                    raise ValueError(f"{name} 的宽高必须至少为 16 像素。")
                references.append(Image.alpha_composite(Image.new("RGBA", image.size, "white"), image).convert("RGB"))
                continue
            raise ValueError(f"{name} 必须是图片对象、文件路径或空值。")

        try:
            import torch

            requested = os.environ.get("OBJECT_INPAINTING_FLUX_DEVICE", "auto")
            device = (
                torch.device("cuda" if torch.cuda.is_available() else "cpu")
                if requested == "auto"
                else torch.device(requested)
            )
            if device.type not in {"cpu", "cuda"}:
                raise ValueError("FLUX 设备必须是 cpu、cuda 或 cuda:0 等逻辑设备。")
            if device.type == "cuda" and (
                not torch.cuda.is_available()
                or (device.index is not None and device.index >= torch.cuda.device_count())
            ):
                raise ValueError("FLUX CUDA 设备不可用，编号相对于 CUDA_VISIBLE_DEVICES。")
            model = os.environ.get("OBJECT_INPAINTING_FLUX_MODEL_DIR", MODEL_ID)
            if Path(model).expanduser().is_dir():
                model = str(Path(model).expanduser().resolve())
            with INFERENCE_LOCK, torch.inference_mode():
                pipeline = get_flux_klein(model, str(device))
                result = pipeline(
                    prompt=prompt,
                    image=references or None,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=float(guidance_scale),
                    generator=torch.Generator(device="cpu").manual_seed(seed),
                    output_type="pil",
                )
                image = result.images[0].convert("RGB")
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            raise ValueError(
                f"FLUX.2 Klein 加载或生成失败：{error}。请准备完整的 Klein-base-9B 本地权重或缓存，"
                "用 OBJECT_INPAINTING_FLUX_MODEL_DIR 指定目录；不会自动联网下载。"
            ) from error
        return {"image": image}
