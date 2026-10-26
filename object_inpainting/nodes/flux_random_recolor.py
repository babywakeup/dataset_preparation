import random
import secrets
from typing import TypedDict

import numpy as np
from PIL import Image, ImageOps

from ..masking import make_source
from .base import Node
from .flux_klein_reference import FluxKleinReferenceNode


RECOLOR_PALETTE = (
    ("burgundy", (128, 32, 64)),
    ("scarlet red", (200, 40, 48)),
    ("burnt orange", (192, 88, 40)),
    ("mustard yellow", (192, 152, 40)),
    ("olive green", (112, 128, 56)),
    ("forest green", (40, 112, 72)),
    ("teal", (32, 136, 136)),
    ("sky blue", (96, 168, 208)),
    ("royal blue", (48, 80, 184)),
    ("navy blue", (40, 56, 104)),
    ("violet", (120, 64, 160)),
    ("dusty pink", (192, 128, 152)),
    ("camel brown", (176, 128, 80)),
    ("chocolate brown", (112, 72, 48)),
    ("cream", (224, 216, 184)),
    ("charcoal gray", (72, 80, 88)),
)


class RandomRecolorOutputs(TypedDict):
    """随机换色输出；sampled_color 是新 source 的颜色条件，不是恢复原 target 的训练条件。"""

    source: Image.Image
    masked_source: Image.Image
    mask: Image.Image
    sampled_color: Image.Image
    color_name: str
    actual_seed: int
    generation_prompt: str


def choose_recolor(image: Image.Image, mask: Image.Image, seed: int) -> tuple[str, tuple[int, int, int]]:
    """从预设色表随机选择一个与编辑区中位 RGB 有明显差异的候选色。

    Args:
        image (PIL.Image.Image): 原尺寸 RGB 图片。
        mask (PIL.Image.Image): 同尺寸二值 L Mask，255 为编辑区，必须非空。
        seed (int): 已解析的非负随机种子，不改变全局随机状态。

    Returns:
        tuple[str, tuple[int, int, int]]: 英文颜色名与 0～255 的 sRGB 色块值。
        优先选择 RGB 欧氏距离至少 80 的颜色；这只是候选筛选，不保证模型实际色差。
    """
    pixels = np.asarray(image)[np.asarray(mask) > 0]
    median = np.median(pixels, axis=0)
    distances = [float(np.sum((np.asarray(rgb, dtype=np.float64) - median) ** 2)) for _, rgb in RECOLOR_PALETTE]
    candidates = [color for color, distance in zip(RECOLOR_PALETTE, distances) if distance >= 80**2]
    if not candidates:
        candidates = [RECOLOR_PALETTE[int(np.argmax(distances))]]
    return random.Random(seed).choice(candidates)


def recover_recolor_mask(target: Image.Image, masked_target: Image.Image) -> Image.Image:
    """从原图与 GrayMaskNode 涂灰图的差异恢复可辨认的编辑区域。

    Args:
        target (PIL.Image.Image): RGB 原图。
        masked_target (PIL.Image.Image): 同尺寸 RGB 涂灰图，修改像素须为 (128, 128, 128)，
            其余像素须与 target 完全一致；不接受黑白 Mask 或有损压缩图。

    Returns:
        PIL.Image.Image: 同尺寸二值 L Mask，发生涂灰变化的位置为 255，其余为 0。
        原图本来就是精确 128 灰色的位置无法辨别是否被涂灰，保守保留，不推测补全。

    Raises:
        ValueError: 图片类型、模式、尺寸、涂灰内容不正确，或没有可恢复的编辑像素。
    """
    if not isinstance(target, Image.Image) or target.mode != "RGB":
        raise ValueError("target 必须是 RGB PIL 图片。")
    if not isinstance(masked_target, Image.Image) or masked_target.mode != "RGB":
        raise ValueError("mask 必须是 GrayMaskNode.source 输出的 RGB 涂灰原图，不接受黑白 Mask。")
    if target.size != masked_target.size:
        raise ValueError("target 与 mask 涂灰原图尺寸必须一致。")
    original = np.asarray(target)
    masked = np.asarray(masked_target)
    changed = np.any(original != masked, axis=2)
    if not np.any(changed):
        raise ValueError("mask 涂灰原图与 target 没有可辨认差异，无法恢复编辑区域；原色已是 128 灰色也无法辨认。")
    if not np.all(masked[changed] == 128):
        raise ValueError("mask 必须是 GrayMaskNode 的涂灰原图：变化像素须为 RGB(128,128,128)，其余与 target 一致。")
    return Image.fromarray(changed.astype(np.uint8) * 255)


class FluxRandomRecolorNode(Node):
    """真实调用 FLUX.2 Klein 随机修改指定衣服区域的颜色，用于反向构造 source。"""

    INPUT_TYPES = {
        "target": "RGB_IMAGE",
        "mask": "RGB_IMAGE",
        "seed": "INT",
        "num_inference_steps": "INT",
        "guidance_scale": "FLOAT",
    }
    INPUT_DEFAULTS = {"seed": -1, "num_inference_steps": 28, "guidance_scale": 4.0}
    OUTPUT_TYPES = {
        "source": "RGB_IMAGE",
        "masked_source": "RGB_IMAGE",
        "mask": "MASK",
        "sampled_color": "RGB_IMAGE",
        "color_name": "STRING",
        "actual_seed": "INT",
        "generation_prompt": "STRING",
    }
    CATEGORY = "dataset"

    def execute(
        self,
        target: Image.Image,
        mask: Image.Image,
        seed: int = -1,
        num_inference_steps: int = 28,
        guidance_scale: float = 4.0,
    ) -> RandomRecolorOutputs:
        """用原图、编辑区涂灰原图和随机色块参考进行换色，强制复用 Mask 外原像素。

        Args:
            target (PIL.Image.Image): RGB 原图，作为真实训练 target，不会被修改，宽高至少 16。
            mask (PIL.Image.Image): 同尺寸 RGB 涂灰原图，直接连接 GrayMaskNode.source；
                变化区域必须为 128 灰色，不接受黑白 Mask。
            seed (int): -1 每次随机；0～2**32-1 固定候选颜色和推理随机种子，默认 -1。
            num_inference_steps (int): FLUX.2 Klein base 去噪步数，默认 28，必须为正整数。
            guidance_scale (float): 有限非负引导强度，默认 4.0。

        Returns:
            RandomRecolorOutputs: source 为原尺寸 RGB 换色结果；masked_source 为其编辑区涂灰图；
            mask 为从涂灰差异恢复的二值 L 图；sampled_color 为 128×128 RGB 随机色块；color_name、actual_seed、
            generation_prompt 记录本次生成条件。不写文件，不保证 Mask 内纹理或色值严格一致。

        Raises:
            ValueError: 输入、参数、Mask 或生成尺寸不正确，或本地模型加载/推理失败。
        """
        binary_mask = recover_recolor_mask(target, mask)
        if min(target.size) < 16:
            raise ValueError("target 宽高必须至少为 16 像素。")
        if type(seed) is not int or not -1 <= seed < 2**32:
            raise ValueError("seed 必须是 -1（随机）或 0～2**32-1 的整数。")
        actual_seed = secrets.randbelow(2**32) if seed == -1 else seed
        color_name, rgb = choose_recolor(target, binary_mask, actual_seed)
        sampled_color = Image.new("RGB", (128, 128), rgb)
        prompt = (
            "Edit image 1, the original photograph. Image 2 is the same photograph with only the editable "
            "clothing region covered in uniform gray. The region replaced by gray compared with image 1 "
            "marks where to recolor; all other regions must stay unchanged, including naturally gray objects. "
            "Image 3 is the desired color swatch. "
            f"Recolor only the marked clothing region to {color_name}, matching image 3. "
            "Use image 1, not the gray fill in image 2, to preserve all original clothing details. "
            "Preserve the exact person's identity, pose, garment silhouette, seams, buttons, prints, logos, "
            "fabric weave, texture, folds, shadows, highlights, camera framing, and background. "
            "Change fabric color only, not material or geometry. Do not add objects, text, mask markings, "
            "or the swatch to the output. Return only the edited photograph."
        )
        width, height = target.size
        padding = (0, 0, (-width) % 16, (-height) % 16)
        reference = ImageOps.expand(target, border=padding, fill=(128, 128, 128))
        masked_reference = ImageOps.expand(mask, border=padding, fill=(128, 128, 128))
        generated = FluxKleinReferenceNode().execute(
            prompt=prompt,
            reference_1=reference,
            reference_2=masked_reference,
            reference_3=sampled_color,
            width=reference.width,
            height=reference.height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=actual_seed,
        )["image"]
        if generated.size != reference.size:
            raise ValueError("FLUX 生成尺寸与请求不一致，无法安全对齐 source 与 target。")
        generated = generated.crop((0, 0, width, height)).convert("RGB")
        source = Image.composite(generated, target, binary_mask)
        masked_source = make_source(source, binary_mask)
        return {
            "source": source,
            "masked_source": masked_source,
            "mask": binary_mask,
            "sampled_color": sampled_color,
            "color_name": color_name,
            "actual_seed": actual_seed,
            "generation_prompt": prompt,
        }
