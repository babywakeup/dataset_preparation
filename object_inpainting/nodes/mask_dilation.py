from PIL import Image

from ..masking import expand_mask
from ..node_contracts import MaskDilationInputs, MaskDilationOutputs
from ..port_types import MaskImage
from .base import Node


class MaskDilationNode(Node):
    INPUT_SCHEMA = MaskDilationInputs
    OUTPUT_SCHEMA = MaskDilationOutputs
    INPUT_DEFAULTS = {"dilation_steps": 1, "dilation_kernel": 3}
    CATEGORY = "dataset"

    def execute(
        self, mask: MaskImage, dilation_steps: int = 1, dilation_kernel: int = 3
    ) -> MaskDilationOutputs:
        if isinstance(dilation_steps, bool) or not isinstance(dilation_steps, int) or dilation_steps < 0:
            raise ValueError("dilation_steps 必须是非负整数，0 表示不膨胀。")
        if (
            isinstance(dilation_kernel, bool)
            or not isinstance(dilation_kernel, int)
            or dilation_kernel < 1
            or dilation_kernel % 2 == 0
        ):
            raise ValueError("dilation_kernel 必须是正奇数，例如 3、5、11；1 表示不扩展。")
        if not isinstance(mask, Image.Image) or mask.mode not in {"1", "L"}:
            raise ValueError("mask 必须是 PIL 黑白或灰度 Mask（1 / L 模式），不能使用 RGB 图或整数标签图。")
        result = mask.convert("L").copy()
        pixels = result.width * result.height
        before = pixels - result.histogram()[0]
        if dilation_kernel > 1:
            for _ in range(dilation_steps):
                result = expand_mask(result, (dilation_kernel - 1) // 2)
        after = pixels - result.histogram()[0]
        return {
            "mask": result,
            "bbox": list(result.getbbox() or (0, 0, 0, 0)),
            "quality": (
                f"膨胀 {dilation_steps} 次，方形核 {dilation_kernel}×{dilation_kernel}；"
                f"非零区域 {before} → {after} 像素；bbox 已按输出 Mask 更新，空 Mask 的框为 [0, 0, 0, 0]。"
            ),
        }
