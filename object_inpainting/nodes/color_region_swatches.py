from collections.abc import Mapping
from typing import cast

from PIL import Image

from ..color_regions import ColorRegionSet
from ..node_contracts import ColorRegionSwatchesInputs, ColorSwatchOutputs, SwatchColors as SwatchColors
from ..port_types import RegionIndices
from .base import Node


class ColorRegionSwatchesNode(Node):
    """按面积排名索引提取区域平均 RGB，每个索引输出一张独立的 32×32 纯色图。"""

    INPUT_SCHEMA = ColorRegionSwatchesInputs
    INPUT_DEFAULTS = {"indices": [0]}
    OUTPUT_SCHEMA = ColorSwatchOutputs
    DYNAMIC_IMAGE_OUTPUTS = True
    CATEGORY = "dataset"

    @staticmethod
    def validate_indices(indices: RegionIndices) -> list[int]:
        if indices is None:
            return [0]
        if not isinstance(indices, list) or not indices or any(type(index) is not int for index in indices):
            raise ValueError("indices 必须是非空整数列表，例如 [0,1]；索引从 0 开始，不接受布尔值或连线。")
        if any(index < 0 for index in indices):
            raise ValueError("indices 索引越界：索引从 0 开始，不能为负数。")
        return list(indices)

    @classmethod
    def prepare_spec(cls, spec: dict) -> None:
        # Port count must be known before graph validation, not after inference.
        spec["inputs"]["indices"] = cls.validate_indices(spec["inputs"].get("indices"))
        if "legacy_swatch_images" in spec and type(spec["legacy_swatch_images"]) is not bool:
            raise ValueError("legacy_swatch_images 必须是布尔值。")

    @classmethod
    def output_types(cls, spec: dict) -> dict[str, str]:
        indices = cls.validate_indices(spec["inputs"].get("indices"))
        return {**{f"img{position}": "RGB_IMAGE" for position in range(1, len(indices) + 1)},
                "colors": "JSON_RECORD", **({"images": "IMAGE_LIST"} if spec.get("legacy_swatch_images") else {})}

    def execute_with_spec(self, inputs: dict, spec: dict) -> Mapping[str, object]:
        result: dict[str, object] = dict(self.execute(**inputs))
        if spec.get("legacy_swatch_images"):
            # Only old workflows referencing the former list port enable this.
            result["images"] = [result[f"img{position}"] for position in range(1, len(inputs["indices"]) + 1)]
        return result

    def execute(self, regions: ColorRegionSet, indices: RegionIndices = None) -> ColorSwatchOutputs:
        """索引从 0 开始，保留输入顺序和重复项；只生成色块，不写入文件。

        Args:
            regions (ColorRegionSet): AutoColorRegionsNode.regions，区域已按本图面积降序排列。
            indices (list[int] | None): 默认 [0]；[0, 1] 选择面积排名前两种颜色。

        Returns:
            ColorSwatchOutputs: img1～imgN 各为一张独立 RGB 32×32 图片，按 indices
            的位置命名；colors 记录索引、1-based 区域 ID、平均 RGB、面积比例与尺寸。
        """
        if not isinstance(regions, ColorRegionSet):
            raise ValueError("regions 必须连接自动分色节点的区域集合输出。")
        indices = self.validate_indices(indices)
        count = len(regions.regions)
        if not count:
            raise ValueError("regions 没有可提取的颜色区域。")
        for index in indices:
            if not 0 <= index < count:
                raise ValueError(f"indices 中的索引 {index} 越界：当前图有 {count} 种颜色，可用索引为 0～{count - 1}。")
        selected = [regions.regions[index] for index in indices]
        result: dict[str, object] = {
            **{f"img{position}": Image.new("RGB", (32, 32), region.color)
               for position, region in enumerate(selected, 1)},
            "colors": {
                "indices": list(indices),
                "region_ids": [region.id for region in selected],
                "rgb": [region.color for region in selected],
                "fractions": [region.fraction for region in selected],
                "size": (32, 32),
            },
        }
        return cast(ColorSwatchOutputs, result)
