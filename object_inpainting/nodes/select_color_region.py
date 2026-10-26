from ..color_regions import ColorRegionSet, SelectedRegionOutputs, select_color_region
from .base import Node


class SelectColorRegionNode(Node):
    """将区域集合中的一个编号转换为普通 Mask 和 RGBA 图片。"""

    INPUT_TYPES = {"regions": "COLOR_REGIONS", "region_id": "INT"}
    INPUT_DEFAULTS = {"region_id": 1}
    OUTPUT_TYPES = {"mask": "MASK", "rgba": "RGBA_IMAGE"}
    CATEGORY = "dataset"

    def execute(self, regions: ColorRegionSet, region_id: int = 1) -> SelectedRegionOutputs:
        """选择一个当前图片中的区域；批量时编号不表示固定颜色。

        Args:
            regions (ColorRegionSet): 连接 AutoColorRegionsNode.regions 的运行时结果。
            region_id (int): 1～N 的区域编号，默认 1（面积最大）。

        Returns:
            SelectedRegionOutputs: 原尺寸的二值 L Mask 和保留原色的 RGBA 抠图。
        """
        return select_color_region(regions, region_id)
