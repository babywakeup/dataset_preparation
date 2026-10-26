from dataclasses import dataclass
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray
from PIL import Image


FloatArray = NDArray[np.float64]
RGBArray = NDArray[np.uint8]
IndexArray = NDArray[np.intp]


@dataclass(frozen=True)
class ColorRegion:
    """单个颜色类别的元数据，不要求空间连通。

    Attributes:
        id (int): 从 1 开始的编号，按面积降序排列，仅对当前图片有意义。
        pixels (int): 当前类别的有效像素数。
        fraction (float): 占输入 Mask 有效面积的比例，范围 (0, 1]。
        color (tuple[int, int, int]): 原图中当前类别的平均 sRGB 色，范围 0～255。
        bbox (tuple[int, int, int, int]): 原图坐标下的左、上、右、下，右下边界不包含。
    """

    id: int
    pixels: int
    fraction: float
    color: tuple[int, int, int]
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class ColorRegionSet:
    """完整分色结果，仅保存一份原图和标签图，不为每个类别复制全尺寸图片。

    Attributes:
        image (PIL.Image.Image): 原尺寸 RGB 原图的副本。
        labels (PIL.Image.Image): 原尺寸 L 模式整数标签图，0 为背景，1～N 为类别编号。
        regions (tuple[ColorRegion, ...]): 面积降序的区域元数据，最多 32 类。
        detail (int): 分色精细度，0 粗、1 中、2 细。
    """

    image: Image.Image
    labels: Image.Image
    regions: tuple[ColorRegion, ...]
    detail: int


class ColorRegionOutputs(TypedDict):
    """分色节点输出：区域集合、RGB 叠加图、RGB 主色图及整数标签图。"""

    regions: ColorRegionSet
    overlay: Image.Image
    palette: Image.Image
    labels: Image.Image


class SelectedRegionOutputs(TypedDict):
    """区域选择输出：全尺寸 L 模式 Mask 和保留原始颜色的 RGBA 抠图。"""

    mask: Image.Image
    rgba: Image.Image


def weighted_lab(rgb: RGBArray) -> FloatArray:
    """将颜色转换为降低亮度影响的标准 CIE Lab 特征。

    Args:
        rgb (NDArray[uint8]): 形状 (N, 3)，有效前景像素的 sRGB，范围 0～255。

    Returns:
        NDArray[float64]: 形状 (N, 3) 的 (0.5 * L, a, b)，不做逐图通道标准化。
    """
    try:
        from skimage.color import rgb2lab
    except ImportError as error:
        raise ValueError("自动分色需要 scikit-image，请安装 Studio requirements.txt 中的依赖。") from error
    features = np.asarray(rgb2lab(rgb.astype(np.float64) / 255.0), dtype=np.float64)
    features[:, 0] *= 0.5
    return features


def color_bins(rgb: RGBArray, features: FloatArray) -> tuple[FloatArray, FloatArray]:
    """按 RGB 量化网格汇总特征，覆盖整张 Mask，避免随机抽样丢掉稀有颜色。

    Args:
        rgb (NDArray[uint8]): 形状 (N, 3) 的有效像素颜色。
        features (NDArray[float64]): 形状 (N, 3)，与 rgb 一一对应的加权 Lab。

    Returns:
        tuple[NDArray[float64], NDArray[float64]]: (B, 3) 的网格均值和 (B,) 像素计数；
        每个 RGB 通道使用 5 位量化，B 不超过 32768。
    """
    quantized = rgb.astype(np.int32) // 8
    keys = quantized[:, 0] * 1024 + quantized[:, 1] * 32 + quantized[:, 2]
    _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    weights = counts.astype(np.float64)
    centers = np.column_stack([np.bincount(inverse, weights=features[:, axis]) / weights for axis in range(3)])
    return centers, weights


def nearest_centers(features: FloatArray, centers: FloatArray) -> IndexArray:
    """分块计算最近中心，避免创建全图像素数乘中心数的大数组。

    Args:
        features (NDArray[float64]): 形状 (N, 3) 的加权 Lab 像素。
        centers (NDArray[float64]): 非空的 (K, 3) 聚类中心。

    Returns:
        NDArray[intp]: 形状 (N,) 的最近中心索引，范围 0～K-1。
    """
    assignments = np.empty(len(features), dtype=np.intp)
    for start in range(0, len(features), 32768):
        block = features[start : start + 32768]
        distances = np.sum((block[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        assignments[start : start + len(block)] = np.argmin(distances, axis=1)
    return assignments


def cluster_colors(rgb: RGBArray, features: FloatArray, detail: int) -> IndexArray:
    """先细分再做完全链接合并，自动确定最终颜色数量。

    Args:
        rgb (NDArray[uint8]): 形状 (N, 3) 的前景 sRGB 像素。
        features (NDArray[float64]): 对应的 (N, 3) 加权 Lab 特征。
        detail (int): 0/1/2 对应粗/中/细，合并距离分别为 24/15/8。

    Returns:
        NDArray[intp]: 形状 (N,) 的颜色类别索引，编号暂未按面积重排。
        内部最多使用 32 个细分中心；完全链接限制的是这些中心之间的距离，
        不是每个原始像素的最大色差，也不是语义颜色识别。
    """
    try:
        from scipy.cluster.hierarchy import fcluster, linkage
        from sklearn.cluster import KMeans
        from threadpoolctl import threadpool_limits
    except ImportError as error:
        raise ValueError("自动分色需要 scipy 和 scikit-learn，请安装 Studio requirements.txt 中的依赖。") from error
    centers, weights = color_bins(rgb, features)
    if len(centers) > 32:
        with threadpool_limits(limits=1):
            model = KMeans(n_clusters=32, random_state=0, n_init=3, max_iter=100)
            model.fit(centers, sample_weight=weights)
        centers = np.asarray(model.cluster_centers_, dtype=np.float64)
    if len(centers) == 1:
        return np.zeros(len(features), dtype=np.intp)
    groups = fcluster(linkage(centers, method="complete"), t=(24.0, 15.0, 8.0)[detail], criterion="distance")
    return np.asarray(groups[nearest_centers(features, centers)], dtype=np.intp)


def segment_colors(image: Image.Image, mask: Image.Image, detail: int = 1) -> ColorRegionSet:
    """只在 Mask 内自动进行 Lab 颜色分区，不加载模型或写入文件。

    Args:
        image (PIL.Image.Image): 原始 RGB 图片，尺寸为 (W, H)。
        mask (PIL.Image.Image): 同尺寸的 1/L 模式 Mask，灰度 >127 的像素参与分色。
        detail (int): 0 粗、1 中、2 细，默认 1，不需要提供主色数量。

    Returns:
        ColorRegionSet: 原图副本、背景为 0 的标签图及面积降序的颜色区域元数据。
        所有区域互斥且完整覆盖有效 Mask；不因区域面积小而删除，也不要求连通。

    Raises:
        ValueError: 输入类型、尺寸、图像模式、精细度不正确，或有效 Mask 为空。
    """
    if not isinstance(image, Image.Image) or not isinstance(mask, Image.Image):
        raise ValueError("image 和 mask 必须是 PIL 图片。")
    if image.mode != "RGB":
        raise ValueError("image 必须是 RGB 图片。")
    if mask.mode not in {"1", "L"}:
        raise ValueError("mask 必须是黑白或 8 位灰度图片。")
    if image.size != mask.size:
        raise ValueError("原图与 Mask 尺寸必须一致。")
    if type(detail) is not int or detail not in (0, 1, 2):
        raise ValueError("detail 必须为 0（粗）、1（中）或 2（细）。")
    active = np.asarray(mask.convert("L")) > 127
    if not np.any(active):
        raise ValueError("Mask 为空：没有灰度大于 127 的有效像素。")
    rgb = np.asarray(image)[active]
    raw_labels = cluster_colors(rgb, weighted_lab(rgb), detail)
    groups, counts = np.unique(raw_labels, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    foreground_labels = np.zeros(len(rgb), dtype=np.uint8)
    regions = []
    ys, xs = np.nonzero(active)
    for region_id, group_index in enumerate(order, 1):
        selected = raw_labels == groups[group_index]
        foreground_labels[selected] = region_id
        mean_rgb = np.rint(rgb[selected].mean(axis=0)).astype(np.uint8)
        bounds = (
            int(xs[selected].min()),
            int(ys[selected].min()),
            int(xs[selected].max()) + 1,
            int(ys[selected].max()) + 1,
        )
        regions.append(
            ColorRegion(
                id=region_id,
                pixels=int(counts[group_index]),
                fraction=float(counts[group_index] / len(rgb)),
                color=(int(mean_rgb[0]), int(mean_rgb[1]), int(mean_rgb[2])),
                bbox=bounds,
            )
        )
    label_image = np.zeros(active.shape, dtype=np.uint8)
    label_image[active] = foreground_labels
    return ColorRegionSet(image.copy(), Image.fromarray(label_image), tuple(regions), detail)


def render_color_regions(result: ColorRegionSet) -> ColorRegionOutputs:
    """生成用于预览或保存的全尺寸可视化图片，不改变原始区域标签。

    Args:
        result (ColorRegionSet): segment_colors 返回的区域集合。

    Returns:
        ColorRegionOutputs: regions 为集合本身；overlay 为 45% 伪彩色叠加的 RGB 图片；
        palette 为实际平均颜色填充的 RGB 主色图；labels 为 L 模式整数标签图。
        两张 RGB 预览图在输入 Mask 外均保留原图。
    """
    import colorsys

    labels = np.asarray(result.labels)
    original = np.asarray(result.image)
    overlay = original.copy()
    palette = original.copy()
    for region in result.regions:
        selected = labels == region.id
        rgb = colorsys.hsv_to_rgb(((region.id - 1) * 0.61803398875) % 1.0, 0.75, 1.0)
        display_color = np.array(rgb) * 255
        overlay[selected] = np.rint(original[selected] * 0.55 + display_color * 0.45).astype(np.uint8)
        palette[selected] = region.color
    return {
        "regions": result,
        "overlay": Image.fromarray(overlay),
        "palette": Image.fromarray(palette),
        "labels": result.labels,
    }


def select_color_region(regions: ColorRegionSet, region_id: int = 1) -> SelectedRegionOutputs:
    """提取一个颜色类别，供抠图、涂灰或保存节点使用。

    Args:
        regions (ColorRegionSet): 自动分色节点的 regions 输出，不接受普通 JSON。
        region_id (int): 当前图片的类别编号，从 1 开始，1 表示面积最大的类别。

    Returns:
        SelectedRegionOutputs: mask 为原尺寸 0/255 的 L 模式 Mask；rgba 为原尺寸 RGBA
        抠图，区域内保留原 RGB，区域外 Alpha=0。不存在自动跨图片颜色身份匹配。

    Raises:
        ValueError: 区域集合类型不正确或编号在当前图片中不存在。
    """
    if not isinstance(regions, ColorRegionSet):
        raise ValueError("regions 必须连接自动分色节点的区域集合输出。")
    if type(region_id) is not int or not any(region.id == region_id for region in regions.regions):
        raise ValueError(f"region_id 必须在 1～{len(regions.regions)} 之间；编号只对当前图片有效。")
    mask = regions.labels.point([255 if value == region_id else 0 for value in range(256)])
    rgba = regions.image.convert("RGBA")
    rgba.putalpha(mask)
    return {"mask": mask, "rgba": rgba}
