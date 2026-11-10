"""Studio 端口显示名；不改变节点 API、连线字段或导出文件名。"""


TYPE_NAMES = {
    "RGB_IMAGE": ("rgb_image", "RGB 图", "rgb_image"),
    "RGBA_IMAGE": ("rgba_image", "透明 RGBA 图", "rgba_image"),
    "MASK": ("mask", "黑白 / 灰度 Mask", "mask"),
    "IMAGE": ("image", "任意模式图片", "image"),
    "IMAGE_LIST": ("image_list", "图片列表", "image_list"),
    "COLOR_REGIONS": ("color_regions", "颜色区域集合", "color_regions"),
    "LABEL_MAP": ("label_map", "整数标签图，非 Mask", "label_map"),
    "BBOX": ("bbox_xyxy", "左上右下像素框", "bbox"),
    "OBJECT_RECORD": ("object_record", "样本记录", "object_record"),
    "PREPARED_SAMPLE": ("sample", "组合训练样本", "sample"),
    "CONFIG": ("config", "共享配置", "config"),
}

FIELD_NAMES = {
    "object_mask": ("mask", "物体区域", "object_mask"),
    "edit_mask": ("mask", "编辑区域", "edit_mask"),
    "reference": ("reference_rgb_image", "RGB 参考图", "reference_rgb_image"),
    "target_path": ("rgb_image_path", "原图路径", "rgb_image_path"),
    "mask_path": ("mask_path", "Mask 路径", "mask_path"),
    "manifest_path": ("jsonl_path", "JSONL 路径", "jsonl_path"),
    "indices": ("indices", "0-based 面积排名列表", "indices"),
    "region_id": ("region_id", "1-based 区域编号", "region_id"),
}

# Only semantic exceptions belong here. Ordinary RGB / RGBA / Mask ports inherit
# their common format name, and a role explains which image a node produces.
PORT_NAMES = {
    "LoopDemoImageNode": {
        "inputs": {"seed": ("seed", "默认 0，七色周期；-1 随机", "INT"),
                   "image": ("rgb_image", "可选文件夹原图，左上角添加颜色标记", "rgb_image")},
        "outputs": {
            "image": ("rgb_image", "原尺寸标记图；无输入时 32×32 色图", "rgb_image"),
            "value": ("value", "本轮实际 seed", "INT"),
        },
    },
    "LoopDemoScoreNode": {
        "inputs": {"image": ("rgb_image", "待评分 RGB 图", "rgb_image")},
        "outputs": {"score": ("score", "左上角 32×32 标记蓝通道均值，蓝为 255", "FLOAT")},
    },
    "LoadTargetNode": {
        "inputs": {"path": ("rgb_image_path", "原图路径", "rgb_image_path")},
        "outputs": {"target": ("rgb_image", "原图", "rgb_image")},
    },
    "LoadInstanceMaskNode": {
        "inputs": {"path": ("mask_path", "Mask 路径", "mask_path")},
    },
    "AlignSquareNode": {
        "outputs": {"target": ("rgb_image", "对齐后的原图", "rgb_image")},
    },
    "GrayMaskNode": {
        "outputs": {"source": ("masked_rgb_image", "编辑区涂灰的 RGB 图", "masked_rgb_image")},
    },
    "AssembleSampleNode": {
        "inputs": {"source": ("masked_rgb_image", "编辑区涂灰的 RGB 图", "masked_rgb_image")},
    },
    "CropNode": {
        "outputs": {"cropped": ("rgb_image", "裁剪后的 RGB 图", "rgb_image")},
    },
    "SAM3TextMaskNode": {
        "inputs": {
            "threshold": ("threshold", "物体置信度阈值，0～1", "threshold"),
            "mask_threshold": ("mask_threshold", "像素 Mask 阈值，0～1", "mask_threshold"),
        },
        "outputs": {
            "object_mask": ("mask", "原始分割区域，未膨胀", "object_mask"),
            "bbox": ("bbox_xyxy", "原始分割 Mask 的像素框", "bbox"),
        },
    },
    "MaskDilationNode": {
        "inputs": {
            "mask": ("mask", "待膨胀黑白 / 灰度区域", "mask"),
            "dilation_steps": ("dilation_steps", "膨胀次数，0=关闭", "dilation_steps"),
            "dilation_kernel": ("dilation_kernel", "方形核边长，正奇数像素", "dilation_kernel"),
        },
        "outputs": {
            "mask": ("mask", "膨胀后的编辑区域", "edit_mask"),
            "bbox": ("bbox_xyxy", "膨胀后 Mask 的像素框", "bbox"),
        },
    },
    "FolderImagesNode": {
        "inputs": {"dir": ("input_dir", "输入文件夹", "input_dir")},
        "outputs": {
            "image": ("rgb_image", "当前原图", "rgb_image"),
            "path": ("rgb_image_path", "当前原图路径", "rgb_image_path"),
        },
    },
    "RGBAToWhiteBackgroundNode": {
        "outputs": {"rgb": ("reference_rgb_image", "白底 RGB 参考图", "reference_rgb_image")},
    },
    "FluxKleinReferenceNode": {
        "outputs": {"image": ("rgb_image", "模型生成图", "rgb_image")},
    },
    "FluxRandomRecolorNode": {
        "inputs": {"mask": ("masked_rgb_image", "涂灰原图，不是黑白 Mask", "masked_rgb_image")},
        "outputs": {
            "source": ("rgb_image", "换色结果，未涂灰", "rgb_image"),
            "masked_source": ("masked_rgb_image", "换色结果的涂灰图", "masked_rgb_image"),
            "mask": ("mask", "恢复的编辑区域", "edit_mask"),
            "sampled_color": ("color_rgb_image", "128×128 随机色块", "color_rgb_image"),
        },
    },
    "AutoColorRegionsNode": {
        "outputs": {
            "overlay": ("overlay_rgb_image", "分区伪彩色叠加，非真实颜色", "overlay_rgb_image"),
            "palette": ("palette_rgb_image", "区域平均颜色填充图", "palette_rgb_image"),
        },
    },
    "SelectColorRegionNode": {
        "outputs": {"mask": ("mask", "所选颜色区域", "mask")},
    },
    "ColorRegionSwatchesNode": {
        "outputs": {"images": ("image_list", "旧流程兼容色块列表", "image_list")},
    },
    "ColorSimilarityNode": {
        "inputs": {
            "image1": ("rgb_image_1", "第一张纯色 RGB 图片", "rgb_image"),
            "image2": ("rgb_image_2", "第二张纯色 RGB 图片", "rgb_image"),
        },
        "outputs": {
            "score": ("score", "CIEDE2000 色差得分（越小越接近）", "score"),
        },
    },
    "SaveImagesNode": {
        "inputs": {"dir": ("output_dir", "图片保存目录", "output_dir")},
        "outputs": {"paths": ("image_paths", "实际保存路径", "image_paths")},
    },
    "SaveSampleNode": {
        "inputs": {"output": ("output_dir", "数据集保存目录", "output_dir")},
    },
    "WriteJsonlNode": {
        "inputs": {"path": ("jsonl_path", "JSONL 路径", "jsonl_path")},
    },
    "SavePreviewNode": {
        "inputs": {"path": ("preview_path", "对照预览图路径", "preview_path")},
        "outputs": {"path": ("preview_path", "对照预览图路径", "preview_path")},
    },
}


def port_labels(node_type: str, inputs: dict[str, str], outputs: dict[str, str]) -> dict:
    """Return complete direction/key metadata without mutating runtime schemas.

    Names describe formats; roles distinguish images with different purposes.
    Kinds are UI recommendations, never new graph validation constraints.
    """
    result = {}
    for direction, ports in (("inputs", inputs), ("outputs", outputs)):
        labels = {}
        for key, port_type in ports.items():
            name, role, kind = TYPE_NAMES.get(port_type, (key, "", port_type))
            name, role, kind = FIELD_NAMES.get(key, (name, role, kind))
            if key == "target" and port_type == "RGB_IMAGE":
                role = "目标原图"
            if key.startswith("reference_") and port_type == "RGB_IMAGE":
                index = key.removeprefix("reference_")
                name, role, kind = f"reference_rgb_image_{index}", f"RGB 参考图 {index}", "reference_rgb_image"
            if node_type == "SaveImagesNode" and key.startswith("img") and key[3:].isdigit():
                name, role, kind = f"image_{key[3:]}", f"保存槽 {key[3:]}，任意模式", "image"
            if node_type == "SaveImagesNode" and key == "filename":
                name, role, kind = "filename", "整组文件名，留空使用随机编号", "filename"
            if node_type == "ColorRegionSwatchesNode" and key.startswith("img") and key[3:].isdigit():
                name, role, kind = key, f"独立 32×32 RGB 色块 {key[3:]}", "rgb_image"
            name, role, kind = PORT_NAMES.get(node_type, {}).get(direction, {}).get(key, (name, role, kind))
            labels[key] = {"name": name, "role": role, "kind": kind}
        result[direction] = labels
    return result
