# 节点入口、出口和公共类型

## 从哪里看

- [port_types.py](../object_inpainting/port_types.py)：`PortType`、`ImageRole` 公共类别，以及图像、路径、坐标框的 Python 类型。
- [node_contracts.py](../object_inpainting/node_contracts.py)：每个节点的 `XXXInputs` / `XXXOutputs` 类。类中的字段就是节点入口 / 出口，字段旁的注解就是对应类型。
- [base.py](../object_inpainting/nodes/base.py)：节点声明 `INPUT_SCHEMA` / `OUTPUT_SCHEMA`；基类自动生成图执行器使用的 `INPUT_TYPES` / `OUTPUT_TYPES`，不再维护两份手写端口表。
- 每个节点实现中的 `execute()`：所有 30 个已注册节点的参数和返回值均有显式注解。Studio 卡片及右侧说明同时显示入口类、出口类名称。

## 公共数据类型

| 类型 | 数据及含义 |
| --- | --- |
| `RGBImage` | PIL RGB 图；原图、普通生成图 |
| `MaskedRGBImage` | 编辑区域填 RGB(128,128,128) 的完整 RGB 图，**不是 Mask** |
| `ReferenceRGBImage` | RGB 参考图 |
| `MaskImage` | PIL 黑白 / 灰度 Mask；具体节点检查阈值与尺寸 |
| `RGBAImage` | PIL 透明图 |
| `OverlayRGBImage` / `PaletteRGBImage` | 伪彩色叠加 / 区域平均颜色填充 RGB 图 |
| `ColorRGBImage` | 候选纯色色块 |
| `LabelMap` | PIL 整数标签图，不是二值 Mask |
| `ImageList` | 图片列表兼容类别；色块节点新接口采用独立 RGB 输出，旧列表连线仍兼容 |
| `BoundingBox` / `FloatBoundingBox` | 整数框 / SAM2 也接受的浮点框；四个坐标，左上右下 |
| `FilePath` / `PathString` | 输入可为 `str` 或 `Path` / 输出字符串但图端口类别仍为 PATH |
| `OptionalReferenceImage` | PIL 图、路径或 None；FLUX 会读取路径并将参考图规范化；空字符串表示不提供 |
| `OptionalImage` / `OptionalImageList` | 保存节点的可选图片 / 图片列表；空字符串图片输入会跳过 |

`PortType` 是图端口类别枚举；`ImageRole` 是 RGB 用途枚举。图像类型用 `Annotated` 携带类别与用途，**底层仍为 PIL Image**，不会复制图像、包装模型输入或改变保存格式。

`XXXInputs` / `XXXOutputs` 使用 `TypedDict`：运行时仍是原有字典，适合 `execute(**inputs)` 和按端口取结果。`TypedDict` 不是自动验证器，`Annotated` 也不会让静态检查器将不同 RGB 用途视为互斥的图像类；它们不代替尺寸、模式、像素内容与模型调用检查。

## 代码中如何表达入口出口

以 [灰色遮挡节点](../object_inpainting/nodes/gray_mask.py) 为例：

```python
class GrayMaskNode(Node):
    INPUT_SCHEMA = GrayMaskInputs
    OUTPUT_SCHEMA = GrayMaskOutputs

    def execute(self, target: RGBImage, edit_mask: MaskImage) -> GrayMaskOutputs:
        return {"source": make_source(target, edit_mask)}
```

对应输入类列出 `target: RGBImage` 和 `edit_mask: MaskImage`，输出类列出 `source: MaskedRGBImage`。随机换色输入类中的 `mask` 同样标成 `MaskedRGBImage`，而输出类中的 `mask` 是 `MaskImage`；不要仅凭历史字段名判断数据。

合同描述图已经补齐默认值后的参数集合；函数自身仍保留原来的默认参数 / 必填参数。例如文件夹 `dir` 在图上有默认空值，独立调用 `execute()` 时仍是必填。

### 保存节点的动态端口例外

`SaveImagesInputs` 描述兼容入口 `img1`～`img4`、`images`、`dir` 和可选整组 `filename: str`。保存节点独立 `execute()` 还接受已注解的 `**extra_images: OptionalImage`，键必须是 `imgN`；其他节点没有这个扩展。

图中每个保存节点的 `save_items: list[SaveImageItem]` 是非输入元数据，每项只有稳定 `id: str` 和子目录 `name: str`；`save_next_id` 保留下一编号。旧图自动补齐四项；显式列表则可以删除全部固定项或声明更多项。按实例生成的 `NodeGraph.input_types[node_id]` 包含各个 `imgN: IMAGE` 及三个公共输入，连线与运行时校验仍严格，不能传入未声明项。

[节点基类](../object_inpainting/nodes/base.py) 的 `prepare_spec()` 归一化元数据，`input_types()` 提供实例端口，`optional_input()` 判定空值，`execute_with_spec()` 在保存时读取名称。默认实现让其他节点继续按静态表执行。名称不是额外的字符串输入口；修改名称不会修改 `id` 或 `paths` 输出键，也不包装 PIL 图像。图片列表仍自动使用 `images_N` 目录。

### 色块节点的动态输出例外

`ColorSwatchOutputs` 是默认 `[0]` 的出口模板：`img1: RGBImage` 和 `colors: SwatchColors`。真实实例依据直接填写的 `indices` 列表扩展 `img1`～`imgN`；例如 `[2,0,2]` 对应三个可独立连线的 RGB 输出，重复索引生成不同 PIL 实例，元数据仍列出原始索引及区域信息。静态 TypedDict 模板不能表达任意数量的字段，实际集合以 `NodeGraph.output_types[node_id]` 为准。

[节点基类](../object_inpainting/nodes/base.py) 新增 `output_types(spec)`，默认仍返回静态 `OUTPUT_TYPES`。图执行器先完成所有节点的元数据 / 默认值 / 实例端口归一化，再检查连线，因此 JSON 中消费者排在色块节点前也能引用 `imgN`。图输出类型、下游连线、运行返回键集合及条件标量检查均采用实例表，不放宽未知端口。

旧工作流的 `images` 引用通过深拷贝、幂等迁移启用 `legacy_swatch_images`，此时 `execute_with_spec()` 额外组装兼容列表。独立 `execute()` 只返回新独立输出；旧导入类路径保持不变。Studio 必须在裁剪预览祖先之前识别完整图的旧引用，避免仅预览色块时丢失兼容端口。

## 节点类对照

| 节点 | 入口类 | 出口类 |
| --- | --- | --- |
| `RecordInputNode` | `RecordInputInputs` | `RecordInputOutputs` |
| `LoadTargetNode` | `LoadTargetInputs` | `LoadTargetOutputs` |
| `LoadInstanceMaskNode` | `LoadInstanceMaskInputs` | `LoadInstanceMaskOutputs` |
| `SelectObjectNode` | `SelectObjectInputs` | `SelectObjectOutputs` |
| `ExtractReferenceNode` | `ExtractReferenceInputs` | `ExtractReferenceOutputs` |
| `ExpandMaskNode` | `ExpandMaskInputs` | `ExpandMaskOutputs` |
| `AlignSquareNode` | `AlignSquareInputs` | `AlignSquareOutputs` |
| `GrayMaskNode` | `GrayMaskInputs` | `GrayMaskOutputs` |
| `BuildPromptNode` | `BuildPromptInputs` | `BuildPromptOutputs` |
| `AssembleSampleNode` | `AssembleSampleInputs` | `SampleOutputs` |
| `ValidateSampleNode` | `ValidateSampleInputs` | `SampleOutputs` |
| `SaveSampleNode` | `SaveSampleInputs` | `SaveSampleOutputs` |
| `WriteJsonlNode` | `WriteJsonlInputs` | `WriteJsonlOutputs` |
| `SavePreviewNode` | `SavePreviewInputs` | `SavePreviewOutputs` |
| `CropNode` | `CropInputs` | `CropOutputs` |
| `SAM2MaskNode` | `SAM2MaskInputs` | `SAM2MaskOutputs` |
| `SAM3TextMaskNode` | `SAM3TextMaskInputs` | `SAM3TextMaskOutputs` |
| `MaskDilationNode` | `MaskDilationInputs` | `MaskDilationOutputs` |
| `MaskToRGBANode` | `MaskToRGBAInputs` | `MaskToRGBAOutputs` |
| `RGBAToWhiteBackgroundNode` | `RGBAToWhiteBackgroundInputs` | `RGBAToWhiteBackgroundOutputs` |
| `FluxKleinReferenceNode` | `FluxKleinReferenceInputs` | `FluxKleinReferenceOutputs` |
| `FluxRandomRecolorNode` | `FluxRandomRecolorInputs` | `RandomRecolorOutputs` |
| `SaveImagesNode` | `SaveImagesInputs` | `SaveImagesOutputs` |
| `FolderImagesNode` | `FolderImagesInputs` | `FolderImagesOutputs` |
| `AutoColorRegionsNode` | `AutoColorRegionsInputs` | `ColorRegionOutputs` |
| `SelectColorRegionNode` | `SelectColorRegionInputs` | `SelectedRegionOutputs` |
| `ColorRegionSwatchesNode` | `ColorRegionSwatchesInputs` | `ColorSwatchOutputs` |
| `ColorSimilarityNode` | `ColorSimilarityInputs` | `ColorSimilarityOutputs` |
| `LoopDemoImageNode` | `LoopDemoImageInputs` | `LoopDemoImageOutputs` |
| `LoopDemoScoreNode` | `LoopDemoScoreInputs` | `LoopDemoScoreOutputs` |

分色已有的两个输出类仍定义在 [color_regions.py](../object_inpainting/color_regions.py)，通过合同模块统一导出；旧的色块 / 换色输出类导入路径也继续指向同一份类，避免重复定义。

[独立膨胀节点](../object_inpainting/nodes/mask_dilation.py) 输入 `mask: MaskImage`、`dilation_steps: int`、`dilation_kernel: int`，返回 `MaskDilationOutputs` 的 `mask: MaskImage`、`bbox: BoundingBox`、`quality: str`。SAM3 合同不再包含膨胀字段，保留两项 float 阈值；原始 Mask 与膨胀 Mask 可以连接不同分支。

## 扩展与验证

1. 在公共合同模块定义输入 / 输出类，在公共类型模块选择或新增有据可查的类别。
2. 节点声明两个 schema 类，为每个 `execute()` 参数和返回值写对应类型。静态节点返回字段须与输出类一致；动态节点通过显式实例钩子定义和校验字段，不使用运行时临时增口。
3. 不支持的字段类型会在注册时明确报错，不会自动当成任意图片。
4. 添加到节点注册表，并更新 Studio 帮助、必要的显示名和测试。字段改名仍需同步工作流 / 文档；仅增加类型注解不改变已有连线。
5. [合同测试](../tests/test_node_contracts.py) 逐个检查全部节点的参数名、参数类型、返回类和生成端口表，并明确验证保存节点的动态参数注解与可选规则；旧式自定义节点仍可直接声明字符串端口表。

本轮未引入图像包装类或新运行依赖，不宣称整个项目或第三方库达到 100% 类型完整；PIL 的外部 `encoderinfo` 等动态字段仍属于第三方类型边界。
