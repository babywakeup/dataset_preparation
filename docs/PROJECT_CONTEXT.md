# 项目架构地图

核对日期：2026-10-27。用途：按需定位代码，不要求每轮全部读取。实现变化时同步修正。

## 1. 两条使用路径

| 路径 | 入口与职责 |
| --- | --- |
| CLI 数据集构建 | [启动文件](../build_object_inpainting_dataset.py) 仅调用 [cli.main](../object_inpainting/cli.py)；解析 manifest、输出目录、可选 workflow，构造 `PipelineConfig` 后调用 `build_dataset` |
| Studio 交互处理 | [启动脚本](../run_object_inpainting_studio.sh) → [模块入口](../object_inpainting_studio/__main__.py) → [FastAPI 服务](../object_inpainting_studio/server.py)；提供图编辑、预览、保存、文件夹批处理和数据集导出 |

两条路径共享节点和图执行器，但“保存当前图片 / 文件夹批处理”不等于 CLI 的完整训练数据集导出。不要把当前打开的薄入口当成主要实现。

## 2. CLI 数据流和保护条件

- [records.py](../object_inpainting/records.py)：JSONL 每行要求 `target`、`mask`、`object_name`；相对路径基于 manifest 所在目录解析。规范化路径与去空格后的对象名产生 24 位哈希 ID；重复记录报错。
- [默认工作流](../object_inpainting/workflows/object_reference.json)：加载图像/掩码 → 选对象 → 提取参考图、扩展掩码 → 方形对齐 → 编辑区填灰 → prompt → 组装与校验 → 保存样本、JSONL、预览。
- [config.py](../object_inpainting/config.py)：`PipelineConfig` 管理尺寸、扩张、参考图 padding、对象比例、验证集比例、种子等；修改参数前检查其校验。
- [quality.py](../object_inpainting/quality.py)：`PreparedSample` 与校验；默认训练样本约束包括二值 mask、非编辑区保持 target、编辑区为灰色 128。不要直接套用到所有 Studio 节点输出。
- [pipeline.py](../object_inpainting/pipeline.py)：`prepare_sample` 只请求 `sample`；`build_dataset` 要求 `sample`、`entry`、`split`，有预览时还要求 `preview` 输出。`workflow` 与 `graph` 参数互斥。
- 输出目录必须为空或不存在；先在同级临时目录生成再迁移，拒绝覆盖已有数据集。全部样本被拒绝时仍保留输出诊断后抛错。
- 输出含 train/validation/rejected JSONL、summary、workflow 和可选 previews；样本导出字段与路径规则见 [exporting.py](../object_inpainting/exporting.py)。

## 3. 图和节点扩展

- [graph.py](../object_inpainting/graph.py)：`NodeGraph` / 图执行；工作流由 `nodes` 和 `outputs` 构成，输入可为字面量、`{"context": "config"}` 或 `{"node": "id", "output": "port"}`；有类型与环路检查。
- 节点约定：参考 [base.py](../object_inpainting/nodes/base.py) 的 `INPUT_TYPES`、`OUTPUT_TYPES`、`execute`；添加实现后在 [节点注册表](../object_inpainting/nodes/__init__.py) 注册。
- 新节点通常还需对应测试；面向 Studio 时检查 [node_help.py](../object_inpainting_studio/node_help.py)、服务端输出序列化及前端输入/预览支持。
- 节点预览有副作用限制；不要为了预览而绕过保存节点的写盘保护。
- CLI 默认图与用户导入的根目录示例工作流不是同一来源；修改前确认实际使用的是哪份 JSON。

## 4. Studio 定位和易混点

| 需求 | 优先查看 |
| --- | --- |
| API、配置解析、节点预览/保存 | [server.py](../object_inpainting_studio/server.py) |
| 后台批处理、取消、结果统计 | [batch.py](../object_inpainting_studio/batch.py) |
| 画布、端口、节点 UI、状态 | [app.js](../object_inpainting_studio/static/app.js)、[index.html](../object_inpainting_studio/static/index.html)、[style.css](../object_inpainting_studio/static/style.css) |
| 文件夹逐张循环、保存终点 | [loop.js](../object_inpainting_studio/static/loop.js) |
| 操作语义与保存目录 | [中文使用指南](../object_inpainting_studio/USAGE.zh-CN.md) |
| 自动颜色区域 / 随机换色 | [颜色区域指南](../object_inpainting_studio/COLOR_REGIONS.zh-CN.md)、[换色指南](../object_inpainting_studio/RANDOM_RECOLOR.zh-CN.md) |

- 批处理状态在进程内，不应假设服务重启能续跑。
- 保存节点的目录与数据集导出目录是不同配置；“只保存当前张”与“逐张循环”也是不同操作。
- 批处理计划检查通过不代表每张图推理成功；不要把单张模型推理步数误认为图片数量。

## 5. 运行与验证导航

- 在本目录启动 Studio：`bash run_object_inpainting_studio.sh`。脚本默认用 `../../.venv/bin/python`，可通过 `OBJECT_INPAINTING_STUDIO_PYTHON` 覆盖；设置 `PYTHONPATH` 与 HF 缓存环境。默认 host/port 见模块入口，不猜测当前是否已有服务。
- CLI 用选定解释器运行入口并传 `--input-manifest` 和 `--output-dir`；其他参数见 [cli.py](../object_inpainting/cli.py)。
- [测试指南](../tests/README.md) 提供 unittest / pytest 命令。在本目录的文档命令为：`PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" ../../.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v`。解释器路径需核实，不能假定环境已装好。
- 单类测试收窄 `-p`；浏览器测试依赖 Playwright Chromium，可通过 `STUDIO_TEST_URL` 指向服务。模型节点测试使用替代输出，不自动证明真实 GPU 推理或图像质量。
- Studio、SAM2、SAM3 的依赖分别见 [基础依赖](../object_inpainting_studio/requirements.txt)、[SAM2 依赖](../object_inpainting_studio/requirements-sam2.txt)、[SAM3 依赖](../object_inpainting_studio/requirements-sam3.txt)。模型目录、设备、缓存等先检查对应节点实现及启动脚本；不要默认下载大模型。

上述命令来自源码/现有文档核对，本次笔记整理没有启动服务、运行测试或验证本机模型环境。