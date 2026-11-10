# 项目架构地图

更新日期：2026-11-10。用途：按需定位代码，不要求每轮全部读取。实现变化时同步修正。

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
- 节点约定：先在 [node_contracts.py](../object_inpainting/node_contracts.py) 定义 TypedDict 入口 / 出口类；节点以 `INPUT_SCHEMA` / `OUTPUT_SCHEMA` 指定，配合显式 `execute()` 参数和返回注解。[base.py](../object_inpainting/nodes/base.py) 自动生成 `INPUT_TYPES` / `OUTPUT_TYPES`，保留旧式自定义节点的手写字符串表支持。所有 28 个内置节点已接入；新增后仍需在 [节点注册表](../object_inpainting/nodes/__init__.py) 注册。
- 公共类型见 [port_types.py](../object_inpainting/port_types.py) 的 `PortType` / `ImageRole` 及 PIL Annotated 别名，不包装图像。涂灰图 `MaskedRGBImage` 与 Mask `MaskImage` 分开标注；SAM2 框使用 `FloatBoundingBox`。完整 [入口出口类型指南](NODE_TYPES.md) 说明合同、限制及全节点对照。
- 动态保存项是静态合同的显式扩展：[基类](../object_inpainting/nodes/base.py) 提供 `prepare_spec`、`input_types`、`optional_input`、`execute_with_spec` 钩子；其他节点默认仍使用原静态端口及 `execute(**inputs)`。`NodeGraph.input_types[node_id]` 为实例端口真相，用于集合 / 连线 / 运行时校验和 Studio 上传解码。只有保存节点依据 `save_items` 生成任意 `imgN: IMAGE`，不放宽其他节点的未知输入校验。
- 动态输出通过基类 `output_types(spec)` 与 `NodeGraph.output_types[node_id]` 定义；所有实例归一化后再检查连线，返回字段集合及条件标量读取同样使用实例表。[色块节点](../object_inpainting/nodes/color_region_swatches.py) 依据字面量 `indices` 生成 `img1`～`imgN: RGB_IMAGE` 和 `colors`，TypedDict 是默认单口模板。输出数在执行前确定，拒绝 context / 连线索引，不按实际区域数悄悄减口。
- 新节点通常还需对应测试；面向 Studio 时检查 [node_help.py](../object_inpainting_studio/node_help.py)、服务端输出序列化及前端输入/预览支持。
- 节点预览有副作用限制；不要为了预览而绕过保存节点的写盘保护。
- 运行异常统一封装为 `NodeExecutionError(ValueError)`，携带原始失败节点 ID / 类型和错误类型 / 原因，保留异常链；上游错误不重复包装。`GraphExecutionCancelled` 是取消控制信号，不作为节点错误。
- CLI 默认图与用户导入的根目录示例工作流不是同一来源；修改前确认实际使用的是哪份 JSON。

## 4. Studio 定位和易混点

| 需求 | 优先查看 |
| --- | --- |
| API、配置解析、节点预览/保存 | [server.py](../object_inpainting_studio/server.py) |
| 跨请求预览上游复用 | [preview_cache.py](../object_inpainting_studio/preview_cache.py)：页面持有不透明 `preview_session`；只复用祖先，显式目标始终执行，保存操作不缓存；CLI / batch 不接入 |
| 统一端口显示名 / 用途元数据 | [port_names.py](../object_inpainting_studio/port_names.py)，API catalog 的 `port_labels.inputs/outputs[原字段]` 提供 `name`、`role`、`kind`；不修改运行时字段 |
| 节点入口 / 出口类 | catalog 的 `input_contract` / `output_contract` 和卡片 / 右侧说明展示实际 schema 类名；见 [类型指南](NODE_TYPES.md) |
| 后台批处理、取消、结果统计 | [batch.py](../object_inpainting_studio/batch.py) |
| 画布、端口、节点 UI、状态 | [app.js](../object_inpainting_studio/static/app.js)、[index.html](../object_inpainting_studio/static/index.html)、[style.css](../object_inpainting_studio/static/style.css) |
| 节点删除与连线清理 | [node_editing.js](../object_inpainting_studio/static/node_editing.js) 的纯工作流编辑函数；[app.js](../object_inpainting_studio/static/app.js) 的按钮、Delete 快捷键、运行锁和状态清理 |
| 文件夹逐张循环、保存终点 | [loop.js](../object_inpainting_studio/static/loop.js) |
| 多条件循环、点击向导与逐轮缓存 | [conditional_loop.py](../object_inpainting_studio/conditional_loop.py)，[multiple_loops.js](../object_inpainting_studio/static/multiple_loops.js)；[loop.js](../object_inpainting_studio/static/loop.js) 提供文件夹工具栏、共享点选与请求；后台复用 [batch.py](../object_inpainting_studio/batch.py) |
| 动态保存项、命名与文件冲突 | [save_images.py](../object_inpainting/nodes/save_images.py)、[app.js](../object_inpainting_studio/static/app.js) 的 `saveItems` / `nodeInfo` 和增删编辑函数 |
| 操作语义与保存目录 | [中文使用指南](../object_inpainting_studio/USAGE.zh-CN.md) |
| 自动颜色区域 / 随机换色 | [颜色区域指南](../object_inpainting_studio/COLOR_REGIONS.zh-CN.md)、[换色指南](../object_inpainting_studio/RANDOM_RECOLOR.zh-CN.md) |
| 面积排名列表 → 独立 32×32 代表色图 | [color_region_swatches.py](../object_inpainting/nodes/color_region_swatches.py)，`indices=[0,2]` → `img1/index=0`、`img2/index=2`，分别接评分 / 保存图片项；旧 `images` 连线自动兼容 |

- 批处理状态在进程内，不应假设服务重启能续跑。
- 色块列表缩短时前端清理已删除输出的下游输入和工作流输出，保留其余连线；重排按位置改变内容。卡片 / inspector / edges / 预览统一采用 `nodeInfo()` 实例输出；拓扑编辑受运行 / 检查锁保护，保留历史预览与会话。旧 `images` 引用由 [兼容迁移](../object_inpainting/workflow_migrations.py) 标记 `legacy_swatch_images`，只对旧节点额外保留列表。预览裁剪前先迁移完整图，保持消费者不在祖先集合时的旧列表输出。
- 条件循环是 Studio 控制层，不改普通 DAG 为任意有环图。新 `conditional_loop={loops:[...]}` 支持 1～32 个 ID 唯一的循环，每个含 `id/name`、`start_nodes/end_nodes/seed_nodes`、`conditions[{node, output, operator, value}]`、`mode=all/any`、`max_iterations`、`timeout_seconds`、`on_exhausted=fail/skip`。旧 flat 格式继续用 `ConditionalPlan` 原语义，两种根字段禁止混写；`build_conditional_plan()` 分派。普通 `NodeGraph.start/run` 继续拒绝条件元数据，不能绕过验收。暂不支持重叠 / 嵌套 / 反馈。
- `prepare_conditional` 裁剪所选 SaveImages、所有条件及显式 end 的祖先，共享归一化后的图给计划与上下文。`MultiConditionalPlan` 取 `descendants(start) ∩ ancestors(end+conditions)` 为各体，验证 start 可达 end、条件 / 种子在体内、非重叠、与所选保存有数据关系，并按传递依赖拓扑排序（含普通 bridge）。中间输出向普通外部消费须声明 end；条件节点可作为普通出口；唯一所选保存允许读取任意已验收体内结果。完整定义仍用于逃逸检查；额外写盘检查仅针对本次执行切片，未选保存分支不执行。布尔只支持 == / !=，数字有限；显式递增种子须非负 INT，逐文件重置。
- `/api/conditional/plan` 只检查结构，不推理或写盘；`/api/conditional` 启动当前图的单图后台任务。启用条件时 `/api/batch/plan` 合并评分分支，`/api/batch/preview` 变为异步单文件任务并使用批量过滤后的同一 index。单图 / 文件夹 / 条件试跑共用 `BATCH_JOBS` 互斥和原有查询 / 取消入口。计划与上下文必须使用同一图对象，避免图片路径解码重写后运行旧图。
- 每轮新 `ConditionalExecution`，固定输出跨尝试深拷贝且交给下游前隔离；多循环通过后强制执行其全部 end、保留已验收体缓存，后续重试不重跑前序。普通 bridge 只有全部依赖循环已验收才可冻结，不可保留失败轮污染。全部循环通过才从已验收缓存执行唯一所选保存一次；耗尽 / 超时不保存，异常不作为评分失败重试。报告兼容 accepted / iterations / history / stop_reason / skipped，增加每循环结果及 history 的 loop_id / loop_name；进度 current_loop_* 逐文件重置，不累计全尺寸历史图片。
- 条件任务超时 / 取消在节点前后及保存依赖就绪后检查，无法打断活跃 GPU 调用；进入保存后的取消不抹去已完成文件记录。普通单次保存在条件启用时拒绝，非保存预览仍允许单次调试。前端复用任务状态待确认锁，配置 / 节点变更使计划失效，轮询不重建聚焦的配置输入。
- [multiple_loops.js](../object_inpainting_studio/static/multiple_loops.js) 管理唯一的点击式多循环界面，旧开关 / 兼容入口 / flat 面板与旧 picker / renderer 已删除。按 `picking.loopId` 精确绑定卡片，向导 start→end→condition；单标量标题直选，多个则点输出，默认数字 <=10 / 布尔 ==true 只是示例。非负起点 INT seed 自动递增，-1 保持随机，高级仍支持多边界 / 条件 / 种子。最终保存独立共享，唯一时自动选择，失效引用不替换。`normalizeImportedConditional()` 在 restore / render / Add 自动一次性深拷贝旧 flat 为 loops，仅补缺省，end 由旧体流向所选保存的来源和条件节点推导；保存选择不明确时不猜，坏值和失效引用留作显式修正。原始非对象配置不丢弃，提供明确删除入口；不改导入原对象或磁盘文件，下载统一新格式。改配置使计划失效但保留预览，删最后循环恢复普通保存；运行锁 / 恢复清 picker，轮询不重建聚焦字段。
- 预览缓存按节点规范化 spec、所用 config / record、文件 stat、上游输出版本判定有效性；文件夹字面量输入还扫描图片清单和文件 stat，动态目录输入保守重算。缓存保存 / 取用均深拷贝真实输出以防下游修改污染；失败输出不缓存，已成功祖先可继续复用。进程锁串行保护，LRU 最多 8 会话 × 64 节点，15 分钟空闲过期在下一次请求清理；执行结束（包括失败）后刷新空闲时间，不能用加载模型前的时间判刚生成的结果过期。
- 普通编辑保留 `previewSession`，交由服务端逐节点校验，不得因为修改保存目录 / 名称而全局清空模型缓存。前端 `nodeResults` 保存当前画布各节点最近一次成功输出；`nodeResultStale` 只标记过期可能，不删除预览；`previewHidden` 管理手动隐藏 / 显示。卡片预览独立于参数折叠和 loop 展示。`resetPreviewState()` 仅用于新建 / 导入流程；删除节点只清该节点显示，batch / 试跑重置执行会话但保留历史预览。修订号防止编辑期间异步旧响应覆盖当前状态；纯节点位移不失效。
- 端口命名是展示层协议：`rgb_image`、`masked_rgb_image`、`mask`、`rgba_image`、`reference_rgb_image`、`image_list` 等，配中文用途；`kind` 用于前端推荐高亮，不是图运行时的新约束。特别是换色输入原 `mask` 实为 `masked_rgb_image`，输出原 `source` 实为未涂灰的 `rgb_image`。画布 / widgets / inspector / 预览 / 连线提示使用同一 metadata；DOM `data-port`、`data-field`、图 JSON 和节点 execute 原字段保持稳定，未知端口有回退。输入定位按 `data-port`，不能再用可重复的显示标题查找。
- 保存节点的目录与数据集导出目录是不同配置；“只保存当前张”与“逐张循环”也是不同操作。
- [SAM3 节点](../object_inpainting/nodes/sam3_text_mask.py) 只分割，保留两项默认 0.5 的 float 阈值；原膨胀参数 / 逻辑已删除。[MaskDilationNode](../object_inpainting/nodes/mask_dilation.py) 用现有方形 MaxFilter 独立 CPU 膨胀，int 参数 steps / kernel 默认 1 / 3，输出同尺寸 L Mask、最终非零 bbox 和面积说明，支持二值及灰度，不改原图；空 Mask 的框为全 0。原数据集 `ExpandMaskNode` 及完整示例保持兼容。详见 [分支设置](../object_inpainting_studio/USAGE.zh-CN.md#sam3-mask-太小时的设置)。
- [工作流迁移](../object_inpainting/workflow_migrations.py) 深拷贝旧图，将 SAM3 的旧膨胀字段移到唯一 ID 的独立节点，并重接全部原 object_mask / bbox 消费与图输出；旧默认 steps=0 被保留，迁移幂等、不改调用者。图构造时统一应用；`/api/workflow/validate` 返回归一化 `workflow`，前端导入采用它，避免隐藏新增执行节点。原 SAM3 quality 仍来自原始模型，独立节点另报膨胀面积。
- `SaveImagesNode` 默认四个保存项，可增删到八项或更多；每项元数据 `save_items: [{id, name}]` 与单调递增 `save_next_id` 随节点保存，不是可连线输入。显式列表是端口集合真相，删除项不被旧四口默认值复活；旧图深拷贝补齐元数据，保留目录与随机编号。`name` 决定子目录，连线和 `paths` 键仍是稳定 `imgN`；前端 `nodeInfo` 为卡片 / inspector / 连线提供实例类型与显示名，增删改名清理结果和 loop 计划，运行期间禁止修改。
- 保存节点 `filename: STRING` 默认空，同轮共享 UUID；可连接文件夹原 `filename`，去掉最后一个扩展名后统一 PNG。全部图片先验证编码，再检查名称重复、路径、符号链接子目录和文件冲突；`xb` 独占写入，不覆盖，OSError 回滚本轮文件（空目录可能残留）。固定名重跑 / 输入同主干会报冲突，不自动覆盖或跳过。
- `SaveImagesNode.images` 仍按列表位置保存到 `images_0`、`images_1` 等自动命名子目录，可与动态项同时保存，共享组名。图片列表复用服务端递归序列化和前端 object 预览；`INT_LIST` 字面量由前端按 JSON 解析。色块取现有区域平均 RGB，顺序保持输入列表；索引越界报错，不能将当前图排名当作跨图颜色身份。
- 批处理计划检查通过不代表每张图推理成功；不要把单张模型推理步数误认为图片数量。
- 错误可观测性：节点失败和 HTTP 错误写入终端日志；API 保留字符串 `detail` 并为节点错误添加 `node_error`；批处理失败项 / 停止任务报告也有同结构字段。单节点 / 当前张试跑在画布与右侧预览显示真正失败的节点，批量运行通过每张结果和下载报告查看。
- 节点删除只编辑当前画布，不删除磁盘文件；断开下游端口并移除直接关联图输出。输入恢复默认值或 null，清理布局 / 结果 / loop 计划；预览或批量任务未结束 / 状态未知时禁止删除。编辑控件内的 Delete 不触发节点删除。

## 5. 运行与验证导航

- 在本目录启动 Studio：`bash run_object_inpainting_studio.sh`。脚本默认用 `../.venv/bin/python`，可通过 `OBJECT_INPAINTING_STUDIO_PYTHON` 覆盖，不要求提前激活虚拟环境；设置 `PYTHONPATH`。HF 缓存默认放在 `../../hf_cache`，允许环境变量覆盖 `HF_HOME`、`HF_HUB_CACHE`（优先于旧名 `HUGGINGFACE_HUB_CACHE`）和 `HF_XET_CACHE`，两个 Hub 变量统一指向同一目录。默认 host/port 见模块入口，不猜测当前是否已有服务。
- CLI 用选定解释器运行入口并传 `--input-manifest` 和 `--output-dir`；其他参数见 [cli.py](../object_inpainting/cli.py)。
- [测试指南](../tests/README.md) 提供 unittest / pytest 命令。在本目录的文档命令为：`PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" ../../.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v`。解释器路径需核实，不能假定环境已装好。
- 单类测试收窄 `-p`；浏览器测试依赖 Playwright Chromium，可通过 `STUDIO_TEST_URL` 指向服务。模型节点测试使用替代输出，不自动证明真实 GPU 推理或图像质量。
- Studio、SAM2、SAM3 的依赖分别见 [基础依赖](../object_inpainting_studio/requirements.txt)、[SAM2 依赖](../object_inpainting_studio/requirements-sam2.txt)、[SAM3 依赖](../object_inpainting_studio/requirements-sam3.txt)。模型目录、设备、缓存等先检查对应节点实现及启动脚本；不要默认下载大模型。

上述命令来自源码/现有文档核对，本次笔记整理没有启动服务、运行测试或验证本机模型环境。
