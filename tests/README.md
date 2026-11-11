# 数据准备测试

本目录集中存放数据准备、节点、API 和浏览器测试。业务模块、启动脚本、工作流和示例图片仍保留在上一级目录。

## 从 Diffusers 仓库根目录运行

```bash
PYTHONPATH="$PWD/examples/dataset_preparation${PYTHONPATH:+:$PYTHONPATH}" \
  .venv/bin/python -m unittest discover \
  -s examples/dataset_preparation/tests -p 'test_*.py' -v
```

`PYTHONPATH` 用于导入上一级的 `object_inpainting` 和 `object_inpainting_studio`。测试模块间的共享 fixture 仍由 unittest 的测试目录搜索路径解析。

若已经进入数据准备目录：

```bash
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  ../../.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

若使用其他 Python 环境，替换解释器路径。不要再使用只搜索数据准备根目录的旧命令；应显式搜索本 tests 目录。

## 只运行一类测试

文件夹全部末端分支：将项目和 tests 放入 `PYTHONPATH`，使用已选解释器执行 `-m unittest test_folder_terminal_branches -q`。覆盖无条件评分、多保存、当前图试跑、条件达标缓存、耗尽不保存、全部目录校验/输出排除、无可视化仍执行、旧 flat 配置、计算失败及多保存冲突清理、条件上游禁止写盘。测试 mock demo 的 10 秒等待，不加载模型。真实浏览器用例 `test_folder_all_terminal_branches_run_and_show_score_without_condition` 检查评分显示与双保存输出。

该模块还覆盖显式结果 / 禁用分支、只评分不保存、普通 / 旧 flat 循环选择中间结果保留达标值且不重跑、无关未完成循环隔离、元数据校验往返、后端节点角色与每文件状态。前端 `test_conditional_loop.js` 覆盖计划过期 / 乱序响应、结果控件运行锁和删节点清理；真实浏览器 `test_batch_result_selection_disable_and_authoritative_plan_without_save` 通过实际按钮禁用错误分支、选择评分、确认不写盘。

修改 `-p` 参数即可，例如 `-p 'test_flux_random_recolor_node.py'`。浏览器测试使用 Playwright Chromium，并在未指定 `STUDIO_TEST_URL` 时自行启动临时服务。

模型节点测试使用替代模型输出验证接口，不自动下载模型或执行真实换色质量评估。

## 区域颜色色块测试

在数据准备目录、已选虚拟环境中运行：

```bash
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  ../.venv/bin/python -m unittest discover -s tests -p 'test_color_region_swatches_node.py' -v
```

[色块测试](test_color_region_swatches_node.py) 覆盖 0-based 面积排名、独立 RGB 32×32 的 `imgN`、顺序和重复索引、非法列表 / 越界、图片预览及不写盘、独立输出连线保存，以及文件夹逐张中缺少排名时报告错误并继续；另保留通用 SaveImages 图片列表的模式 / 不覆盖测试。使用实际 CPU 分色，不加载 SAM 或 FLUX。

新增 [动态输出测试](test_dynamic_swatch_outputs.py) 覆盖超过四口的任意数量、不同节点独立端口表、消费者先于生产者的 JSON 顺序、独立连接色差 / 图片保存、无效端口 / 类型 / 返回集合拒绝、归一化幂等与文件往返、不修改输入工作流、预览仅提供独立图片、旧标记不能恢复列表输出、旧列表连线拒绝、同会话修改索引后的缓存失效及批处理。运行 `-m unittest test_color_region_swatches_node test_dynamic_swatch_outputs -v`，需将项目 / tests 放入 `PYTHONPATH`。

[前端动态输出测试](test_swatch_outputs.js) 调用真实控制函数，覆盖卡片 / inspector 改索引即时增减口、重排与重复、无效输入不改图、运行锁、清理被删输出的下游输入与工作流输出、保留其余连线 / 会话 / 历史预览，以及旧列表输出不恢复和独立端口导入导出；另检查仓库工作流没有旧色卡设置或列表连线、独立保存项的元数据完整。全部前端命令为 `node --test tests/test_port_names.js tests/test_node_editing.js tests/test_conditional_loop.js tests/test_swatch_outputs.js`。

[浏览器测试](test_object_inpainting_studio_browser.py) 的 `test_swatch_dynamic_outputs_connect_score_save_and_shrink` 通过实际 Chromium 页面改 `[0,2,1]` 增加输出、连接独立色块到保存项、与纯红色计算 ΔE00=0、保存真实 PNG 并核对像素，再缩为 `[0]` 验证断线及历史预览保留。`test_color_region_swatches_individual_preview_and_separate_save` 验证仅有独立色块输出，无旧列表预览，保存到 img1/img2。旧 images 标记 / 迁移已删除，动态输出测试覆盖旧连线被拒绝。未运行真实模型推理。

## 跨请求预览缓存回归

[缓存测试](test_preview_cache.py) 通过真实 TestClient API 覆盖：先测上游再测下游的执行次数、真实 PIL 输出复用及防污染、显式重测和中间节点传递失效、参数 / 连线 / 配置 / 同路径文件 / 文件夹变化、会话隔离与过期 / 容量淘汰、失败重试，以及单次保存复用上游但每次写盘。长推理用模拟单调时钟验证执行超过 TTL 后的即时下游请求仍命中，不实际等待。仅 CPU，不加载模型。可按上述 unittest 方法运行 `test_preview_cache` 模块。

[前端交互测试](test_port_names.js) 额外覆盖生成后修改保存目录 / 名称 / 项数再保存仍传递同一会话、编辑和失败保留最近成功预览、历史状态及隐藏 / 显示、参数折叠不隐藏预览、异步旧请求不覆盖当前状态。使用 Node.js VM 调用真实控制器。

[浏览器测试](test_object_inpainting_studio_browser.py) 的 `test_flux_then_edit_save_reuses_model_and_keeps_preview_until_hidden` 使用真实页面和临时服务，以 CPU 图片替代 FLUX.execute，检查生成 → 修改保存目录和名称 → 保存时模型调用次数为 1，实际 PNG 像素一致，预览持续显示且手动隐藏跨重绘保留。不验证真实模型生成质量；外部 STUDIO_TEST_URL 模式跳过此替代模型用例。

## 条件循环回归

### 本轮实际验证（2026-11-10）

以下为本轮已实际执行的结果；核心测试之外补跑了扩展回归，不代表完整套件通过。

- **前端：五个文件共 140 项，全部通过。** 命令：`node --test tests/test_port_names.js tests/test_node_editing.js tests/test_conditional_loop.js tests/test_swatch_outputs.js tests/test_multiple_loops.js`。覆盖三项点选 / condition operator-value、旧高级控件已删除而 JSON 参数不丢失、flat 迁 loops、自动条件保存、节点预览请求与异步任务锁、逐轮快照 / revision、关闭可视化及焦点 / 历史快照。
- **Python：已选环境 Python 3.12.13，149 项通过，无跳过。** 项目及 tests 在 `PYTHONPATH`；PowerShell 执行 `& $selectedenv -m unittest test_loop_preview test_loop_downstream test_loop_demo test_multiple_conditional_loops test_conditional_loop test_node_contracts -q`，其中 `$selectedenv` 指向已选解释器，不固定机器路径。
- **扩展 Python：222 项通过，无跳过。** 同一环境执行 `& $selectedenv -m unittest test_loop_preview test_loop_downstream test_loop_demo test_multiple_conditional_loops test_conditional_loop test_node_contracts test_object_inpainting_studio test_folder_batch_node test_preview_cache test_node_errors test_save_images_node test_port_names -q`。包含真实保存像素 / 模式、最新文件夹快照而非全量图片历史、跨请求缓存及 Windows 临时目录别名路径比较；未为通过测试改变生产路径规则。
- **真实 Chromium：inprocess 模式 6 项通过**，位于 [浏览器测试](test_object_inpainting_studio_browser.py)：4 项更新循环用例 + 2 项新增 CPU 下游 / 关闭可视化用例。以 Event 阻塞确保页面实际看到红 / 绿 / 蓝各轮，而非只断言最终结果；核对普通下游值 255、生成 seed=0/1/2 无重跑、阈值 256 显示失败末轮及未通过。更新用例还验证三项向导、直接点 SaveImages 自动校验保存、旧参数迁移与具体实例绑定；不依赖已删除的高级控件或手动计划按钮。

### 覆盖入口与边界

- [无保存循环预览](test_loop_preview.py)：独立 / 依赖循环切片、无保存 / 空目录、无关坏配置隔离、必要上游验收、末轮耗尽 / 超时、JSON 图片序列化、副作用拦截、缓存隔离及共享任务锁 / 取消。
- [普通节点条件预览](test_loop_downstream.py)：普通节点消费任意循环内部输出及循环内部节点测试时，`/api/node/preview` 自动识别必要循环，先验收后用达标缓存算目标，无需追加 end，无额外生成；失败不跑普通下游。直接 flat 普通节点测试返回 422，flat 正式条件保存仍兼容；关闭可视化不序列化图片而不改变计算。
- [演示节点](test_loop_demo.py)：实际 CPU / PIL 红绿蓝周期、蓝通道评分、输入 / 合同 / catalog / API、达标缓存保存及失败不写盘，不下载模型。
- [多循环](test_multiple_conditional_loops.py)、[条件执行](test_conditional_loop.py) 及 [节点合同](test_node_contracts.py)：DAG 依赖排序、固定 / 验收缓存隔离、同轮评分、显式 end、保存一次、种子 / 逐文件重置、失败 / 异常 / 取消 / 超时、副作用及重叠拒绝、API 兼容与 TypedDict 合同。旧 all / any 等后端兼容测试不意味着当前 UI 仍暴露高级设置。
- 测试 agent 已通过工具将 scikit-image、Playwright 及 Chromium 装入已选环境，**无模型下载**。未运行完整 Python 或浏览器套件，部分其他模块缺 diffusers；未真实 SAM / FLUX 推理，CPU 替代输出和演示不证明模型质量。
- 未重启用户业务服务，临时测试服务器已关闭；用户启动脚本及模块入口自有改动保留不覆盖。节点内 GPU 调用不可强杀，不支持重叠 / 嵌套 / 反馈。
- 最终前端 140 项重跑通过；另重跑两项关键下游 / 可视化 Chromium 用例通过。三个前端模块语法、相关代码编辑器诊断及 `git diff --check` 通过。本节仅记录上述实际测试，不能把前次 124 项 JS / Python 未执行或更早 108 / 293 项记录当作当前结果。

## pytest

若环境已安装 pytest 及仓库测试所需依赖，也可以从仓库根目录运行：

```bash
.venv/bin/python -m pytest examples/dataset_preparation/tests -q
```

本目录的 conftest 为 pytest 补充业务模块导入路径；它不会为 unittest 自动配置路径，因此 unittest 仍需使用上面的 `PYTHONPATH`。

## 前端节点删除测试

在数据准备目录运行（需要 Node.js，无需 npm 依赖或浏览器）：

```bash
node --test tests/test_node_editing.js
```

覆盖节点 / 连线 / 工作流输出清理、默认值恢复、重复删除、画布状态、loop 计划清理、运行中保护和 Delete 输入保护。控制函数测试使用最小 DOM 替身，不等于真实浏览器端到端验证；按钮与键盘的浏览器用例位于 [浏览器测试](test_object_inpainting_studio_browser.py)，需 Playwright、Chromium 及系统依赖。

## 统一端口命名测试

- Python：在已选环境下用 `-m unittest discover -s tests -p 'test_port_names.py' -v`，9 项覆盖全节点元数据完整性、RGB / Mask / 涂灰 / RGBA / 参考图 / 图片列表命名、0/1-based 区分、API catalog 和旧工作流字段兼容。
- 前端：`node --test tests/test_port_names.js tests/test_node_editing.js`，共 21 项。端口相关 11 项调用真实 app 函数，使用最小 DOM 验证推荐连线、非法类型拒绝、合法同类型连接不被语义标签阻断、预览标题、右侧定位与原字段编辑、入口 / 出口类名、独立膨胀默认数字与编辑，以及旧工作流导入采用可见的迁移图；这不等于浏览器视觉或端到端验证。

## SAM3 Mask 控制测试

在已选环境与项目 `PYTHONPATH` 下运行 `-m unittest discover -s tests -p 'test_sam3_text_mask_node.py' -v`，共 16 项。覆盖原单实例分割、阈值传递与边缘像素、非法阈值前置拒绝、空 Mask、动态阈值错误、catalog / API 与膨胀接口已移除。使用替代模型和真实 Transformers SAM3 后处理，不下载权重，不证明真实模型或下游生成质量。

## 独立 Mask 膨胀与迁移测试

运行 `-m unittest discover -s tests -p 'test_mask_dilation_node.py' -v`，13 项覆盖 steps / kernel、边界及最终 bbox、不改输入、关闭膨胀、1-bit / 灰度 / 空 Mask、非法参数 / 模式、编辑与原始参考分支、catalog / 练习入口 / API，以及旧 SAM3 字段迁移、Mask / bbox 重接、默认关闭、ID 冲突、多节点与幂等。使用真实 CPU PIL 膨胀；迁移图执行的 SAM3 边界使用替代输出，不运行真实模型。

## 节点输入输出合同测试

使用已选虚拟环境和项目 `PYTHONPATH`，运行 `-m unittest discover -s tests -p 'test_node_contracts.py' -v`。9 项测试覆盖全部 30 个节点：TypedDict schema、每个 `execute` 参数注解与输入类一致、返回注解与输出类一致、自动生成端口表、语义图像类别、默认参数、旧类导入 / 自定义节点兼容和 Studio catalog。类型设计详见 [节点类型指南](../docs/NODE_TYPES.md)。

## 动态保存项与命名测试

- Python：使用选定解释器和项目 / tests 的 `PYTHONPATH`，运行 `-m unittest test_save_images_node test_folder_batch_node test_node_contracts -v`。保存节点 16 项验证旧四口、动态 12 路、RGB / RGBA / Mask 像素保持、稳定 ID、删项不复活、工作流深拷贝 / 幂等 / 磁盘往返、中文命名与 API 上传、组名扩展名处理、非法名称 / 重名 / 不兼容类型、冲突预检、符号链接拒绝和写盘失败回滚。文件夹测试额外覆盖 12 路复用每张原始文件名及重跑冲突；合同测试明确验证动态参数类型。
- 前端：`node --test tests/test_port_names.js tests/test_node_editing.js`，共 29 项；新增 8 项调用真实 app 函数，覆盖任意数量端口、命名不改连线、删项不重编号、不复用 ID、空列表、运行锁 / 控件解锁、结果 / loop 计划失效、工作流往返、删除上游动态连线及组名断开回退。
- 浏览器：[动态保存用例](test_object_inpainting_studio_browser.py) 包含添加 8 项、删项后增加新编号、命名、上传保存、下载 / 导入和继续增项。需要 Playwright Chromium 系统依赖；最小 DOM 测试不等于该端到端用例通过。
- 2026-11-07 实际验证：全部非浏览器 Python 187 项、前端 29 项通过；动态浏览器用例因缺少 `libnspr4.so` 在启动阶段失败（0 项执行）。不加载真实模型，不证明真实模型质量。
