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

修改 `-p` 参数即可，例如 `-p 'test_flux_random_recolor_node.py'`。浏览器测试使用 Playwright Chromium，并在未指定 `STUDIO_TEST_URL` 时自行启动临时服务。

模型节点测试使用替代模型输出验证接口，不自动下载模型或执行真实换色质量评估。

## 区域颜色色块测试

在数据准备目录、已选虚拟环境中运行：

```bash
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  ../.venv/bin/python -m unittest discover -s tests -p 'test_color_region_swatches_node.py' -v
```

原 [色块测试](test_color_region_swatches_node.py) 10 项覆盖 0-based 面积排名、独立 RGB 32×32 的 `imgN`、顺序和重复索引、非法列表 / 越界、图片预览及不写盘、旧图片列表连线兼容保存 / 模式保留 / 不覆盖，以及文件夹逐张中缺少排名时报告错误并继续。使用实际 CPU 分色，不加载 SAM 或 FLUX。

新增 [动态输出测试](test_dynamic_swatch_outputs.py) 15 项覆盖超过四口的任意数量、不同节点独立端口表、消费者先于生产者的 JSON 顺序、独立连接色差 / 图片保存、无效端口 / 类型 / 返回集合拒绝、迁移幂等与文件往返、不修改输入工作流、新预览无列表 / 旧预览保留列表、同会话修改索引后的缓存失效及批处理。运行 `-m unittest test_color_region_swatches_node test_dynamic_swatch_outputs -v`，需将项目 / tests 放入 `PYTHONPATH`。

[前端动态输出测试](test_swatch_outputs.js) 新增 13 项真实控制函数测试，覆盖卡片 / inspector 改索引即时增减口、重排与重复、无效输入不改图、运行锁、清理被删输出的下游输入与工作流输出、保留其余连线 / 会话 / 历史预览，以及旧列表与导入导出兼容。全部前端命令为 `node --test tests/test_port_names.js tests/test_node_editing.js tests/test_conditional_loop.js tests/test_swatch_outputs.js`。

[浏览器测试](test_object_inpainting_studio_browser.py) 的 `test_swatch_dynamic_outputs_connect_score_save_and_shrink` 通过实际 Chromium 页面改 `[0,2,1]` 增加输出、连接独立色块到保存项、与纯红色计算 ΔE00=0、保存真实 PNG 并核对像素，再缩为 `[0]` 验证断线及历史预览保留。原列表保存用例仍验证兼容端口和原目录。2026-11-10 实测非浏览器 Python 246 项、JS 68 项、重点 Chromium 4 项通过；未运行完整浏览器套件或真实模型推理。

## 跨请求预览缓存回归

[缓存测试](test_preview_cache.py) 通过真实 TestClient API 覆盖：先测上游再测下游的执行次数、真实 PIL 输出复用及防污染、显式重测和中间节点传递失效、参数 / 连线 / 配置 / 同路径文件 / 文件夹变化、会话隔离与过期 / 容量淘汰、失败重试，以及单次保存复用上游但每次写盘。长推理用模拟单调时钟验证执行超过 TTL 后的即时下游请求仍命中，不实际等待。仅 CPU，不加载模型。可按上述 unittest 方法运行 `test_preview_cache` 模块。

[前端交互测试](test_port_names.js) 额外覆盖生成后修改保存目录 / 名称 / 项数再保存仍传递同一会话、编辑和失败保留最近成功预览、历史状态及隐藏 / 显示、参数折叠不隐藏预览、异步旧请求不覆盖当前状态。使用 Node.js VM 调用真实控制器。

[浏览器测试](test_object_inpainting_studio_browser.py) 的 `test_flux_then_edit_save_reuses_model_and_keeps_preview_until_hidden` 使用真实页面和临时服务，以 CPU 图片替代 FLUX.execute，检查生成 → 修改保存目录和名称 → 保存时模型调用次数为 1，实际 PNG 像素一致，预览持续显示且手动隐藏跨重绘保留。不验证真实模型生成质量；外部 STUDIO_TEST_URL 模式跳过此替代模型用例。

## 条件循环回归

- [后端测试](test_conditional_loop.py)：26 项 CPU 用例，真实 PIL / 临时 PNG 保存，覆盖固定祖先只执行一次、循环体逐轮执行、输入原地修改隔离、达标图像与评分 / 种子同轮、AND 不跨轮拼接 / OR 全部分支计算、布尔 / 数字条件、作用域与副作用限制、耗尽失败 / 跳过、异常不重试、取消、模拟时钟超时（含保存依赖返回后的拦截）、单图 API / 普通保存防绕过、任务互斥、文件夹逐图独立缓存、字面图片解码和过滤索引一致的异步试跑。用 Event / join 而非等待轮询，不加载模型。
- [前端条件控制测试](test_conditional_loop.js)：21 项覆盖唯一多循环界面的计划检查、标量 / 种子校验、单图 / 文件夹异步提交、普通保存拦截、进度历史、焦点和请求状态不确定时禁止重复提交。移除已不存在的旧控件测试；各 VM harness 均加载正式的多循环控制器，不伪造旧 DOM 保留死代码。
- [浏览器测试](test_object_inpainting_studio_browser.py) 的 `test_conditional_retry_ui_checks_plan_saves_only_accepted_attempt` 从实际控件配置条件，以 CPU 图片替代 FLUX.execute；检查计划零推理 / 零写盘，seed=0 红图失败、seed=1 蓝图达标，仅保存后者。外部 `STUDIO_TEST_URL` 模式跳过，不证明真实模型质量。
- 同文件 `test_conditional_canvas_picker_selects_exact_instances_and_ports` 在 Chromium 中恢复旧 flat 配置，确认旧 DOM 已完全移除、配置自动转成新卡片，保留第二个同类分支及条件 / 种子 / 重试策略；用新按钮、标题和端口验证错误候选不改选、已有连线不变、角色高亮、精确定位、历史预览和往返。不运行模型。单循环达标保存用例也已切换为新点击向导。
- [多循环后端](test_multiple_conditional_loops.py)：47 项 CPU / PIL 测试，覆盖独立重试、串联及反序配置按 DAG 排序、bridge 缓存、显式 end 强制执行、所选保存读取验收中间图、后循环失败无写盘、变异隔离、取消 / 超时、旧 flat 兼容、API 与文件夹逐图重置、32 个循环、非法 ID / 混写 / 重叠 / 逃逸、多个 SaveImages 中仅执行所选切片并写其目录。
- [多循环前端](test_multiple_loops.js)：39 项真实控制器 / DOM 用例，覆盖自动向导、同类实例独立编辑删除、精确条件端口、种子、范围校验、焦点、运行锁、预览与进度，并验证旧 DOM / handler / renderer 不存在、有效 / 无效 flat 自动迁移、只补缺省、不替换失效目标、不改原输入、坏原始值显式删除、下载导入只生成新格式。全部前端运行 `node --test tests/test_port_names.js tests/test_node_editing.js tests/test_conditional_loop.js tests/test_swatch_outputs.js tests/test_multiple_loops.js`。
- 浏览器 `test_multiple_loop_wizard_runs_each_region_and_saves_once` 从折叠面板的 Add 开始，仅点标题配置两个循环，评分节点直接作为 end，逐轮 seed=0/1 和 10/11 分别失败再通过；以 CPU 图替代 FLUX，检查计划无推理写盘，实际保存两张蓝图、前序无重复生成、报告 / 输入焦点 / 真实下载导入 / 删除最后循环恢复普通保存。
- 删除旧界面后的最新实测：全部前端 JS 108 项、重点 Chromium 5 项（多循环向导 / 旧配置自动迁移及实例点选 / 新向导达标保存 / 动态色块 / 缓存保存）通过。后端本轮未修改、未重跑非浏览器 Python；前次 293 项通过。未运行完整浏览器套件，未验证真实模型。
- 在数据准备目录、选定虚拟环境中可运行 `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/tests" ../.venv/bin/python -m unittest test_conditional_loop -v`；前端运行 `node --test tests/test_port_names.js tests/test_node_editing.js tests/test_conditional_loop.js`。

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

使用已选虚拟环境和项目 `PYTHONPATH`，运行 `-m unittest discover -s tests -p 'test_node_contracts.py' -v`。9 项测试覆盖全部 27 个节点：TypedDict schema、每个 `execute` 参数注解与输入类一致、返回注解与输出类一致、自动生成端口表、语义图像类别、默认参数、旧类导入 / 自定义节点兼容和 Studio catalog。类型设计详见 [节点类型指南](../docs/NODE_TYPES.md)。

## 动态保存项与命名测试

- Python：使用选定解释器和项目 / tests 的 `PYTHONPATH`，运行 `-m unittest test_save_images_node test_folder_batch_node test_node_contracts -v`。保存节点 16 项验证旧四口、动态 12 路、RGB / RGBA / Mask 像素保持、稳定 ID、删项不复活、工作流深拷贝 / 幂等 / 磁盘往返、中文命名与 API 上传、组名扩展名处理、非法名称 / 重名 / 不兼容类型、冲突预检、符号链接拒绝和写盘失败回滚。文件夹测试额外覆盖 12 路复用每张原始文件名及重跑冲突；合同测试明确验证动态参数类型。
- 前端：`node --test tests/test_port_names.js tests/test_node_editing.js`，共 29 项；新增 8 项调用真实 app 函数，覆盖任意数量端口、命名不改连线、删项不重编号、不复用 ID、空列表、运行锁 / 控件解锁、结果 / loop 计划失效、工作流往返、删除上游动态连线及组名断开回退。
- 浏览器：[动态保存用例](test_object_inpainting_studio_browser.py) 包含添加 8 项、删项后增加新编号、命名、上传保存、下载 / 导入和继续增项。需要 Playwright Chromium 系统依赖；最小 DOM 测试不等于该端到端用例通过。
- 2026-11-07 实际验证：全部非浏览器 Python 187 项、前端 29 项通过；动态浏览器用例因缺少 `libnspr4.so` 在启动阶段失败（0 项执行）。不加载真实模型，不证明真实模型质量。
