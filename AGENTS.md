# 项目协作入口

本目录包含对象参考图 inpainting 数据集流水线，以及 FastAPI + 原生 JavaScript 的 Studio 节点编辑器；不是模型训练入口。

## 恢复上下文（节省 token）
- 首先读取 [迭代交接](docs/ITERATION_NOTES.md) 的“当前交接”；只在需要时读取 [架构地图](docs/PROJECT_CONTEXT.md) 对应章节和相关源码。
- 不要默认遍历全仓库、全部节点或大型前端文件；按任务定位模块。
- 笔记是导航，不是源码替代品。修改前核对相关实现和 Git 状态；历史验证不代表当前通过。

## 关键入口
- CLI：[启动文件](build_object_inpainting_dataset.py) → [cli.main](object_inpainting/cli.py) → [pipeline.build_dataset](object_inpainting/pipeline.py)。
- Studio：[启动脚本](run_object_inpainting_studio.sh)、[服务端](object_inpainting_studio/server.py)、[前端](object_inpainting_studio/static/app.js)、[循环逻辑](object_inpainting_studio/static/loop.js)。
- 测试命令和依赖说明以 [测试指南](tests/README.md) 为准；模型替代输出测试不能证明真实模型质量。

## 迭代结束时维护
- 更新交接中的目标、改动、验证命令与结果、阻塞、下一步；没有实际任务时明确写“未指定”。
- 新增架构事实或踩坑再更新架构地图；已过时的信息直接纠正，不累积重复摘要。
- 只记录已核实事实；区分“已验证 / 未执行 / 待确认”。不要保存密钥、完整日志、机器专属绝对路径或大量源码。
- 保留最近几次关键迭代即可，长期历史交给 Git；不要每次重复记录整个项目。