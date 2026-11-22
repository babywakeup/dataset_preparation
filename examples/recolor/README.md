# 早期换色对照示例

从项目根目录运行：

```bash
../.venv/bin/python examples/recolor/demo_flux3_recolor_pipeline.py
../.venv/bin/python examples/recolor/demo_qwen_recolor_pipeline.py
```

脚本内保留原实验的模型、输入图片和参数；运行前按需修改。两者会实际加载大模型并执行推理。
单张结果统一写入 `outputs/recolor_demo_results/`；FLUX 工作流对照的运行记录仍写入 `outputs/flux_workflow_aligned_demo/`。

已训练权重的简洁示例为 `examples/recolor/demo_flux3_recolor_trained_pipeline.py`。正式推理入口 `infer_flux3_recolor.py` 和交互入口 `gradio_flux3_recolor.py` 保留在根目录。
