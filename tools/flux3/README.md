# FLUX.3 诊断工具

从项目根目录执行，数据与输出参数仍相对于项目根目录：

```bash
../.venv/bin/python tools/flux3/test_flux3_recolor_pipeline.py --help
../.venv/bin/python tools/flux3/compare_flux3_training_pipeline.py --help
../.venv/bin/python tools/flux3/analyze_flux3_numeric_precision.py --help
../.venv/bin/python tools/flux3/trace_flux3_tensor_path.py --help
```

这些工具用于基础模型测试、训练与 pipeline 数值对照、精度复核和逐层定位。
已训练模型的日常推理请使用根目录 `infer_flux3_recolor.py`；简洁示例见 `examples/recolor/demo_flux3_recolor_trained_pipeline.py`。
详见 [对照说明](../../docs/FLUX3_PIPELINE_COMPARE.zh-CN.md)。
