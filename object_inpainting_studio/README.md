# Object Inpainting Dataset Studio

This is a local, browser-based node editor for the object-inpainting data pipeline. It does **not** require Node.js or an external CDN.

## 中文使用手册

See [数据生成 Web 使用手册](USAGE.zh-CN.md) for step-by-step instructions, the garment workflow, server-side input/output directories, single-image testing versus folder iteration, saved outputs, and troubleshooting.

See [Mask 内自动颜色分区](COLOR_REGIONS.zh-CN.md) for automatic masked Lab clustering, region selection, and typed Python interfaces. No color count or model weights are required.

See [FLUX.2 衣服随机换色](RANDOM_RECOLOR.zh-CN.md) for generating a randomly recolored source from a real target and the RGB output of GrayMaskNode using local FLUX.2 weights.

For folder iteration, use **↻ 逐张** on the folder node and **运行全部 · 写入磁盘** at the top of the canvas. The save node's **仅保存当前张 · 不循环** button saves only one group; it does not start a loop. The Use section below describes the separate, complete JSONL dataset workflow.

## Run

From the Diffusers repository root:

```bash
bash examples/dataset_preparation/run_object_inpainting_studio.sh
```

Open http://127.0.0.1:7860 in a browser. Pass `--host` and `--port` to change the listener:

```bash
bash examples/dataset_preparation/run_object_inpainting_studio.sh --host 0.0.0.0 --port 7860
```

## Use

1. The default workflow is pre-wired.
2. Drag node headers to arrange the canvas.
3. Click an output port then a same-type input port to create a connection.
4. Select a node to edit literal or context bindings in the inspector.
5. Enter an absolute target image path, mask image path, and object name, then choose **Run preview**.
6. Review target, source, mask, and reference images before export.
7. Enter an input JSONL path and empty output directory, then choose **Export JSONL dataset**.

The input JSONL uses `target`, `mask`, and `object_name`; paths are relative to the JSONL directory. The exported JSONL is compatible with the Klein inpainting trainer.

## Workflow files

Use **Download workflow** to save the nodes, bindings, node positions, global configuration, and iteration settings. Use **Load workflow** to restore them. Images, model weights, and job reports are not embedded in the workflow. The backend validates node class names, port names, port types, and cycles before it executes a workflow.

## Tests

Dataset preparation tests are grouped under [tests](../tests/README.md). That guide includes the updated discovery command and import-path setup.
