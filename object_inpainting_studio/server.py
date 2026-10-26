import base64
import hashlib
import io
import json
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from object_inpainting.color_regions import ColorRegionSet
from object_inpainting.config import PipelineConfig
from object_inpainting.graph import NodeGraph, is_link
from object_inpainting.nodes import NODE_CLASS_MAPPINGS
from object_inpainting.nodes.folder_images import folder_image, scan_folder
from object_inpainting.pipeline import DEFAULT_WORKFLOW, build_dataset
from object_inpainting.quality import PreparedSample, validate_sample
from object_inpainting.records import ObjectRecord, load_target
from PIL import Image, ImageOps

from .batch import BatchJobs
from .node_help import CONFIG_HELP, NODE_HELP


ROOT = Path(__file__).parent
STATIC = ROOT / "static"
UPLOADS = TemporaryDirectory(prefix="object-inpainting-studio-uploads-")
app = FastAPI(title="Object Inpainting Dataset Studio")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
BATCH_JOBS = BatchJobs()


def record_from_payload(payload: dict) -> ObjectRecord:
    if not isinstance(payload, dict):
        raise ValueError("record must be an object.")
    try:
        target_path = Path(payload["target_path"]).expanduser().resolve()
        mask_path = Path(payload["mask_path"]).expanduser().resolve()
        object_name = payload["object_name"].strip()
    except (KeyError, AttributeError, TypeError) as error:
        raise ValueError("Record requires target_path, mask_path, and a non-empty object_name.") from error
    if not object_name:
        raise ValueError("object_name must be non-empty.")
    digest = hashlib.sha256(f"{target_path}\0{mask_path}\0{object_name}".encode()).hexdigest()[:24]
    return ObjectRecord(digest, target_path, mask_path, object_name)


def config_from_payload(payload: dict | None) -> PipelineConfig:
    if payload is not None and not isinstance(payload, dict):
        raise ValueError("config must be an object.")
    values = payload or {}
    allowed = set(PipelineConfig.__dataclass_fields__)
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown configuration values: {sorted(unknown)}.")
    defaults = PipelineConfig()
    for name, value in values.items():
        expected = type(getattr(defaults, name))
        if isinstance(value, bool) or not isinstance(value, (int, float) if expected is float else int):
            raise ValueError(f"Configuration {name} must be {expected.__name__}.")
    return PipelineConfig(**values)


def graph_from_payload(payload: dict | None) -> NodeGraph:
    if isinstance(payload, dict) and "workflow" in payload:
        payload = payload["workflow"]
    return NodeGraph(payload if payload is not None else json.loads(DEFAULT_WORKFLOW.read_text()))


def image_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{encoded}"


def node_catalog() -> dict:
    return {
        name: {
            "inputs": node.INPUT_TYPES,
            "outputs": node.OUTPUT_TYPES,
            "input_defaults": getattr(node, "INPUT_DEFAULTS", {}),
            "category": node.CATEGORY,
            "help": NODE_HELP[name],
            "side_effects": getattr(node, "SIDE_EFFECTS", False),
        }
        for name, node in NODE_CLASS_MAPPINGS.items()
    }


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/node-types")
def get_node_types():
    return node_catalog()


@app.get("/api/workflow/default")
def get_default_workflow():
    return json.loads(DEFAULT_WORKFLOW.read_text())


@app.post("/api/workflow/validate")
def validate_workflow(payload: dict):
    try:
        graph = graph_from_payload(payload.get("workflow"))
        if payload.get("require_dataset") is not False:
            graph.require_outputs({"sample": "PREPARED_SAMPLE", "entry": "JSON_RECORD", "split": "STRING"})
        config = config_from_payload(payload.get("config"))
    except (KeyError, TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return {"valid": True, "execution_order": graph.order, "outputs": list(graph.outputs), "config": asdict(config)}


@app.post("/api/preview")
def preview(payload: dict):
    try:
        graph = graph_from_payload(payload.get("workflow"))
        graph.require_outputs({"sample": "PREPARED_SAMPLE"})
        record = record_from_payload(payload["record"])
        config = config_from_payload(payload.get("config"))
        sample = graph.run({"record": record, "config": config}, outputs=["sample"], allow_side_effects=False)[
            "sample"
        ]
        validate_sample(sample, sample.mask, config)
    except (KeyError, OSError, ValueError, TypeError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return {
        "prompt": sample.prompt,
        "target": image_data_url(sample.target),
        "source": image_data_url(sample.source),
        "mask": image_data_url(sample.mask),
        "reference": image_data_url(sample.reference),
    }


@app.post("/api/export")
def export_dataset(payload: dict):
    try:
        graph = graph_from_payload(payload.get("workflow"))
        config = config_from_payload(payload.get("config"))
        if not isinstance(payload.get("manifest_path"), str) or not payload["manifest_path"].strip():
            raise ValueError("manifest_path must be a non-empty path.")
        if not isinstance(payload.get("output_dir"), str) or not payload["output_dir"].strip():
            raise ValueError("output_dir must be a non-empty path.")
        manifest = Path(payload["manifest_path"]).expanduser()
        output = Path(payload["output_dir"]).expanduser()
        preview_count = int(payload.get("preview_count", 16))
        summary = build_dataset(manifest, output, config, preview_count, graph=graph)
    except (KeyError, OSError, ValueError, TypeError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return summary


@app.post("/api/upload")
async def upload_image(file: UploadFile):
    contents = await file.read(20 * 1024 * 1024 + 1)
    await file.close()
    if len(contents) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image uploads must be at most 20 MiB.")
    try:
        with Image.open(io.BytesIO(contents)) as image:
            image.load()
            path = Path(UPLOADS.name) / f"{uuid4().hex}.png"
            image.save(path, format="PNG")
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=422, detail="Upload must be a valid image.") from error
    return {"path": str(path)}


def serialize_color_regions(regions: ColorRegionSet) -> dict[str, Any]:
    """把区域集合转换成轻量预览 JSON，保留全尺寸结果供后续节点使用。

    Args:
        regions (ColorRegionSet): 包含原图、标签图和区域元数据的运行时对象。

    Returns:
        dict[str, Any]: kind=color_regions、类别数量、原尺寸及区域卡片列表；
        每张卡片包含编号、RGB 代表色、有效面积占比及最长边不超过 256 的裁剪预览。
        预览不是全尺寸数据导出，不改变 regions 中的任何图片。
    """
    items = []
    for region in regions.regions:
        mask = regions.labels.crop(region.bbox).point([255 if value == region.id else 0 for value in range(256)])
        rgba = regions.image.crop(region.bbox).convert("RGBA")
        rgba.putalpha(mask)
        mask.thumbnail((256, 256), Image.Resampling.NEAREST)
        rgba.thumbnail((256, 256), Image.Resampling.LANCZOS)
        items.append(
            {
                "id": region.id,
                "pixels": region.pixels,
                "fraction": region.fraction,
                "color": region.color,
                "mask": image_data_url(mask),
                "rgba": image_data_url(rgba),
            }
        )
    return {"kind": "color_regions", "count": len(items), "size": regions.image.size, "items": items}


def serialize_node_output(value) -> dict:
    if isinstance(value, ColorRegionSet):
        return serialize_color_regions(value)
    if isinstance(value, Image.Image):
        return {"kind": "image", "image": image_data_url(value), "size": value.size, "mode": value.mode}
    if isinstance(value, (str, Path)):
        try:
            path = Path(value).expanduser()
            if path.is_file():
                with Image.open(path) as image:
                    result = serialize_node_output(ImageOps.exif_transpose(image).copy())
                    result["path"] = str(path)
                    return result
        except (OSError, ValueError):
            pass
    if isinstance(value, PreparedSample):
        return {
            "kind": "sample",
            "prompt": value.prompt,
            "images": {
                name: image_data_url(getattr(value, name)) for name in ("target", "source", "mask", "reference")
            },
        }
    if is_dataclass(value) and not isinstance(value, type):
        value = {field.name: getattr(value, field.name) for field in fields(value)}
    if isinstance(value, dict):
        return {"kind": "object", "items": {str(name): serialize_node_output(item) for name, item in value.items()}}
    if isinstance(value, (list, tuple)) and any(isinstance(item, (Image.Image, dict)) for item in value):
        return {
            "kind": "object",
            "items": {str(index): serialize_node_output(item) for index, item in enumerate(value)},
        }
    return {"kind": "text", "value": str(value) if isinstance(value, Path) else value}


def node_preview_graph(definition: dict | None, node_id: str) -> NodeGraph:
    if definition is None:
        definition = json.loads(DEFAULT_WORKFLOW.read_text())
    if isinstance(definition, dict) and "workflow" in definition:
        definition = definition["workflow"]
    if not isinstance(definition, dict) or not isinstance(definition.get("nodes"), dict):
        raise ValueError("Workflow must contain a nodes object.")
    selected = {}

    def visit(current):
        if not isinstance(current, str) or current not in definition["nodes"]:
            raise ValueError(f"Unknown node: {current}.")
        if current in selected:
            return
        spec = definition["nodes"][current]
        if not isinstance(spec, dict) or not isinstance(spec.get("inputs"), dict):
            raise ValueError(f"Node {current}: inputs must be an object.")
        selected[current] = spec
        for binding in spec["inputs"].values():
            if is_link(binding):
                visit(binding["node"])

    visit(node_id)
    return NodeGraph({"nodes": selected, "outputs": {}})


def node_preview_context(graph: NodeGraph, payload: dict, allowed_save_node: str | None = None) -> dict:
    context = {}
    bindings = [binding for spec in graph.nodes.values() for binding in spec["inputs"].values()]
    if {"context": "config"} in bindings:
        context["config"] = config_from_payload(payload.get("config"))
    if {"context": "record"} in bindings:
        context["record"] = record_from_payload(payload.get("record"))
    for node_id, spec in graph.nodes.items():
        node_class = graph.registry[spec["class_type"]]
        if getattr(node_class, "SIDE_EFFECTS", False) and node_id != allowed_save_node:
            raise ValueError(f"Node {node_id}: file-writing nodes are not allowed during preview.")
        for port, expected in node_class.INPUT_TYPES.items():
            binding = spec["inputs"][port]
            if expected not in {"RGB_IMAGE", "MASK", "IMAGE"} or not isinstance(binding, str):
                continue
            if not binding.strip():
                if port in getattr(node_class, "OPTIONAL_INPUTS", set()):
                    continue
                raise ValueError(f"Node {node_id}.{port}: choose an image or connect an upstream output.")
            path = Path(binding).expanduser()
            if expected == "RGB_IMAGE":
                image = load_target(path)
            elif expected == "IMAGE":
                with Image.open(path) as opened:
                    image = ImageOps.exif_transpose(opened).copy()
            else:
                with Image.open(path) as opened:
                    image = ImageOps.exif_transpose(opened).convert("L")
                    if not getattr(node_class, "PRESERVE_MASK_VALUES", False):
                        image = image.point(lambda pixel: 255 if pixel > 127 else 0)
            key = f"studio_image:{node_id}:{port}"
            context[key] = image
            spec["inputs"][port] = {"context": key}
    return context


@app.post("/api/node/preview")
def preview_node(payload: dict):
    try:
        graph = node_preview_graph(payload.get("workflow"), payload["node_id"])
        execution = graph.start(node_preview_context(graph, payload), allow_side_effects=False)
        results = execution.get_node(payload["node_id"])
        return {
            "node_id": payload["node_id"],
            "outputs": {name: serialize_node_output(value) for name, value in results.items()},
            "executed_nodes": list(execution.cache),
        }
    except (KeyError, OSError, ValueError, TypeError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/api/node/save-images")
def save_node_images(payload: dict):
    try:
        node_id = payload["node_id"]
        graph = node_preview_graph(payload.get("workflow"), node_id)
        if graph.nodes[node_id]["class_type"] != "SaveImagesNode":
            raise ValueError("此入口仅允许执行 SaveImagesNode。")
        context = node_preview_context(graph, payload, allowed_save_node=node_id)
        execution = graph.start(context, allow_side_effects=True)
        results = execution.get_node(node_id)
        return {
            "node_id": node_id,
            "outputs": {name: serialize_node_output(value) for name, value in results.items()},
            "executed_nodes": list(execution.cache),
        }
    except (KeyError, OSError, ValueError, TypeError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


def prepare_batch(payload: dict):
    try:
        target_id = payload["node_id"]
        graph = node_preview_graph(payload.get("workflow"), target_id)
        if graph.nodes[target_id]["class_type"] != "SaveImagesNode":
            raise ValueError("请从 SaveImagesNode 启动批量保存。")
        sources = [node_id for node_id, spec in graph.nodes.items() if spec["class_type"] == "FolderImagesNode"]
        if len(sources) != 1:
            raise ValueError("保存节点的上游必须连接且只能连接一个文件夹图片循环输入节点。")
        source_id = sources[0]
        inputs = graph.nodes[source_id]["inputs"]
        if any(isinstance(value, dict) for value in inputs.values()):
            raise ValueError("批量运行时，文件夹节点的 dir、recursive、index 请直接填写，不要连接其他节点。")
        output_dir = graph.nodes[target_id]["inputs"]["dir"]
        if not isinstance(output_dir, str) or not output_dir.strip():
            raise ValueError("批量运行前请在保存节点直接填写 dir 输出目录。")
        output_root = Path(output_dir).expanduser().resolve()
        input_root = Path(inputs["dir"]).expanduser().resolve()
        if input_root.is_relative_to(output_root):
            raise ValueError("保存目录不能是输入目录或其上级目录；请选择单独的输出目录。")
        paths = [
            path
            for path in scan_folder(inputs["dir"], inputs["recursive"])
            if not path.resolve().is_relative_to(output_root)
        ]
        start_index = inputs["index"]
        if type(start_index) is not int or not 0 <= start_index < len(paths):
            raise ValueError(f"index 必须是有效的起始序号：0～{len(paths) - 1}。")
        continue_on_error = payload.get("continue_on_error", True)
        if not isinstance(continue_on_error, bool):
            raise ValueError("continue_on_error 必须是布尔值。")
        for node_id, spec in graph.nodes.items():
            if getattr(graph.registry[spec["class_type"]], "SIDE_EFFECTS", False) and node_id != target_id:
                raise ValueError("循环范围内只能有所选的一个保存节点。")
        return graph, source_id, target_id, paths, start_index, continue_on_error
    except (KeyError, OSError, ValueError, TypeError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/api/batch/plan")
def plan_batch(payload: dict):
    graph, source_id, target_id, paths, start_index, _ = prepare_batch(payload)
    return {
        "source_id": source_id,
        "target_id": target_id,
        "total": len(paths) - start_index,
        "start_index": start_index,
        "current_path": str(paths[start_index]),
        "output_dir": graph.nodes[target_id]["inputs"]["dir"],
        "executed_nodes": list(graph.order),
    }


@app.post("/api/batch/preview")
def preview_batch(payload: dict):
    graph, source_id, target_id, paths, start_index, _ = prepare_batch(payload)
    try:
        context = node_preview_context(graph, payload, allowed_save_node=target_id)
        execution = graph.start(context, allow_side_effects=True)
        execution.cache[source_id] = folder_image(paths[start_index], start_index, len(paths))
        results = execution.get_node(target_id)
        return {
            "node_id": target_id,
            "outputs": {name: serialize_node_output(value) for name, value in results.items()},
            "executed_nodes": list(execution.cache),
        }
    except (KeyError, OSError, ValueError, TypeError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/api/batch")
def start_batch(payload: dict):
    graph, source_id, target_id, paths, start_index, continue_on_error = prepare_batch(payload)
    try:
        context = node_preview_context(graph, payload, allowed_save_node=target_id)
        return BATCH_JOBS.start(graph, context, source_id, target_id, paths, start_index, continue_on_error)
    except (KeyError, OSError, ValueError, TypeError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/api/batch/latest")
def latest_batch():
    return {"job": BATCH_JOBS.latest()}


@app.get("/api/batch/{job_id}")
def get_batch(job_id: str):
    try:
        return BATCH_JOBS.get(job_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.post("/api/batch/{job_id}/cancel")
def cancel_batch(job_id: str):
    try:
        return BATCH_JOBS.cancel(job_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.get("/api/default-config")
def get_default_config():
    return asdict(PipelineConfig())


@app.get("/api/config-help")
def get_config_help():
    return CONFIG_HELP
