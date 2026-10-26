import json
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

from .config import PipelineConfig
from .graph import NodeGraph
from .quality import PreparedSample, validate_sample
from .records import ObjectRecord, read_records


DEFAULT_WORKFLOW = Path(__file__).parent / "workflows" / "object_reference.json"


def prepare_sample(record: ObjectRecord, config: PipelineConfig, graph: NodeGraph | None = None) -> PreparedSample:
    graph = graph if graph is not None else NodeGraph.from_file(DEFAULT_WORKFLOW)
    return graph.run({"record": record, "config": config}, outputs=["sample"])["sample"]


def build_dataset(
    manifest: Path,
    output: Path,
    config: PipelineConfig,
    preview_count: int = 16,
    workflow: Path | None = None,
    graph: NodeGraph | None = None,
) -> dict:
    if preview_count < 0:
        raise ValueError("preview_count must be non-negative.")
    if workflow is not None and graph is not None:
        raise ValueError("Specify either workflow or graph, not both.")
    graph = graph if graph is not None else NodeGraph.from_file(workflow if workflow is not None else DEFAULT_WORKFLOW)
    required_outputs = {"sample": "PREPARED_SAMPLE", "entry": "JSON_RECORD", "split": "STRING"}
    if preview_count:
        required_outputs["preview"] = "PATH"
    graph.require_outputs(required_outputs)
    manifest = manifest.resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    output = output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output directory must be empty; existing datasets are never overwritten.")
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "validation": 0, "rejected": 0}
    with TemporaryDirectory(prefix=".object-inpainting-", dir=output.parent) as directory:
        staging = Path(directory)
        for name in counts:
            (staging / f"{name}.jsonl").touch()
        graph.save(staging / "workflow.json")
        with (staging / "rejected.jsonl").open("w", encoding="utf-8") as rejected:
            for record in read_records(manifest):
                execution = graph.start(
                    {
                        "record": record,
                        "config": config,
                        "output": staging,
                        "preview_path": staging / "previews" / f"{record.sample_id}.png",
                    }
                )
                try:
                    sample = execution.get("sample")
                    validate_sample(sample, sample.mask, config)
                except (ValueError, OSError) as error:
                    counts["rejected"] += 1
                    rejection = {
                        "sample_id": record.sample_id,
                        "target": str(record.target_path),
                        "mask": str(record.mask_path),
                        "reason": str(error),
                    }
                    rejected.write(json.dumps(rejection, ensure_ascii=False) + "\n")
                    continue
                execution.get("entry")
                split = execution.get("split")
                if split not in {"train", "validation"}:
                    raise ValueError("Workflow split must be train or validation.")
                counts[split] += 1
                if counts["train"] + counts["validation"] <= preview_count:
                    execution.get("preview")
        summary = {"counts": counts, "input_manifest": str(manifest), "config": asdict(config)}
        with (staging / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        if output.exists():
            output.rmdir()
        staging.replace(output)
    if not counts["train"] + counts["validation"]:
        raise ValueError("No usable samples; inspect rejected.jsonl in the output directory.")
    return summary
