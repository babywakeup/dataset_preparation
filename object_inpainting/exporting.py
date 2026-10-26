import hashlib
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from .quality import PreparedSample
from .records import ObjectRecord


def choose_split(target_path: Path, validation_fraction: float, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{target_path.resolve()}".encode()).digest()
    fraction = int.from_bytes(digest[:8], "big") / 2**64
    return "validation" if fraction < validation_fraction else "train"


def export_sample(sample: PreparedSample, record: ObjectRecord, output: Path, split: str) -> dict:
    paths = {}
    for name, image in (
        ("target", sample.target),
        ("source", sample.source),
        ("mask", sample.mask),
        ("objects", sample.reference),
    ):
        relative = Path(split, name, f"{record.sample_id}.png")
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(destination, format="PNG")
        paths[name] = relative.as_posix()
    return {
        "source": paths["source"],
        "references": [paths["objects"]],
        "prompt": sample.prompt,
        "target": paths["target"],
        "mask": paths["mask"],
    }


def export_preview(sample: PreparedSample, destination: Path) -> None:
    cell = 256
    canvas = Image.new("RGB", (4 * cell, cell + 24), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (name, image) in enumerate(
        (("target", sample.target), ("source", sample.source), ("mask", sample.mask), ("object", sample.reference))
    ):
        tile = ImageOps.contain(image.convert("RGB"), (cell, cell), Image.Resampling.LANCZOS)
        canvas.paste(tile, (index * cell + (cell - tile.width) // 2, 24 + (cell - tile.height) // 2))
        draw.text((index * cell + 8, 4), name, fill="black")
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="PNG")
