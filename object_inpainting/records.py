import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


@dataclass(frozen=True)
class ObjectRecord:
    sample_id: str
    target_path: Path
    mask_path: Path
    object_name: str


def read_records(manifest: Path):
    root = manifest.resolve().parent
    seen = set()
    with manifest.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            target = (root / row["target"]).resolve()
            mask = (root / row["mask"]).resolve()
            object_name = row["object_name"]
            if not isinstance(object_name, str) or not object_name.strip():
                raise ValueError(f"Record {line_number}: object_name must be a non-empty string.")
            object_name = object_name.strip()
            key = json.dumps([str(target), str(mask), object_name], ensure_ascii=False)
            sample_id = hashlib.sha256(key.encode()).hexdigest()[:24]
            if sample_id in seen:
                raise ValueError(f"Record {line_number}: duplicate target/mask/object_name.")
            seen.add(sample_id)
            yield ObjectRecord(sample_id, target, mask, object_name)


def load_target(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def load_instance_mask(path: Path, target: Image.Image) -> Image.Image:
    with Image.open(path) as image:
        mask = ImageOps.exif_transpose(image).convert("L")
    if target.size != mask.size:
        raise ValueError("Target and instance mask must have identical aligned dimensions.")
    return Image.fromarray(np.where(np.asarray(mask) > 127, 255, 0).astype(np.uint8))


def load_images(record: ObjectRecord) -> tuple[Image.Image, Image.Image]:
    target = load_target(record.target_path)
    return target, load_instance_mask(record.mask_path, target)
