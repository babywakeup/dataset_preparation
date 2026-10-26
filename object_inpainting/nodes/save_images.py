import io
from pathlib import Path
from uuid import uuid4

from PIL import Image

from .base import Node


class SaveImagesNode(Node):
    SIDE_EFFECTS = True
    INPUT_TYPES = {"img1": "IMAGE", "img2": "IMAGE", "img3": "IMAGE", "img4": "IMAGE", "dir": "PATH"}
    OPTIONAL_INPUTS = {"img1", "img2", "img3", "img4"}
    OUTPUT_TYPES = {"paths": "JSON_RECORD"}
    CATEGORY = "dataset"

    def execute(self, dir: str | Path, img1=None, img2=None, img3=None, img4=None) -> dict:
        if not isinstance(dir, (str, Path)) or not str(dir).strip():
            raise ValueError("请填写保存目录 dir。")
        encoded = {}
        for name, image in (("img1", img1), ("img2", img2), ("img3", img3), ("img4", img4)):
            if image is None or (isinstance(image, str) and not image.strip()):
                continue
            if not isinstance(image, Image.Image):
                raise ValueError(f"{name} 必须是图片。")
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            encoded[name] = buffer.getvalue()
        if not encoded:
            raise ValueError("至少需要一张图片。")

        root = Path(dir).expanduser().resolve()
        filename = f"{uuid4().hex}.png"
        paths = {}
        created = []
        try:
            for name, contents in encoded.items():
                folder = root / name
                folder.mkdir(parents=True, exist_ok=True)
                path = folder / filename
                with path.open("xb") as handle:
                    created.append(path)
                    handle.write(contents)
                paths[name] = str(path)
        except OSError:
            for path in created:
                path.unlink(missing_ok=True)
            raise
        return {"paths": paths}
