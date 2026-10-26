from pathlib import Path

from ..records import load_target
from .base import Node


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def scan_folder(dir: str | Path, recursive: int = 0) -> list[Path]:
    if not isinstance(dir, (str, Path)) or not str(dir).strip():
        raise ValueError("请填写图片文件夹 dir（服务器路径）。")
    if type(recursive) is not int or recursive not in (0, 1):
        raise ValueError("recursive 必须是 0 或 1。")
    root = Path(dir).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"图片文件夹不存在：{root}")
    paths = root.rglob("*") if recursive else root.iterdir()
    images = sorted(
        (path for path in paths if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: (path.relative_to(root).as_posix().casefold(), path.relative_to(root).as_posix()),
    )
    if not images:
        raise ValueError("文件夹中没有支持的图片：PNG/JPEG/WebP/BMP/TIFF。")
    return images


def folder_image(path: Path, index: int, total: int) -> dict:
    return {"image": load_target(path), "path": str(path), "filename": path.name, "index": index, "total": total}


class FolderImagesNode(Node):
    INPUT_TYPES = {"dir": "PATH", "recursive": "INT", "index": "INT"}
    INPUT_DEFAULTS = {"dir": "", "recursive": 0, "index": 0}
    OUTPUT_TYPES = {"image": "RGB_IMAGE", "path": "PATH", "filename": "STRING", "index": "INT", "total": "INT"}
    CATEGORY = "input"

    def execute(self, dir: str | Path, recursive: int = 0, index: int = 0) -> dict:
        paths = scan_folder(dir, recursive)
        if type(index) is not int or not 0 <= index < len(paths):
            raise ValueError(f"index 必须在 0～{len(paths) - 1} 之间。")
        return folder_image(paths[index], index, len(paths))
