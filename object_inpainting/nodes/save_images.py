import io
import re
from pathlib import Path
from uuid import uuid4

from PIL import Image

from ..node_contracts import SaveImageItem, SaveImagesInputs, SaveImagesOutputs
from ..port_types import FilePath, OptionalImage, OptionalImageList
from .base import Node


IMAGE_SLOT = re.compile(r"img[1-9][0-9]*\Z")


def safe_name(value: str, label: str) -> str:
    """A single safe path component; never accept a relative/absolute path."""
    if not isinstance(value, str):
        raise ValueError(f"{label} 必须是字符串。")
    name = value.strip()
    if (
        not name or name in {".", ".."} or name.endswith(".")
        or any(char in '/\\<>:"|?*' or ord(char) < 32 or ord(char) == 127 for char in name)
    ):
        raise ValueError(f"{label} 必须是单个有效名称，不能包含路径分隔符、控制字符或 . / ..。")
    return name


class SaveImagesNode(Node):
    SIDE_EFFECTS = True
    INPUT_SCHEMA = SaveImagesInputs
    OPTIONAL_INPUTS = {"img1", "img2", "img3", "img4", "images"}
    INPUT_DEFAULTS = {"filename": ""}
    DYNAMIC_IMAGE_INPUTS = True
    OUTPUT_SCHEMA = SaveImagesOutputs
    CATEGORY = "dataset"

    @classmethod
    def prepare_spec(cls, spec: dict) -> None:
        if "save_items" not in spec:
            # Old graphs retain their original four slots and list input.
            ids = set(f"img{index}" for index in range(1, 5))
            ids.update(key for key in spec["inputs"] if isinstance(key, str) and IMAGE_SLOT.fullmatch(key))
            spec["save_items"] = [{"id": key, "name": ""} for key in sorted(ids, key=lambda key: int(key[3:]))]
        items = spec["save_items"]
        if not isinstance(items, list):
            raise ValueError("save_items 必须是图片保存项列表。")
        ids, names = set(), set()
        for item in items:
            if not isinstance(item, dict) or set(item) != {"id", "name"}:
                raise ValueError("每个保存项必须包含 id 和 name。")
            key = item["id"]
            if not isinstance(key, str) or not IMAGE_SLOT.fullmatch(key) or key in ids:
                raise ValueError("保存项 id 必须是唯一的 imgN（N 为正整数）。")
            ids.add(key)
            if not isinstance(item["name"], str):
                raise ValueError(f"{key} 保存名称必须是字符串。")
            name = safe_name(item["name"].strip() or key, f"{key} 保存名称")
            if name.casefold() in names:
                raise ValueError(f"保存名称重复：{name}。请为每项填写不同名称。")
            names.add(name.casefold())
        minimum = max((int(key[3:]) for key in ids), default=0) + 1
        next_id = spec.get("save_next_id", minimum)
        if type(next_id) is not int or next_id < minimum:
            raise ValueError("save_next_id 必须大于所有现有图片项编号。")
        spec["save_next_id"] = next_id

    @classmethod
    def input_types(cls, spec: dict) -> dict[str, str]:
        return {
            **{item["id"]: "IMAGE" for item in spec["save_items"]},
            "images": "IMAGE_LIST", "dir": "PATH", "filename": "STRING",
        }

    @classmethod
    def optional_input(cls, port: str) -> bool:
        return port == "images" or IMAGE_SLOT.fullmatch(port) is not None

    def execute_with_spec(self, inputs: dict, spec: dict) -> SaveImagesOutputs:
        items: list[SaveImageItem] = spec["save_items"]
        return self._save(
            inputs["dir"], [(item["id"], inputs[item["id"]]) for item in items],
            inputs.get("images"), inputs["filename"], {item["id"]: item["name"] for item in items},
        )

    def execute(
        self, dir: FilePath, img1: OptionalImage = None, img2: OptionalImage = None,
        img3: OptionalImage = None, img4: OptionalImage = None, images: OptionalImageList = None,
        filename: str = "", **extra_images: OptionalImage,
    ) -> SaveImagesOutputs:
        if any(not IMAGE_SLOT.fullmatch(key) for key in extra_images):
            raise ValueError("额外图片输入必须命名为 imgN。")
        inputs = [("img1", img1), ("img2", img2), ("img3", img3), ("img4", img4), *extra_images.items()]
        return self._save(dir, inputs, images, filename, {})

    def _save(
        self, dir: FilePath, inputs: list[tuple[str, OptionalImage]], images: OptionalImageList,
        filename: str, names: dict[str, str],
    ) -> SaveImagesOutputs:
        if not isinstance(dir, (str, Path)) or not str(dir).strip():
            raise ValueError("请填写保存目录 dir。")
        if not isinstance(filename, str):
            raise ValueError("filename 必须是字符串。")
        if filename.strip():
            given = safe_name(filename, "整组文件名")
            filename = f"{safe_name(Path(given).stem, '整组文件名主干')}.png"
        else:
            filename = f"{uuid4().hex}.png"
        encoded = {}
        if images is not None:
            if not isinstance(images, list) or any(not isinstance(image, Image.Image) for image in images):
                raise ValueError("images 必须连接图片列表输出；列表中的每项都必须是图片。")
            inputs.extend((f"images_{index}", image) for index, image in enumerate(images))
        for name, image in inputs:
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
        destinations = {}
        used_names = set()
        for key in encoded:
            folder_name = safe_name(names.get(key, "").strip() or key, f"{key} 保存名称")
            if folder_name.casefold() in used_names:
                raise ValueError(f"保存名称重复：{folder_name}。")
            used_names.add(folder_name.casefold())
            folder = root / folder_name
            if folder.is_symlink() or folder.resolve().parent != root:
                raise ValueError(f"保存子目录不能使用符号链接或指向根目录之外：{folder_name}。")
            if folder.exists() and not folder.is_dir():
                raise ValueError(f"保存子目录已被文件占用：{folder_name}。")
            path = folder / filename
            if path.exists() or path.is_symlink():
                raise ValueError(f"保存文件已存在，不会覆盖：{path}。请修改整组文件名。")
            destinations[key] = path
        paths = {}
        created = []
        try:
            for name, contents in encoded.items():
                path = destinations[name]
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as handle:
                    created.append(path)
                    handle.write(contents)
                paths[name] = str(path)
        except OSError:
            for path in created:
                path.unlink(missing_ok=True)
            raise
        return {"paths": paths}
