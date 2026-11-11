"""Default Studio image filenames to the unambiguous original input name."""

from pathlib import Path

from object_inpainting.graph import is_link


def infer_original_filenames(graph):
    for node, spec in graph.nodes.items():
        if spec["class_type"] != "SaveImagesNode":
            continue
        filename = spec["inputs"].get("filename", "")
        if not isinstance(filename, str) or filename.strip():
            continue
        ancestors, pending = set(), [node]
        while pending:
            current = pending.pop()
            if current in ancestors:
                continue
            ancestors.add(current)
            pending.extend(value["node"] for value in graph.nodes[current]["inputs"].values() if is_link(value))
        folders = [source for source in ancestors if graph.nodes[source]["class_type"] == "FolderImagesNode"]
        if len(folders) == 1:
            spec["inputs"]["filename"] = {"node": folders[0], "output": "filename"}
        elif not folders:
            sources = {source["inputs"]["path"] for source in (graph.nodes[id] for id in ancestors)
                       if source["class_type"] == "LoadTargetNode"
                       and isinstance(source["inputs"]["path"], str) and source["inputs"]["path"].strip()}
            if len(sources) == 1:
                spec["inputs"]["filename"] = Path(next(iter(sources))).name
