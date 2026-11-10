"""Bounded, process-local preview results; never reconstruct runtime objects from JSON."""

import hashlib
import json
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from threading import RLock
from time import monotonic
from uuid import uuid4

from object_inpainting.graph import NodeExecutionError, is_link
from object_inpainting.nodes.folder_images import scan_folder


def file_versions(value):
    """Include external file changes even when an input path stays unchanged."""
    if isinstance(value, dict):
        return {key: file_versions(item) for key, item in value.items()}
    if isinstance(value, list):
        return [file_versions(item) for item in value]
    if isinstance(value, str) and value.strip():
        try:
            path = Path(value).expanduser()
            stat = path.stat()
            return [value, str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
        except (OSError, ValueError):
            pass
    return value


class PreviewCache:
    def __init__(self, max_sessions=8, max_nodes=64, ttl=900):
        self.sessions = OrderedDict()
        self.lock = RLock()
        self.max_sessions = max_sessions
        self.max_nodes = max_nodes
        self.ttl = ttl

    def run(self, graph, payload, context_factory, allow_side_effects=False):
        """Reuse ancestors only. An explicit test always reruns its target node."""
        with self.lock:
            now = monotonic()
            for key, (touched, _) in list(self.sessions.items()):
                if now - touched > self.ttl:
                    del self.sessions[key]
            session_id = payload.get("preview_session")
            if not isinstance(session_id, str) or session_id not in self.sessions:
                session_id = uuid4().hex
                self.sessions[session_id] = (now, OrderedDict())
            _, entries = self.sessions[session_id]
            self.sessions[session_id] = (now, entries)
            self.sessions.move_to_end(session_id)
            while len(self.sessions) > self.max_sessions:
                self.sessions.popitem(last=False)

            # Context decoding rewrites image path bindings, so snapshot first.
            specs = deepcopy(graph.nodes)
            execution = graph.start(context_factory(), allow_side_effects=allow_side_effects)
            revisions = {}
            executed, reused = [], []
            target = payload["node_id"]
            for node_id in graph.order:
                spec = specs[node_id]
                cls = graph.registry[spec["class_type"]]
                inputs = spec["inputs"]
                dependencies = {
                    binding["node"]: revisions[binding["node"]]
                    for binding in inputs.values() if is_link(binding)
                }
                contexts = {
                    binding["context"]: payload.get(binding["context"])
                    for binding in inputs.values()
                    if isinstance(binding, dict) and set(binding) == {"context"}
                }
                files = None
                literal_folder = (
                    isinstance(inputs.get("dir"), str) and type(inputs.get("recursive")) is int
                )
                if spec["class_type"] == "FolderImagesNode" and literal_folder:
                    try:
                        files = [str(path) for path in scan_folder(inputs["dir"], inputs["recursive"])]
                    except (OSError, ValueError, TypeError) as error:
                        raise NodeExecutionError(node_id, spec["class_type"], error) from error
                signature = hashlib.sha256(json.dumps(
                    [id(cls), file_versions(spec), file_versions(contexts),
                     dependencies, file_versions(files)], sort_keys=True,
                ).encode()).hexdigest()
                cached = entries.get(node_id)
                side_effects = getattr(cls, "SIDE_EFFECTS", False)
                reusable = not side_effects and (spec["class_type"] != "FolderImagesNode" or literal_folder)
                if node_id != target and reusable and cached and cached[0] == signature:
                    execution.cache[node_id] = deepcopy(cached[2])
                    revisions[node_id] = cached[1]
                    entries.move_to_end(node_id)
                    reused.append(node_id)
                    continue
                # Remove stale output before executing, including failed retests.
                entries.pop(node_id, None)
                try:
                    result = execution.get_node(node_id)
                finally:
                    # TTL measures idle time, not loading/inference (even on failure).
                    self.sessions[session_id] = (monotonic(), entries)
                executed.append(node_id)
                revisions[node_id] = uuid4().hex
                if reusable:
                    entries[node_id] = (signature, revisions[node_id], deepcopy(result))
                    while len(entries) > self.max_nodes:
                        entries.popitem(last=False)
            self.sessions[session_id] = (monotonic(), entries)
            return execution.cache[target], session_id, executed, reused
