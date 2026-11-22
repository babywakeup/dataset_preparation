import logging
from copy import deepcopy
from threading import Event, Lock, Thread
from uuid import uuid4

from object_inpainting.graph import GraphExecution, GraphExecutionCancelled, NodeExecutionError
from object_inpainting.nodes.folder_images import folder_image

from .conditional_loop import run_conditional
from .batch_scope import describe_batch, finish_batch_outputs


LOGGER = logging.getLogger(__name__)


class BatchCancelled(GraphExecutionCancelled):
    pass


class BatchExecution(GraphExecution):
    def __init__(self, graph, context, cancel, progress=None):
        super().__init__(graph, context, allow_side_effects=True)
        self.cancel = cancel
        self.progress = progress

    def _execute(self, node_id):
        if self.cancel.is_set():
            raise BatchCancelled()
        if self.progress is not None and node_id not in self.cache:
            self.progress({"current_node": node_id})
        return super()._execute(node_id)

    def _resolve(self, binding):
        result = super()._resolve(binding)
        if self.cancel.is_set():
            raise BatchCancelled()
        return result


class BatchJobs:
    def __init__(self):
        self.lock = Lock()
        self.jobs = {}
        self.cancels = {}
        self.threads = {}

    def latest(self):
        with self.lock:
            return deepcopy(next(reversed(self.jobs.values()))) if self.jobs else None

    def get(self, job_id):
        with self.lock:
            if job_id not in self.jobs:
                raise KeyError("批量任务不存在；服务重启后任务记录会清空。")
            return deepcopy(self.jobs[job_id])

    def cancel(self, job_id):
        with self.lock:
            if job_id not in self.jobs:
                raise KeyError("批量任务不存在。")
            job = self.jobs[job_id]
            if job["status"] in {"running", "cancelling"}:
                self.cancels[job_id].set()
                job["status"] = "cancelling"
            return deepcopy(job)

    def start(self, graph, context, source_id, target_id, paths, start_index, continue_on_error, *,
              conditional=None, preview_serializer=None, visualize=True):
        if type(visualize) is not bool:
            raise ValueError("visualize 必须是布尔值。")
        with self.lock:
            if any(job["status"] in {"running", "cancelling"} for job in self.jobs.values()):
                raise ValueError("已有批量任务运行中，请先等待完成或停止。")
            while len(self.jobs) >= 20:
                oldest = next(iter(self.jobs))
                del self.jobs[oldest]
                self.cancels.pop(oldest, None)
                self.threads.pop(oldest, None)
            job_id = uuid4().hex
            cancel = Event()
            job = {
                "id": job_id,
                "source_id": source_id,
                "target_id": target_id,
                "status": "running",
                "total": len(paths) - start_index,
                "processed": 0,
                "succeeded": 0,
                "failed": 0,
                "current_path": None,
                "current_node": None,
                "results": [],
                "error": None,
                "node_error": None,
                "continue_on_error": continue_on_error,
                "skipped": 0,
                "conditional_loop": conditional.describe() if conditional is not None else None,
                "current_iteration": 0,
                "current_loop_id": None,
                "current_loop_name": None,
                "current_loop_index": 0,
                "current_loop_count": 0,
                "max_iterations": conditional.max_iterations if conditional is not None else 0,
                "iteration_history": [],
                "stop_reason": None,
                "visualize": visualize,
                "loop_previews": {},
                "visualization_revision": 0,
                "execution_plan": describe_batch(graph, source_id, target_id, conditional)
                                  if hasattr(graph, "batch_terminals") else None,
                "execution_phase": "input",
                "current_item": None,
            }
            if getattr(conditional, "preview_loop_id", None) is not None:
                job.update(preview_loop_id=conditional.preview_loop_id,
                           required_loop_ids=[item["id"] for item in conditional.loops])
            if getattr(conditional, "preview_node_id", None) is not None:
                job.update(preview_node_id=conditional.preview_node_id,
                           required_loop_ids=[item["id"] for item in conditional.loops])
            self.jobs[job_id] = job
            self.cancels[job_id] = cancel
            thread = Thread(
                target=self._run,
                args=(job_id, graph, context, source_id, target_id, tuple(paths), start_index,
                        conditional, preview_serializer),
                daemon=True,
            )
            self.threads[job_id] = thread
            snapshot = deepcopy(job)
            thread.start()
            return snapshot

    def _run(self, job_id, graph, context, source_id, target_id, paths, start_index, conditional=None,
             preview_serializer=None):
        cancel = self.cancels[job_id]
        job = self.jobs[job_id]

        def serialize(outputs):
            if not job["visualize"] or preview_serializer is None:
                return {}
            return {node: {name: preview_serializer(value) for name, value in values.items()}
                    for node, values in outputs.items()}

        def progress(update):
            update = dict(update)
            snapshot = update.pop("loop_preview", None)
            if snapshot is not None and job["visualize"] and preview_serializer is not None:
                snapshot = {**snapshot, "outputs": serialize(snapshot["outputs"])}
            else:
                snapshot = None
            with self.lock:
                node = update.get("current_node")
                if node is not None:
                    job["execution_phase"] = ("input" if node == source_id else "retry"
                                               if conditional is not None and node in conditional.body else "results")
                job.update(update)
                if snapshot is not None:
                    job["loop_previews"][snapshot["loop_id"]] = snapshot
                    job["visualization_revision"] += 1

        try:
            for index in range(start_index, len(paths)):
                if cancel.is_set():
                    break
                path = paths[index]
                with self.lock:
                    job["current_path"] = str(path) if path is not None else "current"
                    job["execution_phase"] = "input"
                    job["current_item"] = {"id": f"{job_id}:{index}", "index": index,
                                           "total": len(paths), "path": job["current_path"], "status": "running"}
                    job["current_node"] = source_id
                    job["current_iteration"] = 0
                    job["current_loop_id"] = None
                    job["current_loop_name"] = None
                    job["current_loop_index"] = 0
                    job["current_loop_count"] = 0
                    job["iteration_history"] = []
                    job["stop_reason"] = None
                    job["loop_previews"] = {}
                    # Keep images for the current input only, not a folder's
                    # entire image history. Scalar reports remain available.
                    for previous in job["results"]:
                        if "preview" in previous:
                            previous["preview"] = {}
                try:
                    if conditional is not None:
                        initial = {}
                        if source_id is not None:
                            try:
                                initial[source_id] = folder_image(path, index, len(paths))
                            except Exception as error:
                                raise NodeExecutionError(source_id, graph.nodes[source_id]["class_type"], error) from error

                        result = {"source": str(path) if path is not None else "current",
                                  **run_conditional(conditional, context, cancel, progress, initial)}
                        if "preview" in result:
                            if hasattr(graph, "batch_terminals"):
                                progress({"loop_preview": {"kind": "batch", "loop_id": "batch",
                                          "iteration": index + 1, "accepted": result.get("accepted", False),
                                          "outputs": result["preview"]}})
                            result["preview"] = serialize(result["preview"])
                    else:
                        execution = BatchExecution(graph, context, cancel, progress)
                        try:
                            execution.cache[source_id] = folder_image(path, index, len(paths))
                        except Exception as error:
                            raise NodeExecutionError(source_id, graph.nodes[source_id]["class_type"], error) from error
                        writers = getattr(graph, "batch_write_nodes", {target_id})
                        ordered = [n for n in graph.order if n not in writers]
                        for node_id in ordered:
                            with self.lock:
                                job["current_node"] = node_id
                            execution.get_node(node_id)
                            progress({"loop_preview": {"kind": "batch", "loop_id": "batch", "iteration": index + 1,
                                      "accepted": node_id == target_id,
                                      "outputs": {node_id: execution.cache[node_id]}}})
                        outputs = finish_batch_outputs(execution, target_id)
                        progress({"loop_preview": {"kind": "batch", "loop_id": "batch",
                                  "iteration": index + 1, "accepted": True, "outputs": outputs["preview"]}})
                        result = {"source": str(path), **outputs, "preview": serialize(outputs["preview"])}
                except GraphExecutionCancelled:
                    with self.lock:
                        job["stop_reason"] = "cancelled"
                        job["current_item"]["status"] = "cancelled"
                    break
                except Exception as error:
                    LOGGER.exception("批量任务 %s，输入 %s 失败：%s", job_id, path, error)
                    result = {"source": str(path) if path is not None else "current", "error": str(error)}
                    if conditional is not None:
                        result.update(accepted=False, stop_reason="error", iterations=job["current_iteration"],
                                      history=deepcopy(job["iteration_history"]))
                        with self.lock:
                            job["stop_reason"] = "error"
                    if isinstance(error, NodeExecutionError):
                        result["node_error"] = error.as_dict()
                with self.lock:
                    status = "failed" if "error" in result else "skipped" if result.get("skipped") else "succeeded"
                    job["current_item"]["status"] = status
                    job["execution_phase"] = "completed"
                    result.update(item_id=job["current_item"]["id"], item_index=index)
                    job["results"].append(result)
                    job["processed"] += 1
                    job["failed" if "error" in result else "skipped" if result.get("skipped") else "succeeded"] += 1
                if "error" in result and not job["continue_on_error"]:
                    with self.lock:
                        job["status"] = "failed"
                        job["error"] = result["error"]
                        job["node_error"] = result.get("node_error")
                    return
            with self.lock:
                job["status"] = "cancelled" if cancel.is_set() else "completed"
        except Exception as error:
            LOGGER.exception("批量任务 %s 失败：%s", job_id, error)
            with self.lock:
                job["status"] = "failed"
                job["error"] = str(error)
                if isinstance(error, NodeExecutionError):
                    job["node_error"] = error.as_dict()
        finally:
            with self.lock:
                job["current_node"] = None
