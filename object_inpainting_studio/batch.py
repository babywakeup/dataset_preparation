from copy import deepcopy
from threading import Event, Lock, Thread
from uuid import uuid4

from object_inpainting.graph import GraphExecution
from object_inpainting.nodes.folder_images import folder_image


class BatchCancelled(Exception):
    pass


class BatchExecution(GraphExecution):
    def __init__(self, graph, context, cancel):
        super().__init__(graph, context, allow_side_effects=True)
        self.cancel = cancel

    def _execute(self, node_id):
        if self.cancel.is_set():
            raise BatchCancelled()
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

    def start(self, graph, context, source_id, target_id, paths, start_index, continue_on_error):
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
                "continue_on_error": continue_on_error,
            }
            self.jobs[job_id] = job
            self.cancels[job_id] = cancel
            thread = Thread(
                target=self._run,
                args=(job_id, graph, context, source_id, target_id, tuple(paths), start_index),
                daemon=True,
            )
            self.threads[job_id] = thread
            snapshot = deepcopy(job)
            thread.start()
            return snapshot

    def _run(self, job_id, graph, context, source_id, target_id, paths, start_index):
        cancel = self.cancels[job_id]
        job = self.jobs[job_id]
        try:
            for index in range(start_index, len(paths)):
                if cancel.is_set():
                    break
                path = paths[index]
                with self.lock:
                    job["current_path"] = str(path)
                    job["current_node"] = source_id
                try:
                    execution = BatchExecution(graph, context, cancel)
                    execution.cache[source_id] = folder_image(path, index, len(paths))
                    for node_id in graph.order:
                        with self.lock:
                            job["current_node"] = node_id
                        execution.get_node(node_id)
                    result = {"source": str(path), "paths": execution.cache[target_id]["paths"]}
                except BatchCancelled:
                    break
                except Exception as error:
                    result = {"source": str(path), "error": str(error)}
                with self.lock:
                    job["results"].append(result)
                    job["processed"] += 1
                    job["failed" if "error" in result else "succeeded"] += 1
                if "error" in result and not job["continue_on_error"]:
                    with self.lock:
                        job["status"] = "failed"
                        job["error"] = result["error"]
                    return
            with self.lock:
                job["status"] = "cancelled" if cancel.is_set() else "completed"
        except Exception as error:
            with self.lock:
                job["status"] = "failed"
                job["error"] = str(error)
        finally:
            with self.lock:
                job["current_node"] = None
