"""Bounded Studio retry regions. The surrounding workflow remains a DAG.

No image history is retained: only the current attempt, accepted regions and
isolated fixed inputs live in memory before saving or returning a preview.
"""

import math
import operator
from copy import deepcopy
from graphlib import CycleError, TopologicalSorter
from time import monotonic

from object_inpainting.graph import GraphExecution, GraphExecutionCancelled, is_link

from .batch_scope import finish_batch_outputs


OPERATORS = {"<=": operator.le, ">=": operator.ge, "<": operator.lt,
             ">": operator.gt, "==": operator.eq, "!=": operator.ne}
SCALARS = {"FLOAT", "INT", "BOOL", "BOOLEAN"}


def reject_early_writers(graph, body, writers):
    pending, visited = list(body), set()
    while pending:
        node = pending.pop()
        if node in visited:
            continue
        visited.add(node)
        if node in writers:
            raise ValueError("条件验收的上游不能包含写盘节点；请将结果写盘放在验收之后。")
        pending.extend(value["node"] for value in graph.nodes[node]["inputs"].values() if is_link(value))


def loop_settings(definition):
    if isinstance(definition, dict) and "workflow" in definition:
        definition = definition["workflow"]
    settings = definition.get("conditional_loop") if isinstance(definition, dict) else None
    if settings is not None and not isinstance(settings, dict):
        raise ValueError("conditional_loop 必须是对象或 null。")
    if settings is not None and "loops" in settings:
        _multiple_settings(settings)
    return settings


def _multiple_settings(settings):
    if set(settings) != {"loops"}:
        raise ValueError("conditional_loop 不能混合 flat 字段和 loops。")
    loops = settings["loops"]
    if not isinstance(loops, list) or not 1 <= len(loops) <= 32:
        raise ValueError("loops 必须包含 1～32 个循环。")
    ids = set()
    for item in loops:
        if not isinstance(item, dict) or "loops" in item:
            raise ValueError("每个循环必须是非嵌套对象。")
        loop_id = item.get("id")
        if not isinstance(loop_id, str) or not loop_id.strip() or loop_id in ids:
            raise ValueError("循环 id 必须非空且唯一。")
        ids.add(loop_id)
        if "name" in item and (not isinstance(item["name"], str) or not item["name"].strip()):
            raise ValueError("循环 name 必须为非空字符串。")
    return loops


def condition_nodes(settings):
    if "loops" in settings:
        nodes = []
        for item in _multiple_settings(settings):
            ends = item.get("end_nodes")
            if not isinstance(ends, list) or not ends or any(not isinstance(n, str) or not n for n in ends):
                raise ValueError("每个循环必须显式指定至少一个 end_nodes。")
            nodes.extend(condition_nodes(item))
            nodes.extend(ends)
        return list(dict.fromkeys(nodes))
    conditions = settings.get("conditions")
    if not isinstance(conditions, list) or not 1 <= len(conditions) <= 64:
        raise ValueError("条件循环需要 1～64 个评分条件。")
    if any(not isinstance(item, dict) or not isinstance(item.get("node"), str) for item in conditions):
        raise ValueError("每个条件必须指定 node、output、operator 和 value。")
    return [item["node"] for item in conditions]


class ConditionalPlan:
    def __init__(self, graph, target_id, settings, definition):
        self.graph = graph
        self.target_id = target_id
        self.settings = deepcopy(settings)
        batch_writers = getattr(graph, "batch_write_nodes", {target_id})
        condition_nodes(settings)
        if graph.nodes[target_id]["class_type"] != "SaveImagesNode" and not hasattr(graph, "batch_terminals"):
            raise ValueError("条件循环必须以 SaveImagesNode 为保存终点。")
        for key in ("start_nodes", "seed_nodes"):
            values = settings.get(key, [])
            if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
                raise ValueError(f"{key} 必须是节点 ID 列表。")
            if len(set(values)) != len(values):
                raise ValueError(f"{key} 不能包含重复节点。")
        starts = settings.get("start_nodes", [])
        if not starts or any(node not in graph.nodes or (node == target_id and not hasattr(graph, "batch_terminals")) for node in starts):
            raise ValueError("请选择有效的循环开始节点，且不能是保存终点。")
        self.mode = settings.get("mode", "all")
        if self.mode not in {"all", "any"}:
            raise ValueError("条件 mode 必须为 all 或 any。")
        self.max_iterations = settings.get("max_iterations", 10)
        if type(self.max_iterations) is not int or not 1 <= self.max_iterations <= 1000:
            raise ValueError("max_iterations 必须是 1～1000 的整数。")
        self.timeout = settings.get("timeout_seconds", 0)
        if type(self.timeout) not in (int, float) or not math.isfinite(self.timeout) or not 0 <= self.timeout <= 86400:
            raise ValueError("timeout_seconds 必须是 0～86400 的有限数字。")
        self.on_exhausted = settings.get("on_exhausted", "fail")
        if self.on_exhausted not in {"fail", "skip"}:
            raise ValueError("on_exhausted 必须为 fail 或 skip。")

        body = set(starts)
        for node in graph.order:
            if node not in batch_writers and any(is_link(value) and value["node"] in body
                                         for value in graph.nodes[node]["inputs"].values()):
                body.add(node)
        self.body = body
        reject_early_writers(graph, body, batch_writers)
        self.fixed = set(graph.nodes) - body - batch_writers
        if any(graph.nodes[node]["class_type"] == "FolderImagesNode" for node in body):
            raise ValueError("文件夹输入必须位于条件循环外，每张图片有独立的尝试。")
        for node, spec in graph.nodes.items():
            if node not in batch_writers and getattr(graph.registry[spec["class_type"]], "SIDE_EFFECTS", False):
                raise ValueError("条件循环范围只允许终点保存，不能有其他写盘节点。")
        # Reject bypasses even on branches not selected for this run.
        escape_nodes = graph.nodes if hasattr(graph, "batch_terminals") else definition["nodes"]
        for node, spec in escape_nodes.items():
            if node in body or node in batch_writers:
                continue
            if any(is_link(value) and value["node"] in body for value in spec.get("inputs", {}).values()):
                raise ValueError(f"循环输出不能绕过保存终点逃逸到节点 {node}。")
        if any(is_link(value) and value["node"] in body for value in definition.get("outputs", {}).values()):
            raise ValueError("循环体输出不能直接暴露为工作流输出，请通过保存终点。")
        if target_id not in body and not any(is_link(value) and value["node"] in body for value in graph.nodes[target_id]["inputs"].values()):
            raise ValueError("保存终点必须连接条件循环中的输出。")

        self.conditions = deepcopy(settings["conditions"])
        for condition in self.conditions:
            node, output = condition.get("node"), condition.get("output")
            if node not in body or not isinstance(output, str):
                raise ValueError("所有评分条件必须来自本轮循环体中的输出。")
            kind = graph.output_types[node].get(output)
            if kind not in SCALARS:
                raise ValueError(f"条件 {node}.{output} 必须是数字或布尔输出。")
            op, value = condition.get("operator"), condition.get("value")
            if not isinstance(op, str) or op not in OPERATORS:
                raise ValueError("条件比较运算符无效。")
            if kind in {"BOOL", "BOOLEAN"}:
                if type(value) is not bool or op not in {"==", "!="}:
                    raise ValueError("布尔条件仅支持 == / !=，比较值必须是布尔值。")
            elif type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("数字条件的比较值必须是有限数字，不能是布尔值。")
        self.seeds = {}
        for node in settings.get("seed_nodes", []):
            if node not in body or graph.input_types[node].get("seed") != "INT":
                raise ValueError("种子节点必须在循环体内并包含 INT seed 输入。")
            seed = graph.nodes[node]["inputs"]["seed"]
            if type(seed) is not int or not 0 <= seed <= 2**63 - self.max_iterations:
                raise ValueError("循环种子必须直接填写非负整数，且逐轮递增不能超过 2**63-1。")
            self.seeds[node] = seed

    def describe(self):
        return {"target_id": self.target_id, "body_nodes": [n for n in self.graph.order if n in self.body],
                "fixed_nodes": [n for n in self.graph.order if n in self.fixed],
                "max_iterations": self.max_iterations, "mode": self.mode,
                "timeout_seconds": self.timeout, "on_exhausted": self.on_exhausted,
                "seed_nodes": list(self.seeds), "conditions": self.conditions}


class LoopDeadline(GraphExecutionCancelled):
    pass


def build_conditional_plan(graph, target_id, settings, definition):
    """Keep the flat plan and its execution semantics entirely unchanged."""
    if "loops" in settings:
        return MultiConditionalPlan(graph, target_id, settings, definition)
    return ConditionalPlan(graph, target_id, settings, definition)


def _closure(nodes, edges):
    seen = set(nodes)
    pending = list(seen)
    while pending:
        for node in edges[pending.pop()]:
            if node not in seen:
                seen.add(node)
                pending.append(node)
    return seen


def preview_settings(definition, settings, loop_id):
    """Select required regions before validating/building any unrelated graph."""
    return required_preview_settings(definition, settings, loop_id=loop_id)


def required_preview_settings(definition, settings, *, loop_id=None, node_id=None):
    """Discover region ownership from complete bodies, not just start ancestry.

    Only selected regions are validated. Incomplete unrelated cards may coexist
    with a runnable slice; an incomplete card whose start is needed must fail.
    """
    items = _multiple_settings(settings)
    if node_id is None and (not isinstance(loop_id, str) or not any(item["id"] == loop_id for item in items)):
        raise ValueError("未知循环 ID。")

    def ancestry(roots, *, strict=True):
        needed, pending = set(), list(roots)
        while pending:
            node = pending.pop()
            if not isinstance(node, str) or node in needed:
                continue
            needed.add(node)
            spec = definition["nodes"].get(node)
            if not isinstance(spec, dict) or not isinstance(spec.get("inputs"), dict):
                if strict:
                    raise ValueError(f"Node {node}: missing node or invalid inputs.")
                continue
            pending.extend(v["node"] for v in spec["inputs"].values() if is_link(v))
        return needed

    bodies = {}
    for item in items:
        starts = item.get("start_nodes", [])
        starts = {n for n in starts if isinstance(n, str)} if isinstance(starts, list) else set()
        ends = item.get("end_nodes", [])
        ends = ends if isinstance(ends, list) else []
        conditions = item.get("conditions", [])
        conditions = conditions if isinstance(conditions, list) else []
        roots = ends + [c.get("node") for c in conditions if isinstance(c, dict)]
        relevant = ancestry(roots, strict=False)
        anchors = {n for n in roots if isinstance(n, str)}
        body = {n for n in relevant if ancestry([n], strict=False) & starts} if starts else relevant
        bodies[item["id"]] = starts | anchors | body
    required = {loop_id} if node_id is None else set()
    while True:
        selected = {"loops": [item for item in items if item["id"] in required]}
        roots = condition_nodes(selected) if required else []
        needed = ancestry(roots + ([node_id] if node_id is not None else []))
        upstream = {key for key, body in bodies.items() if body & needed}
        if upstream <= required:
            return deepcopy(selected) if required else None
        required.update(upstream)


class MultiConditionalPlan:
    """Disjoint, explicitly bounded retry regions in a single acyclic graph."""

    def __init__(self, graph, target_id, settings, definition, *, preview_loop_id=None, preview_node_id=None):
        items = _multiple_settings(settings)
        condition_nodes(settings)
        self.preview_loop_id = preview_loop_id
        self.preview_node_id = preview_node_id
        computation_only = preview_loop_id is not None or preview_node_id is not None
        self.settings = deepcopy(settings)
        if computation_only and target_id is not None:
            raise ValueError("循环预览不能指定保存终点。")
        if preview_loop_id is not None:
            by_id = {item["id"]: item for item in items}
            if not isinstance(preview_loop_id, str) or preview_loop_id not in by_id:
                raise ValueError("未知循环 ID。")
        self.graph, self.target_id = graph, target_id
        batch_writers = getattr(graph, "batch_write_nodes", {target_id})
        if not computation_only and graph.nodes[target_id]["class_type"] != "SaveImagesNode" and not hasattr(graph, "batch_terminals"):
            raise ValueError("条件循环必须以 SaveImagesNode 为保存终点。")
        # Only the selected execution slice may run side effects. The full
        # definition is still used below to protect workflow outputs.
        for node, spec in graph.nodes.items():
            cls = graph.registry.get(spec["class_type"])
            if node not in batch_writers and getattr(cls, "SIDE_EFFECTS", False):
                raise ValueError("执行图只允许一个所选保存终点，不能有其他写盘节点。")
        parents = {n: {v["node"] for v in spec["inputs"].values() if is_link(v)}
                   for n, spec in graph.nodes.items()}
        children = {n: set() for n in graph.nodes}
        for node, sources in parents.items():
            for source in sources:
                children[source].add(node)
        self.ancestors = {n: _closure([n], parents) for n in graph.nodes}
        targets = getattr(graph, "batch_terminals", [target_id])
        save_ancestors = set().union(*(self.ancestors[node] for node in targets)) if not computation_only else set(graph.nodes)
        self.loops = []
        self.body = set()
        for item in items:
            item = deepcopy(item)
            for key in ("start_nodes", "end_nodes", "seed_nodes"):
                values = item.get(key, [])
                if (not isinstance(values, list) or any(not isinstance(n, str) or not n for n in values)
                        or len(set(values)) != len(values)):
                    raise ValueError(f"{key} 必须是无重复的节点 ID 列表。")
                if any(n not in graph.nodes or (n == target_id and not hasattr(graph, "batch_terminals")) for n in values):
                    raise ValueError(f"{key} 必须指定有效的普通计算节点。")
            starts, ends = item.get("start_nodes", []), item["end_nodes"]
            if not starts:
                raise ValueError("请选择有效的循环开始节点。")
            conditions = condition_nodes(item)
            if any(n not in graph.nodes for n in conditions):
                raise ValueError("所有评分条件必须来自本轮循环体中的输出。")
            descendants = _closure(starts, children)
            relevant = set().union(*(self.ancestors[n] for n in ends + conditions))
            body = descendants & relevant
            if any(not any(s in self.ancestors[e] for e in ends) for s in starts):
                raise ValueError("每个 start 必须能够到达某个 end。")
            if not set(ends + conditions) <= body:
                raise ValueError("所有 end 和评分条件必须在本轮循环体内。")
            if body & self.body:
                raise ValueError("多个循环体不能重叠或嵌套。")
            if not body & save_ancestors:
                raise ValueError("每个循环必须通过数据路径关联所选保存终点，不能是无关循环。")
            if any(graph.nodes[n]["class_type"] == "FolderImagesNode" for n in body):
                raise ValueError("文件夹输入必须位于条件循环外。")
            if any(getattr(graph.registry[graph.nodes[n]["class_type"]], "SIDE_EFFECTS", False) for n in body):
                raise ValueError("循环体不能包含副作用节点。")
            reject_early_writers(graph, body, batch_writers)
            # Ordinary consumers may use any accepted intermediate. Execution
            # retains the whole accepted body; end_nodes are acceptance roots,
            # not a whitelist of legal outgoing edges.
            exits = set(ends + conditions)
            exposed = body if not computation_only else body - exits
            if any(is_link(v) and v["node"] in exposed for v in definition.get("outputs", {}).values()):
                raise ValueError("循环体输出不能直接暴露为工作流输出，请通过保存终点。")
            mode = item.get("mode", "all")
            maximum = item.get("max_iterations", 10)
            timeout = item.get("timeout_seconds", 0)
            exhausted = item.get("on_exhausted", "fail")
            if mode not in {"all", "any"}:
                raise ValueError("条件 mode 必须为 all 或 any。")
            if type(maximum) is not int or not 1 <= maximum <= 1000:
                raise ValueError("max_iterations 必须是 1～1000 的整数。")
            if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 <= timeout <= 86400:
                raise ValueError("timeout_seconds 必须是 0～86400 的有限数字。")
            if exhausted not in {"fail", "skip"}:
                raise ValueError("on_exhausted 必须为 fail 或 skip。")
            for condition in item["conditions"]:
                node, output = condition.get("node"), condition.get("output")
                if not isinstance(output, str) or graph.output_types[node].get(output) not in SCALARS:
                    raise ValueError("条件必须使用数字或布尔输出。")
                kind = graph.output_types[node][output]
                op, value = condition.get("operator"), condition.get("value")
                if not isinstance(op, str) or op not in OPERATORS:
                    raise ValueError("条件比较运算符无效。")
                if kind in {"BOOL", "BOOLEAN"}:
                    if type(value) is not bool or op not in {"==", "!="}:
                        raise ValueError("布尔条件仅支持 == / !=，比较值必须是布尔值。")
                elif type(value) not in (int, float) or not math.isfinite(value):
                    raise ValueError("数字条件的比较值必须是有限数字，不能是布尔值。")
            seeds = {}
            for node in item.get("seed_nodes", []):
                if node not in body or graph.input_types[node].get("seed") != "INT":
                    raise ValueError("种子节点必须在循环体内并包含 INT seed 输入。")
                seed = graph.nodes[node]["inputs"]["seed"]
                if type(seed) is not int or not 0 <= seed <= 2**63 - maximum:
                    raise ValueError("循环种子必须直接填写非负整数，且逐轮递增不能超过 2**63-1。")
                seeds[node] = seed
            item.update(name=item.get("name", item["id"]), body=body, seeds=seeds,
                        mode=mode, max_iterations=maximum, timeout_seconds=timeout, on_exhausted=exhausted)
            self.loops.append(item)
            self.body.update(body)
        # Transitive ancestry includes ordinary fixed bridges between regions.
        dependencies = {item["id"]: {other["id"] for other in self.loops if other is not item
                                    and any(self.ancestors[n] & other["body"] for n in item["body"])}
                        for item in self.loops}
        try:
            order = tuple(TopologicalSorter(dependencies).static_order())
        except CycleError as error:
            raise ValueError("循环存在交叉未来依赖，不能按完整区域顺序执行。") from error
        by_id = {item["id"]: item for item in self.loops}
        self.loops = [by_id[key] for key in order]
        self.fixed = set(graph.nodes) - self.body - batch_writers
        self.max_iterations = sum(item["max_iterations"] for item in self.loops)

    def describe(self):
        loops = [{"id": item["id"], "name": item["name"], "start_nodes": list(item["start_nodes"]),
              "end_nodes": list(item["end_nodes"]), "body_nodes": [n for n in self.graph.order if n in item["body"]],
                  "seed_nodes": list(item["seeds"]), "conditions": deepcopy(item["conditions"]),
                  **{k: item[k] for k in ("mode", "max_iterations", "timeout_seconds", "on_exhausted")}}
                 for item in self.loops]
        return {"target_id": self.target_id,
                **({"preview_loop_id": self.preview_loop_id,
                    "required_loop_ids": [item["id"] for item in self.loops]}
                   if self.preview_loop_id is not None else {}),
                **({"preview_node_id": self.preview_node_id,
                    "required_loop_ids": [item["id"] for item in self.loops]}
                   if self.preview_node_id is not None else {}),
                "body_nodes": [n for n in self.graph.order if n in self.body],
                "fixed_nodes": [n for n in self.graph.order if n in self.fixed],
                "max_iterations": self.max_iterations, "mode": "all", "timeout_seconds": 0,
                "on_exhausted": "fail", "seed_nodes": [n for item in self.loops for n in item["seeds"]],
                "conditions": [deepcopy(c) for item in self.loops for c in item["conditions"]], "loops": loops}


class ConditionalExecution(GraphExecution):
    def __init__(self, graph, context, cancel, deadline, progress, target_id, fixed_nodes):
        super().__init__(graph, context, allow_side_effects=target_id is not None)
        self.cancel = cancel
        self.deadline = deadline
        self.progress = progress
        self.target_id = target_id
        self.fixed_nodes = fixed_nodes

    def checkpoint(self):
        if self.cancel.is_set():
            raise GraphExecutionCancelled()
        if self.deadline is not None and monotonic() >= self.deadline:
            raise LoopDeadline()

    def _execute(self, node_id):
        self.checkpoint()
        if node_id not in self.cache:
            self.progress({"current_node": node_id})
        result = super()._execute(node_id)
        # A completed save must be reported even if cancellation arrives while
        # writing. All earlier computations are checked again before saving.
        if not getattr(self.graph.registry[self.graph.nodes[node_id]["class_type"]], "SIDE_EFFECTS", False):
            self.checkpoint()
        # A downstream node may modify a PIL image in place. Never hand out
        # the retained fixed-cache object, including during the first attempt.
        return deepcopy(result) if node_id in self.fixed_nodes else result

    def _execute_checked(self, node_id):
        # Resolve dependencies first: an elapsed deadline must never enter a
        # side-effecting save after a long dependency finishes.
        if getattr(self.graph.registry[self.graph.nodes[node_id]["class_type"]], "SIDE_EFFECTS", False) and node_id not in self.cache:
            for binding in self.graph.nodes[node_id]["inputs"].values():
                self._resolve(binding)
            self.checkpoint()
        return super()._execute_checked(node_id)


def _loop_preview(loop_id, iteration, accepted, execution, body):
    return {"loop_id": loop_id, "iteration": iteration, "accepted": accepted,
            "outputs": {n: value for n, value in execution.cache.items() if n in body}}


def run_conditional(plan, context, cancel, progress=lambda _update: None, initial_cache=None):
    if isinstance(plan, MultiConditionalPlan):
        return _run_multiple(plan, context, cancel, progress, initial_cache)
    started = monotonic()
    deadline = started + plan.timeout if plan.timeout else None
    fixed_cache = {n: deepcopy(v) for n, v in (initial_cache or {}).items() if n in plan.fixed}
    history = []
    execution = None
    reason = "max_iterations"
    for iteration in range(1, plan.max_iterations + 1):
        # Per-run snapshots: never mutate a shared graph or carry generated
        # output into the next retry. Only explicitly fixed nodes survive.
        graph = deepcopy(plan.graph)
        seeds = {node: seed + iteration - 1 for node, seed in plan.seeds.items()}
        for node, seed in seeds.items():
            graph.nodes[node]["inputs"]["seed"] = seed
        execution = ConditionalExecution(graph, deepcopy(context), cancel, deadline, progress, plan.target_id, plan.fixed)
        execution.cache.update(deepcopy(fixed_cache))
        progress({"current_iteration": iteration, "max_iterations": plan.max_iterations,
                  "iteration_history": deepcopy(history), "stop_reason": None})
        try:
            scores = []
            # Evaluate every condition in this attempt; no cross-round mixing
            # and no short-circuit that leaves a selected branch unexecuted.
            for condition in plan.conditions:
                value = execution.get_node(condition["node"])[condition["output"]]
                expected = condition["value"]
                valid_type = type(value) is bool if type(expected) is bool else type(value) in (int, float)
                if not valid_type:
                    raise ValueError("评分输出的运行时类型与条件不符。")
                if type(value) is not bool and not math.isfinite(value):
                    raise ValueError("评分输出必须是有限数字。")
                scores.append({"node": condition["node"], "output": condition["output"],
                               "value": value, "passed": bool(OPERATORS[condition["operator"]](value, expected))})
            accepted = (all if plan.mode == "all" else any)(score["passed"] for score in scores)
            entry = {"iteration": iteration, "seeds": seeds, "scores": scores,
                     "accepted": accepted, "elapsed_seconds": round(monotonic() - started, 3)}
            history.append(entry)
            progress({"iteration_history": deepcopy(history),
                      "loop_preview": _loop_preview("flat", iteration, accepted, execution, plan.body)})
            if accepted:
                outputs = (finish_batch_outputs(execution, plan.target_id) if hasattr(plan.graph, "batch_terminals")
                           else {"paths": execution.get_node(plan.target_id)["paths"]})
                progress({"stop_reason": "accepted"})
                return {"accepted": True, "iterations": iteration, "history": history,
                        "stop_reason": "accepted", **outputs}
            fixed_cache = {node: deepcopy(result) for node, result in execution.cache.items() if node in plan.fixed}
        except LoopDeadline:
            progress({"loop_preview": _loop_preview("flat", iteration, False, execution, plan.body)})
            reason = "timeout"
            break
        except Exception:
            progress({"loop_preview": _loop_preview("flat", iteration, False, execution, plan.body)})
            raise
    progress({"stop_reason": reason})
    result = {"accepted": False, "iterations": iteration, "history": history,
              "stop_reason": reason, "paths": {}}
    if plan.on_exhausted == "fail":
        result["error"] = "条件循环超时，未保存图片。" if reason == "timeout" else "达到最大尝试次数仍未达标，未保存图片。"
    else:
        result["skipped"] = True
    return result


def _run_multiple(plan, context, cancel, progress, initial_cache):
    # Cache objects are never returned in reports or handed to consumers by
    # reference. This includes accepted images used by later mutating regions.
    retained = {n: deepcopy(v) for n, v in (initial_cache or {}).items()
                if n in plan.fixed and not plan.ancestors[n] & plan.body}
    accepted_body = set()
    history, results = [], []
    all_nodes = set(plan.graph.nodes)
    execution = None
    deadline = None

    def report(accepted, reason, item, iteration):
        result = {"accepted": accepted, "iterations": len(history), "history": deepcopy(history),
                  "stop_reason": reason, "paths": {}, "loops": deepcopy(results)}
        if plan.preview_loop_id is not None:
            result.update(preview_loop_id=plan.preview_loop_id,
                          required_loop_ids=[loop["id"] for loop in plan.loops], preview={})
            if item["id"] == plan.preview_loop_id and execution is not None:
                nodes = dict.fromkeys(item["end_nodes"] + condition_nodes(item))
                # A score is often the end node. Still show computed images
                # from this region, without rerunning generators for display.
                nodes.update({n: None for n in plan.graph.order if n in item["body"]
                              and any(kind in {"RGB_IMAGE", "IMAGE", "MASK"}
                                      for kind in plan.graph.output_types[n].values())})
                result["preview"] = {n: execution.cache[n] for n in nodes if n in execution.cache}
        if plan.preview_node_id is not None:
            result.update(preview_node_id=plan.preview_node_id,
                          required_loop_ids=[loop["id"] for loop in plan.loops], preview={})
            if accepted and execution is not None and plan.preview_node_id in execution.cache:
                result["preview"] = {plan.preview_node_id: execution.cache[plan.preview_node_id]}
        if not accepted:
            failed = {"id": item["id"], "name": item["name"], "accepted": False,
                      "iterations": iteration, "stop_reason": reason,
                      "history": [deepcopy(e) for e in history if e["loop_id"] == item["id"]]}
            if results and results[-1]["id"] == item["id"]:
                result["loops"][-1] = failed
            else:
                result["loops"].append(failed)
            # Include interrupted attempts even when no scalar score completed.
            result["iterations"] = sum(e["iterations"] for e in result["loops"])
            if item["on_exhausted"] == "skip":
                result["skipped"] = True
            else:
                result["error"] = "条件循环超时，未保存图片。" if reason == "timeout" else "达到最大尝试次数仍未达标，未保存图片。"
        progress({"stop_reason": reason})
        return result

    for index, item in enumerate(plan.loops, 1):
        started = monotonic()
        deadline = started + item["timeout_seconds"] if item["timeout_seconds"] else None
        progress({"current_loop_id": item["id"], "current_loop_name": item["name"],
                  "current_loop_index": index, "current_loop_count": len(plan.loops)})
        for iteration in range(1, item["max_iterations"] + 1):
            graph = deepcopy(plan.graph)
            seeds = {n: seed + iteration - 1 for n, seed in item["seeds"].items()}
            for node, seed in seeds.items():
                graph.nodes[node]["inputs"]["seed"] = seed
            execution = ConditionalExecution(graph, deepcopy(context), cancel, deadline, progress,
                                             plan.target_id, all_nodes)
            execution.cache.update(deepcopy(retained))
            progress({"current_iteration": iteration, "max_iterations": item["max_iterations"],
                      "iteration_history": deepcopy(history), "stop_reason": None})
            try:
                execution.checkpoint()
                scores = []
                for condition in item["conditions"]:
                    value = execution.get_node(condition["node"])[condition["output"]]
                    expected = condition["value"]
                    valid = type(value) is bool if type(expected) is bool else type(value) in (int, float)
                    if not valid:
                        raise ValueError("评分输出的运行时类型与条件不符。")
                    if type(value) is not bool and not math.isfinite(value):
                        raise ValueError("评分输出必须是有限数字。")
                    scores.append({"node": condition["node"], "output": condition["output"], "value": value,
                                   "passed": bool(OPERATORS[condition["operator"]](value, expected))})
                accepted = (all if item["mode"] == "all" else any)(s["passed"] for s in scores)
                if accepted or (item["id"] == plan.preview_loop_id
                                and iteration == item["max_iterations"]):
                    for node in item["end_nodes"]:
                        execution.get_node(node)
                    execution.checkpoint()
                history.append({"loop_id": item["id"], "loop_name": item["name"], "iteration": iteration,
                                "seeds": seeds, "scores": scores, "accepted": accepted,
                                "elapsed_seconds": round(monotonic() - started, 3)})
                progress({"iteration_history": deepcopy(history),
                          "loop_preview": _loop_preview(item["id"], iteration, accepted, execution, item["body"])})
                if accepted:
                    accepted_body.update(item["body"])
                # Do not freeze bridge nodes depending on failed/unaccepted bodies.
                retained = {n: deepcopy(v) for n, v in execution.cache.items()
                            if n not in getattr(plan.graph, "batch_write_nodes", {plan.target_id}) and (n in accepted_body or n in plan.fixed)
                            and not (plan.ancestors[n] & plan.body) - accepted_body}
                if accepted:
                    results.append({"id": item["id"], "name": item["name"], "accepted": True,
                                    "iterations": iteration, "stop_reason": "accepted",
                                    "history": [deepcopy(e) for e in history if e["loop_id"] == item["id"]]})
                    break
            except LoopDeadline:
                progress({"loop_preview": _loop_preview(item["id"], iteration, False, execution, item["body"])})
                return report(False, "timeout", item, iteration)
            except Exception:
                progress({"loop_preview": _loop_preview(item["id"], iteration, False, execution, item["body"])})
                raise
        else:
            return report(False, "max_iterations", item, iteration)
    if plan.preview_loop_id is not None:
        return report(True, "accepted", item, iteration)
    # Save only after every region and every explicit end has been accepted.
    # The last region's deadline also covers late save-only dependencies.
    execution = ConditionalExecution(deepcopy(plan.graph), deepcopy(context), cancel, deadline,
                                     progress, plan.target_id, all_nodes)
    execution.cache.update(deepcopy(retained))
    try:
        if plan.preview_node_id is not None:
            execution.get_node(plan.preview_node_id)
            return report(True, "accepted", item, iteration)
        outputs = (finish_batch_outputs(execution, plan.target_id) if hasattr(plan.graph, "batch_terminals")
                   else {"paths": execution.get_node(plan.target_id)["paths"]})
    except LoopDeadline:
        return report(False, "timeout", item, iteration)
    result = report(True, "accepted", item, iteration)
    result.update(outputs)
    return result

