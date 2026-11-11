"""Connected folder branches and their final outputs for one batch execution."""

from pathlib import Path

from object_inpainting.graph import GraphExecutionCancelled, is_link


def batch_options(definition):
    options = definition.get("batch_execution", {})
    if not isinstance(options, dict) or set(options) - {"mode", "result_nodes", "disabled_nodes"}:
        raise ValueError("batch_execution 必须是包含 mode / result_nodes / disabled_nodes 的对象。")
    mode = options.get("mode", "all_terminals")
    if mode not in {"all_terminals", "selected_results"}:
        raise ValueError("逐张结果模式必须是 all_terminals 或 selected_results。")
    lists = []
    for field in ("result_nodes", "disabled_nodes"):
        values = options.get(field, [])
        if (not isinstance(values, list) or any(not isinstance(node, str) or node not in definition["nodes"] for node in values)
                or len(set(values)) != len(values)):
            raise ValueError(f"{field} 必须是不重复的有效节点 ID 列表。")
        lists.append(values)
    return mode, *lists


def upstream_ids(definition, roots):
    seen, pending = set(), list(roots)
    while pending:
        node = pending.pop()
        if node in seen:
            continue
        if node not in definition["nodes"]:
            raise ValueError(f"Unknown node: {node}.")
        seen.add(node)
        pending.extend(value["node"] for value in definition["nodes"][node].get("inputs", {}).values() if is_link(value))
    return seen


def select_batch_results(definition, source_id):
    mode, selected, disabled = batch_options(definition)
    if source_id in disabled:
        raise ValueError("逐张输入文件夹不能被禁用。")
    descendants = folder_descendants(definition, source_id)
    blocked = set()
    for node in disabled:
        blocked.update(folder_descendants(definition, node))
    active = descendants - blocked
    if mode == "selected_results":
        if not selected:
            raise ValueError("指定结果模式请至少选择一个结果节点。")
        if any(node not in descendants for node in selected):
            raise ValueError("结果节点必须连接所选文件夹。")
        if set(selected) & blocked:
            raise ValueError("指定结果依赖已禁用节点，请启用该分支或取消结果选择。")
        targets = list(selected)
        active = upstream_ids(definition, targets)
    else:
        consumed = {value["node"] for node in active for value in definition["nodes"][node].get("inputs", {}).values()
                    if is_link(value)}
        targets = [node for node in definition["nodes"] if node in active - consumed and node != source_id]
    if not targets:
        raise ValueError("没有可执行的结果节点，请启用分支或选择结果。")
    return targets, active, blocked, mode


def describe_batch(graph, source_id, target_id, conditional=None):
    repeated = {source_id}
    for node in graph.order:
        if any(is_link(value) and value["node"] in repeated for value in graph.nodes[node]["inputs"].values()):
            repeated.add(node)
    retry = conditional.body if conditional is not None else set()
    roles = {node: "source" if node == source_id else "retry" if node in retry else
             "result" if node in graph.batch_terminals else "per_item" if node in repeated else "fixed"
             for node in graph.order}
    return {
        "source_id": source_id, "target_id": target_id,
        "executed_nodes": list(graph.order), "terminal_nodes": graph.batch_terminals,
        "repeated_nodes": [node for node in graph.order if node in repeated], "node_roles": roles,
        "result_mode": graph.batch_result_mode,
        "disabled_nodes": sorted(graph.batch_disabled_nodes),
        "write_nodes": [node for node in graph.order if node in graph.batch_write_nodes],
        "output_dirs": {node: spec["inputs"]["dir"] for node, spec in graph.nodes.items()
                        if spec["class_type"] == "SaveImagesNode"},
        "stages": [{"id": "input", "nodes": [source_id]},
                   {"id": "retry", "nodes": [node for node in graph.order if node in retry],
                    "regions": conditional.describe() if conditional is not None else None},
                   {"id": "results", "nodes": [node for node in graph.order if node not in retry and node != source_id]}],
    }


def folder_descendants(definition, source_id):
    nodes = definition["nodes"]
    descendants = {source_id}
    while True:
        expanded = descendants | {
            node for node, spec in nodes.items()
            if any(is_link(value) and value["node"] in descendants
                   for value in spec.get("inputs", {}).values())
        }
        if expanded == descendants:
            break
        descendants = expanded
    return descendants


def folder_terminals(definition, source_id):
    nodes = definition["nodes"]
    descendants = folder_descendants(definition, source_id)
    consumed = {value["node"] for node in descendants
                for value in nodes[node].get("inputs", {}).values() if is_link(value)}
    return [node for node in nodes if node in descendants and node not in consumed]


def finish_batch_outputs(execution, target_id):
    """Compute all remaining branches first, then execute their result writers."""
    graph = execution.graph
    terminals = getattr(graph, "batch_terminals", [target_id])
    writers = {node for node, spec in graph.nodes.items()
               if getattr(graph.registry[spec["class_type"]], "SIDE_EFFECTS", False)}
    for node in graph.order:
        if node not in writers:
            execution.get_node(node)
    previous = set(execution.cache)
    try:
        for node in graph.order:
            if node in writers:
                execution.get_node(node)
    except GraphExecutionCancelled:
        # Preserve already completed outputs on explicit cancellation, matching
        # existing single-writer batch behavior.
        raise
    except Exception:
        # SaveImages never overwrites files and returns paths only after a
        # successful write. These are exclusively files created by this input.
        for node in set(execution.cache) - previous:
            if node in writers and graph.nodes[node]["class_type"] == "SaveImagesNode":
                for path in execution.cache[node].get("paths", {}).values():
                    Path(path).unlink(missing_ok=True)
        raise
    return {
        "paths": execution.cache[target_id].get("paths", {}),
        "paths_by_node": {node: execution.cache[node]["paths"] for node in graph.order
                          if node in writers and "paths" in execution.cache[node]},
        "preview": {node: execution.cache[node] for node in terminals},
    }
