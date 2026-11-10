/**
 * 删除节点及相关连线；保留下游节点，其断开的输入恢复默认值或 null。
 * 仅编辑工作流，不删除磁盘文件、不执行节点。
 */
function removeWorkflowNode(workflow, nodeId, catalog) {
  if (!workflow?.nodes || !Object.hasOwn(workflow.nodes, nodeId)) return null;
  const linkedToNode = value => value && typeof value === "object" && Object.keys(value).length === 2 && value.node === nodeId && Object.hasOwn(value, "output");
  const disconnected = [];
  const removedOutputs = [];
  delete workflow.nodes[nodeId];
  Object.entries(workflow.nodes).forEach(([id, spec]) => {
    const defaults = catalog[spec.class_type]?.input_defaults || {};
    Object.entries(spec.inputs).forEach(([port, binding]) => {
      if (!linkedToNode(binding)) return;
      spec.inputs[port] = Object.hasOwn(defaults, port) ? structuredClone(defaults[port]) : null;
      disconnected.push({ node_id: id, port });
    });
  });
  Object.entries(workflow.outputs || {}).forEach(([name, binding]) => {
    if (!linkedToNode(binding)) return;
    delete workflow.outputs[name];
    removedOutputs.push(name);
  });
  return { disconnected, removedOutputs };
}