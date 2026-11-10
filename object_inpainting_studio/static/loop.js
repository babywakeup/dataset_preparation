const loopUI = { sourceId: null, targetId: null, enabled: false, plan: null, planKey: null, error: "", timer: null, busy: false, checking: false };
const conditionalUI = { targetId: null, controlsKey: null, plan: null, planKey: null, error: "", checking: false, picking: null };
const conditionalEnabled = () => state.workflow?.conditional_loop != null;
const conditionalOperators = ["<=", ">=", "<", ">", "==", "!="];
const conditionalScalarTypes = ["FLOAT", "INT", "BOOLEAN", "BOOL"];

function loopSettings() {
  return { sourceId: loopUI.sourceId, targetId: loopUI.targetId, enabled: loopUI.enabled, conditionalTargetId: conditionalUI.targetId };
}

function restoreLoop(settings = {}) {
  clearTimeout(loopUI.timer);
  Object.assign(loopUI, { sourceId: null, targetId: null, enabled: false, plan: null, planKey: null, error: "", checking: false });
  Object.assign(conditionalUI, { targetId: settings?.conditionalTargetId ?? null, controlsKey: null, plan: null, planKey: null, error: "", checking: false, picking: null });
  if (state.workflow.nodes[settings?.sourceId]?.class_type === "FolderImagesNode") {
    loopUI.sourceId = settings.sourceId;
    loopUI.enabled = settings.enabled === true;
    if (state.workflow.nodes[settings.targetId]?.class_type === "SaveImagesNode" || (conditionalEnabled() && typeof settings.targetId === "string")) loopUI.targetId = settings.targetId;
  }
  resetMultipleLoops();
}

function upstreamNodes(targetId) {
  const visited = new Set(), order = [];
  const visit = id => {
    if (visited.has(id) || !state.workflow.nodes[id]) return;
    visited.add(id);
    Object.values(state.workflow.nodes[id].inputs || {}).filter(isLink).forEach(binding => visit(binding.node));
    order.push(id);
  };
  visit(targetId);
  return order;
}

function batchUpstreamNodes(targetId) {
  if (!state.workflow.nodes[targetId]) return [];
  // Batch execution includes condition/end branches, even when their outputs
  // are not wired into the save node. Match the server's condition_nodes roots.
  const settings = state.workflow.conditional_loop;
  const loops = Array.isArray(settings?.loops) ? settings.loops : settings ? [settings] : [];
  const roots = [];
  loops.forEach(item => {
    if (Array.isArray(item?.conditions)) roots.push(...item.conditions.map(condition => condition?.node));
    if (Array.isArray(settings?.loops) && Array.isArray(item?.end_nodes)) roots.push(...item.end_nodes);
  });
  roots.push(targetId);
  return [...new Set(roots.flatMap(id => upstreamNodes(id)))];
}

function loopScope() {
  const nodes = state.workflow.nodes;
  if (nodes[loopUI.sourceId]?.class_type !== "FolderImagesNode") {
    loopUI.sourceId = Object.keys(nodes).find(id => nodes[id].class_type === "FolderImagesNode") || null;
    loopUI.enabled = false;
  }
  const sourceId = loopUI.sourceId;
  const targets = Object.keys(nodes).filter(id => nodes[id].class_type === "SaveImagesNode" && batchUpstreamNodes(id).includes(sourceId));
  if (!targets.includes(loopUI.targetId) && !(conditionalEnabled() && loopUI.targetId)) loopUI.targetId = targets.length === 1 ? targets[0] : null;
  const order = batchUpstreamNodes(loopUI.targetId);
  const repeated = new Set(sourceId ? [sourceId] : []);
  order.forEach(id => {
    if (Object.values(nodes[id].inputs).some(value => isLink(value) && repeated.has(value.node))) repeated.add(id);
  });
  const folders = order.filter(id => nodes[id].class_type === "FolderImagesNode");
  let problem = "";
  if (!sourceId) problem = "先添加「文件夹图片」节点，再连接处理流程和保存节点。";
  else if (!loopUI.enabled) problem = "当前为单张调试。点击文件夹上的「↻ 逐张」开启遍历。";
  else if (!targets.length) problem = "请把文件夹输出连接到处理节点，再连接到「保存图片」节点。";
  else if (!loopUI.targetId) problem = "存在多个保存分支，请选择本次执行的保存终点。";
  else if (!targets.includes(loopUI.targetId)) problem = "条件任务的保存终点已失效或断开；请明确重新选择，不会自动切换保存分支。";
  else if (folders.length !== 1) problem = "此保存终点依赖多个文件夹，无法确定逐张对应关系；请只保留一个文件夹输入。";
  return { sourceId, targets, order, repeated, ancestors: new Set(order), problem };
}

function chooseLoop(sourceId, enabled, targetId = null) {
  if (nodeDeletionDisabled()) return;
  loopUI.sourceId = sourceId; loopUI.enabled = enabled; loopUI.targetId = targetId;
  state.expanded = {};
  render();
}

function folderLoopSwitch(id) {
  const modes = document.createElement("div");
  modes.className = "folder-loop-switch";
  modes.setAttribute("role", "group"); modes.setAttribute("aria-label", "文件夹运行模式");
  const active = loopUI.sourceId === id && loopUI.enabled;
  [[false, "单张"], [true, "↻ 逐张"]].forEach(([enabled, text]) => {
    const button = Object.assign(document.createElement("button"), { textContent: text, disabled: loopUI.busy });
    button.dataset.loopMode = enabled ? "each" : "single";
    button.setAttribute("aria-pressed", String(enabled === active));
    button.title = enabled ? "对文件夹中的每张图片执行通往保存终点的流程" : "仅测试 index 指定的当前图片";
    button.onclick = () => chooseLoop(id, enabled, loopUI.sourceId === id ? loopUI.targetId : null);
    modes.append(button);
  });
  return modes;
}

function loopPayload() { return { ...workflowPayload(), node_id: loopUI.targetId }; }

function updateLoopPlan(scope) {
  const conditionalProblem = conditionalEnabled() ? conditionalValidation(loopUI.targetId) : "";
  const key = scope.problem || conditionalProblem ? null : JSON.stringify(loopPayload());
  if (key === loopUI.planKey) return;
  clearTimeout(loopUI.timer);
  loopUI.planKey = key; loopUI.plan = null; loopUI.checking = false; loopUI.error = conditionalProblem;
  if (!key) return;
  loopUI.timer = setTimeout(async () => {
    loopUI.checking = true; renderBatch();
    try {
      const plan = await api("/api/batch/plan", { method: "POST", body: key });
      if (key !== loopUI.planKey) return;
      loopUI.plan = plan;
    } catch (error) {
      if (key !== loopUI.planKey) return;
      loopUI.error = error.message;
    } finally {
      if (key === loopUI.planKey) { loopUI.checking = false; renderBatch(); }
    }
  }, 250);
}

function renderLoop() {
  if (!state.workflow) return;
  const scope = loopScope();
  updateLoopPlan(scope);
  const enabled = loopUI.enabled;
  document.querySelector("#loop-mode").textContent = enabled ? "↻ 逐张模式" : "单张调试";
  document.querySelector("#loop-toolbar").classList.toggle("enabled", enabled);
  const target = document.querySelector("#loop-target");
  const staleTarget = loopUI.targetId && !scope.targets.includes(loopUI.targetId) ? loopUI.targetId : null;
  const targetsKey = JSON.stringify([scope.targets, staleTarget]);
  if (target.dataset.optionsKey !== targetsKey) {
    target.dataset.optionsKey = targetsKey;
    target.replaceChildren(new Option(scope.targets.length ? "选择保存终点…" : "尚未连接保存节点", ""));
    scope.targets.forEach(id => target.add(new Option(id, id)));
    if (staleTarget) target.add(new Option(`失效配置：${staleTarget}`, staleTarget));
  }
  target.value = loopUI.targetId || ""; target.disabled = !enabled || nodeDeletionDisabled();
  const plan = loopUI.plan;
  document.querySelector("#loop-summary").textContent = plan ? `↻ ${plan.total} 张输入 × ${conditionalEnabled() ? "每张条件重试，通过后保存" : "每张执行一次"} · 从第 ${plan.start_index + 1} 张开始（index=${plan.start_index}）` : scope.problem || (loopUI.error ? "运行前检查未通过" : "正在检查文件数量和保存目录…");
  const message = document.querySelector("#loop-message");
  message.textContent = loopUI.error || (plan ? `当前文件：${plan.current_path}\n保存到：${plan.output_dir} · 试跑也会写入磁盘` : "");
  message.classList.toggle("error", Boolean(loopUI.error));
  const steps = document.querySelector("#loop-steps"); steps.replaceChildren();
  if (enabled && loopUI.targetId) scope.order.filter(id => scope.repeated.has(id)).forEach((id, index) => {
    const step = Object.assign(document.createElement("button"), { textContent: `${index + 1} · ${state.catalog[state.workflow.nodes[id].class_type].help.title}` });
    step.title = id; step.onclick = () => focusLoopNode(id);
    const item = document.createElement("li"); item.append(step); steps.append(item);
  });
  const disabled = !enabled || !plan || Boolean(scope.problem) || batchStartDisabled() || loopUI.busy;
  document.querySelector("#loop-trial").disabled = disabled; document.querySelector("#loop-run").disabled = disabled;
  document.querySelector("#loop-focus").disabled = !scope.sourceId;
  document.querySelector("#loop-arrange").disabled = !loopUI.targetId;
  document.querySelector("#loop-check").disabled = !enabled || Boolean(scope.problem) || nodeDeletionDisabled();
  document.querySelectorAll(".folder-loop-switch button").forEach(button => { button.disabled = nodeDeletionDisabled(); });
  document.querySelectorAll(".node").forEach(card => {
    const id = card.dataset.nodeId;
    if (!state.workflow.nodes[id]) return;
    card.classList.toggle("loop-member", enabled && scope.repeated.has(id));
    card.classList.toggle("loop-shared", enabled && scope.ancestors.has(id) && !scope.repeated.has(id));
    card.classList.toggle("loop-outside", enabled && Boolean(loopUI.targetId) && !scope.ancestors.has(id));
    card.classList.toggle("loop-current", enabled && batchActive() && !batch.uncertain && batch.job?.source_id === scope.sourceId && batch.job?.target_id === loopUI.targetId && batch.job.current_node === id);
    card.querySelector(".loop-role").textContent = !enabled ? "" : scope.repeated.has(id) ? id === scope.sourceId ? "↻ 循环输入" : id === loopUI.targetId ? "保存终点" : "每张执行" : scope.ancestors.has(id) ? "固定输入 · 每轮读取" : "不执行";
    const path = card.querySelector(".node-path");
    if (path) {
      path.hidden = !enabled;
      const inputs = state.workflow.nodes[id].inputs;
      path.textContent = `目录：${typeof inputs.dir === "string" && inputs.dir ? inputs.dir : "请展开参数填写 dir"}${id === scope.sourceId ? `\nindex=${inputs.index ?? 0} · ${inputs.recursive === 1 ? "递归子目录" : "仅当前目录"}` : ""}`;
    }
  });
  renderLoopRegion(scope); renderConditional();
}

function conditionalValidation(targetId) {
  if (!conditionalEnabled()) return "条件重试未启用。";
  normalizeImportedConditional();
  return validateMultiple(targetId === undefined ? conditionalUI.targetId : targetId);
}

function conditionalChanged() {
  conditionalUI.plan = null; conditionalUI.planKey = null; conditionalUI.error = "";
  invalidateResults(); // Keep successful previews and the preview session.
  renderBatch();
}
function conditionalNodeCaption(id) {
  const spec = state.workflow.nodes[id];
  return spec ? `${id} · ${state.catalog[spec.class_type]?.help?.title || spec.class_type} (${spec.class_type})` : `失效配置：${id}`;
}
function cancelConditionalPick() {
  if (!conditionalUI.picking) return;
  conditionalUI.picking = null;
  document.querySelector("#connection-hint").textContent = "画布点选已结束；点击输出端口可开始连线。";
  render();
}
function consumeConditionalPick(id, output = null) {
  if (!conditionalUI.picking) return false;
  return consumeMultiplePick(id, output);
}
function conditionalAction(caption, action) {
  const button = Object.assign(document.createElement("button"), { textContent: caption, type: "button" });
  button.onclick = action; return button;
}
function conditionalSelect(options, value, label) {
  const select = document.createElement("select"); select.setAttribute("aria-label", label);
  options.forEach(([id, caption]) => select.add(new Option(caption, id)));
  if (!options.some(([id]) => id === value) && value != null && value !== "") select.add(new Option(`失效配置：${value}`, value));
  select.value = value ?? ""; return select;
}
function renderConditional() {
  renderMultipleEntry();
  if (conditionalEnabled()) { renderMulti(); return; }
  conditionalUI.picking = null; conditionalUI.plan = null; conditionalUI.planKey = null; conditionalUI.error = "";
  document.querySelector("#conditional-badge").textContent = "";
  document.querySelector("#conditional-error").textContent = "";
  document.querySelectorAll(".node").forEach(card => {
    ["conditional-member", "conditional-boundary", "conditional-pickable"].forEach(name => card.classList.remove(name));
    const badge = card.querySelector(".conditional-role"); if (badge) badge.textContent = "";
  });
  document.querySelectorAll(".port.output").forEach(port => {
    port.classList.remove("conditional-pick-port"); port.classList.remove("conditional-selected-port");
  });
}
function renderLoopRegion(scope = loopScope()) {
  const region = document.querySelector("#loop-region");
  const cards = [...document.querySelectorAll(".node")].filter(card => scope.repeated.has(card.dataset.nodeId));
  region.hidden = !loopUI.enabled || !loopUI.targetId || !cards.length;
  if (region.hidden) return;
  const left = Math.max(0, Math.min(...cards.map(card => card.offsetLeft)) - 18);
  const top = Math.max(0, Math.min(...cards.map(card => card.offsetTop)) - 42);
  const right = Math.max(...cards.map(card => card.offsetLeft + card.offsetWidth)) + 18;
  const bottom = Math.max(...cards.map(card => card.offsetTop + card.offsetHeight)) + 18;
  Object.assign(region.style, { left: `${left}px`, top: `${top}px`, width: `${right - left}px`, height: `${bottom - top}px` });
  region.querySelector("span").textContent = loopUI.plan ? ` ${loopUI.plan.total} 张 · ${scope.sourceId} → ${loopUI.targetId}` : ` ${scope.sourceId} → ${loopUI.targetId}`;
}
function focusLoopNode(id) {
  state.selected = id; render();
  requestAnimationFrame(() => document.querySelector(`.node[data-node-id="${CSS.escape(id)}"]`)?.scrollIntoView({ block: "center", inline: "center", behavior: "smooth" }));
}
function arrangeLoop() {
  const scope = loopScope(), depths = {}, rows = {};
  scope.order.forEach(id => {
    const parents = Object.values(state.workflow.nodes[id].inputs).filter(isLink).map(binding => depths[binding.node] ?? -1);
    depths[id] = Math.max(-1, ...parents) + 1;
  });
  const arrange = (ids, startY) => ids.forEach(id => {
    const row = depths[id], top = rows[row] ?? startY;
    state.layout[id] = { x: 50 + row * 360, y: top };
    rows[row] = top + document.querySelector(`.node[data-node-id="${CSS.escape(id)}"]`).offsetHeight + 70;
  });
  arrange(scope.order.filter(id => scope.repeated.has(id)), 60);
  const fixedTop = Math.max(60, ...Object.values(rows)) + 30;
  Object.keys(rows).forEach(key => delete rows[key]);
  arrange(scope.order.filter(id => !scope.repeated.has(id)), fixedTop);
  render(); focusLoopNode(scope.sourceId);
}
document.querySelector("#loop-target").onchange = event => { if (nodeDeletionDisabled()) return; loopUI.targetId = event.target.value || null; render(); };
document.querySelector("#loop-focus").onclick = () => { if (loopUI.sourceId) focusLoopNode(loopUI.sourceId); };
document.querySelector("#loop-arrange").onclick = arrangeLoop;
document.querySelector("#loop-check").onclick = () => { loopUI.planKey = null; renderLoop(); };
document.querySelector("#loop-trial").onclick = () => previewNode(loopUI.targetId, true);
document.querySelector("#loop-run").onclick = () => batchRequest("start", loopUI.targetId);
document.querySelector("#loop-example").onclick = () => {
  startPractice();
  state.workflow.nodes = {
    folder: { class_type: "FolderImagesNode", inputs: { dir: "", recursive: 0, index: 0 } },
    save: { class_type: "SaveImagesNode", inputs: { img1: link("folder", "image"), dir: "" } },
  };
  state.layout = { folder: { x: 50, y: 60 }, save: { x: 410, y: 60 } }; state.selected = "folder";
  chooseLoop("folder", true, "save");
  setStatus("逐张复制示例：填写文件夹和保存节点的 dir 后，先试跑当前张，再运行全部。可以在两者之间插入分割、抠图或生成节点；当前示例不运行模型。已有画布已替换。先下载可保留原流程。");
};
