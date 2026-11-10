// Required production controller, loaded between loop.js and app.js.
const multipleUI = { nextId: 0, activeId: null };
const multipleColors = ["#a996ff", "#62d4bd", "#f4b76a", "#79baff", "#ef91bd", "#c2d875"];
const isMultipleLoops = () => Object.hasOwn(state.workflow?.conditional_loop || {}, "loops");
const multipleLoops = () => Array.isArray(state.workflow?.conditional_loop?.loops) ? state.workflow.conditional_loop.loops : [];
const multipleLoop = id => multipleLoops().find(item => item?.id === id);
const multipleValues = (item, field) => Array.isArray(item?.[field]) ? item[field] : [];
const multipleColor = id => {
  let hash = 0;
  for (const char of String(id)) hash = (hash * 31 + char.charCodeAt(0)) >>> 0;
  return multipleColors[hash % multipleColors.length];
};

function resetMultipleLoops() {
  multipleUI.activeId = null;
  normalizeImportedConditional();
  multipleLoops().forEach(item => {
    const number = /^loop_(\d+)$/.exec(item?.id)?.[1];
    if (number) multipleUI.nextId = Math.max(multipleUI.nextId, Number(number));
  });
  // nextId deliberately survives imports / deletion so IDs are never recycled.
}

function nextMultipleId() {
  let id;
  do { id = `loop_${++multipleUI.nextId}`; } while (multipleLoop(id));
  return id;
}

// Convert once by replacing the flat object, never mutating imported input objects.
// Explicit invalid values and missing references stay visible for correction.
function normalizeImportedConditional() {
  const old = state.workflow?.conditional_loop;
  if (old == null || typeof old !== "object" || Array.isArray(old) || Object.hasOwn(old, "loops")) return;
  const saves = Object.keys(state.workflow.nodes).filter(id => state.workflow.nodes[id].class_type === "SaveImagesNode");
  if (conditionalUI.targetId == null) {
    if (loopUI.enabled && loopUI.targetId != null) conditionalUI.targetId = loopUI.targetId;
    else if (saves.length === 1) conditionalUI.targetId = saves[0];
  }
  const item = structuredClone(old);
  const defaults = { start_nodes: [], seed_nodes: [], conditions: [], mode: "all", max_iterations: 10, timeout_seconds: 0, on_exhausted: "fail" };
  for (const [field, value] of Object.entries(defaults)) if (!Object.hasOwn(item, field)) item[field] = value;
  if (!Object.hasOwn(item, "end_nodes")) {
    // Legacy body = start descendants intersect save/condition ancestors.
    const body = multipleBody({ ...item, end_nodes: [conditionalUI.targetId] });
    const saved = Object.values(state.workflow.nodes[conditionalUI.targetId]?.inputs || {})
      .filter(v => isLink(v) && body.has(v.node)).map(v => v.node);
    const conditions = multipleValues(item, "conditions").map(c => c?.node).filter(id => id !== undefined);
    item.end_nodes = [...new Set([...saved, ...conditions])];
  }
  if (!Object.hasOwn(item, "id")) item.id = nextMultipleId();
  if (!Object.hasOwn(item, "name")) item.name = "导入的循环";
  state.workflow.conditional_loop = { loops: [item] };
  conditionalUI.controlsKey = null;
}

function multipleBody(config) {
  const relevant = new Set();
  const ends = Array.isArray(config.end_nodes) ? config.end_nodes : [];
  const conditions = Array.isArray(config.conditions) ? config.conditions : [];
  [...ends, ...conditions.map(c => c?.node)].forEach(id => upstreamNodes(id).forEach(n => relevant.add(n)));
  const descendants = new Set(Array.isArray(config.start_nodes) ? config.start_nodes : []);
  let changed = true;
  while (changed) {
    changed = false;
    Object.entries(state.workflow.nodes).forEach(([id, spec]) => {
      if (!descendants.has(id) && Object.values(spec.inputs || {}).some(v => isLink(v) && descendants.has(v.node))) {
        descendants.add(id); changed = true;
      }
    });
  }
  return new Set([...descendants].filter(id => relevant.has(id)));
}

// Read-only early feedback. The server plan is authoritative (including DAG ordering).
function validateMultiple(targetId = conditionalUI.targetId) {
  const config = state.workflow.conditional_loop;
  if (!config || typeof config !== "object" || Array.isArray(config)) return "conditional_loop 必须是配置对象；请明确删除无效配置后重新添加。";
  if (Object.keys(config).some(key => key !== "loops")) return "conditional_loop 不能混合 flat 字段和 loops。";
  if (!Array.isArray(config.loops) || config.loops.length < 1 || config.loops.length > 32) return "请配置 1–32 个循环。";
  const nodes = state.workflow.nodes;
  const ids = new Set(), occupied = new Set(), regions = [];
  for (const item of config.loops) {
    if (!item || typeof item !== "object" || typeof item.id !== "string" || !item.id.trim() || ids.has(item.id)) return "循环 ID 必须非空且唯一。";
    ids.add(item.id);
    const error = message => `${item.name || item.id}：${message}`;
    if (Object.hasOwn(item, "loops")) return error("不支持嵌套 loops 配置。");
    if (Object.hasOwn(item, "name") && (typeof item.name !== "string" || !item.name.trim())) return error("名称必须是非空文本。");
    for (const field of ["start_nodes", "end_nodes", "seed_nodes"]) {
      if (!Array.isArray(item[field]) || item[field].some(id => typeof id !== "string") || new Set(item[field]).size !== item[field].length) return error(`${field} 必须是无重复的节点 ID 列表。`);
      if (item[field].some(id => !nodes[id] || ["SaveImagesNode", "FolderImagesNode"].includes(nodes[id].class_type))) return error(`${field} 包含失效 / 非普通计算节点；请明确重新点选，不会自动替换。`);
    }
    if (!item.start_nodes.length || !item.end_nodes.length) return error("请依次点选起点 → 终点 → 条件。");
    if (!Array.isArray(item.conditions) || !item.conditions.length) return error("请添加至少一个数字 / 布尔条件。");
    if (!["all", "any"].includes(item.mode)) return error("条件组合必须是 all / any。");
    if (!Number.isInteger(item.max_iterations) || item.max_iterations < 1 || item.max_iterations > 1000) return error("最大尝试次数必须为 1–1000 的整数。");
    if (!Number.isFinite(item.timeout_seconds) || item.timeout_seconds < 0 || item.timeout_seconds > 86400) return error("超时必须为 0–86400 秒。");
    if (!["fail", "skip"].includes(item.on_exhausted)) return error("耗尽策略必须是 fail / skip。");
    for (const c of item.conditions) {
      if (!c || !nodes[c.node]) return error("条件节点失效；请明确重新点选。");
      const type = nodeInfo(c.node).outputs[c.output];
      if (!conditionalScalarTypes.includes(type)) return error(`条件 ${c.node}.${c.output} 必须选择数字 / 布尔输出，不能用图片或文本。`);
      if (!conditionalOperators.includes(c.operator)) return error("条件运算符无效。");
      if (["BOOL", "BOOLEAN"].includes(type)) {
        if (typeof c.value !== "boolean" || !["==", "!="].includes(c.operator)) return error("布尔条件仅支持 == / !=，比较值必须是 true / false。");
      } else if (typeof c.value !== "number" || !Number.isFinite(c.value)) return error("数字条件阈值必须是有限数字。");
    }
    if (nodes[targetId]?.class_type !== "SaveImagesNode") return "请选择公用 SaveImages 保存目标；失效目标不会自动替换。";
    const body = multipleBody(item);
    if (item.start_nodes.some(s => !item.end_nodes.some(e => upstreamNodes(e).includes(s)))) return error("每个起点必须能到达至少一个终点。");
    if ([...item.end_nodes, ...item.conditions.map(c => c.node)].some(id => !body.has(id))) return error("终点和条件必须来自本轮循环体。");
    if ([...body].some(id => occupied.has(id))) return error("循环体重叠 / 嵌套；请选择独立或串联的非重叠区域。");
    if ([...body].some(id => nodes[id].class_type === "FolderImagesNode" || nodeInfo(id).side_effects)) return error("循环体不能包含文件夹 / 保存等副作用节点。");
    for (const id of item.seed_nodes) {
      if (!body.has(id) || nodeInfo(id).inputs.seed !== "INT" || !Number.isSafeInteger(nodes[id].inputs.seed) || nodes[id].inputs.seed < 0) return error(`种子 ${id} 须在循环体内并直接填写非负 INT seed；-1 保持随机，不加入递增种子。`);
    }
    const ends = new Set(item.end_nodes), exits = new Set([...ends, ...item.conditions.map(c => c.node)]);
    for (const [id, spec] of Object.entries(nodes)) {
      if (body.has(id)) continue;
      const allowed = id === targetId ? body : exits;
      if (Object.values(spec.inputs || {}).some(v => isLink(v) && body.has(v.node) && !allowed.has(v.node))) return error(`循环中间输出被 ${id} 消费；请在高级设置中将来源显式加入 end_nodes。`);
    }
    if (Object.values(state.workflow.outputs || {}).some(v => isLink(v) && body.has(v.node))) return error("循环体输出不能直接暴露为工作流输出，请通过保存目标。");
    body.forEach(id => occupied.add(id)); regions.push({ item, body });
  }
  const saveAncestors = new Set(upstreamNodes(targetId));
  if (regions.some(r => ![...r.body].some(id => saveAncestors.has(id)))) return "每个循环必须通过数据路径关联最终保存目标。";
  const pending = new Set(regions);
  while (pending.size) {
    const ready = [...pending].find(r => ![...pending].some(other => other !== r && [...r.body].some(id => upstreamNodes(id).some(n => other.body.has(n)))));
    if (!ready) return "循环存在交叉依赖，无法按完整区域顺序执行。";
    pending.delete(ready);
  }
  return "";
}

function multipleChanged() {
  conditionalUI.controlsKey = null;
  clearTimeout(loopUI.timer);
  loopUI.plan = null; loopUI.planKey = null; loopUI.error = "";
  conditionalChanged();
  render();
}

function editMultiple(id, update) {
  if (nodeDeletionDisabled()) return;
  const item = multipleLoop(id);
  if (!item) return;
  conditionalUI.picking = null;
  update(item);
  multipleChanged();
}

function addMultipleLoop() {
  if (nodeDeletionDisabled() || !state.workflow) return;
  normalizeImportedConditional();
  if (conditionalEnabled() && !isMultipleLoops()) {
    setStatus("conditional_loop 配置无效，原值已保留；请明确删除无效配置后重新添加。", true); return;
  }
  if (isMultipleLoops() && (!Array.isArray(state.workflow.conditional_loop.loops) || multipleLoops().length >= 32)) {
    setStatus("最多支持 32 个循环；请先修正无效配置或删除循环。", true); return;
  }
  if (!isMultipleLoops()) state.workflow.conditional_loop = { loops: [] };
  const id = nextMultipleId();
  multipleLoops().push({ id, name: `循环 ${multipleUI.nextId}`, start_nodes: [], end_nodes: [], seed_nodes: [], conditions: [], mode: "all", max_iterations: 10, timeout_seconds: 0, on_exhausted: "fail" });
  multipleUI.activeId = id;
  document.querySelector("#conditional-panel").open = true;
  beginMultiplePick(id, "start_nodes", null, true);
  multipleChanged();
}

function deleteMultipleLoop(id) {
  if (nodeDeletionDisabled() || !multipleLoop(id)) return;
  if (conditionalUI.picking?.loopId === id) conditionalUI.picking = null;
  state.workflow.conditional_loop.loops = multipleLoops().filter(item => item.id !== id);
  if (multipleUI.activeId === id) multipleUI.activeId = null;
  if (!multipleLoops().length) delete state.workflow.conditional_loop;
  multipleChanged();
}

function beginMultiplePick(loopId, kind, index = null, wizard = false, many = false) {
  if (nodeDeletionDisabled() || !isMultipleLoops() || (loopId !== "$target" && !multipleLoop(loopId))) return;
  multipleUI.activeId = loopId;
  conditionalUI.picking = { loopId, kind, index, wizard, many };
  state.connecting = null;
  render();
}

function multiplePickHint(pick = conditionalUI.picking) {
  if (!pick) return "点击卡片按钮可重新选择具体实例；Esc 退出点选。";
  if (pick.kind === "target") return "选择公用保存目标：点击具体 SaveImages 节点标题。Esc 取消。";
  const name = multipleLoop(pick.loopId)?.name || pick.loopId;
  const role = { start_nodes: "① 起点", end_nodes: "② 终点", seed_nodes: "递增种子", condition: "③ 条件" }[pick.kind];
  return `${name} · ${role}：点击具体节点标题${pick.kind === "condition" ? "；单个数字 / 布尔输出直接选中，多个时点击具体输出圆点" : "（不拖动）"}。${pick.many ? "可连续增删；完成后按 Esc。" : "Esc 取消。"}`;
}

function consumeMultiplePick(id, output = null) {
  const pick = conditionalUI.picking;
  if (!pick?.loopId) return false;
  if (nodeDeletionDisabled() || !isMultipleLoops()) {
    conditionalUI.picking = null; state.connecting = null; return true;
  }
  const spec = state.workflow.nodes[id], item = multipleLoop(pick.loopId);
  if (!spec || (pick.kind !== "target" && !item)) { conditionalUI.picking = null; return true; }
  const reject = message => { setStatus(message, true); return true; };
  if (pick.kind === "target") {
    if (output != null || spec.class_type !== "SaveImagesNode") return reject("请点击具体 SaveImages 保存节点的标题。");
    conditionalUI.targetId = id;
    if (loopUI.enabled) loopUI.targetId = id;
    conditionalUI.picking = null;
  } else if (pick.kind === "condition") {
    const ports = Object.entries(nodeInfo(id).outputs).filter(([, type]) => conditionalScalarTypes.includes(type));
    if (output == null) {
      if (ports.length !== 1) return reject(ports.length ? "此节点有多个数字 / 布尔输出，请点击具体输出圆点。" : "图片 / 文本不能作为条件，请选择数字 / 布尔输出。");
      output = ports[0][0];
    }
    const type = nodeInfo(id).outputs[output];
    if (!conditionalScalarTypes.includes(type)) return reject("图片 / 文本不能作为条件，请选择数字 / 布尔输出圆点。");
    if (!Array.isArray(item.conditions)) item.conditions = [];
    const index = pick.index ?? item.conditions.length;
    const previous = item.conditions[index];
    const boolean = ["BOOL", "BOOLEAN"].includes(type);
    const sameKind = previous && typeof previous.value === (boolean ? "boolean" : "number");
    item.conditions[index] = { node: id, output, operator: sameKind ? previous.operator : boolean ? "==" : "<=", value: sameKind ? previous.value : boolean ? true : 10 };
    conditionalUI.picking = null;
  } else {
    if (output != null) return reject("请点击节点标题，不是输出端口。");
    if (["SaveImagesNode", "FolderImagesNode"].includes(spec.class_type) || nodeInfo(id).side_effects) return reject("起点 / 终点必须是普通计算节点，不是保存或文件夹节点。");
    if (pick.kind === "seed_nodes" && (nodeInfo(id).inputs.seed !== "INT" || !Number.isSafeInteger(spec.inputs.seed) || spec.inputs.seed < 0)) return reject("递增种子须直接填写非负 INT seed；-1 保持随机，无需选择种子。");
    const previous = multipleValues(item, pick.kind);
    item[pick.kind] = pick.many ? previous.includes(id) ? previous.filter(n => n !== id) : [...previous, id] : [id];
    if (pick.kind === "start_nodes") {
      item.seed_nodes = multipleValues(item, "seed_nodes");
      if (!pick.many) item.seed_nodes = item.seed_nodes.filter(n => !previous.includes(n));
      if (item.start_nodes.includes(id) && nodeInfo(id).inputs.seed === "INT" && Number.isSafeInteger(spec.inputs.seed) && spec.inputs.seed >= 0 && !item.seed_nodes.includes(id)) item.seed_nodes.push(id);
    }
    conditionalUI.picking = pick.many ? pick : pick.wizard ? { ...pick, kind: pick.kind === "start_nodes" ? "end_nodes" : "condition", index: 0 } : null;
  }
  state.selected = id;
  if (!conditionalUI.picking) document.querySelector("#connection-hint").textContent = "已指定节点实例；点击输出端口可开始连线。";
  multipleChanged();
  return true;
}

function multipleButton(text, handler, parent) {
  const button = conditionalAction(text, () => { if (!nodeDeletionDisabled()) handler(); });
  button.disabled = nodeDeletionDisabled();
  parent.append(button); return button;
}

function multipleField(parent, text, control) {
  const label = Object.assign(document.createElement("label"), { textContent: text });
  control.setAttribute("aria-label", text);
  label.append(control); parent.append(label); return control;
}

function renderMultipleEntry() {
  normalizeImportedConditional();
  const busy = nodeDeletionDisabled();
  const add = document.querySelector("#conditional-add");
  add.disabled = busy; // Remains visible even when the details panel is collapsed.
  const root = document.querySelector("#multiple-loops");
  root.hidden = false;
  if (!conditionalEnabled()) {
    root.replaceChildren();
    conditionalUI.controlsKey = null;
    root.append(Object.assign(document.createElement("p"), { className: "multiple-empty", textContent: "暂无循环。点击「＋ 添加循环」，然后在画布依次点选起点 → 终点 → 条件。无需先启用或选择种子。" }));
  }
}

function renderMultiFields() {
  const root = document.querySelector("#multiple-loops"); root.replaceChildren();
  if (!isMultipleLoops() || !Array.isArray(state.workflow.conditional_loop.loops) || multipleLoops().some(item => !item || typeof item !== "object" || Array.isArray(item))) {
    root.append(Object.assign(document.createElement("p"), { textContent: "配置无效，原始数据仍保留。删除后可重新添加循环。" }));
    multipleButton("删除无效配置", () => { delete state.workflow.conditional_loop; conditionalUI.picking = null; multipleChanged(); }, root);
    return;
  }
  const hint = Object.assign(document.createElement("p"), { className: "multiple-hint", textContent: multiplePickHint() });
  root.append(hint);
  if (conditionalUI.picking) multipleButton("取消点选 (Esc)", cancelConditionalPick, root);
  const target = Object.assign(document.createElement("section"), { className: "multiple-target" });
  target.append(Object.assign(document.createElement("strong"), { textContent: "最终保存目标 · 全部循环通过后仅保存一次" }));
  const saves = Object.entries(state.workflow.nodes).filter(([, spec]) => spec.class_type === "SaveImagesNode");
  const select = conditionalSelect([["", "请选择公用保存目标…"], ...saves.map(([id]) => [id, conditionalNodeCaption(id)])], conditionalUI.targetId, "公用最终保存目标");
  select.onchange = () => {
    if (nodeDeletionDisabled()) return;
    conditionalUI.targetId = select.value || null;
    if (loopUI.enabled) loopUI.targetId = conditionalUI.targetId;
    conditionalUI.picking = null; multipleChanged();
  };
  target.append(select);
  multipleButton("画布选择最终保存目标", () => beginMultiplePick("$target", "target"), target);
  root.append(target);
  multipleLoops().forEach((item, index) => {
    if (!item || typeof item !== "object") return;
    const id = item.id;
    const card = Object.assign(document.createElement("section"), { className: "multiple-loop-card" });
    card.dataset.loopId = id;
    card.style.setProperty("--conditional-color", multipleColor(id));
    card.append(Object.assign(document.createElement("strong"), { textContent: `↻ ${index + 1} · ${item.name || id}` }));
    const name = Object.assign(document.createElement("input"), { value: item.name ?? id });
    name.onchange = () => editMultiple(id, loop => { loop.name = name.value; });
    multipleField(card, "循环名称", name);
    multipleButton("删除循环", () => deleteMultipleLoop(id), card);
    for (const [field, title] of [["start_nodes", "起点"], ["end_nodes", "终点"]]) {
      const row = Object.assign(document.createElement("div"), { className: "multiple-role-row" });
      multipleButton(`重选${title}`, () => beginMultiplePick(id, field), row);
      (Array.isArray(item[field]) ? item[field] : []).forEach(node => multipleButton(`${title} · ${conditionalNodeCaption(node)}`, () => { if (state.workflow.nodes[node]) focusLoopNode(node); }, row));
      card.append(row);
    }
    (Array.isArray(item.conditions) ? item.conditions : []).forEach((c, i) => {
      const row = Object.assign(document.createElement("div"), { className: "multiple-condition" });
      multipleButton(`重选条件 ${i + 1}`, () => beginMultiplePick(id, "condition", i), row);
      row.append(Object.assign(document.createElement("span"), { textContent: `${conditionalNodeCaption(c?.node)} · ${c?.output || "未选输出"}` }));
      const operator = conditionalSelect(conditionalOperators.map(op => [op, op]), c?.operator, `条件 ${i + 1} 运算符`);
      operator.onchange = () => editMultiple(id, loop => { loop.conditions[i] = { ...loop.conditions[i], operator: operator.value }; });
      const value = Object.assign(document.createElement("input"), { value: c?.value == null ? "" : String(c.value) });
      value.onchange = () => editMultiple(id, loop => { loop.conditions[i] = { ...loop.conditions[i], value: parseLiteral(value.value) }; });
      multipleField(row, `条件 ${i + 1} 运算符`, operator);
      multipleField(row, `条件 ${i + 1} 比较值`, value);
      multipleButton(`删除条件 ${i + 1}`, () => editMultiple(id, loop => { loop.conditions.splice(i, 1); }), row);
      card.append(row);
    });
    multipleButton(item.conditions?.length ? "＋ 添加条件" : "选择条件", () => beginMultiplePick(id, "condition"), card);
    const mode = conditionalSelect([["all", "全部通过 (all)"], ["any", "任一通过 (any)"]], item.mode, "条件组合");
    mode.onchange = () => editMultiple(id, loop => { loop.mode = mode.value; });
    multipleField(card, "条件组合", mode);
    const advanced = document.createElement("details");
    advanced.append(Object.assign(document.createElement("summary"), { textContent: "高级 · 多起终点 / 种子 / 重试限制" }));
    for (const [field, title] of [["start_nodes", "起点"], ["end_nodes", "终点"], ["seed_nodes", "递增种子"]]) {
      multipleButton(`画布增删${title}`, () => beginMultiplePick(id, field, null, false, true), advanced);
      const candidates = Object.keys(state.workflow.nodes).filter(n => !["SaveImagesNode", "FolderImagesNode"].includes(state.workflow.nodes[n].class_type) && (field !== "seed_nodes" || nodeInfo(n).inputs.seed === "INT"));
      const values = Array.isArray(item[field]) ? item[field] : [];
      const options = [...new Set([...candidates, ...values])].map(n => [n, conditionalNodeCaption(n)]);
      const select = conditionalSelect(options, "", `高级${title}（可多选）`); select.multiple = true;
      [...select.options].forEach(option => { option.selected = values.includes(option.value); });
      select.onchange = () => editMultiple(id, loop => { loop[field] = [...select.selectedOptions].map(o => o.value); });
      multipleField(advanced, `高级${title}（可多选）`, select);
      values.forEach(n => multipleButton(`移除${title} ${n}`, () => editMultiple(id, loop => { loop[field] = loop[field].filter(v => v !== n); }), advanced));
    }
    advanced.append(Object.assign(document.createElement("p"), { textContent: "不必选择种子。起点的非负 INT seed 自动递增；-1 保持随机。固定 seed 且无递增种子的确定性节点可能每轮相同。最终保存可直接读取已验收的中间结果；其他普通下游消费中间输出时，须显式增加终点。" }));
    for (const [field, title, min, max] of [["max_iterations", "最大尝试次数", 1, 1000], ["timeout_seconds", "超时秒数（0 禁用）", 0, 86400]]) {
      const input = Object.assign(document.createElement("input"), { type: "number", min, max, step: field === "max_iterations" ? "1" : "any", value: item[field] ?? "" });
      input.onchange = () => editMultiple(id, loop => { loop[field] = input.value.trim() ? Number(input.value) : null; });
      multipleField(advanced, title, input);
    }
    const exhausted = conditionalSelect([["fail", "记为失败"], ["skip", "跳过，不保存"]], item.on_exhausted, "耗尽 / 超时策略");
    exhausted.onchange = () => editMultiple(id, loop => { loop.on_exhausted = exhausted.value; });
    multipleField(advanced, "耗尽 / 超时策略", exhausted);
    card.append(advanced); root.append(card);
  });
}

function renderMulti() {
  const busy = nodeDeletionDisabled();
  // Remember imported numeric IDs even if the user deletes before adding again.
  multipleLoops().forEach(item => {
    const number = /^loop_(\d+)$/.exec(item?.id)?.[1];
    if (number) multipleUI.nextId = Math.max(multipleUI.nextId, Number(number));
  });
  if (busy) {
    if (conditionalUI.picking || state.connecting) document.querySelector("#connection-hint").textContent = "运行 / 检查期间已退出点选和待连线。";
    conditionalUI.picking = null; state.connecting = null;
  }
  const saves = Object.keys(state.workflow.nodes).filter(id => state.workflow.nodes[id].class_type === "SaveImagesNode");
  if (conditionalUI.targetId == null && saves.length === 1) {
    conditionalUI.targetId = saves[0];
    if (loopUI.enabled) loopUI.targetId = saves[0];
  }
  // Layout is intentionally absent: dragging / polling must not rebuild focused fields.
  const key = JSON.stringify([state.workflow.nodes, state.workflow.conditional_loop, conditionalUI.targetId, conditionalUI.picking]);
  if (key !== conditionalUI.controlsKey) { conditionalUI.controlsKey = key; renderMultiFields(); }
  document.querySelectorAll("#multiple-loops input, #multiple-loops select, #multiple-loops button").forEach(control => { control.disabled = busy; });
  document.querySelector("#conditional-badge").textContent = `${multipleLoops().length} 个`;
  const problem = validateMultiple();
  const requestKey = !problem ? conditionalPlanKey() : null;
  if (conditionalUI.planKey !== requestKey && !conditionalUI.checking) { conditionalUI.plan = null; conditionalUI.planKey = null; conditionalUI.error = ""; }
  document.querySelector("#conditional-error").textContent = problem || conditionalUI.error;
  const plan = conditionalUI.plan;
  document.querySelector("#conditional-plan").textContent = conditionalUI.checking ? "正在检查计划（不推理、不写盘）…" : plan ? `最终保存：${plan.target_id}\n${(plan.loops || []).map((item, i) => `${i + 1}. ${item.name || item.id} (${item.id}) · ${item.start_nodes.join("、")} → ${item.end_nodes.join("、")}\n循环体：${item.body_nodes.join("、")} · 最多 ${item.max_iterations} 次`).join("\n")}\n全部循环通过后保存。` : "配置完成后检查计划；服务器按 DAG 排序循环，不推理、不写盘。";
  document.querySelector("#conditional-check").disabled = Boolean(problem) || busy;
  document.querySelector("#conditional-run").disabled = Boolean(problem) || busy || !plan || conditionalUI.planKey !== requestKey;
  renderMultipleRoles();
}

function renderMultipleRoles() {
  const loops = multipleLoops().filter(item => item && typeof item === "object");
  const regions = loops.map(item => ({ item, body: multipleBody(item) }));
  const pick = conditionalUI.picking;
  document.querySelectorAll(".node").forEach(card => {
    const id = card.dataset.nodeId, spec = state.workflow.nodes[id];
    if (!spec) return;
    const roles = [], members = regions.filter(r => r.body.has(id));
    loops.forEach((item, i) => {
      const tags = [];
      if (multipleValues(item, "start_nodes").includes(id)) tags.push("起点");
      if (multipleValues(item, "end_nodes").includes(id)) tags.push("终点");
      if (multipleValues(item, "seed_nodes").includes(id)) tags.push("递增种子");
      if (multipleValues(item, "conditions").some(c => c?.node === id)) tags.push("条件");
      if (tags.length) roles.push(`${i + 1} · ${item.name || item.id}：${tags.join(" / ")}`);
    });
    if (conditionalUI.targetId === id) roles.push("最终保存 · 全部循环通过后");
    card.style.setProperty("--conditional-color", multipleColor(members[0]?.item.id || loops.find(item => multipleValues(item, "start_nodes").includes(id) || multipleValues(item, "end_nodes").includes(id))?.id || "save"));
    card.classList.toggle("conditional-member", members.length > 0);
    card.classList.toggle("conditional-boundary", roles.length > 0);
    const badge = card.querySelector(".conditional-role"); if (badge) badge.textContent = roles.join(" · ");
    const eligible = pick && (pick.kind === "target" ? spec.class_type === "SaveImagesNode" : pick.kind === "condition" ? Object.values(nodeInfo(id).outputs).some(type => conditionalScalarTypes.includes(type)) : pick.kind === "seed_nodes" ? nodeInfo(id).inputs.seed === "INT" : !["SaveImagesNode", "FolderImagesNode"].includes(spec.class_type) && !nodeInfo(id).side_effects);
    card.classList.toggle("conditional-pickable", Boolean(eligible));
  });
  document.querySelectorAll(".port.output").forEach(port => {
    const { nodeId: id, port: output } = port.dataset;
    port.classList.toggle("conditional-pick-port", Boolean(pick?.kind === "condition" && state.workflow.nodes[id] && conditionalScalarTypes.includes(nodeInfo(id).outputs[output])));
    port.classList.toggle("conditional-selected-port", loops.some(item => multipleValues(item, "conditions").some(c => c?.node === id && c?.output === output)));
  });
  if (pick) document.querySelector("#connection-hint").textContent = multiplePickHint(pick);
}

document.querySelector("#conditional-add").onclick = event => {
  event?.preventDefault?.(); event?.stopPropagation?.(); addMultipleLoop();
};
