const state = { catalog: {}, workflow: null, layout: {}, selected: null, connecting: null, config: {}, configHelp: {}, nodeResults: {}, nodeResultStale: {}, previewHidden: {}, nodeErrors: {}, expanded: {}, previewSession: null, previewRevision: 0 };
const workspace = document.querySelector("#workspace");
const nodesElement = document.querySelector("#nodes");
const connectionsElement = document.querySelector("#connections");
const statusElement = document.querySelector("#status");
const batch = { job: null, starting: false, restoring: true, requesting: false, uncertain: false, error: "", timer: null };
const batchActive = () => ["running", "cancelling"].includes(batch.job?.status);
const batchStartDisabled = () => batch.starting || batch.restoring || batch.requesting || batch.uncertain || batchActive();
const nodeDeletionDisabled = () => loopUI.busy || loopUI.checking || conditionalUI.checking || batchStartDisabled();

const api = async (url, options = {}) => {
  const response = await fetch(url, { headers: { "Content-Type": "application/json" }, ...options });
  const text = await response.text();
  let data;
  try { data = JSON.parse(text); } catch { throw new Error(`HTTP ${response.status}: ${text.slice(0, 300)}`); }
  if (!response.ok) {
    const error = new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail || data));
    error.nodeError = data.node_error || null;
    throw error;
  }
  return data;
};

const link = (node, output) => ({ node, output });
const context = (name) => ({ context: name });
const isLink = (value) => value && typeof value === "object" && Object.keys(value).length === 2 && "node" in value && "output" in value;
const isContext = (value) => value && typeof value === "object" && Object.keys(value).length === 1 && "context" in value;
function saveItems(spec) {
  if (!Object.hasOwn(spec, "save_items")) {
    const ids = new Set(["img1", "img2", "img3", "img4", ...Object.keys(spec.inputs).filter(key => /^img[1-9][0-9]*$/.test(key))]);
    spec.save_items = [...ids].sort((a, b) => Number(a.slice(3)) - Number(b.slice(3))).map(id => ({ id, name: "" }));
  }
  spec.save_next_id ??= Math.max(0, ...spec.save_items.map(item => Number(item.id.slice(3)))) + 1;
  return spec.save_items;
}

function nodeInfo(id) {
  const spec = state.workflow.nodes[id];
  const info = state.catalog[spec.class_type];
  if (spec.class_type === "ColorRegionSwatchesNode") {
    const indices = spec.inputs.indices ?? info.input_defaults?.indices ?? [0];
    const selected = validSwatchIndices(indices) ? indices : [0];
    // Older, not-yet-normalized workflows may still consume the list output.
    const usesImages = binding => isLink(binding) && binding.node === id && binding.output === "images";
    if (Object.values(state.workflow.outputs || {}).some(usesImages) || Object.values(state.workflow.nodes).some(node => Object.values(node.inputs).some(usesImages))) spec.legacy_swatch_images = true;
    const outputs = {};
    const labels = {};
    const helpOutputs = {};
    selected.forEach((index, position) => {
      const name = `img${position + 1}`;
      outputs[name] = "RGB_IMAGE";
      labels[name] = { name, role: `index=${index}`, kind: "rgb_image" };
      helpOutputs[name] = `面积排名 index=${index}（从 0 开始）的独立 RGB 色块；按 indices 列表位置映射，重排会改变此端口内容。`;
    });
    outputs.colors = "JSON_RECORD";
    helpOutputs.colors = info.help.outputs.colors;
    if (info.port_labels?.outputs?.colors) labels.colors = info.port_labels.outputs.colors;
    if (spec.legacy_swatch_images === true) {
      outputs.images = "IMAGE_LIST";
      labels.images = { name: "images", role: "旧流程兼容图片列表", kind: "image_list" };
      helpOutputs.images = "旧流程兼容图片列表；新连接请使用独立 imgN 输出。";
    }
    return { ...info, outputs, port_labels: { ...info.port_labels, outputs: labels }, help: { ...info.help, outputs: helpOutputs } };
  }
  if (spec.class_type !== "SaveImagesNode") return info;
  const items = saveItems(spec);
  const inputs = { ...Object.fromEntries(items.map(item => [item.id, "IMAGE"])), images: "IMAGE_LIST", dir: "PATH", filename: "STRING" };
  const labels = { ...info.port_labels?.inputs };
  const helpInputs = { images: info.help.inputs.images, dir: info.help.inputs.dir, filename: info.help.inputs.filename };
  items.forEach(item => {
    labels[item.id] = { name: item.name.trim() || item.id, role: `保存项 ${item.id.slice(3)}，任意模式`, kind: "image" };
    helpInputs[item.id] = "连接 RGB、RGBA 或 Mask，也可上传；旁边填写保存子目录名称，留空使用内部编号。";
  });
  return { ...info, inputs, port_labels: { ...info.port_labels, inputs: labels }, help: { ...info.help, inputs: helpInputs } };
}

function isSwatchIndices(nodeId, name) {
  return name === "indices" && state.workflow?.nodes[nodeId]?.class_type === "ColorRegionSwatchesNode";
}

function validSwatchIndices(value) {
  return Array.isArray(value) && value.length > 0 && value.every(index => Number.isSafeInteger(index) && index >= 0);
}

function editSwatchIndices(nodeId, text) {
  if (nodeDeletionDisabled()) { setStatus("运行 / 检查任务期间不能修改色块 indices 输出拓扑。", true); return false; }
  if (!isSwatchIndices(nodeId, "indices")) return false;
  let indices;
  try { indices = JSON.parse(text); } catch { /* Report the same literal-only validation below. */ }
  if (!validSwatchIndices(indices)) {
    setStatus("indices 必须是非空的非负整数 JSON 列表（从 0 开始，例如 [0,2,1]）；不接受布尔值、连线或 context。", true);
    return false;
  }
  const spec = state.workflow.nodes[nodeId];
  const previous = nodeInfo(nodeId).outputs;
  // Mark descendants stale before removing edges, so disconnected previews remain historical.
  invalidateResults(nodeId);
  spec.inputs.indices = indices;
  const outputs = nodeInfo(nodeId).outputs;
  const removed = new Set(Object.keys(previous).filter(name => !Object.hasOwn(outputs, name)));
  const removedLink = binding => isLink(binding) && binding.node === nodeId && removed.has(binding.output);
  Object.values(state.workflow.nodes).forEach(node => {
    const defaults = state.catalog[node.class_type]?.input_defaults || {};
    Object.entries(node.inputs).forEach(([name, binding]) => {
      if (removedLink(binding)) node.inputs[name] = Object.hasOwn(defaults, name) ? structuredClone(defaults[name]) : null;
    });
  });
  Object.entries(state.workflow.outputs || {}).forEach(([name, binding]) => { if (removedLink(binding)) delete state.workflow.outputs[name]; });
  if (state.connecting?.node === nodeId && removed.has(state.connecting.output)) {
    state.connecting = null;
    document.querySelector("#connection-hint").textContent = "输出已移除，请重新选择连接起点。";
  }
  clearTimeout(loopUI.timer);
  loopUI.plan = null; loopUI.planKey = null; loopUI.error = "";
  conditionalUI.plan = null; conditionalUI.planKey = null; conditionalUI.error = "";
  // Keep conditional references explicit and stale rather than silently deleting conditions.
  render();
  setStatus(`已更新色块输出：imgN 按 indices 列表位置映射（重排会改变内容）。${removed.size ? `已断开移除的输出：${[...removed].join("、")}；条件引用保留，请明确修正失效配置。` : ""}`);
  return true;
}

function saveItemsChanged(nodeId) {
  clearTimeout(loopUI.timer);
  loopUI.plan = null;
  loopUI.planKey = null;
  loopUI.error = "";
  invalidateResults(nodeId);
  render();
}

function addSaveItem(nodeId) {
  if (nodeDeletionDisabled()) return setStatus("运行 / 检查任务期间不能修改保存项。", true);
  const spec = state.workflow.nodes[nodeId];
  if (spec?.class_type !== "SaveImagesNode") return;
  const items = saveItems(spec);
  if (!Number.isSafeInteger(spec.save_next_id) || spec.save_next_id < 1) return setStatus("保存项编号无效，请重新导入有效工作流。", true);
  const id = `img${spec.save_next_id++}`;
  items.push({ id, name: "" });
  spec.inputs[id] = null;
  saveItemsChanged(nodeId);
  setStatus(`已添加 ${id}；可连接图片并填写保存名称。`);
}

function removeSaveItem(nodeId, itemId) {
  if (nodeDeletionDisabled()) return setStatus("运行 / 检查任务期间不能修改保存项。", true);
  const spec = state.workflow.nodes[nodeId];
  if (spec?.class_type !== "SaveImagesNode") return;
  spec.save_items = saveItems(spec).filter(item => item.id !== itemId);
  delete spec.inputs[itemId];
  saveItemsChanged(nodeId);
  setStatus(`已移除 ${itemId} 及其输入连线；其余项不重新编号，未删除磁盘文件。`);
}

function renameSaveItem(nodeId, itemId, value) {
  if (nodeDeletionDisabled()) { setStatus("运行 / 检查任务期间不能修改保存名称。", true); return false; }
  const spec = state.workflow.nodes[nodeId];
  const items = saveItems(spec);
  const item = items.find(item => item.id === itemId);
  if (!item) return false;
  const name = value.trim();
  if (name && (name === "." || name === ".." || name.endsWith(".") || /[/\\<>:"|?*\x00-\x1f\x7f]/.test(name))) {
    setStatus("保存名称不能包含路径分隔符、控制字符或 . / ..。", true); return false;
  }
  if (items.some(other => other.id !== itemId && (other.name.trim() || other.id).toLowerCase() === (name || itemId).toLowerCase())) {
    setStatus("保存名称重复，请为每项填写不同名称。", true); return false;
  }
  item.name = name;
  saveItemsChanged(nodeId);
  return true;
}

function saveItemNameWidget(nodeId, itemId) {
  const item = saveItems(state.workflow.nodes[nodeId]).find(item => item.id === itemId);
  const input = Object.assign(document.createElement("input"), { value: item.name, placeholder: itemId, className: "save-item-name", disabled: nodeDeletionDisabled() });
  input.dataset.saveName = itemId;
  input.title = `保存子目录名称；留空使用 ${itemId}。改名不影响连线。`;
  input.setAttribute("aria-label", `${itemId} 保存名称`);
  const apply = () => { if (!renameSaveItem(nodeId, itemId, input.value)) input.value = item.name; };
  input.onchange = apply;
  input.onkeydown = event => { if (event.key === "Enter") { input.blur(); } };
  return input;
}

function addSaveItemButton(nodeId) {
  const button = Object.assign(document.createElement("button"), { textContent: "＋添加图片", className: "save-item-add", disabled: nodeDeletionDisabled() });
  button.onclick = () => addSaveItem(nodeId);
  return button;
}

const portType = (nodeId, port) => nodeInfo(nodeId).outputs[port];
const position = (id, index = 0) => state.layout[id] || { x: 50 + (index % 5) * 350, y: 50 + Math.floor(index / 5) * 680 };

function setStatus(value, isError = false) {
  statusElement.textContent = typeof value === "string" ? value : JSON.stringify(value, null, 2);
  statusElement.style.color = isError ? "#ff9f9f" : "#a8b9dd";
}

function bindingLabel(value) {
  if (isLink(value)) return `${value.node} → ${portCaption(value.node, "outputs", value.output)}`;
  if (isContext(value)) return `context: ${value.context}`;
  return JSON.stringify(value);
}

function portMetadata(nodeId, direction, name) {
  const info = state.workflow?.nodes[nodeId] ? nodeInfo(nodeId) : null;
  return info?.port_labels?.[direction]?.[name] || { name, role: "", kind: info?.[direction]?.[name] || "" };
}

function portCaption(nodeId, direction, name) {
  const label = portMetadata(nodeId, direction, name);
  return label.role ? `${label.name} · ${label.role}` : label.name;
}

function compatiblePortTypes(expected, actual) {
  return expected === actual || (expected === "IMAGE" && ["RGB_IMAGE", "RGBA_IMAGE", "MASK", "LABEL_MAP"].includes(actual));
}

function connectionMatch(nodeId, input) {
  if (!state.connecting) return "";
  if (isSwatchIndices(nodeId, input)) return "incompatible";
  const expected = nodeInfo(nodeId).inputs[input];
  const actual = portType(state.connecting.node, state.connecting.output);
  if (nodeId === state.connecting.node || !compatiblePortTypes(expected, actual)) return "incompatible";
  const source = portMetadata(state.connecting.node, "outputs", state.connecting.output);
  const target = portMetadata(nodeId, "inputs", input);
  return source.kind === target.kind || expected === "IMAGE" || (actual === "MASK" && target.kind === "mask") ? "recommended" : "compatible";
}

function render() {
  loopScope();
  nodesElement.replaceChildren();
  connectionsElement.replaceChildren();
  Object.entries(state.workflow.nodes).forEach(([id, spec], index) => renderNode(id, spec, index));
  requestAnimationFrame(renderEdges);
  renderInspector();
  renderBatch();
}

function renderNode(id, spec, index) {
  const info = nodeInfo(id);
  const point = position(id, index);
  state.layout[id] = point;
  const card = document.createElement("article");
  card.className = `node${state.selected === id ? " selected" : ""}`;
  card.dataset.nodeId = id;
  card.style.left = `${point.x}px`;
  card.style.top = `${point.y}px`;
  const header = document.createElement("div");
  header.className = "node-header";
  header.textContent = `${id} · ${spec.class_type.replace(/Node$/, "")}`;
  header.append(Object.assign(document.createElement("span"), { className: "loop-role" }));
  header.append(Object.assign(document.createElement("span"), { className: "conditional-role" }));
  if (spec.class_type === "FolderImagesNode") header.append(folderLoopSwitch(id));
  header.append(nodeDeleteButton(id, true));
  header.addEventListener("pointerdown", event => { if (!event.target.closest("button") && !conditionalUI.picking) beginDrag(event, id); });
  header.addEventListener("click", event => { if (!event.target.closest("button") && !consumeConditionalPick(id)) { state.selected = id; render(); } });
  card.append(header);
  const summary = Object.assign(document.createElement("p"), { className: "node-summary", textContent: `${info.help.title}：${info.help.summary}` });
  card.append(summary);
  if (info.input_contract && info.output_contract) card.append(Object.assign(document.createElement("p"), { className: "node-contract", textContent: `入口 ${info.input_contract} → 出口 ${info.output_contract}` }));
  if (state.nodeErrors[id]) {
    card.classList.add("node-failed");
    card.append(Object.assign(document.createElement("pre"), { className: "node-error", textContent: state.nodeErrors[id] }));
  }
  if (["FolderImagesNode", "SaveImagesNode"].includes(spec.class_type)) card.append(Object.assign(document.createElement("p"), { className: "node-path" }));
  const ports = document.createElement("div");
  ports.className = "ports";
  Object.entries(info.inputs).forEach(([name, type]) => ports.append(portRow(id, name, type, "input")));
  if (spec.class_type === "SaveImagesNode") ports.append(addSaveItemButton(id));
  Object.entries(info.outputs).forEach(([name, type]) => ports.append(portRow(id, name, type, "output")));
  card.append(ports);
  const widgets = document.createElement("div");
  widgets.className = "node-widgets";
  const helpButton = Object.assign(document.createElement("button"), { textContent: "说明与测试", className: "node-help-button" });
  helpButton.onclick = () => { state.selected = id; render(); document.querySelector("#node-help").scrollIntoView({ block: "nearest" }); };
  widgets.append(helpButton);
  if (spec.class_type === "RecordInputNode") {
    [["target_path", "target-path"], ["mask_path", "mask-path"], ["object_name", "object-name"]].forEach(([name, field]) => {
      const input = inputWidget(name, document.querySelector(`#${field}`).value, value => {
        document.querySelector(`#${field}`).value = value;
      }, portCaption(id, "outputs", name));
      if (name.endsWith("_path")) input.append(uploadPicker(path => {
        document.querySelector(`#${field}`).value = path; render();
      }, name));
      widgets.append(input);
    });
  } else {
    Object.entries(info.inputs).forEach(([name, type]) => {
      if (["PATH", "STRING", "RGB_IMAGE", "MASK", "IMAGE", "BBOX", "JSON_RECORD", "INT", "FLOAT", "INT_LIST"].includes(type)) {
        const binding = spec.inputs[name] ?? info.input_defaults?.[name];
        const field = isLink(binding) && binding.node === "record" ? ({ target_path: "target-path", mask_path: "mask-path", object_name: "object-name" })[binding.output] : null;
        const shown = field ? document.querySelector(`#${field}`).value : typeof binding === "string" ? binding : !isLink(binding) && binding != null ? JSON.stringify(binding) : "";
        const input = inputWidget(name, shown, value => { spec.inputs[name] = ["BBOX", "JSON_RECORD", "INT", "FLOAT", "INT_LIST"].includes(type) ? parseLiteral(value) : value; }, portCaption(id, "inputs", name), id);
        input.title = `${portCaption(id, "inputs", name)}\n内部字段：${name}\n${info.help.inputs[name] || ""}`;
        if (["PATH", "RGB_IMAGE", "MASK", "IMAGE"].includes(type) && name !== "dir") input.append(uploadPicker(path => { spec.inputs[name] = path; render(); }, name, id));
        if (isLink(binding)) input.append(Object.assign(document.createElement("small"), { textContent: `当前连接 ${bindingLabel(binding)}；填写或选图将覆盖这条连接。` }));
        widgets.append(input);
      }
      if (type === "CONFIG") {
        const button = Object.assign(document.createElement("button"), { textContent: "编辑相关参数（全局共享）" });
        button.onclick = () => { state.selected = id; render(); document.querySelector(`#config-${info.help.parameters[0] || "resolution"}`).focus(); };
        widgets.append(button);
      }
    });
  }
  const saveImages = spec.class_type === "SaveImagesNode";
  const previewButton = Object.assign(document.createElement("button"), { textContent: saveImages ? conditionalEnabled() ? "条件保存当前张 · 查看计划" : "仅保存当前张 · 不循环" : info.side_effects ? "写入节点：仅导出时执行" : "测试此节点 · 自动预览", className: "node-preview-button", disabled: info.side_effects && !saveImages });
  previewButton.onclick = () => previewNode(id);
  widgets.append(previewButton);
  if (saveImages) {
    widgets.append(Object.assign(document.createElement("small"), { className: "save-run-hint", textContent: "当前张不会遍历文件夹；启用条件重试时，上方按钮会定位条件计划，只有通过才保存。处理所有图片：开启「↻ 逐张」，再点击画布顶部「运行全部 · 写入磁盘」。" }));
    const batchButton = Object.assign(document.createElement("button"), { textContent: "↻ 查看逐张流程", className: "node-batch-button", disabled: loopUI.busy });
    batchButton.onclick = () => {
      const sources = upstreamNodes(id).filter(nodeId => state.workflow.nodes[nodeId].class_type === "FolderImagesNode");
      if (sources.length !== 1) return setStatus("此保存节点的上游必须恰好连接一个文件夹节点，才能开启逐张流程。", true);
      chooseLoop(sources[0], true, id);
      document.querySelector("#loop-toolbar").scrollIntoView({ block: "nearest" });
    };
    widgets.append(batchButton);
  }
  const settings = document.createElement("details");
  settings.className = "node-settings";
  settings.open = state.expanded[id] ?? !loopUI.enabled;
  const settingsTitle = Object.assign(document.createElement("summary"), { textContent: "参数与单节点测试" });
  settingsTitle.onclick = () => { state.expanded[id] = !settings.open; requestAnimationFrame(renderEdges); };
  settings.append(settingsTitle, widgets);
  card.append(settings);
  // Previews are independent of parameter/loop folding and execution validity.
  if (state.nodeResults[id]) card.append(nodePreviewView(id, true));
  nodesElement.append(card);
}

function inputWidget(name, value, update, caption = name, nodeId = null) {
  const label = Object.assign(document.createElement("label"), { textContent: caption });
  const input = Object.assign(document.createElement("input"), { value, placeholder: `Enter ${caption}` });
  input.dataset.field = name;
  const swatchIndices = isSwatchIndices(nodeId, name);
  if (swatchIndices) {
    input.className = "swatch-topology-control";
    input.disabled = nodeDeletionDisabled();
    input.title = "非空非负整数 JSON 列表，例如 [0,2,1]；仅支持字面量。imgN 按列表位置映射，重排会改变输出内容。";
  }
  const apply = () => {
    if (swatchIndices) return editSwatchIndices(nodeId, input.value);
    update(input.value); invalidateResults(nodeId); requestAnimationFrame(renderEdges);
  };
  input.onchange = apply;
  input.onkeydown = event => { if (event.key === "Enter") { apply(); input.blur(); } };
  label.append(input);
  return label;
}

function uploadPicker(update, name, nodeId = null) {
  const label = Object.assign(document.createElement("span"), { className: "button file-picker", textContent: "选择图片…", tabIndex: 0, role: "button" });
  const input = Object.assign(document.createElement("input"), { type: "file", accept: "image/*", hidden: true });
  input.dataset.upload = name;
  input.onchange = async () => {
    if (!input.files?.[0]) return;
    try {
      const data = new FormData(); data.append("file", input.files[0]);
      const result = await api("/api/upload", { method: "POST", headers: {}, body: data });
      invalidateResults(nodeId); update(result.path); setStatus("图片已上传，点击“测试此节点”查看输出。上传文件保留到服务停止。");
    } catch (error) { setStatus(error.message, true); }
  };
  label.onclick = event => { if (event.target !== input) input.click(); };
  label.onkeydown = event => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); input.click(); } };
  label.append(input);
  return label;
}

function portRow(nodeId, name, type, direction) {
  const row = document.createElement("div");
  row.className = "port-row";
  const port = document.createElement("button");
  port.className = `port ${direction}${direction === "output" && state.connecting?.node === nodeId && state.connecting?.output === name ? " pending" : ""}`;
  const caption = portCaption(nodeId, `${direction}s`, name);
  const match = direction === "input" ? connectionMatch(nodeId, name) : "";
  if (match) port.classList.add(`port-${match}`);
  row.dataset.direction = direction;
  row.dataset.match = match;
  const matchText = { recommended: "推荐：用途匹配；仍需核对尺寸和内容。", compatible: "类型兼容，但用途不同，请确认后连接。", incompatible: "类型不兼容或不能连接自身。" }[match] || "";
  port.title = `${direction === "input" ? "输入" : "输出"}：${caption}\n类型：${type}\n内部字段：${name}${matchText ? `\n${matchText}` : ""}`;
  port.setAttribute("aria-label", `${direction === "input" ? "输入" : "输出"} ${caption}`);
  port.dataset.nodeId = nodeId;
  port.dataset.port = name;
  port.dataset.direction = direction;
  port.addEventListener("click", event => {
    event.stopPropagation();
    if (conditionalUI.picking) {
      if (direction === "output") consumeConditionalPick(nodeId, name);
      else setStatus("点选模式下请点击节点标题，或在指定条件时点击输出圆点；输入端口不会被修改。", true);
      return;
    }
    direction === "output" ? beginConnection(nodeId, name) : completeConnection(nodeId, name);
  });
  const label = document.createElement("span");
  label.className = "port-label";
  label.textContent = caption;
  label.title = port.title;
  const typeLabel = document.createElement("span");
  typeLabel.className = "port-type";
  typeLabel.textContent = type;
  if (direction === "input") row.append(port, label, typeLabel); else row.append(typeLabel, label, port);
  if (direction === "input" && state.workflow.nodes[nodeId].class_type === "SaveImagesNode" && /^img[1-9][0-9]*$/.test(name)) {
    row.classList.add("save-item-row");
    const remove = Object.assign(document.createElement("button"), { textContent: "×", className: "save-item-remove", disabled: nodeDeletionDisabled() });
    remove.title = `移除 ${name} 及其连线；不删除磁盘文件`;
    remove.setAttribute("aria-label", `移除保存项 ${name}`);
    remove.onclick = event => { event.stopPropagation(); removeSaveItem(nodeId, name); };
    row.replaceChildren(port, saveItemNameWidget(nodeId, name), remove);
  }
  return row;
}

function beginConnection(nodeId, output) {
  state.connecting = { node: nodeId, output };
  document.querySelector("#connection-hint").textContent = `正在连接 ${nodeId} → ${portCaption(nodeId, "outputs", output)}。亮绿色：用途匹配；蓝色：类型可连，请核对用途；暗色：类型不兼容或自身。`;
  render();
}

function completeConnection(nodeId, input) {
  if (isSwatchIndices(nodeId, input) && state.connecting) return setStatus("indices 仅支持非空非负整数字面量列表，不能连接输出或 context。", true);
  if (!state.connecting) {
    state.selected = nodeId;
    render();
    const block = Array.from(document.querySelectorAll("#node-fields .binding")).find(element => element.dataset.port === input);
    block?.querySelector("input")?.focus();
    return;
  }
  const expected = nodeInfo(nodeId).inputs[input];
  const actual = portType(state.connecting.node, state.connecting.output);
  if (!compatiblePortTypes(expected, actual)) return setStatus(`无法连接 ${portCaption(state.connecting.node, "outputs", state.connecting.output)} (${actual}) → ${portCaption(nodeId, "inputs", input)} (${expected})。`, true);
  if (state.connecting.node === nodeId) return setStatus("A node cannot connect to itself.", true);
  if (upstreamNodes(state.connecting.node).includes(nodeId)) return setStatus("这条连接会形成回路。请通过文件夹上的「↻ 逐张」表达循环，不需要回流连线。", true);
  state.workflow.nodes[nodeId].inputs[input] = link(state.connecting.node, state.connecting.output);
  invalidateResults(nodeId);
  state.connecting = null;
  document.querySelector("#connection-hint").textContent = `已连接到 ${nodeId} → ${portCaption(nodeId, "inputs", input)}。`;
  render();
}

function renderEdges() {
  connectionsElement.replaceChildren();
  const scope = loopScope();
  const definitions = document.createElementNS("http://www.w3.org/2000/svg", "defs");
  definitions.innerHTML = '<marker id="loop-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#74d8b5" /></marker>';
  connectionsElement.append(definitions);
  Object.entries(state.workflow.nodes).forEach(([targetId, spec]) => {
    Object.entries(spec.inputs).forEach(([input, value]) => {
      if (!isLink(value)) return;
      const source = document.querySelector(`.port.output[data-node-id="${CSS.escape(value.node)}"][data-port="${CSS.escape(value.output)}"]`);
      const target = document.querySelector(`.port.input[data-node-id="${CSS.escape(targetId)}"][data-port="${CSS.escape(input)}"]`);
      if (!source || !target) return;
      const sourceRect = source.getBoundingClientRect();
      const targetRect = target.getBoundingClientRect();
      const workspaceRect = workspace.getBoundingClientRect();
      const x1 = sourceRect.left + sourceRect.width / 2 - workspaceRect.left;
      const y1 = sourceRect.top + sourceRect.height / 2 - workspaceRect.top;
      const x2 = targetRect.left + targetRect.width / 2 - workspaceRect.left;
      const y2 = targetRect.top + targetRect.height / 2 - workspaceRect.top;
      const bend = Math.max(70, Math.abs(x2 - x1) * 0.45);
      const edge = document.createElementNS("http://www.w3.org/2000/svg", "path");
      const repeated = loopUI.enabled && scope.repeated.has(value.node) && scope.repeated.has(targetId);
      const shared = loopUI.enabled && !repeated && scope.ancestors.has(targetId);
      const outside = loopUI.enabled && loopUI.targetId && !scope.ancestors.has(targetId);
      edge.setAttribute("class", `edge${repeated ? " loop-edge" : shared ? " shared-edge" : outside ? " outside-edge" : ""}`);
      if (repeated) edge.setAttribute("marker-end", "url(#loop-arrow)");
      edge.dataset.source = value.node;
      edge.dataset.target = targetId;
      edge.setAttribute("d", `M ${x1} ${y1} C ${x1 + bend} ${y1}, ${x2 - bend} ${y2}, ${x2} ${y2}`);
      connectionsElement.append(edge);
    });
  });
  renderLoopRegion(scope);
  const cards = [...document.querySelectorAll(".node")];
  workspace.style.width = `${Math.max(3000, ...cards.map(card => card.offsetLeft + card.offsetWidth + 60))}px`;
  workspace.style.height = `${Math.max(2600, ...cards.map(card => card.offsetTop + card.offsetHeight + 60))}px`;
}

function beginDrag(event, id) {
  event.preventDefault();
  state.selected = id;
  document.querySelectorAll(".node").forEach(card => card.classList.toggle("selected", card.dataset.nodeId === id));
  renderInspector();
  const initial = position(id);
  const startX = event.clientX;
  const startY = event.clientY;
  const move = moveEvent => {
    state.layout[id] = { x: Math.max(0, initial.x + moveEvent.clientX - startX), y: Math.max(0, initial.y + moveEvent.clientY - startY) };
    const card = document.querySelector(`.node[data-node-id="${CSS.escape(id)}"]`);
    card.style.left = `${state.layout[id].x}px`;
    card.style.top = `${state.layout[id].y}px`;
    connectionsElement.replaceChildren();
    renderEdges();
  };
  const up = () => { window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); };
  window.addEventListener("pointermove", move);
  window.addEventListener("pointerup", up);
}

function renderInspector() {
  const fields = document.querySelector("#node-fields");
  fields.replaceChildren();
  const help = document.querySelector("#node-help");
  help.replaceChildren();
  const preview = document.querySelector("#node-preview");
  preview.replaceChildren();
  if (state.selected && state.nodeErrors[state.selected]) preview.append(Object.assign(document.createElement("pre"), { className: "node-error", textContent: state.nodeErrors[state.selected] }));
  if (state.selected && state.nodeResults[state.selected]) preview.append(nodePreviewView(state.selected));
  if (!state.selected) return fields.append(Object.assign(document.createElement("p"), { textContent: "Select a node to edit its input bindings." }));
  const spec = state.workflow.nodes[state.selected];
  const info = nodeInfo(state.selected);
  help.append(Object.assign(document.createElement("h3"), { textContent: info.help.title }));
  help.append(Object.assign(document.createElement("p"), { textContent: info.help.summary }));
  if (info.input_contract && info.output_contract) help.append(Object.assign(document.createElement("p"), { className: "node-contract", textContent: `入口类：${info.input_contract}\n出口类：${info.output_contract}` }));
  help.append(Object.assign(document.createElement("p"), { className: "test-guide", textContent: `怎么单独测试：${info.help.test}` }));
  for (const [kind, ports] of [["输入", info.help.inputs], ["输出", info.help.outputs]]) {
    help.append(Object.assign(document.createElement("strong"), { textContent: kind }));
    const list = document.createElement("dl");
    Object.entries(ports).forEach(([name, text]) => {
      const direction = kind === "输入" ? "inputs" : "outputs";
      list.append(Object.assign(document.createElement("dt"), { textContent: portCaption(state.selected, direction, name) }), Object.assign(document.createElement("dd"), { textContent: text }));
    });
    help.append(list);
  }
  if (info.help.parameters.length) help.append(Object.assign(document.createElement("p"), { textContent: `相关参数（右侧下方全局共享）：${info.help.parameters.map(name => `${state.configHelp[name]} (${name})`).join("；")}` }));
  fields.append(Object.assign(document.createElement("h3"), { textContent: state.selected }));
  fields.append(nodeDeleteButton(state.selected));
  if (spec.class_type === "SaveImagesNode") fields.append(addSaveItemButton(state.selected));
  Object.entries(info.inputs).forEach(([name, type]) => {
    const block = document.createElement("div");
    block.className = "binding";
    block.dataset.port = name;
    const title = Object.assign(document.createElement("strong"), { textContent: portCaption(state.selected, "inputs", name) });
    const description = Object.assign(document.createElement("small"), { textContent: `${type} · 内部字段：${name} · ${bindingLabel(spec.inputs[name])}` });
    block.append(title, description);
    if (spec.class_type === "SaveImagesNode" && /^img[1-9][0-9]*$/.test(name)) block.append(saveItemNameWidget(state.selected, name));
    const mode = document.createElement("select");
    const swatchIndices = isSwatchIndices(state.selected, name);
    const nodeId = state.selected;
    (swatchIndices ? [["literal", "Literal only · 非空非负整数列表"]] : [["context", "Context"], ["literal", "Literal"], ["link", "Link via output port"]]).forEach(([value, text]) => mode.add(new Option(text, value)));
    mode.value = isLink(spec.inputs[name]) ? "link" : isContext(spec.inputs[name]) ? "context" : "literal";
    if (swatchIndices) { mode.value = "literal"; mode.disabled = true; }
    const value = document.createElement("input");
    value.value = isContext(spec.inputs[name]) ? spec.inputs[name].context : isLink(spec.inputs[name]) ? bindingLabel(spec.inputs[name]) : JSON.stringify(spec.inputs[name] ?? (swatchIndices ? info.input_defaults?.indices ?? [0] : undefined)) ?? "";
    const actions = document.createElement("div");
    actions.className = "binding-actions";
    const apply = document.createElement("button");
    apply.textContent = "Apply";
    apply.onclick = () => {
      if (swatchIndices) {
        if (mode.value !== "literal") return setStatus("indices 仅支持字面量列表，不能使用连线或 context。", true);
        return editSwatchIndices(nodeId, value.value);
      }
      if (mode.value === "context") {
        if (!value.value.trim()) return setStatus("Context name cannot be empty.", true);
        spec.inputs[name] = context(value.value.trim());
      }
      if (mode.value === "literal") spec.inputs[name] = parseLiteral(value.value);
      if (mode.value === "link") return setStatus("Click an output port, then this input port to create a link.");
      invalidateResults(state.selected); render();
    };
    const remove = document.createElement("button");
    remove.textContent = "Disconnect";
    remove.disabled = swatchIndices;
    remove.onclick = () => {
      if (swatchIndices) return setStatus("indices 不能为空；请填写非空非负整数列表。", true);
      spec.inputs[name] = spec.class_type === "SaveImagesNode" && name === "filename" ? "" : null; invalidateResults(state.selected); render();
    };
    if (swatchIndices) {
      [value, apply].forEach(control => { control.className = "swatch-topology-control"; control.disabled = nodeDeletionDisabled(); });
      value.dataset.field = "indices";
      value.title = "imgN 按 indices 列表位置映射；重排会改变内容。仅支持非空非负整数 JSON 列表。";
    }
    actions.append(apply, remove);
    block.append(mode, value, actions);
    fields.append(block);
  });
}

/**
 * 渲染自动分色区域卡片；缩略图已按区域包围框裁剪，不用于数据导出。
 * @param {{count: number, items: Array<{id: number, pixels: number, fraction: number, color: number[], mask: string, rgba: string}>}} output 后端 color_regions JSON。
 * @param {boolean} compact true 时只显示摘要；详情面板显示区域卡片。
 * @returns {HTMLElement} 包含类别数、代表色、面积占比和可放大的 Mask / RGBA 预览。
 */
function colorRegionView(output, compact) {
  const section = Object.assign(document.createElement("section"), { className: "color-region-view" });
  section.append(Object.assign(document.createElement("strong"), { textContent: `自动分色 · ${output.count} 个颜色区域` }));
  const colors = Object.assign(document.createElement("div"), { className: "region-swatches" });
  output.items.forEach(region => {
    const swatch = Object.assign(document.createElement("span"), { className: "region-swatch" });
    swatch.style.backgroundColor = `rgb(${region.color.join(",")})`;
    swatch.title = `区域 ${region.id} · RGB(${region.color.join(", ")}) · ${(region.fraction * 100).toFixed(2)}%`;
    colors.append(swatch);
  });
  section.append(colors);
  if (compact) {
    section.append(Object.assign(document.createElement("p"), { textContent: "完整区域卡片请查看右侧当前节点输出；编号仅对本图有效。" }));
    return section;
  }
  section.append(Object.assign(document.createElement("p"), { textContent: "色块为原图平均颜色；叠加图的伪彩色只标识分区。面积以输入 Mask 为分母。预览已裁剪，完整 Mask / RGBA 请连接「选择颜色区域」。" }));
  output.items.forEach((region, index) => {
    const card = Object.assign(document.createElement("details"), { className: "color-region-card", open: index < 3 });
    card.dataset.regionId = region.id;
    card.append(Object.assign(document.createElement("summary"), { textContent: `区域 ${region.id} · ${(region.fraction * 100).toFixed(2)}% · ${region.pixels} 像素` }));
    card.append(Object.assign(document.createElement("p"), { textContent: `实际代表色 RGB(${region.color.join(", ")})` }));
    card.append(nodeOutputView({
      "区域 Mask": { kind: "image", image: region.mask, mode: "L" },
      "原色抠图": { kind: "image", image: region.rgba, mode: "RGBA" },
    }));
    section.append(card);
  });
  return section;
}

function refreshPreviewState() {
  document.querySelectorAll(".node-preview-shell").forEach(shell => {
    const id = shell.dataset.nodeId;
    shell.querySelector(".preview-state").textContent = state.nodeResultStale[id]
      ? "历史预览 · 输入或上游可能已变化，需重新测试确认" : "最近一次成功输出";
    shell.querySelector(".preview-toggle").textContent = state.previewHidden[id] ? "显示预览" : "隐藏预览";
    shell.querySelector(".preview-body").hidden = state.previewHidden[id] === true;
  });
  requestAnimationFrame(renderEdges);
}

function toggleNodePreview(nodeId) {
  state.previewHidden[nodeId] = !state.previewHidden[nodeId];
  refreshPreviewState();
}

function nodePreviewView(nodeId, compact = false) {
  const shell = Object.assign(document.createElement("section"), { className: "node-preview-shell" });
  shell.dataset.nodeId = nodeId;
  const toolbar = Object.assign(document.createElement("div"), { className: "preview-toolbar" });
  const status = Object.assign(document.createElement("small"), {
    className: "preview-state",
    textContent: state.nodeResultStale[nodeId] ? "历史预览 · 输入或上游可能已变化，需重新测试确认" : "最近一次成功输出",
  });
  const toggle = Object.assign(document.createElement("button"), {
    type: "button", className: "preview-toggle",
    textContent: state.previewHidden[nodeId] ? "显示预览" : "隐藏预览",
  });
  toggle.onclick = event => { event.stopPropagation(); toggleNodePreview(nodeId); };
  toolbar.append(status, toggle);
  const body = Object.assign(document.createElement("div"), {
    className: "preview-body", hidden: state.previewHidden[nodeId] === true,
  });
  body.append(nodeOutputView(state.nodeResults[nodeId], compact, nodeId));
  shell.append(toolbar, body);
  return shell;
}

function nodeOutputView(outputs, compact = false, nodeId = null) {
  const container = document.createElement("div");
  container.className = "node-output-view";
  const addImage = (name, url, details = "", transparent = false, outputKey = name) => {
    const figure = document.createElement("figure");
    const caption = Object.assign(document.createElement("figcaption"), { textContent: `${name} · 图像 ${details}` });
    const image = Object.assign(document.createElement("img"), { src: url, alt: outputKey });
    image.setAttribute("aria-label", name);
    image.classList.toggle("alpha-preview", transparent);
    image.title = "点击查看大图";
    image.onclick = () => { const dialog = document.querySelector("#image-dialog"); const enlarged = dialog.querySelector("img"); enlarged.src = url; enlarged.classList.toggle("alpha-preview", transparent); dialog.showModal(); };
    figure.append(caption, image); container.append(figure);
  };
  Object.entries(outputs).forEach(([name, output]) => {
    const caption = nodeId ? portCaption(nodeId, "outputs", name) : name;
    if (output.kind === "color_regions") {
      container.append(colorRegionView(output, compact));
    } else if (name === "labels" && output.kind === "image") {
      const details = Object.assign(document.createElement("details"), { className: "label-map-preview" });
      details.append(Object.assign(document.createElement("summary"), { textContent: "整数标签图 · 非二值 Mask（0=背景，1～N=区域）" }));
      details.append(nodeOutputView({ "标签值，非伪彩色": output }, compact));
      container.append(details);
    } else if (output.kind === "image") {
      addImage(caption, output.image, output.size ? `${output.size.join(" × ")} ${output.mode || ""}` : "", output.mode === "RGBA", name);
      if (output.path && !compact) container.append(Object.assign(document.createElement("pre"), { textContent: output.path }));
    }
    else if (output.kind === "sample") {
      const sampleNames = { target: "rgb_image · 目标原图", source: "masked_rgb_image · 涂灰图", mask: "mask · 编辑区域", reference: "reference_rgb_image · RGB 参考图" };
      Object.entries(output.images).forEach(([field, image]) => addImage(sampleNames[field] || field, image, "", false, field));
      container.append(Object.assign(document.createElement("pre"), { textContent: `${name}.prompt · 文本\n${output.prompt}` }));
    } else if (output.kind === "object") {
      const section = Object.assign(document.createElement("section"), { className: "composite-output" });
      section.append(Object.assign(document.createElement("strong"), { textContent: caption }), nodeOutputView(output.items, compact));
      container.append(section);
    } else container.append(Object.assign(document.createElement("pre"), { textContent: `${caption} · 文本\n${typeof output.value === "string" ? output.value : JSON.stringify(output.value, null, 2)}` }));
  });
  return container;
}

function recordPayload() {
  return { target_path: document.querySelector("#target-path").value, mask_path: document.querySelector("#mask-path").value, object_name: document.querySelector("#object-name").value };
}

function renderBatch() {
  document.querySelectorAll(".node-delete-button").forEach(button => { button.disabled = nodeDeletionDisabled(); });
  document.querySelectorAll(".swatch-topology-control").forEach(control => { control.disabled = nodeDeletionDisabled(); });
  document.querySelectorAll(".save-item-add, .save-item-remove, .save-item-name").forEach(control => { control.disabled = nodeDeletionDisabled(); });
  document.querySelectorAll(".node-batch-button").forEach(button => { button.disabled = loopUI.busy; });
  document.querySelectorAll('.node[data-node-id] .node-preview-button').forEach(button => {
    const spec = state.workflow?.nodes[button.closest(".node").dataset.nodeId];
    if (spec?.class_type === "SaveImagesNode") {
      button.disabled = nodeDeletionDisabled();
      button.textContent = conditionalEnabled() ? "条件保存当前张 · 查看计划" : "仅保存当前张 · 不循环";
    }
  });
  const job = batch.job;
  const labels = { running: "运行中", cancelling: "正在停止", completed: "已完成", cancelled: "已停止", failed: "失败" };
  document.querySelector("#batch-status").textContent = batch.starting ? "正在提交批量任务…" : batch.restoring ? "正在恢复最近任务…" : job ? `${labels[job.status] || job.status} · ${job.id}` : "暂无批量任务";
  const progress = document.querySelector("#batch-progress");
  progress.max = Math.max(1, job?.total || 0);
  progress.value = job?.processed || 0;
  document.querySelector("#batch-counts").textContent = job ? `已处理 ${job.processed} / ${job.total} · 成功 ${job.succeeded} · 失败 ${job.failed} · 跳过 ${job.skipped || 0}` : "已处理 0 / 0";
  document.querySelector("#batch-current").textContent = job ? `当前文件：${job.current_path || "—"}\n当前节点：${job.current_node || "—"}\n出错后继续：${job.continue_on_error ? "是" : "否"}${job.current_loop_id ? `\n当前循环：${job.current_loop_index || 1} / ${job.current_loop_count || 1} · ${job.current_loop_name || job.current_loop_id} (${job.current_loop_id})` : ""}${job.max_iterations != null ? `\n条件尝试：${job.current_iteration || 0} / ${job.max_iterations}` : ""}${job.stop_reason ? `\n停止原因：${job.stop_reason}` : ""}` : "";
  const attempts = document.querySelector("#conditional-attempts");
  attempts.replaceChildren();
  (job?.iteration_history || []).forEach(attempt => {
    const item = document.createElement("li");
    item.textContent = `${attempt.loop_id ? `${attempt.loop_name || attempt.loop_id} (${attempt.loop_id}) · ` : ""}第 ${attempt.iteration} 次 · ${attempt.accepted ? "通过" : "未通过"} · ${attempt.elapsed_seconds ?? "—"} 秒\n种子：${JSON.stringify(attempt.seeds || {})}\n${(attempt.scores || []).map(score => `${score.node}.${score.output} = ${JSON.stringify(score.value)} · ${score.passed ? "通过" : "未通过"}`).join("\n")}`;
    attempts.append(item);
  });
  document.querySelector("#batch-error").textContent = [batch.error, job?.error ? `任务错误：${job.error}` : ""].filter(Boolean).join("\n");
  document.querySelector("#batch-cancel").disabled = !job || job.status !== "running" || batch.requesting;
  document.querySelector("#batch-refresh").disabled = batch.requesting || batch.starting;
  document.querySelector("#batch-download").disabled = !job;
  document.querySelector("#batch-continue-on-error").disabled = batchStartDisabled();
  const results = document.querySelector("#batch-results");
  results.replaceChildren();
  const entries = job?.results || [];
  entries.slice(-20).forEach(result => {
    const item = document.createElement("li");
    item.textContent = `${result.source || "当前图像"}\n${result.error != null ? `失败：${result.error}` : result.accepted === false ? "已跳过 · 未保存" : `已写入：${JSON.stringify(result.paths)}`}${result.iterations != null ? `\n尝试 ${result.iterations} 次 · ${result.accepted ? "通过" : "未通过"}` : ""}${result.stop_reason ? `\n停止原因：${result.stop_reason}` : ""}`;
    item.classList.toggle("batch-result-error", result.error != null);
    results.append(item);
  });
  document.querySelector("#batch-results-label").textContent = `文件结果（${entries.length} 条，仅显示最近 20 条；完整记录请下载报告）`;
  renderLoop();
}

async function batchRequest(operation, nodeId = null, folderTrial = false) {
  const starting = operation === "start" || operation === "start-condition";
  if (starting && nodeDeletionDisabled()) return;
  if ((operation === "start" || (operation === "start-condition" && folderTrial)) && (!loopUI.enabled || !loopUI.plan || loopScope().problem || nodeId !== loopUI.targetId || loopUI.planKey !== JSON.stringify(loopPayload()))) return;
  if (operation === "start" && conditionalEnabled() && conditionalValidation(nodeId)) return;
  if (operation === "start-condition" && (!conditionalEnabled() || conditionalValidation(nodeId) || (!folderTrial && (nodeId !== conditionalUI.targetId || !conditionalUI.plan || conditionalUI.planKey !== conditionalPlanKey())))) return;
  if (batch.requesting || (starting && batchStartDisabled())) return;
  if (operation === "cancel" && batch.job?.status !== "running") return;
  clearTimeout(batch.timer);
  batch.timer = null;
  batch.requesting = true;
  batch.starting = starting;
  batch.error = "";
  renderBatch();
  let succeeded = false;
  try {
    if (starting) {
      state.previewSession = null;
      invalidateResults();
      batch.job = await api(operation === "start-condition" ? folderTrial ? "/api/batch/preview" : "/api/conditional" : "/api/batch", { method: "POST", body: JSON.stringify({ ...workflowPayload(), record: recordPayload(), config: currentConfig(), node_id: nodeId, continue_on_error: document.querySelector("#batch-continue-on-error").checked }) });
    } else if (operation === "cancel") {
      batch.job = await api(`/api/batch/${encodeURIComponent(batch.job.id)}/cancel`, { method: "POST" });
    } else if (operation === "poll" && batch.job) {
      batch.job = await api(`/api/batch/${encodeURIComponent(batch.job.id)}`);
    } else {
      const latest = await api("/api/batch/latest");
      batch.job = latest.job;
    }
    batch.uncertain = false;
    succeeded = true;
  } catch (error) {
    batch.uncertain = true;
    const action = starting ? "提交失败或响应丢失，任务可能已启动；已锁定重复启动" : operation === "cancel" ? "停止请求未确认，任务可能仍在运行" : "无法读取任务进度，自动轮询已暂停";
    batch.error = `${action}：${error.message}。请点击“刷新 / 重试”确认服务器上的最近任务；服务重启会丢失任务记录。`;
  } finally {
    batch.requesting = false;
    batch.starting = false;
    batch.restoring = false;
    renderBatch();
    if (succeeded && batchActive()) batch.timer = setTimeout(() => batchRequest("poll"), 700);
  }
}

function downloadBatchReport() {
  if (!batch.job) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(batch.job, null, 2)], { type: "application/json" }));
  const anchor = Object.assign(document.createElement("a"), { href: url, download: "object-inpainting-batch-report.json" });
  document.body.append(anchor);
  anchor.click(); anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function previewNode(nodeId, loopTrial = false) {
  if (loopUI.busy || !state.workflow.nodes[nodeId]) return;
  const button = document.querySelector(`.node[data-node-id="${CSS.escape(nodeId)}"] .node-preview-button`);
  const saving = state.workflow.nodes[nodeId].class_type === "SaveImagesNode";
  if (saving && nodeDeletionDisabled()) return;
  if (loopTrial && (!loopUI.enabled || !loopUI.plan || loopScope().problem)) return;
  if (saving && conditionalEnabled()) {
    if (loopTrial) return batchRequest("start-condition", nodeId, true);
    // Never call the ordinary save endpoint while acceptance is enabled.
    conditionalUI.targetId = nodeId;
    renderBatch();
    const panel = document.querySelector("#conditional-panel");
    panel.open = true;
    panel.scrollIntoView({ block: "nearest" });
    return setStatus("条件重试已启用：请检查单图计划，再点击「条件运行当前张」。普通保存已阻止，避免写入未通过的结果。");
  }
  state.nodeErrors = {};
  if (loopTrial) { state.previewSession = null; invalidateResults(); }
  // Keep the last successful image visible during reruns and on failure.
  Object.keys(state.nodeResults).forEach(id => {
    if (upstreamNodes(id).includes(nodeId)) state.nodeResultStale[id] = true;
  });
  refreshPreviewState();
  const revision = state.previewRevision;
  loopUI.busy = true;
  button.disabled = true;
  button.textContent = "Running…";
  renderBatch();
  try {
    const result = await api(loopTrial ? "/api/batch/preview" : saving ? "/api/node/save-images" : "/api/node/preview", { method: "POST", body: JSON.stringify({ ...workflowPayload(), node_id: nodeId, record: recordPayload(), config: currentConfig(), preview_session: state.previewSession }) });
    if (revision !== state.previewRevision) return;
    state.previewSession = result.preview_session || null;
    const executed = new Set(result.executed_nodes);
    Object.keys(state.nodeResults).forEach(id => {
      if (upstreamNodes(id).some(ancestor => executed.has(ancestor))) state.nodeResultStale[id] = true;
    });
    state.nodeResults[nodeId] = result.outputs;
    delete state.nodeResultStale[nodeId];
    state.selected = nodeId;
    render(); setStatus(`${saving ? "保存成功" : "测试成功"}：${nodeId}。实际执行：${result.executed_nodes.join(" → ")}。${result.reused_nodes?.length ? `复用上游：${result.reused_nodes.join("、")}。` : ""}${saving ? "本次仅保存当前张，未启动循环。遍历文件夹请点击画布顶部「运行全部 · 写入磁盘」。" : "输出已按实际内容自动显示为图像或文本。"}`);
  } catch (error) {
    if (revision !== state.previewRevision) return;
    const failedId = error.nodeError?.node_id || nodeId;
    state.nodeErrors[failedId] = error.message;
    Object.keys(state.nodeResults).forEach(id => {
      if (upstreamNodes(id).includes(failedId)) state.nodeResultStale[id] = true;
    });
    state.selected = state.workflow.nodes[failedId] ? failedId : nodeId;
    render();
    setStatus(`${saving ? "保存失败" : "测试失败"}（请求节点 ${nodeId}）：${error.message}`, true);
  }
  finally { loopUI.busy = false; renderBatch(); }
}

function invalidateResults(changedNode = null) {
  state.previewRevision += 1;
  // Keep the session: the server fingerprints each node and its ancestors.
  // Changing a save path/name must not invalidate an unchanged FLUX output.
  Object.keys(state.nodeResults).forEach(id => {
    if (typeof changedNode !== "string" || upstreamNodes(id).includes(changedNode)) state.nodeResultStale[id] = true;
  });
  state.nodeErrors = {};
  document.querySelectorAll(".node-error").forEach(element => element.remove());
  document.querySelectorAll(".node-failed").forEach(element => element.classList.remove("node-failed"));
  refreshPreviewState();
  renderLoop();
  requestAnimationFrame(renderEdges);
}

function resetPreviewState() {
  state.previewSession = null;
  state.nodeResults = {};
  state.nodeResultStale = {};
  state.previewHidden = {};
  document.querySelector("#preview-grid").replaceChildren();
  invalidateResults();
}

function parseLiteral(value) { try { return JSON.parse(value); } catch { return value; } }

function nodeDeleteButton(nodeId, compact = false) {
  const button = Object.assign(document.createElement("button"), {
    type: "button",
    className: `node-delete-button${compact ? " node-delete-compact" : ""}`,
    textContent: compact ? "×" : "删除此节点",
    title: `删除节点 ${nodeId}，并断开相关连线（Delete）`,
    disabled: nodeDeletionDisabled(),
  });
  button.setAttribute("aria-label", `删除节点 ${nodeId}`);
  button.onclick = event => { event.stopPropagation(); deleteNode(nodeId); };
  return button;
}

function deleteNode(nodeId) {
  if (!state.workflow?.nodes[nodeId]) return;
  if (nodeDeletionDisabled()) return setStatus("运行 / 检查任务期间不能删除节点；请等待完成或先停止任务。", true);
  const removed = removeWorkflowNode(state.workflow, nodeId, state.catalog);
  delete state.layout[nodeId];
  delete state.expanded[nodeId];
  delete state.nodeResults[nodeId];
  delete state.nodeResultStale[nodeId];
  delete state.previewHidden[nodeId];
  if (state.selected === nodeId) state.selected = null;
  state.connecting = null;
  clearTimeout(loopUI.timer);
  loopUI.plan = null;
  loopUI.planKey = null;
  loopUI.error = "";
  if (loopUI.sourceId === nodeId) {
    loopUI.sourceId = null;
    loopUI.enabled = false;
  }
  if (loopUI.targetId === nodeId && !conditionalEnabled()) loopUI.targetId = null;
  invalidateResults();
  render();
  document.querySelector("#connection-hint").textContent = "Select an output port to begin a connection.";
  const affected = removed.disconnected.map(item => `${item.node_id}.${item.port}`);
  setStatus(`已删除节点 ${nodeId}。${affected.length ? `已断开下游输入：${affected.join("、")}；请重新连接或填写。` : ""}${removed.removedOutputs.length ? `已移除工作流输出：${removed.removedOutputs.join("、")}。` : ""}未删除磁盘文件。`);
}

document.addEventListener("keydown", event => {
  if (event.key === "Escape" && conditionalUI.picking) { event.preventDefault(); cancelConditionalPick(); return; }
  if (event.key !== "Delete" || event.repeat || event.isComposing || event.ctrlKey || event.metaKey || event.altKey || event.shiftKey) return;
  const target = event.target;
  if (target instanceof Element && target.closest('input, textarea, select, button, [contenteditable]:not([contenteditable="false"]), [role="textbox"], dialog')) return;
  if (!state.selected || !state.workflow?.nodes[state.selected]) return;
  event.preventDefault();
  deleteNode(state.selected);
});

function addNode(classType) {
  const base = classType.replace(/Node$/, "").replace(/[^A-Za-z0-9_]/g, "_").toLowerCase();
  let id = base;
  let number = 2;
  while (state.workflow.nodes[id]) id = `${base}_${number++}`;
  const defaults = classType === "FolderImagesNode" ? { dir: "", recursive: 0, index: 0, ...state.catalog[classType].input_defaults } : state.catalog[classType].input_defaults;
  const inputs = Object.fromEntries(Object.keys(state.catalog[classType].inputs).map(name => [name, defaults?.[name] ?? (name === "config" || name === "record" ? context(name) : name === "object_name" || (classType === "SAM3TextMaskNode" && name === "text") ? "cup" : null)]));
  state.workflow.nodes[id] = { class_type: classType, inputs };
  if (classType === "ColorRegionSwatchesNode") state.workflow.nodes[id].inputs.indices = structuredClone(defaults?.indices ?? [0]);
  if (classType === "SaveImagesNode") saveItems(state.workflow.nodes[id]);
  state.layout[id] = { x: 140 + Object.keys(state.workflow.nodes).length * 15, y: 120 + Object.keys(state.workflow.nodes).length * 15 };
  state.selected = id;
  render();
}

function renderCatalog() {
  const catalog = document.querySelector("#catalog");
  Object.keys(state.catalog).sort().forEach(classType => {
    const button = document.createElement("button");
    button.className = "catalog-button";
    button.textContent = `+ ${state.catalog[classType].help.title}`;
    button.title = `${classType}\n${state.catalog[classType].help.summary}`;
    button.onclick = () => addNode(classType);
    catalog.append(button);
  });
}

function startPractice(classType = null) {
  state.workflow = { nodes: {}, outputs: {} };
  restoreLoop();
  state.expanded = {};
  state.layout = {}; state.selected = null; state.connecting = null;
  resetPreviewState();
  if (classType) {
    addNode(classType);
    state.layout[state.selected] = { x: 50, y: 50 };
  }
  render();
  setStatus(classType ? `单节点练习：${state.catalog[classType].help.test}` : "空白画布：从左侧添加一个节点开始。可随时下载并重新导入练习流程。");
}

async function restoreDefault() {
  try {
    const workflow = await api("/api/workflow/default");
    startPractice(); state.workflow = workflow; render();
    setStatus("已加载完整示例。可先查看单个节点的说明，再测试它；只会执行必要的上游。");
  } catch (error) { setStatus(error.message, true); }
}

function workflowPayload() { return { workflow: state.workflow }; }
function currentConfig() {
  return Object.fromEntries(Object.keys(state.config).map(name => [name, parseLiteral(document.querySelector(`#config-${name}`).value)]));
}

async function validate() { try { setStatus(await api("/api/workflow/validate", { method: "POST", body: JSON.stringify({ ...workflowPayload(), config: currentConfig(), require_dataset: false }) })); } catch (error) { setStatus(error.message, true); } }
async function preview() {
  try {
    const result = await api("/api/preview", { method: "POST", body: JSON.stringify({ ...workflowPayload(), record: { target_path: document.querySelector("#target-path").value, mask_path: document.querySelector("#mask-path").value, object_name: document.querySelector("#object-name").value }, config: currentConfig() }) });
    const grid = document.querySelector("#preview-grid"); grid.replaceChildren();
    ["target", "source", "mask", "reference"].forEach(name => { const figure = document.createElement("figure"); figure.innerHTML = `<figcaption>${name}</figcaption><img alt="${name}" src="${result[name]}">`; grid.append(figure); });
    setStatus({ prompt: result.prompt });
  } catch (error) { setStatus(error.message, true); }
}
async function exportDataset() {
  try {
    const result = await api("/api/export", { method: "POST", body: JSON.stringify({ ...workflowPayload(), config: currentConfig(), manifest_path: document.querySelector("#manifest-path").value, output_dir: document.querySelector("#output-dir").value, preview_count: Number(document.querySelector("#preview-count").value) }) });
    setStatus(result);
  } catch (error) { setStatus(error.message, true); }
}

function saveWorkflow() {
  const content = JSON.stringify({ version: 1, workflow: state.workflow, layout: state.layout, config: currentConfig(), loop: loopSettings() }, null, 2);
  const anchor = Object.assign(document.createElement("a"), { href: URL.createObjectURL(new Blob([content], { type: "application/json" })), download: "object-inpainting-workflow.json" });
  document.body.append(anchor);
  anchor.click(); anchor.remove();
  setTimeout(() => URL.revokeObjectURL(anchor.href), 1000);
}

async function loadWorkflow(event) {
  const file = event.target.files?.[0];
  if (!file) return;
  try {
    const saved = JSON.parse(await file.text());
    const workflow = saved?.workflow || saved;
    const validation = await api("/api/workflow/validate", { method: "POST", body: JSON.stringify({ workflow, config: saved?.config || currentConfig(), require_dataset: false }) });
    const layout = saved?.layout || {};
    if (typeof layout !== "object" || Array.isArray(layout)) throw new Error("Layout must be an object.");
    for (const point of Object.values(layout)) {
      if (!point || !Number.isFinite(point.x) || !Number.isFinite(point.y) || point.x < 0 || point.y < 0) throw new Error("Layout coordinates must be non-negative numbers.");
    }
    state.workflow = validation.workflow || workflow;
    restoreLoop(saved?.loop);
    state.expanded = {};
    state.layout = layout;
    state.config = validation.config;
    state.selected = null;
    state.connecting = null;
    resetPreviewState();
    renderConfig(); render();
    document.querySelector("#preview-grid").replaceChildren();
    setStatus(`Imported workflow: ${file.name}`);
  } catch (error) {
    setStatus(`Import failed: ${error.message}. Current workflow unchanged.`, true);
  } finally { event.target.value = ""; }
}

function renderConfig() {
  const fields = document.querySelector("#config-fields");
  fields.replaceChildren();
  Object.entries(state.config).forEach(([name, value]) => {
    const label = Object.assign(document.createElement("label"), { textContent: `${state.configHelp[name] || name} · ${name}` });
    const input = Object.assign(document.createElement("input"), { id: `config-${name}`, value });
    input.onchange = invalidateResults;
    label.append(input); fields.append(label);
  });
}

async function initialize() {
  [state.catalog, state.workflow, state.config, state.configHelp] = await Promise.all([api("/api/node-types"), api("/api/workflow/default"), api("/api/default-config"), api("/api/config-help")]);
  await api("/api/workflow/validate", { method: "POST", body: JSON.stringify({ workflow: state.workflow, config: state.config }) });
  renderCatalog();
  renderConfig();
  render(); setStatus("Ready. Load a target and mask, then run preview.");
}

document.querySelector("#validate-workflow").onclick = validate;
document.querySelector("#preview").onclick = preview;
document.querySelector("#export").onclick = exportDataset;
document.querySelector("#save-workflow").onclick = saveWorkflow;
document.querySelector("#load-workflow").onchange = loadWorkflow;
document.querySelector("#new-workflow").onclick = () => startPractice();
document.querySelector("#default-workflow").onclick = restoreDefault;
document.querySelectorAll("[data-starter]").forEach(button => { button.onclick = () => startPractice(button.dataset.starter); });
document.querySelectorAll("#target-path, #mask-path, #object-name").forEach(input => { input.onchange = invalidateResults; });
document.querySelector("#close-image-dialog").onclick = () => document.querySelector("#image-dialog").close();
document.querySelector("#batch-cancel").onclick = () => batchRequest("cancel");
document.querySelector("#batch-refresh").onclick = () => batchRequest("refresh");
document.querySelector("#batch-download").onclick = downloadBatchReport;
workspace.addEventListener("click", event => { if (event.target === workspace) { state.connecting = null; render(); } });
batchRequest("refresh");
initialize().catch(error => setStatus(error.message, true));
