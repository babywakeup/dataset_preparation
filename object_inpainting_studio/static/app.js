const state = { catalog: {}, workflow: null, layout: {}, selected: null, connecting: null, config: {}, configHelp: {}, nodeResults: {}, expanded: {} };
const workspace = document.querySelector("#workspace");
const nodesElement = document.querySelector("#nodes");
const connectionsElement = document.querySelector("#connections");
const statusElement = document.querySelector("#status");
const batch = { job: null, starting: false, restoring: true, requesting: false, uncertain: false, error: "", timer: null };
const batchActive = () => ["running", "cancelling"].includes(batch.job?.status);
const batchStartDisabled = () => batch.starting || batch.restoring || batch.requesting || batch.uncertain || batchActive();

const api = async (url, options = {}) => {
  const response = await fetch(url, { headers: { "Content-Type": "application/json" }, ...options });
  const text = await response.text();
  let data;
  try { data = JSON.parse(text); } catch { throw new Error(`HTTP ${response.status}: ${text.slice(0, 300)}`); }
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail || data));
  return data;
};

const link = (node, output) => ({ node, output });
const context = (name) => ({ context: name });
const isLink = (value) => value && typeof value === "object" && Object.keys(value).length === 2 && "node" in value && "output" in value;
const isContext = (value) => value && typeof value === "object" && Object.keys(value).length === 1 && "context" in value;
const nodeInfo = (id) => state.catalog[state.workflow.nodes[id].class_type];
const portType = (nodeId, port) => nodeInfo(nodeId).outputs[port];
const position = (id, index = 0) => state.layout[id] || { x: 50 + (index % 5) * 350, y: 50 + Math.floor(index / 5) * 680 };

function setStatus(value, isError = false) {
  statusElement.textContent = typeof value === "string" ? value : JSON.stringify(value, null, 2);
  statusElement.style.color = isError ? "#ff9f9f" : "#a8b9dd";
}

function bindingLabel(value) {
  if (isLink(value)) return `${value.node}.${value.output}`;
  if (isContext(value)) return `context: ${value.context}`;
  return JSON.stringify(value);
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
  const info = state.catalog[spec.class_type];
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
  if (spec.class_type === "FolderImagesNode") header.append(folderLoopSwitch(id));
  header.addEventListener("pointerdown", event => { if (!event.target.closest("button")) beginDrag(event, id); });
  header.addEventListener("click", event => { if (!event.target.closest("button")) { state.selected = id; render(); } });
  card.append(header);
  const summary = Object.assign(document.createElement("p"), { className: "node-summary", textContent: `${info.help.title}：${info.help.summary}` });
  card.append(summary);
  if (["FolderImagesNode", "SaveImagesNode"].includes(spec.class_type)) card.append(Object.assign(document.createElement("p"), { className: "node-path" }));
  const ports = document.createElement("div");
  ports.className = "ports";
  Object.entries(info.inputs).forEach(([name, type]) => ports.append(portRow(id, name, type, "input")));
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
      });
      if (name.endsWith("_path")) input.append(uploadPicker(path => {
        document.querySelector(`#${field}`).value = path; render();
      }, name));
      widgets.append(input);
    });
  } else {
    Object.entries(info.inputs).forEach(([name, type]) => {
      if (["PATH", "STRING", "RGB_IMAGE", "MASK", "IMAGE", "BBOX", "JSON_RECORD", "INT", "FLOAT"].includes(type)) {
        const binding = spec.inputs[name] ?? info.input_defaults?.[name];
        const field = isLink(binding) && binding.node === "record" ? ({ target_path: "target-path", mask_path: "mask-path", object_name: "object-name" })[binding.output] : null;
        const shown = field ? document.querySelector(`#${field}`).value : typeof binding === "string" ? binding : !isLink(binding) && binding != null ? JSON.stringify(binding) : "";
        const input = inputWidget(name, shown, value => { spec.inputs[name] = ["BBOX", "JSON_RECORD", "INT", "FLOAT"].includes(type) ? parseLiteral(value) : value; });
        input.title = info.help.inputs[name] || name;
        if (["PATH", "RGB_IMAGE", "MASK", "IMAGE"].includes(type) && name !== "dir") input.append(uploadPicker(path => { spec.inputs[name] = path; render(); }, name));
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
  const previewButton = Object.assign(document.createElement("button"), { textContent: saveImages ? "仅保存当前张 · 不循环" : info.side_effects ? "写入节点：仅导出时执行" : "测试此节点 · 自动预览", className: "node-preview-button", disabled: info.side_effects && !saveImages });
  previewButton.onclick = () => previewNode(id);
  widgets.append(previewButton);
  if (saveImages) {
    widgets.append(Object.assign(document.createElement("small"), { className: "save-run-hint", textContent: "上方按钮只保存当前输入，不会遍历文件夹。处理所有图片：开启「↻ 逐张」，再点击画布顶部「运行全部 · 写入磁盘」。" }));
    const batchButton = Object.assign(document.createElement("button"), { textContent: "↻ 查看逐张流程", className: "node-batch-button", disabled: loopUI.busy });
    batchButton.onclick = () => {
      const sources = upstreamNodes(id).filter(nodeId => state.workflow.nodes[nodeId].class_type === "FolderImagesNode");
      if (sources.length !== 1) return setStatus("此保存节点的上游必须恰好连接一个文件夹节点，才能开启逐张流程。", true);
      chooseLoop(sources[0], true, id);
      document.querySelector("#loop-toolbar").scrollIntoView({ block: "nearest" });
    };
    widgets.append(batchButton);
  }
  if (state.nodeResults[id]) widgets.append(nodeOutputView(state.nodeResults[id], true));
  const settings = document.createElement("details");
  settings.className = "node-settings";
  settings.open = state.expanded[id] ?? !loopUI.enabled;
  const settingsTitle = Object.assign(document.createElement("summary"), { textContent: "参数与单节点测试" });
  settingsTitle.onclick = () => { state.expanded[id] = !settings.open; requestAnimationFrame(renderEdges); };
  settings.append(settingsTitle, widgets);
  card.append(settings);
  nodesElement.append(card);
}

function inputWidget(name, value, update) {
  const label = Object.assign(document.createElement("label"), { textContent: name });
  const input = Object.assign(document.createElement("input"), { value, placeholder: `Enter ${name}` });
  input.dataset.field = name;
  const apply = () => { update(input.value); invalidateResults(); requestAnimationFrame(renderEdges); };
  input.onchange = apply;
  input.onkeydown = event => { if (event.key === "Enter") { apply(); input.blur(); } };
  label.append(input);
  return label;
}

function uploadPicker(update, name) {
  const label = Object.assign(document.createElement("span"), { className: "button file-picker", textContent: "选择图片…", tabIndex: 0, role: "button" });
  const input = Object.assign(document.createElement("input"), { type: "file", accept: "image/*", hidden: true });
  input.dataset.upload = name;
  input.onchange = async () => {
    if (!input.files?.[0]) return;
    try {
      const data = new FormData(); data.append("file", input.files[0]);
      const result = await api("/api/upload", { method: "POST", headers: {}, body: data });
      invalidateResults(); update(result.path); setStatus("图片已上传，点击“测试此节点”查看输出。上传文件保留到服务停止。");
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
  port.className = `port ${direction}${state.connecting?.node === nodeId && state.connecting?.output === name ? " pending" : ""}`;
  port.title = `${direction}: ${name} (${type})`;
  port.dataset.nodeId = nodeId;
  port.dataset.port = name;
  port.dataset.direction = direction;
  port.addEventListener("click", event => { event.stopPropagation(); direction === "output" ? beginConnection(nodeId, name) : completeConnection(nodeId, name); });
  const label = document.createElement("span");
  label.textContent = name;
  const typeLabel = document.createElement("span");
  typeLabel.className = "port-type";
  typeLabel.textContent = type;
  if (direction === "input") row.append(port, label, typeLabel); else row.append(typeLabel, label, port);
  return row;
}

function beginConnection(nodeId, output) {
  state.connecting = { node: nodeId, output };
  document.querySelector("#connection-hint").textContent = `Connecting ${nodeId}.${output}; choose a matching input port.`;
  render();
}

function completeConnection(nodeId, input) {
  if (!state.connecting) {
    state.selected = nodeId;
    render();
    const block = Array.from(document.querySelectorAll("#node-fields .binding")).find(element => element.querySelector("strong").textContent === input);
    block?.querySelector("input")?.focus();
    return;
  }
  const expected = nodeInfo(nodeId).inputs[input];
  const actual = portType(state.connecting.node, state.connecting.output);
  if (expected !== actual && !(expected === "IMAGE" && ["RGB_IMAGE", "RGBA_IMAGE", "MASK", "LABEL_MAP"].includes(actual))) return setStatus(`Cannot connect ${actual} to ${expected}.`, true);
  if (state.connecting.node === nodeId) return setStatus("A node cannot connect to itself.", true);
  if (upstreamNodes(state.connecting.node).includes(nodeId)) return setStatus("这条连接会形成回路。请通过文件夹上的「↻ 逐张」表达循环，不需要回流连线。", true);
  state.workflow.nodes[nodeId].inputs[input] = link(state.connecting.node, state.connecting.output);
  invalidateResults();
  state.connecting = null;
  document.querySelector("#connection-hint").textContent = "Connection added.";
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
  if (state.selected && state.nodeResults[state.selected]) preview.append(nodeOutputView(state.nodeResults[state.selected]));
  if (!state.selected) return fields.append(Object.assign(document.createElement("p"), { textContent: "Select a node to edit its input bindings." }));
  const spec = state.workflow.nodes[state.selected];
  const info = state.catalog[spec.class_type];
  help.append(Object.assign(document.createElement("h3"), { textContent: info.help.title }));
  help.append(Object.assign(document.createElement("p"), { textContent: info.help.summary }));
  help.append(Object.assign(document.createElement("p"), { className: "test-guide", textContent: `怎么单独测试：${info.help.test}` }));
  for (const [kind, ports] of [["输入", info.help.inputs], ["输出", info.help.outputs]]) {
    help.append(Object.assign(document.createElement("strong"), { textContent: kind }));
    const list = document.createElement("dl");
    Object.entries(ports).forEach(([name, text]) => {
      list.append(Object.assign(document.createElement("dt"), { textContent: name }), Object.assign(document.createElement("dd"), { textContent: text }));
    });
    help.append(list);
  }
  if (info.help.parameters.length) help.append(Object.assign(document.createElement("p"), { textContent: `相关参数（右侧下方全局共享）：${info.help.parameters.map(name => `${state.configHelp[name]} (${name})`).join("；")}` }));
  fields.append(Object.assign(document.createElement("h3"), { textContent: state.selected }));
  Object.entries(info.inputs).forEach(([name, type]) => {
    const block = document.createElement("div");
    block.className = "binding";
    const title = Object.assign(document.createElement("strong"), { textContent: name });
    const description = Object.assign(document.createElement("small"), { textContent: `${type} · ${bindingLabel(spec.inputs[name])}` });
    block.append(title, description);
    const mode = document.createElement("select");
    [["context", "Context"], ["literal", "Literal"], ["link", "Link via output port"]].forEach(([value, text]) => mode.add(new Option(text, value)));
    mode.value = isLink(spec.inputs[name]) ? "link" : isContext(spec.inputs[name]) ? "context" : "literal";
    const value = document.createElement("input");
    value.value = isContext(spec.inputs[name]) ? spec.inputs[name].context : isLink(spec.inputs[name]) ? bindingLabel(spec.inputs[name]) : JSON.stringify(spec.inputs[name]) ?? "";
    const actions = document.createElement("div");
    actions.className = "binding-actions";
    const apply = document.createElement("button");
    apply.textContent = "Apply";
    apply.onclick = () => {
      if (mode.value === "context") {
        if (!value.value.trim()) return setStatus("Context name cannot be empty.", true);
        spec.inputs[name] = context(value.value.trim());
      }
      if (mode.value === "literal") spec.inputs[name] = parseLiteral(value.value);
      if (mode.value === "link") return setStatus("Click an output port, then this input port to create a link.");
      invalidateResults(); render();
    };
    const remove = document.createElement("button");
    remove.textContent = "Disconnect";
    remove.onclick = () => { spec.inputs[name] = null; invalidateResults(); render(); };
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

function nodeOutputView(outputs, compact = false) {
  const container = document.createElement("div");
  container.className = "node-output-view";
  const addImage = (name, url, details = "", transparent = false) => {
    const figure = document.createElement("figure");
    const caption = Object.assign(document.createElement("figcaption"), { textContent: `${name} · 图像 ${details}` });
    const image = Object.assign(document.createElement("img"), { src: url, alt: name });
    image.classList.toggle("alpha-preview", transparent);
    image.title = "点击查看大图";
    image.onclick = () => { const dialog = document.querySelector("#image-dialog"); const enlarged = dialog.querySelector("img"); enlarged.src = url; enlarged.classList.toggle("alpha-preview", transparent); dialog.showModal(); };
    figure.append(caption, image); container.append(figure);
  };
  Object.entries(outputs).forEach(([name, output]) => {
    if (output.kind === "color_regions") {
      container.append(colorRegionView(output, compact));
    } else if (name === "labels" && output.kind === "image") {
      const details = Object.assign(document.createElement("details"), { className: "label-map-preview" });
      details.append(Object.assign(document.createElement("summary"), { textContent: "整数标签图 · 非二值 Mask（0=背景，1～N=区域）" }));
      details.append(nodeOutputView({ "标签值，非伪彩色": output }, compact));
      container.append(details);
    } else if (output.kind === "image") {
      addImage(name, output.image, output.size ? `${output.size.join(" × ")} ${output.mode || ""}` : "", output.mode === "RGBA");
      if (output.path && !compact) container.append(Object.assign(document.createElement("pre"), { textContent: output.path }));
    }
    else if (output.kind === "sample") {
      Object.entries(output.images).forEach(([field, image]) => addImage(field, image));
      container.append(Object.assign(document.createElement("pre"), { textContent: `${name}.prompt · 文本\n${output.prompt}` }));
    } else if (output.kind === "object") {
      const section = Object.assign(document.createElement("section"), { className: "composite-output" });
      section.append(Object.assign(document.createElement("strong"), { textContent: name }), nodeOutputView(output.items, compact));
      container.append(section);
    } else container.append(Object.assign(document.createElement("pre"), { textContent: `${name} · 文本\n${typeof output.value === "string" ? output.value : JSON.stringify(output.value, null, 2)}` }));
  });
  return container;
}

function recordPayload() {
  return { target_path: document.querySelector("#target-path").value, mask_path: document.querySelector("#mask-path").value, object_name: document.querySelector("#object-name").value };
}

function renderBatch() {
  document.querySelectorAll(".node-batch-button").forEach(button => { button.disabled = loopUI.busy; });
  document.querySelectorAll('.node[data-node-id] .node-preview-button').forEach(button => {
    const spec = state.workflow?.nodes[button.closest(".node").dataset.nodeId];
    if (spec?.class_type === "SaveImagesNode") button.disabled = batchStartDisabled() || loopUI.busy;
  });
  const job = batch.job;
  const labels = { running: "运行中", cancelling: "正在停止", completed: "已完成", cancelled: "已停止", failed: "失败" };
  document.querySelector("#batch-status").textContent = batch.starting ? "正在提交批量任务…" : batch.restoring ? "正在恢复最近任务…" : job ? `${labels[job.status] || job.status} · ${job.id}` : "暂无批量任务";
  const progress = document.querySelector("#batch-progress");
  progress.max = Math.max(1, job?.total || 0);
  progress.value = job?.processed || 0;
  document.querySelector("#batch-counts").textContent = job ? `已处理 ${job.processed} / ${job.total} · 成功 ${job.succeeded} · 失败 ${job.failed}` : "已处理 0 / 0";
  document.querySelector("#batch-current").textContent = job ? `当前文件：${job.current_path || "—"}\n当前节点：${job.current_node || "—"}\n出错后继续：${job.continue_on_error ? "是" : "否"}` : "";
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
    item.textContent = `${result.source}\n${result.error != null ? `失败：${result.error}` : `已写入：${JSON.stringify(result.paths)}`}`;
    item.classList.toggle("batch-result-error", result.error != null);
    results.append(item);
  });
  document.querySelector("#batch-results-label").textContent = `文件结果（${entries.length} 条，仅显示最近 20 条；完整记录请下载报告）`;
  renderLoop();
}

async function batchRequest(operation, nodeId = null) {
  if (operation === "start" && (loopUI.busy || !loopUI.enabled || !loopUI.plan || loopScope().problem || nodeId !== loopUI.targetId)) return;
  if (batch.requesting || (operation === "start" && batchStartDisabled())) return;
  if (operation === "cancel" && batch.job?.status !== "running") return;
  clearTimeout(batch.timer);
  batch.timer = null;
  batch.requesting = true;
  batch.starting = operation === "start";
  batch.error = "";
  renderBatch();
  let succeeded = false;
  try {
    if (operation === "start") {
      batch.job = await api("/api/batch", { method: "POST", body: JSON.stringify({ ...workflowPayload(), record: recordPayload(), config: currentConfig(), node_id: nodeId, continue_on_error: document.querySelector("#batch-continue-on-error").checked }) });
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
    const action = operation === "start" ? "提交失败或响应丢失，任务可能已启动；已锁定重复启动" : operation === "cancel" ? "停止请求未确认，任务可能仍在运行" : "无法读取任务进度，自动轮询已暂停";
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
  if (saving && batchStartDisabled()) return;
  if (loopTrial && (!loopUI.enabled || !loopUI.plan || loopScope().problem)) return;
  loopUI.busy = true;
  button.disabled = true;
  button.textContent = "Running…";
  renderBatch();
  try {
    const result = await api(loopTrial ? "/api/batch/preview" : saving ? "/api/node/save-images" : "/api/node/preview", { method: "POST", body: JSON.stringify({ ...workflowPayload(), node_id: nodeId, record: recordPayload(), config: currentConfig() }) });
    state.nodeResults[nodeId] = result.outputs;
    state.selected = nodeId;
    render(); setStatus(`${saving ? "保存成功" : "测试成功"}：${nodeId}。实际执行：${result.executed_nodes.join(" → ")}。${saving ? "本次仅保存当前张，未启动循环。遍历文件夹请点击画布顶部「运行全部 · 写入磁盘」。" : "输出已按实际内容自动显示为图像或文本。"}`);
  } catch (error) { setStatus(`${saving ? "保存失败" : "测试失败"}：${error.message}`, true); button.disabled = false; button.textContent = saving ? "仅保存当前张 · 不循环" : "测试此节点 · 自动预览"; }
  finally { loopUI.busy = false; renderBatch(); }
}

function invalidateResults() {
  state.nodeResults = {};
  document.querySelectorAll(".node-output-view").forEach(element => element.remove());
  document.querySelector("#preview-grid").replaceChildren();
  renderLoop();
  requestAnimationFrame(renderEdges);
}

function parseLiteral(value) { try { return JSON.parse(value); } catch { return value; } }

function addNode(classType) {
  const base = classType.replace(/Node$/, "").replace(/[^A-Za-z0-9_]/g, "_").toLowerCase();
  let id = base;
  let number = 2;
  while (state.workflow.nodes[id]) id = `${base}_${number++}`;
  const defaults = classType === "FolderImagesNode" ? { dir: "", recursive: 0, index: 0, ...state.catalog[classType].input_defaults } : state.catalog[classType].input_defaults;
  const inputs = Object.fromEntries(Object.keys(state.catalog[classType].inputs).map(name => [name, defaults?.[name] ?? (name === "config" || name === "record" ? context(name) : name === "object_name" || (classType === "SAM3TextMaskNode" && name === "text") ? "cup" : null)]));
  state.workflow.nodes[id] = { class_type: classType, inputs };
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
  invalidateResults();
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
    state.workflow = workflow;
    restoreLoop(saved?.loop);
    state.expanded = {};
    state.layout = layout;
    state.config = validation.config;
    state.selected = null;
    state.connecting = null;
    state.nodeResults = {};
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