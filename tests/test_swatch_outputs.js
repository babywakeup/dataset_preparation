const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const plain = value => JSON.parse(JSON.stringify(value));
const walk = element => [element, ...element.children.flatMap(walk)];
const text = element => walk(element).map(item => item.textContent || "").join(" ");

// Exercise the real card, inspector, connection, preview and import/export controllers.
function controller() {
  class Element {
    constructor(tag = "div") {
      this.tagName = tag;
      this.children = [];
      this.style = { setProperty(name, value) { this[name] = value; } };
      this.dataset = {};
      this.attributes = {};
      this.className = "";
      this.value = "";
      this.classList = {
        add: value => { this.className += ` ${value}`; },
        remove: value => { this.className = this.className.split(" ").filter(item => item !== value).join(" "); },
        toggle: (value, force) => { if (force) this.classList.add(value); else this.classList.remove(value); },
      };
    }
    append(...children) { this.children.push(...children); }
    add(child) { this.append(child); }
    replaceChildren(...children) { this.children = children; }
    setAttribute(name, value) { this.attributes[name] = value; }
    addEventListener(name, callback) { this[`on${name}`] = callback; }
    querySelector(selector) {
      return walk(this).find(item => selector.startsWith(".") ? item.className.split(" ").includes(selector.slice(1)) : item.tagName === selector);
    }
    blur() {}
    click() {}
    remove() {}
    focus() {}
  }
  const elements = new Map();
  const document = {
    body: new Element("body"),
    createElement: tag => new Element(tag),
    createElementNS: (_, tag) => new Element(tag),
    querySelector(selector) {
      if (!elements.has(selector)) elements.set(selector, new Element());
      return elements.get(selector);
    },
    querySelectorAll(selector) {
      const all = ["#nodes", "#node-fields", "#node-preview"].flatMap(id => walk(document.querySelector(id)));
      if (selector.startsWith(".") && !selector.includes(" ")) return all.filter(item => selector.split(",").some(part => item.className.split(" ").includes(part.slice(1))));
      return [];
    },
    addEventListener() {},
  };
  let blob;
  const timers = new Map();
  let nextTimer = 0;
  const sandbox = vm.createContext({
    document, Element, structuredClone, Blob,
    CSS: { escape: value => value },
    URL: { createObjectURL(value) { blob = value; return "blob:swatches"; }, revokeObjectURL() {} },
    setTimeout(callback) { timers.set(++nextTimer, callback); return nextTimer; },
    clearTimeout(id) { timers.delete(id); },
    requestAnimationFrame() {},
    fetch: () => new Promise(() => {}),
    Option: function (caption, value) { const option = new Element("option"); option.textContent = caption; option.value = value; return option; },
  });
  const run = code => vm.runInContext(code, sandbox);
  for (const file of ["node_editing.js", "loop.js", "multiple_loops.js", "app.js"]) run(fs.readFileSync(path.join(__dirname, "../object_inpainting_studio/static", file), "utf8"));
  run(`
    // Startup requests intentionally never resolve: reset every idle-state flag.
    batch.requesting = false; batch.restoring = false;
    renderLoop = () => {};
    const actualRender = render;
    globalThis.renderCount = 0;
    render = () => { renderCount++; actualRender(); };
    const info = (inputs, outputs, defaults = {}) => ({
      inputs, outputs, input_defaults: defaults,
      help: {title: 'test', summary: 'test', test: 'test', inputs, outputs, parameters: []},
    });
    state.catalog = {
      ColorRegionSwatchesNode: {...info({regions: 'COLOR_REGIONS', indices: 'INT_LIST'}, {img1: 'RGB_IMAGE', colors: 'JSON_RECORD'}, {indices: [0]}), dynamic_image_outputs: true},
      Target: info({first: 'IMAGE', second: 'RGB_IMAGE', third: 'RGB_IMAGE', list: 'IMAGE_LIST'}, {image: 'RGB_IMAGE'}, {second: [4, 5]}),
      Source: info({}, {indices: 'INT_LIST', image: 'RGB_IMAGE'}),
    };
    state.workflow = {nodes: {
      swatches: {class_type: 'ColorRegionSwatchesNode', inputs: {regions: null, indices: [0, 2, 1]}},
      target: {class_type: 'Target', inputs: {first: link('swatches', 'img1'), second: link('swatches', 'img2'), third: link('swatches', 'img3'), list: null}},
      tail: {class_type: 'Target', inputs: {first: link('target', 'image')}},
      source: {class_type: 'Source', inputs: {}},
    }, outputs: {first: link('swatches', 'img1'), third: link('swatches', 'img3'), colors: link('swatches', 'colors'), unrelated: link('source', 'image')}};
    state.selected = 'swatches';
    state.config = {};
  `);
  return { run, document, blob: () => blob };
}

function cardIndices(ui) {
  ui.run("render()");
  return walk(ui.document.querySelector("#nodes")).find(item => item.dataset.field === "indices");
}

function inspector(ui) {
  ui.run("renderInspector()");
  const block = ui.document.querySelector("#node-fields").children.find(item => item.dataset.port === "indices");
  return {
    mode: block.querySelector("select"), value: block.querySelector("input"),
    apply: walk(block).find(item => item.textContent === "Apply"),
    disconnect: walk(block).find(item => item.textContent === "Disconnect"),
  };
}

test("nodeInfo derives positional RGB outputs, labels and help without changing the catalog", () => {
  const ui = controller();
  assert.deepEqual(plain(ui.run("nodeInfo('swatches').outputs")), {img1: "RGB_IMAGE", img2: "RGB_IMAGE", img3: "RGB_IMAGE", colors: "JSON_RECORD"});
  assert.equal(ui.run("portType('swatches', 'img3')"), "RGB_IMAGE");
  assert.equal(ui.run("portCaption('swatches', 'outputs', 'img2')"), "img2 · index=2");
  assert.deepEqual(plain(ui.run("Object.keys(nodeInfo('swatches').help.outputs)")), ["img1", "img2", "img3", "colors"]);
  assert.deepEqual(plain(ui.run("state.catalog.ColorRegionSwatchesNode.outputs")), {img1: "RGB_IMAGE", colors: "JSON_RECORD"});
  assert.equal(ui.run("Object.hasOwn(state.workflow.nodes.swatches, 'legacy_swatch_images')"), false);
  ui.run("render()");
  const nodes = ui.document.querySelector("#nodes");
  assert.match(text(nodes), /img3 · index=1/);
  assert.match(text(ui.document.querySelector("#node-help")), /img2 · index=2/);
  assert.equal(walk(nodes).filter(item => item.dataset.direction === "output" && item.dataset.nodeId === "swatches" && item.dataset.port === "img3").length, 1);
});

test("new nodes use independent default outputs and do not share the catalog indices array", () => {
  const ui = controller();
  ui.run("addNode('ColorRegionSwatchesNode')");
  assert.deepEqual(plain(ui.run("nodeInfo(state.selected).outputs")), {img1: "RGB_IMAGE", colors: "JSON_RECORD"});
  assert.deepEqual(plain(ui.run("state.workflow.nodes[state.selected].inputs.indices")), [0]);
  assert.equal(ui.run("Object.hasOwn(state.workflow.nodes[state.selected], 'legacy_swatch_images')"), false);
  ui.run("state.workflow.nodes[state.selected].inputs.indices.push(2)");
  assert.deepEqual(plain(ui.run("state.catalog.ColorRegionSwatchesNode.input_defaults.indices")), [0]);
  ui.run("delete state.catalog.ColorRegionSwatchesNode.input_defaults.indices; addNode('ColorRegionSwatchesNode')");
  assert.deepEqual(plain(ui.run("state.workflow.nodes[state.selected].inputs.indices")), [0]);
});

test("card expansion immediately rerenders ports, preserves edges and has no fixed slot ceiling", () => {
  const ui = controller();
  const input = cardIndices(ui);
  const before = ui.run("renderCount");
  input.value = JSON.stringify(Array.from({length: 40}, (_, index) => index));
  input.onchange();
  assert.equal(ui.run("renderCount"), before + 1);
  assert.equal(ui.run("portType('swatches', 'img40')"), "RGB_IMAGE");
  assert.deepEqual(plain(ui.run("state.workflow.nodes.target.inputs.third")), {node: "swatches", output: "img3"});
  assert.match(text(ui.document.querySelector("#nodes")), /img40 · index=39/);
});

test("inspector shrink cleans only removed output links and clones defaults; stale conditions remain explicit", () => {
  const ui = controller();
  ui.run(`
    state.workflow.nodes.target.inputs.object = {node: 'swatches', output: 'img3', literal: true};
    state.workflow.nodes.target.inputs.context = {context: 'img3'};
    state.workflow.conditional_loop = {start_nodes: ['swatches'], seed_nodes: [], conditions: [{node: 'swatches', output: 'img3', operator: '==', value: 1}], mode: 'all', max_iterations: 2, timeout_seconds: 0, on_exhausted: 'fail'};
    state.connecting = link('swatches', 'img3');
    loopUI.plan = {old: true}; loopUI.planKey = 'old'; loopUI.timer = setTimeout(() => {}, 250);
    conditionalUI.plan = {old: true}; conditionalUI.planKey = 'old';
    state.previewSession = 'keep-session';
    state.nodeResults = {swatches: {colors: {kind: 'text', value: 'old'}}, target: {}, tail: {}, source: {}};
    state.previewHidden.swatches = true;
  `);
  const controls = inspector(ui);
  assert.equal(controls.mode.value, "literal");
  assert.equal(controls.mode.children.length, 1);
  assert.equal(controls.disconnect.disabled, true);
  controls.value.value = "[2]";
  controls.apply.onclick();
  assert.deepEqual(plain(ui.run("state.workflow.nodes.target.inputs.first")), {node: "swatches", output: "img1"});
  assert.deepEqual(plain(ui.run("state.workflow.nodes.target.inputs.second")), [4, 5]);
  assert.equal(ui.run("state.workflow.nodes.target.inputs.third"), null);
  ui.run("state.workflow.nodes.target.inputs.second.push(6)");
  assert.deepEqual(plain(ui.run("state.catalog.Target.input_defaults.second")), [4, 5]);
  assert.equal(ui.run("state.workflow.nodes.target.inputs.object.literal"), true);
  assert.deepEqual(plain(ui.run("state.workflow.nodes.target.inputs.context")), {context: "img3"});
  assert.deepEqual(plain(ui.run("Object.keys(state.workflow.outputs)")), ["first", "colors", "unrelated"]);
  assert.equal(ui.run("state.connecting"), null);
  assert.equal(ui.run("loopUI.plan"), null);
  assert.equal(ui.run("loopUI.planKey"), null);
  assert.equal(ui.run("conditionalUI.plan"), null);
  assert.equal(ui.run("conditionalUI.planKey"), null);
  assert.equal(ui.run("state.workflow.conditional_loop.conditions[0].output"), "img3");
  assert.match(ui.run("conditionalValidation()"), /swatches.img3/);
  assert.equal(ui.run("state.previewSession"), "keep-session");
  assert.equal(ui.run("state.previewHidden.swatches"), true);
  assert.deepEqual(plain(ui.run("Object.keys(state.nodeResults)")), ["swatches", "target", "tail", "source"]);
  assert.deepEqual(plain(ui.run("state.nodeResultStale")), {swatches: true, target: true, tail: true});
});

test("reorder and duplicates change labels by position without changing surviving connections", () => {
  const ui = controller();
  const edges = ui.run("JSON.stringify([state.workflow.nodes.target.inputs, state.workflow.outputs])");
  ui.run("state.connecting = link('swatches', 'img2')");
  const input = cardIndices(ui);
  input.value = "[1,0,1]";
  input.onkeydown({key: "Enter"});
  assert.deepEqual(plain(ui.run("state.workflow.nodes.swatches.inputs.indices")), [1, 0, 1]);
  assert.equal(ui.run("JSON.stringify([state.workflow.nodes.target.inputs, state.workflow.outputs])"), edges);
  assert.equal(ui.run("portCaption('swatches', 'outputs', 'img1')"), "img1 · index=1");
  assert.equal(ui.run("portCaption('swatches', 'outputs', 'img2')"), "img2 · index=0");
  assert.equal(ui.run("state.connecting.output"), "img2");
  assert.match(ui.document.querySelector("#status").textContent, /重排会改变内容/);
});

test("invalid card and inspector edits leave graph, links, previews, revision and plans untouched", () => {
  for (const invalid of ["", "[]", "[true]", "[false]", "[-1]", "[0.1]", "[null]", '["0"]', "0", "null", "[0,]", "[NaN]", "[Infinity]", "[9007199254740992]", "[0]; alert(1)", '{"context":"indices"}', '{"node":"source","output":"indices"}']) {
    for (const entry of ["card", "inspector"]) {
      const ui = controller();
      const input = entry === "card" ? cardIndices(ui) : inspector(ui);
      ui.run("state.previewSession = 'stable'; state.nodeResults.swatches = {}; loopUI.planKey = 'stable'; conditionalUI.planKey = 'stable';");
      const before = ui.run("JSON.stringify([state.workflow, state.nodeResults, state.nodeResultStale, state.previewRevision, loopUI.planKey, conditionalUI.planKey])");
      if (entry === "card") { input.value = invalid; input.onchange(); }
      else { input.value.value = invalid; input.apply.onclick(); }
      assert.equal(ui.run("JSON.stringify([state.workflow, state.nodeResults, state.nodeResultStale, state.previewRevision, loopUI.planKey, conditionalUI.planKey])"), before, `${entry}: ${invalid}`);
      assert.equal(ui.run("state.previewSession"), "stable");
      assert.match(ui.document.querySelector("#status").textContent, /非空.*非负整数/);
    }
  }
});

test("preview, task submission, polling uncertainty and both plan checks guard all topology edits", () => {
  for (const setup of ["loopUI.busy = true", "loopUI.checking = true", "conditionalUI.checking = true", "batch.starting = true", "batch.requesting = true", "batch.restoring = true", "batch.uncertain = true", "batch.job = {status: 'running'}", "batch.job = {status: 'cancelling'}"]) {
    const ui = controller();
    const card = cardIndices(ui);
    const controls = inspector(ui);
    const before = ui.run("JSON.stringify(state.workflow)");
    ui.run(setup);
    ui.run("renderBatch()");
    assert.equal(card.disabled, true, setup);
    assert.equal(controls.value.disabled, true, setup);
    assert.equal(controls.apply.disabled, true, setup);
    card.value = "[0]"; card.onchange();
    controls.value.value = "[1]"; controls.apply.onclick();
    assert.equal(ui.run("editSwatchIndices('swatches', '[2]')"), false);
    assert.equal(ui.run("JSON.stringify(state.workflow)"), before, setup);
    assert.match(ui.document.querySelector("#status").textContent, /运行 \/ 检查/);
    ui.run("loopUI.busy = false; loopUI.checking = false; conditionalUI.checking = false; Object.assign(batch, {starting: false, requesting: false, restoring: false, uncertain: false, job: null}); renderBatch()");
    assert.equal(card.disabled, false);
    assert.equal(controls.value.disabled, false);
    assert.equal(controls.apply.disabled, false);
  }
});

test("controls rendered during execution are disabled, including after a full render", () => {
  const ui = controller();
  ui.run("loopUI.busy = true; render()");
  const controls = [...walk(ui.document.querySelector("#nodes")), ...walk(ui.document.querySelector("#node-fields"))].filter(item => item.className === "swatch-topology-control");
  assert.equal(controls.length, 3);
  assert.equal(controls.every(item => item.disabled), true);
});

test("indices rejects INT_LIST connections, context mode and disconnect without touching graph", () => {
  const ui = controller();
  ui.run("state.connecting = link('source', 'indices')");
  const before = ui.run("JSON.stringify(state.workflow)");
  assert.equal(ui.run("connectionMatch('swatches', 'indices')"), "incompatible");
  ui.run("completeConnection('swatches', 'indices')");
  assert.equal(ui.run("JSON.stringify(state.workflow)"), before);
  assert.equal(ui.run("state.connecting.node"), "source");
  assert.match(ui.document.querySelector("#status").textContent, /仅支持.*字面量/);
  const controls = inspector(ui);
  controls.mode.value = "context"; controls.value.value = "indices"; controls.apply.onclick();
  controls.disconnect.onclick();
  assert.equal(ui.run("JSON.stringify(state.workflow)"), before);
});

test("dynamic outputs connect by instance type and render positional preview captions", () => {
  const ui = controller();
  ui.run("editSwatchIndices('swatches', '[2,0,2,1]'); state.connecting = link('swatches', 'img4')");
  assert.equal(ui.run("connectionMatch('target', 'first')"), "recommended");
  ui.run("completeConnection('target', 'first')");
  assert.deepEqual(plain(ui.run("state.workflow.nodes.target.inputs.first")), {node: "swatches", output: "img4"});
  const view = ui.run("nodeOutputView({img4: {kind: 'image', image: 'data:image/png;base64,test', mode: 'RGB', size: [32, 32]}, colors: {kind: 'text', value: {indices: [2,0,2,1]}}}, false, 'swatches')");
  assert.match(text(view), /img4 · index=1/);
  assert.equal(walk(view).find(item => item.tagName === "img").alt, "img4");
});

test("legacy metadata and pre-normalization consumer edges retain images and list previews", () => {
  for (const setup of ["state.workflow.nodes.swatches.legacy_swatch_images = true", "state.workflow.nodes.target.inputs.list = link('swatches', 'images')", "state.workflow.outputs.legacy = link('swatches', 'images')"]) {
    const ui = controller();
    ui.run(setup);
    assert.equal(ui.run("nodeInfo('swatches').outputs.images"), "IMAGE_LIST");
    assert.equal(ui.run("state.workflow.nodes.swatches.legacy_swatch_images"), true);
    assert.equal(ui.run("editSwatchIndices('swatches', '[0]')"), true);
    assert.equal(ui.run("portType('swatches', 'images')"), "IMAGE_LIST");
    const view = ui.run("nodeOutputView({images: {kind: 'object', items: {'0': {kind: 'image', image: 'legacy.png', mode: 'RGB'}}}}, false, 'swatches')");
    assert.equal(walk(view).filter(item => item.tagName === "img").length, 1);
    assert.match(text(view), /旧流程兼容/);
    if (setup.includes("target")) assert.equal(ui.run("state.workflow.nodes.target.inputs.list.output"), "images");
    if (setup.includes("outputs")) assert.equal(ui.run("state.workflow.outputs.legacy.output"), "images");
  }
});

test("swatch edits retain the preview session for the next preview request", async () => {
  const ui = controller();
  ui.run(`
    recordPayload = () => ({});
    state.previewSession = 'cached-models';
    globalThis.requests = [];
    fetch = async (url, options) => {
      requests.push(JSON.parse(options.body));
      return {ok: true, text: async () => JSON.stringify({preview_session: 'cached-models', executed_nodes: ['swatches'], reused_nodes: ['source'], outputs: {img1: {kind: 'image', image: 'swatch.png', mode: 'RGB'}}})};
    };
    editSwatchIndices('swatches', '[1]');
  `);
  await ui.run("previewNode('swatches')");
  assert.equal(ui.run("requests[0].preview_session"), "cached-models");
  assert.deepEqual(plain(ui.run("requests[0].workflow.nodes.swatches.inputs.indices")), [1]);
  assert.equal(ui.run("state.nodeResults.swatches.img1.image"), "swatch.png");
  assert.equal(ui.run("state.nodeResultStale.swatches"), undefined);
  assert.equal(ui.run("loopUI.busy"), false);
});

test("export/import keeps dynamic indices, duplicate mapping, workflow outputs and legacy metadata", async () => {
  for (const legacy of [false, true]) {
    const ui = controller();
    ui.run(`
      editSwatchIndices('swatches', '[2,0,2]');
      ${legacy ? "state.workflow.nodes.swatches.legacy_swatch_images = true; state.workflow.outputs.list = link('swatches', 'images');" : ""}
      saveWorkflow();
    `);
    const saved = JSON.parse(await ui.blob().text());
    assert.deepEqual(saved.workflow.nodes.swatches.inputs.indices, [2, 0, 2]);
    assert.equal(saved.workflow.nodes.swatches.legacy_swatch_images, legacy ? true : undefined);
    const importing = controller();
    importing.run(`
      globalThis.saved = ${JSON.stringify(saved)};
      fetch = async (url, options) => {
        const request = JSON.parse(options.body);
        globalThis.importedPayload = request;
        return {ok: true, text: async () => JSON.stringify({workflow: request.workflow, config: {}})};
      };
      state.previewSession = 'old-workflow'; state.nodeResults.swatches = {old: true};
    `);
    await importing.run("loadWorkflow({target: {files: [{name: 'swatches.json', text: async () => JSON.stringify(saved)}], value: 'file'}})");
    assert.deepEqual(plain(importing.run("state.workflow")), saved.workflow);
    assert.deepEqual(plain(importing.run("importedPayload.workflow")), saved.workflow);
    assert.equal(importing.run("portCaption('swatches', 'outputs', 'img3')"), "img3 · index=2");
    assert.equal(importing.run("portType('swatches', 'images')"), legacy ? "IMAGE_LIST" : undefined);
    assert.equal(importing.run("state.previewSession"), null);
    assert.deepEqual(plain(importing.run("state.nodeResults")), {});
  }
});
