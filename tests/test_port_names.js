const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const plain = value => JSON.parse(JSON.stringify(value));
const text = element => [element.textContent || "", ...element.children.map(text)].join(" ");
const walk = element => [element, ...element.children.flatMap(walk)];

function controller() {
  class Element {
    constructor(tag = "div") {
      this.tagName = tag;
      this.children = [];
      this.style = { setProperty(name, value) { this[name] = value; } };
      this.dataset = {};
      this.attributes = {};
      this.className = "";
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
    focus() { this.focused = true; }
    querySelector(selector) {
      return walk(this).find(item => selector.startsWith(".")
        ? item.className.split(" ").includes(selector.slice(1)) : item.tagName === selector);
    }
  }
  const elements = new Map();
  const document = {
    createElement: tag => new Element(tag),
    querySelector(selector) {
      if (!elements.has(selector)) elements.set(selector, new Element());
      return elements.get(selector);
    },
    querySelectorAll(selector) {
      if (selector === "#node-fields .binding") return document.querySelector("#node-fields").children.filter(item => item.className === "binding");
      if (selector === ".save-item-add, .save-item-remove, .save-item-name") return [...walk(document.querySelector("#nodes")), ...walk(document.querySelector("#node-fields"))].filter(item => ["save-item-add", "save-item-remove", "save-item-name"].includes(item.className));
      if (selector === ".node-preview-shell") return [...walk(document.querySelector("#nodes")), ...walk(document.querySelector("#node-preview"))].filter(item => item.className === "node-preview-shell");
      return [];
    },
    addEventListener() {},
  };
  const sandbox = vm.createContext({
    document, Element, structuredClone, setTimeout, clearTimeout,
    requestAnimationFrame() {}, fetch: () => new Promise(() => {}),
    Option: function (label, value) { const option = new Element("option"); option.textContent = label; option.value = value; return option; },
  });
  for (const file of ["node_editing.js", "loop.js", "multiple_loops.js", "app.js"]) vm.runInContext(fs.readFileSync(path.join(__dirname, "../object_inpainting_studio/static", file), "utf8"), sandbox);
  vm.runInContext(`
    render = () => {}; renderLoop = () => {}; batch.restoring = false;
    const meta = (name, role, kind = name) => ({name, role, kind});
    const info = (inputs, outputs, inputLabels, outputLabels) => ({
      inputs, outputs, port_labels: {inputs: inputLabels, outputs: outputLabels},
      help: {title: 'test', summary: 'test', inputs, outputs, test: 'test', parameters: []}
    });
    state.catalog = {
      Gray: info({target: 'RGB_IMAGE', edit_mask: 'MASK'}, {source: 'RGB_IMAGE'},
        {target: meta('rgb_image', '原图'), edit_mask: meta('mask', '编辑区域', 'edit_mask')},
        {source: meta('masked_rgb_image', '编辑区涂灰的 RGB 图')}),
      Recolor: info({target: 'RGB_IMAGE', mask: 'RGB_IMAGE'}, {source: 'RGB_IMAGE', mask: 'MASK'},
        {target: meta('rgb_image', '原图'), mask: meta('masked_rgb_image', '涂灰原图，不是黑白 Mask')},
        {source: meta('rgb_image', '换色结果，未涂灰'), mask: meta('mask', '恢复的编辑区域', 'edit_mask')}),
      Segment: info({image: 'RGB_IMAGE'}, {object_mask: 'MASK'},
        {image: meta('rgb_image', 'RGB 图')}, {object_mask: meta('mask', '物体区域', 'object_mask')}),
      Reference: info({}, {rgb: 'RGB_IMAGE'}, {}, {rgb: meta('reference_rgb_image', '白底 RGB 参考图')}),
      Swatches: info({}, {images: 'IMAGE_LIST'}, {}, {images: meta('image_list', '独立色块')}),
    };
    state.workflow = {nodes: {
      gray: {class_type: 'Gray', inputs: {target: null, edit_mask: null}},
      recolor: {class_type: 'Recolor', inputs: {target: null, mask: null}},
      segment: {class_type: 'Segment', inputs: {image: null}},
      reference: {class_type: 'Reference', inputs: {}},
      swatches: {class_type: 'Swatches', inputs: {}},
    }, outputs: {}};
  `, sandbox);
  return { run: code => vm.runInContext(code, sandbox), document };
}

function preparePreview(ui) {
  ui.run(`
    globalThis.CSS = {escape: value => value};
    renderBatch = () => {}; recordPayload = () => ({}); currentConfig = () => ({});
    batch.requesting = false; batch.restoring = false; loopUI.busy = false;
    state.workflow.nodes.recolor.inputs.mask = {node: 'gray', output: 'source'};
    globalThis.requests = [];
    fetch = async (url, options) => {
      const request = JSON.parse(options.body); requests.push(request);
      return {ok: true, text: async () => JSON.stringify({
        outputs: {source: {kind: 'text', value: request.node_id}},
        executed_nodes: [request.node_id], reused_nodes: request.node_id === 'recolor' ? ['gray'] : [],
        preview_session: 'server-session'
      })};
    };
  `);
}

test("saving after editing save-only fields keeps the model session and preview", async () => {
  const ui = saveController(); preparePreview(ui);
  await ui.run("previewNode('gray')");
  ui.run("renderNode('save', state.workflow.nodes.save, 0)");
  const directory = walk(ui.document.querySelector("#nodes")).find(item => item.dataset.field === "dir");
  directory.value = "new-output";
  directory.onchange();
  ui.run("renameSaveItem('save', 'img1', 'original'); addSaveItem('save')");
  await ui.run("previewNode('save')");
  assert.equal(ui.run("requests.at(-1).preview_session"), "server-session");
  assert.equal(ui.run("state.nodeResults.gray.source.value"), "gray");
});

test("downstream preview sends session and preserves upstream display", async () => {
  const ui = controller(); preparePreview(ui);
  await ui.run("previewNode('gray')");
  await ui.run("previewNode('recolor')");
  assert.equal(ui.run("requests[0].preview_session"), null);
  assert.equal(ui.run("requests[1].preview_session"), "server-session");
  assert.deepEqual(plain(ui.run("Object.keys(state.nodeResults)")), ["gray", "recolor"]);
  await ui.run("previewNode('gray')");
  assert.deepEqual(plain(ui.run("Object.keys(state.nodeResults)")), ["gray", "recolor"]);
  assert.equal(ui.run("state.nodeResultStale.recolor"), true);
});

test("editing preserves session and last preview but rejects an in-flight obsolete response", async () => {
  const ui = controller(); preparePreview(ui);
  await ui.run("previewNode('gray')");
  ui.run("invalidateResults()");
  assert.equal(ui.run("state.previewSession"), "server-session");
  assert.equal(ui.run("state.nodeResultStale.gray"), true);
  ui.run(`fetch = () => new Promise(resolve => {
    globalThis.finishPreview = () => resolve({ok: true, text: async () => JSON.stringify({
      outputs: {}, executed_nodes: ['gray'], preview_session: 'stale-session'
    })});
  });`);
  const pending = ui.run("previewNode('gray')");
  ui.run("invalidateResults(); finishPreview()");
  await pending;
  assert.equal(ui.run("state.previewSession"), "server-session");
  assert.equal(ui.run("state.nodeResults.gray.source.value"), "gray");
});

test("previews remain visible after input edits and hide only through explicit controls", async () => {
  const ui = controller(); preparePreview(ui);
  await ui.run("previewNode('gray')");
  ui.run("document.querySelector('#node-preview').append(nodePreviewView('gray')); invalidateResults('gray')");
  let shell = ui.document.querySelector("#node-preview").children[0];
  assert.match(text(shell), /历史预览/);
  assert.equal(shell.querySelector(".preview-body").hidden, false);
  shell.querySelector(".preview-toggle").onclick({stopPropagation() {}});
  assert.equal(shell.querySelector(".preview-body").hidden, true);
  assert.match(text(shell), /显示预览/);
  await ui.run("previewNode('gray')");
  ui.run("document.querySelector('#node-preview').replaceChildren(nodePreviewView('gray'))");
  shell = ui.document.querySelector("#node-preview").children[0];
  assert.equal(shell.querySelector(".preview-body").hidden, true);
  shell.querySelector(".preview-toggle").onclick({stopPropagation() {}});
  assert.equal(shell.querySelector(".preview-body").hidden, false);
  assert.equal(ui.run("state.nodeResults.gray.source.value"), "gray");
});

test("failed retest retains last successful preview and marks it historical", async () => {
  const ui = controller(); preparePreview(ui);
  await ui.run("previewNode('gray')");
  ui.run(`fetch = async () => ({ok: false, text: async () => JSON.stringify({detail: 'inference failed'})});`);
  await ui.run("previewNode('gray')");
  assert.equal(ui.run("state.nodeResults.gray.source.value"), "gray");
  assert.equal(ui.run("state.nodeResultStale.gray"), true);
  assert.match(ui.run("state.nodeErrors.gray"), /inference failed/);
});

test("parameter folding and loop view do not fold the retained preview", () => {
  const ui = controller();
  ui.run("state.nodeResults.gray = {source: {kind: 'text', value: 'retained'}}; state.expanded.gray = false; loopUI.enabled = true; renderNode('gray', state.workflow.nodes.gray, 0)");
  const card = ui.document.querySelector("#nodes").children[0];
  const settings = card.children.find(item => item.className === "node-settings");
  const shell = card.children.find(item => item.className === "node-preview-shell");
  assert.equal(settings.open, false);
  assert.ok(shell);
  assert.equal(shell.querySelector(".preview-body").hidden, false);
});

test("captions and link descriptions use semantic names with raw keys preserved", () => {
  const ui = controller();
  assert.match(ui.run("portCaption('gray', 'outputs', 'source')"), /^masked_rgb_image/);
  assert.match(ui.run("portCaption('recolor', 'inputs', 'mask')"), /^masked_rgb_image.*不是黑白 Mask/);
  assert.match(ui.run("bindingLabel({node: 'gray', output: 'source'})"), /gray → masked_rgb_image/);
  assert.equal(ui.run("portCaption('missing', 'outputs', 'new_port')"), "new_port");
});

test("recommended ports expose semantic labels, raw dataset keys and accessible descriptions", () => {
  const ui = controller();
  ui.run("state.connecting = {node: 'gray', output: 'source'}");
  const row = ui.run("portRow('recolor', 'mask', 'RGB_IMAGE', 'input')");
  assert.equal(row.dataset.match, "recommended");
  assert.match(text(row), /masked_rgb_image.*不是黑白 Mask/);
  const button = row.children[0];
  assert.equal(button.dataset.port, "mask");
  assert.match(button.className, /port-recommended/);
  assert.match(button.title, /内部字段：mask/);
  assert.match(button.attributes["aria-label"], /输入 masked_rgb_image/);
  ui.run("state.connecting = {node: 'recolor', output: 'mask'}");
  assert.doesNotMatch(ui.run("portRow('recolor', 'mask', 'RGB_IMAGE', 'input')").children[0].className, /pending/);
  assert.match(ui.run("portRow('recolor', 'mask', 'MASK', 'output')").children[2].className, /pending/);
});

test("connecting friendly aliases still stores the legacy workflow keys", () => {
  const ui = controller();
  ui.run("beginConnection('gray', 'source'); completeConnection('recolor', 'mask')");
  assert.deepEqual(plain(ui.run("state.workflow.nodes.recolor.inputs.mask")), {node: "gray", output: "source"});
  assert.equal(ui.run("Object.hasOwn(state.workflow.nodes.recolor.inputs, 'masked_rgb_image')"), false);
});

test("binary mask cannot connect to gray-masked RGB and is not recommended", () => {
  const ui = controller();
  ui.run("beginConnection('segment', 'object_mask')");
  assert.equal(ui.run("connectionMatch('recolor', 'mask')"), "incompatible");
  ui.run("completeConnection('recolor', 'mask')");
  assert.equal(ui.run("state.workflow.nodes.recolor.inputs.mask"), null);
  assert.match(ui.document.querySelector("#status").textContent, /无法连接.*masked_rgb_image/);
});

test("semantic hints do not forbid valid same-format connections", () => {
  const ui = controller();
  ui.run("beginConnection('reference', 'rgb')");
  assert.equal(ui.run("connectionMatch('segment', 'image')"), "compatible");
  ui.run("completeConnection('segment', 'image')");
  assert.deepEqual(plain(ui.run("state.workflow.nodes.segment.inputs.image")), {node: "reference", output: "rgb"});
  ui.run("beginConnection('segment', 'object_mask')");
  assert.equal(ui.run("connectionMatch('gray', 'edit_mask')"), "compatible");
  assert.equal(ui.run("connectionMatch('segment', 'image')"), "incompatible");
});

test("output preview captions are renamed but image keys and list indices remain stable", () => {
  const ui = controller();
  const preview = ui.run("nodeOutputView({source: {kind: 'image', image: 'data:image/png;base64,test', size: [32,32], mode: 'RGB'}}, false, 'gray')");
  assert.match(text(preview), /masked_rgb_image.*32 × 32 RGB/);
  assert.equal(walk(preview).find(item => item.tagName === "img").alt, "source");
  const list = ui.run("nodeOutputView({images: {kind: 'object', items: {'0': {kind: 'image', image: 'test'}}}}, false, 'swatches')");
  assert.match(text(list), /image_list/);
  assert.equal(walk(list).find(item => item.tagName === "img").alt, "0");
});

test("inspector uses friendly names and focuses inputs by raw field rather than title", () => {
  const ui = controller();
  ui.run("state.selected = 'recolor'; renderInspector()");
  const block = ui.document.querySelectorAll("#node-fields .binding").find(item => item.dataset.port === "mask");
  assert.match(block.children[0].textContent, /^masked_rgb_image/);
  ui.run("completeConnection('recolor', 'mask')");
  assert.equal(block.querySelector("input").focused, true);
});

test("node widgets display aliases but edits update the original field", () => {
  const ui = controller();
  ui.run("renderNode('recolor', state.workflow.nodes.recolor, 0)");
  const widgets = walk(ui.document.querySelector("#nodes"));
  const field = widgets.find(item => item.tagName === "input" && item.dataset.field === "mask");
  const label = widgets.find(item => item.tagName === "label" && item.children.includes(field));
  assert.match(label.textContent, /^masked_rgb_image/);
  field.value = "/uploaded.png";
  field.onchange();
  assert.equal(ui.run("state.workflow.nodes.recolor.inputs.mask"), "/uploaded.png");
  assert.equal(ui.run("Object.hasOwn(state.workflow.nodes.recolor.inputs, 'masked_rgb_image')"), false);
});

test("node cards and inspector expose the concrete input and output contract classes", () => {
  const ui = controller();
  ui.run("state.catalog.Recolor.input_contract = 'FluxRandomRecolorInputs'; state.catalog.Recolor.output_contract = 'RandomRecolorOutputs'; renderNode('recolor', state.workflow.nodes.recolor, 0); state.selected = 'recolor'; renderInspector()");
  assert.match(text(ui.document.querySelector("#nodes")), /入口 FluxRandomRecolorInputs → 出口 RandomRecolorOutputs/);
  assert.match(text(ui.document.querySelector("#node-help")), /入口类：FluxRandomRecolorInputs/);
  assert.match(text(ui.document.querySelector("#node-help")), /出口类：RandomRecolorOutputs/);
});

test("independent MaskDilation controls render defaults and store numeric node-local edits", () => {
  const ui = controller();
  ui.run(`
    state.catalog.MaskDilationNode = info(
      {mask: 'MASK', dilation_steps: 'INT', dilation_kernel: 'INT'},
      {mask: 'MASK', bbox: 'BBOX', quality: 'STRING'},
      {dilation_steps: meta('dilation_steps', '膨胀次数'), dilation_kernel: meta('dilation_kernel', '方形核边长')}, {}
    );
    state.catalog.MaskDilationNode.input_defaults = {dilation_steps: 1, dilation_kernel: 3};
    state.workflow.nodes.dilate = {class_type: 'MaskDilationNode', inputs: {mask: null}};
    renderNode('dilate', state.workflow.nodes.dilate, 0);
  `);
  const widgets = walk(ui.document.querySelector("#nodes"));
  for (const [key, original, updated] of [
    ["dilation_steps", "1", "2"], ["dilation_kernel", "3", "11"],
  ]) {
    const input = widgets.find(item => item.tagName === "input" && item.dataset.field === key);
    assert.ok(input, key);
    assert.equal(input.value, original);
    input.value = updated;
    input.onchange();
    assert.equal(ui.run(`state.workflow.nodes.dilate.inputs.${key}`), Number(updated));
  }
  assert.match(text(ui.document.querySelector("#nodes")), /膨胀次数/);
  assert.equal(widgets.some(item => item.dataset.field === "mask_threshold"), false);
});

test("workflow import uses the backend migrated graph instead of hidden legacy SAM3 controls", async () => {
  const ui = controller();
  ui.run(`
    fetch = async () => ({
      ok: true, status: 200,
      text: async () => JSON.stringify({config: {}, workflow: {
        nodes: {
          sam: {class_type: 'SAM3TextMaskNode', inputs: {image: null, text: 'bag', threshold: 0.5, mask_threshold: 0.5}},
          sam_dilation: {class_type: 'MaskDilationNode', inputs: {mask: {node: 'sam', output: 'object_mask'}, dilation_steps: 2, dilation_kernel: 11}}
        },
        outputs: {mask: {node: 'sam_dilation', output: 'mask'}}
      }})
    });
  `);
  await ui.run(`loadWorkflow({target: {files: [{name: 'legacy.json', text: async () => JSON.stringify({nodes: {
    sam: {class_type: 'SAM3TextMaskNode', inputs: {image: null, text: 'bag', dilation_steps: 2, dilation_kernel: 11}}
  }, outputs: {mask: {node: 'sam', output: 'object_mask'}}})}]}})`);
  assert.equal(ui.run("Object.hasOwn(state.workflow.nodes.sam.inputs, 'dilation_steps')"), false);
  assert.equal(ui.run("state.workflow.nodes.sam_dilation.inputs.dilation_steps"), 2);
  assert.equal(ui.run("state.workflow.outputs.mask.node"), "sam_dilation");
  assert.match(ui.document.querySelector("#status").textContent, /Imported workflow/);
});

function saveController() {
  const ui = controller();
  ui.run(`
    batch.requesting = false;
    state.catalog.SaveImagesNode = info(
      {img1: 'IMAGE', img2: 'IMAGE', img3: 'IMAGE', img4: 'IMAGE', images: 'IMAGE_LIST', dir: 'PATH', filename: 'STRING'},
      {paths: 'JSON_RECORD'}, {images: meta('image_list', '图片列表')}, {}
    );
    state.catalog.SaveImagesNode.input_defaults = {filename: ''};
    state.catalog.SaveImagesNode.side_effects = true;
    state.workflow.nodes.save = {class_type: 'SaveImagesNode', inputs: {dir: '/out', img1: {node: 'gray', output: 'source'}}};
  `);
  return ui;
}

test("dynamic save ports grow beyond eight with per-instance metadata and no naming ports", () => {
  const ui = saveController();
  ui.run("for (let i = 0; i < 8; i++) addSaveItem('save'); renderNode('save', state.workflow.nodes.save, 0)");
  assert.equal(ui.run("saveItems(state.workflow.nodes.save).length"), 12);
  assert.equal(ui.run("nodeInfo('save').inputs.img12"), "IMAGE");
  assert.equal(ui.run("state.catalog.SaveImagesNode.inputs.img12"), undefined);
  const elements = walk(ui.document.querySelector("#nodes"));
  assert.equal(elements.filter(item => item.dataset.saveName).length, 12);
  assert.equal(elements.filter(item => item.className === "save-item-add").length, 1);
  assert.equal(ui.run("Object.keys(nodeInfo('save').inputs).some(key => key.startsWith('name'))"), false);
  ui.run("beginConnection('segment', 'object_mask'); completeConnection('save', 'img12')");
  assert.deepEqual(plain(ui.run("state.workflow.nodes.save.inputs.img12")), {node: "segment", output: "object_mask"});
  ui.run("state.connecting = {node: 'swatches', output: 'images'}");
  assert.equal(ui.run("connectionMatch('save', 'img12')"), "incompatible");
  assert.equal(ui.run("connectionMatch('save', 'images')"), "recommended");
});

test("save name widgets update directories without changing input IDs or connections", () => {
  const ui = saveController();
  ui.run("renderNode('save', state.workflow.nodes.save, 0)");
  const input = walk(ui.document.querySelector("#nodes")).find(item => item.dataset.saveName === "img1");
  assert.equal(input.placeholder, "img1");
  assert.equal(input.attributes["aria-label"], "img1 保存名称");
  input.value = "目标图";
  input.onchange();
  assert.equal(ui.run("state.workflow.nodes.save.save_items[0].name"), "目标图");
  assert.deepEqual(plain(ui.run("state.workflow.nodes.save.inputs.img1")), {node: "gray", output: "source"});
  assert.equal(ui.run("portMetadata('save', 'inputs', 'img1').name"), "目标图");
  ui.run("state.selected = 'save'; renderInspector()");
  assert.match(text(ui.document.querySelector("#node-fields")), /目标图/);
  assert.equal(ui.run("renameSaveItem('save', 'img1', '')"), true);
  assert.equal(ui.run("portMetadata('save', 'inputs', 'img1').name"), "img1");
});

test("deleting save items disconnects only those items and never reuses deleted IDs", () => {
  const ui = saveController();
  ui.run("for (let i = 0; i < 4; i++) addSaveItem('save'); beginConnection('segment', 'object_mask'); completeConnection('save', 'img8'); removeSaveItem('save', 'img8'); removeSaveItem('save', 'img2'); addSaveItem('save')");
  assert.equal(ui.run("Object.hasOwn(state.workflow.nodes.save.inputs, 'img8')"), false);
  assert.equal(ui.run("Object.hasOwn(nodeInfo('save').inputs, 'img2')"), false);
  assert.equal(ui.run("Object.hasOwn(state.workflow.nodes.save.inputs, 'img9')"), true);
  assert.deepEqual(plain(ui.run("state.workflow.nodes.save.inputs.img1")), {node: "gray", output: "source"});
  assert.equal(ui.run("state.workflow.nodes.save.save_next_id"), 10);
  ui.run("for (const item of [...saveItems(state.workflow.nodes.save)]) removeSaveItem('save', item.id)");
  assert.deepEqual(plain(ui.run("Object.keys(nodeInfo('save').inputs)")), ["images", "dir", "filename"]);
  ui.run("addSaveItem('save')");
  assert.equal(ui.run("state.workflow.nodes.save.save_items[0].id"), "img10");
});

test("save item mutations mark preview stale without discarding model session and clear loop plans", () => {
  const ui = saveController();
  ui.run("saveItems(state.workflow.nodes.save); state.previewSession = 'existing-session'; state.nodeResults.save = {old: true}; loopUI.plan = {old: true}; loopUI.planKey = 'old'; renameSaveItem('save', 'img1', 'target')");
  assert.equal(ui.run("state.nodeResults.save.old"), true);
  assert.equal(ui.run("state.nodeResultStale.save"), true);
  assert.equal(ui.run("state.previewSession"), "existing-session");
  assert.equal(ui.run("loopUI.plan"), null);
  assert.equal(ui.run("loopUI.planKey"), null);
  for (const setup of ["loopUI.busy = true", "batch.job = {status: 'running'}", "batch.uncertain = true"]) {
    const ui = saveController();
    ui.run("saveItems(state.workflow.nodes.save)");
    const before = plain(ui.run("state.workflow"));
    ui.run(`${setup}; addSaveItem('save'); removeSaveItem('save', 'img1'); renameSaveItem('save', 'img1', 'new')`);
    assert.deepEqual(plain(ui.run("state.workflow")), before);
    assert.match(ui.document.querySelector("#status").textContent, /不能修改/);
  }
});

test("save name duplicates and unsafe paths are rejected without changing valid names", () => {
  const ui = saveController();
  ui.run("renameSaveItem('save', 'img1', 'Mask')");
  for (const value of ["mask", "../escape", "a\\b", ".", "..", "bad:", "a\u0000b"]) {
    assert.equal(ui.run(`renameSaveItem('save', 'img2', ${JSON.stringify(value)})`), false);
    assert.equal(ui.run("state.workflow.nodes.save.save_items[1].name"), "");
  }
});

test("dynamic save metadata survives import/export including deleted slots and filename links", async () => {
  const ui = saveController();
  ui.run(`
    for (let i = 0; i < 4; i++) addSaveItem('save');
    renameSaveItem('save', 'img8', 'reference'); removeSaveItem('save', 'img2');
    state.workflow.nodes.save.inputs.filename = {node: 'folder', output: 'filename'};
    fetch = async (url, options) => ({ok: true, status: 200, text: async () => JSON.stringify({config: {}, workflow: JSON.parse(options.body).workflow})});
    const serializedSaveWorkflow = JSON.stringify({workflow: state.workflow, config: {}, layout: {}});
  `);
  const before = plain(ui.run("state.workflow.nodes.save"));
  await ui.run("loadWorkflow({target: {files: [{name: 'saved.json', text: async () => serializedSaveWorkflow}]}})");
  assert.deepEqual(plain(ui.run("state.workflow.nodes.save")), before);
  assert.equal(ui.run("Object.hasOwn(nodeInfo('save').inputs, 'img2')"), false);
  assert.equal(ui.run("portMetadata('save', 'inputs', 'img8').name"), "reference");
  ui.run("addSaveItem('save')");
  assert.equal(ui.run("state.workflow.nodes.save.save_items.at(-1).id"), "img9");
});

test("source node deletion clears dynamic links while preserving names and IDs", () => {
  const ui = saveController();
  ui.run("for (let i = 0; i < 4; i++) addSaveItem('save'); beginConnection('gray', 'source'); completeConnection('save', 'img8'); renameSaveItem('save', 'img8', 'source'); removeWorkflowNode(state.workflow, 'gray', state.catalog)");
  assert.equal(ui.run("state.workflow.nodes.save.inputs.img8"), null);
  assert.equal(ui.run("state.workflow.nodes.save.save_items.at(-1).name"), "source");
  assert.equal(ui.run("nodeInfo('save').inputs.img8"), "IMAGE");
});

test("save item controls re-enable when preview finishes and filename disconnect restores default", () => {
  const ui = saveController();
  ui.run("renderNode('save', state.workflow.nodes.save, 0); loopUI.busy = true; renderBatch()");
  const controls = ui.document.querySelectorAll(".save-item-add, .save-item-remove, .save-item-name");
  assert.ok(controls.length > 0);
  assert.ok(controls.every(item => item.disabled));
  ui.run("loopUI.busy = false; renderBatch()");
  assert.ok(controls.every(item => !item.disabled));
  ui.run("state.workflow.nodes.save.inputs.filename = {node: 'folder', output: 'filename'}; state.selected = 'save'; renderInspector()");
  const block = ui.document.querySelectorAll("#node-fields .binding").find(item => item.dataset.port === "filename");
  walk(block).find(item => item.textContent === "Disconnect").onclick();
  assert.equal(ui.run("state.workflow.nodes.save.inputs.filename"), "");
});
