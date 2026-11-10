const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const source = fs.readFileSync(path.join(__dirname, "../object_inpainting_studio/static/node_editing.js"), "utf8");
const removeWorkflowNode = vm.runInNewContext(`${source}\nremoveWorkflowNode;`, { structuredClone });
const plain = value => JSON.parse(JSON.stringify(value));

function fixture() {
  return {
    nodes: {
      source: { class_type: "Source", inputs: {} },
      middle: { class_type: "Middle", inputs: { image: { node: "source", output: "image" } } },
      target: {
        class_type: "Target",
        inputs: {
          first: { node: "middle", output: "image" },
          second: { node: "middle", output: "mask" },
          untouched: { node: "source", output: "image" },
        },
      },
    },
    outputs: { removed: { node: "middle", output: "image" }, kept: { node: "target", output: "image" } },
  };
}

test("deletes only the selected node and its edges/outputs, not downstream nodes", () => {
  const workflow = fixture();
  const result = removeWorkflowNode(workflow, "middle", {});
  assert.deepEqual(Object.keys(workflow.nodes), ["source", "target"]);
  assert.equal(workflow.nodes.target.inputs.first, null);
  assert.equal(workflow.nodes.target.inputs.second, null);
  assert.deepEqual(workflow.nodes.target.inputs.untouched, { node: "source", output: "image" });
  assert.deepEqual(workflow.outputs, { kept: { node: "target", output: "image" } });
  assert.deepEqual(plain(result), {
    disconnected: [{ node_id: "target", port: "first" }, { node_id: "target", port: "second" }],
    removedOutputs: ["removed"],
  });
});

test("restores explicit defaults without sharing mutable catalog values", () => {
  const workflow = fixture();
  const defaults = { first: [1, 2], second: 0 };
  removeWorkflowNode(workflow, "middle", { Target: { input_defaults: defaults } });
  assert.deepEqual(workflow.nodes.target.inputs.first, [1, 2]);
  assert.equal(workflow.nodes.target.inputs.second, 0);
  workflow.nodes.target.inputs.first.push(3);
  assert.deepEqual(defaults.first, [1, 2]);
});

test("preserves false, empty-string and null defaults", () => {
  for (const value of [false, "", null]) {
    const workflow = fixture();
    removeWorkflowNode(workflow, "middle", { Target: { input_defaults: { first: value } } });
    assert.equal(workflow.nodes.target.inputs.first, value);
  }
});

test("unknown/already deleted node is a no-op", () => {
  const workflow = fixture();
  const before = structuredClone(workflow);
  assert.equal(removeWorkflowNode(workflow, "missing", {}), null);
  assert.deepEqual(workflow, before);
  removeWorkflowNode(workflow, "middle", {});
  const after = structuredClone(workflow);
  assert.equal(removeWorkflowNode(workflow, "middle", {}), null);
  assert.deepEqual(workflow, after);
});

test("context and literal objects are not mistaken for links", () => {
  const workflow = fixture();
  workflow.nodes.target.inputs.literal = { node: "middle", output: "image", extra: true };
  workflow.nodes.target.inputs.context = { context: "middle" };
  removeWorkflowNode(workflow, "middle", {});
  assert.deepEqual(workflow.nodes.target.inputs.literal, { node: "middle", output: "image", extra: true });
  assert.deepEqual(workflow.nodes.target.inputs.context, { context: "middle" });
});

test("source, target and final-node deletion leaves no dangling links", () => {
  const workflow = fixture();
  removeWorkflowNode(workflow, "source", {});
  assert.equal(workflow.nodes.middle.inputs.image, null);
  assert.equal(workflow.nodes.target.inputs.untouched, null);
  removeWorkflowNode(workflow, "target", {});
  assert.deepEqual(workflow.outputs, { removed: { node: "middle", output: "image" } });
  removeWorkflowNode(workflow, "middle", {});
  assert.deepEqual(workflow, { nodes: {}, outputs: {} });
});

// Exercise the real UI controller with a minimal DOM, without browser/system dependencies.
function controller() {
  class Element {
    constructor(editable = false) {
      this.editable = editable;
      this.style = {};
      this.classList = { toggle() {}, remove() {}, add() {} };
    }
    closest() { return this.editable ? this : null; }
    replaceChildren() {}
    addEventListener() {}
  }
  const elements = new Map();
  const events = {};
  const document = {
    querySelector(selector) {
      if (!elements.has(selector)) elements.set(selector, new Element());
      return elements.get(selector);
    },
    querySelectorAll: () => [],
    addEventListener: (type, handler) => { events[type] = handler; },
  };
  const sandbox = vm.createContext({
    document, Element, structuredClone, clearTimeout, setTimeout,
    requestAnimationFrame() {},
    fetch: () => new Promise(() => {}),
  });
  for (const name of ["node_editing.js", "loop.js", "multiple_loops.js", "app.js"]) {
    vm.runInContext(fs.readFileSync(path.join(__dirname, "../object_inpainting_studio/static", name), "utf8"), sandbox);
  }
  vm.runInContext("render = () => {}; renderLoop = () => {}; batch.restoring = false; batch.requesting = false;", sandbox);
  sandbox.workflowForTest = fixture();
  vm.runInContext("state.workflow = workflowForTest; state.catalog = {};", sandbox);
  return { run: code => vm.runInContext(code, sandbox), events, Element };
}

test("UI deletion clears selection, connections, layout, previews and stale loop plans", () => {
  const ui = controller();
  ui.run(`
    state.selected = 'middle'; state.connecting = {node: 'middle', output: 'image'};
    state.layout.middle = {x: 1, y: 2}; state.expanded.middle = true;
    state.nodeResults.middle = {image: 'old preview'}; state.nodeErrors.target = 'old error';
    loopUI.sourceId = 'source'; loopUI.targetId = 'middle'; loopUI.enabled = true;
    loopUI.plan = {total: 5}; loopUI.planKey = 'old plan'; loopUI.error = 'old error';
    deleteNode('middle');
  `);
  const result = plain(ui.run("({selected: state.selected, connecting: state.connecting, layout: state.layout, expanded: state.expanded, results: state.nodeResults, errors: state.nodeErrors, target: loopUI.targetId, plan: loopUI.plan, planKey: loopUI.planKey})"));
  assert.deepEqual(result, { selected: null, connecting: null, layout: {}, expanded: {}, results: {}, errors: {}, target: null, plan: null, planKey: null });
  assert.equal(ui.run("state.workflow.nodes.target.inputs.first"), null);
});

test("deleting the loop source disables traversal", () => {
  const ui = controller();
  ui.run("loopUI.sourceId = 'source'; loopUI.enabled = true; deleteNode('source');");
  assert.equal(ui.run("loopUI.sourceId"), null);
  assert.equal(ui.run("loopUI.enabled"), false);
});

test("preview, active/cancelling batch and uncertain task state block deletion", () => {
  for (const setup of ["loopUI.busy = true", "batch.job = {status: 'running'}", "batch.job = {status: 'cancelling'}", "batch.uncertain = true", "batch.starting = true"]) {
    const ui = controller();
    ui.run(`${setup}; deleteNode('middle');`);
    assert.equal(ui.run("Object.hasOwn(state.workflow.nodes, 'middle')"), true, setup);
  }
});

test("Delete shortcut ignores editable targets, composition, repeat and modifiers", () => {
  for (const extra of [{ target: "editable" }, { isComposing: true }, { repeat: true }, { ctrlKey: true }, { metaKey: true }, { altKey: true }, { shiftKey: true }]) {
    const ui = controller();
    ui.run("state.selected = 'middle';");
    const event = { key: "Delete", target: new ui.Element(extra.target === "editable"), preventDefault() { this.prevented = true; }, ...extra };
    if (extra.target) event.target = new ui.Element(true);
    ui.events.keydown(event);
    assert.equal(ui.run("Object.hasOwn(state.workflow.nodes, 'middle')"), true);
  }
  const ui = controller();
  ui.run("state.selected = 'middle';");
  const event = { key: "Delete", target: new ui.Element(), preventDefault() { this.prevented = true; } };
  ui.events.keydown(event);
  assert.equal(ui.run("Object.hasOwn(state.workflow.nodes, 'middle')"), false);
  assert.equal(event.prevented, true);
});
