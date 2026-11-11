const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const plain = value => JSON.parse(JSON.stringify(value));
const walk = element => [element, ...element.children.flatMap(walk)];

// Run the actual controllers with an intentionally small DOM, as in test_port_names.js.
function controller() {
  class EventTarget {
    constructor() { this.listeners = new Map(); }
    addEventListener(type, callback) {
      if (!this.listeners.has(type)) this.listeners.set(type, []);
      this.listeners.get(type).push(callback);
    }
    dispatchEvent(event) {
      event.target ??= this;
      event.preventDefault = () => { event.defaultPrevented = true; };
      event.stopPropagation = () => { event.propagationStopped = true; };
      this[`on${event.type}`]?.(event);
      for (const callback of this.listeners.get(event.type) || []) callback(event);
      return !event.defaultPrevented;
    }
  }
  class Element extends EventTarget {
    constructor(tag = "div") {
      super();
      this.tagName = tag;
      this.children = [];
      this.dataset = {};
      this.style = { setProperty(name, value) { this[name] = value; } };
      this.attributes = {};
      this.value = "";
      this.checked = false;
      this.classList = {
        contains: name => (this.className || "").split(/\s+/).includes(name),
        toggle: (name, force) => {
          const classes = new Set((this.className || "").split(/\s+/).filter(Boolean));
          const enabled = force ?? !classes.has(name);
          if (enabled) classes.add(name); else classes.delete(name);
          this.className = [...classes].join(" ");
          return enabled;
        },
        add: name => this.classList.toggle(name, true),
        remove: name => this.classList.toggle(name, false),
      };
    }
    set value(value) { this._value = String(value); }
    get value() { return this._value; }
    get options() { return this.children; }
    get selectedOptions() { return this.children.filter(option => option.selected); }
    append(...children) { children.forEach(child => { child.parentElement = this; this.children.push(child); }); }
    add(child) { this.append(child); }
    replaceChildren(...children) { this.children.forEach(child => { child.parentElement = null; }); this.children = []; this.append(...children); }
    setAttribute(name, value) { this.attributes[name] = value; }
    matches(selector) {
      return selector.split(",").some(part => {
        const simple = part.trim();
        const ancestors = simple.split(/\s+/);
        if (ancestors.length > 1) {
          if (!this.matches(ancestors.pop())) return false;
          for (let parent = this.parentElement; parent; parent = parent.parentElement) {
            if (parent.matches(ancestors.join(" "))) return true;
          }
          return false;
        }
        const id = simple.match(/^#([\w-]+)/)?.[1];
        if (id && this.id !== id) return false;
        const tag = simple.match(/^[a-z]+/i)?.[0];
        if (tag && this.tagName !== tag) return false;
        if ([...simple.matchAll(/\.([\w-]+)/g)].some(([, name]) => !this.classList.contains(name))) return false;
        for (const [, attribute, value] of simple.matchAll(/\[([\w-]+)(?:="([^"]*)")?\]/g)) {
          const actual = attribute.startsWith("data-") ? this.dataset[attribute.slice(5).replace(/-([a-z])/g, (_, letter) => letter.toUpperCase())] : this.attributes[attribute];
          if (value == null ? actual == null : actual !== value) return false;
        }
        return true;
      });
    }
    querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
    querySelectorAll(selector) { return walk(this).slice(1).filter(child => child.matches(selector)); }
    closest(selector) { return this.matches(selector) ? this : this.parentElement?.closest(selector) || null; }
    scrollIntoView(options) { this.scrolled = options; }
    click() { if (!this.disabled) this.dispatchEvent({type: "click"}); }
    remove() {
      if (!this.parentElement) return;
      const siblings = this.parentElement.children;
      const index = siblings.indexOf(this);
      if (index !== -1) siblings.splice(index, 1);
      this.parentElement = null;
    }
    focus() { document.activeElement = this; }
  }
  const elements = new Map();
  const document = Object.assign(new EventTarget(), {
    body: new Element(),
    createElement: tag => new Element(tag),
    createElementNS: (_, tag) => new Element(tag),
    querySelector(selector) {
      if (/^#conditional-(enabled|legacy|legacy-controls|fields|check|run|plan)$/.test(selector)) return null;
      if (!/^#[\w-]+$/.test(selector)) return this.querySelectorAll(selector)[0] || null;
      if (!elements.has(selector)) elements.set(selector, Object.assign(new Element(), {id: selector.slice(1)}));
      return elements.get(selector);
    },
    querySelectorAll(selector) {
      return [...new Set([document.body, ...elements.values()].flatMap(walk))].filter(item => item.matches(selector));
    },
  });
  const timers = new Map();
  const frames = [];
  let nextTimer = 0;
  let blob;
  const sandbox = vm.createContext({
    document, Element, structuredClone, Blob,
    URL: { createObjectURL(value) { blob = value; return "blob:test"; }, revokeObjectURL() {} },
    CSS: { escape: value => value },
    setTimeout(callback) { timers.set(++nextTimer, callback); return nextTimer; },
    clearTimeout(id) { timers.delete(id); },
    requestAnimationFrame(callback) { frames.push(callback); return frames.length; },
    fetch: () => new Promise(() => {}),
    Option: function (caption, value) { const option = new Element("option"); option.textContent = caption; option.value = value; return option; },
  });
  const run = code => vm.runInContext(code, sandbox);
  for (const file of ["node_editing.js", "loop.js", "multiple_loops.js", "app.js"]) run(fs.readFileSync(path.join(__dirname, "../object_inpainting_studio/static", file), "utf8"));
  run(`
    render = () => {};
    batch.requesting = false; batch.restoring = false;
    state.config = {};
    const info = (inputs, outputs) => ({inputs, outputs, help: {title: 'test', inputs, outputs, parameters: []}});
    state.catalog = {
      Source: info({}, {image: 'IMAGE'}),
      Generate: info({image: 'IMAGE', seed: 'INT'}, {image: 'IMAGE'}),
      Score: info({image: 'IMAGE'}, {value: 'FLOAT', count: 'INT', ok: 'BOOL', valid: 'BOOLEAN', text: 'STRING'}),
      SaveImagesNode: info({image: 'IMAGE'}, {paths: 'JSON_RECORD'}),
      FolderImagesNode: info({}, {image: 'IMAGE'}),
    };
    state.workflow = {nodes: {
      fixed: {class_type: 'Source', inputs: {}},
      gen: {class_type: 'Generate', inputs: {image: link('fixed', 'image'), seed: 12}},
      alt: {class_type: 'Generate', inputs: {image: link('fixed', 'image'), seed: 23}},
      score: {class_type: 'Score', inputs: {image: link('gen', 'image'), other: link('alt', 'image')}},
      save: {class_type: 'SaveImagesNode', inputs: {img1: link('gen', 'image'), dir: '/out'}},
      save2: {class_type: 'SaveImagesNode', inputs: {img1: link('gen', 'image'), dir: '/out2'}},
    }, outputs: {}, conditional_loop: {
      start_nodes: ['gen'], seed_nodes: ['gen'],
      conditions: [{node: 'score', output: 'value', operator: '<=', value: 0.3}],
      mode: 'all', max_iterations: 10, timeout_seconds: 0, on_exhausted: 'skip',
    }};
    conditionalUI.targetId = 'save';
    normalizeImportedConditional();
    // The real app normalizes dynamic save ports while first rendering the canvas.
    // Do it before snapshots/plan keys so read-only panel renders do not migrate fixtures.
    Object.keys(state.workflow.nodes).forEach(id => nodeInfo(id));
    globalThis.requests = [];
    globalThis.reply = {status: 'completed', id: 'job-1', total: 1, processed: 1, succeeded: 1, failed: 0, results: []};
    fetch = async (url, options = {}) => {
      const body = options.body ? JSON.parse(options.body) : undefined;
      requests.push({url, body});
      return {ok: true, text: async () => JSON.stringify(reply)};
    };
  `);
  return { run, document, timers, frames, blob: () => blob };
}

// Render the real node/card and dispatch its registered handlers, not a copy of them.
function canvasCard(ui, id) {
  ui.run(`renderNode(${JSON.stringify(id)}, state.workflow.nodes[${JSON.stringify(id)}], 0)`);
  return ui.document.querySelector("#nodes").children.at(-1);
}

function clickPort(ui, id, output, direction = "output") {
  const type = ui.run(`nodeInfo(${JSON.stringify(id)})[${JSON.stringify(`${direction}s`)}][${JSON.stringify(output)}]`);
  assert.ok(type, `${id}.${output} must be a real ${direction} port`);
  const row = ui.run(`portRow(${JSON.stringify(id)}, ${JSON.stringify(output)}, ${JSON.stringify(type)}, ${JSON.stringify(direction)})`);
  const port = row.querySelector(".port");
  const event = {type: "click"};
  port.dispatchEvent(event);
  assert.equal(event.propagationStopped, true);
  return port;
}

function conditionalButton(ui, caption) {
  ui.run("renderConditional()");
  const button = walk(ui.document.querySelector("#multiple-loops")).find(element => element.tagName === "button" && element.textContent === caption);
  assert.ok(button, caption);
  return button;
}

function folder(ui) {
  ui.run(`
    state.workflow.nodes.folder = {class_type: 'FolderImagesNode', inputs: {dir: '/in', index: 0}};
    state.workflow.nodes.gen.inputs.image = link('folder', 'image');
    loopUI.sourceId = 'folder'; loopUI.targetId = 'save'; loopUI.enabled = true;
    loopUI.plan = {total: 3, start_index: 0, current_path: '/in/a.png', output_dir: '/out',
      executed_nodes: ['fixed', 'alt', 'folder', 'gen', 'score', 'save', 'save2'],
      repeated_nodes: ['folder', 'gen', 'score', 'save', 'save2'], terminal_nodes: ['score', 'save', 'save2'],
      node_roles: {fixed: 'fixed', alt: 'fixed', folder: 'source', gen: 'retry', score: 'retry', save: 'result', save2: 'result'}};
    loopUI.planKey = JSON.stringify(loopPayload());
  `);
}

test("folder scope includes conditional scoring branches outside save ancestry", () => {
  const ui = controller(); folder(ui);
  assert.equal(ui.run("loopScope().ancestors.has('score')"), true);
  assert.equal(ui.run("loopScope().repeated.has('score')"), true);
  assert.equal(ui.run("loopScope().ancestors.has('alt')"), true);
  assert.equal(ui.run("loopScope().repeated.has('alt')"), false);
  const score = canvasCard(ui, "score");
  ui.document.querySelector("#loop-region").append(ui.document.createElement("span"));
  ui.run("renderLoop()");
  assert.equal(score.querySelector(".loop-role").textContent, "条件重试");
  assert.equal(score.classList.contains("loop-outside"), false);
  ui.run("delete state.workflow.conditional_loop");
  assert.equal(ui.run("loopScope().repeated.has('score')"), false);
  assert.equal(ui.run("loopScope().plan"), null);
});

test("batch roles follow the backend plan and stale plans never color branches", () => {
  const ui = controller(); folder(ui);
  ui.run("loopUI.plan.executed_nodes = ['folder', 'gen', 'save']; loopUI.plan.repeated_nodes = ['folder', 'gen', 'save']; delete loopUI.plan.node_roles.score");
  assert.equal(ui.run("loopScope().ancestors.has('score')"), false);
  ui.run("state.workflow.nodes.gen.inputs.seed = 99");
  assert.equal(ui.run("loopScope().order.length"), 0);
});

test("an older batch plan response cannot replace the current execution scope", async () => {
  const ui = controller(); folder(ui);
  ui.run(`globalThis.finishPlans = []; loopUI.planKey = null;
    fetch = () => new Promise(resolve => finishPlans.push(plan => resolve({ok: true, text: async () => JSON.stringify(plan)})));
    renderLoop()`);
  const oldRequest = ui.timers.get(ui.run("loopUI.timer"))();
  ui.run("state.workflow.nodes.gen.inputs.seed = 77; renderLoop()");
  const currentRequest = ui.timers.get(ui.run("loopUI.timer"))();
  ui.run("finishPlans[1]({executed_nodes:['folder','gen','save'], repeated_nodes:['folder','gen','save'], node_roles:{folder:'source',gen:'per_item',save:'result'}, terminal_nodes:['save']})");
  await currentRequest;
  ui.run("finishPlans[0]({executed_nodes:['score'], repeated_nodes:['score'], node_roles:{score:'result'}, terminal_nodes:['score']})");
  await oldRequest;
  assert.deepEqual(plain(ui.run("loopScope().order")), ['folder', 'gen', 'save']);
});

test("result and disabled controls persist metadata, invalidate plans and respect running lock", () => {
  const ui = controller(); folder(ui);
  const controls = ui.run("batchNodeControls('score')");
  controls.querySelector(".result_nodes").click();
  assert.equal(ui.run("state.workflow.batch_execution.mode"), "selected_results");
  assert.deepEqual(plain(ui.run("state.workflow.batch_execution.result_nodes")), ['score']);
  assert.equal(ui.run("loopUI.plan"), null);
  ui.run("batchNodeControls('alt').querySelector('.disabled_nodes').click()");
  assert.deepEqual(plain(ui.run("state.workflow.batch_execution.disabled_nodes")), ['alt']);
  ui.run("batch.job = {status: 'running'}; batchNodeControls('gen').querySelector('.result_nodes').click()");
  assert.deepEqual(plain(ui.run("state.workflow.batch_execution.result_nodes")), ['score']);
  ui.run("batch.job = null; deleteNode('score')");
  assert.deepEqual(plain(ui.run("state.workflow.batch_execution.result_nodes")), []);
});

test("folder trial applies every terminal output to its canvas node", async () => {
  const ui = controller(); folder(ui);
  ui.document.querySelector("#loop-region").append(ui.document.createElement("span"));
  ui.run("delete state.workflow.conditional_loop");
  canvasCard(ui, "save");
  ui.run(`reply = {outputs: {paths: {kind: 'text', value: '/out/a.png'}},
    executed_nodes: ['folder', 'gen', 'score', 'save', 'save2'],
    terminal_outputs: {score: {value: {kind: 'text', value: 42}}}}`);
  await ui.run("previewNode('save', true)");
  assert.equal(ui.run("requests[0].url"), "/api/batch/preview");
  assert.equal(ui.run("state.nodeResults.score.value.value"), 42);
  assert.equal(ui.run("state.nodeResultStale.score"), undefined);
});

test("repeat boundary uses explicit multi-roots and relevant descendants, never the save", () => {
  const ui = controller();
  assert.deepEqual(plain(ui.run("[...multipleBody(multipleLoops()[0])].sort()")), ["gen", "score"]);
  ui.run("multipleLoops()[0].start_nodes.push('alt'); multipleLoops()[0].seed_nodes.push('alt');");
  assert.deepEqual(plain(ui.run("[...multipleBody(multipleLoops()[0])].sort()")), ["alt", "gen", "score"]);
  assert.equal(ui.run("conditionalValidation()"), "");
});

test("scalar comparisons and iteration bounds are validated on loop cards", () => {
  const ui = controller();
  for (const output of ["value", "count", "ok", "valid"]) {
    const boolean = ["ok", "valid"].includes(output);
    for (const operator of boolean ? ["==", "!="] : ["<=", ">=", "<", ">", "==", "!="]) {
      ui.run(`Object.assign(multipleLoops()[0].conditions[0], {output: '${output}', operator: '${operator}', value: ${boolean ? "true" : "0.5"}})`);
      assert.equal(ui.run("conditionalValidation()"), "");
    }
  }
  for (const [max_iterations, timeout_seconds] of [[1, 0], [1000, 86400]]) {
    ui.run(`Object.assign(multipleLoops()[0], {max_iterations: ${max_iterations}, timeout_seconds: ${timeout_seconds}, mode: 'any', on_exhausted: 'fail'})`);
    assert.equal(ui.run("conditionalValidation()"), "");
  }
});

test("invalid card fields block without silently repairing user values", () => {
  for (const mutation of [
    "start_nodes = []", "start_nodes = ['missing']", "start_nodes = ['gen', 'gen']",
    "seed_nodes = ['alt']", "seed_nodes = ['missing']", "seed_nodes = ['gen', 'gen']",
    "conditions = []", "conditions[0].node = 'fixed'", "conditions[0].node = 'missing'",
    "conditions[0].output = 'text'", "conditions[0].operator = '='", "conditions[0].value = '0.5'",
    "conditions[0].value = Infinity", "conditions[0].value = true",
    "conditions[0] = {node: 'score', output: 'ok', operator: '<=', value: true}",
    "conditions[0] = {node: 'score', output: 'ok', operator: '==', value: 1}",
    "max_iterations = 0", "max_iterations = 1001", "max_iterations = 1.5",
    "timeout_seconds = -1", "timeout_seconds = 86401", "mode = 'other'", "on_exhausted = 'other'",
  ]) {
    const ui = controller(); ui.run(`multipleLoops()[0].${mutation}`);
    const before = ui.run("JSON.stringify(state.workflow.conditional_loop)");
    assert.notEqual(ui.run("conditionalValidation()"), "", mutation);
    ui.run("renderConditional()");
    assert.equal(ui.run("JSON.stringify(state.workflow.conditional_loop)"), before, mutation);
    assert.notEqual(ui.document.querySelector("#conditional-error").textContent, "", mutation);
  }
});

test("seed must be a nonnegative literal INT inside the loop", () => {
  for (const seed of ["link('fixed', 'image')", "null", "1.5", "'12'", "undefined", "-1"]) {
    const ui = controller(); ui.run(`state.workflow.nodes.gen.inputs.seed = ${seed}`);
    assert.match(ui.run("conditionalValidation()"), /非负 INT seed/);
  }
});

test("deleting folder target never silently selects the remaining save branch", async () => {
  const ui = controller(); folder(ui); ui.run("deleteNode('save'); renderLoop()");
  assert.equal(ui.run("loopUI.targetId"), "save");
  assert.equal(ui.run("conditionalUI.targetId"), "save");
  assert.match(ui.run("loopScope().problem"), /不会自动切换/);
  await ui.run("batchRequest('start', 'save2')");
  assert.equal(ui.run("requests.length"), 0);
});

test("null disables acceptance, Add replaces it, new workflow resets transient state", () => {
  const ui = controller(); ui.run("state.workflow.conditional_loop = null; renderConditional()");
  assert.equal(ui.run("conditionalEnabled()"), false);
  assert.equal(ui.document.querySelector("#conditional-check"), null);
  ui.document.querySelector("#conditional-add").click();
  assert.equal(ui.run("multipleLoops().length"), 1);
  ui.run("startPractice()");
  assert.equal(ui.run("conditionalEnabled()"), false);
  assert.equal(ui.run("conditionalUI.targetId"), null);
  assert.equal(ui.run("multipleUI.activeId"), null);
});

test("SaveImages automatically submits validation with exact payload and locks editing until response", async () => {
  const ui = controller();
  ui.run(`fetch = (url, options) => { requests.push({url, body: JSON.parse(options.body)}); return new Promise(resolve => {globalThis.finish = () => resolve({ok: true, text: async () => JSON.stringify(reply)});}); };`);
  const before = plain(ui.run("state.workflow"));
  const promise = ui.run("previewNode('save')");
  assert.equal(ui.run("batch.starting && batch.requesting"), true);
  assert.equal(ui.run("nodeDeletionDisabled()"), true);
  assert.equal(ui.document.querySelector("#conditional-add").disabled, true);
  assert.equal(ui.run("requests[0].url"), "/api/conditional");
  assert.deepEqual(plain(ui.run("Object.keys(requests[0].body).sort()")), ["config", "continue_on_error", "node_id", "record", "visualize", "workflow"]);
  assert.equal(ui.run("requests[0].body.node_id"), "save");
  assert.equal(ui.run("requests[0].body.visualize"), true);
  assert.deepEqual(plain(ui.run("requests[0].body.workflow")), before);
  assert.deepEqual(plain(ui.run("Object.keys(requests[0].body.workflow.conditional_loop)")), ["loops"]);
  ui.run("editMultiple(multipleLoops()[0].id, loop => {loop.max_iterations = 2})");
  assert.equal(ui.run("multipleLoops()[0].max_iterations"), 10);
  ui.run("finish()"); await promise;
  assert.equal(ui.run("batch.starting || batch.requesting"), false);
  assert.equal(ui.document.querySelector("#conditional-add").disabled, false);
});

test("server validation failure never falls back to ordinary save and requires status confirmation", async () => {
  const ui = controller();
  ui.run("fetch = async (url, options) => {requests.push({url, body: JSON.parse(options.body)}); return {ok: false, text: async () => JSON.stringify({detail: 'bad plan'})};}");
  await ui.run("previewNode('save')");
  assert.match(ui.run("batch.error"), /bad plan/);
  assert.equal(ui.run("batch.uncertain"), true);
  await ui.run("previewNode('save')");
  assert.deepEqual(plain(ui.run("requests.map(r => r.url)")), ["/api/conditional"]);
  assert.equal(ui.document.querySelector("#conditional-add").disabled, true);
  ui.run("reply = {job: null}; fetch = async () => ({ok: true, text: async () => JSON.stringify(reply)})");
  await ui.run("batchRequest('refresh')");
  assert.equal(ui.run("batch.uncertain"), false);
  assert.equal(ui.document.querySelector("#conditional-add").disabled, false);
});

test("single-image start shares batch state, poll, cancel and latest endpoints", async () => {
  const ui = controller(); ui.run("reply.status = 'running'");
  await ui.run("batchRequest('start-condition', 'save')");
  assert.equal(ui.run("requests[0].url"), "/api/conditional");
  assert.equal(ui.run("requests[0].body.node_id"), "save");
  assert.equal(ui.run("batch.job.id"), "job-1");
  assert.equal(ui.run("batchActive()"), true);
  await ui.run("batchRequest('start-condition', 'save')");
  assert.equal(ui.run("requests.length"), 1);
  await ui.run("batchRequest('poll')");
  assert.equal(ui.run("requests.at(-1).url"), "/api/batch/job-1");
  ui.run("reply.status = 'cancelling'"); await ui.run("batchRequest('cancel')");
  assert.equal(ui.run("requests.at(-1).url"), "/api/batch/job-1/cancel");
  ui.run("reply = {job: null}"); await ui.run("batchRequest('refresh')");
  assert.equal(ui.run("requests.at(-1).url"), "/api/batch/latest");
  assert.equal(ui.run("batch.job"), null);
});

test("submission uncertainty locks both starts until latest confirms idle", async () => {
  const ui = controller(); folder(ui);
  ui.run("fetch = async () => { throw new Error('lost response'); }");
  await ui.run("batchRequest('start-condition', 'save')");
  assert.equal(ui.run("batch.uncertain"), true);
  assert.match(ui.run("batch.error"), /任务可能已启动/);
  ui.run("fetch = async url => {requests.push({url}); return {ok: true, text: async () => JSON.stringify({job: null})};}");
  await ui.run("batchRequest('start', 'save')"); await ui.run("batchRequest('start-condition', 'save')");
  assert.equal(ui.run("requests.length"), 0);
  await ui.run("batchRequest('refresh')"); assert.equal(ui.run("batch.uncertain"), false);
});

test("submission locks synchronously without discarding successful previews", async () => {
  const ui = controller();
  ui.run("state.nodeResults.gen = {image: {kind: 'image', image: 'retained'}}; fetch = () => new Promise(resolve => {globalThis.finish = () => resolve({ok: true, text: async () => JSON.stringify(reply)});})");
  const pending = ui.run("batchRequest('start-condition', 'save')");
  assert.equal(ui.run("batch.starting && batch.requesting"), true);
  assert.equal(ui.run("state.nodeResults.gen.image.image"), "retained");
  assert.equal(ui.document.querySelector("#conditional-add").disabled, true);
  ui.run("finish()"); await pending;
  assert.equal(ui.run("batch.starting || batch.requesting"), false);
});

test("invalid acceptance, active previews and job checks cannot start", async () => {
  for (const setup of ["multipleLoops()[0].conditions = []", "loopUI.busy = true", "loopUI.checking = true", "conditionalUI.checking = true", "batch.job = {status: 'cancelling'}"]) {
    const ui = controller(); ui.run(setup);
    await ui.run("batchRequest('start-condition', 'save')"); assert.equal(ui.run("requests.length"), 0, setup);
  }
});

test("ordinary save cannot bypass malformed acceptance; valid save automatically uses conditional endpoint", async () => {
  const ui = controller();
  ui.run("multipleLoops()[0].conditions = []");
  await ui.run("previewNode('save')");
  assert.equal(ui.run("requests.length"), 0);
  ui.run("multipleLoops()[0].conditions = [{node: 'score', output: 'value', operator: '<=', value: 0.3}]; conditionalUI.targetId = 'save2'");
  await ui.run("previewNode('save')");
  assert.equal(ui.run("conditionalUI.targetId"), "save");
  assert.deepEqual(plain(ui.run("requests.map(r => r.url)")), ["/api/conditional"]);
  assert.equal(ui.run("requests[0].body.node_id"), "save");
});

test("folder batch carries loops metadata and conditional trial uses async job lock", async () => {
  const ui = controller(); folder(ui);
  await ui.run("batchRequest('start', 'save')");
  assert.equal(ui.run("requests[0].url"), "/api/batch");
  assert.deepEqual(plain(ui.run("requests[0].body.workflow.conditional_loop.loops[0].start_nodes")), ["gen"]);
  await ui.run("previewNode('save', true)");
  assert.equal(ui.run("requests[1].url"), "/api/batch/preview");
  assert.equal(ui.run("requests[1].body.node_id"), "save");
  ui.run("multipleLoops()[0].max_iterations = 4");
  await ui.run("batchRequest('start', 'save')"); assert.equal(ui.run("requests.length"), 2);
});

test("folder checks lock cards and release on API errors", async () => {
  const ui = controller(); folder(ui);
  ui.run("loopUI.planKey = null; fetch = () => new Promise(resolve => {globalThis.finish = () => resolve({ok: false, text: async () => JSON.stringify({detail: 'bad plan'})});}); renderLoop()");
  const pending = ui.timers.get(ui.run("loopUI.timer"))();
  assert.equal(ui.run("loopUI.checking"), true);
  assert.equal(ui.document.querySelector("#conditional-add").disabled, true);
  ui.run("finish()"); await pending;
  assert.equal(ui.run("loopUI.checking"), false);
  assert.equal(ui.document.querySelector("#conditional-add").disabled, false);
  assert.equal(ui.run("loopUI.error"), "bad plan");
});

test("poll rendering preserves focused card controls and uncommitted values", () => {
  const ui = controller(); ui.run("renderBatch()");
  const root = ui.document.querySelector("#multiple-loops");
  const input = walk(root).find(e => e.attributes["aria-label"] === "条件 1 比较值");
  input.value = "123"; input.focus(); const original = [...root.children];
  ui.run("batch.job = {id: 'j', status: 'completed', total: 1, processed: 1, succeeded: 1, failed: 0}; renderBatch(); renderBatch()");
  assert.equal(ui.document.activeElement, input); assert.equal(input.value, "123");
  assert.deepEqual(root.children, original);
});

test("cancel helper, panel and Escape preserve configuration and preview session", () => {
  for (const method of ["helper", "button", "Escape"]) {
    const ui = controller();
    ui.run("state.previewSession = 'existing-session'; beginMultiplePick(multipleLoops()[0].id, 'condition', 0)");
    const before = ui.run("JSON.stringify([state.workflow, state.previewRevision, state.previewSession])");
    if (method === "helper") ui.run("cancelConditionalPick()");
    if (method === "button") conditionalButton(ui, "取消点选 (Esc)").click();
    if (method === "Escape") {
      const event = {type: "keydown", key: "Escape", target: ui.document.body}; ui.document.dispatchEvent(event);
      assert.equal(event.defaultPrevented, true);
    }
    assert.equal(ui.run("conditionalUI.picking"), null);
    assert.equal(ui.run("JSON.stringify([state.workflow, state.previewRevision, state.previewSession])"), before);
  }
});

test("actual port edit invalidates folder plan but preserves history image identity and session", () => {
  const ui = controller();
  ui.run("state.previewSession = 'reuse'; state.nodeResults.gen = {image: {kind: 'image', image: 'retained'}}; globalThis.retained = state.nodeResults.gen.image; state.previewHidden.gen = true; loopUI.plan = {}; loopUI.planKey = 'old'");
  const revision = ui.run("state.previewRevision");
  conditionalButton(ui, "重选条件").click(); clickPort(ui, "score", "count");
  assert.equal(ui.run("conditionalUI.plan"), null); assert.equal(ui.run("loopUI.plan"), null);
  assert.equal(ui.run("conditionalUI.planKey"), null); assert.equal(ui.run("loopUI.planKey"), null);
  assert.equal(ui.run("state.nodeResults.gen.image === retained"), true);
  assert.equal(ui.run("state.nodeResultStale.gen"), true);
  assert.equal(ui.run("state.previewHidden.gen"), true);
  assert.equal(ui.run("state.previewSession"), "reuse");
  assert.equal(ui.run("state.previewRevision"), revision + 1);
});

test("progress retains loop identity, scores, seeds, stop reason and skipped-not-saved", () => {
  const ui = controller();
  ui.run(`batch.job = {id: 'j', status: 'completed', total: 1, processed: 1, succeeded: 0, failed: 0, skipped: 1,
    current_iteration: 3, max_iterations: 3, stop_reason: 'max_iterations',
    iteration_history: [{loop_id: 'loop_1', loop_name: '已运行', iteration: 3, seeds: {gen: 14}, scores: [{node: 'score', output: 'value', value: 0.8, passed: false}], accepted: false, elapsed_seconds: 1.2}],
    results: [{source: 'current', accepted: false, iterations: 3, stop_reason: 'max_iterations', paths: {}, history: []}]}; renderBatch()`);
  assert.match(ui.document.querySelector("#batch-counts").textContent, /跳过 1/);
  assert.match(ui.document.querySelector("#batch-current").textContent, /条件尝试：3 \/ 3/);
  const attempt = ui.document.querySelector("#conditional-attempts").children[0].textContent;
  assert.match(attempt, /已运行 \(loop_1\)/); assert.match(attempt, /score.value = 0.8 · 未通过/); assert.match(attempt, /"gen":14/);
  const result = ui.document.querySelector("#batch-results").children[0].textContent;
  assert.match(result, /已跳过 · 未保存/); assert.doesNotMatch(result, /已写入/);
});

test("download/import preserves loops-only schema and explicit shared target", async () => {
  const ui = controller(); const before = plain(ui.run("state.workflow.conditional_loop"));
  ui.run("saveWorkflow()"); const saved = JSON.parse(await ui.blob().text());
  assert.deepEqual(saved.workflow.conditional_loop, before);
  assert.deepEqual(Object.keys(saved.workflow.conditional_loop), ["loops"]);
  assert.equal(saved.loop.conditionalTargetId, "save");
  ui.run("fetch = async (url, options) => ({ok: true, text: async () => JSON.stringify({workflow: JSON.parse(options.body).workflow, config: {}})})");
  ui.run(`globalThis.importEvent = {target: {files: [{name: 'retry.json', text: async () => ${JSON.stringify(JSON.stringify(saved))}}]}}`);
  await ui.run("loadWorkflow(importEvent)");
  assert.deepEqual(plain(ui.run("state.workflow.conditional_loop")), before);
  assert.equal(ui.run("conditionalUI.targetId"), "save");
  assert.equal(ui.run("conditionalUI.picking"), null);
});

for (const nodeId of ["gen", "score", "bridge", "downstream"]) {
  for (const visualize of [true, false]) {
    test(`actual previewNode(${nodeId}) uses an async conditional job with visualize=${visualize}`, async () => {
      const ui = controller();
      ui.run(`
        state.workflow.nodes.bridge = {class_type: 'Generate', inputs: {image: link('gen', 'image'), seed: 0}};
        state.workflow.nodes.downstream = {class_type: 'Score', inputs: {image: link('bridge', 'image')}};
        ${visualize ? "" : "state.visualize = false;"}
        state.config = {demo_limit: 0}; document.querySelector('#config-demo_limit').value = '7';
        document.querySelector('#target-path').value = 'input.png';
        document.querySelector('#mask-path').value = 'mask.png';
        document.querySelector('#object-name').value = '对象';
        document.querySelector('#batch-continue-on-error').checked = true;
        state.previewSession = 'must-not-reuse-unaccepted-cache'; reply.status = 'running';
      `);
      const before = plain(ui.run("state.workflow"));
      await ui.run(`previewNode('${nodeId}')`);
      assert.deepEqual(plain(ui.run("requests")), [{url: "/api/node/preview", body: {
        workflow: before, visualize, node_id: nodeId, continue_on_error: true,
        config: {demo_limit: 7}, record: {target_path: "input.png", mask_path: "mask.png", object_name: "对象"},
      }}]);
      assert.deepEqual(plain(ui.run("state.workflow")), before);
      assert.equal(ui.run("batchActive()"), true);
      assert.equal(ui.run("batch.uncertain"), false);
      assert.equal(ui.run("state.activeJobId"), "job-1");
      assert.equal(ui.run("state.activeJobRevision === state.previewRevision"), true);
      assert.equal(ui.run("state.selected"), nodeId);
      assert.equal(ui.run("state.previewSession"), null);
    });
  }
}

test("an unrelated ordinary node remains synchronous and carries the existing preview session", async () => {
  const ui = controller();
  const card = canvasCard(ui, "alt");
  ui.run(`state.previewSession = 'reuse-fixed-input';
    fetch = (url, options) => {
      requests.push({url, body: JSON.parse(options.body)});
      return new Promise(resolve => {globalThis.finish = () => resolve({ok: true, text: async () => JSON.stringify({
        outputs: {image: {kind: 'image', image: 'data:image/png;base64,unrelated'}},
        executed_nodes: ['fixed', 'alt'], reused_nodes: [], preview_session: 'still-synchronous'
      })});});
    };`);
  const before = plain(ui.run("state.workflow"));
  const pending = ui.run("previewNode('alt')");
  assert.equal(ui.run("loopUI.busy"), true);
  assert.equal(ui.run("batch.starting || batch.requesting"), false);
  assert.equal(card.querySelector(".node-preview-button").disabled, true);
  assert.equal(ui.run("requests[0].url"), "/api/node/preview");
  assert.deepEqual(plain(ui.run("requests[0].body")), {
    workflow: before, visualize: true, node_id: 'alt', preview_session: 'reuse-fixed-input',
    config: {}, record: {target_path: '', mask_path: '', object_name: ''},
  });
  ui.run("finish()"); await pending;
  assert.equal(ui.run("loopUI.busy"), false);
  assert.equal(ui.run("batch.job"), null);
  assert.equal(ui.run("state.previewSession"), "still-synchronous");
  assert.equal(ui.run("state.nodeResults.alt.image.image"), "data:image/png;base64,unrelated");
});
