const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");
const plain = value => JSON.parse(JSON.stringify(value));
const walk = el => [el, ...el.children.flatMap(walk)];

// Standalone DOM / VM fixture; dispatch actual app title / port / panel handlers.
function controller() {
  class Events {
    constructor() { this.listeners = {}; }
    addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); }
    dispatchEvent(event) {
      event.target ??= this;
      event.preventDefault = () => { event.defaultPrevented = true; };
      event.stopPropagation = () => { event.propagationStopped = true; };
      this[`on${event.type}`]?.(event);
      (this.listeners[event.type] || []).forEach(fn => fn(event));
      return !event.defaultPrevented;
    }
  }
  class Element extends Events {
    constructor(tag = "div") {
      super(); this.tagName = tag; this.children = []; this.dataset = {}; this.attributes = {}; this.value = "";
      this.checked = false;
      this.style = { setProperty(name, value) { this[name] = value; } };
      this.classList = {
        contains: name => (this.className || "").split(/\s+/).includes(name),
        toggle: (name, force) => {
          const names = new Set((this.className || "").split(/\s+/).filter(Boolean));
          const enabled = force ?? !names.has(name);
          if (enabled) names.add(name); else names.delete(name);
          this.className = [...names].join(" "); return enabled;
        },
        add: name => this.classList.toggle(name, true), remove: name => this.classList.toggle(name, false),
      };
    }
    set value(v) { this._value = String(v); }
    get value() { return this._value; }
    get options() { return this.children; }
    get selectedOptions() { return this.children.filter(o => o.selected); }
    append(...children) { children.forEach(c => { c.parentElement = this; this.children.push(c); }); }
    replaceChildren(...children) { this.children.forEach(child => { child.parentElement = null; }); this.children = []; this.append(...children); }
    add(child) { this.append(child); }
    setAttribute(name, value) { this.attributes[name] = value; }
    matches(selector) {
      return selector.split(",").some(part => {
        const parts = part.trim().split(/\s+/);
        if (parts.length > 1) {
          if (!this.matches(parts.pop())) return false;
          for (let p = this.parentElement; p; p = p.parentElement) if (p.matches(parts.join(" "))) return true;
          return false;
        }
        const s = parts[0], id = s.match(/^#([\w-]+)/)?.[1], tag = s.match(/^[a-z]+/i)?.[0];
        if (id && this.id !== id || tag && this.tagName !== tag) return false;
        if ([...s.matchAll(/\.([\w-]+)/g)].some(([, cls]) => !this.classList.contains(cls))) return false;
        return [...s.matchAll(/\[([\w-]+)(?:="([^"]*)")?\]/g)].every(([, attr, value]) => {
          const actual = attr.startsWith("data-") ? this.dataset[attr.slice(5).replace(/-([a-z])/g, (_, c) => c.toUpperCase())] : this.attributes[attr];
          return value == null ? actual != null : actual === value;
        });
      });
    }
    querySelectorAll(s) { return walk(this).slice(1).filter(c => c.matches(s)); }
    querySelector(s) { return this.querySelectorAll(s)[0] || null; }
    closest(s) { return this.matches(s) ? this : this.parentElement?.closest(s) || null; }
    click() { if (!this.disabled) this.dispatchEvent({ type: "click" }); }
    focus() { document.activeElement = this; }
    scrollIntoView() {}
    remove() {
      if (!this.parentElement) return;
      const siblings = this.parentElement.children;
      const index = siblings.indexOf(this);
      if (index !== -1) siblings.splice(index, 1);
      this.parentElement = null;
    }
  }
  const elements = new Map();
  const document = Object.assign(new Events(), {
    body: new Element(), createElement: tag => new Element(tag),
    querySelector(s) {
      if (/^#conditional-(enabled|legacy|legacy-controls|fields|check|run|plan)$/.test(s)) return null;
      if (!/^#[\w-]+$/.test(s)) return this.querySelectorAll(s)[0] || null;
      if (!elements.has(s)) elements.set(s, Object.assign(new Element(), { id: s.slice(1) }));
      return elements.get(s);
    },
    querySelectorAll(s) { return [...new Set([document.body, ...elements.values()].flatMap(walk))].filter(c => c.matches(s)); },
  });
  let blob, timer = 0;
  const sandbox = vm.createContext({
    document, structuredClone, Blob, CSS: { escape: v => v },
    URL: { createObjectURL(v) { blob = v; return "blob:test"; }, revokeObjectURL() {} },
    setTimeout: () => ++timer, clearTimeout() {}, requestAnimationFrame() {},
    fetch: () => new Promise(() => {}),
    Option: function(textContent, value) { return Object.assign(new Element("option"), { textContent, value }); },
  });
  const run = code => vm.runInContext(code, sandbox);
  for (const file of ["node_editing.js", "loop.js", "multiple_loops.js", "app.js"]) run(fs.readFileSync(path.join(__dirname, "../object_inpainting_studio/static", file), "utf8"));
  run(`
    render = () => {}; batch.requesting = false; batch.restoring = false;
    state.config = {};
    const info = (inputs, outputs) => ({inputs, outputs, help: {title: '同类测试节点', inputs, outputs, parameters: []}});
    state.catalog = {
      Source: info({}, {image: 'IMAGE'}),
      Generate: info({image: 'IMAGE', seed: 'INT'}, {image: 'IMAGE'}),
      Score: info({image: 'IMAGE'}, {value: 'FLOAT'}),
      MultiScore: info({image: 'IMAGE'}, {value: 'FLOAT', ok: 'BOOL', text: 'STRING'}),
      BoolScore: info({image: 'IMAGE'}, {ok: 'BOOLEAN'}),
      SaveImagesNode: info({image: 'IMAGE'}, {paths: 'JSON_RECORD'}),
      FolderImagesNode: info({}, {image: 'IMAGE'}),
      LoopDemoImageNode: info({seed: 'INT'}, {image: 'RGB_IMAGE'}),
      LoopDemoScoreNode: info({image: 'RGB_IMAGE'}, {score: 'FLOAT'}),
    };
    state.workflow = {nodes: {
      fixed: {class_type: 'Source', inputs: {}},
      gen: {class_type: 'Generate', inputs: {image: link('fixed', 'image'), seed: 12}},
      alt: {class_type: 'Generate', inputs: {image: link('fixed', 'image'), seed: -1}},
      score: {class_type: 'Score', inputs: {image: link('gen', 'image')}},
      score2: {class_type: 'Score', inputs: {image: link('alt', 'image')}},
      multi: {class_type: 'MultiScore', inputs: {image: link('gen', 'image')}},
      bool: {class_type: 'BoolScore', inputs: {image: link('gen', 'image')}},
      save: {class_type: 'SaveImagesNode', inputs: {img1: link('gen', 'image'), img2: link('alt', 'image'), dir: '/out'}},
    }, outputs: {}};
    Object.keys(state.workflow.nodes).forEach(id => nodeInfo(id));
    state.previewSession = 'keep-session';
    state.nodeResults = {gen: {outputs: {image: {type: 'image', data: 'old'}}}};
    globalThis.requests = [];
    fetch = async (url, options = {}) => {
      requests.push({url, body: options.body ? JSON.parse(options.body) : null});
      return {ok: true, text: async () => JSON.stringify({target_id: 'save', loops: [], body_nodes: ['gen'], fixed_nodes: [], max_iterations: 10})};
    };
    renderConditional();
  `);
  const ui = { run, document, blob: () => blob };
  ui.button = (text, loopId) => {
    run("renderConditional()");
    const root = loopId ? document.querySelectorAll(".multiple-loop-card").find(c => c.dataset.loopId === loopId) : document.querySelector("#multiple-loops");
    const found = walk(root).find(c => c.tagName === "button" && c.textContent === text);
    assert.ok(found, text); return found;
  };
  ui.field = (label, loopId) => {
    run("renderConditional()");
    const root = document.querySelectorAll(".multiple-loop-card").find(c => c.dataset.loopId === loopId);
    const found = walk(root).find(c => c.attributes["aria-label"] === label);
    assert.ok(found, label); return found;
  };
  ui.add = () => { document.querySelector("#conditional-add").click(); return run("multipleLoops().at(-1).id"); };
  ui.header = id => {
    document.querySelectorAll(`.node[data-node-id="${id}"]`).forEach(card => card.remove());
    run(`renderNode(${JSON.stringify(id)}, state.workflow.nodes[${JSON.stringify(id)}], 0)`);
    const header = document.querySelector("#nodes").children.at(-1).querySelector(".node-header");
    header.dispatchEvent({type: "pointerdown"}); header.click(); return header;
  };
  ui.port = (id, output, direction = "output") => {
    const row = run(`portRow('${id}', '${output}', nodeInfo('${id}').${direction}s['${output}'], '${direction}')`);
    const event = {type: "click"}; row.querySelector(".port").dispatchEvent(event);
    assert.equal(event.propagationStopped, true); return row;
  };
  ui.complete = (start = "gen", end = start, condition = start === "gen" ? "score" : "score2") => {
    const id = ui.add(); ui.header(start); ui.header(end); ui.header(condition); return id;
  };
  return ui;
}

test("empty state exposes only Add; production has no legacy DOM, handlers or flat renderer", () => {
  const ui = controller();
  assert.equal(ui.run("conditionalEnabled()"), false);
  assert.match(ui.document.querySelector("#multiple-loops").children[0].textContent, /暂无循环/);
  assert.equal(ui.document.querySelector("#conditional-add").disabled, false);
  const html = fs.readFileSync(path.join(__dirname, "../object_inpainting_studio/static/index.html"), "utf8");
  const scripts = ["loop.js", "multiple_loops.js"].map(file => fs.readFileSync(path.join(__dirname, "../object_inpainting_studio/static", file), "utf8")).join("\n");
  for (const id of ["conditional-enabled", "conditional-legacy", "conditional-legacy-controls", "conditional-fields", "conditional-check", "conditional-run", "conditional-plan"]) {
    assert.equal(ui.document.querySelector(`#${id}`), null);
    assert.doesNotMatch(html, new RegExp(`id="${id}"`));
    assert.doesNotMatch(scripts, new RegExp(id));
  }
  assert.doesNotMatch(scripts, /function (renderConditionalFields|renderConditionalRoles|beginConditionalPick|editConditional|conditionalReferences|conditionalPickHint)\(/);
  assert.doesNotMatch(scripts, /\b(checkConditionalPlan|conditionalPlanKey|conditionalPayload)\b/);
  assert.match(html, /<summary>↻ 条件循环/);
  assert.ok(html.indexOf('src="/static/loop.js"') < html.indexOf('src="/static/multiple_loops.js"'));
  assert.ok(html.indexOf('src="/static/multiple_loops.js"') < html.indexOf('src="/static/app.js"'));
});

test("Add starts title-only wizard and advances start → ordinary end → single scalar condition", () => {
  const ui = controller(), id = ui.add();
  assert.equal(ui.run("conditionalUI.picking.kind"), "start_nodes");
  assert.equal(ui.run("state.connecting"), null);
  ui.header("save"); assert.equal(ui.run("conditionalUI.picking.kind"), "start_nodes");
  ui.header("gen"); assert.equal(ui.run("conditionalUI.picking.kind"), "end_nodes");
  ui.header("save"); assert.equal(ui.run("conditionalUI.picking.kind"), "end_nodes");
  ui.header("gen"); assert.equal(ui.run("conditionalUI.picking.kind"), "condition");
  ui.header("score"); assert.equal(ui.run("conditionalUI.picking"), null);
  assert.deepEqual(plain(ui.run(`multipleLoop('${id}').conditions`)), [{node: "score", output: "value", operator: "<=", value: 10}]);
  assert.equal(ui.run("conditionalUI.targetId"), "save");
  assert.equal(ui.run("conditionalValidation()"), "");
});

test("two same-type instances only change their own ID-bound loop on reselect", () => {
  const ui = controller(), a = ui.complete(), b = ui.complete("alt");
  const before = ui.run(`JSON.stringify(multipleLoop('${a}'))`);
  ui.button("重选起点", b).click(); ui.header("gen");
  assert.equal(ui.run(`JSON.stringify(multipleLoop('${a}'))`), before);
  assert.deepEqual(plain(ui.run(`multipleLoop('${b}').start_nodes`)), ["gen"]);
  assert.equal(ui.run("conditionalUI.picking"), null);
  assert.match(ui.button("起点 · gen · 同类测试节点 (Generate)", b).textContent, /gen.*同类测试节点.*Generate/);
});

test("multi-scalar header asks for output, rejects image/text and accepts real scalar port", () => {
  const ui = controller(), id = ui.add(); ui.header("gen"); ui.header("gen");
  ui.header("multi"); assert.equal(ui.run("conditionalUI.picking.kind"), "condition");
  assert.match(ui.document.querySelector("#status").textContent, /多个/);
  ui.port("multi", "text"); assert.equal(ui.run("conditionalUI.picking.kind"), "condition");
  ui.port("gen", "image"); assert.equal(ui.run("conditionalUI.picking.kind"), "condition");
  ui.port("multi", "ok");
  assert.deepEqual(plain(ui.run(`multipleLoop('${id}').conditions[0]`)), {node: "multi", output: "ok", operator: "==", value: true});
  assert.equal(ui.run("conditionalUI.picking"), null);
});

test("condition type switches reset numeric/bool defaults and still validate comparisons", () => {
  const ui = controller(), id = ui.complete();
  ui.button("重选条件", id).click(); ui.header("bool");
  assert.equal(ui.run(`multipleLoop('${id}').conditions[0].value`), true);
  let op = ui.field("条件 1 运算符", id); op.value = "<="; op.onchange();
  assert.match(ui.run("conditionalValidation()"), /布尔条件/);
  ui.button("重选条件", id).click(); ui.header("score");
  assert.deepEqual(plain(ui.run(`multipleLoop('${id}').conditions[0]`)), {node: "score", output: "value", operator: "<=", value: 10});
  let value = ui.field("条件 1 比较值", id); value.value = "true"; value.onchange();
  assert.match(ui.run("conditionalValidation()"), /有限数字/);
});

test("auto seed uses only nonnegative literal start seed and never rewrites -1", () => {
  const ui = controller(), a = ui.complete(), b = ui.complete("alt");
  assert.deepEqual(plain(ui.run(`multipleLoop('${a}').seed_nodes`)), ["gen"]);
  assert.deepEqual(plain(ui.run(`multipleLoop('${b}').seed_nodes`)), []);
  assert.equal(ui.run("state.workflow.nodes.alt.inputs.seed"), -1);
  assert.equal(ui.run("conditionalValidation()"), "");
  ui.run(`state.workflow.nodes.gen.inputs.seed = link('fixed', 'image');`);
  assert.match(ui.run("conditionalValidation()"), /非负 INT seed/);
});

test("no-seed start works without exposing a seed picker", () => {
  const ui = controller(), id = ui.add();
  ui.header("score"); ui.header("score"); ui.header("score");
  assert.deepEqual(plain(ui.run(`multipleLoop('${id}').seed_nodes`)), []);
  ui.run(`beginMultiplePick('${id}', 'seed_nodes')`);
  assert.equal(ui.run("conditionalUI.picking"), null);
  assert.deepEqual(plain(ui.run(`multipleLoop('${id}').seed_nodes`)), []);
  assert.equal(ui.run("state.workflow.nodes.alt.inputs.seed"), -1);
});

test("deleting active loop exits picker and preserves other loop, graph, links and historical previews", () => {
  const ui = controller(), a = ui.complete(), b = ui.complete("alt");
  const graph = ui.run("JSON.stringify(state.workflow.nodes)"), other = ui.run(`JSON.stringify(multipleLoop('${b}'))`);
  const preview = ui.run("JSON.stringify(state.nodeResults)");
  ui.button("重选终点", a).click(); ui.button("删除循环", a).click();
  assert.equal(ui.run("conditionalUI.picking"), null);
  assert.equal(ui.run("JSON.stringify(state.workflow.nodes)"), graph);
  assert.equal(ui.run(`JSON.stringify(multipleLoop('${b}'))`), other);
  assert.equal(ui.run("JSON.stringify(state.nodeResults)"), preview);
  assert.equal(ui.run("state.previewSession"), "keep-session");
  ui.button("重选起点", b).click(); ui.header("alt");
  assert.deepEqual(plain(ui.run(`multipleLoop('${b}').start_nodes`)), ["alt"]);
});

test("deleting another loop does not cancel active pick or retarget by array index", () => {
  const ui = controller(), a = ui.complete(), b = ui.complete("alt");
  ui.button("重选终点", b).click(); ui.button("删除循环", a).click();
  assert.equal(ui.run("conditionalUI.picking.loopId"), b);
  ui.header("score2");
  assert.deepEqual(plain(ui.run(`multipleLoop('${b}').end_nodes`)), ["score2"]);
});

test("deleting final loop restores ordinary save and never reuses IDs", () => {
  const ui = controller(), id = ui.complete(); ui.button("删除循环", id).click();
  assert.equal(ui.run("Object.hasOwn(state.workflow, 'conditional_loop')"), false);
  assert.equal(ui.run("conditionalUI.picking"), null);
  assert.notEqual(ui.add(), id);
});

function legacy(ui) {
  ui.run(`state.workflow.conditional_loop = {
    start_nodes: ['gen'], seed_nodes: ['gen'], conditions: [{node: 'score', output: 'value', operator: '<', value: 0.3}],
    mode: 'any', max_iterations: 7, timeout_seconds: 12, on_exhausted: 'skip'
  }; conditionalUI.targetId = 'save'; globalThis.originalLegacy = state.workflow.conditional_loop; renderConditional();`);
}

test("legacy auto migration preserves settings and original object; Add appends after migration", () => {
  const ui = controller(); legacy(ui);
  const old = plain(ui.run("originalLegacy"));
  assert.equal(ui.run("multipleLoops().length"), 1);
  ui.add();
  const migrated = plain(ui.run("multipleLoops()[0]"));
  for (const [key, value] of Object.entries(old)) assert.deepEqual(migrated[key], value);
  assert.deepEqual(migrated.end_nodes.sort(), ["gen", "score"]);
  assert.equal(migrated.name, "导入的循环");
  assert.deepEqual(plain(ui.run("originalLegacy")), old);
  assert.deepEqual(plain(ui.run("Object.keys(state.workflow.conditional_loop)")), ["loops"]);
  assert.equal(ui.run("multipleLoops().length"), 2);
});

test("invalid legacy start still becomes a repairable card and does not block Add", () => {
  const ui = controller();
  ui.run("state.workflow.conditional_loop = {start_nodes: ['missing'], conditions: [{node: 'deleted-score', output: 'value', operator: '<', value: 3}]}; renderConditional()");
  assert.equal(ui.run("isMultipleLoops()"), true);
  assert.deepEqual(plain(ui.run("multipleLoops()[0].start_nodes")), ["missing"]);
  assert.deepEqual(plain(ui.run("multipleLoops()[0].end_nodes")), ["deleted-score"]);
  assert.match(ui.run("conditionalValidation()"), /失效/);
  const old = ui.run("JSON.stringify(multipleLoops()[0])");
  ui.add();
  assert.equal(ui.run("multipleLoops().length"), 2);
  assert.equal(ui.run("JSON.stringify(multipleLoops()[0])"), old);
});

test("schema export/import roundtrip retains multiple loops and resets transient picker", async () => {
  const ui = controller(); ui.complete(); ui.complete("alt");
  const before = ui.run("JSON.stringify(state.workflow.conditional_loop)");
  ui.run("saveWorkflow()");
  const saved = JSON.parse(await ui.blob().text());
  assert.equal(JSON.stringify(saved.workflow.conditional_loop), before);
  ui.run(`state.workflow = ${JSON.stringify(saved.workflow)}; restoreLoop({conditionalTargetId: 'save'}); renderConditional();`);
  assert.equal(ui.run("JSON.stringify(state.workflow.conditional_loop)"), before);
  assert.equal(ui.run("conditionalUI.picking"), null);
  assert.equal(ui.run("multipleUI.activeId"), null);
  assert.equal(ui.run("conditionalValidation()"), "");
});

test("condition change invalidates folder plan but preserves history and preview session", () => {
  const ui = controller(), id = ui.complete();
  ui.run("batch.job = {status: 'completed', iteration_history: [{loop_id: 'old', loop_name: '旧历史', iteration: 2}]};");
  for (const [label, value, field] of [["条件 1 比较值", "8", "value"], ["条件 1 运算符", ">=", "operator"]]) {
    ui.run("loopUI.plan = {}; loopUI.planKey = 'old';");
    const control = ui.field(label, id); control.value = value; control.onchange();
    assert.equal(ui.run(`String(multipleLoop('${id}').conditions[0].${field})`), value);
    assert.equal(ui.run("conditionalUI.plan"), null);
    assert.equal(ui.run("loopUI.plan"), null);
    assert.equal(ui.run("state.previewSession"), "keep-session");
    assert.equal(ui.run("batch.job.iteration_history[0].loop_name"), "旧历史");
  }
});

test("legacy multi-conditions remain read-only and reselect explicitly replaces only their own list", async () => {
  const ui = controller(), a = ui.complete(), b = ui.complete("alt");
  ui.run(`multipleLoop('${a}').conditions.push({node: 'multi', output: 'ok', operator: '==', value: true});
    Object.assign(multipleLoop('${a}'), {mode: 'any', max_iterations: 7, timeout_seconds: 12, on_exhausted: 'skip'});`);
  const before = ui.run(`JSON.stringify(multipleLoop('${a}'))`);
  const other = ui.run(`JSON.stringify(multipleLoop('${b}'))`);
  ui.run("renderConditional(); renderBatch(); renderConditional()");
  const card = ui.document.querySelectorAll('.multiple-loop-card').find(c => c.dataset.loopId === a);
  assert.match(textOf(card), /2 个条件（any）.*原列表保持不变/);
  assert.match(textOf(card), /score.value <= 10；multi.ok == true/);
  assert.equal(card.querySelectorAll('input, select').length, 0);
  assert.equal(ui.run(`JSON.stringify(multipleLoop('${a}'))`), before);
  ui.run('saveWorkflow()');
  const downloaded = JSON.parse(await ui.blob().text());
  assert.deepEqual(downloaded.workflow.conditional_loop.loops.find(item => item.id === a), JSON.parse(before));
  ui.button("重选条件", a).click(); ui.header("multi"); ui.port("multi", "value");
  assert.equal(ui.run(`multipleLoop('${a}').conditions.length`), 1);
  assert.equal(ui.run(`multipleLoop('${a}').conditions[0].node`), "multi");
  assert.equal(ui.run(`multipleLoop('${a}').mode`), "any");
  const after = plain(ui.run(`multipleLoop('${a}')`));
  assert.deepEqual({...after, conditions: JSON.parse(before).conditions}, JSON.parse(before));
  assert.equal(ui.run(`JSON.stringify(multipleLoop('${b}'))`), other);
});

test("legacy multiple boundaries remain visible and unchanged until explicit single reselect", () => {
  const ui = controller(), id = ui.complete();
  const graph = ui.run("JSON.stringify(state.workflow.nodes)");
  ui.run(`multipleLoop('${id}').end_nodes = ['gen', 'score']; renderConditional()`);
  assert.deepEqual(plain(ui.run(`multipleLoop('${id}').end_nodes`)), ["gen", "score"]);
  ui.button("终点 · score · 同类测试节点 (Score)", id);
  ui.button("重选终点", id).click(); ui.header("score");
  assert.deepEqual(plain(ui.run(`multipleLoop('${id}').end_nodes`)), ["score"]);
  assert.equal(ui.run("JSON.stringify(state.workflow.nodes)"), graph);
});

test("stale nodes / target do not silently select another same-type instance", () => {
  const ui = controller(), id = ui.complete();
  ui.run("delete state.workflow.nodes.gen; renderConditional();");
  assert.deepEqual(plain(ui.run(`multipleLoop('${id}').start_nodes`)), ["gen"]);
  assert.match(ui.run("conditionalValidation()"), /失效/);
  ui.run("conditionalUI.targetId = 'deleted'; renderConditional();");
  assert.equal(ui.run("conditionalUI.targetId"), "deleted");
  assert.equal(ui.document.querySelector("#conditional-check"), null);
});

test("multiple SaveImages do not expose target controls or block loop previews", () => {
  const ui = controller();
  ui.run("state.workflow.nodes.save2 = structuredClone(state.workflow.nodes.save);");
  const id = ui.complete(); assert.equal(ui.run("conditionalUI.targetId"), null);
  assert.equal(ui.document.querySelectorAll('.multiple-target').length, 0);
  assert.equal(ui.document.querySelector('#conditional-error').textContent, '');
  assert.equal(ui.button('预览此循环 · 不写盘', id).disabled, false);
  ui.run(`beginMultiplePick('${id}', 'target')`);
  assert.equal(ui.run('conditionalUI.picking'), null);
  assert.match(ui.run('conditionalValidation()'), /保存目标/);
});

test("overlap/nesting is rejected but independent and serial regions pass", () => {
  const ui = controller(); ui.complete(); ui.complete();
  assert.match(ui.run("conditionalValidation()"), /重叠/);
  const ids = plain(ui.run("multipleLoops().map(l => l.id)"));
  ui.button("删除循环", ids[1]).click(); ui.complete("alt");
  assert.equal(ui.run("conditionalValidation()"), "");
  ui.run("state.workflow.nodes.alt.inputs.image = link('gen', 'image');");
  assert.equal(ui.run("conditionalValidation()"), "");
});

test("ordinary downstream may consume any accepted internal output without adding end_nodes", () => {
  const ui = controller(), id = ui.complete();
  ui.run("delete state.workflow.nodes.multi; delete state.workflow.nodes.bool;");
  ui.run(`multipleLoop('${id}').end_nodes = ['score'];`);
  assert.equal(ui.run("conditionalValidation()"), "");
  ui.run("state.workflow.nodes.alt.inputs.image = link('gen', 'image');");
  assert.equal(ui.run("conditionalValidation()"), "");
  assert.equal(ui.run(`validateMultiple(null, '${id}')`), "");
  assert.deepEqual(plain(ui.run(`multipleLoop('${id}').end_nodes`)), ['score']);
  ui.run(`multipleLoop('${id}').end_nodes = ['gen'];`);
  ui.run("state.workflow.nodes.alt.inputs.metric = link('score', 'value');");
  assert.equal(ui.run("conditionalValidation()"), "");
});

test("ordinary downstream allowance does not remove workflow-output and side-effect protections", () => {
  const ui = controller(), id = ui.complete();
  ui.run("state.workflow.outputs.leak = link('gen', 'image')");
  assert.match(ui.run('conditionalValidation()'), /工作流输出/);
  ui.run("delete state.workflow.outputs.leak; state.catalog.Generate.side_effects = true");
  assert.match(ui.run(`validateMultiple(null, '${id}')`), /副作用/);
});

test("cross-region future dependency rejects interleaved non-overlapping regions", () => {
  const ui = controller(); ui.complete(); ui.complete("alt");
  ui.run("state.workflow.nodes.score.inputs.extra = link('alt', 'image'); state.workflow.nodes.score2.inputs.extra = link('gen', 'image');");
  assert.match(ui.run("conditionalValidation()"), /交叉依赖/);
});

test("all busy states reject Add/delete/pick/condition edits and cancel pending picker", () => {
  for (const lock of ["loopUI.busy", "loopUI.checking", "conditionalUI.checking", "batch.starting", "batch.requesting", "batch.restoring", "batch.uncertain"]) {
    const ui = controller(), id = ui.complete();
    const threshold = ui.field("条件 1 比较值", id), operator = ui.field("条件 1 运算符", id);
    const del = ui.button("删除循环", id), reselect = ui.button("重选起点", id);
    reselect.click();
    const before = ui.run("JSON.stringify(state.workflow)");
    ui.run(`${lock} = true; state.connecting = link('gen', 'image'); renderConditional();`);
    assert.equal(ui.run("conditionalUI.picking"), null, lock);
    assert.equal(ui.run("state.connecting"), null, lock);
    assert.equal(ui.document.querySelector("#conditional-add").disabled, true, lock);
    ui.run("addMultipleLoop()"); del.onclick(); reselect.onclick();
    threshold.value = "3"; threshold.onchange(); operator.value = ">="; operator.onchange();
    assert.equal(ui.run("JSON.stringify(state.workflow)"), before, lock);
  }
});

test("input clicks during picker do not bind links; entering picker clears old connection", () => {
  const ui = controller(), id = ui.complete();
  ui.run("state.connecting = link('alt', 'image');");
  ui.button("重选起点", id).click();
  assert.equal(ui.run("state.connecting"), null);
  const before = ui.run("JSON.stringify(state.workflow.nodes)");
  ui.port("gen", "image", "input");
  assert.equal(ui.run("JSON.stringify(state.workflow.nodes)"), before);
  assert.equal(ui.run("conditionalUI.picking.loopId"), id);
  ui.document.dispatchEvent({type: "keydown", key: "Escape"});
  assert.equal(ui.run("conditionalUI.picking"), null);
});

test("polling / layout changes preserve focused threshold and commit on change only", () => {
  const ui = controller(), id = ui.complete();
  const field = ui.field("条件 1 比较值", id); field.focus(); field.value = "25";
  assert.equal(field.oninput, undefined);
  ui.run("state.layout.gen = {x: 350, y: 200}; renderBatch(); renderConditional();");
  assert.equal(ui.field("条件 1 比较值", id), field);
  assert.equal(field.value, "25");
  assert.equal(ui.run(`multipleLoop('${id}').conditions[0].value`), 10);
  field.onchange(); assert.equal(ui.run(`multipleLoop('${id}').conditions[0].value`), 25);
});

test("current loop progress and each history entry show loop name and stable ID", () => {
  const ui = controller(); ui.complete();
  ui.run(`batch.job = {id: 'task', status: 'running', total: 1, processed: 0, current_loop_id: 'loop_b', current_loop_name: '第二段', current_loop_index: 2, current_loop_count: 3,
    iteration_history: [{loop_id: 'loop_a', loop_name: '第一段', iteration: 1, accepted: true}, {loop_id: 'loop_b', loop_name: '第二段', iteration: 2, accepted: false}]}; renderBatch();`);
  assert.match(ui.document.querySelector("#batch-current").textContent, /当前循环：2 \/ 3 · 第二段 \(loop_b\)/);
  assert.match(ui.document.querySelector("#conditional-attempts").children[0].textContent, /第一段 \(loop_a\)/);
  assert.match(ui.document.querySelector("#conditional-attempts").children[1].textContent, /第二段 \(loop_b\)/);
});

test("SaveImages submits loops-only schema without a manual plan check", async () => {
  const ui = controller(); ui.complete(); ui.complete("alt");
  ui.run("fetch = async (url, options) => { requests.push({url, body: JSON.parse(options.body)}); return {ok: true, text: async () => JSON.stringify({id: 'save-task', status: 'completed', results: []})}; }");
  const before = plain(ui.run("state.workflow"));
  await ui.run("previewNode('save')");
  assert.deepEqual(plain(ui.run("Object.keys(requests[0].body.workflow.conditional_loop)")), ["loops"]);
  assert.deepEqual(plain(ui.run("requests[0].body.workflow")), before);
  assert.equal(ui.run("requests[0].url"), "/api/conditional");
  assert.equal(ui.run("requests[0].body.node_id"), "save");
  assert.equal(ui.run("requests[0].body.visualize"), true);
  assert.equal(ui.document.querySelector("#conditional-run"), null);
});

test("bounds, duplicate IDs and mixed flat/loops configuration block planning", () => {
  for (const mutation of ["l.max_iterations = 0", "l.timeout_seconds = -1", "l.mode = 'bad'", "l.on_exhausted = 'bad'", "l.start_nodes.push('gen')", "state.workflow.conditional_loop.mode = 'all'", "multipleLoops().push(structuredClone(l))"]) {
    const ui = controller(); ui.complete();
    ui.run(`{ const l = multipleLoops()[0]; ${mutation}; renderConditional(); }`);
    assert.notEqual(ui.run("conditionalValidation()"), "", mutation);
    assert.notEqual(ui.document.querySelector("#conditional-error").textContent, "", mutation);
  }
});

test("imported malformed lists display validation errors without repairing configuration", () => {
  for (const [field, value] of [["start_nodes", 7], ["end_nodes", "gen"], ["conditions", "score"], ["seed_nodes", null]]) {
    const ui = controller(), id = ui.complete();
    // Include real cards so role rendering also encounters the invalid config.
    ui.run("renderNode('gen', state.workflow.nodes.gen, 0)");
    ui.run(`multipleLoop('${id}').${field} = ${JSON.stringify(value)};`);
    const before = ui.run("JSON.stringify(state.workflow.conditional_loop)");
    ui.run("renderConditional()");
    assert.notEqual(ui.run("conditionalValidation()"), "");
    assert.equal(ui.run("JSON.stringify(state.workflow.conditional_loop)"), before);
  }
});

test("imported IDs survive delete / add without recycling and run lock clears connection-only state", () => {
  const ui = controller(), id = ui.complete();
  ui.run(`multipleLoop('${id}').id = 'loop_50'; restoreLoop({conditionalTargetId: 'save'}); renderConditional();`);
  ui.button("删除循环", "loop_50").click();
  assert.equal(ui.add(), "loop_51");
  ui.run("cancelConditionalPick(); state.connecting = link('gen', 'image'); batch.uncertain = true; renderConditional();");
  assert.equal(ui.run("state.connecting"), null);
});

test("restore migrates once with stable ID and preserves explicitly supplied end/name", () => {
  const ui = controller();
  ui.run("state.workflow.conditional_loop = {start_nodes: ['gen'], end_nodes: ['missing-end'], name: '原名称', seed_nodes: [], conditions: []}; restoreLoop({conditionalTargetId: 'deleted-save'}); renderConditional()");
  const before = ui.run("JSON.stringify(state.workflow.conditional_loop)");
  const id = ui.run("multipleLoops()[0].id");
  assert.deepEqual(plain(ui.run("multipleLoops()[0].end_nodes")), ["missing-end"]);
  assert.equal(ui.run("multipleLoops()[0].name"), "原名称");
  assert.equal(ui.run("conditionalUI.targetId"), "deleted-save");
  ui.run("renderConditional(); renderBatch(); restoreLoop({conditionalTargetId: 'deleted-save'}); renderConditional()");
  assert.equal(ui.run("JSON.stringify(state.workflow.conditional_loop)"), before);
  assert.equal(ui.run("multipleLoops()[0].id"), id);
  assert.equal(ui.run("state.previewSession"), "keep-session");
});

test("migration defaults fill absent fields only; explicit null/string/list errors remain", () => {
  const defaults = {start_nodes: [], seed_nodes: [], conditions: [], mode: "all", max_iterations: 10, timeout_seconds: 0, on_exhausted: "fail"};
  const ui = controller();
  ui.run("state.workflow.conditional_loop = {}; restoreLoop(); renderConditional()");
  for (const [field, value] of Object.entries(defaults)) assert.deepEqual(plain(ui.run(`multipleLoops()[0].${field}`)), value);
  assert.deepEqual(plain(ui.run("multipleLoops()[0].end_nodes")), []);
  assert.notEqual(ui.run("conditionalValidation()"), "");
  for (const [field, value] of [
    ["start_nodes", null], ["seed_nodes", "gen"], ["conditions", null], ["end_nodes", null],
    ["mode", null], ["max_iterations", "10"], ["timeout_seconds", null], ["on_exhausted", ""], ["name", null],
  ]) {
    const ui = controller();
    ui.run(`state.workflow.conditional_loop = {start_nodes: ['gen'], conditions: [{node: 'score', output: 'value', operator: '<', value: 3}], ${field}: ${JSON.stringify(value)}}; renderConditional()`);
    assert.deepEqual(plain(ui.run(`multipleLoops()[0].${field}`)), value, field);
    assert.notEqual(ui.run("conditionalValidation()"), "", field);
    assert.notEqual(ui.document.querySelector("#conditional-error").textContent, "", field);
    ui.add(); assert.equal(ui.run("multipleLoops().length"), 2, field);
  }
});

test("legacy derivation uses the old body, excludes unrelated saved sources and keeps missing conditions", () => {
  const ui = controller();
  ui.run("state.workflow.conditional_loop = {start_nodes: ['gen'], conditions: [{node: 'score', output: 'value', operator: '<', value: 3}, {node: 'gone', output: 'value', operator: '<', value: 2}]}; renderConditional()");
  assert.deepEqual(plain(ui.run("multipleLoops()[0].end_nodes")), ["gen", "score", "gone"]);
  assert.equal(ui.run("multipleLoops()[0].end_nodes.includes('alt')"), false);
  assert.match(ui.run("conditionalValidation()"), /失效/);
});

test("legacy import never guesses among multiple saves or replaces stale target", () => {
  for (const target of [null, "gone"]) {
    const ui = controller();
    ui.run(`state.workflow.nodes.save2 = structuredClone(state.workflow.nodes.save);
      state.workflow.conditional_loop = {start_nodes: ['gen'], seed_nodes: [], conditions: [{node: 'score', output: 'value', operator: '<', value: 3}]};
      restoreLoop({conditionalTargetId: ${JSON.stringify(target)}}); renderConditional()`);
    assert.equal(ui.run("conditionalUI.targetId"), target);
    assert.deepEqual(plain(ui.run("multipleLoops()[0].end_nodes")), ["score"]);
    assert.match(ui.run("conditionalValidation()"), /保存目标/);
    assert.equal(ui.document.querySelector("#conditional-check"), null);
  }
});

test("folder legacy import derives using the explicitly selected folder save", () => {
  const ui = controller();
  ui.run("state.workflow.nodes.folder = {class_type: 'FolderImagesNode', inputs: {}}; state.workflow.nodes.gen.inputs.image = link('folder', 'image'); state.workflow.nodes.save2 = structuredClone(state.workflow.nodes.save); state.workflow.conditional_loop = {start_nodes: ['gen'], conditions: []}; restoreLoop({sourceId: 'folder', targetId: 'save2', enabled: true}); renderConditional()");
  assert.equal(ui.run("conditionalUI.targetId"), "save2");
  assert.deepEqual(plain(ui.run("multipleLoops()[0].end_nodes")), ["gen"]);
});

test("invalid raw conditional values are retained across polls and need explicit deletion", async () => {
  for (const raw of [[], ["bad"], "invalid", 7, false, {loops: null}, {loops: [null]}, {loops: []}, {unknown: "keep-me"}]) {
    const ui = controller();
    ui.run(`state.workflow.conditional_loop = ${JSON.stringify(raw)}; restoreLoop(); renderConditional()`);
    const before = ui.run("JSON.stringify(state.workflow.conditional_loop)");
    ui.run("renderConditional(); renderBatch(); renderConditional()");
    assert.equal(ui.run("JSON.stringify(state.workflow.conditional_loop)"), before);
    assert.equal(ui.run("conditionalEnabled()"), true);
    assert.notEqual(ui.run("conditionalValidation()"), "");
    await ui.run("previewNode('save')"); assert.equal(ui.run("requests.length"), 0);
    ui.run("saveWorkflow()");
    const saved = JSON.parse(await ui.blob().text());
    assert.equal(JSON.stringify(saved.workflow.conditional_loop), before);
    if (raw && !Array.isArray(raw) && typeof raw === "object" && !Object.hasOwn(raw, "loops")) {
      assert.equal(ui.run("multipleLoops()[0].unknown"), "keep-me");
      ui.button("删除循环").click();
    } else if (raw?.loops?.length === 0) {
      ui.add(); ui.button("删除循环").click();
    } else {
      ui.button("删除无效配置").click();
    }
    assert.equal(ui.run("conditionalEnabled()"), false);
  }
});

test("null conditional metadata stays disabled, unlike malformed raw values", () => {
  const ui = controller(); ui.run("state.workflow.conditional_loop = null; restoreLoop(); renderConditional()");
  assert.equal(ui.run("conditionalEnabled()"), false);
  assert.equal(ui.run("state.workflow.conditional_loop"), null);
  assert.equal(ui.document.querySelector("#conditional-check"), null);
  assert.equal(ui.document.querySelector("#conditional-error").textContent, "");
  assert.equal(ui.document.querySelector("#conditional-plan"), null);
});

test("old workflow download and actual import produce only new loops without mutating source", async () => {
  const ui = controller(); legacy(ui);
  const original = ui.run("JSON.stringify(originalLegacy)");
  ui.run("saveWorkflow()"); const downloaded = JSON.parse(await ui.blob().text());
  assert.deepEqual(Object.keys(downloaded.workflow.conditional_loop), ["loops"]);
  const oldSaved = {...downloaded, workflow: {...downloaded.workflow, conditional_loop: JSON.parse(original)}};
  ui.run(`globalThis.sourceImport = ${JSON.stringify(oldSaved)};
    fetch = async (url, options) => ({ok: true, text: async () => JSON.stringify({workflow: JSON.parse(options.body).workflow, config: {}})});
    globalThis.importEvent = {target: {files: [{name: 'legacy.json', text: async () => JSON.stringify(sourceImport)}]}}`);
  await ui.run("loadWorkflow(importEvent)");
  assert.equal(ui.run("isMultipleLoops()"), true);
  assert.equal(ui.run("multipleLoops().length"), 1);
  assert.equal(ui.run("JSON.stringify(sourceImport. workflow.conditional_loop)"), original);
  ui.run("saveWorkflow()"); const again = JSON.parse(await ui.blob().text());
  assert.deepEqual(Object.keys(again.workflow.conditional_loop), ["loops"]);
});

test("Add migrates current flat data even before rendering and enforces 32-loop ceiling", () => {
  const ui = controller();
  ui.run("state.workflow.conditional_loop = {start_nodes: ['missing']}; addMultipleLoop()");
  assert.equal(ui.run("multipleLoops().length"), 2);
  assert.deepEqual(plain(ui.run("multipleLoops()[0].start_nodes")), ["missing"]);
  ui.run("while (multipleLoops().length < 32) addMultipleLoop()");
  const before = ui.run("JSON.stringify(state.workflow.conditional_loop)");
  ui.run("addMultipleLoop()");
  assert.equal(ui.run("JSON.stringify(state.workflow.conditional_loop)"), before);
  assert.match(ui.document.querySelector("#status").textContent, /32/);
  assert.equal(new Set(plain(ui.run("multipleLoops().map(l => l.id)"))).size, 32);
});

test("cards expose only start/end/condition with preview/delete and no advanced or save controls", () => {
  const ui = controller(), id = ui.complete();
  const card = ui.document.querySelectorAll(".multiple-loop-card").find(c => c.dataset.loopId === id);
  assert.equal(walk(card).some(c => c.tagName === "select" && (c.multiple || Object.hasOwn(c.attributes, "multiple"))), false);
  assert.deepEqual(card.querySelectorAll("select").map(c => c.attributes["aria-label"]), ["条件 1 运算符"]);
  assert.deepEqual(card.querySelectorAll("input").map(c => c.attributes["aria-label"]), ["条件 1 比较值"]);
  assert.equal(card.querySelectorAll('details').length, 0);
  assert.deepEqual(card.querySelectorAll('button').map(c => c.textContent), [
    '预览此循环 · 不写盘', '删除循环', '重选起点', '起点 · gen · 同类测试节点 (Generate)',
    '重选终点', '终点 · gen · 同类测试节点 (Generate)', '重选条件',
  ]);
  assert.equal(ui.document.querySelectorAll('.multiple-target').length, 0);
  assert.doesNotMatch(textOf(card), /高级|循环名称|条件组合|添加条件|删除条件|最大尝试|超时|耗尽|增删|保存目标/);
  assert.deepEqual(plain(ui.run(`multipleLoop('${id}').seed_nodes`)), ['gen']);
  assert.equal(ui.run(`multipleLoop('${id}').max_iterations`), 10);
});

// Mock only the HTTP boundary; validation, submission, polling and rendering stay real.
function replies(ui, values) {
  ui.run(`
    requests.length = 0;
    globalThis.responses = ${JSON.stringify(values)};
    fetch = async (url, options = {}) => {
      requests.push({url, method: options.method || 'GET', body: options.body ? JSON.parse(options.body) : null});
      if (!responses.length) throw new Error('Unexpected request: ' + url);
      const value = responses.shift();
      const response = data => ({ok: true, text: async () => JSON.stringify(data)});
      if (value.pending) return new Promise((resolve, reject) => {
        globalThis.resolveRequest = data => resolve(response(data));
        globalThis.rejectRequest = reject;
      });
      if (value.networkError) throw new Error(value.networkError);
      return response(value);
    };
  `);
}
const settled = () => new Promise(resolve => setImmediate(resolve));
const previewJob = (id, status = "running", result = null) => ({
  id: "preview/task", status, preview_loop_id: id, required_loop_ids: [id],
  total: 1, processed: result ? 1 : 0, succeeded: result?.accepted ? 1 : 0,
  failed: status === "failed" ? 1 : 0, results: result ? [result] : [],
});
const previewCard = (ui, id) => ui.document.querySelectorAll(".multiple-preview").find(c => c.dataset.previewLoopId === id);
const textOf = element => walk(element).map(c => c.textContent || "").join("\n");
const canvasCard = (ui, id) => {
  // Full render() clears the real canvas. This partial-render fixture must also
  // keep one card per ID, or production querySelector() updates an obsolete card.
  ui.document.querySelectorAll(`.node[data-node-id="${id}"]`).forEach(card => card.remove());
  ui.run(`renderNode(${JSON.stringify(id)}, state.workflow.nodes[${JSON.stringify(id)}], 0)`);
  return ui.document.querySelector('#nodes').children.at(-1);
};

const liveSnapshot = (id, iteration, accepted = false) => ({
  loop_id: id, iteration, accepted, outputs: {
    gen: {image: {kind: 'image', image: `data:image/png;base64,${id}-round-${iteration}`, size: [32, 32], mode: 'RGB'}},
    score: {value: {kind: 'scalar', value: iteration}},
  },
});

test("ordinary batch previews show the input number and completion without an acceptance verdict", () => {
  const ui = controller();
  const score = canvasCard(ui, 'score');
  ui.run(`delete state.workflow.conditional_loop;
    state.activeJobId = 'batch-job';
    state.activeJobRevision = state.previewRevision;
    batch.job = {id: 'batch-job', status: 'running', visualization_revision: 1,
      loop_previews: {batch: {kind: 'batch', loop_id: 'batch', iteration: 2, accepted: false,
        outputs: {score: {value: {kind: 'scalar', value: 0.5}}}}}};
    applyJobVisualization()`);
  assert.equal(score.querySelector('.preview-state').textContent, '第 2 张输入 · 节点输出');
  ui.run(`batch.job.status = 'completed';
    batch.job.results = [{preview: batch.job.loop_previews.batch.outputs}];
    applyJobVisualization()`);
  assert.equal(score.querySelector('.preview-state').textContent, '本次运行 · 节点输出');
});

test("each running poll renders its round immediately, replaces rather than appends, then renders final preview", async () => {
  const ui = controller(), id = ui.complete(), other = ui.complete('alt');
  const gen = canvasCard(ui, 'gen'), score = canvasCard(ui, 'score');
  ui.run("state.selected = 'score'");
  const first = liveSnapshot(id, 1), second = liveSnapshot(id, 2);
  const final = {preview_loop_id: id, accepted: true, iterations: 3, stop_reason: 'accepted', preview: liveSnapshot(id, 3, true).outputs};
  replies(ui, [
    {...previewJob(id), visualization_revision: 1, loop_previews: {[id]: first}},
    {...previewJob(id), visualization_revision: 2, loop_previews: {[id]: second}},
    {...previewJob(id, 'completed', final), visualization_revision: 3, loop_previews: {[id]: second}},
  ]);
  await ui.run(`batchRequest('preview-loop', '${id}')`);
  assert.equal(previewCard(ui, id).hidden, false);
  assert.equal(previewCard(ui, id).querySelector('img').src, first.outputs.gen.image.image);
  assert.equal(gen.querySelector('.node-preview-shell img').src, first.outputs.gen.image.image);
  assert.deepEqual(plain(ui.run('state.nodeResults.score')), first.outputs.score);
  assert.match(textOf(score.querySelector('.node-preview-shell')), /value · 文本\n1/);
  assert.match(gen.querySelector('.preview-state').textContent, /第 1 次 · 未通过/);
  assert.doesNotMatch(gen.querySelector('.preview-state').textContent, /成功|达标结果/);
  assert.match(textOf(previewCard(ui, id)), /运行中.*第 1 轮/);
  assert.match(textOf(previewCard(ui, id)), /实时最新快照/);
  assert.equal(previewCard(ui, other).children.length, 0);
  const firstShell = gen.querySelector('.node-preview-shell');
  await ui.run("batchRequest('poll')");
  assert.equal(previewCard(ui, id).querySelectorAll('img').length, 1);
  assert.equal(previewCard(ui, id).querySelector('img').src, second.outputs.gen.image.image);
  assert.equal(gen.querySelectorAll('.node-preview-shell').length, 1);
  assert.equal(firstShell.parentElement, null);
  assert.equal(gen.querySelector('.node-preview-shell img').src, second.outputs.gen.image.image);
  assert.deepEqual(plain(ui.run('state.nodeResults.gen')), second.outputs.gen);
  assert.deepEqual(plain(ui.run('state.nodeResults.score')), second.outputs.score);
  assert.match(textOf(score.querySelector('.node-preview-shell')), /value · 文本\n2/);
  assert.match(textOf(ui.document.querySelector('#node-preview')), /value · 文本\n2/);
  assert.equal(ui.document.querySelector('#run-visualization').querySelector('img').src, second.outputs.gen.image.image);
  assert.match(textOf(ui.document.querySelector('#run-visualization')), /value · 文本\n2/);
  assert.match(gen.querySelector('.preview-state').textContent, /第 2 次 · 未通过/);
  assert.match(textOf(previewCard(ui, id)), /运行中.*第 2 轮/);
  assert.equal(ui.run(`multipleUI.previews.get('${id}').result.iterations`), 2);
  await ui.run("batchRequest('poll')");
  assert.equal(previewCard(ui, id).querySelectorAll('img').length, 1);
  assert.equal(previewCard(ui, id).querySelector('img').src, final.preview.gen.image.image);
  assert.equal(gen.querySelectorAll('.node-preview-shell').length, 1);
  assert.equal(gen.querySelector('.node-preview-shell img').src, final.preview.gen.image.image);
  assert.deepEqual(plain(ui.run('state.nodeResults.score')), final.preview.score);
  assert.match(gen.querySelector('.preview-state').textContent, /达标结果/);
  assert.match(textOf(previewCard(ui, id)), /已通过.*第 3 轮/);
  assert.equal(ui.run('multipleUI.previews.size'), 1);
});

test("visualization off hides without forgetting; on restores latest hidden-round snapshots for every loop", () => {
  const ui = controller(), first = ui.complete(), second = ui.complete('alt');
  const a = liveSnapshot(first, 1), b = liveSnapshot(second, 2, true);
  ui.run(`batch.job = {id: 'save-task', status: 'running', visualization_revision: 1,
    loop_previews: ${JSON.stringify({[first]: a, [second]: b})}}; renderConditional()`);
  for (const [id, snapshot] of [[first, a], [second, b]]) {
    assert.equal(previewCard(ui, id).hidden, false);
    assert.equal(previewCard(ui, id).querySelector('img').src, snapshot.outputs.gen.image.image);
    assert.doesNotMatch(textOf(previewCard(ui, id)), /不写盘/);
  }
  ui.run('state.visualize = false; renderConditional()');
  assert.equal(ui.run('multipleUI.previews.size'), 2);
  assert.equal(previewCard(ui, first).hidden, true);
  assert.equal(previewCard(ui, second).hidden, true);
  const update = liveSnapshot(first, 3, true);
  ui.run(`batch.job.loop_previews['${first}'] = ${JSON.stringify(update)}; batch.job.visualization_revision++; renderConditional()`);
  assert.equal(previewCard(ui, first).hidden, true);
  assert.equal(ui.run(`multipleUI.previews.get('${first}').result.iterations`), 3);
  ui.run('state.visualize = true; renderConditional()');
  assert.equal(previewCard(ui, first).hidden, false);
  assert.equal(previewCard(ui, second).hidden, false);
  assert.equal(previewCard(ui, first).querySelector('img').src, update.outputs.gen.image.image);
  assert.equal(previewCard(ui, second).querySelector('img').src, b.outputs.gen.image.image);
  assert.equal(previewCard(ui, first).querySelectorAll('img').length, 1);
  ui.run('batch.job = null; renderConditional()');
  assert.equal(previewCard(ui, first).querySelector('img').src, update.outputs.gen.image.image);
  assert.equal(previewCard(ui, second).querySelector('img').src, b.outputs.gen.image.image);
  assert.doesNotMatch(textOf(previewCard(ui, first)), /不写盘/);
});

test("a new preview request with identical status and no live snapshot still displays its new final outputs", async () => {
  const ui = controller(), id = ui.complete();
  const result = iteration => ({preview_loop_id: id, accepted: true, iterations: iteration, preview: liveSnapshot(id, iteration).outputs});
  replies(ui, [previewJob(id, 'completed', result(1)), {...previewJob(id, 'completed', result(2)), id: 'preview/task-2'}]);
  await ui.run(`batchRequest('preview-loop', '${id}')`);
  assert.equal(previewCard(ui, id).querySelector('img').src, result(1).preview.gen.image.image);
  await ui.run(`batchRequest('preview-loop', '${id}')`);
  assert.equal(previewCard(ui, id).querySelector('img').src, result(2).preview.gen.image.image);
  assert.equal(ui.run('multipleUI.previews.size'), 1);
});

for (const hasSave of [false, true]) {
  test(`preview button posts only loop_id to preview endpoint ${hasSave ? "with a SaveImages node" : "without any SaveImages"}`, async () => {
    const ui = controller(), id = ui.complete(), other = ui.complete("alt");
    if (!hasSave) {
      ui.run("delete state.workflow.nodes.save; conditionalUI.targetId = null; renderConditional()");
      assert.match(ui.run("conditionalValidation()"), /保存目标/);
      assert.equal(ui.run("conditionalUI.plan"), null);
      assert.equal(ui.document.querySelector('#conditional-error').textContent, '');
    }
    ui.run(`state.config = {demo_limit: 0}; document.querySelector('#config-demo_limit').value = '7';
      document.querySelector('#target-path').value = 'input.png';
      document.querySelector('#mask-path').value = 'mask.png';
      document.querySelector('#object-name').value = '对象';`);
    const before = plain(ui.run("state.workflow"));
    const otherBefore = plain(ui.run(`multipleLoop('${other}')`));
    replies(ui, [previewJob(id)]);
    const button = ui.button("预览此循环 · 不写盘", id);
    assert.equal(button.disabled, false);
    button.click(); await settled();
    const requests = plain(ui.run("requests"));
    assert.equal(requests.length, 1);
    assert.equal(requests[0].url, "/api/conditional/preview");
    assert.equal(requests[0].method, "POST");
    const payload = requests[0].body;
    assert.equal(payload.loop_id, id);
    assert.equal(payload.visualize, true);
    assert.equal(Object.hasOwn(payload, "node_id"), false);
    assert.equal(Object.hasOwn(payload, "continue_on_error"), false);
    assert.deepEqual(payload.workflow, before);
    assert.deepEqual(payload.record, {target_path: "input.png", mask_path: "mask.png", object_name: "对象"});
    assert.deepEqual(payload.config, {demo_limit: 7});
    assert.deepEqual(plain(ui.run("state.workflow")), before);
    assert.deepEqual(plain(ui.run(`multipleLoop('${other}')`)), otherBefore);
    assert.equal(ui.run("batch.uncertain"), false);
    assert.equal(ui.run("batch.job.preview_loop_id"), id);
    assert.equal(ui.run("state.previewSession"), null);
  });
}

for (const downstream of [false, true]) {
  test(`selected preview ignores an incomplete ${downstream ? "downstream" : "independent"} loop without removing its configuration`, async () => {
    const ui = controller(), id = ui.complete(), other = ui.complete("alt");
    ui.run(`multipleLoop('${other}').conditions = []; multipleLoop('${other}').max_iterations = 0;
      ${downstream ? "state.workflow.nodes.alt.inputs.image = link('gen', 'image');" : ""}
      renderConditional();`);
    const before = ui.run("JSON.stringify(state.workflow)");
    assert.notEqual(ui.run("conditionalValidation()"), "");
    assert.equal(ui.run(`validateMultiple(null, '${id}')`), "");
    assert.equal(ui.button("预览此循环 · 不写盘", other).disabled, true);
    const button = ui.button("预览此循环 · 不写盘", id);
    assert.equal(button.disabled, false);
    replies(ui, [previewJob(id)]); button.click(); await settled();
    assert.deepEqual(plain(ui.run("requests.map(r => [r.url, r.body.loop_id])")), [["/api/conditional/preview", id]]);
    assert.equal(ui.run("requests[0].body.workflow.conditional_loop.loops.length"), 2);
    assert.equal(ui.run("JSON.stringify(state.workflow)"), before);
    assert.equal(ui.run("batch.uncertain"), false);
  });
}

test("invalid necessary upstream blocks selected preview through ordinary bridge nodes, then permits it after repair", async () => {
  const ui = controller(), upstream = ui.complete(), id = ui.complete("alt");
  ui.run(`state.workflow.nodes.bridge = {class_type: 'Source', inputs: {image: link('gen', 'image')}};
    state.workflow.nodes.alt.inputs.image = link('bridge', 'image');
    multipleLoop('${upstream}').max_iterations = 0; renderConditional();`);
  const before = ui.run("JSON.stringify(state.workflow)");
  assert.match(ui.run(`validateMultiple(null, '${id}')`), /最大尝试次数/);
  const button = ui.button("预览此循环 · 不写盘", id);
  assert.equal(button.disabled, true);
  replies(ui, [previewJob(id)]);
  button.click(); button.onclick();
  await ui.run(`batchRequest('preview-loop', '${id}')`);
  assert.equal(ui.run("requests.length"), 0);
  assert.equal(ui.run("JSON.stringify(state.workflow)"), before);
  ui.run(`multipleLoop('${upstream}').max_iterations = 10; renderConditional()`);
  assert.equal(ui.run(`validateMultiple(null, '${id}')`), "");
  ui.button("预览此循环 · 不写盘", id).click(); await settled();
  assert.deepEqual(plain(ui.run("requests.map(r => [r.url, r.body.loop_id])")), [["/api/conditional/preview", id]]);
  assert.equal(ui.run("batch.uncertain"), false);
});

test("pending preview synchronously acquires the shared lock before a second button or save submission", async () => {
  const ui = controller(), id = ui.complete(), other = ui.complete("alt");
  const first = ui.button("预览此循环 · 不写盘", id), second = ui.button("预览此循环 · 不写盘", other);
  const before = ui.run("JSON.stringify(state.workflow)");
  replies(ui, [{pending: true}]);
  first.click();
  assert.equal(ui.run("batch.requesting"), true);
  assert.equal(ui.run("batch.starting"), true);
  assert.equal(first.disabled, true);
  assert.equal(second.disabled, true);
  assert.equal(ui.run("nodeDeletionDisabled()"), true);
  first.onclick(); second.onclick();
  await ui.run(`Promise.all([batchRequest('preview-loop', '${other}'), batchRequest('start-condition', 'save'), batchRequest('start', 'save'), previewNode('save')])`);
  assert.equal(ui.run("requests.length"), 1);
  assert.equal(ui.run("JSON.stringify(state.workflow)"), before);
  ui.run(`resolveRequest(${JSON.stringify(previewJob(id))})`); await settled();
  assert.equal(ui.run("batch.requesting"), false);
  assert.equal(ui.run("batch.starting"), false);
  assert.equal(ui.run("batchActive()"), true);
  assert.equal(ui.run("batch.uncertain"), false);
  assert.equal(ui.button("预览此循环 · 不写盘", other).disabled, true);
});

for (const status of ["running", "cancelling"]) {
  test(`shared ${status} task blocks loop preview and conditional save, regardless of task kind`, async () => {
    const ui = controller(), id = ui.complete();
    const staleButton = ui.button("预览此循环 · 不写盘", id);
    replies(ui, []);
    // An ordinary batch job (not a loop-preview job) owns exactly the same lock.
    ui.run(`batch.job = {id: 'ordinary-batch', status: '${status}'}; renderBatch()`);
    assert.equal(ui.run("batchStartDisabled()"), true);
    assert.equal(staleButton.disabled, true);
    assert.equal(ui.run("nodeDeletionDisabled()"), true);
    staleButton.onclick();
    await ui.run(`Promise.all([batchRequest('preview-loop', '${id}'), batchRequest('start-condition', 'save'), batchRequest('start', 'save'), previewNode('save')])`);
    assert.equal(ui.run("requests.length"), 0);
    assert.equal(ui.run("batch.job.id"), "ordinary-batch");
  });
}

test("lost preview response keeps uncertain lock until refresh confirms no task, then allows a new preview", async () => {
  const ui = controller(), id = ui.complete();
  const staleButton = ui.button("预览此循环 · 不写盘", id);
  replies(ui, [{networkError: "response lost"}, {job: null}, previewJob(id)]);
  staleButton.click(); await settled();
  assert.equal(ui.run("batch.uncertain"), true);
  assert.equal(ui.run("batch.requesting || batch.starting"), false);
  assert.match(ui.document.querySelector("#batch-error").textContent, /任务可能已启动.*锁定重复启动/);
  assert.equal(ui.button("预览此循环 · 不写盘", id).disabled, true);
  staleButton.onclick();
  await ui.run(`batchRequest('preview-loop', '${id}')`);
  assert.equal(ui.run("requests.length"), 1);
  ui.document.querySelector("#batch-refresh").click(); await settled();
  assert.equal(ui.run("batch.uncertain"), false);
  assert.equal(ui.button("预览此循环 · 不写盘", id).disabled, false);
  ui.button("预览此循环 · 不写盘", id).click(); await settled();
  assert.deepEqual(plain(ui.run("requests.map(r => r.url)")), ["/api/conditional/preview", "/api/batch/latest", "/api/conditional/preview"]);
  assert.equal(ui.run("batch.uncertain"), false);
});

for (const status of ["completed", "failed"]) {
  test(`${status} preview renders its own loop card and actual canvas image and score, never as written`, async () => {
    const ui = controller(), id = ui.complete(), other = ui.complete("alt");
    const gen = canvasCard(ui, 'gen'), score = canvasCard(ui, 'score');
    const accepted = status === "completed";
    const result = {
      preview_loop_id: id, accepted, iterations: accepted ? 3 : 1,
      stop_reason: accepted ? "accepted" : "max_iterations", paths: {},
      ...(accepted ? {} : {error: "maximum attempts reached"}),
      preview: {
        gen: {image: {kind: "image", image: "data:image/png;base64,last-round", size: [32, 32], mode: "RGB"}},
        score: {value: {kind: "scalar", value: accepted ? 255 : 0}},
      },
    };
    // Keep an unrelated node preview to verify that visualization updates only returned nodes.
    ui.run("state.nodeResults.alt = {image: {kind: 'image', image: 'unrelated'}}");
    replies(ui, [previewJob(id), previewJob(id, status, result)]);
    ui.button("预览此循环 · 不写盘", id).click(); await settled();
    assert.equal(previewCard(ui, id).querySelectorAll("img").length, 0);
    assert.match(textOf(previewCard(ui, id)), /运行中/);
    await ui.run("batchRequest('poll')");
    const card = previewCard(ui, id), text = textOf(card);
    assert.equal(card.querySelectorAll("img").length, 1);
    assert.equal(card.querySelector("img").src, result.preview.gen.image.image);
    assert.match(text, new RegExp(`value · 文本\\n${accepted ? 255 : 0}`));
    assert.match(text, accepted ? /已通过/ : /未通过 \/ 已停止/);
    assert.match(text, /本次任务快照的末轮结果/);
    assert.match(text, /不写盘/);
    assert.match(text, new RegExp(`停止原因：${result.stop_reason}`));
    assert.equal(previewCard(ui, other).children.length, 0);
    const summary = textOf(ui.document.querySelector("#batch-results"));
    assert.match(summary, new RegExp(`循环预览：${id} · 未写盘`));
    assert.doesNotMatch(summary + text, /已写入：/);
    if (!accepted) assert.match(summary, /失败：maximum attempts reached/);
    assert.deepEqual(plain(ui.run("state.nodeResults.gen")), result.preview.gen);
    assert.deepEqual(plain(ui.run("state.nodeResults.score")), result.preview.score);
    assert.equal(ui.run("state.nodeResults.alt.image.image"), 'unrelated');
    assert.match(ui.run("state.nodePreviewStatus.gen"), accepted ? /达标结果/ : /未达标末轮/);
    assert.equal(gen.querySelector('.node-preview-shell img').src, result.preview.gen.image.image);
    assert.match(textOf(score.querySelector('.node-preview-shell')), new RegExp(`value · 文本\\n${accepted ? 255 : 0}`));
    assert.match(gen.querySelector('.preview-state').textContent, accepted ? /达标结果/ : /未达标末轮/);
    if (!accepted) assert.doesNotMatch(gen.querySelector('.preview-state').textContent, /成功|达标结果/);
    assert.equal(ui.run("batch.uncertain"), false);
    assert.deepEqual(plain(ui.run("requests.map(r => r.url)")), ["/api/conditional/preview", "/api/batch/preview%2Ftask"]);
  });
}

test("each loop keeps its own last preview when another runs; deletion and restore clear retained cards", async () => {
  const ui = controller(), first = ui.complete(), second = ui.complete("alt");
  const result = id => ({preview_loop_id: id, accepted: true, iterations: 1, stop_reason: "accepted", preview: {
    gen: {image: {kind: "image", image: `data:image/png;base64,${id}`, mode: "RGB", size: [32, 32]}},
  }});
  replies(ui, [previewJob(first, "completed", result(first)), previewJob(second, "completed", result(second))]);
  await ui.run(`batchRequest('preview-loop', '${first}')`);
  const firstUrl = previewCard(ui, first).querySelector("img").src;
  await ui.run(`batchRequest('preview-loop', '${second}')`);
  assert.equal(previewCard(ui, first).querySelector("img").src, firstUrl);
  assert.equal(previewCard(ui, second).querySelector("img").src, result(second).preview.gen.image.image);
  assert.equal(ui.run("multipleUI.previews.size"), 2);
  ui.run(`deleteMultipleLoop('${first}')`);
  assert.equal(ui.run(`multipleUI.previews.has('${first}')`), false);
  ui.run("restoreLoop()");
  assert.equal(ui.run("multipleUI.previews.size"), 0);
});

test("real preview polling preserves existing focused input identity, uncommitted text and selection", async () => {
  const ui = controller(), id = ui.complete();
  const input = ui.field("条件 1 比较值", id);
  const originalValue = ui.run(`multipleLoop('${id}').conditions[0].value`);
  const result = {preview_loop_id: id, accepted: false, iterations: 1, preview: {}, stop_reason: "max_iterations"};
  replies(ui, [previewJob(id), previewJob(id), previewJob(id, "failed", result)]);
  input.focus(); input.value = "123456"; input.selectionStart = 2; input.selectionEnd = 5;
  ui.button("预览此循环 · 不写盘", id).click(); await settled();
  for (let poll = 0; poll < 2; poll++) {
    await ui.run("batchRequest('poll')");
    assert.equal(ui.field("条件 1 比较值", id), input);
    assert.equal(ui.document.activeElement, input);
    assert.equal(input.value, "123456");
    assert.deepEqual([input.selectionStart, input.selectionEnd], [2, 5]);
    assert.ok(walk(ui.document.querySelector("#multiple-loops")).includes(input));
    assert.equal(ui.run(`multipleLoop('${id}').conditions[0].value`), originalValue);
  }
  assert.equal(input.disabled, false);
  assert.equal(ui.run("batch.uncertain"), false);
  assert.deepEqual(plain(ui.run("requests.map(r => r.url)")), ["/api/conditional/preview", "/api/batch/preview%2Ftask", "/api/batch/preview%2Ftask"]);
});

test("demo button appends two runnable loops and four CPU nodes, retaining original graph, loops and seed 0/1", () => {
  const ui = controller(), original = ui.complete();
  ui.run("state.layout.gen = {x: 15, y: 25}; state.workflow.outputs.original = link('fixed', 'image')");
  const nodes = plain(ui.run("state.workflow.nodes")), outputs = plain(ui.run("state.workflow.outputs"));
  const layout = plain(ui.run("state.layout")), config = plain(ui.run(`multipleLoop('${original}')`));
  const preview = ui.run("JSON.stringify(state.nodeResults)");
  const button = ui.document.querySelector("#conditional-demo");
  assert.equal(button.disabled, false); button.click(); ui.run("renderConditional()");
  assert.equal(ui.run("multipleLoops().length"), 3);
  assert.equal(ui.run("Object.keys(state.workflow.nodes).length"), Object.keys(nodes).length + 4);
  assert.deepEqual(plain(ui.run(`multipleLoop('${original}')`)), config);
  for (const [id, spec] of Object.entries(nodes)) assert.deepEqual(plain(ui.run(`state.workflow.nodes['${id}']`)), spec);
  for (const [id, pos] of Object.entries(layout)) assert.deepEqual(plain(ui.run(`state.layout['${id}']`)), pos);
  assert.deepEqual(plain(ui.run("state.workflow.outputs")), outputs);
  assert.equal(ui.run("JSON.stringify(state.nodeResults)"), preview);
  assert.equal(ui.run("state.previewSession"), "keep-session");
  const demos = plain(ui.run("multipleLoops().slice(1)"));
  demos.forEach((loop, index) => {
    const image = loop.start_nodes[0], score = loop.end_nodes[0];
    assert.deepEqual(plain(ui.run(`state.workflow.nodes['${image}']`)), {class_type: "LoopDemoImageNode", inputs: {seed: index}});
    assert.deepEqual(plain(ui.run(`state.workflow.nodes['${score}']`)), {class_type: "LoopDemoScoreNode", inputs: {image: {node: image, output: "image"}}});
    assert.deepEqual(loop.seed_nodes, [image]);
    assert.deepEqual(loop.conditions, [{node: score, output: "score", operator: ">=", value: 255}]);
    assert.equal(loop.max_iterations, 3);
    assert.equal(ui.run(`validateMultiple(null, '${loop.id}')`), "");
    assert.equal(ui.button("预览此循环 · 不写盘", loop.id).disabled, false);
  });
  assert.equal(new Set(demos.flatMap(l => [...l.start_nodes, ...l.end_nodes])).size, 4);
  assert.equal(ui.run("conditionalUI.picking"), null);
  assert.match(ui.document.querySelector("#status").textContent, /不替换原流程/);
});

test("demo previews need no SaveImages and retain seed zero in the actual preview payload", async () => {
  const ui = controller();
  ui.run("delete state.workflow.nodes.save; conditionalUI.targetId = null");
  ui.document.querySelector("#conditional-demo").click(); ui.run("renderConditional()");
  const id = ui.run("multipleLoops()[0].id"), image = ui.run("multipleLoops()[0].start_nodes[0]");
  replies(ui, [previewJob(id)]);
  ui.button("预览此循环 · 不写盘", id).click(); await settled();
  assert.equal(ui.run("requests.length"), 1);
  assert.equal(ui.run("requests[0].url"), "/api/conditional/preview");
  assert.equal(ui.run(`requests[0].body.workflow.nodes['${image}'].inputs.seed`), 0);
  assert.equal(ui.run("Object.hasOwn(requests[0].body, 'node_id')"), false);
  assert.equal(ui.run("batch.uncertain"), false);
});

test("demo button and direct handler respect every shared busy lock, including active jobs", () => {
  for (const lock of ["loopUI.busy = true", "loopUI.checking = true", "conditionalUI.checking = true", "batch.starting = true", "batch.requesting = true", "batch.restoring = true", "batch.uncertain = true", "batch.job = {status: 'running'}", "batch.job = {status: 'cancelling'}"]) {
    const ui = controller(); ui.complete();
    const before = ui.run("JSON.stringify([state.workflow, state.layout])");
    ui.run(`${lock}; renderConditional()`);
    const button = ui.document.querySelector("#conditional-demo");
    assert.equal(button.disabled, true, lock);
    button.click(); button.onclick(); ui.run("addLoopDemo()");
    assert.equal(ui.run("JSON.stringify([state.workflow, state.layout])"), before, lock);
  }
});

test("actual visualize yes/no handler hides and restores latest canvas and run snapshots without editing graph", async () => {
  const ui = controller(), id = ui.complete();
  const gen = canvasCard(ui, 'gen'), score = canvasCard(ui, 'score');
  const first = liveSnapshot(id, 1), second = liveSnapshot(id, 2);
  replies(ui, [{...previewJob(id), visualization_revision: 1, loop_previews: {[id]: first}},
    {...previewJob(id), visualization_revision: 2, loop_previews: {[id]: second}}]);
  const workflow = ui.run('JSON.stringify(state.workflow)');
  await ui.run(`batchRequest('preview-loop', '${id}')`);
  const revision = ui.run('state.previewRevision');
  const firstShell = gen.querySelector('.node-preview-shell');
  const toggle = ui.document.querySelector('#visualize');
  toggle.value = 'no'; toggle.dispatchEvent({type: 'change'});
  assert.equal(ui.run('visualizeEnabled()'), false);
  assert.equal(ui.document.body.classList.contains('visualization-off'), true);
  assert.equal(previewCard(ui, id).hidden, true);
  assert.equal(gen.querySelector('.node-preview-shell'), firstShell);
  await ui.run("batchRequest('poll')");
  assert.deepEqual(plain(ui.run('state.nodeResults.gen')), first.outputs.gen);
  assert.equal(gen.querySelector('.node-preview-shell img').src, first.outputs.gen.image.image);
  assert.equal(ui.run(`multipleUI.previews.get('${id}').result.iterations`), 2);
  toggle.value = 'yes'; toggle.dispatchEvent({type: 'change'});
  assert.equal(ui.run('visualizeEnabled()'), true);
  assert.equal(ui.document.body.classList.contains('visualization-off'), false);
  assert.equal(previewCard(ui, id).hidden, false);
  assert.deepEqual(plain(ui.run('state.nodeResults.gen')), second.outputs.gen);
  assert.deepEqual(plain(ui.run('state.nodeResults.score')), second.outputs.score);
  assert.equal(gen.querySelectorAll('.node-preview-shell').length, 1);
  assert.equal(gen.querySelector('.node-preview-shell img').src, second.outputs.gen.image.image);
  assert.match(textOf(score.querySelector('.node-preview-shell')), /value · 文本\n2/);
  assert.equal(ui.document.querySelector('#run-visualization').querySelector('img').src, second.outputs.gen.image.image);
  assert.equal(ui.run('state.previewRevision'), revision);
  assert.equal(ui.run('JSON.stringify(state.workflow)'), workflow);
  assert.equal(ui.run('requests.length'), 2);
});

for (const mismatch of ['edited graph', 'different active job']) {
  test(`${mismatch} prevents a historical job from reviving current canvas results`, async () => {
    const ui = controller(), id = ui.complete();
    const gen = canvasCard(ui, 'gen'), score = canvasCard(ui, 'score');
    const first = liveSnapshot(id, 1), second = liveSnapshot(id, 2, true);
    const job = snapshot => ({...previewJob(id, 'completed'), visualization_revision: snapshot.iteration, loop_previews: {[id]: snapshot}});
    replies(ui, [job(first), job(second)]);
    await ui.run(`batchRequest('preview-loop', '${id}')`);
    const firstShell = gen.querySelector('.node-preview-shell');
    const previous = ui.run('JSON.stringify(state.nodeResults)');
    if (mismatch === 'edited graph') {
      const threshold = ui.field('条件 1 比较值', id);
      threshold.value = '8'; threshold.onchange();
      assert.equal(ui.run('state.nodeResultStale.gen'), true);
      assert.match(firstShell.querySelector('.preview-state').textContent, /历史预览/);
    } else {
      ui.run("state.activeJobId = 'new-job'; invalidateResults()");
      // Isolate job identity from revision protection: revisions match but IDs do not.
      ui.run('state.activeJobRevision = state.previewRevision');
    }
    const workflow = ui.run('JSON.stringify(state.workflow)');
    await ui.run("batchRequest('poll')");
    assert.equal(ui.run('JSON.stringify(state.nodeResults)'), previous);
    assert.equal(ui.run('state.nodeResultStale.gen'), true);
    assert.equal(ui.run('state.nodeResultStale.score'), true);
    assert.equal(gen.querySelector('.node-preview-shell'), firstShell);
    assert.equal(gen.querySelector('.node-preview-shell img').src, first.outputs.gen.image.image);
    assert.match(score.querySelector('.preview-state').textContent, /历史预览/);
    assert.doesNotMatch(gen.querySelector('.preview-state').textContent, /达标结果|最近一次成功输出/);
    assert.equal(ui.run('JSON.stringify(state.workflow)'), workflow);
  });
}

test("a job with visualize=false cannot apply unsolicited snapshots to canvas or run visualization", async () => {
  const ui = controller(), id = ui.complete();
  const gen = canvasCard(ui, 'gen');
  const before = ui.run('JSON.stringify(state.nodeResults)');
  replies(ui, [{...previewJob(id), visualize: false, visualization_revision: 1, loop_previews: {[id]: liveSnapshot(id, 1)}}]);
  await ui.run(`batchRequest('preview-loop', '${id}')`);
  assert.equal(ui.run('JSON.stringify(state.nodeResults)'), before);
  assert.equal(gen.querySelector('img'), null);
  assert.equal(ui.document.querySelector('#run-visualization').querySelectorAll('img').length, 0);
  assert.match(textOf(ui.document.querySelector('#run-visualization')), /本次运行关闭了可视化/);
  assert.equal(previewCard(ui, id).querySelectorAll('img').length, 0);
});
