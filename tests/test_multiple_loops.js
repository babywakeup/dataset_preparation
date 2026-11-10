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
    replaceChildren(...children) { this.children = []; this.append(...children); }
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
    remove() {}
  }
  const elements = new Map();
  const document = Object.assign(new Events(), {
    body: new Element(), createElement: tag => new Element(tag),
    querySelector(s) {
      if (/^#conditional-(enabled|legacy|legacy-controls|fields)$/.test(s)) return null;
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
    const info = (inputs, outputs) => ({inputs, outputs, help: {title: '同类测试节点', inputs, outputs, parameters: []}});
    state.catalog = {
      Source: info({}, {image: 'IMAGE'}),
      Generate: info({image: 'IMAGE', seed: 'INT'}, {image: 'IMAGE'}),
      Score: info({image: 'IMAGE'}, {value: 'FLOAT'}),
      MultiScore: info({image: 'IMAGE'}, {value: 'FLOAT', ok: 'BOOL', text: 'STRING'}),
      BoolScore: info({image: 'IMAGE'}, {ok: 'BOOLEAN'}),
      SaveImagesNode: info({image: 'IMAGE'}, {paths: 'JSON_RECORD'}),
      FolderImagesNode: info({}, {image: 'IMAGE'}),
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
  for (const id of ["conditional-enabled", "conditional-legacy", "conditional-legacy-controls", "conditional-fields"]) {
    assert.equal(ui.document.querySelector(`#${id}`), null);
    assert.doesNotMatch(html, new RegExp(`id="${id}"`));
    assert.doesNotMatch(scripts, new RegExp(id));
  }
  assert.doesNotMatch(scripts, /function (renderConditionalFields|renderConditionalRoles|beginConditionalPick|editConditional|conditionalReferences|conditionalPickHint)\(/);
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
  assert.match(ui.field("高级起点（可多选）", b).options.find(o => o.value === "alt").textContent, /alt.*同类测试节点.*Generate/);
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
  ui.button("重选条件 1", id).click(); ui.header("bool");
  assert.equal(ui.run(`multipleLoop('${id}').conditions[0].value`), true);
  let op = ui.field("条件 1 运算符", id); op.value = "<="; op.onchange();
  assert.match(ui.run("conditionalValidation()"), /布尔条件/);
  ui.button("重选条件 1", id).click(); ui.header("score");
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

test("no-seed start works; advanced seed rejects random or linked seed", () => {
  const ui = controller(), id = ui.add();
  ui.header("score"); ui.header("score"); ui.header("score");
  assert.deepEqual(plain(ui.run(`multipleLoop('${id}').seed_nodes`)), []);
  ui.button("画布增删递增种子", id).click(); ui.header("alt");
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

test("rename / advanced change invalidates both plans but preserves history and preview session", () => {
  const ui = controller(), id = ui.complete();
  ui.run("batch.job = {status: 'completed', iteration_history: [{loop_id: 'old', loop_name: '旧历史', iteration: 2}]};");
  for (const [label, value, field] of [["循环名称", "新名称", "name"], ["最大尝试次数", "8", "max_iterations"], ["超时秒数（0 禁用）", "5", "timeout_seconds"], ["耗尽 / 超时策略", "skip", "on_exhausted"]]) {
    ui.run("conditionalUI.plan = {}; conditionalUI.planKey = conditionalPlanKey(); loopUI.plan = {}; loopUI.planKey = 'old';");
    const control = ui.field(label, id); control.value = value; control.onchange();
    assert.equal(ui.run(`String(multipleLoop('${id}').${field})`), value);
    assert.equal(ui.run("conditionalUI.plan"), null);
    assert.equal(ui.run("loopUI.plan"), null);
    assert.equal(ui.run("state.previewSession"), "keep-session");
    assert.equal(ui.run("batch.job.iteration_history[0].loop_name"), "旧历史");
  }
});

test("condition add/remove and all/any edit only the requested loop", () => {
  const ui = controller(), a = ui.complete(), b = ui.complete("alt");
  const other = ui.run(`JSON.stringify(multipleLoop('${b}'))`);
  ui.button("＋ 添加条件", a).click(); ui.header("multi"); ui.port("multi", "value");
  assert.equal(ui.run(`multipleLoop('${a}').conditions.length`), 2);
  const mode = ui.field("条件组合", a); mode.value = "any"; mode.onchange();
  ui.button("删除条件 1", a).click();
  assert.equal(ui.run(`multipleLoop('${a}').conditions[0].node`), "multi");
  assert.equal(ui.run(`multipleLoop('${a}').mode`), "any");
  assert.equal(ui.run(`JSON.stringify(multipleLoop('${b}'))`), other);
});

test("advanced multi-start/end controls support explicit add/remove without rewriting bindings", () => {
  const ui = controller(), id = ui.complete();
  const graph = ui.run("JSON.stringify(state.workflow.nodes)");
  ui.button("画布增删终点", id).click(); ui.header("score");
  assert.deepEqual(plain(ui.run(`multipleLoop('${id}').end_nodes`)), ["gen", "score"]);
  ui.header("score");
  assert.deepEqual(plain(ui.run(`multipleLoop('${id}').end_nodes`)), ["gen"]);
  ui.button("画布增删起点", id).click(); ui.header("alt");
  ui.button("移除起点 alt", id).click();
  assert.deepEqual(plain(ui.run(`multipleLoop('${id}').start_nodes`)), ["gen"]);
  assert.equal(ui.run("JSON.stringify(state.workflow.nodes)"), graph);
});

test("stale nodes / target do not silently select another same-type instance", () => {
  const ui = controller(), id = ui.complete();
  ui.run("delete state.workflow.nodes.gen; renderConditional();");
  assert.deepEqual(plain(ui.run(`multipleLoop('${id}').start_nodes`)), ["gen"]);
  assert.match(ui.run("conditionalValidation()"), /失效/);
  ui.run("conditionalUI.targetId = 'deleted'; renderConditional();");
  assert.equal(ui.run("conditionalUI.targetId"), "deleted");
  assert.equal(ui.document.querySelector("#conditional-check").disabled, true);
});

test("multiple SaveImages require explicit title pick, shared by all loops", () => {
  const ui = controller();
  ui.run("state.workflow.nodes.save2 = structuredClone(state.workflow.nodes.save);");
  ui.complete(); assert.equal(ui.run("conditionalUI.targetId"), null);
  ui.button("画布选择最终保存目标").click(); ui.header("gen");
  assert.equal(ui.run("conditionalUI.picking.kind"), "target");
  ui.header("save2"); assert.equal(ui.run("conditionalUI.targetId"), "save2");
  assert.equal(ui.run("conditionalValidation()"), "");
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

test("selected save reads accepted intermediates; ordinary mid-body escape still requires explicit end", () => {
  const ui = controller(), id = ui.complete();
  ui.run("delete state.workflow.nodes.multi; delete state.workflow.nodes.bool;");
  ui.run(`multipleLoop('${id}').end_nodes = ['score'];`);
  assert.equal(ui.run("conditionalValidation()"), "");
  ui.run("state.workflow.nodes.alt.inputs.image = link('gen', 'image');");
  assert.match(ui.run("conditionalValidation()"), /end_nodes/);
  ui.run(`multipleLoop('${id}').end_nodes.push('gen');`);
  assert.equal(ui.run("conditionalValidation()"), "");
  ui.run(`multipleLoop('${id}').end_nodes = ['gen'];`);
  ui.run("state.workflow.nodes.alt.inputs.metric = link('score', 'value');");
  assert.equal(ui.run("conditionalValidation()"), "");
});

test("cross-region future dependency rejects interleaved non-overlapping regions", () => {
  const ui = controller(); ui.complete(); ui.complete("alt");
  ui.run("state.workflow.nodes.score.inputs.extra = link('alt', 'image'); state.workflow.nodes.score2.inputs.extra = link('gen', 'image');");
  assert.match(ui.run("conditionalValidation()"), /交叉依赖/);
});

test("all busy states reject Add/delete/rename/pick/advanced edits and cancel pending picker", () => {
  for (const lock of ["loopUI.busy", "loopUI.checking", "conditionalUI.checking", "batch.starting", "batch.requesting", "batch.restoring", "batch.uncertain"]) {
    const ui = controller(), id = ui.complete();
    const name = ui.field("循环名称", id), max = ui.field("最大尝试次数", id);
    const del = ui.button("删除循环", id), reselect = ui.button("重选起点", id);
    reselect.click();
    const before = ui.run("JSON.stringify(state.workflow)");
    ui.run(`${lock} = true; state.connecting = link('gen', 'image'); renderConditional();`);
    assert.equal(ui.run("conditionalUI.picking"), null, lock);
    assert.equal(ui.run("state.connecting"), null, lock);
    assert.equal(ui.document.querySelector("#conditional-add").disabled, true, lock);
    ui.run("addMultipleLoop()"); del.onclick(); reselect.onclick();
    name.value = "blocked"; name.onchange(); max.value = "3"; max.onchange();
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

test("polling / layout changes preserve focused fields and name commits on change only", () => {
  const ui = controller(), id = ui.complete();
  const field = ui.field("循环名称", id); field.focus(); field.value = "正在输入";
  assert.equal(field.oninput, undefined);
  ui.run("state.layout.gen = {x: 350, y: 200}; renderBatch(); renderConditional();");
  assert.equal(ui.field("循环名称", id), field);
  assert.equal(field.value, "正在输入");
  assert.notEqual(ui.run(`multipleLoop('${id}').name`), field.value);
  field.onchange(); assert.equal(ui.run(`multipleLoop('${id}').name`), "正在输入");
});

test("current loop progress and each history entry show loop name and stable ID", () => {
  const ui = controller(); ui.complete();
  ui.run(`batch.job = {id: 'task', status: 'running', total: 1, processed: 0, current_loop_id: 'loop_b', current_loop_name: '第二段', current_loop_index: 2, current_loop_count: 3,
    iteration_history: [{loop_id: 'loop_a', loop_name: '第一段', iteration: 1, accepted: true}, {loop_id: 'loop_b', loop_name: '第二段', iteration: 2, accepted: false}]}; renderBatch();`);
  assert.match(ui.document.querySelector("#batch-current").textContent, /当前循环：2 \/ 3 · 第二段 \(loop_b\)/);
  assert.match(ui.document.querySelector("#conditional-attempts").children[0].textContent, /第一段 \(loop_a\)/);
  assert.match(ui.document.querySelector("#conditional-attempts").children[1].textContent, /第二段 \(loop_b\)/);
});

test("plan request sends loops-only schema, renders server order and gates run", async () => {
  const ui = controller(); ui.complete(); ui.complete("alt");
  assert.equal(ui.document.querySelector("#conditional-run").disabled, true);
  await ui.run("checkConditionalPlan()");
  assert.deepEqual(plain(ui.run("Object.keys(requests[0].body.workflow.conditional_loop)")), ["loops"]);
  assert.equal(ui.run("requests[0].url"), "/api/conditional/plan");
  assert.equal(ui.document.querySelector("#conditional-run").disabled, false);
});

test("bounds, duplicate IDs and mixed flat/loops configuration block planning", () => {
  for (const mutation of ["l.max_iterations = 0", "l.timeout_seconds = -1", "l.mode = 'bad'", "l.on_exhausted = 'bad'", "l.start_nodes.push('gen')", "state.workflow.conditional_loop.mode = 'all'", "multipleLoops().push(structuredClone(l))"]) {
    const ui = controller(); ui.complete();
    ui.run(`{ const l = multipleLoops()[0]; ${mutation}; renderConditional(); }`);
    assert.notEqual(ui.run("conditionalValidation()"), "", mutation);
    assert.equal(ui.document.querySelector("#conditional-check").disabled, true, mutation);
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
    assert.equal(ui.document.querySelector("#conditional-check").disabled, true, field);
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
    assert.equal(ui.document.querySelector("#conditional-check").disabled, true);
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
    await ui.run("checkConditionalPlan()"); assert.equal(ui.run("requests.length"), 0);
    assert.equal(ui.document.querySelector("#conditional-run").disabled, true);
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
  assert.equal(ui.document.querySelector("#conditional-check").disabled, true);
  assert.equal(ui.document.querySelector("#conditional-error").textContent, "");
  assert.equal(ui.document.querySelector("#conditional-plan").textContent, "");
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

test("loop.js tolerates missing optional controller without falling back to flat UI", () => {
  const ui = controller();
  ui.run("renderMultipleEntry = undefined; consumeMultiplePick = undefined; renderConditional()");
  assert.equal(ui.document.querySelector("#conditional-check").disabled, true);
  assert.equal(ui.document.querySelector("#conditional-run").disabled, true);
  ui.run("conditionalUI.picking = {loopId: 'loop_1', kind: 'condition'}");
  assert.equal(ui.run("consumeConditionalPick('score', 'value')"), true);
  assert.equal(ui.run("conditionalUI.picking"), null);
});
