'use strict';

const $ = id => document.getElementById(id);
const primitiveNames = ['identity', 'rot90', 'rot180', 'rot270', 'hmirror', 'vmirror', 'dmirror', 'cmirror'];
const roleDefaults = {translate: 'anthropic/claude-fable-5.2', critic: 'openai/gpt-6.2-sol', alternatives: 'google/gemini-4-neon'};
const arcColors = ['#000000','#1e93ff','#fa3d31','#4ecc30','#ffdd00','#999999','#e53ba3','#ff851c','#88d8f1','#931131'];
const icons = {
  code: '<path d="m8 6-6 6 6 6m8-12 6 6-6 6m-3-16-2 20"/>',
  grid: '<path d="M3 3h7v7H3zm11 0h7v7h-7zM3 14h7v7H3zm11 0h7v7h-7z"/>',
  spark: '<path d="m12 3 2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5z"/>',
  book: '<path d="M3 4c4-1 7 0 9 2 2-2 5-3 9-2v15c-4-1-7 0-9 2-2-2-5-3-9-2zm9 2v15"/>',
  check: '<path d="m4 12 5 5L20 6"/>',
  play: '<path d="m6 3 15 9-15 9z" fill="currentColor" stroke="none"/>',
  table: '<rect x="2.5" y="4" width="19" height="16" rx="2"/><path d="M3 9h18M3 14h18M9 9v11"/>'
};

function icon(name) {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', '0 0 24 24');
  svg.setAttribute('fill', 'none');
  svg.setAttribute('stroke', 'currentColor');
  svg.setAttribute('stroke-width', '1.6');
  svg.setAttribute('stroke-linecap', 'round');
  svg.setAttribute('stroke-linejoin', 'round');
  svg.setAttribute('class', 'icon');
  svg.setAttribute('aria-hidden', 'true');
  svg.innerHTML = icons[name]; // All glyphs are local constants.
  return svg;
}

function el(tag, className = '', text = null) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== null && text !== undefined) node.textContent = text;
  return node;
}
function button(text, action, cls = '') {
  const node = el('button', cls, text);
  node.type = 'button';
  node.addEventListener('click', action);
  return node;
}
function option(value, label = value) { return new Option(label, value); }
function select(values, label, chosen = values[0]) {
  const node = el('select');
  node.setAttribute('aria-label', label);
  for (const value of values) node.add(option(value));
  node.value = chosen;
  return node;
}
function field(label, input) {
  const node = el('label', 'field', label);
  node.append(input);
  return node;
}
function notice(text) { return el('p', 'notice', text); }
function empty(message, glyph = 'table') {
  const container = el('div', 'empty');
  const mark = el('span', 'empty-mark');
  mark.append(icon(glyph));
  container.append(mark, el('p', '', message));
  return container;
}
function report(error) {
  $('global-error').hidden = false;
  $('global-error').textContent = error.message;
}
async function request(url, body) {
  const response = await fetch(url, body === undefined ? {} : {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)
  });
  let data;
  try { data = await response.json(); } catch { throw new Error('Invalid response from ' + url); }
  if (!response.ok) throw new Error(data.error ? data.error.code + ' · ' + data.error.message : String(response.status));
  return data;
}
function apiQuery(path, query) {
  return request(path + '?' + new URLSearchParams(query));
}

let language = {ops: [], examples: []};
let family = 'neox';
let activeTab;
const editor = $('source');
const initialScript = [
  'model neox-tiny', 'seed 0', '',
  '# Rotate every reader and writer together.',
  'R1: gauge fold_ln(); rotate(resid, orth1)', '',
  '# Compare the same rotation with LayerNorm intact.',
  'R2: gauge rotate(resid, orth1)', '',
  '# Measure the effect of deleting the middle layer.',
  'R3: edit delete(L[mid])'
].join('\n');

function navigate() {
  const id = location.hash.slice(1);
  activeTab = ['editor','arc','assist','explore','agent','language'].includes(id) ? id : 'editor';
  document.querySelectorAll('.tab-page').forEach(page => { page.hidden = page.id !== activeTab; });
  document.querySelectorAll('.tabs a').forEach(link => {
    if (link.hash === '#' + activeTab) link.setAttribute('aria-current', 'page');
    else link.removeAttribute('aria-current');
  });
  const tab = document.querySelector('.tabs a[aria-current]');
  document.title = 'WSBench · ' + tab.textContent;
  if (activeTab === 'arc') loadTasks();
}
function showHealth(health) {
  $('health').replaceChildren();
  for (const [key, value, available] of [
    ['pythia-14m', health.models_local['pythia-14m'] ? 'yes' : 'no', health.models_local['pythia-14m']],
    ['ARC data', health.arc_local ? 'yes' : 'no', health.arc_local],
    ['LLM', health.llm, health.llm !== 'no-key'],
    ['Claude direct', health.anthropic ? 'yes' : 'no', health.anthropic]
  ]) {
    const item = el('span', 'health-item');
    item.append(el('i', 'dot' + (available ? ' on' : '')), el('b','',key), document.createTextNode(value));
    $('health').append(item);
  }
}
function drawPalette() {
  const visible = language.ops.filter(op => op.family === family);
  $('families').replaceChildren();
  for (const name of ['neox','modadd','grid']) {
    const tab = button(name, () => { family = name; drawPalette(); });
    tab.setAttribute('aria-pressed', String(family === name));
    $('families').append(tab);
  }
  $('ops').replaceChildren();
  for (const op of visible) {
    const siblings = visible.filter(other => other.call.split('(')[0] === op.call.split('(')[0]);
    const name = siblings.length > 1 ? op.call.split(',')[0] + ', …)' : op.call.includes('(') ? op.call.split('(')[0] + '()' : op.call;
    const item = button(name, () => insertText(op.call));
    item.title = op.call + '\n' + op.args + '\n' + op.doc;
    item.setAttribute('aria-label', 'Insert ' + op.call);
    $('ops').append(item);
  }
  $('palette-count').textContent = visible.length + ' operations';
}
function setupLanguage(data) {
  language = data;
  $('language-card').textContent = data.card;
  for (const ex of data.examples) $('examples').add(option(ex.name));
  drawPalette();
  const {table, tbody} = makeTable(['Family','Call','Arguments','Description']);
  for (const op of data.ops) {
    const row = tbody.insertRow();
    [op.family, op.call, op.args, op.doc].forEach((value, i) => {
      const cell = row.insertCell();
      cell.append(el(i === 1 ? 'code' : 'span','',value));
    });
  }
  $('op-table').replaceChildren(table);
}
function openInEditor(script) {
  location.hash = 'editor';
  $('examples').value = '';
  setScript(script);
}
function setScript(script) {
  editor.value = script;
  editor.setSelectionRange(0, 0);
  editor.scrollLeft = editor.scrollTop = 0;
  sourceChanged();
}
function insertText(text, nextLine = false) {
  if (nextLine) {
    const end = editor.value.indexOf('\n', editor.selectionStart);
    const pos = end < 0 ? editor.value.length : end;
    editor.setRangeText('\n' + text, pos, pos, 'end');
  } else {
    editor.setRangeText(text, editor.selectionStart, editor.selectionEnd, 'end');
  }
  editor.focus({preventScroll:true});
  sourceChanged();
}

const syntax = /("(?:\\.|[^"\\])*")|(#[^\n]*)|([A-Za-z_]\w*(?=\s*:))|\b(model|seed|gauge|edit|reparam|approx|train|path|solve)\b|\b(true|false|\d+(?:\.\d+)?)\b|\b([A-Za-z_]\w*)(?=\s*\()/g;
function syntaxLine(line) {
  const fragment = document.createDocumentFragment();
  let pos = 0;
  for (const match of line.matchAll(syntax)) {
    fragment.append(document.createTextNode(line.slice(pos, match.index)));
    const type = match[1] ? 'literal' : match[2] ? 'comment' : match[3] ? 'label' : match[4] ? 'claim' : match[5] ? 'literal' : 'call';
    fragment.append(el('span', 't-' + type, match[0]));
    pos = match.index + match[0].length;
  }
  fragment.append(document.createTextNode(line.slice(pos)));
  return fragment;
}
let diagnostics = [];
let parseToken = 0;
let parseTimer;
function syncScroll() {
  const mirror = $('highlight').parentElement;
  mirror.scrollLeft = editor.scrollLeft;
  mirror.scrollTop = editor.scrollTop;
  $('line-numbers').style.transform = 'translateY(' + (-editor.scrollTop) + 'px)';
}
function updateCursor() {
  const before = editor.value.slice(0, editor.selectionStart);
  $('cursor').textContent = 'Ln ' + before.split('\n').length + ', Col ' + (before.length - before.lastIndexOf('\n'));
}
function drawSource() {
  $('line-numbers').replaceChildren();
  $('highlight').replaceChildren();
  const lines = editor.value.split('\n');
  lines.forEach((line, index) => {
    const issues = diagnostics.filter(issue => issue.line === index + 1);
    const kind = issues.some(issue => !issue.isHint) ? 'error' : issues.length ? 'hint' : '';
    const gutter = el('span', kind ? 'has-' + kind : '', String(index + 1));
    gutter.title = issues.map(issue => issue.code + ' · ' + issue.message).join('\n');
    const content = el('span', kind ? kind + '-line' : '');
    content.append(syntaxLine(line));
    $('line-numbers').append(gutter);
    $('highlight').append(content, document.createTextNode('\n'));
  });
  $('source-context').textContent = 'WSL v0.1  ·  ' + lines.length + (lines.length === 1 ? ' line' : ' lines');
  updateCursor();
  syncScroll();
}
function sourceChanged() {
  diagnostics = [];
  parseToken++;
  clearTimeout(parseTimer);
  $('problems').replaceChildren();
  drawSource();
  parseTimer = setTimeout(() => checkScript(), 300);
}
function gotoIssue(issue) {
  const lines = editor.value.split('\n');
  const pos = lines.slice(0, issue.line - 1).reduce((n, line) => n + line.length + 1, 0) + issue.col - 1;
  editor.focus();
  editor.setSelectionRange(pos, pos);
  editor.scrollTop = Math.max(0, (issue.line - 3) * 24);
  syncScroll();
}
async function checkScript(explicit = false) {
  clearTimeout(parseTimer);
  const token = ++parseToken;
  try {
    const data = await request('/api/parse', {script: editor.value});
    if (token !== parseToken) return;
    diagnostics = [...data.errors, ...data.hints.map(h => ({...h, isHint: true}))].sort((a,b) => a.line-b.line || a.col-b.col);
    drawSource();
    const rows = diagnostics.map(issue => {
      const node = button('', () => gotoIssue(issue), 'issue' + (issue.isHint ? ' hint' : ''));
      node.append(el('b', '', 'Ln ' + issue.line + ':' + issue.col + ' · ' + issue.code), document.createTextNode('  ' + issue.message));
      return node;
    });
    if (explicit && !rows.length) rows.push(el('span', '', 'Checked · ' + data.statements.length + ' statements'));
    $('problems').replaceChildren(...rows);
  } catch (error) { if (token === parseToken) $('problems').replaceChildren(notice(error.message)); }
}
function makeTable(headers, className = '') {
  const table = el('table', className);
  const head = table.createTHead().insertRow();
  for (const name of headers) head.append(el('th', '', name));
  return {table, tbody: table.createTBody()};
}
function numberText(n, signed = false) {
  if (n == null) return '—';
  const abs = Math.abs(n);
  const value = abs && (abs < .0001 || abs >= 10000) ? n.toExponential(2) : Number(n.toPrecision(4)).toString();
  return signed && n > 0 ? '+' + value : value;
}
function messageDetails(list) {
  const box = el('div');
  if (!list.length) { box.textContent = '—'; return box; }
  for (const item of list) {
    const detail = el('details');
    detail.append(el('summary','',item.code), el('p','',item.message));
    box.append(detail);
  }
  return box;
}
const panes = {};
function clearPane(key) {
  if (panes[key]) panes[key].active = false;
  delete panes[key];
}
function newPane(key, target, status, predicted = false) {
  clearPane(key);
  const pane = {key, target, status, predicted, active: true, id: null, selected: null, verdicts: [], revision: 0};
  pane.table = el('div');
  pane.detail = el('div', 'results-detail');
  pane.detail.hidden = true;
  target.replaceChildren(pane.table, pane.detail);
  panes[key] = pane;
  return pane;
}
function isCurrent(pane) { return pane.active && panes[pane.key] === pane; }
function statusPill(status) { return el('span', 'status s-' + status.toLowerCase(), status); }
function selectVerdict(pane, v) {
  pane.selected = v.line;
  pane.table.querySelectorAll('tr[data-line]').forEach(row => row.classList.toggle('selected', +row.dataset.line === v.line));
  showDetail(pane, v);
}
function drawVerdicts(pane, rows) {
  pane.verdicts = rows;
  if (!rows.length) { pane.table.replaceChildren(empty('This run has no statements.')); return; }
  const headers = ['Label','Claim','Source','Status','Measured','Δloss','Rel','Control ratio','Hints / warnings'];
  if (pane.predicted) headers.push('Predicted','Agree?');
  const {table, tbody} = makeTable(headers, 'verdicts');
  for (const v of rows) {
    const m = v.metrics || {};
    const row = tbody.insertRow();
    row.dataset.line = v.line;
    row.tabIndex = 0;
    row.setAttribute('aria-label', 'Details for ' + v.label);
    row.classList.toggle('selected', v.line === pane.selected);
    row.onclick = () => selectVerdict(pane, v);
    row.onkeydown = event => {
      if (event.target === row && ['Enter',' '].includes(event.key)) { event.preventDefault(); selectVerdict(pane,v); }
    };
    for (const [value, cls] of [
      [v.label,''],[v.claim,''],[v.source,'source-cell'],[statusPill(v.status),''],[v.measured ?? (m.total != null ? m.passed + '/' + m.total : null),''],
      [numberText(m.dloss,true),''],[numberText(m.rel),''],[numberText(m.control?.ratio),''],
      [messageDetails([...(v.hints || []), ...(v.warnings || []), ...(v.error ? [v.error] : [])]),'info-cell'],
      ...(pane.predicted ? [[v.predicted,''],[v.agree,'']] : [])
    ]) {
      const cell = row.insertCell();
      cell.className = cls;
      cell.append(value == null ? '—' : value);
    }
  }
  const wrap = el('div', 'table-wrap');
  wrap.append(table);
  const summary = el('div','result-summary');
  for (const stat of [...new Set(rows.map(v => v.status))]) {
    const item = el('span');
    item.append(el('b','',rows.filter(v => v.status === stat).length), ' ' + stat.toLowerCase());
    summary.append(item);
  }
  const previous = pane.table.querySelector('.table-wrap');
  const scrollLeft = previous?.scrollLeft || 0;
  const scrollTop = previous?.scrollTop || 0;
  pane.table.replaceChildren(summary, wrap);
  wrap.scrollLeft = scrollLeft;
  wrap.scrollTop = scrollTop;
}
async function poll(pane, id) {
  pane.id = id;
  while (isCurrent(pane)) {
    const run = await request('/api/runs/' + encodeURIComponent(id));
    if (!isCurrent(pane)) return;
    pane.status.textContent = run.status + (run.current_line != null ? ' · line ' + run.current_line : '');
    const previous = pane.verdicts.find(v => v.line === pane.selected);
    drawVerdicts(pane, run.verdicts);
    const selected = run.verdicts.find(v => v.line === pane.selected);
    if (selected && (selected.status !== previous?.status || selected.metrics?.curves?.step?.length !== previous?.metrics?.curves?.step?.length)) showDetail(pane, selected);
    if (['done','cancelled'].includes(run.status)) return;
    await new Promise(resolve => setTimeout(resolve, 400));
  }
}
async function runScript(key, script, target, status) {
  const pane = newPane(key, target, status);
  status.textContent = 'Submitting…';
  if (key === 'editor') { $('run').disabled = true; $('cancel').hidden = false; $('cancel').disabled = true; }
  try {
    const {run_id} = await request('/api/run', {script, source: key === 'editor' ? 'editor' : 'arc'});
    if (!isCurrent(pane)) return;
    if (key === 'editor') $('cancel').disabled = false;
    await poll(pane, run_id);
  } catch (error) { if (isCurrent(pane)) { status.textContent = 'Error'; pane.table.replaceChildren(notice(error.message)); } }
  finally { if (isCurrent(pane) && key === 'editor') { $('run').disabled = false; $('cancel').hidden = true; } }
}

function canvas(width, height, label, cls) {
  const node = el('canvas', cls);
  node.width = width;
  node.height = height;
  node.setAttribute('aria-label', label);
  node.setAttribute('role', 'img');
  return node;
}
function figure(caption, contents, note = '') {
  const node = el('figure','figure');
  node.append(el('figcaption','',caption), contents);
  if (note) node.append(el('small','',note));
  return node;
}
function arcGrid(grid, label) {
  if (!Array.isArray(grid) || !grid.length || !Array.isArray(grid[0])) return el('p','notice', typeof grid === 'string' ? grid : 'Undecodable grid');
  const h = grid.length, w = grid[0].length, side = Math.max(h,w) <= 15 ? 16 : 10;
  const node = canvas((w + 2) * side, (h + 2) * side, label, 'arc-grid');
  const ctx = node.getContext('2d');
  ctx.fillStyle = '#222';
  ctx.fillRect(0, 0, node.width, node.height);
  ctx.fillStyle = '#555';
  ctx.fillRect(side, side, w*side+1, h*side+1);
  grid.forEach((row, y) => row.forEach((color, x) => {
    ctx.fillStyle = arcColors[color] || '#222';
    ctx.fillRect((x+1)*side+1, (y+1)*side+1, side-1, side-1);
  }));
  return node;
}
function heatmap(data, scale, label) {
  const h = data.length, w = data[0].length;
  const node = canvas(w, h, label, 'heatmap');
  const ctx = node.getContext('2d'), pixels = ctx.createImageData(w,h);
  const blue = [33,102,172], red = [178,24,43];
  let i = 0;
  for (const row of data) for (const v of row) {
    const t = Math.min(1, Math.abs(v) / (scale || 1));
    const color = v < 0 ? blue : red;
    for (const c of color) pixels.data[i++] = Math.round(247 + t * (c-247));
    pixels.data[i++] = 255;
  }
  ctx.putImageData(pixels,0,0);
  return node;
}

function chart(title, xs, series, xLabel, log = false) {
  const node = document.createElementNS('http://www.w3.org/2000/svg','svg');
  node.setAttribute('viewBox','0 0 640 320');
  node.setAttribute('width','640');
  node.setAttribute('height','320');
  node.setAttribute('class','chart');
  node.setAttribute('role','img');
  node.setAttribute('aria-label',title);
  function draw(tag, attrs, content) {
    const child = document.createElementNS(node.namespaceURI,tag);
    for (const [k,val] of Object.entries(attrs)) child.setAttribute(k,val);
    if (content != null) child.textContent = content;
    node.append(child);
    return child;
  }
  const colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd'];
  const x0=66, x1=618, y0=247, y1=33;
  const xMin = xs.reduce((min,x) => Math.min(min,x),Infinity);
  const xLast = xs.reduce((max,x) => Math.max(max,x),-Infinity);
  const xMax = xLast === xMin ? xMin + 1 : xLast;
  const positive = series.flatMap(s => s.values).filter(y => y > 0);
  const low = log ? Math.floor(Math.log10(positive.length ? Math.min(...positive) : 1e-6)) : 0;
  const high = log ? Math.max(low + 1, Math.ceil(Math.log10(positive.length ? Math.max(...positive) : 1))) : 1;
  const px = val => x0 + (val - xMin) / (xMax - xMin || 1) * (x1-x0);
  const py = val => y0 - ((log ? Math.log10(Math.max(val,10**low)) : val) - low) / (high-low) * (y0-y1);
  const label = (x,y,text,anchor='start') => draw('text',{x,y,'text-anchor':anchor,fill:'#536170','font-size':11,'font-family':'monospace'},text);
  draw('title',{},title);
  label(x0,17,title);
  for (let i=0; i<5; i++) {
    const x=x0+i*(x1-x0)/4, y=y0-i*(y0-y1)/4, exp=low+i*(high-low)/4;
    draw('line',{x1:x0,x2:x1,y1:y,y2:y,stroke:'#e4e8ed'});
    draw('line',{x1:x,x2:x,y1:y0,y2:y0+4,stroke:'#758398'});
    label(x0-9,y+4,numberText(log ? 10**exp : exp),'end');
    label(x,y0+17,numberText(xMin+i*(xMax-xMin)/4),'middle');
  }
  label((x0+x1)/2,y0+35,xLabel,'middle');
  series.forEach(({name,values},i) => {
    const points = xs.map((x,j) => [px(x),py(values[j])]).filter(([x,y]) => Number.isFinite(x) && Number.isFinite(y));
    draw('polyline',{points:points.map(p=>p.join(',')).join(' '),stroke:colors[i],'stroke-width':2.5,fill:'none'});
    if (points.length===1) draw('circle',{cx:points[0][0],cy:points[0][1],r:3,fill:colors[i]});
    const x = x0+i*140;
    draw('line',{x1:x,x2:x+16,y1:305,y2:305,stroke:colors[i],'stroke-width':3});
    label(x+22,309,name);
  });
  return node;
}
function showDetail(pane, verdict) {
  const detail = pane.detail;
  pane.revision++;
  detail.hidden = false;
  const heading = el('h3','',verdict.label + ' · ' + (verdict.claim || 'Statement'));
  detail.replaceChildren(heading);
  if (verdict.error) detail.append(notice(verdict.error.code + ' · ' + verdict.error.message));
  const metrics = verdict.metrics || {};
  if (metrics.changed_tensors) {
    if (!metrics.changed_tensors.length) { detail.append(el('p','legend','No tensors changed.')); return; }
    const picker = el('select');
    picker.setAttribute('aria-label','Changed tensor');
    for (const t of metrics.changed_tensors) picker.add(option(t.name, t.name + '  [' + t.shape.join(' × ') + ']'));
    const info = el('div');
    const draw = () => {
      const tensor = metrics.changed_tensors.find(t => t.name === picker.value);
      tensorPanel(pane, verdict, tensor, info);
    };
    picker.onchange = draw;
    detail.append(field('Changed tensors (' + metrics.changed_tensors.length + ')', picker), info);
    draw();
  } else if (metrics.curves?.step?.length) {
    const c = metrics.curves;
    detail.append(chart('Accuracy', c.step, [{name:'train_acc',values:c.train_acc},{name:'test_acc',values:c.test_acc}], 'Step'),
      chart('Loss (log scale)', c.step, [{name:'train_loss',values:c.train_loss},{name:'test_loss',values:c.test_loss}], 'Step', true));
  } else if (metrics.points?.length) {
    const key = 't' in metrics.points[0] ? 't' : 'c';
    detail.append(chart('Path accuracy', metrics.points.map(p=>p[key]), ['train_acc','test_acc'].map(name=>({name,values:metrics.points.map(p=>p[name])})), key));
  } else if (metrics.cosine != null) {
    detail.append(el('p','result-selection','Cosine ' + numberText(metrics.cosine) + '  ·  ‖A‖ ' + numberText(metrics.norm_a) + '  ·  ‖B‖ ' + numberText(metrics.norm_b)));
  } else if (metrics.by_split) {
    detail.append(el('p','result-selection',Object.entries(metrics.by_split).map(([s,[n,total]])=>s+' '+n+'/'+total).join('  ·  ')));
    const f = metrics.first_failure;
    if (f) {
      detail.append(el('h4','','First failure · ' + f.split + ' [' + f.index + ']'));
      const grid = el('div','triple');
      for (const [key,value] of [['Input',f.input],['Expected',f.expected],['Got',f.got]]) grid.append(figure(key,arcGrid(value,key)));
      detail.append(grid);
    }
  } else if (!verdict.error) {
    detail.append(el('p','legend',verdict.status === 'RUNNING' ? 'Waiting for measurements…' : 'No measurements available.'));
  }
}
function tensorPanel(pane, verdict, tensor, target) {
  const revision = ++pane.revision;
  let load = 0;
  const gridToggle = el('input');
  gridToggle.type = 'checkbox';
  const bounds = {};
  const controls = el('div','detail-controls');
  const results = el('div','triple'), legend = el('p','legend');
  if (tensor.shape.length === 2) {
    for (const [label,defaultValue,limit] of [
      ['r0',0,tensor.shape[0]],['r1',Math.min(16,tensor.shape[0]),tensor.shape[0]],
      ['c0',0,tensor.shape[1]],['c1',Math.min(16,tensor.shape[1]),tensor.shape[1]]
    ]) {
      const input = el('input');
      input.type = 'number'; input.min = '0'; input.max = limit; input.step = '1'; input.value = defaultValue;
      input.onchange = () => { if (gridToggle.checked) paint(); };
      controls.append(field(label,input));
      bounds[label] = input;
    }
    const toggle = el('label','checkbox');
    toggle.append(gridToggle, ' ARC view');
    gridToggle.onchange = paint;
    controls.append(toggle, button('Insert gridop(...)', () => {
      if (!validBounds()) { legend.textContent = 'Choose valid integer bounds for this tensor.'; return; }
      location.hash = 'editor';
      insertText('edit gridop(rot90, block(' + JSON.stringify(tensor.name) + ', ' + Object.values(bounds).map(n=>n.valueAsNumber).join(', ') + '))',true);
    }));
  }
  function validBounds() {
    const [r0,r1,c0,c1] = Object.values(bounds).map(n=>n.valueAsNumber);
    return [r0,r1,c0,c1].every(Number.isInteger) && r0>=0 && r0<r1 && r1<=tensor.shape[0] && c0>=0 && c0<c1 && c1<=tensor.shape[1];
  }
  async function paint() {
    const current = ++load;
    const query = {run_id:pane.id,line:verdict.line,name:tensor.name};
    const isArc = gridToggle.checked;
    legend.textContent = isArc ? 'ARC view: colour = decile of the value inside this block (0 = lowest tenth, 9 = highest). Colours are bins, not categories.' : '';
    if (isArc) {
      if (!validBounds() || bounds.r1.valueAsNumber-bounds.r0.valueAsNumber>30 || bounds.c1.valueAsNumber-bounds.c0.valueAsNumber>30) {
        results.replaceChildren(notice('Select a valid block of at most 30 × 30 entries.')); return;
      }
      query.view = 'arc';
      for (const [k,n] of Object.entries(bounds)) query[k] = n.value;
    }
    results.replaceChildren(el('p','legend','Loading tensor…'));
    try {
      const views = await Promise.all(['before','after','diff'].map(async which => {
        const data = await apiQuery('/api/tensor',{...query,which});
        const label = which[0].toUpperCase() + which.slice(1);
        return isArc ? figure(label,arcGrid(data.grid,label)) : figure(label,heatmap(data.data,data.scale,label),'scale '+numberText(data.scale)+' · block '+data.block.join('×'));
      }));
      if (isCurrent(pane) && revision === pane.revision && current===load) results.replaceChildren(...views);
    } catch (error) { if (revision===pane.revision && current===load) results.replaceChildren(notice(error.message)); }
  }
  target.replaceChildren(controls,legend,results);
  paint();
}

let taskList = null;
let loadingTasks = false;
let taskName = '';
let taskTicket = 0;
async function loadTasks() {
  if (taskList || loadingTasks) return;
  loadingTasks = true;
  $('tasks').replaceChildren(el('p','legend','Loading tasks…'));
  try {
    taskList = await request('/api/arc/tasks');
    drawTasks();
    if (taskList.length) await selectTask(taskList.find(t => t.task === 'task150')?.task || taskList[0].task);
  } catch (error) {
    $('tasks').replaceChildren(notice(error.message));
    $('task-detail').replaceChildren(empty('ARC task data is unavailable.','grid'));
  } finally { loadingTasks = false; }
}
function drawTasks() {
  const search = $('task-filter').value.trim().toLowerCase();
  const visible = (taskList || []).filter(t => (!search || [t.task,t.original_id || ''].some(s => s.toLowerCase().includes(search))) &&
    (!$('single').checked || t.single_primitive));
  $('task-count').textContent = visible.length + (visible.length===1 ? ' task' : ' tasks');
  const group = document.createDocumentFragment();
  for (const task of visible) {
    const row = button('',()=>selectTask(task.task),'task-row');
    row.setAttribute('aria-pressed',String(taskName === task.task));
    row.dataset.task = task.task;
    row.append(el('strong','',task.task),el('small','',task.bytes_raw != null ? task.bytes_raw+' B' : '—'),el('small','',task.original_id || '—'));
    if (task.single_primitive) row.append(el('span','primitive-tag',task.single_primitive));
    group.append(row);
  }
  if (!visible.length) group.append(empty('No matching tasks.','grid'));
  $('tasks').replaceChildren(group);
}
function updateArcTarget(selected) {
  const target = $('target');
  const wasGrid = target.value.startsWith('grid:');
  Array.from(target.options).filter(o=>o.value.startsWith('grid:')).forEach(o=>o.remove());
  target.add(option('grid:' + selected));
  if (wasGrid) target.value='grid:'+selected;
}
async function selectTask(name) {
  taskName = name;
  const ticket = ++taskTicket;
  clearPane('arc');
  document.querySelectorAll('.task-row').forEach(row=>row.setAttribute('aria-pressed',String(row.dataset.task===name)));
  updateArcTarget(name);
  const area = $('task-detail');
  area.replaceChildren(el('p','legend','Loading ' + name + '…'));
  try {
    const task = await request('/api/arc/task/' + encodeURIComponent(name));
    if (ticket !== taskTicket) return;
    const meta = taskList.find(row => row.task===name);
    const header = el('div','section-heading task-detail-top');
    header.append(el('h2','',name),el('span','meta',task.original_id||''));
    const controls = el('div','task-options'), actions = el('div','detail-controls');
    const primitive = select(primitiveNames,'Primitive',meta.single_primitive||'identity');
    const size = select(['fixed','gated'],'Compile size'), source = select(['all','train'],'Compile from');
    const verdict = el('div'), status = el('span','meta'), preview = el('div','compile-viz');
    let script = 'model grid:'+name+'\nsolve '+(meta.single_primitive?'prim('+meta.single_primitive+')':'golf()');
    let previewTicket=0;
    async function compilePreview(input,shape) {
      const call = ++previewTicket;
      preview.replaceChildren(el('p','legend','Loading compiled network…'));
      try {
        const data = await apiQuery('/api/arc/compiled', {...input,...(shape ? {shape} : {})});
        if (call!==previewTicket || ticket!==taskTicket) return;
        const top = el('div','section-heading');
        top.append(el('h3','','Compiled network'));
        if (input.size==='gated') {
          const picker = select(data.sizes.map(dim=>dim.join('x')),'Compiled input shape',data.shape.join('x'));
          picker.onchange = ()=>compilePreview(input,picker.value);
          top.append(picker);
        }
        const map = el('pre','src-map',data.src_grid.map(row=>row.map(i=>String(i).padStart(3)).join(' ')).join('\n'));
        const scale = data.P_preview.reduce((s,row)=>Math.max(s,...row),0);
        const fig = figure('P preview',heatmap(data.P_preview,scale,'Compiled permutation matrix'));
        fig.style.width='300px';
        preview.replaceChildren(top,figure('Source cell index',map),fig);
      } catch (error) { if (call===previewTicket) preview.replaceChildren(notice(error.message)); }
    }
    function runOp(kind) {
      const selected = {task:name,prim:primitive.value,size:size.value,from:source.value};
      const expr = kind==='golf' ? 'golf()' : kind==='prim' ? 'prim('+selected.prim+')' : 'compile('+selected.prim+', size='+selected.size+', from='+selected.from+')';
      script = 'model grid:'+name+'\nsolve '+expr;
      startScript.textContent = script;
      runScript('arc',script,verdict,status);
      if (kind==='compile') compilePreview(selected);
      else { previewTicket++; preview.replaceChildren(); }
    }
    actions.append(button('Run golf()',()=>runOp('golf')),button('Run prim()',()=>runOp('prim')),button('Compile',()=>runOp('compile'),'primary'),button('Open in editor',()=>openInEditor(script)));
    const selects = el('div','detail-controls');
    selects.append(field('Primitive',primitive),field('Size',size),field('From',source));
    const startScript = el('pre','raw',script);
    controls.append(selects,actions,startScript,status,verdict,preview);
    area.replaceChildren(header,controls);
    for (const [split,examples,total] of [['Train',task.train,task.train.length],['Test',task.test,task.test.length],['ARC-GEN',task.arc_gen_first20.slice(0,3),task.n_arcgen]]) {
      area.append(el('h3','',split+' · '+(total>examples.length ? examples.length+' of ' : '')+total));
      const collection = el('div','examples');
      examples.forEach((example,index)=>{
        const box = el('div','example');
        const pair = el('div','pair');
        pair.append(figure('Input',arcGrid(example.input,split+' input '+index)),figure('Expected',arcGrid(example.output,split+' output '+index)));
        box.append(el('h4','','Example '+(index+1)),pair);
        collection.append(box);
      });
      area.append(collection);
    }
    area.append(el('h3','','Raw solution'),el('small','', 'Raw '+(task.bytes_raw??'—')+' B  ·  Submitted '+(task.bytes_submitted??'—')+' B'),
      el('pre','raw',task.solution_raw??'Solution unavailable.'));
  } catch (error) { if (ticket===taskTicket) area.replaceChildren(notice(error.message)); }
}
function setupRoles() {
  const list = el('datalist');
  list.id = 'model-choices';
  for (const id of Object.values(roleDefaults)) list.append(option(id));
  $('role-fields').append(list);
  for (const [role,id] of Object.entries(roleDefaults)) {
    const input = el('input');
    input.id='role-'+role;
    input.value=id;
    input.required=true;
    input.setAttribute('list','model-choices');
    input.autocomplete='off';
    input.spellcheck=false;
    input.oninput=()=>{
      $('fable-warning').hidden=!['critic','alternatives'].some(r=>$('role-'+r).value===roleDefaults.translate);
    };
    $('role-fields').append(field(role[0].toUpperCase()+role.slice(1),input));
  }
}
function renderAssist(data, output, note) {
  const body = el('div','assist-detail');
  body.append(el('p','cost',(note ? note + ' · ' : '') + 'Cost · $'+Number(data.cost_usd).toFixed(4)));
  for (const role of Object.keys(roleDefaults)) {
    const detail = el('details','details-block');
    const reply=data.roles[role];
    detail.append(el('summary','',role+' · '+(reply?.model||'Skipped')+(reply?.provider ? ' · via '+reply.provider : '')),el('pre','raw',reply?.text || reply?.error || 'No reply.'));
    if (reply && !reply.ok && reply.text && reply.error) detail.append(notice(reply.error));
    if (role === 'translate' && !data.run_id) detail.open = true;   // nothing was measured: show why
    body.append(detail);
  }
  for (const problem of data.parse_errors) body.append(notice(problem.role+(problem.line!=null?' · line '+problem.line+':'+problem.col:'')+' · '+problem.code+' '+problem.message));
  body.append(el('h3','','Combined script'),el('pre','raw',data.script),button('Open in editor',()=>openInEditor(data.script)));
  output.replaceChildren(body);
}
// The models take about half a minute. The page starts the request, then asks for its state every second,
// so a dropped connection or a reload does not lose the answer.
async function followAssist(id, note) {
  const status = $('assist-status'), output = $('assist-output');
  const token = followAssist.token = {};
  const started = Date.now();
  let failures = 0;
  for (;;) {
    let task;
    try { task = await request('/api/assist/' + encodeURIComponent(id)); failures = 0; }
    catch (error) {
      if (followAssist.token !== token) return;
      if (++failures > 30 || /E_NOT_FOUND/.test(error.message)) { output.replaceChildren(notice(error.message)); status.textContent = 'Error'; return; }
      status.textContent = 'Connection lost, retrying…';
      await new Promise(resolve => setTimeout(resolve, 2000));
      continue;
    }
    if (followAssist.token !== token) return;
    if (task.status === 'running') {
      status.textContent = 'Calling models… ' + Math.round((Date.now() - started) / 1000) + ' s';
      await new Promise(resolve => setTimeout(resolve, 1000));
      continue;
    }
    if (task.status === 'error') { output.replaceChildren(notice(task.error.code + ' · ' + task.error.message)); status.textContent = 'Error'; return; }
    const data = task.result;
    renderAssist(data, output, note);
    if (data.run_id) {
      const results = el('div');
      output.append(results);
      try { await poll(newPane('assist',results,status,true),data.run_id); }
      catch (error) { output.append(notice(error.message)); status.textContent = 'Error'; }
    } else status.textContent='Complete';
    return;
  }
}
async function assist(event) {
  event.preventDefault();
  const execute = event.submitter?.id !== 'assist-only';
  const roles = Object.fromEntries(Object.keys(roleDefaults).map(role=>[role,$('role-'+role).value.trim()]));
  const status = $('assist-status'), output = $('assist-output');
  clearPane('assist');
  status.textContent = 'Calling models…';
  output.replaceChildren();
  $('assist-run').disabled = $('assist-only').disabled = true;
  try {
    const {assist_id} = await request('/api/assist/start', {text:$('abstract').value, model:$('target').value, roles, execute});
    await followAssist(assist_id);
  } catch (error) { output.append(notice(error.message)); status.textContent='Error'; }
  finally { $('assist-run').disabled = $('assist-only').disabled = false; }
}
document.querySelectorAll('[data-icon]').forEach(node=>node.prepend(icon(node.dataset.icon)));
$('examples').onchange=()=>{
  const example = language.examples.find(ex=>ex.name===$('examples').value);
  if (example) setScript(example.script);
};
editor.addEventListener('input',sourceChanged);
editor.addEventListener('scroll',syncScroll);
for (const event of ['selectionchange','keyup','click','focus']) editor.addEventListener(event,updateCursor);
$('check').onclick=()=>checkScript(true);
$('run').onclick=()=>runScript('editor',editor.value,$('editor-results'),$('editor-status'));
$('cancel').onclick=async()=>{
  const pane=panes.editor;
  if (!pane?.id) return;
  $('cancel').disabled=true;
  try { await request('/api/runs/'+encodeURIComponent(pane.id)+'/cancel',{}); }
  catch (error) { if (isCurrent(pane)) { $('cancel').disabled=false; report(error); } }
};
$('task-filter').oninput=drawTasks;
$('single').onchange=drawTasks;
$('assist-form').onsubmit=assist;
window.addEventListener('hashchange',()=>{navigate(); window.scrollTo(0,0);});
setupRoles();
setScript(initialScript);
$('editor-results').replaceChildren(empty('Run the script to see its verdicts.'));
$('assist-output').replaceChildren(empty('Translate a source text to see the proposed experiments.','spark'));
navigate();
// After a reload, show the most recent editor run and assist run again (also runs that are still in progress).
async function resumeEditor(run) {
  $('examples').value = '';
  setScript(run.script);
  const pane = newPane('editor', $('editor-results'), $('editor-status'));
  if (['queued','running'].includes(run.status)) { $('run').disabled = true; $('cancel').hidden = false; $('cancel').disabled = false; }
  try { await poll(pane, run.run_id); }
  catch (error) { if (isCurrent(pane)) { $('editor-status').textContent = 'Error'; pane.table.replaceChildren(notice(error.message)); } }
  finally { if (isCurrent(pane)) { $('run').disabled = false; $('cancel').hidden = true; } }
}
async function resumeAssist() {
  let latest;
  try { latest = (await request('/api/assist/latest')).assist; } catch { return; }
  if (!latest || panes.assist || followAssist.token) return;
  if (latest.request?.text && !$('abstract').value) $('abstract').value = latest.request.text;
  if (latest.request?.model && [...$('target').options].some(o => o.value === latest.request.model)) $('target').value = latest.request.model;
  await followAssist(latest.assist_id, 'Restored from ' + new Date(latest.created * 1000).toLocaleString());
}
// ---- Explore: a planner model runs several assist rounds for one idea -------------------------------------------
const exploreView = {id: null, revision: -1, rounds: new Map(), token: null};
function exploreRound(round, output) {
  let view = exploreView.rounds.get(round.n);
  if (!view) {
    const card = el('div','assist-detail explore-round');
    view = {card, head: el('h3'), note: el('p','meta'), question: el('p'), info: el('p','cost'), problems: el('div'),
            script: el('details','details-block'), table: el('div'), pane: null};
    card.append(view.head, view.note, view.question, view.info, view.problems, view.script, view.table);
    output.append(card);
    exploreView.rounds.set(round.n, view);
  }
  view.head.textContent = 'Round ' + round.n + ' · ' + round.status;
  view.note.textContent = round.note || '';
  view.question.textContent = round.question;
  view.info.textContent = round.assist ? 'Cost · $' + Number(round.cost_usd).toFixed(4) : '';
  view.problems.replaceChildren();
  if (round.problem) view.problems.append(notice('No measurement · ' + round.problem));
  if (round.assist && !view.script.childElementCount) {
    view.script.append(el('summary','','Script and model replies'), el('pre','raw',round.assist.script),
      button('Open in editor',()=>openInEditor(round.assist.script)));
    for (const role of Object.keys(roleDefaults)) {
      const reply = round.assist.roles[role];
      view.script.append(el('h4','',role+' · '+(reply?.model||'Skipped')), el('pre','raw',reply?.text || reply?.error || 'No reply.'));
    }
  }
  if (round.verdicts) {
    if (!view.pane) view.pane = newPane('explore-' + round.n, view.table, el('span'), true);
    view.pane.id = round.run_id;
    drawVerdicts(view.pane, round.verdicts);
  }
}
function drawExplore(task) {
  const status = $('explore-status'), output = $('explore-output');
  const live = ['running','waiting'].includes(task.status);
  $('explore-run').disabled = live;
  $('explore-cancel').hidden = !live;
  if (task.revision === exploreView.revision) return;
  exploreView.revision = task.revision;
  const state = task.state;
  if (!state) { status.textContent = task.status === 'error' ? 'Error' : 'Planning round 1…'; }
  else {
    for (const round of state.rounds) exploreRound(round, output);
    status.textContent = task.status + ' · ' + state.rounds.length + '/' + state.max_rounds + ' rounds · $' + Number(state.cost_usd).toFixed(4) + ' of $' + Number(state.budget_usd).toFixed(2);
  }
  output.querySelectorAll('.explore-extra').forEach(node => node.remove());
  const extra = el('div','assist-detail explore-extra');
  for (const item of state?.answers || []) extra.append(el('p','meta','Asked: ' + item.question + ' — Answer: ' + item.answer));
  if (task.status === 'waiting') {
    const input = el('input'); input.type = 'text'; input.id = 'explore-answer'; input.setAttribute('aria-label','Your answer');
    const send = button('Send answer', async () => {
      if (!input.value.trim()) return;
      send.disabled = true;
      try { await request('/api/explore/' + exploreView.id + '/answer', {answer: input.value.trim()}); }
      catch (error) { send.disabled = false; report(error); }
    }, 'primary');
    extra.append(el('h3','','The planner asks'), el('p','',state.waiting), input, send);
  }
  if (task.status === 'error') extra.append(notice(task.error.code + ' · ' + task.error.message));
  if (state?.stopped && !live) extra.append(el('p','meta','Stopped · ' + state.stopped));
  if (state?.report) extra.append(el('h3','','Report'), el('pre','raw report',state.report));
  if (state?.report_error) extra.append(notice('No report · ' + state.report_error));
  if (extra.childElementCount) output.append(extra);
}
async function followExplore(id) {
  const token = exploreView.token = {};
  for (const key of exploreView.rounds.keys()) clearPane('explore-' + key);
  Object.assign(exploreView, {id, revision: -1, rounds: new Map()});
  $('explore-output').replaceChildren();
  let failures = 0;
  for (;;) {
    let task;
    try { task = await request('/api/explore/' + encodeURIComponent(id)); failures = 0; }
    catch (error) {
      if (exploreView.token !== token) return;
      if (++failures > 30 || /E_NOT_FOUND/.test(error.message)) { $('explore-output').append(notice(error.message)); $('explore-status').textContent = 'Error'; return; }
      $('explore-status').textContent = 'Connection lost, retrying…';
      await new Promise(resolve => setTimeout(resolve, 2000));
      continue;
    }
    if (exploreView.token !== token) return;
    drawExplore(task);
    if (!['running','waiting'].includes(task.status)) return;
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
}
async function startExplore(event) {
  event.preventDefault();
  $('explore-run').disabled = true;
  $('explore-status').textContent = 'Starting…';
  try {
    const {explore_id} = await request('/api/explore/start', {text: $('idea').value, model: $('explore-target').value,
      rounds: Number($('explore-rounds').value), budget_usd: Number($('explore-budget').value)});
    await followExplore(explore_id);
  } catch (error) { $('explore-output').replaceChildren(notice(error.message)); $('explore-status').textContent = 'Error'; $('explore-run').disabled = false; }
}
async function resumeExplore() {
  let latest;
  try { latest = (await request('/api/explore/latest')).explore; } catch { return; }
  if (!latest || exploreView.token) return;
  if (latest.request?.text && !$('idea').value) $('idea').value = latest.request.text;
  if ([...$('explore-target').options].some(o => o.value === latest.request?.model)) $('explore-target').value = latest.request.model;
  await followExplore(latest.explore_id);
}
$('explore-form').onsubmit = startExplore;
$('explore-cancel').onclick = async () => {
  if (!exploreView.id) return;
  try { await request('/api/explore/' + exploreView.id + '/cancel', {}); } catch (error) { report(error); }
};
$('explore-output').replaceChildren(empty('Describe an idea to start a series of measured rounds.','spark'));
// ---- Agent: a Claude Managed Agent that uses WSBench as its tools ---------------------------------------------------
const agentView = {id: null, nodes: [], token: null, status: null, canContinue: false, key: null, busy: false, polling: false, wake: null, refresh: false};
function agentError(error) {
  const box = $('agent-error');
  box.hidden = !error;
  box.textContent = error ? (error.message || String(error)) : '';
}
// Send, Stop and the hint under the message box follow the state of the conversation.
function agentButtons() {
  const live = agentView.status === 'running';
  const loading = Boolean(agentView.id) && agentView.status == null;
  $('agent-send').disabled = live || loading || agentView.busy;
  $('agent-stop').hidden = !live;
  let mode, placeholder = 'Describe an idea or answer the agent, in any language.';
  if (live) mode = 'The agent is working. You can stop it or wait for its answer.';
  else if (loading) mode = 'Loading the conversation…';
  else if (!agentView.id) mode = 'Your message starts a new conversation.';
  else if (agentView.canContinue) { mode = 'Your message continues this conversation.'; placeholder = 'Answer the agent or ask a follow-up.'; }
  else { mode = 'This conversation cannot be continued. Your message starts a new one.'; placeholder = 'Describe a new idea, in any language.'; }
  $('agent-mode').textContent = mode;
  $('agent-text').placeholder = placeholder;
}
// A sleep that a tap on Send, the browser coming back online or the tab becoming visible again can cut short.
function nap(ms) {
  return new Promise(resolve => {
    const done = () => { clearTimeout(timer); if (agentView.wake === done) agentView.wake = null; resolve(); };
    const timer = setTimeout(done, ms);
    agentView.wake = done;
  });
}
function showAgentResult(node, item) {
  // the verdict table of a wsl_run stands in for its result, unless the run is gone from the server's history
  if (item.result == null || node.result.childElementCount || (item.name === 'wsl_run' && !node.evicted)) return;
  node.result.append(el('summary','', node.evicted ? 'Result (run no longer stored)' : 'Result'), el('pre','raw', item.result));
}
function agentItem(item, index) {
  let node = agentView.nodes[index];
  if (!node) {
    node = {root: el('div','assist-detail agent-item agent-' + item.kind), pane: null, table: null, drawn: null, evicted: false};
    if (item.kind === 'tool') {
      node.head = el('h4');
      node.input = el('details','details-block');
      node.input.append(el('summary','','Input'), el('pre','raw', item.input?.script ?? JSON.stringify(item.input, null, 1)));
      if (item.input?.script) node.input.append(button('Open in editor',()=>openInEditor(item.input.script)));
      node.table = el('div');
      node.result = el('details','details-block');
      node.root.append(node.head, node.input, node.table, node.result);
    } else {
      node.root.append(el('p','meta', {user:'You', agent:'Agent', note:'Note', error:'Error'}[item.kind] || item.kind), el('pre','raw report', item.text));
    }
    $('agent-output').append(node.root);
    agentView.nodes[index] = node;
  }
  if (item.kind !== 'tool') return;
  node.head.textContent = item.name + (item.result == null ? ' · running…' : '');
  showAgentResult(node, item);
  if (item.run_id && node.drawn !== 'final') {
    if (!node.pane) { node.pane = newPane('agent-' + index, node.table, el('span')); node.pane.id = item.run_id; }
    const pane = node.pane;
    request('/api/runs/' + item.run_id).then(run => {
      if (!isCurrent(pane)) return;
      const previous = pane.verdicts.find(v => v.line === pane.selected);
      drawVerdicts(pane, run.verdicts);
      const selected = run.verdicts.find(v => v.line === pane.selected);
      if (selected && (selected.status !== previous?.status || selected.metrics?.curves?.step?.length !== previous?.metrics?.curves?.step?.length)) showDetail(pane, selected);
      if (['done','cancelled'].includes(run.status)) node.drawn = 'final';
    }).catch(error => {
      if (!/E_NOT_FOUND/.test(error.message)) return;      // the server forgot this run (it keeps only the newest ones)
      node.drawn = 'final'; node.evicted = true;
      showAgentResult(node, item);
    });
  }
}
function drawAgent(record) {
  agentView.status = record.status;
  agentView.canContinue = record.status === 'idle' || (record.status === 'error' && Boolean(record.session_id));
  agentButtons();
  record.items.forEach(agentItem);
  // Everything below is drawn again when the revision, the status or the error changes, or after a failed poll.
  const key = [record.revision, record.status, record.error || '', record.cost_usd ?? ''].join('|');
  if (key === agentView.key) return;
  agentView.key = key;
  $('agent-stop').disabled = false;
  $('agent-status').textContent = record.status + (record.cost_usd != null ? ' · $' + Number(record.cost_usd).toFixed(2) + ' of $' + Number(record.budget_usd).toFixed(2) : '');
  $('agent-output').querySelectorAll('.agent-extra').forEach(node => node.remove());
  const extra = el('div','assist-detail agent-extra');
  const lastItem = record.items[record.items.length - 1];
  if (record.status === 'error' && record.error && !(lastItem?.kind === 'error' && lastItem.text === record.error)) extra.append(notice(record.error));
  const firstMessage = record.items.find(item => item.kind === 'user')?.text;
  if (record.status === 'error' && !record.session_id && firstMessage) {
    const problem = notice('');
    problem.hidden = true;
    problem.setAttribute('role', 'alert');
    const again = button('Send my first message again', async () => {
      again.disabled = true; problem.hidden = true;
      try {
        const {agent_conv_id} = await request('/api/agent/start', {text: firstMessage, budget_usd: Number(record.budget_usd) || Number($('agent-budget').value)});
        followAgent(agent_conv_id, 'running');
      } catch (error) { again.disabled = false; problem.textContent = error.message; problem.hidden = false; }
    }, 'primary');
    extra.append(again, problem);
  }
  if (['budget','terminated'].includes(record.status)) extra.append(el('p','meta','This conversation cannot be continued. Your next message starts a new one.'));
  if (record.status === 'error' && record.session_id) extra.append(el('p','meta','The conversation is still open: send a new message to continue.'));
  if (record.console_url) { const link = el('a','', 'Open the session in the Claude Console'); link.href = record.console_url; link.target = '_blank'; link.rel = 'noopener'; extra.append(link); }
  if (extra.childElementCount) $('agent-output').append(extra);
}
// Polls the conversation while a turn runs, and a few more times until the tool cards have their final numbers.
// It never gives up on network errors; it ends when the conversation is not running any more.
async function agentLoop(id, token) {
  agentView.polling = true;
  let failures = 0, settling = 0;
  try {
    for (;;) {
      let record;
      agentView.refresh = false;
      try { record = await request('/api/agent/' + encodeURIComponent(id)); failures = 0; }
      catch (error) {
        if (agentView.token !== token) return;
        if (/E_NOT_FOUND/.test(error.message)) {
          $('agent-output').append(notice(error.message));
          $('agent-status').textContent = 'Error';
          Object.assign(agentView, {status: 'error', canContinue: false});
          agentButtons();
          return;
        }
        failures++;
        agentView.key = null;                      // the next good answer redraws the label, even at the same revision
        $('agent-status').textContent = 'Connection lost, retrying…';
        await nap(Math.min(2000 * failures, 15000));
        if (agentView.token !== token) return;
        continue;
      }
      if (agentView.token !== token) return;
      drawAgent(record);
      if (agentView.refresh) continue;             // Send or Stop happened while this request was in flight
      if (record.status === 'running') settling = 0;
      else if (!agentView.nodes.some(node => node.pane && node.drawn !== 'final') || ++settling > 5) return;
      await nap(record.status === 'running' ? 1000 : 3000);
      if (agentView.token !== token) return;
    }
  } finally {
    if (agentView.token === token) agentView.polling = false;
  }
}
function pollAgent() {                              // make the current conversation refresh now (and keep following it)
  if (!agentView.id) return;
  if (agentView.polling) { if (agentView.wake) agentView.wake(); else agentView.refresh = true; }
  else agentLoop(agentView.id, agentView.token);
}
function followAgent(id, status = null) {
  const token = agentView.token = {};
  agentView.wake?.();
  agentView.nodes.forEach((_, index) => clearPane('agent-' + index));
  Object.assign(agentView, {id, nodes: [], key: null, status, canContinue: false, polling: false, refresh: false});
  $('agent-output').replaceChildren();
  agentButtons();
  return agentLoop(id, token);
}
async function sendAgent(event) {
  event.preventDefault();
  const text = $('agent-text').value.trim();
  if (!text || agentView.busy || agentView.status === 'running') return;
  agentError(null);
  agentView.busy = true;
  agentButtons();
  try {
    if (agentView.id && agentView.canContinue) {
      await request('/api/agent/' + agentView.id + '/message', {text});
      agentView.status = 'running';                // until the next poll says otherwise
      pollAgent();
    } else {
      const {agent_conv_id} = await request('/api/agent/start', {text, budget_usd: Number($('agent-budget').value)});
      followAgent(agent_conv_id, 'running');
    }
    $('agent-text').value = '';
  } catch (error) { agentError(error); }
  finally {
    agentView.busy = false;
    agentButtons();
    if (matchMedia('(any-pointer: fine)').matches) $('agent-text').focus({preventScroll: true});    // not on a tablet: the keyboard would cover the answer
  }
}
async function resumeAgent() {
  let latest;
  try { latest = (await request('/api/agent/latest')).agent; } catch { return; }
  if (latest && !agentView.token) followAgent(latest.agent_conv_id);
}
$('agent-form').onsubmit = sendAgent;
$('agent-new').onclick = async () => {
  if (agentView.id && agentView.status === 'running') {
    if (!confirm('The agent is still working. Stop it and start a new conversation?')) return;
    try { await request('/api/agent/' + agentView.id + '/interrupt', {}); }
    catch (error) { agentError(error); return; }
  }
  agentError(null);
  agentView.token = {};
  agentView.wake?.();
  agentView.nodes.forEach((_, index) => clearPane('agent-' + index));
  Object.assign(agentView, {id: null, nodes: [], key: null, status: null, canContinue: false, polling: false, refresh: false});
  $('agent-output').replaceChildren(empty('Describe an idea to start a conversation with the agent.','spark'));
  $('agent-status').textContent = '';
  agentButtons();
};
$('agent-stop').onclick = async () => {
  if (!agentView.id) return;
  agentError(null);
  $('agent-stop').disabled = true;                  // until the next redraw; the interrupt is sent once
  try { await request('/api/agent/' + agentView.id + '/interrupt', {}); pollAgent(); }
  catch (error) { $('agent-stop').disabled = false; agentError(error); }
};
window.addEventListener('online', pollAgent);
document.addEventListener('visibilitychange', () => { if (!document.hidden) pollAgent(); });
$('agent-output').replaceChildren(empty('Describe an idea to start a conversation with the agent.','spark'));
agentButtons();
async function restoreRuns() {
  let runs;
  try { runs = await request('/api/runs'); } catch { return; }
  const lastEditor = runs.find(run => run.source === 'editor');
  if (lastEditor && !panes.editor) resumeEditor(lastEditor);
}
restoreRuns();
resumeAssist();
resumeExplore();
resumeAgent();
request('/api/health').then(health => { showHealth(health); $('agent-nokey').hidden = Boolean(health.agent); }).catch(error=>{$('health').textContent=error.message;});
request('/api/language').then(setupLanguage).catch(report);
