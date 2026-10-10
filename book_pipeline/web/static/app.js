'use strict';

const $ = (id) => document.getElementById(id);
const state = { job: null, jobs: [], file: null, blocks: [], types: [], subheadings: [], colors: {}, typography: null, navigation: { toc: [], targets: {} }, page: 1, zoom: 1, uploading: false, panel: 'info', preview: 'samples', revision: null, selection: 0, connected: true };
const busyStates = new Set(['queued', 'processing', 'rendering', 'extracting']);
const statuses = { uploaded: '等待处理', queued: '任务排队中', processing: '识别与整理中', extracting: '补充识别中', ready: '等待设置样式', rendering: '正在生成整书', complete: '书稿已完成', failed: '处理失败', interrupted: '处理已中断' };
const palette = [{ name: '橙色', value: '#FF8618' }, { name: '绿色', value: '#00A652' }, { name: '灰色', value: '#737B86' }, { name: '蓝色', value: '#0E86E1' }, { name: '青色', value: '#00AEF7' }, { name: '红色', value: '#CB5353' }, { name: '深蓝', value: '#0356A1' }];
const icons = () => window.lucide?.createIcons();
const esc = (value) => String(value ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const jobBusy = (job) => busyStates.has(job?.status) || busyStates.has(job?.preview_status);
const busy = () => jobBusy(state.job) || state.uploading || !!state.removingBlock;
const colorName = (color) => palette.find((p) => p.value === color)?.name || '自定义';
const typeColor = (kind) => kind === 'subheading' ? '#3C71B7' : state.colors[kind] || state.types.find((t) => t.kind === kind)?.default || '#737B86';
const sameColors = (a, b) => JSON.stringify(Object.entries(a || {}).sort()) === JSON.stringify(Object.entries(b || {}).sort());
const headingDepth = (pattern) => (pattern.match(/罗马数字|希腊字母|数字|汉字|字母/g) || []).length;
const headingSize = (body, pattern) => body + ({ 1: 4.5, 2: 2.5, 3: 1.5 }[headingDepth(pattern)] || 1);
const typography = (value) => { const body = value?.body_size ?? 10.5; return { body_size: body, subheading_sizes: { '数字.': body + 4.5, '数字.数字': body + 2.5, '数字.数字.数字': body + 1.5, ...value?.subheading_sizes } }; };
const sameTypography = (a, b) => { const x = typography(a), y = typography(b); return x.body_size === y.body_size && sameColors(x.subheading_sizes, y.subheading_sizes); };
const presentation = () => ({ colors: state.colors, typography: state.typography || typography() });

function toast(message) {
  $('toast').textContent = message;
  $('toast').hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { $('toast').hidden = true; }, 5500);
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) {
    const detail = Array.isArray(data.detail) ? data.detail.map((e) => e.msg).join('；') : data.detail;
    throw new Error(detail || '请求失败，请重试。');
  }
  return data;
}

function showPanel(panel) {
  if (panel === 'colors' && !state.job?.has_blocks) return;
  state.panel = panel;
  for (const name of ['info', 'keywords', 'progress', 'colors']) {
    $(`${name}-panel`).hidden = panel !== name;
    $(`${name}-tab`).classList.toggle('active', panel === name);
    $(`${name}-tab`).setAttribute('aria-selected', String(panel === name));
  }
}

const keywordFields = { statements: 'statement-keywords', examples: 'example-keywords', subheadings: 'subheading-keywords' };
let defaultKeywords = null;

function drawKeywords(candidates) {
  if (!candidates) return;
  const common = { statements: ['定理', '公式', '命题', '公理', '定义', '引理', '推论'], examples: ['例题', '例', 'Example'], subheadings: ['数字.', '数字.数字', '数字.数字.数字', '汉字、', '(汉字)', '小写字母.', '大写字母.', '罗马数字.', '希腊字母.'] };
  for (const [group, id] of Object.entries(keywordFields)) {
    const entries = [...candidates[group]].sort((a, b) => {
      const x = common[group].indexOf(a), y = common[group].indexOf(b);
      if (x >= 0 || y >= 0) return (x < 0 ? 999 : x) - (y < 0 ? 999 : y);
      return (group === 'subheadings' ? headingDepth(a) - headingDepth(b) : 0) || a.localeCompare(b, 'zh-CN');
    });
    $(id).value = entries.join('\n');
  }
  $('keyword-error').hidden = true;
  $('keyword-status').textContent = state.job?.keywords_changed && state.job?.has_blocks ? '候选集合已修改，重新识别后生效。' : '';
}

function readKeywords() {
  return Object.fromEntries(Object.entries(keywordFields).map(([group, id]) => [group, $(id).value.split(/\r?\n/).map((s) => s.trim()).filter(Boolean)]));
}

async function saveKeywords() {
  const candidates = readKeywords();
  if (state.job) {
    state.job = await api(`/api/jobs/${state.job.id}/keyword-candidates`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(candidates) });
    drawKeywords(state.job.keyword_candidates);
    paintJob();
  } else {
    drawKeywords(await api('/api/keyword-candidates/validate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(candidates) }));
  }
  $('keyword-status').textContent = state.job?.has_blocks ? '已保存；重新识别后生效。' : '将在开始处理时使用此候选集合。';
  $('keyword-error').hidden = true;
}

async function applyKeywords() {
  if (!state.job?.has_blocks || busy()) return;
  try {
    await saveKeywords();
    state.job = await api(`/api/jobs/${state.job.id}/extract`, { method: 'POST' });
    showPanel('progress');
    paintJob();
    await refreshJobs();
  } catch (error) { $('keyword-error').textContent = error.message; $('keyword-error').hidden = false; toast(error.message); }
}

function showPreview(view) {
  state.preview = view;
  for (const name of ['samples', 'page', 'blocks', 'toc']) $(`${name}-preview`).hidden = name !== view;
  document.querySelectorAll('[data-preview]').forEach((button) => {
    const active = button.dataset.preview === view;
    button.classList.toggle('active', active);
    button.setAttribute('aria-selected', String(active));
  });
  if (view === 'blocks') drawBlocks();
  if (view === 'toc') drawToc();
  updatePreviewStatus();
}

function showView(history) {
  $('workspace-view').hidden = history;
  $('history-view').hidden = !history;
  $('workspace-nav').classList.toggle('active', !history);
  $('history-nav').classList.toggle('active', history);
  $('breadcrumb-current').textContent = history ? '我的书稿' : '书稿工作台';
}

function renderStages() {
  const stages = state.job?.stages || ['MinerU 识别', '标题匹配', '正文整理', '引用映射', '整书排版'].map((label) => ({ label, status: 'pending' }));
  $('stage-count').textContent = `${stages.filter((s) => s.status === 'complete').length} / 5`;
  $('stage-list').innerHTML = stages.map((stage, i) => {
    const symbol = stage.status === 'complete' ? '<i data-lucide="check"></i>' : stage.status === 'interrupted' || (stage.status === 'running' && !state.connected) ? '<i data-lucide="pause"></i>' : stage.status === 'running' ? '<i data-lucide="loader-circle" class="spin"></i>' : stage.status === 'failed' ? '<i data-lucide="x"></i>' : String(i + 1).padStart(2, '0');
    const hasCount = Number.isInteger(stage.done) && Number.isInteger(stage.total) && stage.total > 0;
    const count = ['running', 'interrupted'].includes(stage.status) && hasCount ? `${stage.done} / ${stage.total}${stage.unit ? ' ' + stage.unit : ''}` : '';
    const detail = stage.detail || (stage.status === 'pending' ? (i >= 3 ? '等待生成整本 PDF' : '等待开始') : '');
    const window = stage.window;
    const scope = window ? `第 ${window.index} / ${window.total} 批 · 第 ${window.start}-${window.end} 页` : '';
    const pages = window && Number.isInteger(stage.completed_pages) ? `已处理 ${stage.completed_pages} / ${stage.source_total} 页` : stage.id === 'mineru' && !hasCount && stage.source_total ? `共 ${stage.source_total} 页` : '';
    const meter = stage.status === 'running' && state.connected ? `<progress ${hasCount ? `max="${Number(stage.total)}" value="${Number(stage.done)}"` : ''} aria-label="${esc(detail || stage.label)}进度"></progress>` : '';
    return `<div class="stage ${esc(stage.status)}"><span class="stage-icon">${symbol}</span><div class="stage-copy"><div class="stage-name"><span>${esc(stage.id === 'render' ? '整书排版' : stage.label)}</span><small>${esc(count)}</small></div><div class="stage-detail">${esc(detail)}${scope ? '<br>' + esc(scope) : ''}${pages ? '<br>' + esc(pages) : ''}</div>${meter}</div></div>`;
  }).join('');
  icons();
}

function updateDirty() {
  const dirty = !sameColors(state.colors, state.job?.preview_colors) || !sameTypography(state.typography, state.job?.preview_typography);
  $('dirty-status').textContent = dirty && state.job?.has_blocks ? '待预览' : '';
  $('sample-notice').hidden = !(dirty && state.job?.preview_samples?.length);
  $('preview-notice').hidden = !(state.job?.has_pdf && (state.job.render_content_stale || !sameColors(state.colors, state.job.rendered_colors) || !sameTypography(state.typography, state.job.rendered_typography)));
}

function drawTypography() {
  state.typography ||= typography();
  const body = state.typography.body_size;
  $('body-size').value = body;
  $('subheading-sizes').innerHTML = state.subheadings.map((item, n) => {
    const size = state.typography.subheading_sizes[item.pattern] ?? headingSize(body, item.pattern);
    return `<div class="type-setting"><label class="size-row" for="heading-size-${n}"><span>${esc(item.pattern)}<small>${item.count} 个小标题 · 推荐 ${headingSize(body, item.pattern)} pt</small></span><span><input id="heading-size-${n}" type="number" min="${body + .5}" max="36" step="0.5" value="${size}" data-heading-pattern="${esc(item.pattern)}" aria-label="${esc(item.pattern)}小标题字号"><span>pt</span></span></label><button class="text-button remove-type" data-remove-type="subheading" data-remove-pattern="${esc(item.pattern)}" aria-label="删除${esc(item.pattern)}类小标题" ${busy() ? 'disabled' : ''}>删除此类型</button></div>`;
  }).join('') || '<p class="style-note">尚未提取到编号小标题。</p>';
}

function updatePreviewStatus() {
  const job = state.job;
  $('preview-status').textContent = state.preview === 'samples'
    ? (busyStates.has(job?.preview_status) ? '正在渲染样例' : job?.preview_samples?.length ? `${job.preview_samples.length} 个样例${job.preview_samples.some((s) => s.content_kind === 'body') ? ' · 含正文对照' : ''}` : '等待生成样例预览')
    : job?.has_pdf ? `${job.render_pages} 页 · 整书 PDF` : job ? statuses[job.status] : '等待处理';
}

function drawSamples() {
  const job = state.job;
  const samples = job?.preview_samples || [];
  const running = busyStates.has(job?.preview_status);
  const error = job?.preview_error;
  $('sample-message').classList.toggle('error', !!error);
  $('sample-message').textContent = error || (running ? '正在渲染每种类型的一个特殊块，请稍候…' : samples.length
    ? '真实正文对照与每类一个样例，可比较字号和例题字体。'
    : job?.has_blocks ? state.blocks.length ? '点击“生成预览”，查看每种类型的一个特殊块。' : '正文中未提取到特殊块。' : '完成识别与正文整理后，可预览特殊块样式。');
  const key = `${job?.id || ''}:${job?.preview_revision || 0}`;
  if ($('sample-list').dataset.key !== key) {
    $('sample-list').dataset.key = key;
    $('sample-list').innerHTML = samples.map((sample) => {
      const sizes = typography(job.preview_typography);
      const type = sample.content_kind === 'body' ? '正文对照' : sample.pattern ? '小标题 · ' + sample.pattern : state.types.find((t) => t.kind === sample.kind)?.label || sample.kind;
      const size = sample.pattern ? sizes.subheading_sizes[sample.pattern] ?? headingSize(sizes.body_size, sample.pattern) : sizes.body_size;
      const font = sample.pattern ? '蓝色加粗' : sample.content_kind === 'example' ? '题目楷体 · 解答仿宋' : '正文字号';
      return `<article class="sample-card" data-sample-kind="${esc(sample.kind)}"><header><strong>${esc(type)} · ${size} pt</strong><small>${esc(font)} · ${esc(sample.section)} · 原书第 ${sample.source_page} 页</small></header>${sample.pages.map((page) => `<img src="/api/jobs/${job.id}/samples/${esc(sample.kind)}/${page}.png?v=${job.preview_revision}" alt="${esc(sample.label)}渲染样例${sample.pages.length > 1 ? ' · 第 ' + (sample.pages.indexOf(page) + 1) + ' 页' : ''}" loading="lazy"><div class="sample-image-error" hidden>图片加载失败，请重新生成预览。</div>`).join('')}</article>`;
    }).join('');
  }
}

function drawColors(openKind = null) {
  $('color-summary').textContent = `${state.types.length} 类特殊块 · ${state.subheadings.length} 种小标题`;
  $('color-list').innerHTML = state.types.length ? state.types.map((type) => {
    const color = typeColor(type.kind);
    return `<div class="color-row"><div class="color-row-main"><span class="color-dot" style="background:${color}"></span><span class="color-label">${esc(type.label)}<span class="color-count">${type.count}</span></span><button class="color-trigger" data-color-kind="${esc(type.kind)}" aria-label="设置${esc(type.label)}颜色" aria-expanded="${openKind === type.kind}" ${busy() ? 'disabled' : ''}><span class="swatch" style="background:${color}"></span><span>${colorName(color)}</span><i data-lucide="chevron-down"></i></button></div><button class="text-button remove-type" data-remove-type="${esc(type.kind)}" aria-label="删除${esc(type.label)}类型" ${busy() ? 'disabled' : ''}>删除此类型</button><div class="swatch-panel" data-picker="${esc(type.kind)}" ${openKind !== type.kind ? 'hidden' : ''}><div class="swatches">${palette.map((p) => `<button class="swatch-choice ${p.value === color ? 'selected' : ''}" style="background:${p.value}" data-kind="${esc(type.kind)}" data-color="${p.value}" aria-label="${esc(type.label)}使用${p.name}" title="${p.name}" aria-pressed="${p.value === color}">${p.value === color ? '<i data-lucide="check"></i>' : ''}</button>`).join('')}</div><div class="custom-color"><label for="color-${esc(type.kind)}">自定义</label><input id="color-${esc(type.kind)}" type="color" value="${color}" data-custom-kind="${esc(type.kind)}" aria-label="${esc(type.label)}自定义颜色"><span class="color-hex">${color}</span></div></div></div>`;
  }).join('') : '<div class="list-empty">正文中未识别到特殊块</div>';
  updateDirty();
  drawTypography();
  icons();
}

function drawBlocks() {
  const needle = $('block-search').value.trim().toLowerCase();
  const filter = $('block-filter').value;
  const blocks = state.blocks.filter((b) => (!filter || (b.pattern ? state.subheadings.find((t) => t.pattern === b.pattern)?.kind : b.kind) === filter) && (!needle || `${b.label} ${b.name} ${b.text} ${b.section}`.toLowerCase().includes(needle)));
  $('block-list').innerHTML = blocks.length ? blocks.map((block) => `<article class="block-item" style="--block-color:${typeColor(block.kind)}"><button class="block-open" data-target="${esc(block.id)}"><strong>${esc(block.label)}${block.name && !block.label.includes(block.name) ? ' · ' + esc(block.name) : ''}</strong><p>${esc(block.text)}</p><div class="block-meta"><span>${esc(block.section)}</span><span>原书第 ${block.page} 页</span></div></button></article>`).join('') : '<div class="list-empty">没有符合条件的特殊块</div>';
}

function drawToc() {
  $('toc-list').innerHTML = state.navigation.toc.length ? state.navigation.toc.map(([level, title, page]) => `<button class="toc-item" style="--level:${Math.min(level - 1, 5)}" data-page="${page}"><span>${esc(title)}</span><span>${page}</span></button>`).join('') : '<div class="list-empty">生成书稿后显示目录</div>';
}

function updatePage() {
  const job = state.job;
  state.page = Math.max(1, Math.min(state.page, job?.render_pages || 1));
  $('empty-preview').hidden = !!job?.has_pdf;
  $('page-scroll').hidden = !job?.has_pdf;
  $('page-total').textContent = `/ ${job?.render_pages || 0}`;
  $('page-number').value = state.page;
  $('page-number').max = job?.render_pages || 1;
  $('page-number').disabled = !job?.has_pdf;
  $('previous-page').disabled = !job?.has_pdf || state.page <= 1;
  $('next-page').disabled = !job?.has_pdf || state.page >= job.render_pages;
  for (const id of ['zoom-in', 'zoom-out', 'fit-page']) $(id).disabled = !job?.has_pdf;
  if (job?.has_pdf) {
    const src = `/api/jobs/${job.id}/pages/${state.page}.png?v=${job.revision}`;
    if ($('pdf-page').getAttribute('src') !== src) {
      $('image-error').hidden = true;
      $('pdf-page').hidden = false;
      $('pdf-page').src = src;
      $('pdf-page').alt = `${job.book.title} · 第 ${state.page} 页`;
      $('page-scroll').scrollTop = 0;
    }
  } else {
    $('pdf-page').removeAttribute('src');
  }
  updateZoom();
}

function updateZoom() {
  $('zoom-label').textContent = `${Math.round(state.zoom * 100)}%`;
  const base = Math.min(430, Math.max(200, $('page-scroll').clientWidth - 50));
  $('pdf-page').style.width = `${base * state.zoom}px`;
}

function gotoPage(page) {
  if (!state.job?.has_pdf) return;
  state.page = Math.max(1, Math.min(state.job.render_pages, Math.trunc(Number(page)) || 1));
  showPreview('page');
  updatePage();
}

function paintJob() {
  const job = state.job;
  const offline = !!job && !state.connected;
  const status = offline ? '服务连接中断' : busyStates.has(job?.preview_status) ? '正在生成样例预览' : job ? statuses[job.status] : '等待上传';
  $('job-status').className = `job-status ${offline ? 'interrupted' : job?.status || ''}`;
  $('job-status').innerHTML = `<span class="status-dot"></span>${status}`;
  $('processing-title').textContent = job?.book.title || '尚未上传书籍';
  $('processing-author').textContent = job?.book.author || '';
  $('colors-tab').disabled = !job?.has_blocks;
  $('start-processing').disabled = !!job || state.uploading;
  for (const id of ['book-title', 'book-author', 'remove-file', 'dropzone']) $(id).disabled = !!job || state.uploading;
  $('new-book').disabled = state.uploading;
  $('job-error').hidden = !job?.error && !offline;
  $('job-error').textContent = offline ? '本地服务连接中断，当前显示的是最后收到的进度。' : job?.error || '';
  $('retry-job').hidden = !job || !['failed', 'interrupted', 'uploaded'].includes(job.status);
  $('retry-job').innerHTML = job?.has_blocks ? '<i data-lucide="rotate-ccw"></i>重新生成整本 PDF' : '<i data-lucide="rotate-ccw"></i>继续处理';
  $('go-colors').hidden = !job?.has_blocks || busy();
  $('render-button').disabled = !job?.has_blocks || busy();
  $('render-button').innerHTML = busyStates.has(job?.preview_status) ? '<i data-lucide="loader-circle" class="spin"></i><span>正在生成样例</span>' : '<i data-lucide="refresh-cw"></i><span>生成预览</span>';
  $('render-book-button').disabled = !job?.has_blocks || busy();
  $('refresh-blocks-button').hidden = !job?.has_blocks || (job?.content_method === 'page_content_keywords_v9' && !job?.keywords_changed);
  $('refresh-blocks-button').disabled = busy();
  $('apply-keywords').hidden = !job?.has_blocks;
  for (const id of [...Object.values(keywordFields), 'save-keywords', 'reset-keywords', 'apply-keywords']) $(id).disabled = busy() || state.uploading || !defaultKeywords;
  document.querySelectorAll('#typography-settings input').forEach((control) => { control.disabled = busy(); });
  $('reset-colors').disabled = busy();
  document.querySelectorAll('[data-color-kind], [data-custom-kind], [data-color]').forEach((control) => { control.disabled = busy(); });
  document.querySelectorAll('[data-remove-type]').forEach((control) => { control.disabled = busy(); });
  $('download-pdf').classList.toggle('disabled', !job?.has_pdf);
  $('download-pdf').setAttribute('aria-disabled', String(!job?.has_pdf));
  if (job?.has_pdf) $('download-pdf').href = `/api/jobs/${job.id}/pdf`;
  else $('download-pdf').removeAttribute('href');
  updatePreviewStatus();
  $('preview-book-label').textContent = job?.book.title || 'PDF 回炉所';
  $('empty-detail').textContent = job?.has_blocks ? '点击“生成整本 PDF”后显示书稿' : busy() ? '书籍处理中' : '等待书籍处理';
  const step = !job ? 0 : job.has_blocks ? 2 : 1;
  document.querySelectorAll('[data-step]').forEach((item) => {
    item.classList.toggle('current', Number(item.dataset.step) === step);
    item.classList.toggle('done', Number(item.dataset.step) < step);
  });
  renderStages();
  drawSamples();
  updateDirty();
  updatePage();
  icons();
}

async function loadDetails(job) {
  if (!job.has_blocks) return;
  const selection = state.selection;
  const detail = await api(`/api/jobs/${job.id}/blocks`);
  if (selection !== state.selection) return;
  state.job = detail.job;
  state.blocks = detail.blocks;
  state.types = detail.types;
  state.subheadings = detail.subheadings || [];
  state.colors = Object.fromEntries(Object.entries(state.colors).filter(([kind]) => state.types.some((t) => t.kind === kind)));
  $('block-count').textContent = state.blocks.length;
  $('block-filter').innerHTML = '<option value="">全部类型</option>' + state.types.map((t) => `<option value="${esc(t.kind)}">${esc(t.label)}</option>`).join('') + state.subheadings.map((t) => `<option value="${esc(t.kind)}">小标题 ${esc(t.pattern)}</option>`).join('');
  drawColors();
  drawBlocks();
  $('sample-list').dataset.key = '';
}

async function loadNavigation(job) {
  if (!job.has_pdf) return;
  const selection = state.selection;
  const data = await api(`/api/jobs/${job.id}/navigation`);
  if (selection !== state.selection) return;
  state.navigation = data;
  state.revision = job.revision;
  drawToc();
}

async function selectJob(identifier) {
  state.selection += 1;
  const selection = state.selection;
  const job = await api(`/api/jobs/${identifier}`);
  if (selection !== state.selection) return;
  state.job = job;
  drawKeywords(job.keyword_candidates);
  state.colors = { ...job.colors };
  state.typography = typography(job.typography);
  state.subheadings = [];
  state.blocks = [];
  state.types = [];
  state.navigation = { toc: [], targets: {} };
  state.page = 1;
  state.zoom = 1;
  state.revision = null;
  state.file = null;
  $('book-title').value = job.book.title;
  $('book-author').value = job.book.author;
  $('dropzone').hidden = true;
  $('selected-file').hidden = false;
  $('file-name').textContent = job.filename;
  $('file-size').textContent = `${job.source_pages} 页`;
  $('remove-file').hidden = true;
  $('form-error').hidden = true;
  $('block-search').value = '';
  $('block-count').textContent = '';
  showView(false);
  showPreview('samples');
  paintJob();
  await Promise.all([loadDetails(job), loadNavigation(job)]);
  showPanel(job.has_blocks && !busy() ? 'colors' : 'progress');
  paintJob();
}

function newBook() {
  state.selection += 1;
  Object.assign(state, { job: null, file: null, blocks: [], types: [], subheadings: [], colors: {}, typography: typography(), navigation: { toc: [], targets: {} }, page: 1, zoom: 1, revision: null });
  $('upload-form').reset();
  drawKeywords(defaultKeywords);
  $('file-input').value = '';
  $('dropzone').hidden = false;
  $('selected-file').hidden = true;
  $('remove-file').hidden = false;
  $('form-error').hidden = true;
  $('upload-meter').hidden = true;
  $('block-count').textContent = '';
  showView(false);
  showPanel('info');
  showPreview('samples');
  paintJob();
  drawBlocks();
  drawToc();
}

function chooseFile(file) {
  if (!file || state.job || state.uploading) return;
  if (!file.name.toLowerCase().endsWith('.pdf')) { toast('请选择 PDF 文件。'); return; }
  if (file.size > 1024 ** 3) { toast('文件不能超过 1 GB。'); return; }
  state.file = file;
  $('dropzone').hidden = true;
  $('selected-file').hidden = false;
  $('file-name').textContent = file.name;
  $('file-size').textContent = file.size < 1024 ** 2 ? `${(file.size / 1024).toFixed(0)} KB` : `${(file.size / 1024 ** 2).toFixed(1)} MB`;
  $('form-error').hidden = true;
}

function uploadFile(form) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open('POST', '/api/jobs');
    request.responseType = 'json';
    request.upload.onprogress = (event) => {
      if (!event.lengthComputable) return;
      const percent = Math.round(event.loaded / event.total * 100);
      $('upload-progress').value = percent;
      $('upload-percent').textContent = `${percent}%`;
    };
    request.onload = () => {
      if (request.status >= 200 && request.status < 300) resolve(request.response);
      else reject(new Error(typeof request.response?.detail === 'string' ? request.response.detail : '上传失败，请检查文件与书籍信息。'));
    };
    request.onerror = () => reject(new Error('连接中断，请检查本地服务是否正在运行。'));
    request.send(form);
  });
}

async function submitBook(event) {
  event.preventDefault();
  if (!state.file) { $('form-error').textContent = '请先选择 PDF 文件。'; $('form-error').hidden = false; return; }
  if (!$('book-title').value.trim() || !$('book-author').value.trim()) { toast('请填写书名和作者。'); return; }
  const form = new FormData();
  form.append('file', state.file);
  form.append('title', $('book-title').value.trim());
  form.append('author', $('book-author').value.trim());
  form.append('keyword_candidates', JSON.stringify(readKeywords()));
  state.uploading = true;
  $('upload-meter').hidden = false;
  $('upload-progress').value = 0;
  $('upload-percent').textContent = '0%';
  $('form-error').hidden = true;
  paintJob();
  try {
    const job = await uploadFile(form);
    state.uploading = false;
    $('upload-meter').hidden = true;
    await selectJob(job.id);
    state.job = await api(`/api/jobs/${job.id}/process`, { method: 'POST' });
    showPanel('progress');
    paintJob();
    await refreshJobs();
  } catch (error) {
    $('form-error').textContent = error.message;
    $('form-error').hidden = false;
    if (state.job) toast(error.message);
  } finally {
    state.uploading = false;
    $('upload-meter').hidden = true;
    paintJob();
  }
}

async function renderBook() {
  if (!state.job || busy()) return;
  if (!validSizes()) return;
  try {
    state.job = await api(`/api/jobs/${state.job.id}/render`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(presentation()) });
    showPanel('progress');
    showPreview('page');
    paintJob();
    await refreshJobs();
  } catch (error) { toast(error.message); }
}

async function renderSamples() {
  if (!state.job || busy()) return;
  if (!validSizes()) return;
  const selection = state.selection;
  try {
    const job = await api(`/api/jobs/${state.job.id}/preview`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(presentation()) });
    if (selection !== state.selection) return;
    state.job = job;
    showPreview('samples');
    paintJob();
    await refreshJobs();
  } catch (error) { toast(error.message); }
}

function thumbnail(job) {
  return job.has_pdf ? `<img class="book-thumb" src="/api/jobs/${job.id}/pages/1.png?v=${job.revision}" alt="" loading="lazy">` : '<span class="book-thumb"><i data-lucide="book-open"></i></span>';
}

function drawJobs() {
  $('job-count').textContent = state.jobs.length;
  $('recent-jobs').innerHTML = state.jobs.length ? state.jobs.slice(0, 3).map((job) => `<button class="recent-job" data-job="${job.id}">${thumbnail(job)}<span class="recent-copy"><strong>${esc(job.book.title)}</strong><span>${esc(job.book.author)} · ${statuses[job.status]}</span></span><i data-lucide="chevron-right"></i></button>`).join('') : '<p class="no-jobs">暂无书稿</p>';
  $('all-jobs').innerHTML = state.jobs.length ? state.jobs.map((job) => `<button class="history-row" data-job="${job.id}"><span class="history-book">${thumbnail(job)}<span><strong>${esc(job.book.title)}</strong><small>${esc(job.book.author)}</small></span></span><span>${job.has_pdf ? job.render_pages : job.source_pages}</span><span>${statuses[job.status]}</span><span class="history-time">${new Date(job.updated).toLocaleDateString('zh-CN', { timeZone: 'Asia/Shanghai', month: '2-digit', day: '2-digit' })} ${new Date(job.updated).toLocaleTimeString('zh-CN', { timeZone: 'Asia/Shanghai', hour: '2-digit', minute: '2-digit' })}</span></button>`).join('') : '<p class="no-jobs">暂无书稿</p>';
  icons();
}

async function refreshJobs() {
  state.jobs = await api('/api/jobs');
  drawJobs();
}

async function poll() {
  try {
    if (state.job && (jobBusy(state.job) || !state.connected)) {
      const identifier = state.job.id;
      const job = await api(`/api/jobs/${identifier}`);
      state.connected = true;
      if (state.job?.id === identifier) {
        const newBlocks = job.has_blocks && (!state.job.has_blocks || job.block_revision !== state.job.block_revision || (job.status === 'ready' && (state.job.status === 'extracting' || job.content_method !== state.job.content_method)));
        const previousStatus = state.job.status;
        const previousPreview = state.job.preview_status;
        state.job = job;
        if (newBlocks) { await loadDetails(job); showPanel('colors'); }
        if (job.has_pdf && job.revision !== state.revision) { await loadNavigation(job); updatePage(); }
        if (job.status === 'complete' && previousStatus !== 'complete') { showPanel('colors'); toast('整书 PDF 已生成。'); }
        if (job.preview_status === 'complete' && previousPreview !== 'complete') toast('特殊块样例预览已生成。');
        paintJob();
      }
    }
    if (state.jobs.some(jobBusy)) { await refreshJobs(); state.connected = true; }
  } catch (error) {
    const wasConnected = state.connected;
    state.connected = false;
    paintJob();
    if (wasConnected) toast('本地服务连接中断，正在等待重新连接。');
  }
  finally { setTimeout(poll, 2000); }
}

document.querySelectorAll('[data-panel]').forEach((button) => button.addEventListener('click', () => showPanel(button.dataset.panel)));
document.querySelectorAll('[data-preview]').forEach((button) => button.addEventListener('click', () => showPreview(button.dataset.preview)));
$('workspace-nav').addEventListener('click', () => showView(false));
$('history-nav').addEventListener('click', () => { showView(true); refreshJobs().catch((e) => toast(e.message)); });
$('view-all').addEventListener('click', () => showView(true));
$('new-book').addEventListener('click', newBook);
$('dropzone').addEventListener('click', () => $('file-input').click());
$('file-input').addEventListener('change', (event) => chooseFile(event.target.files[0]));
for (const name of ['dragenter', 'dragover']) $('dropzone').addEventListener(name, (event) => { event.preventDefault(); $('dropzone').classList.add('dragover'); });
for (const name of ['dragleave', 'drop']) $('dropzone').addEventListener(name, (event) => { event.preventDefault(); $('dropzone').classList.remove('dragover'); });
$('dropzone').addEventListener('drop', (event) => chooseFile(event.dataTransfer.files[0]));
$('remove-file').addEventListener('click', () => { state.file = null; $('file-input').value = ''; $('selected-file').hidden = true; $('dropzone').hidden = false; });
$('upload-form').addEventListener('submit', submitBook);
$('reset-keywords').addEventListener('click', () => { drawKeywords(defaultKeywords); $('keyword-status').textContent = '已恢复默认候选，保存后生效。'; });
$('save-keywords').addEventListener('click', () => saveKeywords().catch((error) => { $('keyword-error').textContent = error.message; $('keyword-error').hidden = false; }));
$('apply-keywords').addEventListener('click', applyKeywords);
for (const id of Object.values(keywordFields)) $(id).addEventListener('input', () => { $('keyword-error').hidden = true; $('keyword-status').textContent = '候选集合已修改。'; });
$('go-colors').addEventListener('click', () => showPanel('colors'));
$('render-button').addEventListener('click', renderSamples);
$('render-book-button').addEventListener('click', renderBook);
$('reset-colors').addEventListener('click', () => { state.colors = {}; state.typography = typography(); drawColors(); drawBlocks(); });
$('refresh-blocks-button').addEventListener('click', applyKeywords);
$('body-size').addEventListener('input', (event) => {
  if (!event.target.checkValidity() || busy()) return;
  const body = Number(event.target.value);
  const old = state.typography;
  state.typography = typography({ body_size: body });
  for (const [pattern, size] of Object.entries(old.subheading_sizes)) {
    state.typography.subheading_sizes[pattern] = Math.max(body + .5, size);
  }
  drawTypography(); updateDirty();
});
$('subheading-sizes').addEventListener('input', (event) => {
  if (event.target.dataset.headingPattern && event.target.checkValidity() && !busy()) {
    state.typography.subheading_sizes[event.target.dataset.headingPattern] = Number(event.target.value);
    updateDirty();
  }
});

function validSizes() {
  for (const control of document.querySelectorAll('#typography-settings input')) {
    if (!control.reportValidity()) return false;
  }
  return true;
}
$('retry-job').addEventListener('click', async () => {
  if (state.job?.has_blocks) return renderBook();
  try { state.job = await api(`/api/jobs/${state.job.id}/process`, { method: 'POST' }); paintJob(); await refreshJobs(); }
  catch (error) { toast(error.message); }
});
$('color-list').addEventListener('click', (event) => {
  const trigger = event.target.closest('[data-color-kind]');
  if (trigger) { drawColors(trigger.getAttribute('aria-expanded') === 'true' ? null : trigger.dataset.colorKind); return; }
  const choice = event.target.closest('[data-color]');
  if (choice && !busy()) { state.colors[choice.dataset.kind] = choice.dataset.color; drawColors(choice.dataset.kind); drawBlocks(); }
});
$('color-list').addEventListener('input', (event) => {
  if (event.target.dataset.customKind && !busy()) {
    const kind = event.target.dataset.customKind;
    state.colors[kind] = event.target.value.toUpperCase();
    updateDirty();
  }
});
$('color-list').addEventListener('change', (event) => {
  if (event.target.dataset.customKind && !busy()) { drawColors(event.target.dataset.customKind); drawBlocks(); }
});
for (const id of ['recent-jobs', 'all-jobs']) $(id).addEventListener('click', (event) => {
  const button = event.target.closest('[data-job]');
  if (button) selectJob(button.dataset.job).catch((error) => toast(error.message));
});
$('block-search').addEventListener('input', drawBlocks);
$('block-filter').addEventListener('change', drawBlocks);
$('colors-panel').addEventListener('click', async (event) => {
  const remove = event.target.closest('[data-remove-type]');
  if (remove) {
    if (busy() || !state.job) return;
    const identifier = state.job.id, selection = state.selection;
    state.removingBlock = true;
    paintJob();
    try {
      const selector = { kind: remove.dataset.removeType };
      if (remove.dataset.removePattern) selector.pattern = remove.dataset.removePattern;
      const updated = await api(`/api/jobs/${identifier}/block-types`, { method: 'DELETE', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(selector) });
      if (selection !== state.selection) return;
      state.job = updated;
      await loadDetails(updated);
      state.colors = Object.fromEntries(Object.entries(state.colors).filter(([kind]) => state.types.some((t) => t.kind === kind)));
      drawColors();
      drawBlocks();
      toast('已删除此类型的全部特殊块身份，原文保留。重新生成预览或整本 PDF 后生效。');
      await refreshJobs();
    } catch (error) { toast(error.message); }
    finally { state.removingBlock = false; paintJob(); }
    return;
  }
});
$('block-list').addEventListener('click', (event) => {
  const button = event.target.closest('[data-target]');
  if (!button) return;
  const page = state.navigation.targets[button.dataset.target];
  if (page) gotoPage(page); else toast('生成书稿后可查看这个特殊块所在的页面。');
});
$('toc-list').addEventListener('click', (event) => { const button = event.target.closest('[data-page]'); if (button) gotoPage(button.dataset.page); });
$('previous-page').addEventListener('click', () => gotoPage(state.page - 1));
$('next-page').addEventListener('click', () => gotoPage(state.page + 1));
$('page-number').addEventListener('change', (event) => gotoPage(event.target.value));
$('page-number').addEventListener('keydown', (event) => { if (event.key === 'Enter') { event.preventDefault(); gotoPage(event.target.value); } });
$('zoom-in').addEventListener('click', () => { state.zoom = Math.min(2, state.zoom + .25); updateZoom(); });
$('zoom-out').addEventListener('click', () => { state.zoom = Math.max(.5, state.zoom - .25); updateZoom(); });
$('fit-page').addEventListener('click', () => { state.zoom = 1; updateZoom(); });
$('pdf-page').addEventListener('error', () => { if (state.job?.has_pdf) { $('image-error').hidden = false; $('pdf-page').hidden = true; } });
$('reload-page').addEventListener('click', () => { $('pdf-page').removeAttribute('src'); updatePage(); });
$('sample-list').addEventListener('error', (event) => {
  if (event.target.tagName === 'IMG') { event.target.hidden = true; event.target.nextElementSibling.hidden = false; }
}, true);
window.addEventListener('resize', updateZoom);

async function init() {
  paintJob();
  drawBlocks();
  drawToc();
  try {
    const [environment, candidates] = await Promise.all([api('/api/environment'), api('/api/keyword-candidates'), refreshJobs()]);
    defaultKeywords = candidates;
    if (!state.job) drawKeywords(candidates);
    paintJob();
    const ready = environment.mineru && environment.tex && environment.api_key;
    $('env-label').textContent = ready ? '运行环境就绪' : '运行环境需配置';
    $('env-dot').style.background = ready ? '#24865B' : '#D17B27';
    $('env-details').textContent = ready ? 'MinerU · DeepSeek · XeLaTeX' : [!environment.mineru && 'MinerU 环境', !environment.tex && '排版工具', !environment.api_key && 'API 密钥'].filter(Boolean).join(' · ');
  } catch (error) { $('env-label').textContent = '服务连接失败'; toast(error.message); }
  poll();
}
init();
