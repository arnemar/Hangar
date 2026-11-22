import { api, sessionUrl } from './api.js';
import { attachFile, renderComposer } from './chat.js';
import { button, el, node, notice, state } from './state.js';

const browser = () => el('file-browser');
const panel = () => el('project-panel');
let tab = 'files';
let fileRequest = 0;

function projectUrl(suffix, path = '') {
  return '/api/projects/' + encodeURIComponent(state.session.project_name) + suffix
    + '?' + new URLSearchParams({ path });
}

function fileStatus(text, error = false) {
  browser().replaceChildren(node('p', { text, className: error ? 'notice error' : 'empty-panel' }));
}

function parent(path) {
  return path.split('/').slice(0, -1).join('/');
}

async function browse(path = '') {
  const serial = ++fileRequest;
  fileStatus('Loading files…');
  try {
    const result = await api(projectUrl('/files', path));
    if (serial !== fileRequest) return;
    browser().replaceChildren();
    if (path) browser().append(button('← Parent folder', () => browse(parent(path))));
    browser().append(node('p', { text: path || 'Project root', className: 'folder-path' }));
    for (const file of result.entries) {
      const row = button('', () => file.directory ? browse(file.path) : preview(file.path), 'file-row');
      const icon = node('span', { text: file.directory ? '▸' : '·' });
      icon.setAttribute('aria-hidden', 'true');
      row.append(icon, node('span', { text: file.name }));
      browser().append(row);
    }
    if (!result.entries.length) browser().append(node('p', { text: 'No files in this folder.', className: 'empty-panel' }));
    if (result.truncated) browser().append(node('p', { text: 'Showing the first 300 files and folders.', className: 'empty-panel' }));
  } catch (error) {
    if (serial === fileRequest) fileStatus(error.message, true);
  }
}

async function preview(path) {
  const serial = ++fileRequest;
  fileStatus('Opening file…');
  try {
    const result = await api(projectUrl('/file', path));
    if (serial !== fileRequest) return;
    const actions = node('div', { className: 'preview-actions' });
    const attach = button('Attach excerpt', () => {
      attachFile(path);
      if (state.attachments.includes(path)) {
        attach.textContent = 'Attached';
        if (matchMedia('(max-width: 760px)').matches) closeProject();
      }
    }, 'secondary');
    actions.append(button('← Files', () => browse(parent(path))), attach);
    browser().replaceChildren(
      actions,
      node('p', { text: path, className: 'folder-path' }),
      node('pre', { text: result.content, className: 'file-text' }),
    );
  } catch (error) {
    if (serial !== fileRequest) return;
    fileStatus(error.message, true);
    browser().append(button('← Files', () => browse(parent(path))));
  }
}

export function closeProject() {
  ++fileRequest;
  panel().hidden = true;
  el('toggle-project').setAttribute('aria-expanded', 'false');
}

function selectTab(next) {
  tab = next;
  el('files-tab').setAttribute('aria-pressed', String(tab === 'files'));
  el('changes-tab').setAttribute('aria-pressed', String(tab === 'changes'));
  browser().hidden = tab !== 'files';
  el('change-list').hidden = tab !== 'changes';
  if (tab === 'changes') renderChanges();
}

export function openProject(show = 'files') {
  if (!state.session?.project_name) return;
  panel().hidden = false;
  el('toggle-project').setAttribute('aria-expanded', 'true');
  const project = state.projects.find(project => project.name === state.session.project_name);
  el('project-title').textContent = state.session.project_name;
  el('project-path').textContent = project?.root || '';
  el('project-path').title = project?.root || '';
  selectTab(show);
  if (show === 'files') void browse();
}

async function decide(change, decision, errorTarget) {
  if (state.run || state.switching) return;
  const id = state.session.id;
  const container = errorTarget.parentElement;
  const actions = [...container.querySelectorAll('button')];
  actions.forEach(action => { action.disabled = true; });
  try {
    const changed = await api(sessionUrl(id, '/changes/' + encodeURIComponent(change.id)), {
      method: 'POST', data: { decision },
    });
    if (state.session?.id !== id) return;
    state.changes = state.changes.map(item => item.id === changed.id ? changed : item);
    if (changed.status === 'applied') {
      ++fileRequest;
      browser().replaceChildren();
    }
    renderChanges();
  } catch (error) {
    if (state.session?.id === id) notice(error.message, true, errorTarget);
  } finally {
    actions.forEach(action => { action.disabled = false; });
  }
}

function diffView(diff) {
  const pre = node('pre', { className: 'diff' });
  for (const line of diff.split('\n')) {
    const className = line[0] === '+' ? 'added' : line[0] === '-' ? 'removed' : '';
    pre.append(node('span', { text: line, className }), document.createTextNode('\n'));
  }
  return pre;
}

export function renderChanges() {
  const list = el('change-list');
  const pending = state.changes.filter(change => change.status === 'pending').length;
  el('changes-tab').textContent = pending ? 'Changes (' + pending + ')' : 'Changes';
  if (list.hidden || panel().hidden) return;
  const opened = new Set([...list.querySelectorAll('details[open]')].map(row => row.dataset.id));
  list.replaceChildren();
  if (!state.changes.length) {
    list.append(node('p', { text: 'No file changes have been proposed in this chat.', className: 'empty-panel' }));
    return;
  }
  for (const change of [...state.changes].reverse()) {
    const card = node('details', { className: 'change-card' });
    card.dataset.id = change.id;
    card.open = opened.has(change.id) || change.status === 'pending';
    const summary = node('summary', { text: change.path });
    const status = { pending: 'Review required', applied: 'Applied', dismissed: 'Dismissed' }[change.status];
    summary.append(node('span', { text: status, className: 'change-badge' }));
    card.append(summary, node('p', { text: change.project_name, className: 'change-state' }));
    card.append(diffView(change.diff));
    if (change.status === 'pending') {
      const error = node('p', { className: 'notice error' });
      error.hidden = true;
      error.setAttribute('role', 'alert');
      const actions = node('div', { className: 'preview-actions' });
      const dismiss = button('Dismiss', () => decide(change, 'dismiss', error), 'secondary');
      const apply = button('Apply change', () => decide(change, 'apply', error), 'primary');
      apply.disabled = !!state.run || change.project_name !== state.session?.project_name;
      dismiss.disabled = !!state.run;
      actions.append(dismiss, apply);
      card.append(actions, error);
    }
    list.append(card);
  }
}

export function bindProjects({ setProject, renderChrome }) {
  const dialog = el('projects-dialog');
  const selection = el('registered-project');
  const errorTarget = el('project-error');
  function openDialog() {
    if (state.run || state.switching) return;
    notice('', false, errorTarget);
    selection.replaceChildren(new Option('Choose a project', ''));
    for (const project of state.projects) {
      const option = new Option(project.local ? project.name : project.name + ' (SSH)', project.name);
      option.disabled = !project.local;
      selection.add(option);
    }
    el('detach-project').hidden = !state.session?.project_name;
    el('attach-project').disabled = true;
    dialog.showModal();
  }
  async function attach(value) {
    try {
      await setProject(value);
      dialog.close();
      if (value) openProject();
      else closeProject();
      return true;
    } catch (error) {
      notice(error.message, true, errorTarget);
      return false;
    }
  }
  el('open-projects').addEventListener('click', openDialog);
  el('intro-project').addEventListener('click', openDialog);
  el('toggle-project').addEventListener('click', () => panel().hidden ? openProject() : closeProject());
  el('close-project').addEventListener('click', closeProject);
  el('files-tab').addEventListener('click', () => {
    selectTab('files');
    if (!browser().hasChildNodes()) void browse();
  });
  el('changes-tab').addEventListener('click', () => selectTab('changes'));
  el('detach-project').addEventListener('click', () => attach(null));
  selection.addEventListener('change', () => { el('attach-project').disabled = !selection.value; });
  el('attach-project-form').addEventListener('submit', async event => {
    event.preventDefault();
    if (!selection.value) return;
    const submit = el('attach-project');
    submit.disabled = true;
    await attach(selection.value);
    submit.disabled = !selection.value;
  });
  el('register-project-form').addEventListener('submit', async event => {
    event.preventDefault();
    const submit = event.submitter;
    submit.disabled = true;
    try {
      const record = await api('/api/projects', {
        method: 'POST', data: { name: el('register-name').value, root: el('register-root').value },
      });
      state.projects.push(record);
      const option = new Option(record.name, record.name);
      selection.add(option);
      if (await attach(record.name)) event.target.reset();
      renderChrome();
    } catch (error) {
      notice(error.message, true, errorTarget);
    } finally {
      submit.disabled = false;
    }
  });
}
