import { api, sessionUrl } from './api.js';
import { bindChat, clearDraft, renderComposer } from './chat.js';
import { bindProjects, closeProject, openProject, renderChanges } from './projects.js';
import { bindSettings } from './settings.js';
import { button, el, node, notice, state } from './state.js';
import { renderTranscript } from './transcript.js';

function renderChrome() {
  const locked = !!state.run || state.switching;
  const session = state.session;
  const project = session?.project_name;
  const history = state.sessions.map(chat => {
    const row = node('div', { className: 'chat-item' + (chat.id === session?.id ? ' active' : '') });
    const open = button(chat.title || 'New chat', () => selectSession(chat.id));
    if (chat.id === session?.id) open.setAttribute('aria-current', 'true');
    const remove = button('×', () => deleteSession(chat), 'icon');
    remove.setAttribute('aria-label', 'Delete chat ' + (chat.title || 'New chat'));
    open.disabled = locked;
    remove.disabled = locked;
    row.append(open, remove);
    return row;
  });
  el('chat-list').replaceChildren(...history);
  el('new-chat').disabled = locked;
  el('open-settings').disabled = locked;
  el('open-projects').disabled = locked;
  el('intro-project').disabled = locked;
  el('open-projects').textContent = project || 'Attach project';
  el('open-projects').title = project ? 'Change attached project' : 'Attach project';
  el('toggle-project').hidden = !project;
  el('chat-title').textContent = session?.title || 'New chat';

  const select = el('model-select');
  const model = session?.model || state.draftModel || state.defaultModel;
  select.replaceChildren(new Option(state.models.length ? 'Choose a model' : 'No models', ''));
  for (const available of state.models) select.add(new Option(available, available));
  if (model && !state.models.includes(model)) {
    select.add(new Option(model + ' (unavailable)', model));
  }
  select.value = model;
  select.disabled = locked || !state.models.length;
  renderComposer();
  renderChanges();
}

async function refreshModels() {
  const result = await api('/api/models');
  state.models = result.models;
  state.defaultModel = result.default;
  if (!state.models.includes(state.draftModel)) state.draftModel = '';
  const connection = el('connection');
  connection.dataset.state = state.models.length ? 'ready' : 'offline';
  connection.textContent = state.models.length
    ? 'Ollama connected'
    : result.connected ? 'No models installed' : 'Ollama unreachable';
  renderChrome();
}

function remember(session) {
  state.session = session;
  if (!state.sessions.some(chat => chat.id === session.id)) state.sessions.unshift(session);
  else state.sessions = state.sessions.map(chat => chat.id === session.id ? session : chat);
  return session;
}

async function ensureSession() {
  const model = state.session?.model || state.draftModel || state.defaultModel;
  if (!state.session) return remember(await api('/api/sessions', { method: 'POST', data: { model } }));
  if (!state.session.model && model) {
    return remember(await api(sessionUrl(state.session.id), { method: 'PATCH', data: { model } }));
  }
  return state.session;
}

async function loadCurrent(id) {
  const [transcript, changes, history] = await Promise.all([
    api(sessionUrl(id, '/messages')), api(sessionUrl(id, '/changes')), api('/api/sessions'),
  ]);
  state.sessions = history.sessions;
  state.session = state.sessions.find(chat => chat.id === id) || null;
  state.messages = transcript.messages;
  state.changes = changes.changes;
}

async function selectSession(id) {
  if (state.run || state.switching) return;
  state.switching = true;
  renderChrome();
  try {
    await loadCurrent(id);
    clearDraft();
    closeProject();
    closeSidebar();
    notice();
    renderTranscript({ follow: true });
  } catch (error) {
    notice(error.message, true);
  } finally {
    state.switching = false;
    renderChrome();
  }
}

function newChat() {
  state.session = null;
  state.messages = [];
  state.changes = [];
  clearDraft();
  closeProject();
  closeSidebar();
  notice();
  renderChrome();
  renderTranscript();
  el('message').focus();
}

async function deleteSession(chat) {
  if (state.run || state.switching || !window.confirm('Delete this chat and all of its messages?')) return;
  try {
    await api(sessionUrl(chat.id), { method: 'DELETE' });
    state.sessions = state.sessions.filter(session => session.id !== chat.id);
    if (state.session?.id === chat.id) newChat();
    else renderChrome();
  } catch (error) {
    notice(error.message, true);
  }
}

async function setProject(project_name) {
  if (state.run || state.switching) return;
  state.switching = true;
  renderChrome();
  try {
    const chat = await ensureSession();
    remember(await api(sessionUrl(chat.id), { method: 'PATCH', data: { project_name } }));
    state.attachments = [];
    state.proposeEdits = false;
    notice();
  } finally {
    state.switching = false;
    renderChrome();
  }
}

function closeSidebar() {
  el('sidebar').classList.remove('open');
  el('open-sidebar').setAttribute('aria-expanded', 'false');
}
el('open-sidebar').addEventListener('click', () => {
  const opened = el('sidebar').classList.toggle('open');
  el('open-sidebar').setAttribute('aria-expanded', String(opened));
});
document.addEventListener('pointerdown', event => {
  if (!event.target.closest('#sidebar, #open-sidebar')) closeSidebar();
});
document.querySelectorAll('[data-close]').forEach(close => {
  close.addEventListener('click', () => el(close.dataset.close).close());
});
el('new-chat').addEventListener('click', newChat);
el('model-select').addEventListener('change', async event => {
  const model = event.target.value;
  if (!state.models.includes(model)) {
    renderChrome();
    return;
  }
  if (!state.session) {
    state.draftModel = model;
    renderChrome();
    return;
  }
  state.switching = true;
  renderChrome();
  try {
    remember(await api(sessionUrl(state.session.id), { method: 'PATCH', data: { model } }));
    notice();
  } catch (error) {
    notice(error.message, true);
  } finally {
    state.switching = false;
    renderChrome();
  }
});
bindChat({
  ensureSession, settle: loadCurrent, renderChrome,
  onProposal(change) {
    if (!state.changes.some(item => item.id === change.id)) state.changes.push(change);
    openProject('changes');
  },
});
bindProjects({ setProject, renderChrome });
bindSettings(refreshModels);
renderChrome();
void Promise.allSettled([
  refreshModels(),
  api('/api/projects').then(result => { state.projects = result.projects; }),
  api('/api/sessions').then(result => {
    state.sessions = result.sessions;
    if (state.sessions[0] && !state.session) return selectSession(state.sessions[0].id);
    renderChrome();
  }),
]).then(results => {
  const failed = results.find(result => result.status === 'rejected');
  if (failed) notice(failed.reason.message, true);
});
