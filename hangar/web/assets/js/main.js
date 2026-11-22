import { api } from './api.js';
import { bindChat, renderComposer } from './chat.js';
import { renderChanges } from './changes.js';
import { refreshModels, renderModels } from './models.js';
import { bindProjects, openProject } from './projects.js';
import { ensureSession, loadSession, sessionActions } from './sessions.js';
import { bindSettings } from './settings.js';
import { button, el, locked, node, notice, state } from './state.js';
import { renderTranscript } from './transcript.js';

const sessions = sessionActions(render, closeSidebar);
const updateModels = () => refreshModels(render);

function closeSidebar() {
  el('sidebar').classList.remove('open');
  el('open-sidebar').setAttribute('aria-expanded', 'false');
}

function renderHistory(busy) {
  const rows = state.sessions.map(chat => {
    const active = chat.id === state.session?.id;
    const row = node('div', { className: active ? 'chat-item active' : 'chat-item' });
    const open = button(chat.title || 'New chat', () => sessions.open(chat.id));
    if (active) open.setAttribute('aria-current', 'true');
    const remove = button('×', () => sessions.remove(chat), 'icon');
    remove.setAttribute('aria-label', 'Delete chat ' + (chat.title || 'New chat'));
    open.disabled = remove.disabled = busy;
    row.append(open, remove);
    return row;
  });
  el('chat-list').replaceChildren(...rows);
}

function render() {
  const busy = locked();
  const project = state.session?.project_name;
  renderHistory(busy);
  for (const id of ['new-chat', 'open-settings', 'open-projects', 'intro-project']) {
    el(id).disabled = busy;
  }
  el('open-projects').textContent = project || 'Attach project';
  el('open-projects').title = project ? 'Change attached project' : 'Attach project';
  el('toggle-project').hidden = !project;
  el('chat-title').textContent = state.session?.title || 'New chat';
  renderModels();
  renderComposer();
  renderChanges();
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
el('new-chat').addEventListener('click', sessions.newChat);
el('model-select').addEventListener('change', event => sessions.model(event.target.value));
el('refresh-models').addEventListener('click', async () => {
  const error = await updateModels();
  notice(error || '', !!error);
});

bindChat({
  ensureSession, settle: loadSession, renderChrome: render,
  onProposal(change) {
    if (!state.changes.some(item => item.id === change.id)) state.changes.push(change);
    openProject('changes');
  },
});
bindProjects({ setProject: sessions.project });
bindSettings(async () => {
  const error = await updateModels();
  notice(error || '', !!error);
});

async function start() {
  render();
  const modelCheck = updateModels();
  const results = await Promise.allSettled([
    api('/api/projects').then(result => { state.projects = result.projects; }),
    api('/api/sessions').then(async result => {
      state.sessions = result.sessions;
      if (state.sessions.length) await loadSession(state.sessions[0].id);
    }),
  ]);
  state.switching = false;
  render();
  renderTranscript({ follow: true });
  const failed = results.find(result => result.status === 'rejected');
  if (failed) notice(failed.reason.message, true);
  const connectionError = await modelCheck;
  if (connectionError && !failed && !locked() && el('notice').hidden) notice(connectionError, true);
}

void start();
