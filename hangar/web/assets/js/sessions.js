import { api, sessionUrl } from './api.js';
import { clearDraft } from './chat.js';
import { closeProject } from './projects.js';
import { currentModel, el, locked, notice, state } from './state.js';
import { renderTranscript } from './transcript.js';

function remember(session) {
  state.sessions = state.sessions.some(chat => chat.id === session.id)
    ? state.sessions.map(chat => chat.id === session.id ? session : chat)
    : [session, ...state.sessions];
  state.session = session;
  return session;
}

export async function ensureSession() {
  const model = currentModel();
  if (!state.session) return remember(await api('/api/sessions', { method: 'POST', data: { model } }));
  if (!state.session.model && model) await patchSession({ model });
  return state.session;
}

async function patchSession(fields) {
  return remember(await api(sessionUrl(state.session.id), { method: 'PATCH', data: fields }));
}

export async function loadSession(id) {
  const [transcript, changes, history] = await Promise.all([
    api(sessionUrl(id, '/messages')), api(sessionUrl(id, '/changes')), api('/api/sessions'),
  ]);
  const session = history.sessions.find(chat => chat.id === id);
  if (!session) throw new Error('This chat is no longer available.');
  state.sessions = history.sessions;
  state.session = session;
  state.messages = transcript.messages;
  state.changes = changes.changes;
}

export function sessionActions(render, closeSidebar) {
  async function withSessionLock(action) {
    if (locked()) return;
    state.switching = true;
    render();
    try {
      await action();
    } finally {
      state.switching = false;
      render();
    }
  }

  function showConversation() {
    clearDraft();
    closeProject();
    closeSidebar();
    notice();
    renderTranscript({ follow: true });
  }

  function showNew() {
    state.session = null;
    state.messages = [];
    state.changes = [];
    showConversation();
  }

  return {
    open: id => withSessionLock(async () => {
      await loadSession(id);
      showConversation();
    }).catch(error => notice(error.message, true)),

    newChat() {
      if (locked()) return;
      showNew();
      render();
      el('message').focus();
    },

    remove(chat) {
      if (locked() || !window.confirm('Delete this chat and all of its messages?')) return;
      return withSessionLock(async () => {
        await api(sessionUrl(chat.id), { method: 'DELETE' });
        state.sessions = state.sessions.filter(item => item.id !== chat.id);
        if (state.session?.id === chat.id) showNew();
        else notice();
      }).catch(error => notice(error.message, true));
    },

    async model(model) {
      if (locked() || !state.models.includes(model)) {
        render();
        return;
      }
      if (!state.session) {
        state.draftModel = model;
        render();
        return;
      }
      await withSessionLock(async () => {
        await patchSession({ model });
        notice();
      }).catch(error => notice(error.message, true));
    },

    project: project_name => withSessionLock(async () => {
      if (state.session) await patchSession({ project_name });
      else remember(await api('/api/sessions', {
        method: 'POST', data: { model: currentModel(), project_name },
      }));
      state.attachments = [];
      state.proposeEdits = false;
      notice();
    }),
  };
}
