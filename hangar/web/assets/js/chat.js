import { chatEvents, sessionUrl } from './api.js';
import { button, el, notice, state } from './state.js';
import { renderTranscript, updateStream } from './transcript.js';

export function renderComposer() {
  const busy = !!state.run || state.switching;
  const model = state.session?.model || state.draftModel || state.defaultModel;
  const project = state.session?.project_name;
  const edit = el('suggest-edits');
  const search = el('search-web');
  edit.hidden = !project;
  edit.disabled = busy;
  search.disabled = busy;
  edit.setAttribute('aria-pressed', String(state.proposeEdits));
  search.setAttribute('aria-pressed', String(state.searchWeb));
  el('message').disabled = busy;
  el('send').hidden = !!state.run;
  el('send').disabled = busy || !state.models.includes(model)
    || !(el('message').value.trim() || state.attachments.length);
  el('stop').hidden = !state.run;
  const files = el('attachments');
  files.hidden = !state.attachments.length;
  files.replaceChildren(...state.attachments.map(path => {
    const remove = button(path + ' ×', () => {
      state.attachments = state.attachments.filter(file => file !== path);
      renderComposer();
    }, 'chip');
    remove.setAttribute('aria-label', 'Remove ' + path);
    remove.disabled = busy;
    return remove;
  }));
  el('access-note').textContent = state.searchWeb
    ? 'Search sends the typed question to DuckDuckGo.'
    : project ? 'Project files are read-only. File changes need your approval.'
      : 'Enter to send · Shift+Enter for a new line';
}

export function clearDraft() {
  el('message').value = '';
  state.attachments = [];
  state.proposeEdits = false;
  state.searchWeb = false;
  renderComposer();
}

export function attachFile(path) {
  if (state.run || state.switching || state.attachments.includes(path)) return;
  if (state.attachments.length >= 3) {
    notice('Attach up to three file excerpts per message.', true);
    return;
  }
  state.attachments.push(path);
  renderComposer();
  el('message').focus();
}

export function bindChat({ ensureSession, settle, renderChrome, onProposal }) {
  el('composer').addEventListener('submit', async event => {
    event.preventDefault();
    if (el('send').disabled) return;
    const draft = el('message').value;
    const run = {
      controller: new AbortController(),
      message: draft.trim(), attachments: [...state.attachments],
      answer: '', activity: 'Starting', started: false, done: false,
    };
    const options = { propose_edits: state.proposeEdits, search_web: state.searchWeb };
    if (options.search_web && !run.message) {
      notice('Enter a question to search the web.', true);
      return;
    }
    state.run = run;
    notice();
    renderChrome();
    renderTranscript({ follow: true });
    try {
      const session = await ensureSession();
      run.sessionId = session.id;
      for await (const frame of chatEvents(sessionUrl(session.id, '/chat'), {
        message: run.message,
        attachments: run.attachments,
        ...options,
      }, run.controller.signal)) {
        if (frame.type === 'started') {
          run.started = true;
          clearDraft();
        } else if (frame.type === 'chunk') {
          run.answer += frame.content;
          updateStream();
        } else if (frame.type === 'reset') {
          run.answer = '';
          renderTranscript();
        } else if (frame.type === 'proposal') {
          onProposal(frame.change);
        } else if (frame.type === 'activity') {
          run.activity = frame.content;
          if (!run.answer) renderTranscript();
        } else if (frame.type === 'notice') {
          notice(frame.content);
        } else if (frame.type === 'error') {
          throw new Error(frame.content);
        } else if (frame.type === 'done') {
          run.done = true;
          if (frame.reason === 'length') notice('Ollama reached its generation limit.');
        }
      }
      if (!run.done) throw new Error('The response ended before it was complete.');
    } catch (error) {
      if (!run.started) {
        el('message').value = draft;
        state.attachments = run.attachments;
      }
      if (error.name === 'AbortError') notice('Response stopped.');
      else notice(error.message, true);
    } finally {
      // Keep the controls locked until the authoritative history is loaded.
      try {
        if (run.sessionId) await settle(run.sessionId);
      } catch (error) {
        notice(error.message, true);
      } finally {
        state.run = null;
        renderChrome();
        renderTranscript({ follow: true });
        el('message').focus();
      }
    }
  });
  el('message').addEventListener('input', renderComposer);
  el('message').addEventListener('keydown', event => {
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      if (!el('send').disabled) el('composer').requestSubmit();
    }
  });
  el('stop').addEventListener('click', () => state.run?.controller.abort());
  el('search-web').addEventListener('click', () => {
    state.searchWeb = !state.searchWeb;
    renderComposer();
  });
  el('suggest-edits').addEventListener('click', () => {
    state.proposeEdits = !state.proposeEdits;
    renderComposer();
  });
}
