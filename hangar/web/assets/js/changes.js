import { api, sessionUrl } from './api.js';
import { button, el, locked, node, notice, state } from './state.js';

async function decide(change, decision, card, errorTarget) {
  if (locked()) return;
  const id = state.session.id;
  const actions = [...card.querySelectorAll('button:not(:disabled)')];
  actions.forEach(action => { action.disabled = true; });
  try {
    const updated = await api(sessionUrl(id, '/changes/' + encodeURIComponent(change.id)), {
      method: 'POST', data: { decision },
    });
    if (state.session?.id !== id) return;
    state.changes = state.changes.map(item => item.id === updated.id ? updated : item);
    if (updated.status === 'applied') document.dispatchEvent(new Event('project-file-changed'));
    renderChanges();
  } catch (error) {
    notice(error.message, true, errorTarget);
    actions.forEach(action => { action.disabled = locked(); });
  }
}

function diffView(diff) {
  const pre = node('pre', { className: 'diff' });
  for (const line of diff.split('\n')) {
    const className = line.startsWith('+') ? 'added' : line.startsWith('-') ? 'removed' : '';
    pre.append(node('span', { text: line + '\n', className }));
  }
  return pre;
}

function changeCard(change, opened) {
  const card = node('details', { className: 'change-card' });
  card.dataset.id = change.id;
  card.open = opened ?? change.status === 'pending';
  const summary = node('summary', { text: change.path });
  const status = { pending: 'Review required', applied: 'Applied', dismissed: 'Dismissed' }[change.status];
  summary.append(node('span', { text: status, className: 'change-badge' }));
  card.append(summary, node('p', { text: change.project_name, className: 'change-state' }), diffView(change.diff));
  if (change.status !== 'pending') return card;

  const error = node('p', { className: 'notice error' });
  error.hidden = true;
  error.setAttribute('role', 'alert');
  const actions = node('div', { className: 'preview-actions' });
  const dismiss = button('Dismiss', () => decide(change, 'dismiss', card, error), 'secondary');
  const apply = button('Apply change', () => decide(change, 'apply', card, error), 'primary');
  dismiss.disabled = locked();
  apply.disabled = locked() || change.project_name !== state.session?.project_name;
  actions.append(dismiss, apply);
  card.append(actions, error);
  return card;
}

export function renderChanges() {
  const pending = state.changes.filter(change => change.status === 'pending').length;
  el('changes-tab').textContent = pending ? 'Changes (' + pending + ')' : 'Changes';
  const list = el('change-list');
  if (list.hidden || el('project-panel').hidden) return;
  const opened = new Map([...list.querySelectorAll('details')].map(row => [row.dataset.id, row.open]));
  const cards = [...state.changes].reverse().map(change => changeCard(change, opened.get(change.id)));
  list.replaceChildren(...cards);
  if (!cards.length) list.append(node('p', {
    text: 'No file changes have been proposed in this chat.', className: 'empty-panel',
  }));
}
