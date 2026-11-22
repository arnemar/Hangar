export const state = {
  sessions: [],
  session: null,
  messages: [],
  changes: [],
  projects: [],
  models: [],
  defaultModel: '',
  draftModel: '',
  attachments: [],
  proposeEdits: false,
  searchWeb: false,
  run: null,
  switching: true,
  refreshingModels: false,
};

export function currentModel() {
  return state.session?.model || state.draftModel || state.defaultModel;
}

export function locked() {
  return Boolean(state.run || state.switching);
}

export function el(id) {
  return document.getElementById(id);
}

export function node(tag, { text, className } = {}) {
  const value = document.createElement(tag);
  if (text !== undefined) value.textContent = text;
  if (className) value.className = className;
  return value;
}

export function button(text, action, className = 'quiet') {
  const value = node('button', { text, className });
  value.type = 'button';
  value.addEventListener('click', action);
  return value;
}

export function notice(text = '', error = false, target = el('notice')) {
  target.textContent = text;
  target.hidden = !text;
  target.classList.toggle('error', error);
}
