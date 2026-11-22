import { api } from './api.js';
import { currentModel, el, locked, state } from './state.js';

let pending;

export function renderModels() {
  const select = el('model-select');
  const model = currentModel();
  select.replaceChildren(new Option(state.models.length ? 'Choose a model' : 'No models', ''));
  for (const name of state.models) select.add(new Option(name, name));
  if (model && !state.models.includes(model)) select.add(new Option(model + ' (unavailable)', model));
  select.value = model;
  select.disabled = locked() || state.refreshingModels || !state.models.length;
  el('refresh-models').disabled = locked() || state.refreshingModels;
}

export async function refreshModels(render) {
  pending?.abort();
  const controller = new AbortController();
  pending = controller;
  state.refreshingModels = true;
  const status = el('connection');
  status.dataset.state = 'loading';
  status.textContent = 'Checking Ollama…';
  status.removeAttribute('title');
  render();
  try {
    const result = await api('/api/models', { signal: controller.signal });
    if (pending !== controller) return;
    state.models = result.models;
    state.defaultModel = result.default;
    if (!state.models.includes(state.draftModel)) state.draftModel = '';
    status.dataset.state = result.connected ? 'ready' : 'offline';
    status.textContent = !result.connected ? 'Ollama unavailable'
      : state.models.length ? 'Ollama connected' : 'No models installed';
    if (result.detail) status.title = result.detail;
    return result.detail;
  } catch (error) {
    if (pending !== controller) return;
    state.models = [];
    state.defaultModel = '';
    status.dataset.state = 'offline';
    status.textContent = 'Ollama status unavailable';
    return error.message;
  } finally {
    if (pending === controller) {
      pending = undefined;
      state.refreshingModels = false;
      render();
    }
  }
}
