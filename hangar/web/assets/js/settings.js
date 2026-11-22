import { api } from './api.js';
import { el, locked, node, notice, state } from './state.js';
import { applyTheme, savedTheme, saveTheme } from './theme.js';

const generationFields = {
  temperature: 'temperature',
  top_p: 'top-p',
  max_output_tokens: 'max-output-tokens',
};

function fillSettings(settings) {
  el('theme-select').value = savedTheme;
  el('ollama-url').value = settings.ollama_url;
  el('context-length').value = settings.context_length;
  el('instructions').value = settings.system_prompt_chat;
  el('memory-options').hidden = !settings.saved_facts.length;
  el('use-memory').checked = settings.use_saved_memory;
  el('saved-facts').replaceChildren(...settings.saved_facts.map(text => node('li', { text })));
  for (const [key, id] of Object.entries(generationFields)) el(id).value = settings[key] ?? '';

  const model = el('default-model');
  model.replaceChildren(new Option('Automatic', ''));
  for (const name of state.models) model.add(new Option(name, name));
  const current = settings.default_model;
  if (current && !state.models.includes(current)) model.add(new Option(current + ' (unavailable)', current));
  model.value = current;
}

function readSettings() {
  return {
    ollama_url: el('ollama-url').value,
    default_model: el('default-model').value,
    context_length: Number(el('context-length').value),
    system_prompt_chat: el('instructions').value,
    use_saved_memory: el('use-memory').checked,
    ...Object.fromEntries(Object.entries(generationFields).map(([key, id]) => [
      key, el(id).value === '' ? null : el(id).valueAsNumber,
    ])),
  };
}

export function bindSettings(refreshModels) {
  const dialog = el('settings-dialog');
  const target = el('settings-error');
  const form = el('settings-form');
  const submit = form.querySelector('button[type="submit"]');
  const theme = el('theme-select');
  const tabs = [...dialog.querySelectorAll('[data-settings-tab]')];
  const panels = [...dialog.querySelectorAll('[data-settings-panel]')];
  let loading = false;
  let original;

  function showPanel(name) {
    for (const tab of tabs) tab.setAttribute('aria-pressed', String(tab.dataset.settingsTab === name));
    for (const panel of panels) panel.hidden = panel.dataset.settingsPanel !== name;
  }

  for (const tab of tabs) tab.addEventListener('click', () => showPanel(tab.dataset.settingsTab));
  theme.addEventListener('change', () => applyTheme(theme.value));
  dialog.addEventListener('close', () => applyTheme());

  el('open-settings').addEventListener('click', async () => {
    if (locked() || loading || dialog.open) return;
    loading = true;
    notice('', false, target);
    try {
      const settings = await api('/api/settings');
      if (locked()) return;
      original = settings;
      fillSettings(settings);
      dialog.showModal();
    } catch (error) {
      notice(error.message, true);
    } finally {
      loading = false;
    }
  });

  form.addEventListener('submit', async event => {
    event.preventDefault();
    if (submit.disabled) return;
    const invalid = form.querySelector('input:invalid, select:invalid, textarea:invalid');
    if (invalid) {
      showPanel(invalid.closest('[data-settings-panel]').dataset.settingsPanel);
      invalid.reportValidity();
      return;
    }

    const choice = theme.value;
    const settings = readSettings();
    dialog.inert = submit.disabled = true;
    notice('', false, target);
    try {
      const saved = await api('/api/settings', { method: 'PUT', data: settings });
      saveTheme(choice);
      dialog.close();
      if (saved.ollama_url !== original.ollama_url || saved.default_model !== original.default_model) {
        await refreshModels();
      }
    } catch (error) {
      notice(error.message, true, dialog.open ? target : el('notice'));
    } finally {
      dialog.inert = submit.disabled = false;
    }
  });
  dialog.addEventListener('cancel', event => { if (submit.disabled) event.preventDefault(); });
}
