import { api } from './api.js';
import { el, locked, node, notice } from './state.js';

export function bindSettings(refreshModels) {
  const dialog = el('settings-dialog');
  const target = el('settings-error');
  const form = el('settings-form');
  const submit = form.querySelector('button[type="submit"]');
  let loading = false;
  el('open-settings').addEventListener('click', async () => {
    if (locked() || loading || dialog.open) return;
    loading = true;
    notice('', false, target);
    try {
      const settings = await api('/api/settings');
      if (locked()) return;
      el('ollama-url').value = settings.ollama_url;
      el('context-length').value = settings.context_length;
      el('instructions').value = settings.system_prompt_chat;
      el('memory-options').hidden = !settings.saved_facts.length;
      el('use-memory').checked = settings.use_saved_memory;
      el('saved-facts').replaceChildren(...settings.saved_facts.map(text => node('li', { text })));
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
    submit.disabled = true;
    try {
      await api('/api/settings', {
        method: 'PUT',
        data: {
          ollama_url: el('ollama-url').value,
          context_length: Number(el('context-length').value),
          system_prompt_chat: el('instructions').value,
          use_saved_memory: el('use-memory').checked,
        },
      });
      dialog.close();
      await refreshModels();
    } catch (error) {
      notice(error.message, true, dialog.open ? target : el('notice'));
    } finally {
      submit.disabled = false;
    }
  });
}
