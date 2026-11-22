import { api } from './api.js';
import { folderTree } from './folder-tree.js';
import { el, notice } from './state.js';

const absolutePath = /^(?:\/|\\|~(?:$|[\\/])|[a-z]:[\\/])/i;

function folders(path, query, signal) {
  const params = new URLSearchParams({ path });
  if (query) params.set('q', query);
  return api('/api/projects/folders?' + params, { signal });
}

export function bindFolderPicker() {
  const input = el('folder-query');
  const name = el('register-name');
  const output = el('selected-folder');
  const list = el('folder-list');
  const status = el('folder-status');
  const current = el('folder-current');
  let view, request, selected, selectedButton;
  let viewQuery = '';

  function select(folder, control) {
    selectedButton?.setAttribute('aria-pressed', 'false');
    control?.setAttribute('aria-pressed', 'true');
    selectedButton = control;
    selected = folder;
    name.value = folder?.name.replace(/[\\/]/g, '').slice(0, 100) || (folder ? 'Root folder' : '');
    output.textContent = folder?.path || '';
    output.title = folder?.path || '';
    el('register-project-form').hidden = !folder;
    el('folder-selection-hint').hidden = !!folder;
    el('register-project').disabled = name.disabled = !folder;
  }

  function showView(data, query, controller) {
    const location = el('folder-location');
    location.textContent = data.folder.path;
    location.title = data.folder.path;
    el('folder-up').disabled = !data.parent;
    el('folder-back').hidden = !query;
    current.disabled = current.hidden = false;
    const load = path => folders(path, '', controller.signal);
    list.replaceChildren(folderTree(data.folders, select, load, !!query));
    const message = data.truncated ? 'Showing a limited set of folders. Narrow your search or enter a path.'
      : data.folders.length ? '' : query ? 'No matching folders.' : 'No subfolders.';
    notice(message, false, status);
  }

  async function navigate(path = '', query = '', pick = false) {
    request?.abort();
    const controller = new AbortController();
    request = controller;
    list.replaceChildren();
    current.disabled = true;
    notice(query ? 'Searching folders…' : 'Loading folders…', false, status);
    try {
      const data = await folders(path, query, controller.signal);
      if (request !== controller) return;
      view = data;
      viewQuery = query;
      showView(data, query, controller);
      if (pick) select(view.folder);
    } catch (error) {
      if (request !== controller || error.name === 'AbortError') return;
      if (view) showView(view, viewQuery, controller);
      if (error.status === 404 && !error.detail) {
        if (pick) select({ name: path.split(/[\\/]/).filter(Boolean).at(-1) || '', path });
        notice('Restart Hangar to enable the folder tree. You can still enter an absolute path.', false, status);
      } else {
        notice(error.message, true, status);
      }
    }
  }

  el('folder-query-form').addEventListener('submit', event => {
    event.preventDefault();
    const value = input.value.trim();
    if (absolutePath.test(value)) void navigate(value, '', true);
    else void navigate(view?.folder.path, value);
  });
  el('folder-home').addEventListener('click', () => { input.value = ''; void navigate(); });
  el('folder-up').addEventListener('click', () => { input.value = ''; void navigate(view?.parent); });
  el('folder-back').addEventListener('click', () => { input.value = ''; void navigate(view?.folder.path); });
  current.addEventListener('click', () => { if (view) select(view.folder); });

  return {
    open() {
      view = undefined;
      viewQuery = '';
      input.value = '';
      el('folder-location').textContent = '';
      el('folder-up').disabled = true;
      el('folder-back').hidden = true;
      current.hidden = true;
      select();
      void navigate();
    },
    close() {
      request?.abort();
      request = undefined;
    },
    selectedPath() {
      return selected?.path;
    },
  };
}
