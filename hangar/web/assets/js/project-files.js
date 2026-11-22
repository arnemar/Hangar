import { api } from './api.js';
import { attachFile } from './chat.js';
import { button, el, node, state } from './state.js';

let requestNumber = 0;
let currentLocation;
const browser = () => el('file-browser');

function pathUrl(project, endpoint, path) {
  return '/api/projects/' + encodeURIComponent(project) + endpoint + '?' + new URLSearchParams({ path });
}

function status(text, error = false) {
  browser().replaceChildren(node('p', { text, className: error ? 'notice error' : 'empty-panel' }));
}

function parent(path) {
  return path.split('/').slice(0, -1).join('/');
}

async function openPath(path, file) {
  const project = state.session?.project_name;
  if (!project) return;
  currentLocation = { path, file };
  const number = ++requestNumber;
  status(file ? 'Opening file…' : 'Loading files…');
  try {
    const view = await api(pathUrl(project, file ? '/file' : '/files', path));
    if (number !== requestNumber || project !== state.session?.project_name) return;
    if (file) showFile(view);
    else showFolder(view);
  } catch (error) {
    if (number !== requestNumber || project !== state.session?.project_name) return;
    status(error.message, true);
    if (path) browser().append(button('← Parent folder', () => browse(parent(path))));
  }
}

function showFile({ path, content }) {
  const actions = node('div', { className: 'preview-actions' });
  actions.append(
    button('← Files', () => browse(parent(path))),
    button('Attach excerpt', () => attachFile(path), 'secondary'),
  );
  browser().replaceChildren(
    actions, node('p', { text: path, className: 'folder-path' }),
    node('pre', { text: content, className: 'file-text' }),
  );
}

function showFolder({ path, entries, truncated }) {
  const rows = entries.map(file => {
    const row = button('', () => openPath(file.path, !file.directory), 'file-row');
    const icon = node('span', { text: file.directory ? '▸' : '·' });
    icon.setAttribute('aria-hidden', 'true');
    row.append(icon, node('span', { text: file.name }));
    return row;
  });
  browser().replaceChildren();
  if (path) browser().append(button('← Parent folder', () => browse(parent(path))));
  browser().append(node('p', { text: path || 'Project folder', className: 'folder-path' }), ...rows);
  if (!entries.length) browser().append(node('p', { text: 'This folder is empty.', className: 'empty-panel' }));
  if (truncated) browser().append(node('p', {
    text: 'Showing the first 300 files and folders.', className: 'empty-panel',
  }));
}

export function browse(path = '') {
  return openPath(path, false);
}

export function clearFiles() {
  ++requestNumber;
  currentLocation = undefined;
  browser().replaceChildren();
}

export function refreshFiles() {
  if (currentLocation) void openPath(currentLocation.path, currentLocation.file);
}
