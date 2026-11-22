import { button, node } from './state.js';

export function folderTree(folders, select, load, search = false) {
  const list = node('ul', { className: 'folder-nodes' });
  for (const folder of folders) {
    const item = node('li');
    const row = node('div', { className: 'folder-node' });
    const children = node('div', { className: 'folder-children' });
    children.hidden = true;
    let loaded = false;

    const expand = button('▸', async () => {
      children.hidden = !children.hidden;
      const expanded = !children.hidden;
      expand.setAttribute('aria-expanded', String(expanded));
      expand.textContent = expanded ? '▾' : '▸';
      if (!expanded || loaded) return;

      loaded = true;
      children.replaceChildren(node('p', { className: 'field-help', text: 'Loading folders…' }));
      try {
        const result = await load(folder.path);
        children.replaceChildren(folderTree(result.folders, select, load));
        if (!result.folders.length) children.append(node('p', { className: 'field-help', text: 'No subfolders' }));
        if (result.truncated) children.append(node('p', { className: 'field-help', text: 'More folders available by path or search.' }));
      } catch (error) {
        loaded = false;
        if (error.name !== 'AbortError') {
          children.replaceChildren(node('p', { className: 'field-help', text: error.message }));
        }
      }
    }, 'folder-toggle');
    expand.setAttribute('aria-label', 'Subfolders of ' + folder.name);
    expand.setAttribute('aria-expanded', 'false');

    const pick = button('', () => select(folder, pick), 'folder-pick');
    pick.setAttribute('aria-pressed', 'false');
    pick.title = folder.path;
    const icon = node('span', { className: 'folder-glyph' });
    icon.setAttribute('aria-hidden', 'true');
    const label = node('span', { text: folder.name });
    if (search) label.append(node('small', { text: folder.path }));
    pick.append(icon, label);

    row.append(expand, pick);
    item.append(row, children);
    list.append(item);
  }
  return list;
}
