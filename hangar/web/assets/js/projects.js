import { api } from './api.js';
import { renderChanges } from './changes.js';
import { browse, clearFiles, refreshFiles } from './project-files.js';
import { el, locked, notice, state } from './state.js';

function selectTab(tab) {
  const files = tab === 'files';
  el('files-tab').setAttribute('aria-pressed', String(files));
  el('changes-tab').setAttribute('aria-pressed', String(!files));
  el('file-browser').hidden = !files;
  el('change-list').hidden = files;
  if (files && !el('file-browser').hasChildNodes()) void browse();
  if (!files) renderChanges();
}

export function closeProject() {
  clearFiles();
  el('project-panel').hidden = true;
  el('toggle-project').setAttribute('aria-expanded', 'false');
}

export function openProject(tab = 'files') {
  const name = state.session?.project_name;
  if (!name) return;
  const project = state.projects.find(item => item.name === name);
  el('project-title').textContent = name;
  el('project-path').textContent = project?.root || '';
  el('project-path').title = project?.root || '';
  el('project-panel').hidden = false;
  el('toggle-project').setAttribute('aria-expanded', 'true');
  selectTab(tab);
}

export function bindProjects({ setProject }) {
  const dialog = el('projects-dialog');
  const selection = el('registered-project');
  const errorTarget = el('project-error');
  let busy = false;

  function renderPicker() {
    const current = state.session?.project_name || '';
    selection.replaceChildren(new Option('Choose a project', ''));
    for (const project of state.projects) {
      const option = new Option(project.local ? project.name : project.name + ' (SSH)', project.name);
      option.disabled = !project.local;
      selection.add(option);
    }
    selection.value = state.projects.some(project => project.name === current && project.local) ? current : '';
    el('detach-project').hidden = !current;
    el('attach-project').disabled = !selection.value;
  }

  function openDialog() {
    if (locked() || busy) return;
    notice('', false, errorTarget);
    renderPicker();
    dialog.showModal();
  }

  async function updateProject(action) {
    if (busy || locked()) return;
    busy = true;
    const controls = [...dialog.querySelectorAll('button, input, select')];
    controls.forEach(control => { control.disabled = true; });
    try {
      notice('', false, errorTarget);
      const name = await action();
      await setProject(name);
      closeProject();
      if (name) openProject();
      dialog.close();
    } catch (error) {
      notice(error.message, true, dialog.open ? errorTarget : el('notice'));
    } finally {
      busy = false;
      controls.forEach(control => { control.disabled = false; });
      el('attach-project').disabled = !selection.value;
    }
  }

  dialog.addEventListener('cancel', event => { if (busy) event.preventDefault(); });
  el('open-projects').addEventListener('click', openDialog);
  el('intro-project').addEventListener('click', openDialog);
  el('toggle-project').addEventListener('click', () => {
    if (el('project-panel').hidden) openProject();
    else closeProject();
  });
  el('close-project').addEventListener('click', closeProject);
  el('files-tab').addEventListener('click', () => selectTab('files'));
  el('changes-tab').addEventListener('click', () => selectTab('changes'));
  document.addEventListener('project-file-changed', refreshFiles);
  el('detach-project').addEventListener('click', () => updateProject(() => null));
  selection.addEventListener('change', () => { el('attach-project').disabled = !selection.value; });
  el('attach-project-form').addEventListener('submit', event => {
    event.preventDefault();
    if (selection.value) void updateProject(() => selection.value);
  });
  el('register-project-form').addEventListener('submit', event => {
    event.preventDefault();
    void updateProject(async () => {
      const record = await api('/api/projects', {
        method: 'POST', data: { name: el('register-name').value, root: el('register-root').value },
      });
      state.projects.push(record);
      selection.add(new Option(record.name, record.name));
      selection.value = record.name;
      event.target.reset();
      return record.name;
    });
  });
}
