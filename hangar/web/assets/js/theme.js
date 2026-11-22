const preferenceKey = 'hangar.theme';
const system = window.matchMedia('(prefers-color-scheme: dark)');
const valid = new Set(['system', 'light', 'dark']);

function storedTheme() {
  try {
    const value = localStorage.getItem(preferenceKey);
    return valid.has(value) ? value : 'system';
  } catch {
    return 'system';
  }
}

export let savedTheme = storedTheme();
let preview = savedTheme;

export function applyTheme(choice = savedTheme) {
  preview = choice;
  document.documentElement.dataset.theme = choice === 'system'
    ? (system.matches ? 'dark' : 'light')
    : choice;
}

export function saveTheme(choice) {
  savedTheme = choice;
  applyTheme(choice);
  try {
    localStorage.setItem(preferenceKey, choice);
  } catch {
    // Keep the preference for the current page when browser storage is unavailable.
  }
}

system.addEventListener('change', () => applyTheme(preview));
applyTheme();
