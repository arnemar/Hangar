import { button, node } from './state.js';

function inline(parent, text) {
  const parts = text.split(/(\*\*[^*]+\*\*|\x60[^\x60]+\x60|\[[^\]]+\]\(https?:\/\/[^\s)]+\))/g);
  for (const part of parts) {
    if (part.startsWith('**') && part.endsWith('**') && part.length > 4) {
      parent.append(node('strong', { text: part.slice(2, -2) }));
    } else if (part.startsWith('\x60') && part.endsWith('\x60') && part.length > 2) {
      parent.append(node('code', { text: part.slice(1, -1) }));
    } else {
      const link = part.match(/^\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)$/);
      if (link) {
        const anchor = node('a', { text: link[1] });
        anchor.href = link[2];
        anchor.target = '_blank';
        anchor.rel = 'noopener noreferrer';
        parent.append(anchor);
      } else {
        parent.append(document.createTextNode(part));
      }
    }
  }
}

function codeBlock(lang, lines) {
  const wrapper = node('div', { className: 'code-block' });
  const toolbar = node('div', { className: 'code-toolbar' });
  const source = lines.join('\n');
  const copy = button('Copy', async () => {
    try {
      await navigator.clipboard.writeText(source);
      copy.textContent = 'Copied';
    } catch {
      copy.textContent = 'Could not copy';
    }
  });
  toolbar.append(node('span', { text: lang || 'Code' }), copy);
  const pre = node('pre');
  pre.append(node('code', { text: source }));
  wrapper.append(toolbar, pre);
  return wrapper;
}

export function markdown(parent, content) {
  const lines = content.split('\n');
  let group, kind;
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (line.startsWith('\x60\x60\x60')) {
      const code = [];
      const lang = line.slice(3).trim();
      while (++i < lines.length && !lines[i].startsWith('\x60\x60\x60')) code.push(lines[i]);
      parent.append(codeBlock(lang, code));
      group = kind = null;
      continue;
    }
    const numbered = line.match(/^\d+\.\s+(.*)/);
    const bullet = line.match(/^[-*]\s+(.*)/);
    if (numbered || bullet) {
      const tag = numbered ? 'ol' : 'ul';
      if (kind !== tag) {
        group = node(tag);
        parent.append(group);
        kind = tag;
      }
      const item = node('li');
      inline(item, (numbered || bullet)[1]);
      group.append(item);
      continue;
    }
    group = kind = null;
    if (!line.trim()) continue;
    const heading = line.match(/^(#{1,3})\s+(.*)/);
    const quote = line.match(/^>\s?(.*)/);
    const paragraph = node(heading ? 'h' + (heading[1].length + 1) : quote ? 'blockquote' : 'p');
    inline(paragraph, heading ? heading[2] : quote ? quote[1] : line);
    parent.append(paragraph);
  }
}

