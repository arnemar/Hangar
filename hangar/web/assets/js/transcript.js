import { markdown } from './markdown.js';
import { el, node, state } from './state.js';

function turn(message, streaming = false) {
  const isUser = message.role === 'user';
  const article = node('article', { className: 'turn ' + (isUser ? 'user' : 'assistant') });
  article.setAttribute('aria-label', isUser ? 'You' : 'Hangar');
  if (!isUser) article.append(node('p', { text: 'Hangar', className: 'role' }));
  const answer = node('div', { className: 'answer' + (streaming ? ' streaming' : '') });
  if (isUser || streaming) answer.textContent = message.content ?? '';
  else markdown(answer, message.content ?? '');
  article.append(answer);
  if (message.attachments?.length) {
    const files = node('div', { className: 'turn-meta' });
    for (const path of message.attachments) files.append(node('span', { text: path }));
    article.append(files);
  }
  return article;
}

export function renderTranscript({ follow = false } = {}) {
  const feed = el('feed');
  const nearBottom = feed.scrollHeight - feed.scrollTop - feed.clientHeight < 65;
  const target = el('transcript');
  const run = state.run;
  const messages = [...state.messages];
  if (run && (!run.sessionId || run.sessionId === state.session?.id)) {
    messages.push({ role: 'user', content: run.message, attachments: run.attachments });
  }
  target.replaceChildren(...messages.map(message => turn(message)));
  if (run?.answer) target.append(turn({ role: 'assistant', content: run.answer }, true));
  if (run?.activity && !run.answer) {
    const label = node('p', { text: run.activity, className: 'activity' });
    label.setAttribute('role', 'status');
    target.append(label);
  }
  el('intro').hidden = !!(messages.length || run);
  if (follow || nearBottom) feed.scrollTop = feed.scrollHeight;
}

export function updateStream() {
  const target = el('transcript');
  const answer = target.querySelector('.streaming');
  if (!answer || !state.run?.answer) {
    renderTranscript();
    return;
  }
  const feed = el('feed');
  const atBottom = feed.scrollHeight - feed.clientHeight - feed.scrollTop < 65;
  answer.textContent = state.run.answer;
  if (atBottom) feed.scrollTop = feed.scrollHeight;
}
