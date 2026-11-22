function detail(body, fallback) {
  if (typeof body?.detail === 'string') return body.detail;
  if (Array.isArray(body?.detail)) {
    return body.detail.map(error => error.msg).join(' ');
  }
  return fallback;
}

async function request(url, { method = 'GET', data, signal } = {}) {
  const response = await fetch(url, {
    method, signal,
    headers: data === undefined ? undefined : { 'Content-Type': 'application/json' },
    body: data === undefined ? undefined : JSON.stringify(data),
  });
  if (!response.ok) {
    const fallback = 'Request failed (' + response.status + ').';
    const body = await response.json().catch(() => null);
    const message = detail(body, '');
    throw Object.assign(new Error(message || fallback), { status: response.status, detail: message });
  }
  return response;
}

export async function api(url, options) {
  const response = await request(url, options);
  return response.status === 204 ? null : response.json();
}

export async function* chatEvents(url, data, signal) {
  const response = await request(url, { method: 'POST', data, signal });
  if (!response.headers.get('content-type')?.startsWith('text/event-stream') || !response.body) {
    throw new Error('Hangar returned an invalid chat stream.');
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) return;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();
      for (const line of lines) {
        if (line.startsWith('data: ')) yield JSON.parse(line.slice(6));
      }
    }
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}

export function sessionUrl(id, suffix = '') {
  return '/api/sessions/' + encodeURIComponent(id) + suffix;
}
