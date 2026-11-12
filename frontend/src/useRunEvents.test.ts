import { act, createElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useRunEvents } from './useRunEvents';

class Source {
  static all: Source[] = [];
  listeners = new Map<string, EventListener>();
  onerror: (() => void) | null = null;
  onopen: (() => void) | null = null;
  closed = false;
  constructor(public url: string) { Source.all.push(this); }
  addEventListener(name: string, callback: EventListener) { this.listeners.set(name, callback); }
  close() { this.closed = true; }
  emit(sequence: number, text: string) {
    this.listeners.get('message_delta')?.(new MessageEvent('message_delta', {
      data: JSON.stringify({ sequence, kind: 'message_delta', sdk_item_id: 'reply', payload: { item_id: 'reply', text } }),
    }));
  }
}
let root: Root;
let host: HTMLDivElement;
function Reader({ id }: { id: string }) {
  const state = useRunEvents(id);
  return createElement('div', null, JSON.stringify(state.events));
}
beforeEach(() => {
  vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true);
  vi.stubGlobal('EventSource', Source);
  Source.all = [];
  host = document.createElement('div'); document.body.append(host);
  root = createRoot(host);
});
afterEach(async () => {
  await act(() => root.unmount()); host.remove();
  vi.useRealTimers(); vi.unstubAllGlobals();
});
describe('persisted run events', () => {
  it('deduplicates replay and ignores a previous run after navigation', async () => {
    await act(() => root.render(createElement(Reader, { id: 'run-one' })));
    const first = Source.all[0];
    await act(() => { first.emit(1, 'Hello'); first.emit(1, 'Hello'); });
    expect(JSON.parse(host.textContent ?? '[]')).toHaveLength(1);
    await act(() => root.render(createElement(Reader, { id: 'run-two' })));
    expect(first.closed).toBe(true);
    await act(() => first.emit(2, 'Private old output'));
    expect(host.textContent).not.toContain('Private old output');
    expect(host.textContent).not.toContain('Hello');
  });
  it('reconnects using the displayed cursor rather than restarting replay at zero', async () => {
    vi.useFakeTimers();
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => ({ status: 'running', last_event_sequence: 1 }) }));
    await act(() => root.render(createElement(Reader, { id: 'run-one' })));
    await act(async () => { Source.all[0].emit(1, 'Hello'); Source.all[0].onerror?.(); });
    await act(() => vi.advanceTimersByTime(1000));
    expect(Source.all[1].url).toContain('after=1');
    await act(() => Source.all[0].emit(2, 'Late retired connection'));
    expect(host.textContent).not.toContain('Late retired connection');
  });
});

it('a stale unauthorized response cannot expire a newer signed-in session', async () => {
  const { api, setCsrf } = await import('./api');
  let respond: (value: unknown) => void = () => {};
  vi.stubGlobal('fetch', vi.fn().mockImplementation(() => new Promise(resolve => { respond = resolve; })));
  const expired = vi.fn(); window.addEventListener('session-expired', expired);
  try {
    setCsrf('first-session');
    const pending = api('/runs/old', { method: 'POST' }).catch(() => undefined);
    setCsrf(null); setCsrf('second-session');
    respond({ ok: false, status: 401, json: async () => ({ code: 'login_required', message: 'Sign in again.' }) });
    await pending;
    expect(expired).not.toHaveBeenCalled();
    const current = api('/auth/session').catch(() => undefined);
    respond({ ok: false, status: 401, json: async () => ({ code: 'login_required' }) });
    await current;
    expect(expired).toHaveBeenCalledOnce();
  } finally { window.removeEventListener('session-expired', expired); setCsrf(null); }
});
