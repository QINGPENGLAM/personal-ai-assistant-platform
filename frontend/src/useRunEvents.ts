import { useEffect, useState } from 'react';
import { api } from './api';
import { terminal, type Run, type RunEvent, type EventKind } from './types';

interface State { runId: string | null; events: RunEvent[]; connection: string }
const kinds: EventKind[] = ['message_delta', 'message_snapshot', 'activity_started', 'activity_completed', 'error', 'state'];
export function useRunEvents(runId: string | null) {
  const [state, setState] = useState<State>({ runId, events: [], connection: 'Loading events' });
  useEffect(() => {
    let alive = true;
    let source: EventSource | undefined;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let cursor = 0;
    const controller = new AbortController();
    const events = new Map<number, RunEvent>();
    const update = (connection: string) => {
      if (alive) setState({ runId, events: [...events.values()].sort((a, b) => a.sequence - b.sequence), connection });
    };
    function connect() {
      if (!alive || !runId) return;
      const current = new EventSource(`/api/runs/${encodeURIComponent(runId)}/events?after=${cursor}`, { withCredentials: true });
      source = current;
      current.onopen = () => { if (source === current) update('Connected'); };
      for (const kind of kinds) current.addEventListener(kind, ((message: MessageEvent) => {
        if (!alive || source !== current) return;
        try {
          const event = JSON.parse(message.data) as RunEvent;
          if (!Number.isSafeInteger(event.sequence) || event.sequence <= cursor) return;
          events.set(event.sequence, event); cursor = event.sequence; update('Connected');
        } catch { update('Could not read an event'); }
      }) as EventListener);
      current.onerror = async () => {
        current.close();
        if (!alive || source !== current) return;
        // EOF can mean completion; drain all saved events before closing permanently.
        try {
          const run = await api<Run>(`/runs/${runId}`, { signal: controller.signal });
          if (!alive || source !== current) return;
          if (terminal(run.status) && cursor >= (run.last_event_sequence ?? 0)) { update('Complete'); return; }
        } catch {
          if (!alive || source !== current) return;
        }
        update('Reconnecting');
        timer = setTimeout(connect, 1000);
      };
    }
    update(runId ? 'Loading events' : 'No run selected'); connect();
    return () => { alive = false; controller.abort(); source?.close(); clearTimeout(timer); };
  }, [runId]);
  return state.runId === runId ? state : { runId, events: [], connection: 'Loading events' };
}
