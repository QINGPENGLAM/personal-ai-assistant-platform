import type { RunEvent } from '../types';
export function ActivityPanel({ events, complete }: { events: RunEvent[]; complete: boolean }) {
  const items = new Map<string, RunEvent>();
  let truncated = false;
  for (const event of events) {
    if (event.kind === 'state' && event.payload.reason === 'output_truncated') truncated = true;
    if (event.kind === 'activity_started' || event.kind === 'activity_completed') items.set(event.sdk_item_id ?? String(event.payload.item_id), event);
  }
  return <section className="panel activity"><h2>Activity</h2>{!items.size && <p className="muted">Commands, file changes, and tool calls appear here.</p>}
    {truncated && <p className="notice">Output reached the display limit. Some activity details are incomplete.</p>}
    {[...items.entries()].map(([id, event]) => <details key={id}><summary>{String(event.payload.name ?? 'Activity')} · {event.kind === 'activity_completed' ? String(event.payload.status ?? 'Finished') : complete || truncated ? 'Stopped · output incomplete' : 'Running'}</summary><pre>{String(event.payload.summary ?? '')}</pre>{event.payload.output != null && <pre>{String(event.payload.output).slice(0, 65536)}</pre>}</details>)}
  </section>;
}
