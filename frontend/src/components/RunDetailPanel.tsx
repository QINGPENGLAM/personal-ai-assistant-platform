import { useState } from 'react';
import { write } from '../api';
import { terminal, type Accepted, type Run } from '../types';
import { useRunEvents } from '../useRunEvents';
import { ActivityPanel } from './ActivityPanel';
import { ErrorNotice } from './ErrorNotice';
import { StatusBadge } from './StatusBadge';
export function RunDetailPanel({ run, archived, refresh }: { run: Run; archived: boolean; refresh: () => void }) {
  // Queued runs have no assistant output yet; polling discovers when work starts.
  // Keeping their streams open would exhaust the browser's HTTP/1 connection pool.
  const { events, connection } = useRunEvents(run.status === 'queued' ? null : run.id);
  const [busy, setBusy] = useState(false); const [error, setError] = useState<unknown>(null);
  const [notice, setNotice] = useState(''); const [retryId, setRetryId] = useState<string | null>(null);
  const messages = new Map<string, string>();
  for (const event of events) {
    const id = String(event.payload.item_id ?? event.sdk_item_id ?? 'reply');
    if (id.startsWith('user:')) continue;
    if (event.kind === 'message_delta') messages.set(id, (messages.get(id) ?? '') + String(event.payload.text ?? ''));
    if (event.kind === 'message_snapshot') messages.set(id, String(event.payload.full_text ?? ''));
  }
  const latestState = [...events].reverse().find(event => event.kind === 'state');
  async function stop() {
    setBusy(true); setError(null);
    try { await write(`/runs/${run.id}/stop`); refresh(); } catch (error) { setError(error); } finally { setBusy(false); }
  }
  async function retry() {
    if (!window.confirm('Retry sends this prompt again in the saved conversation. Files and external actions may be repeated. Continue?')) return;
    const id = retryId ?? crypto.randomUUID(); setRetryId(id); setBusy(true); setError(null);
    try {
      const result = await write<Accepted>(`/runs/${run.id}/retry`, undefined, id);
      setNotice(`${result.paused ? 'Retry queued. Continue is still required.' : 'Retry queued.'}${result.context_warning ? ' No prior model context was available.' : ''}`); setRetryId(null); refresh();
    } catch (error) { setError(error); } finally { setBusy(false); }
  }
  return <article className="panel run"><div className="run-heading"><StatusBadge status={run.status} /><small>{new Date(run.created_at).toLocaleString()}</small><small role="status">{run.status === 'queued' ? 'Waiting' : connection}</small></div>
    <h3 className="prompt">{run.prompt}</h3>{run.retry_of && <small>Retry of {run.retry_of}</small>}
    <div className="reply">{[...messages.entries()].map(([id, text]) => <p key={id}>{text}</p>)}{!messages.size && <p className="muted">{run.status === 'queued' ? 'Waiting in the assistant queue.' : 'No saved reply yet.'}</p>}</div>
    {latestState?.payload.history_incomplete === true && <p className="notice">Some model history could not be recovered. Saved output is shown.</p>}
    <ActivityPanel events={events} complete={terminal(run.status) || latestState?.payload.activities_closed === true} />
    {run.error_message && <p className="notice error">{run.error_message}</p>}<ErrorNotice error={error} />{notice && <p role="status">{notice}</p>}
    <div className="actions">{['starting', 'running'].includes(run.status) && <button className="danger" disabled={busy} onClick={stop}>Stop run</button>}
      {!archived && ['failed', 'interrupted', 'completed_with_errors'].includes(run.status) && <button className="secondary" disabled={busy} onClick={retry}>Retry prompt</button>}</div>
  </article>;
}
