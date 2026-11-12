import { useEffect, useState } from 'react';
import { api, ApiError, write } from '../api';
import type { Run } from '../types';
import { ErrorNotice } from './ErrorNotice';
import { RunDetailPanel } from './RunDetailPanel';

type Job = { job_id: string; name: string; prompt: string; created_by: 'user' | 'agent'; run_at: string; interval_seconds: number | null; next_run_at: string | null; enabled: boolean; cancelled_at: string | null; active_run_id: string | null; lateness_ms: number };
type Page<T> = { items: T[]; next_cursor: string | null };
type JobRun = Run & { scheduled_for: string | null; coalesced_count: number; started_at: string | null };
type Pending = { id: string; body: { name: string; prompt: string; run_at: string; interval_seconds?: number } };
const offsetPattern = /^[+-](?:(?:0\d|1[0-3]):[0-5]\d|14:00)$/;

export function JobsPanel({ agentId, refreshAgent }: { agentId: string; refreshAgent: () => void }) {
  const base = `/agents/${agentId}/jobs`;
  const [items, setItems] = useState<Job[]>([]); const [cursor, setCursor] = useState<string | null>(null);
  const [name, setName] = useState(''); const [prompt, setPrompt] = useState('');
  const [localTime, setLocalTime] = useState(''); const [offset, setOffset] = useState('+00:00');
  const [interval, setInterval] = useState(''); const [pending, setPending] = useState<Pending | null>(null);
  const [cancelIds, setCancelIds] = useState<Record<string, string>>({});
  const [selected, setSelected] = useState<string | null>(null); const [runs, setRuns] = useState<JobRun[]>([]);
  const [runCursor, setRunCursor] = useState<string | null>(null);
  const [jobPages, setJobPages] = useState(1); const [runPages, setRunPages] = useState(1);
  const [busy, setBusy] = useState(false); const [error, setError] = useState<unknown>(null);
  const [notice, setNotice] = useState(''); const [revision, setRevision] = useState(0);
  const timestamp = `${localTime}${localTime.length === 16 ? ':00' : ''}${offset}`;
  const refresh = () => { setRevision(value => value + 1); refreshAgent(); };
  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    async function load() {
      try {
        let jobCursor: string | null = null; const loadedJobs: Job[] = [];
        for (let page = 0; page < jobPages; page++) {
          const result: Page<Job> = await api<Page<Job>>(`${base}${jobCursor ? `?cursor=${encodeURIComponent(jobCursor)}` : ''}`, { signal: controller.signal });
          loadedJobs.push(...result.items); jobCursor = result.next_cursor;
          if (!jobCursor) break;
        }
        if (!controller.signal.aborted) { setItems(loadedJobs); setCursor(jobCursor); }
        if (selected) {
          let historyCursor: string | null = null; const loadedRuns: JobRun[] = [];
          for (let page = 0; page < runPages; page++) {
            const history: Page<JobRun> = await api<Page<JobRun>>(`/jobs/${selected}/runs${historyCursor ? `?cursor=${encodeURIComponent(historyCursor)}` : ''}`, { signal: controller.signal });
            loadedRuns.push(...history.items); historyCursor = history.next_cursor;
            if (!historyCursor) break;
          }
          if (!controller.signal.aborted) { setRuns(loadedRuns); setRunCursor(historyCursor); }
        }
      } catch (error) { if (!controller.signal.aborted) setError(error); }
      finally { if (!controller.signal.aborted) timer = setTimeout(load, 5000); }
    }
    void load(); return () => { controller.abort(); clearTimeout(timer); };
  }, [base, selected, revision, jobPages, runPages]);

  async function create(event: React.FormEvent) {
    event.preventDefault(); setBusy(true); setError(null); setNotice('');
    try {
      if (!pending && (!offsetPattern.test(offset) || offset === '-00:00' || !localTime || !Number.isFinite(Date.parse(timestamp)))) throw new Error('Choose a valid date/time and explicit numeric UTC offset.');
      const operation = pending ?? { id: crypto.randomUUID(), body: { name, prompt, run_at: timestamp, ...(interval ? { interval_seconds: Number(interval) } : {}) } };
      setPending(operation);
      await write(base, operation.body, operation.id);
      setPending(null); setName(''); setPrompt(''); setNotice('Scheduled task saved. Paused assistants keep due work waiting.'); refresh();
    } catch (error) {
      setError(error);
      if (error instanceof ApiError && error.status < 500) setPending(null);
    } finally { setBusy(false); }
  }
  async function cancel(job: Job) {
    if (!window.confirm('Cancel this schedule and its queued runs? Work already starting or running will continue until you Stop it separately.')) return;
    const id = cancelIds[job.job_id] ?? crypto.randomUUID();
    setCancelIds(value => ({ ...value, [job.job_id]: id })); setBusy(true); setError(null);
    try {
      const result = await write<{ active_run_id: string | null }>(`/jobs/${job.job_id}/cancel`, undefined, id);
      setNotice(result.active_run_id ? 'Schedule cancelled. An executing run remains; open its history and use Stop run separately.' : 'Schedule cancelled; queued work was cancelled.');
      setCancelIds(value => { const next = { ...value }; delete next[job.job_id]; return next; }); refresh();
    } catch (error) { setError(error); } finally { setBusy(false); }
  }
  function more(history: boolean) {
    if (history) setRunPages(value => value + 1);
    else setJobPages(value => value + 1);
  }
  return <section className="panel stack" aria-label="Scheduled tasks"><h2>Scheduled tasks</h2>
    <p className="muted">One-time or fixed UTC intervals. Tasks share this assistant's queue and use separate model conversations.</p>
    <ErrorNotice error={error} />{notice && <p role="status">{notice}</p>}
    <form className="stack" onSubmit={create}>
      <label>Task name<input required maxLength={80} value={name} disabled={busy || !!pending} onChange={event => setName(event.target.value)} /></label>
      <label>Task instructions<textarea required maxLength={16384} rows={3} value={prompt} disabled={busy || !!pending} onChange={event => setPrompt(event.target.value)} /></label>
      <label>First run date and time<input type="datetime-local" step="1" required value={localTime} disabled={busy || !!pending} onChange={event => setLocalTime(event.target.value)} /></label>
      <label>UTC offset<input required value={offset} placeholder="-05:00" disabled={busy || !!pending} onChange={event => setOffset(event.target.value)} /></label>
      <label>Repeat interval in seconds (optional)<input type="number" min={60} max={31536000} step={1} value={interval} disabled={busy || !!pending} onChange={event => setInterval(event.target.value)} /></label>
      <small>Resolved timestamp: {localTime ? timestamp : 'Choose a time'} · Leave interval empty for one-time work. A 24-hour interval is not a fixed local time across daylight saving changes.</small>
      <button disabled={busy}>{pending ? 'Retry saving task' : 'Save scheduled task'}</button>
      {pending && <p>Original values and request ID are retained until the result is confirmed.</p>}
    </form>
    {items.length === 0 && <p>No scheduled tasks yet.</p>}
    {items.map(job => <article className="stack" key={job.job_id}><h3>{job.name}</h3><small>Created by {job.created_by} · {job.enabled ? 'Enabled' : job.cancelled_at ? 'Cancelled' : 'One-time occurrence queued'}</small><p>{job.prompt}</p><small>First run: {job.run_at}<br />Next run: {job.next_run_at ?? 'None'} · {job.interval_seconds ? `Every ${job.interval_seconds} seconds` : 'One-time'}{job.lateness_ms > 0 && ` · Due ${Math.floor(job.lateness_ms / 1000)} seconds ago`}</small><div className="actions"><button className="secondary" disabled={busy} onClick={() => { setRuns([]); setRunPages(1); setSelected(job.job_id); refresh(); }}>View task runs: {job.name}</button><button className="secondary" disabled={busy || !!job.cancelled_at} onClick={() => cancel(job)}>Cancel task: {job.name}</button></div></article>)}
    {cursor && <button disabled={busy} onClick={() => more(false)}>More scheduled tasks</button>}
    {selected && <div className="stack"><h3>Scheduled run history</h3>{runs.length === 0 && <p>No runs yet.</p>}{runs.map(run => <div key={run.id}><small>Scheduled: {run.scheduled_for ?? 'Manual retry'} · Started: {run.started_at ?? 'Waiting'} · Coalesced missed occurrences: {run.coalesced_count}</small><RunDetailPanel run={run} archived={false} refresh={refresh} /></div>)}{runCursor && <button disabled={busy} onClick={() => more(true)}>More task runs</button>}</div>}
  </section>;
}
