import { useEffect, useRef, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { useAuth } from '../App';
import { api, ApiError, write } from '../api';
import type { Agent, Run } from '../types';
import { JobsPanel } from '../components/JobsPanel';
import { WorkspacePanel } from '../components/WorkspacePanel';
import { ChatPanel } from '../components/ChatPanel';
import { ErrorNotice } from '../components/ErrorNotice';
import { StatusBadge } from '../components/StatusBadge';
type CleanupCheck = { run_count: number; event_count: number; export_bytes: number; eligible: boolean; blockers: string[]; deletion_available: boolean };
type PurgeRequest = { version: 1; id: string; body: { run_ids: string[]; sha256: string; confirmed: true } };
type PurgeResult = { purged: boolean; run_count: number; reclamation_pending: boolean; maintenance_error?: string | null; reclaimed_bytes: number };
function savedPurge(key: string): PurgeRequest | null {
  try {
    const raw = sessionStorage.getItem(key); if (!raw || raw.length > 16000) return null;
    const value = JSON.parse(raw); const uuid = /^[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}$/;
    if (value.version !== 1 || !uuid.test(value.id) || value.body?.confirmed !== true || !/^[0-9a-f]{64}$/.test(value.body.sha256) || !Array.isArray(value.body.run_ids) || value.body.run_ids.length < 1 || value.body.run_ids.length > 100 || !value.body.run_ids.every((id: unknown) => typeof id === 'string' && uuid.test(id))) return null;
    return value;
  } catch { return null; }
}
type Control = 'continue' | 'prepare' | 'new-chat' | 'wake' | 'sleep' | 'restart' | 'reset-model-context';
export function AgentPage() {
  const { session } = useAuth();
  const { id } = useParams();
  const purgeKey = `archive-purge:${session.user.id}:${id}`;
  const [pendingPurge, setPendingPurge] = useState<PurgeRequest | null>(() => savedPurge(purgeKey));
  const [purgeResult, setPurgeResult] = useState<PurgeResult | null>(null);
  const [purgeError, setPurgeError] = useState<unknown>(null);
  const [purging, setPurging] = useState(false);
  const purgeController = useRef<AbortController | null>(null);
  const [agent, setAgent] = useState<Agent | null>(null);
  const [runs, setRuns] = useState<Run[]>([]); const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(true); const [busy, setBusy] = useState(false);
  const [revision, setRevision] = useState(0); const [controlIds, setControlIds] = useState<Partial<Record<Control, string>>>({});
  const [archiveRuns, setArchiveRuns] = useState<Run[]>([]);
  const [archivePages, setArchivePages] = useState<(string | null)[]>([null]);
  const [archiveNext, setArchiveNext] = useState<string | null>(null);
  const [archiveLoading, setArchiveLoading] = useState(true);
  const archiveCursor = archivePages[archivePages.length - 1];
  const [archiveIds, setArchiveIds] = useState<string[]>([]);
  const [archiveError, setArchiveError] = useState<unknown>(null);
  const [archiveNotice, setArchiveNotice] = useState('');
  const [lastArchive, setLastArchive] = useState<{ run_ids: string[]; sha256: string } | null>(null);
  const [cleanupCheck, setCleanupCheck] = useState<CleanupCheck | null>(null);
  const [checking, setChecking] = useState(false);
  const [checkError, setCheckError] = useState<unknown>(null);
  const [exporting, setExporting] = useState(false);
  const exportRequest = useRef<AbortController | null>(null);
  useEffect(() => {
    setArchiveIds([]); setArchiveNotice(''); setLastArchive(null); setCleanupCheck(null);
    return () => { exportRequest.current?.abort(); purgeController.current?.abort(); };
  }, [id]);
  const exportable = archiveRuns.filter(run => !run.retry_of && ['completed', 'completed_with_errors', 'failed', 'interrupted', 'cancelled'].includes(run.status) && !['pending', 'failed'].includes(run.cleanup_state));
  async function downloadHistory() {
    const controller = new AbortController(); exportRequest.current = controller;
    setExporting(true); setArchiveNotice(''); setCleanupCheck(null); setCheckError(null);
    try {
      const response = await api<Response>('/runs/archive/export', { method: 'POST', body: JSON.stringify({ run_ids: archiveIds }), signal: controller.signal }, 'response');
      const bytes = await response.arrayBuffer();
      const hash = await crypto.subtle.digest('SHA-256', bytes);
      const digest = Array.from(new Uint8Array(hash), byte => byte.toString(16).padStart(2, '0')).join('');
      if (digest !== response.headers.get('x-archive-sha256')) throw new Error('Export verification failed. Please download again.');
      if (controller.signal.aborted) return;
      const url = URL.createObjectURL(new Blob([bytes], { type: 'application/json' }));
      const anchor = document.createElement('a'); anchor.href = url; anchor.download = 'run-history.json'; anchor.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      setLastArchive({ run_ids: [...archiveIds], sha256: digest });
      setArchiveNotice(`Export verified · SHA-256 ${digest}. Saved history is kept. Check eligibility before confirming any deletion.`);
    } catch (error) { if (!controller.signal.aborted) setError(error); }
    finally { if (!controller.signal.aborted) setExporting(false); }
  }
  useEffect(() => {
    const controller = new AbortController(); let timer: ReturnType<typeof setTimeout>;
    setArchiveLoading(true); setArchiveRuns([]); setArchiveNext(null);
    async function loadArchive() {
      try {
        const page = await api<{ items: Run[]; next_cursor: string | null }>(`/agents/${id}/runs?archive_roots=true&limit=20${archiveCursor ? `&cursor=${encodeURIComponent(archiveCursor)}` : ''}`, { signal: controller.signal });
        if (!controller.signal.aborted) { setArchiveRuns(page.items); setArchiveNext(page.next_cursor); setArchiveError(null); }
      } catch (error) { if (!controller.signal.aborted) { setArchiveRuns([]); setArchiveNext(null); setArchiveError(error); } }
      finally { if (!controller.signal.aborted) { setArchiveLoading(false); timer = setTimeout(loadArchive, 5000); } }
    }
    void loadArchive(); return () => { controller.abort(); clearTimeout(timer); };
  }, [id, archiveCursor, revision]);
  async function checkDownloadedArchive() {
    if (!lastArchive || checking || exporting) return;
    const controller = new AbortController(); exportRequest.current = controller;
    setChecking(true); setCheckError(null); setCleanupCheck(null);
    try {
      const result = await api<CleanupCheck>('/runs/archive/check', { method: 'POST', body: JSON.stringify(lastArchive), signal: controller.signal });
      if (!controller.signal.aborted) setCleanupCheck(result);
    } catch (error) { if (!controller.signal.aborted) setCheckError(error); }
    finally { if (!controller.signal.aborted) setChecking(false); }
  }
  async function purgeDownloadedArchive() {
    let operation = pendingPurge;
    if (!operation) {
      if (!lastArchive || !cleanupCheck?.deletion_available) return;
      if (!window.confirm(`Permanently delete ${cleanupCheck.run_count} saved run(s) and their exported events? This cannot be undone. Workspace files, model history, assistants and scheduled jobs are kept. SHA-256: ${lastArchive.sha256}`)) return;
      operation = { version: 1, id: crypto.randomUUID(), body: { ...lastArchive, confirmed: true } };
      try { sessionStorage.setItem(purgeKey, JSON.stringify(operation)); }
      catch (error) { setPurgeError(error); return; }
      setPendingPurge(operation);
    }
    const controller = new AbortController(); purgeController.current = controller;
    setPurging(true); setPurgeError(null);
    try {
      const result = await api<PurgeResult>('/runs/archive/purge', { method: 'POST', headers: { 'X-Request-ID': operation.id }, body: JSON.stringify(operation.body), signal: controller.signal });
      if (controller.signal.aborted) return;
      setPurgeResult(result);
      if (result.purged) { setLastArchive(null); setCleanupCheck(null); setArchiveIds([]); setArchivePages([null]); refresh(); }
      if (!result.reclamation_pending) { sessionStorage.removeItem(purgeKey); setPendingPurge(null); }
    } catch (error) { if (!controller.signal.aborted) setPurgeError(error); }
    finally { if (!controller.signal.aborted) setPurging(false); }
  }
  const refresh = () => setRevision(value => value + 1);
  useEffect(() => {
    const controller = new AbortController(); let timer: ReturnType<typeof setTimeout>;
    async function load() {
      try {
        const [next, history] = await Promise.all([api<Agent>(`/agents/${id}`, { signal: controller.signal }), api<Run[]>(`/agents/${id}/chat`, { signal: controller.signal })]);
        if (!controller.signal.aborted) { setAgent(next); setRuns(history); setError(null); }
      } catch (error) { if (!controller.signal.aborted) setError(error); }
      finally { if (!controller.signal.aborted) { setLoading(false); timer = setTimeout(load, 2000); } }
    }
    void load(); return () => { controller.abort(); clearTimeout(timer); };
  }, [id, revision]);
  async function control(action: Control) {
    if (action === 'new-chat' && !window.confirm('Start a new conversation? Old model context will no longer be used. Queued chat messages and retries will be cancelled. Files, saved records, and jobs are kept. The assistant stays paused.')) return;
    if (action === 'sleep' && !window.confirm('Sleep this assistant? Current work will be interrupted and queued tasks will stay paused.')) return;
    if (action === 'restart' && !window.confirm('Restart and repair this assistant? Current work will be interrupted. Files and history are kept; queued tasks stay paused until Continue.')) return;
    if (action === 'reset-model-context' && !window.confirm('Permanently delete this assistant’s model history? Workspace files, credentials, platform chat/events and scheduled jobs are kept. Queued chat is cancelled and old conversations are archived. The assistant stays paused.')) return;
    const requestId = controlIds[action] ?? crypto.randomUUID();
    setControlIds(value => ({ ...value, [action]: requestId }));
    setBusy(true); setError(null);
    try {
      await write(`/agents/${id}/${action}`, ['new-chat', 'reset-model-context'].includes(action) ? { confirmed: true } : undefined, requestId);
      setControlIds(value => { const next = { ...value }; delete next[action]; return next; }); refresh();
    } catch (error) {
      setError(error);
      if (error instanceof ApiError && error.status < 500 && !['control_pending', 'control_uncertain'].includes(error.code)) setControlIds(value => { const next = { ...value }; delete next[action]; return next; });
    } finally { setBusy(false); }
  }
  if (loading) return <p role="status">Loading assistant…</p>;
  if (!agent) return <><Link to="/agents">← Assistants</Link><ErrorNotice error={error} /></>;
  return <><Link className="back" to="/agents">← All assistants</Link><div className="page-heading"><div><div className="eyebrow">PRIVATE ASSISTANT</div><h1>{agent.name}</h1><p className="muted">{agent.model} · Conversation {agent.chat_epoch + 1}</p></div><StatusBadge status={agent.paused ? 'paused' : agent.status} /></div>
    <ErrorNotice error={error} /><div className="agent-grid"><aside className="panel stack"><h2>Assistant controls</h2><p className="muted">{agent.paused ? 'Work is paused until you choose Continue.' : 'Queued tasks run while this page is closed.'}</p>
      {agent.storage_protection?.active && <div className="notice error" role="alert"><strong>Platform storage protection</strong><p>New work and writes are blocked. Saved data remains readable. Repair storage to enable recovery checks; affected assistants stay paused.</p><small>{agent.storage_protection.reason?.replaceAll('_', ' ') ?? 'Checking storage capacity'} · {agent.storage_protection.measurement_verified ? 'Quota measurement verified' : 'Quota measurement unavailable'}</small></div>}
      {agent.run_admission && <div className={agent.run_admission.limited ? 'notice' : 'muted'} role={agent.run_admission.limited ? 'status' : undefined}>
        {agent.run_admission.limited && <p>Model run limit reached. Queued work will be reconsidered after {agent.run_admission.retry_at ? new Date(agent.run_admission.retry_at).toLocaleString() : 'the limit clears'}.</p>}
        <small>Model starts: {agent.run_admission.minute_count}/{agent.run_admission.minute_limit} in the last minute · {agent.run_admission.day_count}/{agent.run_admission.day_limit} in the last 24 hours.</small>
      </div>}
      {agent.blockers.length > 0 && <div className="notice"><strong>Needs attention</strong><ul>{agent.blockers.map(blocker => <li key={blocker}>{blocker.replaceAll('_', ' ')}</li>)}</ul></div>}
      {agent.status !== 'ready' && <button disabled={busy || purging} onClick={() => control('prepare')}>Retry preparation</button>}
      <button disabled={busy || purging || !agent.paused || agent.blockers.length > 0 || agent.status !== 'ready'} onClick={() => control('continue')}>Continue assistant</button>
      <button className="secondary" disabled={busy || purging || !agent.paused || runs.some(run => ['starting', 'running', 'stopping'].includes(run.status))} onClick={() => control('new-chat')}>New chat</button>
      <small>Container: {agent.container_state}</small>
      <button className="secondary" disabled={busy || purging || agent.status !== 'ready' || agent.blockers.includes('recovery_error')} onClick={() => control('wake')}>Wake assistant</button>
      <button className="secondary" disabled={busy || purging || agent.blockers.includes('recovery_error')} onClick={() => control('sleep')}>Sleep assistant</button>
      <button className="secondary" disabled={busy || purging || agent.status !== 'ready'} onClick={() => control('restart')}>Restart assistant</button>
      <button className="secondary" disabled={busy || purging || !agent.paused || agent.container_state !== 'stopped' || agent.blockers.includes('recovery_error') || runs.some(run => ['starting', 'running', 'stopping'].includes(run.status))} onClick={() => control('reset-model-context')}>Reset model context</button>
      {Object.keys(controlIds).length > 0 && <div className="notice"><small>Unconfirmed control requests retain their original IDs. Retry the same action to inspect its result.</small><button className="secondary" disabled={busy || purging} onClick={() => { if (window.confirm('Have you inspected the assistant state? Discarding these IDs allows new logical control operations.')) setControlIds({}); }}>Discard inspected control requests</button></div>}
      <small>Stop an active run from its conversation card. Stop also pauses queued work.</small>

      <section className="stack" aria-labelledby="archive-heading"><h2 id="archive-heading">Export saved history</h2><ErrorNotice error={archiveError} />
        <small>Select completed chat or scheduled-task roots. Each export includes its terminal retries and events. Active or unclean retries must finish first. Up to 100 roots / 32 MiB per download; history stays saved.</small>
        {exportable.map(run => <label key={run.id}><input type="checkbox" checked={archiveIds.includes(run.id)} disabled={purging || exporting || archiveLoading || (!archiveIds.includes(run.id) && archiveIds.length >= 100)} aria-label={`Export: ${run.prompt}`} onChange={event => setArchiveIds(value => event.target.checked ? [...value, run.id] : value.filter(item => item !== run.id))} />{run.source === 'job' ? 'Scheduled task' : 'Chat'} · {run.prompt.slice(0, 80)}</label>)}
        <small>{archiveIds.length}/100 selected across pages · Page {archivePages.length}</small>
        {archiveLoading && <small role="status">Loading saved history…</small>}
        <button className="secondary" disabled={purging || exporting || archiveLoading || archivePages.length === 1} onClick={() => { setArchiveLoading(true); setArchivePages(value => value.slice(0, -1)); }}>Previous history page</button>
        <button className="secondary" disabled={purging || exporting || archiveLoading || !archiveNext} onClick={() => { if (archiveNext) { setArchiveLoading(true); setArchivePages(value => [...value, archiveNext]); } }}>Next history page</button>
        <button className="secondary" disabled={purging || exporting || archiveIds.length === 0} onClick={() => setArchiveIds([])}>Clear history selection</button>
        <button className="secondary" disabled={purging || exporting || checking || !!pendingPurge || archiveIds.length === 0} onClick={() => void downloadHistory()}>{exporting ? 'Preparing export…' : 'Download selected history'}</button>
        {archiveNotice && <small role="status">{archiveNotice}</small>}
        {lastArchive && <><small>Last verified download: {lastArchive.run_ids.length} root(s). The check uses that exact selection and digest.</small><button className="secondary" disabled={purging || exporting || checking} onClick={() => void checkDownloadedArchive()}>{checking ? 'Checking saved history…' : 'Check last downloaded archive'}</button></>}
        <ErrorNotice error={checkError} />
        {cleanupCheck && <div className="notice" role="status"><strong>Cleanup check · {cleanupCheck.run_count} {cleanupCheck.run_count === 1 ? 'run' : 'runs'} · {cleanupCheck.deletion_available ? 'ready for confirmation' : 'deletion blocked'}</strong><p>{cleanupCheck.event_count} saved events · {cleanupCheck.export_bytes} exported bytes.</p>{cleanupCheck.blockers.length > 0 ? <ul>{cleanupCheck.blockers.map(blocker => <li key={blocker}>{blocker.replaceAll('_', ' ').replace(/^./, value => value.toUpperCase())}</li>)}</ul> : <p>No dependency blockers were found in this snapshot.</p>}<small>This check reserves no records. Deletion requires separate confirmation and fresh server checks. Storage pressure or insufficient compaction space blocks new deletion.</small></div>}

        {lastArchive && cleanupCheck?.deletion_available && !pendingPurge && <button disabled={purging || checking || exporting} onClick={() => void purgeDownloadedArchive()}>Permanently delete downloaded records</button>}
        {pendingPurge && <div className="notice"><p>Cleanup request saved for {pendingPurge.body.run_ids.length} root(s). Retry uses the original selection, digest and request ID, including after reload.</p><button disabled={purging} onClick={() => void purgeDownloadedArchive()}>{purging ? 'Processing cleanup…' : 'Retry saved cleanup request'}</button><button className="secondary" disabled={purging} onClick={() => { if (window.confirm('Have you inspected the saved records? Discard this retry ID only. This does not restore deleted records.')) { try { sessionStorage.removeItem(purgeKey); setPendingPurge(null); } catch (error) { setPurgeError(error); } } }}>Discard inspected cleanup request</button></div>}
        <ErrorNotice error={purgeError} />
        {purgeResult && <div className="notice" role="status"><strong>{purgeResult.reclamation_pending ? 'Saved records deleted. Space recovery is pending.' : 'Saved records deleted. Space recovery complete.'}</strong>{purgeResult.reclamation_pending ? <p>Retry the saved cleanup request to continue recovery. {purgeResult.maintenance_error?.replaceAll('_', ' ')}</p> : <p>Database files released {purgeResult.reclaimed_bytes} bytes during compaction. Later bookkeeping writes are excluded.</p>}</div>}
      </section>
    </aside><div className="stack"><ChatPanel agent={agent} runs={runs} refresh={refresh} /><WorkspacePanel key={agent.id} agentId={agent.id} refreshAgent={refresh} /><JobsPanel key={`jobs-${agent.id}`} agentId={agent.id} refreshAgent={refresh} /></div></div></>;
}
