import { useEffect, useRef, useState, type FormEvent } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { api, write } from '../api';
import type { Agent } from '../types';
import { StatusBadge } from '../components/StatusBadge';
import { ErrorNotice } from '../components/ErrorNotice';
export function DashboardPage() {
  const [agents, setAgents] = useState<Agent[]>([]); const [models, setModels] = useState<string[]>([]);
  const [loading, setLoading] = useState(true); const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null); const request = useRef<{ body: string; id: string } | null>(null);
  const navigate = useNavigate();
  useEffect(() => {
    const controller = new AbortController();
    Promise.all([api<Agent[]>('/agents', { signal: controller.signal }), api<{ models: string[] }>('/models', { signal: controller.signal })])
      .then(([agents, result]) => { if (!controller.signal.aborted) { setAgents(agents); setModels(result.models); } })
      .catch(error => { if (!controller.signal.aborted) setError(error); }).finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, []);
  async function create(event: FormEvent<HTMLFormElement>) {
    event.preventDefault(); if (busy) return;
    const data = new FormData(event.currentTarget); const body = { name: data.get('name'), model: data.get('model') };
    const serialized = JSON.stringify(body);
    if (request.current?.body !== serialized) request.current = { body: serialized, id: crypto.randomUUID() };
    setBusy(true); setError(null);
    try { const agent = await write<Agent>('/agents', body, request.current!.id); navigate(`/agents/${agent.id}`); }
    catch (error) { setError(error); } finally { setBusy(false); }
  }
  return <><div className="page-heading"><div><div className="eyebrow">YOUR WORKSPACE</div><h1>Your assistants</h1><p className="muted">A separate workspace and conversation for every assistant.</p></div></div>
    <ErrorNotice error={error} /><div className="dashboard-grid"><section><h2>Assistants <span className="muted">{agents.length}</span></h2>
      {loading ? <p role="status">Loading assistants…</p> : agents.length === 0 ? <div className="panel empty"><h3>Room for your first assistant.</h3><p>Name it, choose an available model, and start a conversation.</p></div> :
        <div className="agent-list">{agents.map(agent => <Link className="panel agent-card" key={agent.id} to={`/agents/${agent.id}`}><div><h3>{agent.name}</h3><small>{agent.model}</small></div><StatusBadge status={agent.paused ? 'paused' : agent.status} /><span aria-hidden="true">↗</span></Link>)}</div>}
    </section><form className="panel stack" onSubmit={create}><h2>Create an assistant</h2><label>Name<input name="name" required maxLength={80} placeholder="Study companion" /></label>
      <label>Model<select name="model" required disabled={!models.length}>{!models.length && <option value="">No verified models available</option>}{models.map(model => <option key={model}>{model}</option>)}</select></label>
      {!models.length && !loading && <p className="notice">No models have passed the current runtime check. Complete the model verification before creating an assistant.</p>}
      <button disabled={busy || !models.length}>{busy ? 'Creating…' : 'Create assistant'}</button>
    </form></div></>;
}
