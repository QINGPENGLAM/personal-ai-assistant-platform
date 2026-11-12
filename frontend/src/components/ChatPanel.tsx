import { useRef, useState, type FormEvent } from 'react';
import { write } from '../api';
import type { Accepted, Agent, Run } from '../types';
import { ErrorNotice } from './ErrorNotice';
import { RunDetailPanel } from './RunDetailPanel';
export function ChatPanel({ agent, runs, refresh }: { agent: Agent; runs: Run[]; refresh: () => void }) {
  const [prompt, setPrompt] = useState(''); const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null); const [notice, setNotice] = useState('');
  const pending = useRef<{ prompt: string; id: string } | null>(null);
  async function send(event: FormEvent) {
    event.preventDefault(); if (busy || !prompt.trim()) return;
    if (pending.current?.prompt !== prompt) pending.current = { prompt, id: crypto.randomUUID() };
    setBusy(true); setError(null);
    try {
      const accepted = await write<Accepted>(`/agents/${agent.id}/messages`, { prompt }, pending.current!.id);
      setPrompt(''); pending.current = null; setNotice(accepted.paused ? 'Message queued. Continue the assistant when its blockers are resolved.' : 'Message queued.'); refresh();
    } catch (error) { setError(error); } finally { setBusy(false); }
  }
  let epoch: number | undefined;
  return <section className="chat"><div className="section-heading"><h2>Conversation</h2><small>Saved automatically</small></div>
    {!runs.length && <div className="panel empty"><h3>What would you like to work on?</h3><p>Messages and activity stay here when you leave the page.</p></div>}
    {[...runs].reverse().map(run => {
      const boundary = epoch !== run.chat_epoch; epoch = run.chat_epoch;
      return <div key={run.id}>{boundary && <p className="epoch">Conversation {run.chat_epoch + 1}{run.chat_epoch !== agent.chat_epoch ? ' · Archived' : ' · Current'}</p>}
        <RunDetailPanel run={run} archived={run.chat_epoch !== agent.chat_epoch} refresh={refresh} /></div>;
    })}
    <form className="panel composer" onSubmit={send}><label htmlFor="prompt">Message your assistant</label><textarea id="prompt" value={prompt} onChange={event => setPrompt(event.target.value)} required maxLength={16384} rows={4} placeholder="Ask a question or describe a task…" disabled={busy} />
      <ErrorNotice error={error} />{notice && <p role="status">{notice}</p>}<div className="composer-footer"><small>{agent.paused ? 'Paused · new messages will wait' : agent.blockers.length ? 'Blocked · new messages will wait' : 'Tasks run independently of this page'}</small><button disabled={busy || !prompt.trim() || agent.status !== 'ready'}>{busy ? 'Sending…' : 'Send message'}</button></div>
    </form></section>;
}
