import { useEffect, useState } from 'react';
import { api, ApiError } from '../api';
import { ErrorNotice } from './ErrorNotice';

type Entry = { name: string; path: string; kind: 'file' | 'directory' | 'unsupported'; size: number | null };
type Listing = { path: string; entries: Entry[]; truncated: boolean };
type Preview = { text: string | null; binary: boolean; truncated: boolean };
type Pending = { id: string; path: string; method: 'POST' | 'DELETE'; hash?: string; file?: File };

export function WorkspacePanel({ agentId, refreshAgent }: { agentId: string; refreshAgent: () => void }) {
  const base = `/agents/${agentId}/files`;
  const [path, setPath] = useState('');
  const [direct, setDirect] = useState('');
  const [listing, setListing] = useState<Listing | null>(null);
  const [selected, setSelected] = useState('');
  const [preview, setPreview] = useState<Preview | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [target, setTarget] = useState('');
  const [pending, setPending] = useState<Pending | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);
  const [revision, setRevision] = useState(0);
  const [notice, setNotice] = useState('');
  useEffect(() => {
    const controller = new AbortController();
    setListing(null); setError(null);
    api<Listing>(`${base}?path=${encodeURIComponent(path)}`, { signal: controller.signal })
      .then(value => { if (!controller.signal.aborted) setListing(value); })
      .catch(error => { if (!controller.signal.aborted) setError(error); });
    return () => controller.abort();
  }, [base, path, revision]);

  async function inspect(value: string) {
    setBusy(true); setError(null); setPreview(null); setSelected(value);
    try { setPreview(await api<Preview>(`${base}/preview?path=${encodeURIComponent(value)}`)); }
    catch (error) { setError(error); } finally { setBusy(false); }
  }
  async function execute(operation: Pending) {
    setPending(operation); setBusy(true); setError(null); setNotice('');
    const query = new URLSearchParams({ path: operation.path });
    if (operation.hash) query.set('expected_sha256', operation.hash);
    if (operation.method === 'POST' && operation.hash) query.set('overwrite', 'true');
    const body = operation.file ? new FormData() : undefined;
    if (body && operation.file) body.append('file', operation.file);
    try {
      await api(`${base}?${query}`, { method: operation.method, body, headers: { 'X-Request-ID': operation.id } });
      setPending(null); setPreview(null); setSelected('');
      setNotice(operation.method === 'POST' ? 'File uploaded.' : 'File deleted.');
      setRevision(value => value + 1); refreshAgent();
    } catch (error) {
      setError(error);
      if (error instanceof ApiError && error.code === 'file_operation_uncertain') {
        setNotice('The result is uncertain. Inspect the target before discarding this request and starting a new one.');
      } else if (error instanceof ApiError && error.status < 500 && ![408, 507].includes(error.status)) {
        // A definitive rejection requires a new confirmation; network/server failures retain the snapshot.
        setPending(null);
      }
    } finally { setBusy(false); }
  }
  async function startUpload() {
    if (!file || !target) return;
    setBusy(true); setError(null);
    try {
      let hash: string | undefined;
      try {
        const metadata = await api<{ sha256: string }>(`${base}/metadata?path=${encodeURIComponent(target)}`);
        hash = metadata.sha256;
        if (!window.confirm(`Replace ${target}?\nCurrent SHA-256: ${hash}\nThe current file will be replaced.`)) return;
      } catch (error) { if (!(error instanceof ApiError && error.status === 404)) throw error; }
      await execute({ id: crypto.randomUUID(), path: target, method: 'POST', hash, file });
    } catch (error) { setError(error); } finally { setBusy(false); }
  }
  async function startDelete() {
    setBusy(true); setError(null);
    try {
      const metadata = await api<{ sha256: string }>(`${base}/metadata?path=${encodeURIComponent(selected)}`);
      if (!window.confirm(`Delete ${selected}?\nSHA-256: ${metadata.sha256}\nThis removes one file permanently.`)) return;
      await execute({ id: crypto.randomUUID(), path: selected, method: 'DELETE', hash: metadata.sha256 });
    } catch (error) { setError(error); } finally { setBusy(false); }
  }
  const parts = path.split('/').filter(Boolean);
  return <section className="panel stack" aria-label="Workspace files">
    <h2>Workspace files</h2><p className="muted">Browse private files. Upload and delete are available while the assistant is idle.</p>
    <ErrorNotice error={error} />{notice && <p role="status">{notice}</p>}
    <nav aria-label="Workspace folders"><button className="secondary" disabled={busy} onClick={() => { setPath(''); setSelected(''); setPreview(null); }}>Workspace</button>{parts.map((part, index) => <button className="secondary" key={index} disabled={busy} onClick={() => { setPath(parts.slice(0, index + 1).join('/')); setSelected(''); setPreview(null); }}>{part}</button>)}</nav>
    <label>Direct workspace path<input value={direct} onChange={event => setDirect(event.target.value)} placeholder="folder/notes.txt" /></label>
    <div className="actions"><button className="secondary" disabled={busy} onClick={() => { setPath(direct); setSelected(''); setPreview(null); setRevision(value => value + 1); }}>Open folder</button><button className="secondary" disabled={busy || !direct} onClick={() => inspect(direct)}>Open file</button><button className="secondary" disabled={busy} onClick={() => setRevision(value => value + 1)}>Refresh files</button></div>
    {listing && <><ul>{listing.entries.map(entry => <li key={entry.path}><button className="secondary" disabled={busy || entry.kind === 'unsupported'} onClick={() => entry.kind === 'directory' ? (setPath(entry.path), setSelected(''), setPreview(null)) : inspect(entry.path)}>{entry.name}{entry.kind === 'directory' ? '/' : ''}</button> <small>{entry.kind === 'file' ? `${entry.size} bytes` : entry.kind}</small></li>)}</ul>{listing.entries.length === 0 && <p>No visible files in this folder.</p>}{listing.truncated && <p>Listing limited to 200 entries. Use a direct path to open an omitted entry.</p>}</>}
    {selected && preview && <div className="stack"><h3>{selected}</h3>{preview.binary ? <p>This binary file is available for download.</p> : <pre style={{ whiteSpace: 'pre-wrap', overflowWrap: 'anywhere', maxHeight: '24rem', overflow: 'auto' }}>{preview.text}</pre>}{preview.truncated && <p>Preview limited to 256 KiB.</p>}<a href={`/api${base}/download?path=${encodeURIComponent(selected)}`}>Download file (up to 20 MiB)</a><button className="secondary" disabled={busy || !!pending} onClick={startDelete}>Delete selected file</button></div>}
    <hr /><label>Upload file (up to 10 MiB)<input type="file" disabled={busy || !!pending} onChange={event => { const next = event.target.files?.[0] ?? null; setFile(next); setTarget(next ? `${path ? `${path}/` : ''}${next.name}` : ''); }} /></label>
    <label>Upload destination<input value={target} disabled={busy || !!pending} onChange={event => setTarget(event.target.value)} /></label>
    <button disabled={busy || !!pending || !file || !target} onClick={startUpload}>Upload file</button>
    {pending && <div className="notice"><p>A request is retained with its original file and confirmation. Retrying uses the same request ID.</p><button disabled={busy} onClick={() => execute(pending)}>Retry file request</button><button className="secondary" disabled={busy} onClick={() => { if (window.confirm('Have you inspected the target? Discarding this request allows a new operation with a new request ID.')) setPending(null); }}>Discard inspected request</button></div>}
  </section>;
}
