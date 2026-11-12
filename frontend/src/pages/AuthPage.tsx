import { useRef, useState, type FormEvent } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { write } from '../api';
import type { Session } from '../types';
import { ErrorNotice } from '../components/ErrorNotice';
export function AuthPage({ onSuccess }: { onSuccess: (session: Session) => void }) {
  const location = useLocation(); const navigate = useNavigate();
  const register = location.pathname === '/register';
  const [busy, setBusy] = useState(false); const lock = useRef(false);
  const [error, setError] = useState<unknown>(null);
  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault(); if (lock.current) return; lock.current = true; setBusy(true); setError(null);
    const data = new FormData(event.currentTarget);
    try {
      const session = await write<Session>(register ? '/auth/register' : '/auth/login', { email: data.get('email'), password: data.get('password') });
      onSuccess(session);
      const from = location.state?.from;
      navigate(typeof from === 'string' && from.startsWith('/agents') ? from : '/agents', { replace: true });
    } catch (error) { setError(error); } finally { lock.current = false; setBusy(false); }
  }
  return <main className="auth-shell"><div className="eyebrow">PERSONAL ASSISTANTS</div><h1>{register ? 'A workspace of your own.' : 'Welcome back.'}</h1>
    <p className="muted">Your assistants, conversations, and work. All in one place.</p>
    <form className="panel stack" onSubmit={submit}><h2>{register ? 'Create an account' : 'Sign in'}</h2><ErrorNotice error={error} />
      <label>Email<input name="email" type="email" required autoComplete="email" maxLength={254} /></label>
      <label>Password<input name="password" type="password" required minLength={register ? 12 : 1} maxLength={128} autoComplete={register ? 'new-password' : 'current-password'} /></label>
      {register && <small>Use at least 12 characters.</small>}<button disabled={busy}>{busy ? 'Please wait…' : register ? 'Create account' : 'Sign in'}</button>
    </form><p>{register ? 'Already have an account?' : 'New here?'} <Link to={register ? '/login' : '/register'}>{register ? 'Sign in' : 'Create an account'}</Link></p>
  </main>;
}
