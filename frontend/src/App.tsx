import { createContext, useContext, useEffect, useState } from 'react';
import { Link, Navigate, Route, Routes, useLocation } from 'react-router-dom';
import { api, ApiError, setCsrf, write } from './api';
import type { Session } from './types';
import { AuthPage } from './pages/AuthPage';
import { DashboardPage } from './pages/DashboardPage';
import { AgentPage } from './pages/AgentPage';
import { ErrorNotice } from './components/ErrorNotice';

const Auth = createContext<{ session: Session; logout: () => Promise<void> } | null>(null);
export const useAuth = () => { const value = useContext(Auth); if (!value) throw new Error('Sign in is required.'); return value; };
export function App() {
  const [session, setSession] = useState<Session | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<unknown>(null);
  const location = useLocation();
  const signIn = (value: Session) => { setCsrf(value.csrf_token); setSession(value); setError(null); };
  useEffect(() => {
    const controller = new AbortController();
    const expired = () => { setCsrf(null); setSession(null); };
    window.addEventListener('session-expired', expired);
    api<Session>('/auth/session', { signal: controller.signal }).then(value => {
      if (!controller.signal.aborted) signIn(value);
    }).catch(error => {
      if (!controller.signal.aborted && !(error instanceof ApiError && error.status === 401)) setError(error);
    }).finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => { controller.abort(); window.removeEventListener('session-expired', expired); };
  }, []);
  if (loading) return <main className="auth-shell"><p role="status">Loading your session…</p></main>;
  if (!session) return <><ErrorNotice error={error} /><Routes>
    <Route path="/login" element={<AuthPage onSuccess={signIn} />} />
    <Route path="/register" element={<AuthPage onSuccess={signIn} />} />
    <Route path="*" element={<Navigate to="/login" state={{ from: location.pathname }} replace />} />
  </Routes></>;
  async function logout() { await write('/auth/logout'); setCsrf(null); setSession(null); }
  return <Auth.Provider value={{ session, logout }}><header className="topbar">
    <Link className="brand" to="/agents">Personal Assistants<span>YOUR PRIVATE WORKSPACE</span></Link>
    <div className="account"><span>{session.user.email}</span><button className="secondary" onClick={() => logout().catch(setError)}>Sign out</button></div>
  </header><ErrorNotice error={error} /><main className="shell"><Routes>
    <Route path="/agents" element={<DashboardPage key={session.user.id} />} />
    <Route path="/agents/:id" element={<AgentPage key={`${session.user.id}:${location.pathname}`} />} />
    <Route path="*" element={<Navigate to="/agents" replace />} />
  </Routes></main></Auth.Provider>;
}
