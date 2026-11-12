let csrf: string | null = null;
let sessionGeneration = 0;
export function setCsrf(value: string | null) {
  if (value !== csrf) sessionGeneration++;
  csrf = value;
}
export class ApiError extends Error {
  constructor(public status: number, public code: string, message: string, public requestId?: string) { super(message); }
}
export async function api<T>(path: string, init: RequestInit = {}, responseType: 'json' | 'response' = 'json'): Promise<T> {
  const generation = sessionGeneration;
  const headers = new Headers(init.headers);
  if (init.body && !(init.body instanceof FormData)) headers.set('Content-Type', 'application/json');
  if (init.method && init.method !== 'GET') {
    if (csrf) headers.set('X-CSRF-Token', csrf);
    if (!headers.has('X-Request-ID')) headers.set('X-Request-ID', crypto.randomUUID());
  }
  const response = await fetch(`/api${path}`, { ...init, headers, credentials: 'same-origin' });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    if (response.status === 401 && generation === sessionGeneration) window.dispatchEvent(new Event('session-expired'));
    throw new ApiError(response.status, body.code ?? 'request_failed', body.message ?? 'The request could not be completed.', body.request_id);
  }
  if (responseType === 'response') return response as T;
  return response.status === 204 ? undefined as T : response.json();
}
export const write = <T>(path: string, body?: unknown, requestId: string = crypto.randomUUID()) => api<T>(path, {
  method: 'POST', headers: { 'X-Request-ID': requestId },
  ...(body === undefined ? {} : { body: JSON.stringify(body) }),
});
export function errorMessage(error: unknown) {
  return error instanceof Error ? error.message : 'The request could not be completed.';
}
