import { ApiError, errorMessage } from '../api';
export function ErrorNotice({ error }: { error: unknown }) {
  if (!error) return null;
  return <div className="notice error" role="alert">{errorMessage(error)}{error instanceof ApiError && error.requestId && <small>Request: {error.requestId}</small>}</div>;
}
