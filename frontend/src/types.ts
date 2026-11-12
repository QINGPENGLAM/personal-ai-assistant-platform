export interface User { id: string; email: string }
export interface Session { user: User; csrf_token: string }
export interface RunAdmission {
  limited: boolean; retry_at: string | null; minute_count: number; day_count: number;
  minute_limit: number; day_limit: number;
}
export interface Agent {
  storage_protection?: { active: boolean; reason: string | null; measurement_verified: boolean; used_bytes?: number; hard_bytes?: number; free_bytes?: number } | null;
  run_admission?: RunAdmission | null;
  id: string; name: string; model: string;
  status: 'provisioning' | 'provisioning_error' | 'ready';
  paused: boolean; blockers: string[]; container_state: string; chat_epoch: number; created_at: string;
}
export type RunStatus = 'queued' | 'starting' | 'running' | 'stopping' | 'completed' | 'completed_with_errors' | 'failed' | 'interrupted' | 'cancelled';
export interface Run {
  source: 'chat' | 'job';
  id: string; agent_id: string; retry_of: string | null; prompt: string; status: RunStatus;
  model: string; chat_epoch: number; created_at: string; paused: boolean;
  error_code: string | null; error_message: string | null;
  cleanup_state: 'not_required' | 'pending' | 'confirmed' | 'failed';
  last_event_sequence?: number;
}
export type EventKind = 'message_delta' | 'message_snapshot' | 'activity_started' | 'activity_completed' | 'error' | 'state';
export interface RunEvent {
  sequence: number; kind: EventKind; sdk_item_id: string | null;
  payload: Record<string, unknown>; created_at?: string;
}
export interface Accepted { run_id: string; paused: boolean; context_warning?: boolean }
export const terminal = (status: string) => ['completed', 'completed_with_errors', 'failed', 'interrupted', 'cancelled'].includes(status);
