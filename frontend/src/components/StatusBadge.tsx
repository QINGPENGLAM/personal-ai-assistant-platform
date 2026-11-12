const labels: Record<string, string> = {
  queued: 'Queued', starting: 'Starting', running: 'Running', stopping: 'Stopping',
  completed: 'Completed', completed_with_errors: 'Completed with errors', interrupted: 'Interrupted',
  failed: 'Failed', cancelled: 'Cancelled', ready: 'Ready', provisioning: 'Preparing',
  provisioning_error: 'Preparation failed', paused: 'Paused',
};
export function StatusBadge({ status }: { status: string }) {
  return <span className={`badge state-${status}`}>{labels[status] ?? 'Unknown'}</span>;
}
