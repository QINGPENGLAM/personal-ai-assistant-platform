#!/bin/sh
set -eu
project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$project_dir"
if [ "$(uname -s)" != Linux ]; then
    printf '%s\n' 'Use the supported Linux/XFS VM for real runtime startup. See README for the fake browser demo.' >&2
    exit 1
fi
sh scripts/build-agent.sh --check-only
for image in "${QPL_AGENT_IMAGE_TAG:-qpl-agent:dev}" "${QPL_IMAGE_TAG:-qpl-platform:dev}"; do
    docker image inspect "$image" >/dev/null
 done
[ -d frontend/node_modules ] || { printf '%s\n' 'Run npm --prefix frontend ci first.' >&2; exit 1; }
socket=${QPL_DOCKER_SOCKET:-/var/run/docker.sock}
helper=${QPL_QUOTA_HELPER_SOCKET:-/run/qpl-quota/helper.sock}
[ -S "$socket" ] && [ -S "$helper" ] || { printf '%s\n' 'Docker and quota-helper sockets are required.' >&2; exit 1; }
# Only this trusted backend receives the daemon socket and its supplemental group.
set -- docker run --detach --rm --name qpl-platform-backend \
    --label qpl.platform=personal-assistant --label qpl.purpose=backend \
    --user 1000:1000 --group-add "$(stat -c %g "$socket")" \
    --publish 127.0.0.1:8000:8000 --cap-drop ALL --security-opt no-new-privileges:true \
    --mount "type=bind,source=$socket,target=/var/run/docker.sock" \
    --mount "type=bind,source=$helper,target=/run/qpl-quota/helper.sock" \
    --mount type=volume,source=qpl-platform-state,target=/platform-state \
    --mount type=volume,source=qpl-agent-data,target=/runtime \
    --tmpfs /upload-staging:rw,noexec,nosuid,nodev,size=64m,mode=0700,uid=1000,gid=1000 \
    --env QPL_ALLOWED_ORIGIN=http://127.0.0.1:5173 \
    --env "QPL_AGENT_IMAGE_TAG=${QPL_AGENT_IMAGE_TAG:-qpl-agent:dev}" \
    --env "QPL_AUTH_MODE=${QPL_AUTH_MODE:-auth_file}"
if [ "${QPL_AUTH_MODE:-auth_file}" = toolkit ]; then
    QPL_TOOLKIT_API_KEY=${QPL_TOOLKIT_API_KEY:-${UMGPT_TOOLKIT_API_KEY:-}}
    export QPL_TOOLKIT_API_KEY
    [ -n "$QPL_TOOLKIT_API_KEY" ] || { printf '%s\n' 'Set QPL_TOOLKIT_API_KEY or UMGPT_TOOLKIT_API_KEY.' >&2; exit 1; }
    set -- "$@" --env QPL_TOOLKIT_API_KEY
else
    [ -n "${QPL_AUTH_FILE_SOURCE:-}" ] || { printf '%s\n' 'Set the absolute Linux host QPL_AUTH_FILE_SOURCE.' >&2; exit 1; }
    case "$QPL_AUTH_FILE_SOURCE" in *,*) printf '%s\n' 'Auth source paths cannot contain commas.' >&2; exit 1 ;; /*) ;; *) printf '%s\n' 'The auth source must be absolute.' >&2; exit 1 ;; esac
    set -- "$@" --mount "type=bind,source=$QPL_AUTH_FILE_SOURCE,target=/run/codex-auth.json,readonly" \
        --env "QPL_AUTH_FILE_SOURCE=$QPL_AUTH_FILE_SOURCE" --env QPL_AUTH_FILE_VALIDATION=/run/codex-auth.json
fi
backend_id=$("$@" "${QPL_IMAGE_TAG:-qpl-platform:dev}")
frontend_pid=
cleanup() {
    if [ -n "$frontend_pid" ]; then kill "$frontend_pid" 2>/dev/null || true; fi
    docker stop --time 90 "$backend_id" >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if ! python3 - <<'PYREADY'
import time
import urllib.request
for _ in range(60):
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=1) as response:
            if response.status == 200:
                break
    except OSError:
        time.sleep(1)
else:
    raise SystemExit("Backend did not become healthy; inspect its Docker logs.")
PYREADY
then exit 1; fi
printf '%s\n' 'Backend is ready. Model verification is an explicit separate command; no probe was started.'
npm --prefix frontend run dev &
frontend_pid=$!
wait "$frontend_pid"
