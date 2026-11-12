#!/bin/sh
# Default checks never provision Docker resources or submit model turns.
set -eu
project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$project_dir"
# Reject ambiguous invocations before selecting a mode.
[ "$#" -le 1 ] || { printf '%s\n' 'Unexpected extra arguments.' >&2; exit 2; }
case "${1:-}" in
    --help|-h)
        printf '%s\n' 'Usage: sh scripts/check-project.sh [--real]' \
            'Default: verify installed dependencies, backend, frontend and simulated browser flows.' \
            '--real: currently incomplete; exits 2 without starting resources or model calls.'
        exit 0
        ;;
    --real)
        # Refuse certification until the independent runtime isolation path exists.
        printf '%s\n' 'Incomplete real acceptance: the independent platform test-image runner and real browser suite are not implemented yet.' \
            'Recorded live acceptance is separate evidence; this command cannot certify the full real suite.' >&2
        exit 2
        ;;
    '') ;;
    *) printf '%s\n' 'Usage: sh scripts/check-project.sh [--real]' >&2; exit 2 ;;
esac
[ "$#" -eq 0 ] || { printf '%s\n' 'Unexpected extra arguments.' >&2; exit 2; }
# Missing tools are prerequisites, never an invitation to install or repair automatically.
for executable in uv npm; do
    command -v "$executable" >/dev/null 2>&1 || {
        printf 'Missing prerequisite: %s. Install dependencies using README first.\n' "$executable" >&2
        exit 2
    }
done
# Do not inherit opt-in real mode from the caller into Playwright collection.
RUN_REAL_AGENT_TESTS=0
export RUN_REAL_AGENT_TESTS
uv lock --check --offline
uv sync --frozen --check
npm --prefix frontend ci --dry-run --ignore-scripts --no-audit --no-fund --offline
npm --prefix frontend ls --depth=0
# Installed dependency checks precede all execution; failures stop the pipeline immediately.
uv run --frozen --no-sync ruff check backend scripts
uv run --frozen --no-sync pytest -q -m 'not real_agent'
npm --prefix frontend test
npm --prefix frontend run build
npm --prefix frontend run test:e2e
printf '%s\n' 'Simulated checks passed. Real container/model acceptance is a separate, incomplete suite.'
