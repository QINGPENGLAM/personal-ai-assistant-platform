#!/bin/sh
set -eu

read_secret_file() {
    SECRET_VALUE=
    EXTRA_VALUE=
    if ! exec 3< "$1"; then
        return 1
    fi
    if ! IFS= read -r SECRET_VALUE <&3 && [ -z "$SECRET_VALUE" ]; then
        exec 3<&-
        return 1
    fi
    if IFS= read -r EXTRA_VALUE <&3; then
        exec 3<&-
        return 1
    fi
    exec 3<&-
    [ -n "$SECRET_VALUE" ]
}

if [ "$#" -eq 1 ] && [ "$1" = "--history-only" ]; then
    unset PLATFORM_TOKEN UMGPT_TOOLKIT_API_KEY HTTP_PROXY HTTPS_PROXY
    unset http_proxy https_proxy ALL_PROXY all_proxy NO_PROXY no_proxy
    exec codex -c mcp_servers.platform.enabled=false app-server --listen stdio://
fi

probe_mode=false
if [ "$#" -eq 1 ] && [ "$1" = "--probe" ]; then
    probe_mode=true
elif [ "$#" -ne 0 ]; then
    printf '%s\n' "Unsupported Codex launcher mode." >&2
    exit 64
fi

unset PLATFORM_TOKEN UMGPT_TOOLKIT_API_KEY HTTP_PROXY HTTPS_PROXY
unset http_proxy https_proxy ALL_PROXY all_proxy NO_PROXY no_proxy

if ! read_secret_file /codex/.platform-run-token; then
    printf '%s\n' "A valid run token is required." >&2
    exit 78
fi
PLATFORM_TOKEN=$SECRET_VALUE
unset SECRET_VALUE EXTRA_VALUE
export PLATFORM_TOKEN

if [ -f /run/credentials/UMGPT_TOOLKIT_API_KEY ]; then
    if ! read_secret_file /run/credentials/UMGPT_TOOLKIT_API_KEY; then
        printf '%s\n' "The configured Toolkit credential is invalid." >&2
        exit 78
    fi
    UMGPT_TOOLKIT_API_KEY=$SECRET_VALUE
    unset SECRET_VALUE EXTRA_VALUE
    export UMGPT_TOOLKIT_API_KEY
fi

HTTP_PROXY=http://127.0.0.1:8765
HTTPS_PROXY=http://127.0.0.1:8765
http_proxy=$HTTP_PROXY
https_proxy=$HTTPS_PROXY
NO_PROXY=127.0.0.1,localhost
no_proxy=$NO_PROXY
export HTTP_PROXY HTTPS_PROXY http_proxy https_proxy NO_PROXY no_proxy

if [ "$probe_mode" = true ]; then
    exec codex -c mcp_servers.platform.enabled=false app-server --listen stdio://
fi
exec codex app-server --listen stdio://
