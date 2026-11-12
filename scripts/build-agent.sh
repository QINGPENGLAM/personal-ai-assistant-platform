#!/bin/sh
set -eu
project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
exec python3 - "$project_dir" "$@" <<'PY'
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tempfile


def docker(*arguments):
    try:
        return subprocess.run(["docker", *arguments], check=True, capture_output=True,
                              text=True, timeout=600 if arguments[0] == "build" else 30).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError("Docker operation failed; verify the daemon and image build prerequisites.") from error


def helper(path, operation):
    with socket.socket(socket.AF_UNIX) as connection:
        connection.settimeout(10)
        connection.connect(path)
        connection.sendall(json.dumps({"operation": operation}).encode() + b"\n")
        data = bytearray()
        while not data.endswith(b"\n") and len(data) <= 4096:
            chunk = connection.recv(4097 - len(data))
            if not chunk:
                break
            data.extend(chunk)
    if len(data) > 4096 or not data.endswith(b"\n") or json.loads(data).get("ok") is not True:
        raise RuntimeError("The quota helper did not confirm filesystem support.")


def context(root, destination):
    files = ["Dockerfile.agent", "Dockerfile.platform", "pyproject.toml", "uv.lock",
             "frontend/package.json", "frontend/package-lock.json", "frontend/tsconfig.json",
             "frontend/vite.config.ts", "frontend/index.html", "scripts/check-runtime.py"]
    for folder, suffixes in [("backend/app", {".py", ".sql"}), ("agent", {".py", ".sh"}),
                             ("frontend/src", {".ts", ".tsx", ".css"})]:
        if (root / folder).is_symlink() or not (root / folder).resolve().is_relative_to(root.resolve()):
            raise RuntimeError("Build source directories must stay inside the checkout.")
        for source in (root / folder).rglob("*"):
            if source.is_symlink():
                raise RuntimeError("Build sources must not contain symbolic links.")
            if source.is_file() and source.suffix in suffixes:
                files.append(source.relative_to(root).as_posix())
    for relative in files:
        source = root / relative
        if (source.is_symlink() or not source.is_file()
                or not source.resolve().is_relative_to(root.resolve())
                or any(parent.is_symlink() for parent in source.parents if parent != root and parent.is_relative_to(root))):
            raise RuntimeError("A required build source is missing or unsafe.")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def main():
    root = Path(sys.argv[1])
    parser = argparse.ArgumentParser(description="Verify the supported Docker/XFS runtime and build source-only images.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--images-only", action="store_true")
    args = parser.parse_args(sys.argv[2:])
    version = docker("version", "--format", "{{.Server.Version}}")
    match = re.match(r"^(\d+)\.", version)
    if match is None or int(match.group(1)) < 26:
        raise RuntimeError("Docker Engine 26 or newer is required.")
    info = json.loads(docker("info", "--format", "{{json .}}"))
    if info.get("OSType") != "linux":
        raise RuntimeError("A Linux Docker builder is required.")
    if not args.images_only:
        if (info.get("Driver") != "overlay2"
                or any("rootless" in str(value) or "userns" in str(value)
                       for value in info.get("SecurityOptions", []))):
            raise RuntimeError("Use the supported rootful Linux Docker/overlay2 runtime.")
        if not args.check_only:
            for volume in ("qpl-platform-state", "qpl-agent-data"):
                docker("volume", "create", volume)
        helper_socket = os.environ.get("QPL_QUOTA_HELPER_SOCKET", "/run/qpl-quota/helper.sock")
        helper(helper_socket, "verify")
        if args.check_only:
            print("Runtime preflight passed; no images or model turns were started.")
            return
        helper(helper_socket, "ensure_platform")
    with tempfile.TemporaryDirectory(prefix="qpl-build-") as temporary:
        staged = Path(temporary)
        context(root, staged)
        for filename, tag in [("Dockerfile.agent", os.environ.get("QPL_AGENT_IMAGE_TAG", "qpl-agent:dev")),
                              ("Dockerfile.platform", os.environ.get("QPL_IMAGE_TAG", "qpl-platform:dev"))]:
            docker("build", "--file", str(staged / filename), "--tag", tag, str(staged))
    print("Agent and platform image builds completed; no model turns were started.")
    if args.images_only:
        print("Images-only mode: runtime/quota acceptance was not performed.")


try:
    main()
except (RuntimeError, OSError, ValueError) as error:
    print(f"Build failed: {error}", file=sys.stderr)
    sys.exit(1)
PY
