from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_probe():
    path = Path(__file__).parents[2] / "scripts/check-runtime.py"
    spec = importlib.util.spec_from_file_location("qpl_runtime_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Session:
    def __init__(self, events):
        self.events = events
        self.prompts = []

    async def prepare_thread(self, model):
        return SimpleNamespace(id="probe-thread")

    async def start_turn(self, thread, model, prompt):
        self.prompts.append(prompt)
        return self

    async def stream(self):
        for method, payload in self.events:
            yield SimpleNamespace(method=method, payload=payload)


def ready_events(text="READY", status="completed"):
    return [
        ("item/completed", {"item": {"type": "agentMessage", "text": text}}),
        ("turn/completed", {"turn": {"status": status}}),
    ]


@pytest.mark.parametrize(
    "events,expected",
    [
        (ready_events(), True),
        (ready_events(" READY\n"), True),
        (ready_events("READY extra"), False),
        (ready_events(status="failed"), False),
        ([("item/started", {"item": {"type": "commandExecution"}})] + ready_events(), False),
        ([("error", {"willRetry": True})] + ready_events(), False),
    ],
)
async def test_probe_accepts_only_exact_ready_without_tools_or_errors(events, expected):
    probe = load_probe()
    session = Session(events)
    assert await probe.ready_turn(session, "gpt-6-luna", deadline=0.1) is expected
    assert session.prompts == ["Reply with exactly READY and do not call tools."]


async def test_probe_attempt_ledger_prevents_same_config_retries_and_crash_replay(tmp_path):
    probe = load_probe()
    ledger = probe.AttemptLedger(tmp_path / "attempts.json")
    assert ledger.previous("binding", "gpt-6-luna") is None
    ledger.begin("binding", "gpt-6-luna")
    reloaded = probe.AttemptLedger(tmp_path / "attempts.json")
    assert reloaded.previous("binding", "gpt-6-luna") == "failed"
    reloaded.finish("binding", "gpt-6-luna", True)
    assert (
        probe.AttemptLedger(tmp_path / "attempts.json").previous(
            "binding",
            "gpt-6-luna",
        )
        == "passed"
    )
    assert reloaded.previous("different-binding", "gpt-6-luna") is None
    assert json.loads((tmp_path / "attempts.json").read_text())["version"] == 2


def test_probe_mounts_only_target_key_report_subpaths(settings):
    from dataclasses import replace
    from uuid import uuid4

    settings = replace(settings, auth_mode="toolkit", toolkit_api_key="test-key")
    probe = load_probe()
    target = str(uuid4())
    instance = str(uuid4())
    arguments = probe.probe_arguments(settings, target, instance, "sha256:" + "a" * 64, 991)
    mounts = [arguments[index + 1] for index, item in enumerate(arguments) if item == "--mount"]
    target_mounts = [mount for mount in mounts if "source=qpl-platform-test," in mount]
    assert target_mounts == [
        f"type=volume,source=qpl-platform-test,target=/target-key,volume-subpath=probe-share/{target}/key,readonly",
        f"type=volume,source=qpl-platform-test,target=/target-report,volume-subpath=probe-share/{target}/report",
    ]
    assert all("source=qpl-agent-test," not in mount for mount in mounts)
    assert "QPL_TOOLKIT_API_KEY" in arguments


async def test_published_probe_keeps_original_evidence_time_and_does_not_retry(settings, tmp_path):
    from dataclasses import replace

    from backend.app.recovery import VerifiedModelReports

    probe = load_probe()
    store = VerifiedModelReports(replace(settings, auth_mode="toolkit", toolkit_api_key="test-key"))
    ledger = probe.AttemptLedger(tmp_path / "attempts.json")
    calls = []

    async def run(model):
        calls.append(model)
        assert ledger.previous(store._configuration_binding(b"k" * 32), model) == "failed"
        return model == "gpt-6-luna"

    destination = tmp_path / "report.json"
    runtime = SimpleNamespace(probe=run)
    await probe.publish_probe(runtime, store, b"k" * 32, ledger, destination)
    original = json.loads(destination.read_text())
    await probe.publish_probe(runtime, store, b"k" * 32, ledger, destination)
    assert calls == ["gpt-6.1-sol", "gpt-6-luna"]
    assert json.loads(destination.read_text())["generated_at"] == original["generated_at"]
    assert original["candidates"] == {"gpt-6.1-sol": "failed", "gpt-6-luna": "passed"}
    assert "test-key" not in destination.read_text()


async def test_changed_credentials_or_failed_cleanup_never_replace_report(settings, tmp_path):
    from dataclasses import replace

    from backend.app.recovery import VerifiedModelReports

    probe = load_probe()
    store = VerifiedModelReports(replace(settings, auth_mode="toolkit", toolkit_api_key="test-key"))
    destination = tmp_path / "report.json"
    destination.write_bytes(b"previous verified report")

    async def changed(model):
        store.settings = replace(store.settings, toolkit_api_key="rotated-key")
        return True

    with pytest.raises(ValueError, match="configuration changed"):
        await probe.publish_probe(
            SimpleNamespace(probe=changed),
            store,
            b"k" * 32,
            probe.AttemptLedger(tmp_path / "attempts.json"),
            destination,
        )
    assert destination.read_bytes() == b"previous verified report"

    async def unclean(model):
        raise probe.CleanupError("Unconfirmed stop")

    with pytest.raises(probe.CleanupError):
        await probe.publish_probe(
            SimpleNamespace(probe=unclean),
            store,
            b"k" * 32,
            probe.AttemptLedger(tmp_path / "other-attempts.json"),
            destination,
        )
    assert destination.read_bytes() == b"previous verified report"


async def test_probe_revokes_tokens_and_confirms_container_and_sdk_cleanup(settings):
    from uuid import uuid4

    from backend.app.network_gateway import GatewayError, TokenAuthority

    probe = load_probe()
    instance = str(uuid4())
    authority = TokenAuthority(instance, settings.allowed_model_hosts)
    session = Session(ready_events())
    closed = []
    stopped = []
    tokens = []

    async def close():
        closed.append(True)

    session.close = close

    async def connect(agent_id, *, probe_only):
        assert probe_only is True
        return session

    async def inventory():
        return []

    async def start(agent_id):
        return None

    async def stop(agent_id):
        stopped.append(agent_id)

    async def docker(*args):
        return "codex-cli 0.166.0"

    def token(agent_id, value):
        assert authority.resolve(value, "connect")[1].scope == "probe"
        with pytest.raises(GatewayError):
            authority.resolve(value, "mcp")
        tokens.append(value)

    containers = SimpleNamespace(
        running_agent_ids=inventory,
        start_agent=start,
        stop_agent=stop,
        _docker=docker,
        _ensure_agent_directories=lambda agent_id: None,
        _write_agent_configuration=lambda agent: None,
        write_run_token=token,
        remove_run_token=lambda agent_id: None,
        container_name=lambda agent_id: f"qpl-agent-{instance}-{agent_id}",
    )
    runtime = probe.ProbeRuntime(
        settings,
        instance,
        containers,
        SimpleNamespace(authority=authority),
        SimpleNamespace(connect=connect),
    )
    assert await runtime.probe("gpt-6-luna") is True
    assert len(stopped) == len(closed) == 1
    with pytest.raises(GatewayError):
        authority.resolve(tokens[0], "connect")

    async def failed_stop(agent_id):
        raise RuntimeError("Unconfirmed container stop")

    containers.stop_agent = failed_stop
    with pytest.raises(probe.CleanupError):
        await runtime.probe("gpt-6-luna")
    assert len(closed) == 2
    with pytest.raises(GatewayError):
        authority.resolve(tokens[-1], "connect")


def load_build_functions():
    import ast

    script = (Path(__file__).parents[2] / "scripts/build-agent.sh").read_text()
    source = script.split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    tree = ast.parse(source)
    # Execute the actual helpers without the shell entry point or Docker side effects.
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))
    ]
    scope = {}
    exec(compile(tree, "build-agent.sh", "exec"), scope)
    return scope


def build_fixture(tmp_path):
    root = tmp_path / "source"
    for relative in [
        "Dockerfile.agent",
        "Dockerfile.platform",
        "pyproject.toml",
        "uv.lock",
        "frontend/package.json",
        "frontend/package-lock.json",
        "frontend/tsconfig.json",
        "frontend/vite.config.ts",
        "frontend/index.html",
        "scripts/check-runtime.py",
        "backend/app/main.py",
        "agent/launch-codex.sh",
        "frontend/src/App.tsx",
    ]:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("safe source")
    return root


def test_build_context_excludes_runtime_credentials_and_dependencies(tmp_path):
    root = build_fixture(tmp_path)
    for relative in [
        ".env",
        ".git/config",
        "runtime/credentials/auth.json",
        ".venv/auth.py",
        "frontend/node_modules/private.ts",
        "frontend/.env",
    ]:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("secret fixture")
    staged = tmp_path / "staged"
    staged.mkdir()
    load_build_functions()["context"](root, staged)
    assert (staged / "frontend/src/App.tsx").read_text() == "safe source"
    assert all(
        "secret fixture" not in path.read_text() for path in staged.rglob("*") if path.is_file()
    )


def test_build_context_rejects_symlinked_source_directories(tmp_path):
    root = build_fixture(tmp_path)
    (root / "frontend/src/App.tsx").unlink()
    (root / "frontend/src").rmdir()
    outside = tmp_path / "private"
    outside.mkdir()
    (outside / "key.ts").write_text("private credential fixture")
    (root / "frontend/src").symlink_to(outside, target_is_directory=True)
    staged = tmp_path / "staged"
    staged.mkdir()
    with pytest.raises(RuntimeError):
        load_build_functions()["context"](root, staged)


def test_probe_configuration_round_trip_preserves_each_attempt_window(tmp_path):
    probe = load_probe()
    path = tmp_path / "attempts.json"
    ledger = probe.AttemptLedger(path)
    ledger.begin("config-A", "gpt-6-luna")
    ledger.finish("config-A", "gpt-6-luna", False)
    ledger.begin("config-B", "gpt-6-luna")
    ledger.finish("config-B", "gpt-6-luna", True)
    reopened = probe.AttemptLedger(path)
    assert reopened.previous("config-A", "gpt-6-luna") == "failed"
    assert reopened.previous("config-B", "gpt-6-luna") == "passed"


def test_probe_config_capacity_never_evicts_a_recent_attempt(tmp_path, monkeypatch):
    probe = load_probe()
    ledger = probe.AttemptLedger(tmp_path / "attempts.json")
    now = ledger.now()
    for index in range(16):
        ledger.begin(f"config-{index}", "gpt-6-luna")
    with pytest.raises(ValueError, match="capacity"):
        ledger.begin("another-config", "gpt-6-luna")
    assert ledger.previous("config-0", "gpt-6-luna") == "failed"
    monkeypatch.setattr(probe.AttemptLedger, "now", staticmethod(lambda: now + probe.DAY_MS + 1000))
    ledger.begin("another-config", "gpt-6-luna")
    assert ledger.previous("config-0", "gpt-6-luna") is None


def test_host_never_deletes_an_unverified_smoke_volume(settings, monkeypatch):
    from uuid import uuid4

    probe = load_probe()
    target = str(uuid4())
    commands = []

    def docker(*args, **kwargs):
        commands.append(args)
        if args[0] == "version":
            return "26.0.0"
        if args[0] == "info":
            return json.dumps({"OSType": "linux", "Driver": "overlay2"})
        if args[:2] == ("image", "inspect"):
            return "sha256:" + "a" * 64
        if args[:2] == ("volume", "inspect"):
            if "qpl-probe-state-" in args[2]:
                return json.dumps(
                    [
                        {
                            "Driver": "local",
                            "Labels": {
                                "qpl.purpose": "probe-state",
                                "qpl.target": target,
                            },
                        }
                    ]
                )
            return json.dumps([{"Driver": "local", "Labels": {"qpl.instance": "another-owner"}}])
        return ""

    class Manager:
        def __init__(self, *args):
            pass

        async def _docker(self, *args):
            return ""

    monkeypatch.setattr(probe, "docker", docker)
    monkeypatch.setattr(probe, "ContainerManager", Manager)
    with pytest.raises(ValueError):
        probe.host(settings, target)
    assert not any(command[:2] == ("volume", "rm") for command in commands)


def test_images_only_build_skips_runtime_provisioning(tmp_path, monkeypatch):
    scope = load_build_functions()
    root = build_fixture(tmp_path)
    commands = []

    def docker(*args):
        commands.append(args)
        if args[0] == "version":
            return "29.4.0"
        if args[0] == "info":
            return json.dumps({"OSType": "linux", "Driver": "overlayfs"})
        return ""

    def forbidden(*args):
        raise AssertionError("Image-only validation must not provision runtime resources.")

    scope["docker"], scope["helper"] = docker, forbidden
    monkeypatch.setattr(scope["sys"], "argv", ["build-agent.sh", str(root), "--images-only"])
    scope["main"]()
    assert sum(command[0] == "build" for command in commands) == 2
    assert all(command[0] != "volume" for command in commands)


@pytest.mark.parametrize("arguments", [[], ["--check-only"]])
def test_normal_build_retains_supported_driver_gate(tmp_path, monkeypatch, arguments):
    scope = load_build_functions()
    commands = []

    def docker(*args):
        commands.append(args)
        return (
            "29.4.0"
            if args[0] == "version"
            else json.dumps({"OSType": "linux", "Driver": "overlayfs"})
        )

    scope["docker"] = docker
    monkeypatch.setattr(scope["sys"], "argv", ["build-agent.sh", str(tmp_path), *arguments])
    with pytest.raises(RuntimeError, match="supported"):
        scope["main"]()
    assert not any(command[0] in {"volume", "build"} for command in commands)


def test_build_modes_are_mutually_exclusive(tmp_path, monkeypatch):
    scope = load_build_functions()
    commands = []
    scope["docker"] = lambda *args: commands.append(args)
    monkeypatch.setattr(
        scope["sys"], "argv", ["build-agent.sh", str(tmp_path), "--check-only", "--images-only"]
    )
    with pytest.raises(SystemExit) as error:
        scope["main"]()
    assert error.value.code == 2 and commands == []


def test_images_only_refuses_non_linux_builder(tmp_path, monkeypatch):
    scope = load_build_functions()
    commands = []

    def docker(*args):
        commands.append(args)
        return "29.4.0" if args[0] == "version" else json.dumps({"OSType": "windows"})

    scope["docker"] = docker
    monkeypatch.setattr(scope["sys"], "argv", ["build-agent.sh", str(tmp_path), "--images-only"])
    with pytest.raises(RuntimeError, match="Linux"):
        scope["main"]()
    assert not any(command[0] in {"volume", "build"} for command in commands)


# Stub external tools so CLI safety/failure checks cannot install packages or run models.
def _project_check_tools(tmp_path, monkeypatch):
    import os

    tools = tmp_path / "tools"
    tools.mkdir()
    log = tmp_path / "commands.log"
    stub = '''#!/bin/sh
name=${0##*/}
line="$name $*"
printf '%s real=%s\n' "$line" "${RUN_REAL_AGENT_TESTS:-unset}" >> "$QPL_CHECK_LOG"
if [ "$line" = "${QPL_CHECK_FAIL:-}" ]; then exit 17; fi
'''
    # One disposable executable per tool records argv instead of executing the real tool.
    for name in ("uv", "npm"):
        executable = tools / name
        executable.write_text(stub)
        executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(tools) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("QPL_CHECK_LOG", str(log))
    return log


# An inherited real opt-in must never turn a default project check into model execution.
def test_project_check_defaults_to_fake_and_runs_all_stages(tmp_path, monkeypatch):
    import subprocess

    log = _project_check_tools(tmp_path, monkeypatch)
    monkeypatch.setenv("RUN_REAL_AGENT_TESTS", "1")
    script = Path(__file__).parents[2] / "scripts/check-project.sh"
    completed = subprocess.run(["sh", str(script)], cwd=tmp_path, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    commands = log.read_text().splitlines()
    # Verify the full ordered boundary, including deliberate reset of inherited real mode.
    assert commands == [
        "uv lock --check --offline real=0",
        "uv sync --frozen --check real=0",
        "npm --prefix frontend ci --dry-run --ignore-scripts --no-audit --no-fund --offline real=0",
        "npm --prefix frontend ls --depth=0 real=0",
        "uv run --frozen --no-sync ruff check backend scripts real=0",
        "uv run --frozen --no-sync pytest -q -m not real_agent real=0",
        "npm --prefix frontend test real=0",
        "npm --prefix frontend run build real=0",
        "npm --prefix frontend run test:e2e real=0",
    ]
    assert "Simulated checks passed" in completed.stdout


# Early failure must prevent later stages and the success banner, not just return nonzero.
@pytest.mark.parametrize("failed_stage", [
    "uv lock --check --offline",
    "uv run --frozen --no-sync pytest -q -m not real_agent",
    "npm --prefix frontend test",
    "npm --prefix frontend run build",
    "npm --prefix frontend run test:e2e",
])
def test_project_check_preserves_failure_and_does_not_continue(tmp_path, monkeypatch, failed_stage):
    import subprocess

    log = _project_check_tools(tmp_path, monkeypatch)
    monkeypatch.setenv("QPL_CHECK_FAIL", failed_stage)
    script = Path(__file__).parents[2] / "scripts/check-project.sh"
    completed = subprocess.run(["sh", str(script)], capture_output=True, text=True)
    assert completed.returncode == 17
    assert log.read_text().splitlines()[-1] == failed_stage + " real=0"
    assert "Simulated checks passed" not in completed.stdout


# Refusing an incomplete real suite must happen before invoking any external tool.
def test_project_check_real_is_incomplete_without_independent_runner(tmp_path, monkeypatch):
    import subprocess

    log = _project_check_tools(tmp_path, monkeypatch)
    script = Path(__file__).parents[2] / "scripts/check-project.sh"
    completed = subprocess.run(["sh", str(script), "--real"], capture_output=True, text=True)
    assert completed.returncode == 2
    assert "Incomplete real acceptance" in completed.stderr
    assert not log.exists()


# Help and invalid modes are parse-only paths with no verification side effects.
@pytest.mark.parametrize("arguments,expected", [
    (["--help"], 0), (["--unknown"], 2), (["--help", "extra"], 2),
])
def test_project_check_cli_modes_do_not_run_checks(tmp_path, monkeypatch, arguments, expected):
    import subprocess

    log = _project_check_tools(tmp_path, monkeypatch)
    script = Path(__file__).parents[2] / "scripts/check-project.sh"
    completed = subprocess.run(["sh", str(script), *arguments], capture_output=True, text=True)
    assert completed.returncode == expected
    assert not log.exists()


def load_browser_server():
    path = Path(__file__).parents[2] / "scripts/serve-test.py"
    spec = importlib.util.spec_from_file_location("qpl_browser_server", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("runtime_argument", [False, True])
def test_real_browser_entry_never_opens_configured_product_runtime(
    settings,
    tmp_path,
    monkeypatch,
    capsys,
    runtime_argument,
):
    import sys

    server = load_browser_server()
    accesses = []
    monkeypatch.setattr(
        server,
        "browser_settings",
        lambda: accesses.append("settings") or settings,
    )
    monkeypatch.setattr(server, "test_app", lambda config: accesses.append("app"))
    monkeypatch.setattr(server.uvicorn, "run", lambda *args, **kwargs: accesses.append("listen"))
    arguments = ["serve-test.py", "--mode", "real"]
    if runtime_argument:
        arguments.extend(["--runtime", str(tmp_path / "preview")])
    monkeypatch.setattr(sys, "argv", arguments)
    with pytest.raises(SystemExit) as stopped:
        server.main()
    assert stopped.value.code == 2
    assert accesses == []
    assert "independent test" in capsys.readouterr().err


@pytest.mark.parametrize("change", [None, "purpose", "session", "extra", "owner", "writable"])
def test_independent_host_marker_is_strict(tmp_path, monkeypatch, change):
    import stat

    server = load_browser_server()
    marker = tmp_path / "host.json"
    content = {
        "project": "hackathon-2-qpl",
        "purpose": "isolated-acceptance",
        "session_id": "570243d3-f325-4a44-9785-621e39127639",
    }
    if change == "purpose":
        content["purpose"] = "preview"
    elif change == "session":
        content["session_id"] = "not-a-uuid"
    elif change == "extra":
        content["allow_product"] = True
    marker.write_text(json.dumps(content))
    monkeypatch.setattr(server, "TEST_HOST_MARKER", marker)
    metadata = SimpleNamespace(
        st_mode=stat.S_IFREG | (0o666 if change == "writable" else 0o644),
        st_uid=1000 if change == "owner" else 0,
        st_size=marker.stat().st_size,
    )
    monkeypatch.setattr(server.os, "fstat", lambda descriptor: metadata)
    if change is None:
        assert server.independent_test_session() == content["session_id"]
    else:
        with pytest.raises(RuntimeError, match="independent test host marker"):
            server.independent_test_session()


def test_independent_host_marker_rejects_symlink(tmp_path, monkeypatch):
    server = load_browser_server()
    target = tmp_path / "target"
    target.write_text("{}")
    marker = tmp_path / "host.json"
    marker.symlink_to(target)
    monkeypatch.setattr(server, "TEST_HOST_MARKER", marker)
    with pytest.raises(RuntimeError, match="independent test host marker"):
        server.independent_test_session()


@pytest.mark.parametrize(
    "change",
    [None, "preview", "other_session", "duplicate", "remote_storage"],
)
def test_independent_volume_identity_is_required(monkeypatch, change):
    server = load_browser_server()
    session = "570243d3-f325-4a44-9785-621e39127639"
    volumes = [
        {
            "Name": name,
            "Driver": "local",
            "Scope": "local",
            "Options": None,
            "Mountpoint": f"/var/lib/docker/volumes/{name}/_data",
            "Labels": {"qpl.purpose": "isolated-acceptance", "qpl.test-session": session},
        }
        for name in ("qpl-platform-state", "qpl-agent-data")
    ]
    if change == "preview":
        volumes[0]["Labels"] = {}
    elif change == "other_session":
        volumes[0]["Labels"]["qpl.test-session"] = "other-session"
    elif change == "duplicate":
        volumes[1] = volumes[0]
    elif change == "remote_storage":
        volumes[0]["Options"] = {"type": "nfs"}
    monkeypatch.setattr(server, "test_docker", lambda *args: json.dumps(volumes))
    if change is None:
        server.verify_test_volumes(session)
    else:
        with pytest.raises(RuntimeError, match="independent test volumes"):
            server.verify_test_volumes(session)


@pytest.mark.parametrize("change", [None, "desktop", "driver", "root", "old", "rootless", "userns"])
def test_independent_preflight_requires_local_supported_daemon(monkeypatch, change):
    server = load_browser_server()
    session = "570243d3-f325-4a44-9785-621e39127639"
    info = {
        "OSType": "linux",
        "Driver": "overlay2",
        "DockerRootDir": "/var/lib/docker",
        "ServerVersion": "29.1.3",
        "SecurityOptions": ["name=seccomp"],
    }
    if change == "desktop":
        info["OSType"] = "windows"
    elif change == "driver":
        info["Driver"] = "overlayfs"
    elif change == "root":
        info["DockerRootDir"] = "/other/docker"
    elif change == "old":
        info["ServerVersion"] = "25.0.0"
    elif change in {"rootless", "userns"}:
        info["SecurityOptions"] = ["name=" + change]
    checked = []
    monkeypatch.setattr(server, "independent_test_session", lambda: session)
    monkeypatch.setattr(server, "test_docker", lambda *args: json.dumps(info))
    monkeypatch.setattr(server, "verify_test_volumes", lambda value: checked.append(value))
    monkeypatch.setattr(server, "verify_test_quota", lambda: checked.append("quota"))
    if change is None:
        assert server.real_test_preflight() == session
        assert checked == [session, "quota"]
    else:
        with pytest.raises(RuntimeError, match="rootful Linux"):
            server.real_test_preflight()
        assert checked == []


def test_test_docker_ignores_remote_context(monkeypatch):
    server = load_browser_server()
    calls = []
    monkeypatch.setenv("DOCKER_HOST", "tcp://other.invalid:2375")
    monkeypatch.setenv("DOCKER_CONTEXT", "preview")
    monkeypatch.setenv("DOCKER_TLS_VERIFY", "1")
    monkeypatch.setenv("DOCKER_CERT_PATH", "/other/certs")
    monkeypatch.setattr(
        server.subprocess,
        "run",
        lambda arguments, **options: (
            calls.append((arguments, options)) or SimpleNamespace(stdout="{}")
        ),
    )
    assert server.test_docker("info") == "{}"
    arguments, options = calls[0]
    assert arguments == ["docker", "--host", "unix:///var/run/docker.sock", "info"]
    assert all(
        name not in options["env"]
        for name in (
            "DOCKER_HOST",
            "DOCKER_CONTEXT",
            "DOCKER_TLS_VERIFY",
            "DOCKER_CERT_PATH",
        )
    )
    assert options["timeout"] == 30 and options["check"] is True


@pytest.mark.parametrize("change", [None, "refused", "hard_limit", "low_space", "boolean"])
def test_independent_quota_preflight_requires_measured_limits(monkeypatch, change):
    server = load_browser_server()
    measurement = {
        "ok": True,
        "free_bytes": 4 * 1024**3,
        "quota": {
            "used_bytes": 4096,
            "hard_bytes": 2 * 1024**3,
            "used_inodes": 3,
            "hard_inodes": 100000,
        },
    }
    if change == "refused":
        measurement["ok"] = False
    elif change == "hard_limit":
        measurement["quota"]["hard_bytes"] = 4 * 1024**3
    elif change == "low_space":
        measurement["free_bytes"] = 1024**2
    elif change == "boolean":
        measurement["quota"]["used_bytes"] = True
    operations = []
    monkeypatch.setattr(
        server,
        "test_quota_request",
        lambda operation: (
            operations.append(operation) or ({"ok": True} if operation == "verify" else measurement)
        ),
    )
    if change is None:
        server.verify_test_quota()
        assert operations == ["verify", "verify_platform"]
    else:
        with pytest.raises(RuntimeError, match="quota"):
            server.verify_test_quota()


@pytest.mark.parametrize("case", ["fragmented", "refused", "truncated", "oversized", "timeout"])
def test_test_quota_transport_bounds_and_closes(monkeypatch, case):
    server = load_browser_server()

    class Connection:
        closed = False
        sent = None

        def __init__(self):
            self.chunks = [b'{"ok":', b"true}\n"]
            if case == "refused":
                self.chunks = [b'{"ok":false}\n']
            elif case == "truncated":
                self.chunks = [b'{"ok":true}']
            elif case == "oversized":
                self.chunks = [b"x" * 4097]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.closed = True

        def settimeout(self, seconds):
            assert 0 < seconds <= 10

        def connect(self, path):
            assert path == "/run/qpl-quota/helper.sock"

        def sendall(self, content):
            self.sent = content

        def recv(self, maximum):
            assert maximum <= 4097
            if case == "timeout":
                raise TimeoutError
            return self.chunks.pop(0) if self.chunks else b""

    connection = Connection()
    monkeypatch.setattr(server.socket, "socket", lambda *args: connection)
    if case == "fragmented":
        assert server.test_quota_request("verify") == {"ok": True}
    else:
        with pytest.raises(RuntimeError, match="quota verification"):
            server.test_quota_request("verify")
    assert connection.closed
    assert json.loads(connection.sent) == {"operation": "verify"}


@pytest.mark.parametrize("explicit_runtime", [False, True])
def test_fake_browser_service_keeps_its_own_runtime(
    settings,
    tmp_path,
    monkeypatch,
    explicit_runtime,
):
    import sys

    server = load_browser_server()
    root = tmp_path / "fake-test"
    calls = []
    monkeypatch.setattr(server, "browser_settings", lambda: settings)
    monkeypatch.setattr(server.tempfile, "mkdtemp", lambda **kwargs: str(root))
    monkeypatch.setattr(
        server,
        "test_app",
        lambda configured: calls.append(configured) or "fake-app",
    )
    monkeypatch.setattr(server.uvicorn, "run", lambda app, **kwargs: calls.append((app, kwargs)))
    arguments = ["serve-test.py", "--mode", "fake", "--port", "8001", "--frontend-port", "5174"]
    if explicit_runtime:
        arguments.extend(["--runtime", str(root)])
    monkeypatch.setattr(sys, "argv", arguments)
    server.main()
    configured, (app, options) = calls
    assert configured.database_path == root / "platform.sqlite"
    assert configured.agent_data_root == root / "agent-data"
    assert configured.database_path != settings.database_path
    assert configured.allowed_origin == "http://127.0.0.1:5174"
    assert app == "fake-app" and options["host"] == "127.0.0.1" and options["port"] == 8001


def test_real_check_only_does_not_read_settings_or_start_service(monkeypatch, capsys):
    import sys

    server = load_browser_server()
    monkeypatch.setattr(server, "real_test_preflight", lambda: "verified-session")
    monkeypatch.setattr(
        server,
        "browser_settings",
        lambda: pytest.fail("product config accessed"),
    )
    monkeypatch.setattr(
        server.uvicorn,
        "run",
        lambda *args, **kwargs: pytest.fail("service started"),
    )
    monkeypatch.setattr(sys, "argv", ["serve-test.py", "--mode", "real", "--check-only"])
    server.main()
    assert "no product or model was started" in capsys.readouterr().out


def test_quota_preflight_has_an_overall_request_deadline(monkeypatch):
    server = load_browser_server()
    clock = [0.0]
    monkeypatch.setattr(server, "time", SimpleNamespace(monotonic=lambda: clock[0]), raising=False)

    class SlowConnection:
        closed = False
        reads = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.closed = True

        def settimeout(self, seconds):
            assert 0 < seconds <= 10

        def connect(self, path):
            pass

        def sendall(self, content):
            pass

        def recv(self, maximum):
            self.reads += 1
            clock[0] += 11
            return b'{"ok":' if self.reads == 1 else b"true}\n"

    connection = SlowConnection()
    monkeypatch.setattr(server.socket, "socket", lambda *args: connection)
    with pytest.raises(RuntimeError, match="quota verification"):
        server.test_quota_request("verify")
    assert connection.reads == 1 and connection.closed


def test_browser_entry_import_does_not_load_product_configuration(monkeypatch):
    import sys

    # A cold import must not evaluate even invalid product configuration defaults.
    monkeypatch.delitem(sys.modules, "backend.app.main", raising=False)
    monkeypatch.delitem(sys.modules, "backend.app.config", raising=False)
    monkeypatch.setenv("QPL_COOKIE_SECURE", "invalid")
    load_browser_server()


def real_backend_controller(server):
    return server.RealTestBackend(
        "570243d3-f325-4a44-9785-621e39127639",
        "sha256:" + "1" * 64,
        "sha256:" + "2" * 64,
        Path("/home/ubuntu/test-auth.json"),
        999,
        8001,
        5174,
    )


def test_real_backend_arguments_are_private_and_session_scoped():
    server = load_browser_server()
    backend = real_backend_controller(server)
    arguments = backend.arguments()
    assert arguments[:4] == ["container", "create", "--name", backend.name]
    assert "127.0.0.1:8001:8000" in arguments
    assert "1000:1000" in arguments
    assert "qpl.purpose=real-test-backend" in arguments
    assert "qpl.test-session=" + backend.session in arguments
    assert "QPL_ALLOWED_ORIGIN=http://127.0.0.1:5174" in arguments
    assert "QPL_AGENT_IMAGE_TAG=sha256:" + "2" * 64 in arguments
    assert arguments[-1] == "sha256:" + "1" * 64
    mounts = [arguments[i + 1] for i, item in enumerate(arguments) if item == "--mount"]
    assert len(mounts) == 5
    assert any("source=qpl-platform-state," in item for item in mounts)
    assert any("source=qpl-agent-data," in item for item in mounts)
    assert any(
        "source=/home/ubuntu/test-auth.json," in item and "readonly" in item for item in mounts
    )
    assert "--privileged" not in arguments


def test_real_backend_will_not_control_a_foreign_container(monkeypatch):
    server = load_browser_server()
    backend = real_backend_controller(server)
    operations = []

    def docker(*arguments, **options):
        operations.append(arguments)
        if arguments[:2] == ("container", "ls"):
            return "" if any("qpl-real-cleanup-" in item for item in arguments) else backend.name
        if arguments[:2] == ("container", "inspect"):
            return json.dumps({"labels": {"qpl.purpose": "preview"}})
        pytest.fail("Foreign container control was attempted")

    monkeypatch.setattr(server, "test_docker", docker)
    with pytest.raises(RuntimeError, match="identity"):
        backend.start()
    assert all(operation[1] not in {"start", "stop", "kill", "rm"} for operation in operations)


def owned_backend_metadata(backend):
    return {
        "labels": {"qpl.platform": "personal-assistant", "qpl.purpose": "real-test-backend",
                   "qpl.test-session": backend.session},
        "image": backend.platform_image, "user": "1000:1000", "running": False,
        "host": {"Privileged": False, "NetworkMode": "bridge", "CapDrop": ["ALL"],
                 "SecurityOpt": ["no-new-privileges:true"],
                 "Tmpfs": {
                     "/upload-staging": (
                         "rw,noexec,nosuid,nodev,size=64m,mode=0700,uid=1000,gid=1000"
                     ),
                 },
                 "PortBindings": {"8000/tcp": [{"HostIp": "127.0.0.1", "HostPort": "8001"}]}},
        "mounts": [
            {"Type": "volume", "Name": "qpl-platform-state",
             "Source": "/var/lib/docker/volumes/qpl-platform-state/_data",
             "Destination": "/platform-state", "RW": True},
            {"Type": "volume", "Name": "qpl-agent-data",
             "Source": "/var/lib/docker/volumes/qpl-agent-data/_data",
             "Destination": "/runtime", "RW": True},
            {"Type": "bind", "Source": "/var/run/docker.sock",
             "Destination": "/var/run/docker.sock", "RW": True},
            {"Type": "bind", "Source": "/run/qpl-quota/helper.sock",
             "Destination": "/run/qpl-quota/helper.sock", "RW": True},
            {"Type": "bind", "Source": "/home/ubuntu/test-auth.json",
             "Destination": "/run/codex-auth.json", "RW": False},
        ],
    }


@pytest.mark.parametrize(
    "change", [None, "image", "auth_rw", "extra_mount", "public_port", "privileged", "tmpfs"],
)
def test_real_backend_reuse_revalidates_isolation(monkeypatch, change):
    server = load_browser_server()
    backend = real_backend_controller(server)
    metadata = owned_backend_metadata(backend)
    if change == "image":
        metadata["image"] = "sha256:" + "3" * 64
    elif change == "auth_rw":
        metadata["mounts"][-1]["RW"] = True
    elif change == "extra_mount":
        metadata["mounts"].append(
            {"Type": "bind", "Source": "/", "Destination": "/host", "RW": True},
        )
    elif change == "public_port":
        metadata["host"]["PortBindings"]["8000/tcp"][0]["HostIp"] = "0.0.0.0"
    elif change == "privileged":
        metadata["host"]["Privileged"] = True
    elif change == "tmpfs":
        metadata["host"]["Tmpfs"] = {}
    calls = []

    def docker(*arguments, **options):
        calls.append(arguments)
        if arguments[:2] == ("container", "ls"):
            return "" if any("qpl-real-cleanup-" in item for item in arguments) else backend.name
        if arguments[:2] == ("container", "inspect"):
            return json.dumps(metadata)
        assert arguments[:2] == ("container", "start")
        return backend.name

    monkeypatch.setattr(server, "test_docker", docker)
    if change is None:
        backend.start()
        assert calls[-1] == ("container", "start", backend.name)
    else:
        with pytest.raises(RuntimeError, match="identity"):
            backend.start()
        assert all(call[1] not in {"start", "stop", "kill", "rm"} for call in calls)


def test_real_cleanup_runs_without_health_identity_and_without_host_credentials(monkeypatch):
    server = load_browser_server()
    backend = real_backend_controller(server)
    calls = []

    def docker(*arguments, **options):
        calls.append((arguments, options))
        return "" if arguments[:2] == ("container", "ls") else "done"

    monkeypatch.setattr(server, "test_docker", docker)
    backend.stop_agents()
    arguments, options = calls[0]
    assert arguments[:2] == ("run", "--name")
    assert arguments[-1] == "-"
    assert "qpl.purpose=real-test-cleanup" in arguments
    assert "none" in arguments and "--read-only" in arguments
    assert options["timeout"] == 90
    mounts = [arguments[i+1] for i, item in enumerate(arguments) if item == "--mount"]
    assert mounts == ["type=bind,source=/var/run/docker.sock,target=/var/run/docker.sock"]
    compile(server.TEST_AGENT_CLEANUP_CODE, "test-agent-cleanup", "exec")


def test_cleanup_timeout_collects_its_explicit_worker(monkeypatch):
    server = load_browser_server()
    backend = real_backend_controller(server)
    name = f"qpl-real-cleanup-{backend.session}"
    removed = []

    def docker(*arguments, **options):
        if arguments[0] == "run":
            assert "--rm" not in arguments
            raise RuntimeError("test cleanup timed out")
        if arguments[:2] == ("container", "ls"):
            return name
        if arguments[:2] == ("container", "inspect"):
            if arguments[3] == "{{json .Config.Labels}}":
                return json.dumps({
                    "qpl.purpose": "real-test-cleanup", "qpl.test-session": backend.session,
                })
            if arguments[3] == "{{.Image}}":
                return backend.platform_image
            assert arguments[3] == "{{json .State.Running}}"
            return "false"
        if arguments[:2] == ("container", "stop"):
            return name
        assert arguments == ("container", "rm", name)
        removed.append(name)
        return name

    monkeypatch.setattr(server, "test_docker", docker)
    with pytest.raises(RuntimeError, match="timed out"):
        backend.stop_agents()
    assert removed == [name]
