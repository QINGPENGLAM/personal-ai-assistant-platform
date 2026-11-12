from __future__ import annotations

import hashlib
import os
from uuid import uuid4

import pytest

from backend.app.auth import AuthService
from backend.app.coordination import Coordination
from backend.app.repository import Repository, RepositoryError


@pytest.fixture
def workspace(settings, database):
    from backend.app.workspace import WorkspaceService

    owner, _ = AuthService(database, settings).register(
        "files@example.test",
        "correct horse battery",
        "127.0.0.1",
    )
    repository = Repository(database)
    agent, _ = repository.create_agent(owner.id, "Files", "gpt-6-luna", str(uuid4()))
    repository.set_agent_status(agent.id, "ready")
    root = settings.agent_data_root / agent.id / "workspace"
    root.mkdir(parents=True)
    service = WorkspaceService(settings, repository, Coordination(), test_mode=True)
    service.initialize()
    yield service, owner.id, agent.id, root
    service.close()


async def test_workspace_unicode_preview_metadata_and_exact_download(workspace):
    service, owner, agent, root = workspace
    data = "Unicode notes: café\n".encode()
    (root / "café.txt").write_bytes(data)
    listing = await service.list_files(owner, agent, "")
    assert listing["truncated"] is False
    assert listing["entries"][0]["name"] == "café.txt"
    metadata = await service.metadata(owner, agent, "café.txt")
    assert metadata["sha256"] == hashlib.sha256(data).hexdigest()
    assert (await service.preview(owner, agent, "café.txt"))["text"] == data.decode()
    download = await service.download(owner, agent, "café.txt")
    assert b"".join(download.chunks()) == data
    assert download.closed


@pytest.mark.parametrize(
    "path", ["/etc/passwd", "../outside", "folder/../../outside", "bad\x00name"]
)
async def test_workspace_rejects_unsafe_paths(workspace, path):
    service, owner, agent, root = workspace
    with pytest.raises(RepositoryError):
        await service.preview(owner, agent, path)


async def test_workspace_never_follows_links_or_opens_fifo(workspace, tmp_path):
    service, owner, agent, root = workspace
    outside = tmp_path / "private.txt"
    outside.write_bytes(b"private data")
    (root / "link").symlink_to(outside)
    (root / "directory-link").symlink_to(tmp_path, target_is_directory=True)
    os.mkfifo(root / "pipe")
    for path in ("link", "directory-link/private.txt", "pipe"):
        with pytest.raises(RepositoryError):
            await service.preview(owner, agent, path)
    listing = await service.list_files(owner, agent, "")
    assert {entry["kind"] for entry in listing["entries"]} == {"unsupported"}


async def test_workspace_checks_ownership_before_filesystem_access(workspace):
    service, owner, agent, root = workspace
    (root / "secret.txt").write_bytes(b"private")
    with pytest.raises(RepositoryError) as denied:
        await service.metadata(str(uuid4()), agent, "secret.txt")
    assert denied.value.status_code == 404


async def test_workspace_listing_is_bounded_and_direct_paths_reach_omitted_files(workspace):
    service, owner, agent, root = workspace
    for index in range(205):
        (root / f"{index:03}.txt").write_bytes(str(index).encode())
    listing = await service.list_files(owner, agent, "")
    assert len(listing["entries"]) == 200
    assert listing["truncated"] is True
    missing = next(
        name
        for name in (f"{index:03}.txt" for index in range(205))
        if name not in {entry["name"] for entry in listing["entries"]}
    )
    download = await service.download(owner, agent, missing)
    assert b"".join(download.chunks()) == str(int(missing[:3])).encode()


async def test_workspace_preview_and_download_limits(workspace):
    service, owner, agent, root = workspace
    (root / "large.txt").write_bytes(b"a" * (256 * 1024 + 1))
    preview = await service.preview(owner, agent, "large.txt")
    assert len(preview["text"].encode()) == 256 * 1024
    assert preview["truncated"] is True
    (root / "binary.bin").write_bytes(b"\x00\xffdata")
    assert (await service.preview(owner, agent, "binary.bin"))["binary"] is True
    with (root / "too-large.bin").open("wb") as file:
        file.truncate(20 * 1024 * 1024 + 1)
    with pytest.raises(RepositoryError) as rejected:
        await service.download(owner, agent, "too-large.bin")
    assert rejected.value.status_code == 413


def upload_request(data=b"hello", *, body=None, headers=None):
    from starlette.requests import Request

    boundary = "qpl-test-boundary"
    payload = (
        body
        if body is not None
        else (
            (
                f"--{boundary}\r\n"
                'Content-Disposition: form-data; name="file"; filename="file.txt"\r\n\r\n'
            ).encode()
            + data
            + f"\r\n--{boundary}--\r\n".encode()
        )
    )
    values = {"content-type": f"multipart/form-data; boundary={boundary}", **(headers or {})}
    sent = False

    async def receive():
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": payload, "more_body": False}

    return Request(
        {"type": "http", "headers": [(k.encode(), v.encode()) for k, v in values.items()]}, receive
    )


async def test_upload_receipt_does_not_repeat_replacement(workspace):
    service, owner, agent, root = workspace
    request_id = str(uuid4())
    receipt = await service.upload(owner, agent, request_id, "new.txt", upload_request())
    assert receipt["sha256"] == hashlib.sha256(b"hello").hexdigest()
    assert (root / "new.txt").read_bytes() == b"hello"
    assert (root / "new.txt").stat().st_mode & 0o777 == 0o600
    (root / "new.txt").write_bytes(b"later changes")
    replay = await service.upload(owner, agent, request_id, "new.txt", upload_request())
    assert replay["replayed"] is True
    assert (root / "new.txt").read_bytes() == b"later changes"
    with pytest.raises(RepositoryError) as conflict:
        await service.upload(owner, agent, request_id, "new.txt", upload_request(b"different"))
    assert conflict.value.code == "idempotency_conflict"
    assert not list(service.staging_root.iterdir())


async def test_upload_requires_current_hash_and_explicit_overwrite(workspace):
    service, owner, agent, root = workspace
    (root / "existing.txt").write_bytes(b"old")
    for overwrite, expected in [(False, None), (True, "0" * 64)]:
        with pytest.raises(RepositoryError) as rejected:
            await service.upload(
                owner,
                agent,
                str(uuid4()),
                "existing.txt",
                upload_request(),
                overwrite=overwrite,
                expected_sha256=expected,
            )
        assert rejected.value.status_code == 409
        assert (root / "existing.txt").read_bytes() == b"old"
    await service.upload(
        owner,
        agent,
        str(uuid4()),
        "existing.txt",
        upload_request(),
        overwrite=True,
        expected_sha256=hashlib.sha256(b"old").hexdigest(),
    )
    assert (root / "existing.txt").read_bytes() == b"hello"


@pytest.mark.parametrize("case", ["length", "truncated", "oversize", "field"])
async def test_rejected_upload_cleans_staging_without_mutation(workspace, case):
    service, owner, agent, root = workspace
    request = {
        "length": lambda: upload_request(headers={"content-length": str(11 * 1024**2)}),
        "truncated": lambda: upload_request(body=b"--qpl-test-boundary\r\n"),
        "oversize": lambda: upload_request(b"x" * (10 * 1024**2 + 1)),
        "field": lambda: upload_request(
            body=(
                b"--qpl-test-boundary\r\nContent-Disposition: form-data; name=wrong"
                b"\r\n\r\nx\r\n--qpl-test-boundary--\r\n"
            )
        ),
    }[case]()
    with pytest.raises(RepositoryError):
        await service.upload(owner, agent, str(uuid4()), "new.txt", request)
    assert not (root / "new.txt").exists()
    assert not list(service.staging_root.iterdir())
    assert not service._uploads


async def test_delete_receipt_never_removes_recreated_file(workspace):
    service, owner, agent, root = workspace
    target = root / "notes.txt"
    target.write_bytes(b"keep")
    digest = hashlib.sha256(b"keep").hexdigest()
    request_id = str(uuid4())
    await service.delete(owner, agent, request_id, "notes.txt", digest)
    assert not target.exists()
    target.write_bytes(b"keep")
    receipt = await service.delete(owner, agent, request_id, "notes.txt", digest)
    assert receipt["replayed"] is True
    assert target.read_bytes() == b"keep"


async def test_interrupted_operation_stays_uncertain_and_cleans_only_temp(workspace):
    service, owner, agent, root = workspace
    request_id = str(uuid4())
    service.repository.reserve_workspace_operation(
        owner, agent, request_id, "workspace.upload", "hash", {"path": "notes.txt"}
    )
    temporary = root / f".qpl-upload-{request_id}.tmp"
    temporary.write_bytes(b"partial")
    (root / "notes.txt").write_bytes(b"original")
    await service.recover_operations()
    assert not temporary.exists()
    assert (root / "notes.txt").read_bytes() == b"original"
    with pytest.raises(RepositoryError) as uncertain:
        service.repository.workspace_receipt(owner, agent, request_id, "workspace.upload", "hash")
    assert uncertain.value.code == "file_operation_uncertain"


async def test_quota_usage_must_be_verified_before_upload(workspace):
    service, owner, agent, root = workspace

    async def missing_usage(agent):
        return {"ok": True}

    service.quota_check = missing_usage
    with pytest.raises(RepositoryError) as denied:
        await service.upload(owner, agent, str(uuid4()), "new.txt", upload_request())
    assert denied.value.code == "quota_unavailable"
    assert not (root / "new.txt").exists()


async def test_cancelled_download_preparation_closes_open_descriptor(workspace, monkeypatch):
    import asyncio
    import threading

    service, owner, agent, root = workspace
    (root / "notes.txt").write_bytes(b"notes")
    opened, release = threading.Event(), threading.Event()
    original = service._download
    results = []

    def delayed(*args):
        result = original(*args)
        results.append(result)
        opened.set()
        release.wait(2)
        return result

    monkeypatch.setattr(service, "_download", delayed)
    task = asyncio.create_task(service.download(owner, agent, "notes.txt"))
    await asyncio.to_thread(opened.wait, 2)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert results[0].closed


async def test_uncertain_temporary_cleanup_is_retried_after_recovery_blocker(workspace):
    import json

    service, owner, agent, root = workspace
    request_id = str(uuid4())
    service.repository.reserve_workspace_operation(
        owner, agent, request_id, "workspace.upload", "hash", {"path": "notes.txt"}
    )
    temporary = root / f".qpl-upload-{request_id}.tmp"
    temporary.write_bytes(b"partial")
    with service.repository.database.transaction() as connection:
        connection.execute(
            "UPDATE agents SET blockers_json = ? WHERE id = ?",
            (json.dumps(["recovery_error"]), agent),
        )
    await service.recover_operations()
    assert temporary.exists()
    with service.repository.database.transaction() as connection:
        connection.execute("UPDATE agents SET blockers_json = '[]' WHERE id = ?", (agent,))
    await service.recover_operations()
    assert not temporary.exists()


def test_workspace_api_auth_csrf_ownership_and_streaming(settings):
    from fastapi.testclient import TestClient

    from backend.app.main import create_app

    with TestClient(
        create_app(settings, test_mode=True, verified_models=("gpt-6-luna",))
    ) as client:
        headers = {"Origin": settings.allowed_origin, "X-Request-ID": str(uuid4())}
        registered = client.post(
            "/api/auth/register",
            headers=headers,
            json={
                "email": "api-files@example.test",
                "password": "correct horse battery",
            },
        )
        headers["X-CSRF-Token"] = registered.json()["csrf_token"]
        created = client.post(
            "/api/agents", headers=headers, json={"name": "Files", "model": "gpt-6-luna"}
        )
        assert created.status_code == 201
        base = f"/api/agents/{created.json()['id']}/files"
        assert (
            client.post(base, params={"path": "notes.txt"}, files={"file": b"notes"}).status_code
            == 403
        )
        uploaded = client.post(
            base, params={"path": "notes.txt"}, files={"file": b"notes"}, headers=headers
        )
        assert uploaded.status_code == 200, uploaded.text
        assert client.get(base + "/download", params={"path": "notes.txt"}).content == b"notes"
        assert client.get(base + "/preview", params={"path": "notes.txt"}).json()["text"] == "notes"
        client.post("/api/auth/logout", headers=headers)
        other = client.post(
            "/api/auth/register",
            headers={"Origin": settings.allowed_origin, "X-Request-ID": str(uuid4())},
            json={
                "email": "other-files@example.test",
                "password": "correct horse battery",
            },
        )
        for route in ("", "/metadata", "/preview", "/download"):
            assert client.get(base + route, params={"path": "notes.txt"}).status_code == 404
        headers["X-CSRF-Token"] = other.json()["csrf_token"]
        assert (
            client.delete(
                base,
                params={
                    "path": "notes.txt",
                    "expected_sha256": hashlib.sha256(b"notes").hexdigest(),
                },
                headers=headers,
            ).status_code
            == 404
        )


async def test_reception_does_not_lock_agent_and_second_user_upload_is_bounded(workspace):
    import asyncio

    from starlette.requests import Request

    service, owner, agent, root = workspace
    receiving, release = asyncio.Event(), asyncio.Event()

    async def receive():
        receiving.set()
        await release.wait()
        return {"type": "http.request", "body": b"", "more_body": False}

    request = Request(
        {
            "type": "http",
            "headers": [
                (b"content-type", b"multipart/form-data; boundary=example"),
            ],
        },
        receive,
    )
    task = asyncio.create_task(service.upload(owner, agent, str(uuid4()), "slow.txt", request))
    try:
        await asyncio.wait_for(receiving.wait(), 1)
        async with asyncio.timeout(1), service.coordination.agent_locks([agent]):
            pass
        with pytest.raises(RepositoryError) as denied:
            await service.upload(owner, agent, str(uuid4()), "second.txt", upload_request())
        assert denied.value.status_code == 429
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert not service._uploads
    assert not list(service.staging_root.iterdir())


async def test_listing_scans_only_201_entries_including_internal_files(workspace, monkeypatch):
    service, owner, agent, root = workspace
    for _index in range(250):
        (root / f".qpl-upload-{uuid4()}.tmp").write_bytes(b"")
    original = os.scandir
    scanned = []

    class Scan:
        def __enter__(self):
            self.scan = original(root)
            return self

        def __iter__(self):
            return self

        def __next__(self):
            entry = next(self.scan)
            scanned.append(entry.name)
            return entry

        def __exit__(self, *args):
            self.scan.close()

    monkeypatch.setattr(os, "scandir", lambda fd: Scan())
    listing = await service.list_files(owner, agent, "")
    assert listing["truncated"] is True
    assert len(scanned) == 201


async def test_global_storage_protection_does_not_add_agent_quota_blocker(workspace):
    service, owner, agent, root = workspace
    before = service.repository.get_owned_agent(owner, agent)
    with service.repository.database.transaction() as connection:
        connection.execute("UPDATE platform_storage_state SET protection_latched = 1 WHERE id = 1")
    with pytest.raises(RepositoryError) as denied:
        await service.upload(owner, agent, str(uuid4()), "new.txt", upload_request())
    assert denied.value.status_code == 507
    after = service.repository.get_owned_agent(owner, agent)
    assert after.blockers == before.blockers
    assert after.paused == before.paused


async def test_protection_triggered_before_publication_preserves_original(workspace, monkeypatch):
    from backend.app.db import StorageWriteBlocked

    service, owner, agent, root = workspace
    target = root / "notes.txt"
    target.write_bytes(b"original")
    guard = service.repository.database.storage
    guard.enabled = True
    guard.observe(
        {"quota": {"used_bytes": 0, "hard_bytes": 2 * 1024**3}, "free_bytes": 3 * 1024**3}
    )
    original = os.fsync

    def full_during_sync(descriptor):
        original(descriptor)
        guard.trip("simulated_disk_pressure")

    monkeypatch.setattr(os, "fsync", full_during_sync)
    with pytest.raises(StorageWriteBlocked):
        await service.upload(
            owner,
            agent,
            str(uuid4()),
            "notes.txt",
            upload_request(b"new"),
            overwrite=True,
            expected_sha256=hashlib.sha256(b"original").hexdigest(),
        )
    assert target.read_bytes() == b"original"
