#!/usr/bin/env python3
"""M4.3-REQ-005.3/5 — production-image security/mock smoke (Design.md §7.5).

Runs the *same* production image twice: once with the deterministic test
embedding seam mounted read-only via `PYTHONPATH` (positive: build -> serve
-> query 200), once without any harness mount at all (negative: production
must reject `EMBEDDING_PROVIDER=deterministic_test` with a bounded 503, since
the seam module physically does not exist in the image). If `docker` is not
installed, exits 0 with `{"status": "SKIPPED", "reason": "docker_unavailable"}`
— a local development convenience that never applies to hosted CI, which is
expected to always have Docker.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

SECURITY_FLAGS = [
    "--read-only",
    "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
    "--security-opt", "no-new-privileges",
    "--cap-drop", "ALL",
]

# A real, always-present vendored static asset (Design.md §7.5 step 6): if
# `deploy/Dockerfile`'s production stage ever drops `COPY web/static/`, this
# GET regresses to 404 and `static_asset_ok` goes false — the exact
# regression this check exists to catch (DR-I1-MAJ-06).
STATIC_ASSET_PATH = "/static/app.js"
_STATIC_ASSET_CONTENT_TYPE_PREFIXES = ("text/javascript", "application/javascript")

# The five bools every hosted gate depends on (§7.5 step 9) — kept as a
# single named tuple so `all_ok`'s wiring is itself a testable, inspectable
# value rather than an inline literal.
_ALL_OK_KEYS = (
    "host_gateway_reachable", "mock_query_ok", "root_page_ok",
    "static_asset_ok", "production_test_seam_sealed",
)


def build_docker_run_argv(image: str, *, index_root: str, host_port: int, mock_port: int,
                           test_seam_dir: str) -> list[str]:
    return [
        "docker", "run", "-d",
        "--user", "10001:10001",
        *SECURITY_FLAGS,
        "--add-host", "host.docker.internal:host-gateway",
        "-v", f"{index_root}:/app/runtime/index:ro",
        "-v", f"{test_seam_dir}:/opt/m43-test-seam:ro",
        "-e", "PYTHONPATH=/opt/m43-test-seam",
        "-e", "SIMPLE_QNA_RAG_INDEX_ROOT=/app/runtime/index",
        "-e", "SIMPLE_QNA_RAG_EMBEDDING_PROVIDER=deterministic_test",
        "-e", "SIMPLE_QNA_RAG_ALLOW_TEST_EMBEDDING=1",
        "-e", f"SIMPLE_QNA_RAG_OLLAMA_BASE_URL=http://host.docker.internal:{mock_port}",
        "-p", f"{host_port}:8000",
        image,
    ]


def build_negative_activation_argv(image: str, *, neg_host_port: int) -> list[str]:
    return [
        "docker", "run", "-d",
        "--user", "10001:10001",
        *SECURITY_FLAGS,
        "-e", "SIMPLE_QNA_RAG_EMBEDDING_PROVIDER=deterministic_test",
        "-e", "SIMPLE_QNA_RAG_ALLOW_TEST_EMBEDDING=1",
        "-p", f"{neg_host_port}:8000",
        image,
    ]


def build_reachability_probe_argv(container_id: str, mock_port: int) -> list[str]:
    return [
        "docker", "exec", container_id, "python", "-c",
        "import urllib.request as u; "
        f"u.urlopen('http://host.docker.internal:{mock_port}/mock/ping', timeout=5).read()",
    ]


def _run(argv: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True, **kwargs)


def _poll_ready(port: int, *, expect_status: int, expect_reason: str | None = None,
                 max_seconds: int = 10) -> tuple[bool, int | None, str | None]:
    deadline = time.monotonic() + max_seconds
    last_status, last_body = None, None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health/ready", timeout=2) as resp:
                last_status = resp.status
                last_body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_status = exc.code
            try:
                last_body = json.loads(exc.read().decode("utf-8"))
            except Exception:
                last_body = None
        except Exception:
            time.sleep(1)
            continue
        if last_status == expect_status and (
                expect_reason is None or (last_body or {}).get("reason") == expect_reason):
            return True, last_status, (last_body or {}).get("reason")
        time.sleep(1)
    return False, last_status, (last_body or {}).get("reason") if last_body else None


def check_static_asset(port: int, *, timeout: float = 5.0) -> bool:
    """`GET STATIC_ASSET_PATH` must be 200 with a JS content-type and a
    non-empty body — a 404 (missing `COPY web/static/`), a wrong
    content-type (StaticFiles misconfigured/wrong mount), or an empty body
    are all treated as failure, never as a soft pass."""
    try:
        with urllib.request.urlopen(
                f"http://127.0.0.1:{port}{STATIC_ASSET_PATH}", timeout=timeout) as resp:
            if resp.status != 200:
                return False
            content_type = resp.headers.get("Content-Type", "")
            body = resp.read()
    except Exception:
        return False
    if not body:
        return False
    if not any(content_type.startswith(prefix) for prefix in _STATIC_ASSET_CONTENT_TYPE_PREFIXES):
        return False
    return True


def compute_all_ok(result: dict) -> bool:
    return all(result.get(key) for key in _ALL_OK_KEYS)


def run_smoke(image: str) -> dict:
    if shutil.which("docker") is None:
        return {"status": "SKIPPED", "reason": "docker_unavailable"}

    from tests.support.mock_ollama import start_mock_ollama_server

    mock_server = start_mock_ollama_server()
    mock_port = mock_server.server_port
    import threading

    threading.Thread(target=mock_server.serve_forever, daemon=True).start()

    result: dict = {"schema": "m43-container-smoke-v1", "image": image}
    containers: list[str] = []
    try:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            index_root = Path(tmp) / "index_root"
            _build_fixture_index(index_root)

            host_port = _free_port()
            argv = build_docker_run_argv(
                image, index_root=str(index_root), host_port=host_port, mock_port=mock_port,
                test_seam_dir=str(REPO_ROOT / "tests" / "support"))
            proc = _run(argv)
            if proc.returncode != 0:
                result["status"] = "FAIL"
                result["error"] = "docker_run_failed"
                return result
            container_id = proc.stdout.strip()
            containers.append(container_id)

            probe = _run(build_reachability_probe_argv(container_id, mock_port))
            result["host_gateway_reachable"] = probe.returncode == 0

            ready_ok, _, _ = _poll_ready(host_port, expect_status=200)
            live_ok = False
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{host_port}/health/live", timeout=5) as resp:
                    live_ok = resp.status == 200
            except Exception:
                live_ok = False
            result["readiness_sequence"] = {"live": live_ok, "ready": ready_ok}

            root_ok = False
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{host_port}/", timeout=5) as resp:
                    root_ok = resp.status == 200 and b"<title>" in resp.read()
            except Exception:
                root_ok = False
            result["root_page_ok"] = root_ok

            mock_query_ok = False
            try:
                req = urllib.request.Request(
                    f"http://127.0.0.1:{host_port}/rag",
                    data=json.dumps({"question": "mock question"}).encode("utf-8"),
                    headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(req, timeout=30) as resp:
                    mock_query_ok = resp.status == 200
            except Exception:
                mock_query_ok = False
            result["mock_query_ok"] = mock_query_ok
            result["static_asset_ok"] = check_static_asset(host_port)
            result["embedding_provider"] = "deterministic_test"

            t0 = time.monotonic()
            _run(["docker", "stop", "-t", "10", container_id])
            result["graceful_stop_seconds"] = round(time.monotonic() - t0, 2)

            inspect = _run(["docker", "inspect", "--format", "{{.Id}}", image])
            result["image_digest"] = inspect.stdout.strip()

            neg_port = _free_port()
            neg_argv = build_negative_activation_argv(image, neg_host_port=neg_port)
            neg_proc = _run(neg_argv)
            sealed = False
            if neg_proc.returncode == 0:
                neg_container = neg_proc.stdout.strip()
                containers.append(neg_container)
                ok, status, reason = _poll_ready(
                    neg_port, expect_status=503,
                    expect_reason="artifact_test_embedding_seam_unavailable")
                sealed = ok
                _run(["docker", "stop", "-t", "5", neg_container])
            result["production_test_seam_sealed"] = sealed

            result["status"] = "PASS" if compute_all_ok(result) else "FAIL"
            return result
    finally:
        for cid in containers:
            _run(["docker", "rm", "-f", cid])
        mock_server.shutdown()


def _free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _build_fixture_index(index_root: Path) -> None:
    import sys as _sys

    _sys.path.insert(0, str(REPO_ROOT / "tests" / "support"))
    from simple_qna_rag_test_seam.deterministic_embeddings import DeterministicTestEmbeddings

    import faiss
    import numpy as np
    import pickle
    import hashlib
    from langchain_community.docstore.in_memory import InMemoryDocstore

    from simple_qna_rag.index import lifecycle
    from simple_qna_rag.index.manifest import canonical_json_bytes

    texts = ["container smoke fixture document one", "container smoke fixture document two"]
    embeddings = DeterministicTestEmbeddings()
    vectors = np.array(embeddings.embed_documents(texts), dtype="float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss_bytes = faiss.serialize_index(index).tobytes()
    docstore = InMemoryDocstore({str(i): t for i, t in enumerate(texts)})
    pkl_bytes = pickle.dumps((docstore, {i: str(i) for i in range(len(texts))}))

    binding = {
        "embedding_model_name": "deterministic-test-fixture",
        "embedding_provider": "deterministic_test",
        "normalize_embeddings": True,
        "chunk_size": 1000,
        "chunk_overlap": 200,
    }
    identity = {
        "corpus_manifest_sha256": None,
        "source_document_count": len(texts),
        "chunk_count": len(texts),
        "embedding_model_name": binding["embedding_model_name"],
        "embedding_model_revision": "unknown",
        "embedding_provider": binding["embedding_provider"],
        "normalize_embeddings": binding["normalize_embeddings"],
        "chunk_size": binding["chunk_size"],
        "chunk_overlap": binding["chunk_overlap"],
        "faiss_index_type": type(index).__name__,
        "faiss_dimension": int(index.d),
        "settings_hash": hashlib.sha256(canonical_json_bytes(binding)).hexdigest(),
        "dependency_lock_sha256": "0" * 64,
        "builder_git_sha": "0" * 40,
        "builder_git_dirty": False,
        "index_faiss": {"size_bytes": len(faiss_bytes), "sha256": hashlib.sha256(faiss_bytes).hexdigest()},
        "index_pkl": {"size_bytes": len(pkl_bytes), "sha256": hashlib.sha256(pkl_bytes).hexdigest()},
        "source": "build",
        "legacy_baseline_id": None,
    }
    manifest = lifecycle.build(index_root, faiss_bytes=faiss_bytes, pkl_bytes=pkl_bytes,
                                identity_fields=identity)
    lifecycle.activate(index_root, manifest["version_id"], operation="activate",
                        settings_snapshot=binding)
    # `_publish` already leaves version files world-readable (0o444) and
    # version dirs world-readable+executable (0o555, §4.3) — that already
    # satisfies the container's non-owner UID 10001 read-only mount without
    # loosening permissions further (a wider chmod here would fail
    # `verify_version`'s own `allowed_mode_mask=0o022` check).
    import os

    os.chmod(index_root, 0o755)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    result = run_smoke(args.image)
    text = json.dumps(result, sort_keys=True, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    if result.get("status") == "SKIPPED":
        return 0
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
