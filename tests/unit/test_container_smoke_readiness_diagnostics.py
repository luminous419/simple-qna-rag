"""Hosted_Container_Remediation_Iteration_5.md — readiness-failure
observability regression (no docker required, network calls to unbound
loopback ports fail fast and are caught by `run_smoke()`'s existing
try/except branches, exactly as they do in production).

Hosted run 31800982514 (commit c51387a) reproduced the exact failure
`container_smoke.py` was already extended to 60s for
(Hosted_CI_Remediation_Iteration_4.md §2): `readiness_sequence.live=true`,
`ready=false`, `mock_query_ok=false`. That extension did not fix the hosted
failure. The root defect this module closes is not the poll budget: it is
that `run_smoke()` discarded `_poll_ready`'s own last observed HTTP
status/reason (`ready_ok, _, _ = _poll_ready(...)`) and never captured the
container's own structured startup log, so a hosted failure carried no
signal distinguishing "still cold-starting past the budget" from "the app
came up and `/health/ready` is deterministically rejecting with a concrete,
already-computed reason" (`evaluate_readiness()`'s `artifact_*`/
`settings_invalid`/`engine_init_failed`/... branches, observability/health.py).
These tests pin the fix: the discarded fields are now captured and
propagated, and a `docker logs` tail is attached exactly when (not
unconditionally spammed on) a poll fails to reach its expected status.
"""

from __future__ import annotations

import subprocess

from scripts import container_smoke


def _fake_run_factory(*, container_log_tail: str = "startup log line\n"):
    """Docker-argv dispatcher standing in for `container_smoke._run` across
    every subprocess shape `run_smoke()` issues — `docker run` (positive and
    negative), `docker exec` (reachability probe), `docker logs`, `docker
    stop`, `docker inspect`, and `docker rm` cleanup."""

    def fake_run(argv, **kwargs):
        assert argv[0] == "docker"
        sub = argv[1]
        if sub == "run" and "-d" in argv:
            # Negative-activation argv omits the reachability/index-mount
            # flags the positive argv carries (build_negative_activation_argv
            # vs build_docker_run_argv) — use that to tell them apart.
            is_negative = "--add-host" not in argv
            stdout = "fake-container-neg\n" if is_negative else "fake-container-pos\n"
            return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")
        if sub == "exec":
            return subprocess.CompletedProcess(argv, 0, stdout="pong", stderr="")
        if sub == "logs":
            return subprocess.CompletedProcess(argv, 0, stdout=container_log_tail, stderr="")
        if sub == "stop":
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        if sub == "inspect":
            return subprocess.CompletedProcess(argv, 0, stdout="sha256:fakedigest", stderr="")
        if sub == "rm":
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected docker subcommand in test fake: {argv}")

    return fake_run


def _install_docker_available(monkeypatch):
    monkeypatch.setattr(container_smoke.shutil, "which", lambda _: "/usr/bin/docker")


def test_ready_poll_last_status_and_reason_propagate_when_readiness_never_converges(monkeypatch):
    """Regression for the exact hosted defect: `ready_ok, _, _ = _poll_ready(...)`
    used to throw away the poll's last observed HTTP status/reason. A hosted
    `ready=false` result must now say *why* — the concrete reason
    `evaluate_readiness()` already computed and put on `/health/ready`'s JSON
    body, not just an opaque boolean."""
    _install_docker_available(monkeypatch)
    monkeypatch.setattr(container_smoke, "_run", _fake_run_factory())
    monkeypatch.setattr(container_smoke, "_build_fixture_index", lambda index_root: None)

    def fake_poll_ready(port, *, expect_status, expect_reason=None, max_seconds=10):
        if expect_status == 200:
            return False, 503, "artifact_index_trust_error"
        return True, 503, "artifact_test_embedding_seam_unavailable"

    monkeypatch.setattr(container_smoke, "_poll_ready", fake_poll_ready)

    result = container_smoke.run_smoke("simple-qna-rag:test")

    sequence = result["readiness_sequence"]
    assert sequence["ready"] is False
    assert sequence["ready_last_http_status"] == 503
    assert sequence["ready_last_reason"] == "artifact_index_trust_error"
    assert isinstance(sequence["ready_poll_elapsed_seconds"], float)
    assert sequence["ready_poll_elapsed_seconds"] >= 0.0


def test_container_log_tail_attached_only_when_ready_poll_fails(monkeypatch):
    """`docker logs` capture must fire exactly when readiness failed to
    reach 200 — never unconditionally (that would spam every passing run's
    evidence artifact with a container log dump for no reason)."""
    _install_docker_available(monkeypatch)
    monkeypatch.setattr(container_smoke, "_run", _fake_run_factory(container_log_tail="the smoking gun\n"))
    monkeypatch.setattr(container_smoke, "_build_fixture_index", lambda index_root: None)
    monkeypatch.setattr(
        container_smoke, "_poll_ready",
        lambda port, *, expect_status, expect_reason=None, max_seconds=10:
            (False, 503, "engine_init_failed") if expect_status == 200 else (True, 503, "artifact_test_embedding_seam_unavailable"))

    result = container_smoke.run_smoke("simple-qna-rag:test")

    assert result["container_log_tail"] == "the smoking gun\n"


def test_container_log_tail_absent_when_ready_poll_succeeds(monkeypatch):
    _install_docker_available(monkeypatch)
    monkeypatch.setattr(container_smoke, "_run", _fake_run_factory())
    monkeypatch.setattr(container_smoke, "_build_fixture_index", lambda index_root: None)
    monkeypatch.setattr(
        container_smoke, "_poll_ready",
        lambda port, *, expect_status, expect_reason=None, max_seconds=10:
            (True, 200, "ok") if expect_status == 200 else (True, 503, "artifact_test_embedding_seam_unavailable"))

    result = container_smoke.run_smoke("simple-qna-rag:test")

    assert "container_log_tail" not in result
    assert result["readiness_sequence"]["ready_last_http_status"] == 200
    assert result["readiness_sequence"]["ready_last_reason"] == "ok"


def test_negative_control_last_status_reason_and_log_tail_propagate_on_seal_failure(monkeypatch):
    """The negative-control seal check discarded its own `_poll_ready`
    reason too (`ok, status, reason = _poll_ready(...)`, only `ok` used).
    A seal failure (production image somehow answering 200 for the
    deterministic-test seam) must now be diagnosable the same way the
    positive path is."""
    _install_docker_available(monkeypatch)
    monkeypatch.setattr(container_smoke, "_run", _fake_run_factory(container_log_tail="neg log\n"))
    monkeypatch.setattr(container_smoke, "_build_fixture_index", lambda index_root: None)
    monkeypatch.setattr(
        container_smoke, "_poll_ready",
        lambda port, *, expect_status, expect_reason=None, max_seconds=10:
            (True, 200, "ok") if expect_status == 200 else (False, 200, None))

    result = container_smoke.run_smoke("simple-qna-rag:test")

    assert result["production_test_seam_sealed"] is False
    assert result["production_test_seam_seal_last_http_status"] == 200
    assert result["production_test_seam_seal_last_reason"] is None
    assert result["negative_control_log_tail"] == "neg log\n"


def test_negative_control_log_tail_absent_when_seal_holds(monkeypatch):
    _install_docker_available(monkeypatch)
    monkeypatch.setattr(container_smoke, "_run", _fake_run_factory())
    monkeypatch.setattr(container_smoke, "_build_fixture_index", lambda index_root: None)
    monkeypatch.setattr(
        container_smoke, "_poll_ready",
        lambda port, *, expect_status, expect_reason=None, max_seconds=10:
            (True, 200, "ok") if expect_status == 200 else
            (True, 503, "artifact_test_embedding_seam_unavailable"))

    result = container_smoke.run_smoke("simple-qna-rag:test")

    assert result["production_test_seam_sealed"] is True
    assert "negative_control_log_tail" not in result
    assert result["production_test_seam_seal_last_http_status"] == 503
    assert result["production_test_seam_seal_last_reason"] == "artifact_test_embedding_seam_unavailable"


# ---------------------------------------------------------------------------
# `_capture_container_logs` — pure unit coverage of the tail-truncation
# contract (no `run_smoke()` scaffolding needed).
# ---------------------------------------------------------------------------


def test_capture_container_logs_combines_stdout_and_stderr(monkeypatch):
    def fake_run(argv, **kwargs):
        assert argv == ["docker", "logs", "--tail", "200", "cid123"]
        return subprocess.CompletedProcess(argv, 0, stdout="out-line\n", stderr="err-line\n")

    monkeypatch.setattr(container_smoke, "_run", fake_run)
    combined = container_smoke._capture_container_logs("cid123")
    assert combined == "out-line\nerr-line\n"


def test_capture_container_logs_truncates_to_tail_not_head(monkeypatch):
    long_output = "x" * 20000

    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, stdout=long_output, stderr="")

    monkeypatch.setattr(container_smoke, "_run", fake_run)
    combined = container_smoke._capture_container_logs("cid123", max_bytes=100)
    assert len(combined) == 100
    assert combined == long_output[-100:]


def test_capture_container_logs_preserves_startup_line_under_health_check_spam(monkeypatch):
    """Hosted run 31804369490's actual container_smoke.json evidence
    (Hosted_CI_Remediation_Iteration_6.md): a full readiness-poll budget
    logs one request_start/request_end pair per second, which alone filled
    the entire 16000-byte tail and evicted the single `"event": "startup"`
    line — the one line carrying `reason`/`engine_error_type` — before it
    ever reached the captured tail. The startup line must survive
    regardless of how much health-check noise follows it."""
    startup_line = '{"event": "startup", "reason": "engine_init_failed", "engine_error_type": "PermissionError"}\n'
    noise_line = '{"event": "request_end", "route": "health", "status_code": 503}\n'
    combined = startup_line + (noise_line * 400)  # far more than max_bytes alone

    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, stdout=combined, stderr="")

    monkeypatch.setattr(container_smoke, "_run", fake_run)
    tail = container_smoke._capture_container_logs("cid123", max_bytes=16000)
    assert startup_line in tail
    assert len(tail) <= 16000
    # The trailing noise (most recent exchange) is still present too.
    assert tail.endswith(noise_line)


def test_capture_container_logs_falls_back_to_plain_tail_when_no_startup_line(monkeypatch):
    """No behavior change for the common byte-truncation case that was
    already covered before this iteration — a log with no startup event at
    all still gets a plain last-N-bytes cut."""
    long_output = "x" * 20000

    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, stdout=long_output, stderr="")

    monkeypatch.setattr(container_smoke, "_run", fake_run)
    combined = container_smoke._capture_container_logs("cid123", max_bytes=100)
    assert len(combined) == 100
    assert combined == long_output[-100:]


def test_capture_container_logs_never_raises_on_docker_failure(monkeypatch):
    def fake_run(argv, **kwargs):
        raise OSError("docker binary vanished")

    monkeypatch.setattr(container_smoke, "_run", fake_run)
    assert container_smoke._capture_container_logs("cid123") == ""


# ---------------------------------------------------------------------------
# CR-HCIR6-MAJ-02 — strict UTF-8-byte-bounded retention. The prior
# implementation measured `len(str)` characters, not encoded bytes, and
# concatenated every matching startup line before budgeting, so multibyte
# content, one oversized startup event, or repeated startup events could all
# independently push the result past `max_bytes` once re-encoded.
# ---------------------------------------------------------------------------


def test_capture_container_logs_bounds_encoded_bytes_not_character_count(monkeypatch):
    """100 Korean characters is 100 `len()` characters but 300 UTF-8 bytes —
    the exact multibyte counterexample from Code_Review_Hosted_CI_
    Remediation_6.md (`100 characters / 300 UTF-8 bytes with max_bytes=100`).
    The previous `len(combined) <= max_bytes` short-circuit compared
    characters to a byte budget and returned the whole thing unbounded."""
    multibyte_line = "가" * 100 + "\n"

    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, stdout=multibyte_line, stderr="")

    monkeypatch.setattr(container_smoke, "_run", fake_run)
    result = container_smoke._capture_container_logs("cid123", max_bytes=100)
    assert len(result.encode("utf-8")) <= 100


def test_capture_container_logs_oversized_single_startup_event_stays_within_budget(monkeypatch):
    """A single `startup` event line larger than `max_bytes` on its own must
    still be truncated to the budget, not returned whole (the prior
    `startup_blob + tail_blob` concatenation returned it unbounded whenever
    the preserved startup line(s) alone already exceeded `max_bytes`)."""
    startup_line = '{"event": "startup", "reason": "x' + ("y" * 300) + '"}\n'
    assert len(startup_line.encode("utf-8")) > 100

    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, stdout=startup_line, stderr="")

    monkeypatch.setattr(container_smoke, "_run", fake_run)
    result = container_smoke._capture_container_logs("cid123", max_bytes=100)
    assert len(result.encode("utf-8")) <= 100
    # Evidence retained where it fits: the diagnostic marker survives the cut.
    assert '"event": "startup"' in result


def test_capture_container_logs_retains_only_latest_of_duplicate_startup_events(monkeypatch):
    """A crash-looping container can log the `startup` event many times
    before finally staying up. Concatenating every occurrence (the prior
    defect) can by itself blow the byte budget; only the single latest
    occurrence must be retained."""
    startup_template = '{"event": "startup", "reason": "engine_init_failed", "attempt": %d}\n'
    startup_lines = [startup_template % i for i in range(5)]
    noise = '{"event": "request_end", "route": "health"}\n' * 50
    combined = "".join(startup_lines) + noise

    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, stdout=combined, stderr="")

    monkeypatch.setattr(container_smoke, "_run", fake_run)
    result = container_smoke._capture_container_logs("cid123", max_bytes=500)
    assert len(result.encode("utf-8")) <= 500
    assert result.count('"event": "startup"') == 1
    assert startup_lines[-1] in result


def test_capture_container_logs_avoids_duplicating_startup_line_present_in_tail_window(monkeypatch):
    """When the retained `startup` event line already falls entirely inside
    the recomputed tail window (it happened recently, close enough to the
    end of the log), it must not also be double-counted by being prepended
    separately — a naive "startup blob + tail" concatenation would include
    the same bytes twice and waste budget without adding evidence."""
    startup_line = '{"event": "startup", "reason": "engine_init_failed"}\n'
    trail_noise = '{"event": "request_end"}\n'
    lead_noise = "n" * 300 + "\n"
    combined = lead_noise + startup_line + trail_noise

    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, stdout=combined, stderr="")

    monkeypatch.setattr(container_smoke, "_run", fake_run)
    startup_bytes_len = len(startup_line.encode("utf-8"))
    trail_bytes_len = len(trail_noise.encode("utf-8"))
    # Generous enough that the tail window, sized after reserving room for
    # the startup line, reaches back far enough to already contain the
    # whole startup line verbatim.
    max_bytes = (2 * startup_bytes_len) + trail_bytes_len
    result = container_smoke._capture_container_logs("cid123", max_bytes=max_bytes)
    assert len(result.encode("utf-8")) <= max_bytes
    assert result.count('"event": "startup"') == 1
    assert result.endswith(trail_noise)


# ---------------------------------------------------------------------------
# CR-HCIR7-MAJ-01 — the two boundary counterexamples the fresh Codex review
# found still open in CR-HCIR6-MAJ-02's remediation: an under-budget log
# with >1 startup events bypassed dedup via the early return, and a tail
# window that starts *inside* (not at, and not fully containing) the
# retained startup line duplicated the overlapping bytes instead of
# trimming them.
# ---------------------------------------------------------------------------


def test_capture_container_logs_dedupes_repeated_startup_events_even_under_budget(monkeypatch):
    """Codex's independent probe: two startup events totalling 70 encoded
    bytes with max_bytes=1000 is comfortably under budget, so the prior
    implementation's `len(combined_bytes) <= max_bytes` early return handed
    both events back verbatim — dedup only ran on the over-budget path.
    "Retain only the latest startup event" must hold unconditionally, not
    only when the log happens to exceed the byte ceiling."""
    startup_template = '{"event": "startup", "attempt": %d}\n'
    combined = (startup_template % 1) + (startup_template % 2)
    assert len(combined.encode("utf-8")) < 1000

    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, stdout=combined, stderr="")

    monkeypatch.setattr(container_smoke, "_run", fake_run)
    result = container_smoke._capture_container_logs("cid123", max_bytes=1000)
    assert len(result.encode("utf-8")) <= 1000
    assert result.count('"event": "startup"') == 1
    assert (startup_template % 2) in result
    assert (startup_template % 1) not in result


def test_capture_container_logs_trims_partial_overlap_when_tail_starts_inside_startup_line(monkeypatch):
    """Codex's independent probe reproduced this exact output:
    `{"event": "startup", ...}\\n_init_failed"}\\nTAIL` — the recomputed tail
    window's start byte lands in the *middle* of the retained startup line
    (not at its start, and not containing it wholly), so the old
    `startup in tail` full-containment check never fires and the trailing
    fragment of the startup line gets duplicated verbatim ahead of the real
    tail content. The maximal suffix-of-startup/prefix-of-tail overlap must
    be trimmed instead.

    Forcing the tail window to start inside (not at, not past) the startup
    line requires: enough *leading* content before the startup line to push
    the whole log over `max_bytes` (so the truncation branch runs at all),
    while `remaining = max_bytes - len(startup_bytes)` is sized strictly
    between `len(trail_noise)` (so the window reaches back past the trail
    into the startup line) and `len(startup_bytes) + len(trail_noise)` (so
    it does not reach back far enough to contain the whole startup line)."""
    startup_line = '{"event": "startup", "reason": "engine_init_failed"}\n'
    trail_noise = "TAIL\n"
    lead_noise = "n" * 300 + "\n"
    combined = lead_noise + startup_line + trail_noise

    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, stdout=combined, stderr="")

    monkeypatch.setattr(container_smoke, "_run", fake_run)
    startup_bytes_len = len(startup_line.encode("utf-8"))
    trail_bytes_len = len(trail_noise.encode("utf-8"))
    overlap_bytes = 15
    assert overlap_bytes < startup_bytes_len
    remaining = trail_bytes_len + overlap_bytes
    max_bytes = startup_bytes_len + remaining

    result = container_smoke._capture_container_logs("cid123", max_bytes=max_bytes)
    assert len(result.encode("utf-8")) <= max_bytes
    assert result.count('"event": "startup"') == 1
    assert result.count("engine_init_failed") == 1
    # The correct, non-duplicated reconstruction is the startup line
    # immediately followed by the trailing noise — no repeated fragment of
    # the startup line's own tail sitting between them.
    assert result == startup_line + trail_noise
