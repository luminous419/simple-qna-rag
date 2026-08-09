"""CR-I3-MAJ-01 — runner/environment/external-dir provisioning check and
Actions run receipt (provenance+job+artifact) verification, both driven
through `gh api` so the same commands an operator would type by hand are
exercised.

CR-I4-MAJ-01/02 (Code_Review_Iteration_4.md) closed two fail-open gaps:
(1) `--require-conclusion ""` used to be passed straight through instead of
being normalized/rejected, so it silently behaved like `--require-conclusion
success` rather than skipping the check as the runbook claimed; (2) every
provenance argument (jobs, artifacts, commit sha, branch, event, workflow
path, provisioning external dirs) used to default to "not checked", so a
`provisioning OK`/`receipt OK` could be produced while verifying nothing.
The tests below cover both the happy paths and these previously-missing
fail-closed/adversarial paths.
"""

import inspect
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import ci_acceptance_contract as cac  # noqa: E402


def _fake_gh(responses: dict[str, object]):
    """`responses`는 `gh api <path>` 경로 -> (JSON-encodable 응답 또는
    (returncode, stderr) 실패 튜플) 매핑이다."""

    def _run(cmd: list[str]) -> tuple[int, str, str]:
        assert cmd[:2] == ["gh", "api"]
        path = cmd[2]
        if path not in responses:
            raise AssertionError(f"unexpected gh api path: {path}")
        value = responses[path]
        if isinstance(value, tuple):
            returncode, stderr = value
            return returncode, "", stderr
        return 0, json.dumps(value), ""

    return _run


# ---------------------------------------------------------------------------
# provisioning
# ---------------------------------------------------------------------------


def test_check_runner_registered_finds_online_runner_with_required_labels():
    responses = {
        "repos/o/r/actions/runners": {
            "runners": [
                {"name": "r1", "status": "online", "labels": [{"name": "self-hosted"}, {"name": "ollama-m3"}]},
            ]
        }
    }
    runner = cac.check_runner_registered("o/r", {"self-hosted", "ollama-m3"}, run=_fake_gh(responses))
    assert runner["name"] == "r1"


def test_check_runner_registered_raises_when_no_runners():
    responses = {"repos/o/r/actions/runners": {"runners": []}}
    with pytest.raises(cac.ContractError, match="no online runner"):
        cac.check_runner_registered("o/r", {"ollama-m3"}, run=_fake_gh(responses))


def test_check_runner_registered_raises_when_labels_dont_match():
    responses = {
        "repos/o/r/actions/runners": {
            "runners": [{"name": "r1", "status": "online", "labels": [{"name": "self-hosted"}]}]
        }
    }
    with pytest.raises(cac.ContractError, match="no online runner"):
        cac.check_runner_registered("o/r", {"self-hosted", "ollama-m3"}, run=_fake_gh(responses))


def test_check_runner_registered_raises_when_runner_offline():
    responses = {
        "repos/o/r/actions/runners": {
            "runners": [
                {"name": "r1", "status": "offline", "labels": [{"name": "self-hosted"}, {"name": "ollama-m3"}]}
            ]
        }
    }
    with pytest.raises(cac.ContractError, match="no online runner"):
        cac.check_runner_registered("o/r", {"self-hosted", "ollama-m3"}, run=_fake_gh(responses))


def test_check_runner_registered_raises_when_required_labels_empty():
    with pytest.raises(cac.ContractError, match="must not be empty"):
        cac.check_runner_registered("o/r", set(), run=_fake_gh({}))


def test_check_environment_protected_passes_with_required_reviewers_rule():
    responses = {
        "repos/o/r/environments/m3-live-regression": {
            "protection_rules": [{"type": "required_reviewers", "reviewers": [{"type": "User"}]}]
        }
    }
    data = cac.check_environment_protected("o/r", "m3-live-regression", run=_fake_gh(responses))
    assert data["protection_rules"][0]["type"] == "required_reviewers"


def test_check_environment_protected_raises_when_environment_missing():
    responses = {"repos/o/r/environments/m3-live-regression": (404, "Not Found")}
    with pytest.raises(cac.ContractError, match="does not exist"):
        cac.check_environment_protected("o/r", "m3-live-regression", run=_fake_gh(responses))


def test_check_environment_protected_raises_when_no_reviewer_rule():
    responses = {
        "repos/o/r/environments/m3-live-regression": {
            "protection_rules": [{"type": "wait_timer"}]
        }
    }
    with pytest.raises(cac.ContractError, match="no required_reviewers"):
        cac.check_environment_protected("o/r", "m3-live-regression", run=_fake_gh(responses))


def test_check_environment_protected_raises_when_reviewer_rule_has_no_reviewers():
    """CR-I4-MAJ-02 adversarial case: a required_reviewers rule with an
    empty reviewers list exists but blocks nobody — must not pass."""
    responses = {
        "repos/o/r/environments/m3-live-regression": {
            "protection_rules": [{"type": "required_reviewers", "reviewers": []}]
        }
    }
    with pytest.raises(cac.ContractError, match="no reviewers are configured"):
        cac.check_environment_protected("o/r", "m3-live-regression", run=_fake_gh(responses))


def test_check_external_dir_readonly_raises_when_missing(tmp_path):
    missing = tmp_path / "does-not-exist"
    with pytest.raises(cac.ContractError, match="does not exist"):
        cac.check_external_dir_readonly(missing)


def test_check_external_dir_readonly_raises_when_writable(tmp_path):
    with pytest.raises(cac.ContractError, match="writable"):
        cac.check_external_dir_readonly(tmp_path)


def test_check_external_dir_readonly_passes_when_not_writable(tmp_path, monkeypatch):
    monkeypatch.setattr(cac.os, "access", lambda path, mode: False)
    cac.check_external_dir_readonly(tmp_path)


def test_run_provisioning_runs_all_checks_in_order(tmp_path, monkeypatch):
    responses = {
        "repos/o/r/actions/runners": {
            "runners": [
                {"name": "r1", "status": "online", "labels": [{"name": "ollama-m3"}]}
            ]
        },
        "repos/o/r/environments/m3-live-regression": {
            "protection_rules": [{"type": "required_reviewers", "reviewers": [{"type": "User"}]}]
        },
    }
    vectorstore_dir = tmp_path / "vectorstore"
    documents_dir = tmp_path / "documents"
    vectorstore_dir.mkdir()
    documents_dir.mkdir()

    calls = []
    fake = _fake_gh(responses)

    def _tracking_run(cmd):
        calls.append(cmd[2])
        return fake(cmd)

    monkeypatch.setattr(cac.os, "access", lambda path, mode: False)

    cac.run_provisioning(
        "o/r",
        runner_labels={"ollama-m3"},
        environment="m3-live-regression",
        external_dirs={"vectorstore": vectorstore_dir, "documents": documents_dir},
        run=_tracking_run,
    )

    assert calls == [
        "repos/o/r/actions/runners",
        "repos/o/r/environments/m3-live-regression",
    ]


def test_run_provisioning_raises_when_external_dir_name_missing(tmp_path):
    """CR-I4-MAJ-02 adversarial case: calling the function directly (not
    through the CLI) with only one named external dir must still fail
    closed — this must be rejected before any gh api call is made."""
    only_dir = tmp_path / "vectorstore"
    only_dir.mkdir()

    def _unexpected_run(cmd):
        raise AssertionError("gh api must not be called when external_dirs is incomplete")

    with pytest.raises(cac.ContractError, match="fixed profile keys"):
        cac.run_provisioning(
            "o/r",
            runner_labels={"ollama-m3"},
            environment="m3-live-regression",
            external_dirs={"vectorstore": only_dir},
            run=_unexpected_run,
        )


def test_run_provisioning_raises_when_no_external_dirs():
    def _unexpected_run(cmd):
        raise AssertionError("gh api must not be called when external_dirs is empty")

    with pytest.raises(cac.ContractError, match="fixed profile keys"):
        cac.run_provisioning(
            "o/r",
            runner_labels={"ollama-m3"},
            environment="m3-live-regression",
            external_dirs={},
            run=_unexpected_run,
        )


def test_run_provisioning_raises_when_external_dir_name_not_in_fixed_profile(tmp_path):
    """A dict with the right *count* of entries but a name outside
    EXTERNAL_DIR_NAMES must still fail closed — count alone is not the
    profile."""
    vectorstore_dir = tmp_path / "vectorstore"
    other_dir = tmp_path / "other"
    vectorstore_dir.mkdir()
    other_dir.mkdir()

    def _unexpected_run(cmd):
        raise AssertionError("gh api must not be called when external_dirs names are wrong")

    with pytest.raises(cac.ContractError, match="fixed profile keys"):
        cac.run_provisioning(
            "o/r",
            runner_labels={"ollama-m3"},
            environment="m3-live-regression",
            external_dirs={"vectorstore": vectorstore_dir, "other": other_dir},
            run=_unexpected_run,
        )


def test_run_provisioning_raises_when_same_path_used_for_two_external_dirs(tmp_path):
    """CR-I5-MAJ-02 regression: the review reproduced a direct-call bypass
    where the same path was passed twice and both "different" directory
    checks silently passed against the identical directory. A named dict
    closes the shape of that bypass (no more unnamed positional list), but
    the value-level bypass — two distinct required names pointing at the
    same path — must be rejected too, independent of the CLI."""
    same_dir = tmp_path / "same"
    same_dir.mkdir()

    def _unexpected_run(cmd):
        raise AssertionError("gh api must not be called when external_dirs paths are not distinct")

    with pytest.raises(cac.ContractError, match="must be distinct paths"):
        cac.run_provisioning(
            "o/r",
            runner_labels={"ollama-m3"},
            environment="m3-live-regression",
            external_dirs={"vectorstore": same_dir, "documents": same_dir},
            run=_unexpected_run,
        )


def test_external_dir_identity_raises_when_missing(tmp_path):
    missing = tmp_path / "does-not-exist"
    with pytest.raises(cac.ContractError, match="does not exist"):
        cac._external_dir_identity("vectorstore", missing)


def test_external_dir_identity_returns_same_value_for_symlink_and_target(tmp_path):
    target = tmp_path / "shared"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target)
    assert cac._external_dir_identity("vectorstore", target) == cac._external_dir_identity(
        "documents", link
    )


def test_run_provisioning_raises_when_external_dirs_are_symlink_alias_to_same_target(tmp_path):
    """CR-R1-MAJ-01 adversarial case: `vectorstore` and `documents` are two
    *different* symlinks that both resolve to the same real directory. The
    lexical `Path` distinctness check above cannot see this (the two Path
    values are textually different), so only a canonical filesystem
    identity comparison catches it — and it must do so before any `gh api`
    call, exactly like the other structural provisioning rejections above."""
    real_target = tmp_path / "shared"
    real_target.mkdir()
    vectorstore_link = tmp_path / "vectorstore"
    documents_link = tmp_path / "documents"
    vectorstore_link.symlink_to(real_target)
    documents_link.symlink_to(real_target)

    def _unexpected_run(cmd):
        raise AssertionError(
            "gh api must not be called when external_dirs alias the same filesystem identity"
        )

    with pytest.raises(cac.ContractError, match="distinct filesystem identities"):
        cac.run_provisioning(
            "o/r",
            runner_labels={"ollama-m3"},
            environment="m3-live-regression",
            external_dirs={"vectorstore": vectorstore_link, "documents": documents_link},
            run=_unexpected_run,
        )


def test_run_provisioning_raises_when_external_dir_alias_via_nested_symlink(tmp_path):
    """A symlink pointing at another symlink that ultimately resolves to the
    same target must be caught too — `Path.resolve(strict=True)` fully
    unwinds symlink chains, not just a single hop."""
    real_target = tmp_path / "shared"
    real_target.mkdir()
    intermediate_link = tmp_path / "intermediate"
    intermediate_link.symlink_to(real_target)
    vectorstore_link = tmp_path / "vectorstore"
    vectorstore_link.symlink_to(intermediate_link)

    def _unexpected_run(cmd):
        raise AssertionError(
            "gh api must not be called when external_dirs alias the same filesystem identity"
        )

    with pytest.raises(cac.ContractError, match="distinct filesystem identities"):
        cac.run_provisioning(
            "o/r",
            runner_labels={"ollama-m3"},
            environment="m3-live-regression",
            external_dirs={"vectorstore": vectorstore_link, "documents": real_target},
            run=_unexpected_run,
        )


# ---------------------------------------------------------------------------
# receipt
# ---------------------------------------------------------------------------

RECEIPT_SHA = "a" * 40
RECEIPT_PATH = ".github/workflows/ci.yml"


def _receipt_responses(
    *,
    conclusion="success",
    job_conclusion="success",
    artifact_expired=False,
    head_sha=RECEIPT_SHA,
    head_branch="master",
    event="push",
    workflow_path=RECEIPT_PATH,
    job_labels=None,
    artifact_digest=None,
):
    job = {"name": "m3-live-regression-gate", "conclusion": job_conclusion}
    if job_labels is not None:
        job["labels"] = job_labels
    artifact = {"name": "m4-regression-report", "expired": artifact_expired}
    if artifact_digest is not None:
        artifact["digest"] = {"sha256": artifact_digest}
    return {
        "repos/o/r/actions/runs/123": {
            "conclusion": conclusion,
            "id": 123,
            "head_sha": head_sha,
            "head_branch": head_branch,
            "event": event,
            "path": workflow_path,
        },
        "repos/o/r/actions/runs/123/jobs": {"jobs": [job]},
        "repos/o/r/actions/runs/123/artifacts": {"artifacts": [artifact]},
    }


def _receipt_kwargs(**overrides):
    kwargs = dict(
        require_jobs=["m3-live-regression-gate"],
        require_artifacts=["m4-regression-report"],
        expected_sha=RECEIPT_SHA,
        expected_branch="master",
        expected_workflow_path=RECEIPT_PATH,
    )
    kwargs.update(overrides)
    return kwargs


def test_check_run_receipt_passes_when_run_job_and_artifact_all_succeed():
    result = cac.check_run_receipt(
        "o/r",
        "123",
        run=_fake_gh(_receipt_responses()),
        **_receipt_kwargs(),
    )
    assert result["artifacts"] == ["m4-regression-report"]
    assert result["jobs"]["m3-live-regression-gate"]["conclusion"] == "success"


def test_check_run_receipt_raises_when_run_conclusion_mismatches():
    with pytest.raises(cac.ContractError, match="expected 'success'"):
        cac.check_run_receipt(
            "o/r",
            "123",
            run=_fake_gh(_receipt_responses(conclusion="failure")),
            **_receipt_kwargs(),
        )


def test_check_run_receipt_raises_when_required_job_missing():
    with pytest.raises(cac.ContractError, match="missing expected job"):
        cac.check_run_receipt(
            "o/r",
            "123",
            run=_fake_gh(_receipt_responses()),
            **_receipt_kwargs(require_jobs=["python-tests"]),
        )


def test_check_run_receipt_raises_when_job_conclusion_not_success():
    with pytest.raises(cac.ContractError, match="conclusion='failure'"):
        cac.check_run_receipt(
            "o/r",
            "123",
            run=_fake_gh(_receipt_responses(job_conclusion="failure")),
            **_receipt_kwargs(),
        )


def test_check_run_receipt_raises_when_artifact_expired():
    with pytest.raises(cac.ContractError, match="missing non-expired artifact"):
        cac.check_run_receipt(
            "o/r",
            "123",
            run=_fake_gh(_receipt_responses(artifact_expired=True)),
            **_receipt_kwargs(),
        )


def test_check_run_receipt_allows_null_required_conclusion_for_always_artifact_contract():
    result = cac.check_run_receipt(
        "o/r",
        "123",
        run=_fake_gh(_receipt_responses(conclusion="failure")),
        **_receipt_kwargs(require_conclusion=None),
    )
    assert result["artifacts"] == ["m4-regression-report"]


# --- CR-I5-MAJ-01: realistic failed-run/failed-job artifact contract ------


def test_check_run_receipt_raises_when_skip_conclusion_but_required_job_actually_failed():
    """CR-I5-MAJ-01 regression: Code_Review_Iteration_5.md found that all
    prior failed-run tests used run conclusion='failure' with the required
    job's own conclusion left at the default 'success' — an unrealistic
    combination that hid the real bug. A genuinely failed required job
    (the actual Runbook §4 step 6 scenario) must still be rejected by
    --skip-conclusion alone, since --skip-conclusion only ever covered the
    overall run conclusion, never the per-job conclusion."""
    with pytest.raises(cac.ContractError, match="conclusion='failure'"):
        cac.check_run_receipt(
            "o/r",
            "123",
            run=_fake_gh(_receipt_responses(conclusion="failure", job_conclusion="failure")),
            **_receipt_kwargs(require_conclusion=None),
        )


def test_check_run_receipt_allows_actually_failed_job_when_conclusion_explicitly_permitted():
    """CR-I5-MAJ-01 fix: require_job_conclusions narrowly widens exactly the
    named job's allowed conclusion set, closing the always()-artifact gap
    the previous unrealistic test missed."""
    result = cac.check_run_receipt(
        "o/r",
        "123",
        run=_fake_gh(_receipt_responses(conclusion="failure", job_conclusion="failure")),
        **_receipt_kwargs(
            require_conclusion=None,
            require_job_conclusions={"m3-live-regression-gate": {"failure"}},
        ),
    )
    assert result["jobs"]["m3-live-regression-gate"]["conclusion"] == "failure"
    assert result["artifacts"] == ["m4-regression-report"]


def test_check_run_receipt_raises_when_require_job_conclusions_references_unknown_job():
    with pytest.raises(cac.ContractError, match="require_job_conclusions references job"):
        cac.check_run_receipt(
            "o/r",
            "123",
            run=_fake_gh(_receipt_responses()),
            **_receipt_kwargs(require_job_conclusions={"some-other-job": {"failure"}}),
        )


def test_check_run_receipt_raises_when_require_job_conclusions_value_empty():
    with pytest.raises(cac.ContractError, match="empty allowed-conclusion set"):
        cac.check_run_receipt(
            "o/r",
            "123",
            run=_fake_gh(_receipt_responses()),
            **_receipt_kwargs(require_job_conclusions={"m3-live-regression-gate": set()}),
        )


def test_check_run_receipt_default_job_conclusion_still_requires_success_when_unlisted():
    """Widening one job's allowed conclusions must not loosen a second,
    unlisted required job's default 'success' requirement."""
    responses = _receipt_responses(job_conclusion="success")
    responses["repos/o/r/actions/runs/123/jobs"]["jobs"].append(
        {"name": "python-tests", "conclusion": "failure"}
    )
    with pytest.raises(cac.ContractError, match="job 'python-tests'.*conclusion='failure'"):
        cac.check_run_receipt(
            "o/r",
            "123",
            run=_fake_gh(responses),
            **_receipt_kwargs(
                require_jobs=["m3-live-regression-gate", "python-tests"],
                require_job_conclusions={"m3-live-regression-gate": {"success", "failure"}},
            ),
        )


# --- CR-I4-MAJ-02 adversarial provenance/required-evidence cases ----------


def test_check_run_receipt_raises_when_require_jobs_empty():
    with pytest.raises(cac.ContractError, match="require_jobs must not be empty"):
        cac.check_run_receipt(
            "o/r",
            "123",
            run=_fake_gh(_receipt_responses()),
            **_receipt_kwargs(require_jobs=[]),
        )


def test_check_run_receipt_raises_when_require_artifacts_empty():
    with pytest.raises(cac.ContractError, match="require_artifacts must not be empty"):
        cac.check_run_receipt(
            "o/r",
            "123",
            run=_fake_gh(_receipt_responses()),
            **_receipt_kwargs(require_artifacts=[]),
        )


def test_check_run_receipt_raises_when_head_sha_mismatches():
    """A same-name job succeeding on a different commit must not pass as
    the current merge's receipt."""
    with pytest.raises(cac.ContractError, match="head_sha"):
        cac.check_run_receipt(
            "o/r",
            "123",
            run=_fake_gh(_receipt_responses(head_sha="b" * 40)),
            **_receipt_kwargs(),
        )


def test_check_run_receipt_raises_when_head_branch_mismatches():
    with pytest.raises(cac.ContractError, match="head_branch"):
        cac.check_run_receipt(
            "o/r",
            "123",
            run=_fake_gh(_receipt_responses(head_branch="feature/other")),
            **_receipt_kwargs(),
        )


def test_check_run_receipt_raises_when_event_not_in_expected_set():
    with pytest.raises(cac.ContractError, match="event="):
        cac.check_run_receipt(
            "o/r",
            "123",
            run=_fake_gh(_receipt_responses(event="pull_request")),
            **_receipt_kwargs(),
        )


def test_check_run_receipt_accepts_workflow_dispatch_event_by_default():
    result = cac.check_run_receipt(
        "o/r",
        "123",
        run=_fake_gh(_receipt_responses(event="workflow_dispatch")),
        **_receipt_kwargs(),
    )
    assert result["artifacts"] == ["m4-regression-report"]


def test_check_run_receipt_raises_when_expected_events_explicitly_empty():
    """CR-I5-MAJ-02 regression: passing expected_events=set() (as opposed
    to omitting it, i.e. None) used to fall through the previous `if
    expected_events else DEFAULT_EXPECTED_EVENTS` truthiness check and
    silently widen back to the permissive default instead of being
    rejected or checking nothing — a fail-open gap distinct from the
    require_jobs/require_artifacts empty-list checks."""
    with pytest.raises(cac.ContractError, match="expected_events must not be empty"):
        cac.check_run_receipt(
            "o/r",
            "123",
            run=_fake_gh(_receipt_responses()),
            **_receipt_kwargs(expected_events=set()),
        )


def test_check_run_receipt_raises_when_workflow_path_mismatches():
    with pytest.raises(cac.ContractError, match="workflow path"):
        cac.check_run_receipt(
            "o/r",
            "123",
            run=_fake_gh(_receipt_responses(workflow_path=".github/workflows/other.yml")),
            **_receipt_kwargs(),
        )


def test_check_run_receipt_raises_when_run_not_found():
    responses = {"repos/o/r/actions/runs/123": (0, "")}

    def _run(cmd):
        # Simulate `gh api` succeeding (exit 0) with an empty body.
        return 0, "", ""

    with pytest.raises(cac.ContractError, match="not found"):
        cac.check_run_receipt("o/r", "123", run=_run, **_receipt_kwargs())


def test_check_run_receipt_raises_when_expected_sha_empty():
    with pytest.raises(cac.ContractError, match="expected_sha is required"):
        cac.check_run_receipt(
            "o/r",
            "123",
            run=_fake_gh(_receipt_responses()),
            **_receipt_kwargs(expected_sha=""),
        )


def test_check_run_receipt_raises_when_expected_branch_empty():
    with pytest.raises(cac.ContractError, match="expected_branch is required"):
        cac.check_run_receipt(
            "o/r",
            "123",
            run=_fake_gh(_receipt_responses()),
            **_receipt_kwargs(expected_branch=""),
        )


def test_check_run_receipt_raises_when_expected_workflow_path_empty():
    with pytest.raises(cac.ContractError, match="expected_workflow_path is required"):
        cac.check_run_receipt(
            "o/r",
            "123",
            run=_fake_gh(_receipt_responses()),
            **_receipt_kwargs(expected_workflow_path=""),
        )


def test_check_run_receipt_passes_when_required_job_label_present():
    result = cac.check_run_receipt(
        "o/r",
        "123",
        run=_fake_gh(_receipt_responses(job_labels=["self-hosted", "ollama-m3"])),
        **_receipt_kwargs(require_job_labels={"m3-live-regression-gate": {"self-hosted", "ollama-m3"}}),
    )
    assert result["jobs"]["m3-live-regression-gate"]["labels"] == ["self-hosted", "ollama-m3"]


def test_check_run_receipt_raises_when_required_job_label_missing():
    with pytest.raises(cac.ContractError, match="missing required runner label"):
        cac.check_run_receipt(
            "o/r",
            "123",
            run=_fake_gh(_receipt_responses(job_labels=["self-hosted"])),
            **_receipt_kwargs(require_job_labels={"m3-live-regression-gate": {"self-hosted", "ollama-m3"}}),
        )


def test_check_run_receipt_raises_when_job_labels_reference_unknown_job():
    with pytest.raises(cac.ContractError, match="require_job_labels references job"):
        cac.check_run_receipt(
            "o/r",
            "123",
            run=_fake_gh(_receipt_responses()),
            **_receipt_kwargs(require_job_labels={"some-other-job": {"self-hosted"}}),
        )


def test_check_run_receipt_passes_when_artifact_hash_matches():
    result = cac.check_run_receipt(
        "o/r",
        "123",
        run=_fake_gh(_receipt_responses(artifact_digest="c" * 64)),
        **_receipt_kwargs(require_artifact_hashes={"m4-regression-report": "c" * 64}),
    )
    assert result["artifacts"] == ["m4-regression-report"]


def test_check_run_receipt_raises_when_artifact_hash_mismatches():
    with pytest.raises(cac.ContractError, match="sha256="):
        cac.check_run_receipt(
            "o/r",
            "123",
            run=_fake_gh(_receipt_responses(artifact_digest="c" * 64)),
            **_receipt_kwargs(require_artifact_hashes={"m4-regression-report": "d" * 64}),
        )


def test_check_run_receipt_raises_when_artifact_hash_required_but_no_digest_field():
    with pytest.raises(cac.ContractError, match="no sha256 digest"):
        cac.check_run_receipt(
            "o/r",
            "123",
            run=_fake_gh(_receipt_responses()),
            **_receipt_kwargs(require_artifact_hashes={"m4-regression-report": "c" * 64}),
        )


def test_check_run_receipt_raises_when_artifact_hashes_reference_unknown_artifact():
    with pytest.raises(cac.ContractError, match="require_artifact_hashes references artifact"):
        cac.check_run_receipt(
            "o/r",
            "123",
            run=_fake_gh(_receipt_responses()),
            **_receipt_kwargs(require_artifact_hashes={"some-other-artifact": "c" * 64}),
        )


# ---------------------------------------------------------------------------
# CR-I5-MAJ-02: check_m41_receipt — non-configurable M4.1 fixed profile
# ---------------------------------------------------------------------------

M41_RUN_ID = "456"


def _m41_receipt_responses(*, missing_job=None, missing_artifact=None, job_conclusion="success", conclusion="success"):
    jobs = [
        {"name": name, "conclusion": job_conclusion}
        for name in cac.M41_RECEIPT_JOBS
        if name != missing_job
    ]
    artifacts = [
        {"name": name, "expired": False}
        for name in cac.M41_RECEIPT_ARTIFACTS
        if name != missing_artifact
    ]
    return {
        f"repos/o/r/actions/runs/{M41_RUN_ID}": {
            "conclusion": conclusion,
            "id": int(M41_RUN_ID),
            "head_sha": RECEIPT_SHA,
            "head_branch": "master",
            "event": "push",
            "path": RECEIPT_PATH,
        },
        f"repos/o/r/actions/runs/{M41_RUN_ID}/jobs": {"jobs": jobs},
        f"repos/o/r/actions/runs/{M41_RUN_ID}/artifacts": {"artifacts": artifacts},
    }


def test_m41_receipt_jobs_have_no_duplicates():
    assert len(cac.M41_RECEIPT_JOBS) == len(set(cac.M41_RECEIPT_JOBS))


def test_m41_receipt_artifacts_have_no_duplicates():
    assert len(cac.M41_RECEIPT_ARTIFACTS) == len(set(cac.M41_RECEIPT_ARTIFACTS))


def test_check_m41_receipt_signature_has_no_configurable_job_or_artifact_list():
    """Structural parity guarantee: neither a direct caller nor the CLI can
    pass a partial job/artifact list — the parameter to do so does not
    exist, closing CR-I5-MAJ-02's "one job and one artifact is accepted as
    the required list" gap by construction rather than by validation."""
    params = inspect.signature(cac.check_m41_receipt).parameters
    assert "require_jobs" not in params
    assert "require_artifacts" not in params


def test_check_m41_receipt_passes_when_all_three_jobs_and_two_artifacts_present():
    result = cac.check_m41_receipt(
        "o/r",
        M41_RUN_ID,
        expected_sha=RECEIPT_SHA,
        run=_fake_gh(_m41_receipt_responses()),
    )
    assert sorted(result["jobs"]) == sorted(cac.M41_RECEIPT_JOBS)
    assert result["artifacts"] == sorted(cac.M41_RECEIPT_ARTIFACTS)


def test_check_m41_receipt_raises_when_a_required_job_missing():
    with pytest.raises(cac.ContractError, match="missing expected job 'm3-live-regression-gate'"):
        cac.check_m41_receipt(
            "o/r",
            M41_RUN_ID,
            expected_sha=RECEIPT_SHA,
            run=_fake_gh(_m41_receipt_responses(missing_job="m3-live-regression-gate")),
        )


def test_check_m41_receipt_raises_when_a_required_artifact_missing():
    with pytest.raises(cac.ContractError, match="missing non-expired artifact"):
        cac.check_m41_receipt(
            "o/r",
            M41_RUN_ID,
            expected_sha=RECEIPT_SHA,
            run=_fake_gh(_m41_receipt_responses(missing_artifact="dependency-snapshot")),
        )


def test_check_m41_receipt_allows_failed_live_job_via_require_job_conclusions():
    """CR-I5-MAJ-01 + CR-I5-MAJ-02 together: the Runbook §4 step 6
    always()-artifact check against an intentionally-failed live job, run
    through the fixed M4.1 profile instead of the generic receipt call."""
    responses = _m41_receipt_responses(conclusion="failure")
    for job in responses[f"repos/o/r/actions/runs/{M41_RUN_ID}/jobs"]["jobs"]:
        if job["name"] == "m3-live-regression-gate":
            job["conclusion"] = "failure"
    result = cac.check_m41_receipt(
        "o/r",
        M41_RUN_ID,
        expected_sha=RECEIPT_SHA,
        require_conclusion=None,
        require_job_conclusions={"m3-live-regression-gate": {"failure"}},
        run=_fake_gh(responses),
    )
    assert result["jobs"]["m3-live-regression-gate"]["conclusion"] == "failure"


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def test_main_provisioning_exit_1_when_check_fails(capsys, monkeypatch):
    def _raise(*args, **kwargs):
        raise cac.ContractError("no online runner")

    monkeypatch.setattr(cac, "run_provisioning", _raise)
    exit_code = cac.main(
        [
            "--repo",
            "o/r",
            "provisioning",
            "--runner-label",
            "ollama-m3",
            "--environment",
            "m3-live-regression",
            "--external-dir",
            "vectorstore=/opt/data/vectorstore",
            "--external-dir",
            "documents=/opt/data/documents",
        ]
    )
    assert exit_code == 1
    assert "no online runner" in capsys.readouterr().err


def test_main_provisioning_exit_2_when_external_dir_missing(capsys):
    with pytest.raises(SystemExit) as exc_info:
        cac.main(
            [
                "--repo",
                "o/r",
                "provisioning",
                "--runner-label",
                "ollama-m3",
                "--environment",
                "m3-live-regression",
                "--external-dir",
                "vectorstore=/opt/data/vectorstore",
            ]
        )
    assert exc_info.value.code == 2
    assert "requires --external-dir for" in capsys.readouterr().err


def test_main_provisioning_exit_2_when_external_dir_name_invalid():
    with pytest.raises(SystemExit) as exc_info:
        cac.main(
            [
                "--repo",
                "o/r",
                "provisioning",
                "--runner-label",
                "ollama-m3",
                "--environment",
                "m3-live-regression",
                "--external-dir",
                "wrong-name=/opt/data/x",
                "--external-dir",
                "documents=/opt/data/documents",
            ]
        )
    assert exc_info.value.code == 2


def test_main_provisioning_exit_2_when_external_dir_names_duplicated(capsys):
    with pytest.raises(SystemExit) as exc_info:
        cac.main(
            [
                "--repo",
                "o/r",
                "provisioning",
                "--runner-label",
                "ollama-m3",
                "--environment",
                "m3-live-regression",
                "--external-dir",
                "vectorstore=/opt/data/vectorstore",
                "--external-dir",
                "vectorstore=/opt/data/vectorstore-2",
                "--external-dir",
                "documents=/opt/data/documents",
            ]
        )
    assert exc_info.value.code == 2
    assert "must not repeat" in capsys.readouterr().err


def test_main_provisioning_exit_0_prints_summary(capsys, monkeypatch):
    monkeypatch.setattr(cac, "run_provisioning", lambda *a, **k: None)
    exit_code = cac.main(
        [
            "--repo",
            "o/r",
            "provisioning",
            "--runner-label",
            "ollama-m3",
            "--environment",
            "m3-live-regression",
            "--external-dir",
            "vectorstore=/opt/data/vectorstore",
            "--external-dir",
            "documents=/opt/data/documents",
        ]
    )
    assert exit_code == 0
    assert "provisioning OK" in capsys.readouterr().out


def test_main_provisioning_exit_1_when_external_dir_paths_identical(tmp_path, capsys):
    """CR-I5-MAJ-02 regression via the CLI argv path: distinct required
    names pointing at the same physical directory pass argparse's
    missing/duplicate-name checks (the names themselves are fine) but must
    still be rejected by run_provisioning's value-level distinctness
    check — exercising the real function, not a mock."""
    same_dir = tmp_path / "same"
    same_dir.mkdir()
    exit_code = cac.main(
        [
            "--repo",
            "o/r",
            "provisioning",
            "--runner-label",
            "ollama-m3",
            "--environment",
            "m3-live-regression",
            "--external-dir",
            f"vectorstore={same_dir}",
            "--external-dir",
            f"documents={same_dir}",
        ]
    )
    assert exit_code == 1
    assert "must be distinct paths" in capsys.readouterr().err


def test_main_provisioning_exit_1_when_external_dir_paths_are_symlink_alias(tmp_path, capsys):
    """CR-R1-MAJ-01 regression via the real CLI argv path: `vectorstore`
    and `documents` are passed as two different symlink paths (so argparse
    and the lexical `Path` distinctness check both see "different" values),
    but both resolve to the same real directory. `main()` must still reject
    this before any `gh api` call, exercising the actual CLI entrypoint
    rather than a mock of `run_provisioning`."""
    real_target = tmp_path / "shared"
    real_target.mkdir()
    vectorstore_link = tmp_path / "vectorstore"
    documents_link = tmp_path / "documents"
    vectorstore_link.symlink_to(real_target)
    documents_link.symlink_to(real_target)

    exit_code = cac.main(
        [
            "--repo",
            "o/r",
            "provisioning",
            "--runner-label",
            "ollama-m3",
            "--environment",
            "m3-live-regression",
            "--external-dir",
            f"vectorstore={vectorstore_link}",
            "--external-dir",
            f"documents={documents_link}",
        ]
    )
    assert exit_code == 1
    assert "distinct filesystem identities" in capsys.readouterr().err


def _base_receipt_argv(*extra: str) -> list[str]:
    return [
        "--repo",
        "o/r",
        "receipt",
        "--run-id",
        "1",
        "--require-job",
        "m3-live-regression-gate",
        "--require-artifact",
        "m4-regression-report",
        "--expected-sha",
        RECEIPT_SHA,
        *extra,
    ]


def test_main_receipt_exit_0_and_prints_summary(capsys, monkeypatch):
    def _fake_check_run_receipt(*args, **kwargs):
        return {"run": {}, "jobs": {"python-tests": {}}, "artifacts": ["dependency-snapshot"]}

    monkeypatch.setattr(cac, "check_run_receipt", _fake_check_run_receipt)
    exit_code = cac.main(_base_receipt_argv())
    assert exit_code == 0
    assert "receipt OK" in capsys.readouterr().out


def test_main_missing_required_args_exit_2():
    with pytest.raises(SystemExit) as exc_info:
        cac.main(["--repo", "o/r", "provisioning"])
    assert exc_info.value.code == 2


def test_main_receipt_exit_2_when_require_job_missing():
    with pytest.raises(SystemExit) as exc_info:
        cac.main(
            [
                "--repo",
                "o/r",
                "receipt",
                "--run-id",
                "1",
                "--require-artifact",
                "m4-regression-report",
                "--expected-sha",
                RECEIPT_SHA,
            ]
        )
    assert exc_info.value.code == 2


def test_main_receipt_exit_2_when_require_artifact_missing():
    with pytest.raises(SystemExit) as exc_info:
        cac.main(
            [
                "--repo",
                "o/r",
                "receipt",
                "--run-id",
                "1",
                "--require-job",
                "m3-live-regression-gate",
                "--expected-sha",
                RECEIPT_SHA,
            ]
        )
    assert exc_info.value.code == 2


def test_main_receipt_exit_2_when_expected_sha_missing():
    with pytest.raises(SystemExit) as exc_info:
        cac.main(
            [
                "--repo",
                "o/r",
                "receipt",
                "--run-id",
                "1",
                "--require-job",
                "m3-live-regression-gate",
                "--require-artifact",
                "m4-regression-report",
            ]
        )
    assert exc_info.value.code == 2


def test_main_receipt_exit_2_when_require_conclusion_is_empty_string(capsys):
    """CR-I4-MAJ-01 regression: `--require-conclusion ""` used to be
    silently passed through as an empty-string comparison (always failing
    against a real conclusion) instead of skipping the check as documented.
    It must now be a hard parse error, not a silent behavior change."""
    with pytest.raises(SystemExit) as exc_info:
        cac.main(_base_receipt_argv("--require-conclusion", ""))
    assert exc_info.value.code == 2
    assert "--skip-conclusion" in capsys.readouterr().err


def test_main_receipt_exit_2_when_skip_conclusion_combined_with_require_conclusion():
    with pytest.raises(SystemExit) as exc_info:
        cac.main(_base_receipt_argv("--skip-conclusion", "--require-conclusion", "success"))
    assert exc_info.value.code == 2


def test_main_receipt_skip_conclusion_passes_require_conclusion_none(monkeypatch):
    captured = {}

    def _fake_check_run_receipt(*args, **kwargs):
        captured.update(kwargs)
        return {"run": {}, "jobs": {}, "artifacts": []}

    monkeypatch.setattr(cac, "check_run_receipt", _fake_check_run_receipt)
    exit_code = cac.main(_base_receipt_argv("--skip-conclusion"))
    assert exit_code == 0
    assert captured["require_conclusion"] is None


def test_main_receipt_default_require_conclusion_is_success(monkeypatch):
    captured = {}

    def _fake_check_run_receipt(*args, **kwargs):
        captured.update(kwargs)
        return {"run": {}, "jobs": {}, "artifacts": []}

    monkeypatch.setattr(cac, "check_run_receipt", _fake_check_run_receipt)
    cac.main(_base_receipt_argv())
    assert captured["require_conclusion"] == "success"


def test_main_receipt_parses_job_label_and_artifact_hash_flags(monkeypatch):
    captured = {}

    def _fake_check_run_receipt(*args, **kwargs):
        captured.update(kwargs)
        return {"run": {}, "jobs": {}, "artifacts": []}

    monkeypatch.setattr(cac, "check_run_receipt", _fake_check_run_receipt)
    cac.main(
        _base_receipt_argv(
            "--require-job-label",
            "m3-live-regression-gate=self-hosted",
            "--require-job-label",
            "m3-live-regression-gate=ollama-m3",
            "--require-artifact-hash",
            f"m4-regression-report={'c' * 64}",
        )
    )
    assert captured["require_job_labels"] == {"m3-live-regression-gate": {"self-hosted", "ollama-m3"}}
    assert captured["require_artifact_hashes"] == {"m4-regression-report": "c" * 64}


def test_main_receipt_exit_2_when_require_job_label_malformed():
    with pytest.raises(SystemExit) as exc_info:
        cac.main(_base_receipt_argv("--require-job-label", "no-equals-sign"))
    assert exc_info.value.code == 2


# --- CR-I5-MAJ-01: --allow-job-conclusion CLI wiring -----------------------


def test_main_receipt_parses_allow_job_conclusion_flag(monkeypatch):
    captured = {}

    def _fake_check_run_receipt(*args, **kwargs):
        captured.update(kwargs)
        return {"run": {}, "jobs": {}, "artifacts": []}

    monkeypatch.setattr(cac, "check_run_receipt", _fake_check_run_receipt)
    cac.main(
        _base_receipt_argv(
            "--skip-conclusion",
            "--allow-job-conclusion",
            "m3-live-regression-gate=failure",
        )
    )
    assert captured["require_job_conclusions"] == {"m3-live-regression-gate": {"failure"}}


def test_main_receipt_exit_2_when_allow_job_conclusion_malformed():
    with pytest.raises(SystemExit) as exc_info:
        cac.main(_base_receipt_argv("--allow-job-conclusion", "no-equals-sign"))
    assert exc_info.value.code == 2


def _failed_live_job_argv(*extra: str) -> list[str]:
    return [
        "--repo",
        "o/r",
        "receipt",
        "--run-id",
        "123",
        "--require-job",
        "m3-live-regression-gate",
        "--require-artifact",
        "m4-regression-report",
        "--expected-sha",
        RECEIPT_SHA,
        *extra,
    ]


def test_main_receipt_allow_job_conclusion_permits_actual_failed_job(capsys, monkeypatch):
    """CR-I5-MAJ-01 positive regression: exercises the *real*
    check_run_receipt (only the underlying gh CLI call is faked) through
    argv, reproducing the exact Runbook §4 step 6 scenario — an
    intentionally-failed live job whose if: always() artifact must still
    be verifiable. Mocking check_run_receipt entirely (as the pre-existing
    CLI tests do) would not have caught CR-I5-MAJ-01, since the bug was
    inside check_run_receipt's per-job conclusion check itself."""
    responses = _receipt_responses(conclusion="failure", job_conclusion="failure")
    monkeypatch.setattr(cac, "_real_gh_runner", _fake_gh(responses))
    exit_code = cac.main(
        _failed_live_job_argv(
            "--skip-conclusion",
            "--allow-job-conclusion",
            "m3-live-regression-gate=failure",
        )
    )
    assert exit_code == 0
    assert "receipt OK" in capsys.readouterr().out


def test_main_receipt_exit_1_when_actual_failed_job_without_allow_job_conclusion(capsys, monkeypatch):
    """CR-I5-MAJ-01 negative regression: the same real failed-job scenario
    must still fail closed via argv when --allow-job-conclusion is not
    given, even with --skip-conclusion — proving --skip-conclusion alone
    never silently widened the per-job check."""
    responses = _receipt_responses(conclusion="failure", job_conclusion="failure")
    monkeypatch.setattr(cac, "_real_gh_runner", _fake_gh(responses))
    exit_code = cac.main(_failed_live_job_argv("--skip-conclusion"))
    assert exit_code == 1
    assert "conclusion='failure'" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# CR-I5-MAJ-02: receipt-m41 CLI wiring
# ---------------------------------------------------------------------------


def _base_receipt_m41_argv(*extra: str) -> list[str]:
    return [
        "--repo",
        "o/r",
        "receipt-m41",
        "--run-id",
        M41_RUN_ID,
        "--expected-sha",
        RECEIPT_SHA,
        *extra,
    ]


def test_main_receipt_m41_exit_0_and_prints_summary(capsys, monkeypatch):
    def _fake_check_m41_receipt(*args, **kwargs):
        return {"run": {}, "jobs": {name: {} for name in cac.M41_RECEIPT_JOBS}, "artifacts": sorted(cac.M41_RECEIPT_ARTIFACTS)}

    monkeypatch.setattr(cac, "check_m41_receipt", _fake_check_m41_receipt)
    exit_code = cac.main(_base_receipt_m41_argv())
    assert exit_code == 0
    assert "receipt-m41 OK" in capsys.readouterr().out


def test_main_receipt_m41_exit_1_when_check_fails(capsys, monkeypatch):
    def _raise(*args, **kwargs):
        raise cac.ContractError("missing expected job 'm3-live-regression-gate'")

    monkeypatch.setattr(cac, "check_m41_receipt", _raise)
    exit_code = cac.main(_base_receipt_m41_argv())
    assert exit_code == 1
    assert "missing expected job" in capsys.readouterr().err


def test_main_receipt_m41_exit_2_when_expected_sha_missing():
    with pytest.raises(SystemExit) as exc_info:
        cac.main(["--repo", "o/r", "receipt-m41", "--run-id", M41_RUN_ID])
    assert exc_info.value.code == 2


def test_main_receipt_m41_has_no_require_job_or_require_artifact_flags():
    """CLI/direct-call parity: receipt-m41 must not expose a way to
    override the fixed job/artifact list — the same structural guarantee
    tested at the function level in
    test_check_m41_receipt_signature_has_no_configurable_job_or_artifact_list."""
    with pytest.raises(SystemExit) as exc_info:
        cac.main(_base_receipt_m41_argv("--require-job", "python-tests"))
    assert exc_info.value.code == 2


def test_main_receipt_m41_shares_conclusion_and_label_parsing_with_receipt(monkeypatch):
    """CLI/direct-call parity: receipt-m41 parses --skip-conclusion,
    --allow-job-conclusion, --require-job-label, and --require-artifact-hash
    identically to receipt (both go through _parse_receipt_common_args)."""
    captured = {}

    def _fake_check_m41_receipt(*args, **kwargs):
        captured.update(kwargs)
        return {"run": {}, "jobs": {}, "artifacts": []}

    monkeypatch.setattr(cac, "check_m41_receipt", _fake_check_m41_receipt)
    cac.main(
        _base_receipt_m41_argv(
            "--skip-conclusion",
            "--allow-job-conclusion",
            "m3-live-regression-gate=failure",
            "--require-job-label",
            "m3-live-regression-gate=ollama-m3",
            "--require-artifact-hash",
            f"m4-regression-report={'c' * 64}",
        )
    )
    assert captured["require_conclusion"] is None
    assert captured["require_job_conclusions"] == {"m3-live-regression-gate": {"failure"}}
    assert captured["require_job_labels"] == {"m3-live-regression-gate": {"ollama-m3"}}
    assert captured["require_artifact_hashes"] == {"m4-regression-report": "c" * 64}
