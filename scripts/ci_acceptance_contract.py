#!/usr/bin/env python3
"""CR-I3-MAJ-01 — single executable acceptance contract for the M3 live
regression gate's *operational* preconditions and receipts.

Code_Review_Iteration_3.md's CR-I3-MAJ-01 draws a line the earlier
CR-I2-MAJ-01 code fix could not close by itself: a workflow file that is
*correctly written* (trust-boundary allow-list, protected environment,
external read-only data, two preflights, timeout/concurrency, always-upload
artifact) is not the same as evidence that a *real* GitHub Actions run
exercised it on a provisioned self-hosted runner. This script is the
single, testable contract for that operational evidence, split into two
subcommands so each can be exercised independently and honestly:

  provisioning   verify the runner is registered+online with the required
                 labels, the target environment has a required_reviewers
                 protection rule with at least one configured reviewer, and
                 both external data directories (vectorstore + documents)
                 are present and not writable by the current process (the
                 read-only posture CR-I3-MAJ-01 asks for). Run this on the
                 self-hosted runner host as the runner's own service
                 account for a real answer; run elsewhere and it will
                 correctly report "not provisioned" rather than fabricate a
                 pass.
  receipt        verify a specific GitHub Actions run id: overall
                 conclusion, expected event/branch/commit-sha/workflow-path
                 provenance, named job conclusions (optionally their
                 requested runner labels), and non-expired artifact
                 uploads (optionally their sha256 digest). Works against
                 *any* completed run, so it can be exercised today against
                 an existing CI run to prove the checking logic is
                 correct, and later against the real
                 m3-live-regression-gate run once one exists.

CR-I4-MAJ-01/02 (Code_Review_Iteration_4.md) closed two fail-open gaps in
this contract: (1) the CLI's `--require-conclusion ""` used to be passed
straight through and silently mismatched `success` instead of skipping the
check as the runbook claimed — an explicit `--skip-conclusion` flag now
does that, and an empty `--require-conclusion` value is now a hard parse
error, never a silent no-op; (2) every provenance field this contract
exists to check (job/artifact lists, commit sha, branch, event, workflow
path) used to default to "not checked" — they are now required arguments
(with fail-closed validation duplicated inside the library functions, not
just argparse, so direct Python callers can't bypass it either), and
`provisioning` now refuses to run unless both named external directories
are supplied.

Code_Review_Iteration_5.md's CR-I5-MAJ-01/02 closed two remaining fail-open
gaps: (1) `--skip-conclusion` used to skip only the overall run conclusion
check while every `--require-job` still had to be `success`, so the
Runbook's own recipe for verifying an `if: always()` artifact on an
intentionally-failed live job could never actually pass — a new, narrow
`--allow-job-conclusion JOB=CONCLUSION` flag (and `require_job_conclusions`
parameter) lets a specific job's allowed conclusion be widened without
touching any other job's default `success` requirement; (2) `provisioning`
and the merge receipt both accepted partial/duplicate evidence for the
M4.1 fixed profile when called directly (bypassing the CLI's parse-time
checks) — `run_provisioning` now takes a named `external_dirs: dict[str,
Path]` keyed by `EXTERNAL_DIR_NAMES` and rejects reused paths, and a new
`check_m41_receipt`/`receipt-m41` pair fixes the merge receipt's three
required jobs and two required artifacts (`M41_RECEIPT_JOBS`/
`M41_RECEIPT_ARTIFACTS`) as a non-configurable profile instead of relying
on a caller to remember to pass the complete list to the generic `receipt`
subcommand.

Code_Review_Recovery_Cycle_1.md's CR-R1-MAJ-01 found that the named
`external_dirs` distinctness check above only compared `Path` values
lexically, so two different names whose paths were symlinks/mount aliases
to the same real directory still passed as "distinct" — `run_provisioning`
now also resolves each named dir to its canonical filesystem identity
(`Path.resolve(strict=True)` + `os.stat()`'s `(st_dev, st_ino)`, see
`_external_dir_identity`) and rejects any name group that resolves to the
same identity, before any `gh api` call, for both direct callers and the
real CLI argv path.

Both subcommands shell out to the already-authenticated `gh` CLI (`gh api`)
instead of re-implementing a GitHub REST client, so the exact commands an
operator would type by hand are the ones actually exercised here. Neither
subcommand fabricates a passing receipt: every failure raises with the
`gh api` response that drove the verdict.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


class ContractError(RuntimeError):
    """A provisioning or receipt check failed; message explains why."""


def gh_api(path: str, *, run: "GhRunner | None" = None) -> object:
    """`gh api <path>`를 실행하고 JSON을 파싱해 반환한다. 실패하면
    `ContractError`를 던진다. `run`은 테스트에서 subprocess 호출을 대체하는
    콜러블이며, 기본값은 실제 `gh` CLI를 호출한다."""
    caller = run if run is not None else _real_gh_runner
    returncode, stdout, stderr = caller(["gh", "api", path])
    if returncode != 0:
        raise ContractError(f"gh api {path} failed (exit {returncode}): {stderr.strip()}")
    text = stdout.strip()
    if not text:
        return None
    return json.loads(text)


def _real_gh_runner(cmd: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    return proc.returncode, proc.stdout, proc.stderr


GhRunner = type(_real_gh_runner)


# ---------------------------------------------------------------------------
# provisioning
# ---------------------------------------------------------------------------

# CR-I4-MAJ-02: the M4.1 workflow (`.github/workflows/ci.yml`,
# `m3-live-regression-gate`) reads exactly these two runner-local external
# paths (`SIMPLE_QNA_RAG_VECTORSTORE_DIR`/`SIMPLE_QNA_RAG_DOCUMENTS_DIR`).
# provisioning is a fixed profile for that job, not a generic N-directory
# checker, so both names are mandatory rather than an optional free list.
EXTERNAL_DIR_NAMES = ("vectorstore", "documents")


def check_runner_registered(repo: str, required_labels: set[str], *, run=None) -> dict:
    """`required_labels`를 모두 가진 online self-hosted runner를 찾아 반환한다.
    없으면 `ContractError`."""
    if not required_labels:
        raise ContractError(
            "required_labels must not be empty — a runner check with no required labels "
            "would match any online runner and prove nothing"
        )
    data = gh_api(f"repos/{repo}/actions/runners", run=run)
    runners = (data or {}).get("runners", []) if isinstance(data, dict) else []
    for runner in runners:
        labels = {label.get("name") for label in runner.get("labels", [])}
        if required_labels <= labels and runner.get("status") == "online":
            return runner
    raise ContractError(
        f"no online runner with labels {sorted(required_labels)} registered on {repo} "
        f"(found {len(runners)} runner(s): {[r.get('name') for r in runners]})"
    )


def check_environment_protected(repo: str, environment: str, *, run=None) -> dict:
    """`environment`가 존재하고 `required_reviewers` protection rule을
    가졌으며 그 rule에 실제 reviewer가 최소 1명 이상 설정돼 있는지 확인한다.
    셋 중 하나라도 아니면 `ContractError`."""
    try:
        data = gh_api(f"repos/{repo}/environments/{environment}", run=run)
    except ContractError as exc:
        raise ContractError(f"environment {environment!r} does not exist on {repo}: {exc}") from exc
    if not isinstance(data, dict):
        raise ContractError(
            f"environment {environment!r} does not exist on {repo} (gh api returned empty response)"
        )
    rules = data.get("protection_rules", [])
    reviewer_rules = [r for r in rules if r.get("type") == "required_reviewers"]
    if not reviewer_rules:
        raise ContractError(
            f"environment {environment!r} exists but has no required_reviewers protection rule "
            f"(rules found: {[r.get('type') for r in rules]})"
        )
    reviewers = [rv for rule in reviewer_rules for rv in (rule.get("reviewers") or [])]
    if not reviewers:
        raise ContractError(
            f"environment {environment!r} has a required_reviewers protection rule but no reviewers "
            f"are configured on it — the approval gate would not actually block anyone"
        )
    return data


def check_external_dir_readonly(path: Path) -> None:
    """`path`가 존재하고 현재 프로세스에 쓰기 불가능한지 확인한다(runner
    service account 기준 read-only 계약). 개발자 워크스테이션에서 이 체크를
    실행하면 그 계정으로는 정상적으로 쓰기가 가능하므로 의도적으로
    실패한다 — 이 체크는 runner 호스트에서 runner service account로 실행할
    때만 실제 답을 준다."""
    if not path.is_dir():
        raise ContractError(f"external data dir does not exist: {path}")
    if os.access(path, os.W_OK):
        raise ContractError(
            f"external data dir is writable by the current process — expected read-only "
            f"for the runner service account: {path}"
        )


def _external_dir_identity(name: str, path: Path) -> tuple[int, int]:
    """CR-R1-MAJ-01: `path`의 canonical filesystem identity(`(st_dev,
    st_ino)`)를 반환한다. lexical `Path` 비교(`__eq__`/`__hash__`)는 서로
    다른 symlink나 mount alias가 같은 실제 디렉터리를 가리켜도 다른 값으로
    취급하므로, 여기서는 `Path.resolve(strict=True)`로 symlink를 완전히
    풀고 `os.stat()`의 `(st_dev, st_ino)`로 실제 inode identity를 비교한다
    — 같은 st_dev/st_ino는 bind mount로 이중 마운트된 경우까지 포함해
    동일 디렉터리로 간주된다. `check_external_dir_readonly`가 이미
    `path.is_dir()`로 존재를 확인한 뒤 호출되므로 여기서 실패하는 경우는
    resolve 도중의 권한 오류나 호출 사이의 레이스뿐이며, 두 경우 모두
    기존 fail-closed 계약과 동일하게 `ContractError`로 변환한다."""
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ContractError(f"external data dir does not exist: {path}") from exc
    except OSError as exc:
        raise ContractError(
            f"external data dir {name!r} could not be resolved to a canonical path: "
            f"{path} ({exc})"
        ) from exc
    try:
        stat_result = resolved.stat()
    except OSError as exc:
        raise ContractError(
            f"external data dir {name!r} could not be stat'd: {resolved} ({exc})"
        ) from exc
    return (stat_result.st_dev, stat_result.st_ino)


def run_provisioning(
    repo: str,
    *,
    runner_labels: set[str],
    environment: str,
    external_dirs: dict[str, Path],
    run=None,
) -> None:
    """CR-I4-MAJ-02/CR-I5-MAJ-02 fail-closed 계약: `check_runner_registered`/
    `check_environment_protected`는 각자 빈 필수값을 거부하고, 여기서는
    `external_dirs`가 M4.1 고정 profile(`EXTERNAL_DIR_NAMES` — vectorstore/
    documents 각각 이름 있는 항목)과 정확히 일치하는지, 그리고 서로 다른
    이름이 동일 경로를 가리키지 않는지(두 디렉터리를 실제로 검사하는 척만
    하는 것을 방지)를 개별 dir 체크로 내려가기 전에 즉시 거부한다.

    CR-I5-MAJ-02: 이전에는 `list[Path]`를 받아 길이만 검사했으므로 CLI를
    거치지 않고 `run_provisioning(..., external_dirs=[Path('/same'),
    Path('/same')])`처럼 이름 없이 같은 경로를 두 번 넘겨도 통과했다 —
    이름 있는 `dict[str, Path]`로 바꿔 CLI가 이미 강제하던 것과 동일한
    profile을 함수 경계에서도 구조적으로 강제한다.

    CR-R1-MAJ-01: 위 distinctness 체크는 `Path.__eq__`에 의한 lexical
    비교라서 `vectorstore=/data/a`와 `documents=/data/b`가 각각 같은 실제
    디렉터리(`/data/shared`)를 가리키는 symlink여도 서로 다른 값으로
    통과했다 — 두 dir을 "검사하는 척"만 하는 것을 막으려던 원래 의도를
    symlink/mount alias로 우회할 수 있었다. `_external_dir_identity`로 각
    dir을 `Path.resolve(strict=True)` + `os.stat()`의 `(st_dev, st_ino)`
    까지 canonical하게 풀어 실제 filesystem identity가 겹치는 이름 조합을
    gh api 호출(runner/environment 체크) 전에 즉시 fail-closed로 거부한다
    — 존재하지 않음/권한 오류도 동일하게 `ContractError`로 fail-closed
    처리된다(`_external_dir_identity` 참고)."""
    missing_names = sorted(set(EXTERNAL_DIR_NAMES) - set(external_dirs))
    extra_names = sorted(set(external_dirs) - set(EXTERNAL_DIR_NAMES))
    if missing_names or extra_names:
        raise ContractError(
            f"external_dirs must have exactly the M4.1 fixed profile keys "
            f"{EXTERNAL_DIR_NAMES!r}, got {sorted(external_dirs)!r} "
            f"(missing={missing_names}, extra={extra_names})"
        )
    if len(set(external_dirs.values())) != len(external_dirs):
        raise ContractError(
            f"external_dirs values must be distinct paths — reusing the same path for more "
            f"than one of {EXTERNAL_DIR_NAMES!r} does not actually check two directories: "
            f"{ {name: str(path) for name, path in external_dirs.items()} }"
        )
    identities: dict[tuple[int, int], list[str]] = {}
    for name in EXTERNAL_DIR_NAMES:
        identity = _external_dir_identity(name, external_dirs[name])
        identities.setdefault(identity, []).append(name)
    alias_groups = [sorted(names) for names in identities.values() if len(names) > 1]
    if alias_groups:
        raise ContractError(
            f"external_dirs values must be distinct filesystem identities — the following "
            f"name group(s) resolve to the same canonical directory via symlink/mount alias: "
            f"{alias_groups} "
            f"{ {name: str(path) for name, path in external_dirs.items()} }"
        )
    check_runner_registered(repo, runner_labels, run=run)
    check_environment_protected(repo, environment, run=run)
    for name in EXTERNAL_DIR_NAMES:
        check_external_dir_readonly(external_dirs[name])


def _parse_external_dir(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"--external-dir must be NAME=PATH (NAME one of {EXTERNAL_DIR_NAMES!r}), got {value!r}"
        )
    name, _, raw_path = value.partition("=")
    if name not in EXTERNAL_DIR_NAMES:
        raise argparse.ArgumentTypeError(
            f"--external-dir NAME must be one of {EXTERNAL_DIR_NAMES!r}, got {name!r}"
        )
    if not raw_path:
        raise argparse.ArgumentTypeError(f"--external-dir {name}= is missing a PATH")
    return name, Path(raw_path)


# ---------------------------------------------------------------------------
# receipt
# ---------------------------------------------------------------------------

DEFAULT_EXPECTED_EVENTS = frozenset({"push", "workflow_dispatch"})


def check_run_receipt(
    repo: str,
    run_id: str,
    *,
    require_jobs: list[str],
    require_artifacts: list[str],
    expected_sha: str,
    expected_branch: str,
    expected_workflow_path: str,
    expected_events: set[str] | None = None,
    require_conclusion: str | None = "success",
    require_job_conclusions: dict[str, set[str]] | None = None,
    require_job_labels: dict[str, set[str]] | None = None,
    require_artifact_hashes: dict[str, str] | None = None,
    run=None,
) -> dict:
    """`run_id` Actions run의 provenance(event/branch/head_sha/workflow
    path), conclusion, 지정한 job들의 conclusion(+선택적 runner label),
    만료되지 않은 artifact 업로드(+선택적 sha256 digest)를 확인한다. 실패
    조건마다 구체적인 `ContractError`를 던진다.

    CR-I4-MAJ-02: `require_jobs`/`require_artifacts`/`expected_sha`/
    `expected_branch`/`expected_workflow_path`는 모두 필수이며 비어 있으면
    `ContractError`다 — 예전에는 이 값들의 기본값이 빈 리스트/생략 가능이라
    job과 artifact를 하나도 검사하지 않은 "receipt OK", 그리고 event/
    branch/sha가 다른 과거 run을 현재 merge receipt로 받아들이는 통과가
    가능했다. 이 함수는 CLI를 거치지 않고 직접 호출해도 동일하게
    fail-closed다.

    CR-I5-MAJ-01: `require_conclusion=None`(`--skip-conclusion`)은 전체 run
    conclusion 검사만 생략하고, 각 `require_jobs` 항목의 conclusion은
    여전히 기본으로 `success`만 허용한다 — 그래서 의도적으로 실패시킨
    `if: always()` artifact 검증 run에서는 그 필수 job 자체가 실패했다는
    이유만으로 artifact 조회 전에 거부됐다. `require_job_conclusions`는 이
    기본값을 job 단위로 좁게 override하는 명시적 계약이다(전체 conclusion
    검사와는 독립적인 별도 파라미터) — 명시적으로 나열한 job만 지정한
    conclusion 집합을 허용하고, 나머지 job은 여전히 `success`만 허용한다.
    """
    if not require_jobs:
        raise ContractError(
            "require_jobs must not be empty — a receipt that checks zero jobs proves nothing"
        )
    if not require_artifacts:
        raise ContractError(
            "require_artifacts must not be empty — a receipt that checks zero artifacts proves nothing"
        )
    if not expected_sha:
        raise ContractError("expected_sha is required to bind this receipt to a specific commit")
    if not expected_branch:
        raise ContractError("expected_branch is required")
    if not expected_workflow_path:
        raise ContractError("expected_workflow_path is required")
    if expected_events is not None and not expected_events:
        raise ContractError(
            "expected_events must not be empty when provided — an empty set would check "
            "nothing; omit expected_events (None) to use the default event set "
            f"{sorted(DEFAULT_EXPECTED_EVENTS)}"
        )

    events = set(expected_events) if expected_events is not None else set(DEFAULT_EXPECTED_EVENTS)
    job_labels = require_job_labels or {}
    artifact_hashes = require_artifact_hashes or {}
    job_conclusions = require_job_conclusions or {}

    unknown_label_jobs = set(job_labels) - set(require_jobs)
    if unknown_label_jobs:
        raise ContractError(
            f"require_job_labels references job(s) not in require_jobs: {sorted(unknown_label_jobs)}"
        )
    unknown_hash_artifacts = set(artifact_hashes) - set(require_artifacts)
    if unknown_hash_artifacts:
        raise ContractError(
            f"require_artifact_hashes references artifact(s) not in require_artifacts: "
            f"{sorted(unknown_hash_artifacts)}"
        )
    unknown_conclusion_jobs = set(job_conclusions) - set(require_jobs)
    if unknown_conclusion_jobs:
        raise ContractError(
            f"require_job_conclusions references job(s) not in require_jobs: "
            f"{sorted(unknown_conclusion_jobs)}"
        )
    empty_conclusion_jobs = sorted(name for name, allowed in job_conclusions.items() if not allowed)
    if empty_conclusion_jobs:
        raise ContractError(
            f"require_job_conclusions must not have an empty allowed-conclusion set for job(s) "
            f"{empty_conclusion_jobs} — an empty set would reject every possible conclusion"
        )

    run_data = gh_api(f"repos/{repo}/actions/runs/{run_id}", run=run)
    if not isinstance(run_data, dict):
        raise ContractError(f"run {run_id} not found (gh api returned empty response)")

    actual_conclusion = run_data.get("conclusion")
    if require_conclusion is not None and actual_conclusion != require_conclusion:
        raise ContractError(
            f"run {run_id} conclusion={actual_conclusion!r}, expected {require_conclusion!r}"
        )

    actual_sha = run_data.get("head_sha")
    if actual_sha != expected_sha:
        raise ContractError(
            f"run {run_id} head_sha={actual_sha!r}, expected {expected_sha!r} — this run is not "
            f"evidence for the expected commit"
        )

    actual_branch = run_data.get("head_branch")
    if actual_branch != expected_branch:
        raise ContractError(
            f"run {run_id} head_branch={actual_branch!r}, expected {expected_branch!r}"
        )

    actual_event = run_data.get("event")
    if actual_event not in events:
        raise ContractError(
            f"run {run_id} event={actual_event!r}, expected one of {sorted(events)}"
        )

    actual_path = run_data.get("path")
    if actual_path != expected_workflow_path:
        raise ContractError(
            f"run {run_id} workflow path={actual_path!r}, expected {expected_workflow_path!r}"
        )

    jobs_data = gh_api(f"repos/{repo}/actions/runs/{run_id}/jobs", run=run)
    jobs = {j["name"]: j for j in (jobs_data or {}).get("jobs", [])}
    for name in require_jobs:
        if name not in jobs:
            raise ContractError(
                f"run {run_id} missing expected job {name!r} (found {sorted(jobs)})"
            )
        job = jobs[name]
        job_conclusion = job.get("conclusion")
        allowed_conclusions = job_conclusions.get(name, {"success"})
        if job_conclusion not in allowed_conclusions:
            raise ContractError(
                f"job {name!r} in run {run_id} conclusion={job_conclusion!r}, expected one of "
                f"{sorted(allowed_conclusions)}"
            )
        required_labels = job_labels.get(name)
        if required_labels:
            actual_labels = set(job.get("labels") or [])
            missing_labels = required_labels - actual_labels
            if missing_labels:
                raise ContractError(
                    f"job {name!r} in run {run_id} missing required runner label(s) "
                    f"{sorted(missing_labels)} (found {sorted(actual_labels)})"
                )

    artifacts_data = gh_api(f"repos/{repo}/actions/runs/{run_id}/artifacts", run=run)
    all_artifacts = {
        a["name"]: a for a in (artifacts_data or {}).get("artifacts", []) if not a.get("expired")
    }
    missing = [name for name in require_artifacts if name not in all_artifacts]
    if missing:
        raise ContractError(
            f"run {run_id} missing non-expired artifact(s) {missing} (found {sorted(all_artifacts)})"
        )
    for name, expected_hash in artifact_hashes.items():
        digest = ((all_artifacts[name].get("digest") or {}).get("sha256"))
        if not digest:
            raise ContractError(
                f"artifact {name!r} in run {run_id} has no sha256 digest to verify against "
                f"expected {expected_hash!r} (gh api artifacts response lacked 'digest')"
            )
        if digest != expected_hash:
            raise ContractError(
                f"artifact {name!r} in run {run_id} sha256={digest!r}, expected {expected_hash!r}"
            )

    return {"run": run_data, "jobs": jobs, "artifacts": sorted(all_artifacts)}


# CR-I5-MAJ-02: the M4.1 merge receipt (Runbook §4 step 5) is supposed to
# prove three named jobs and two named artifacts all landed on one run —
# not "some non-empty list of jobs/artifacts". `check_run_receipt` stays a
# generic tool (it is also used against unrelated historical runs, e.g.
# Runbook §7's replay of run 31204494509 with a different job/artifact
# list), so this fixed profile lives in its own named structure and
# wrapper function instead of narrowing the generic contract.
M41_RECEIPT_JOBS = ("python-tests", "frontend-tests", "m3-live-regression-gate")
M41_RECEIPT_ARTIFACTS = ("dependency-snapshot", "m4-regression-report")

assert len(M41_RECEIPT_JOBS) == len(set(M41_RECEIPT_JOBS)), "M41_RECEIPT_JOBS must not repeat"
assert len(M41_RECEIPT_ARTIFACTS) == len(set(M41_RECEIPT_ARTIFACTS)), "M41_RECEIPT_ARTIFACTS must not repeat"


def check_m41_receipt(
    repo: str,
    run_id: str,
    *,
    expected_sha: str,
    expected_branch: str = "master",
    expected_workflow_path: str = ".github/workflows/ci.yml",
    expected_events: set[str] | None = None,
    require_conclusion: str | None = "success",
    require_job_conclusions: dict[str, set[str]] | None = None,
    require_job_labels: dict[str, set[str]] | None = None,
    require_artifact_hashes: dict[str, str] | None = None,
    run=None,
) -> dict:
    """M4.1 고정 merge-receipt profile: `M41_RECEIPT_JOBS`(3개)/
    `M41_RECEIPT_ARTIFACTS`(2개)를 caller가 고를 수 없게 고정해서
    `check_run_receipt`에 위임한다 — `run_provisioning`의 고정
    `EXTERNAL_DIR_NAMES` profile과 동일한 패턴이다. CR-I5-MAJ-02가 지적한
    "job 하나·artifact 하나만 있어도 필수 목록으로 인정" 문제는
    `require_jobs`/`require_artifacts`를 애초에 인자로 받지 않음으로써
    CLI(`receipt-m41` 서브커맨드)와 direct call 양쪽에서 구조적으로
    막는다 — 부분집합을 넘길 방법 자체가 없다."""
    return check_run_receipt(
        repo,
        run_id,
        require_jobs=list(M41_RECEIPT_JOBS),
        require_artifacts=list(M41_RECEIPT_ARTIFACTS),
        expected_sha=expected_sha,
        expected_branch=expected_branch,
        expected_workflow_path=expected_workflow_path,
        expected_events=expected_events,
        require_conclusion=require_conclusion,
        require_job_conclusions=require_job_conclusions,
        require_job_labels=require_job_labels,
        require_artifact_hashes=require_artifact_hashes,
        run=run,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="owner/name, e.g. luminous419/simple-qna-rag")
    sub = parser.add_subparsers(dest="command", required=True)

    provisioning = sub.add_parser("provisioning", help="verify runner + environment + external data readonly")
    provisioning.add_argument("--runner-label", action="append", required=True, dest="runner_labels")
    provisioning.add_argument("--environment", required=True)
    provisioning.add_argument(
        "--external-dir",
        action="append",
        required=True,
        dest="external_dirs_raw",
        type=_parse_external_dir,
        metavar="NAME=PATH",
        help=f"repeatable, required for both of {EXTERNAL_DIR_NAMES!r}, e.g. vectorstore=/opt/.../vectorstore",
    )

    receipt = sub.add_parser("receipt", help="verify a completed Actions run's provenance, jobs, and artifacts")
    receipt.add_argument("--run-id", required=True)
    receipt.add_argument("--require-job", action="append", required=True, dest="require_jobs")
    receipt.add_argument("--require-artifact", action="append", required=True, dest="require_artifacts")
    _add_receipt_common_arguments(receipt)

    receipt_m41 = sub.add_parser(
        "receipt-m41",
        help=(
            "verify the M4.1 fixed merge-receipt profile "
            f"({M41_RECEIPT_JOBS!r} + {M41_RECEIPT_ARTIFACTS!r}) for a completed Actions run — "
            "unlike 'receipt', the job/artifact list is not caller-configurable"
        ),
    )
    receipt_m41.add_argument("--run-id", required=True)
    _add_receipt_common_arguments(receipt_m41)

    return parser


def _add_receipt_common_arguments(receipt: argparse.ArgumentParser) -> None:
    """`receipt`와 `receipt-m41`이 공유하는 provenance/conclusion/label/hash
    인자 정의 — 두 서브커맨드가 같은 파싱 규칙을 갖도록(parity) 한 곳에
    모은다. `receipt-m41`은 `--require-job`/`--require-artifact`를 여기에
    포함하지 않는다(M41_RECEIPT_JOBS/M41_RECEIPT_ARTIFACTS로 고정, CLI에서
    선택할 방법이 없음 — CR-I5-MAJ-02)."""
    receipt.add_argument(
        "--require-job-label",
        action="append",
        default=[],
        dest="require_job_labels_raw",
        metavar="JOB=LABEL",
        help="repeatable; asserts the named job's requested runner labels include LABEL",
    )
    receipt.add_argument(
        "--require-artifact-hash",
        action="append",
        default=[],
        dest="require_artifact_hashes_raw",
        metavar="NAME=SHA256",
        help="repeatable; asserts the named artifact's reported sha256 digest matches",
    )
    receipt.add_argument(
        "--allow-job-conclusion",
        action="append",
        default=[],
        dest="allow_job_conclusions_raw",
        metavar="JOB=CONCLUSION",
        help="repeatable; CR-I5-MAJ-01 — narrows the default 'success'-only conclusion "
        "requirement for the named job to the explicitly listed conclusion(s), e.g. "
        "JOB=failure. Independent of --skip-conclusion (which only affects the overall run "
        "conclusion): use both together to verify an if: always() artifact on a run whose "
        "required job actually failed.",
    )
    receipt.add_argument("--expected-sha", required=True, help="commit head_sha the run must correspond to")
    receipt.add_argument("--expected-branch", default="master")
    receipt.add_argument(
        "--expected-event",
        action="append",
        dest="expected_events",
        default=None,
        help="repeatable; default: push, workflow_dispatch",
    )
    receipt.add_argument("--expected-workflow-path", default=".github/workflows/ci.yml")
    receipt.add_argument(
        "--require-conclusion",
        default=None,
        dest="require_conclusion_raw",
        help="expected overall run conclusion (default: success). An empty string is rejected — "
        "use --skip-conclusion to intentionally skip this check.",
    )
    receipt.add_argument(
        "--skip-conclusion",
        action="store_true",
        help="skip the overall run-conclusion check (only for verifying if: always() artifacts "
        "on a run of any conclusion); cannot be combined with --require-conclusion",
    )


def _parse_key_value(parser: argparse.ArgumentParser, flag: str, raw: str) -> tuple[str, str]:
    if "=" not in raw:
        parser.error(f"{flag} must be KEY=VALUE, got {raw!r}")
    key, _, value = raw.partition("=")
    if not key or not value:
        parser.error(f"{flag} must be KEY=VALUE, got {raw!r}")
    return key, value


def _resolve_require_conclusion(
    parser: argparse.ArgumentParser, *, skip_conclusion: bool, require_conclusion_raw: str | None
) -> str | None:
    if skip_conclusion:
        if require_conclusion_raw is not None:
            parser.error("--skip-conclusion cannot be combined with --require-conclusion")
        return None
    raw_conclusion = require_conclusion_raw if require_conclusion_raw is not None else "success"
    if raw_conclusion == "":
        parser.error(
            "--require-conclusion cannot be an empty string; pass --skip-conclusion to "
            "intentionally skip the conclusion check"
        )
    return raw_conclusion


def _parse_receipt_common_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> dict:
    """`receipt`/`receipt-m41`이 공유하는 argv -> kwargs 변환. 두 서브커맨드가
    같은 헬퍼를 거치므로 conclusion/label/hash/job-conclusion 파싱 규칙이
    구조적으로 어긋날 수 없다(parity)."""
    require_conclusion = _resolve_require_conclusion(
        parser, skip_conclusion=args.skip_conclusion, require_conclusion_raw=args.require_conclusion_raw
    )

    require_job_labels: dict[str, set[str]] = {}
    for raw in args.require_job_labels_raw:
        job, label = _parse_key_value(parser, "--require-job-label", raw)
        require_job_labels.setdefault(job, set()).add(label)

    require_artifact_hashes: dict[str, str] = {}
    for raw in args.require_artifact_hashes_raw:
        name, digest = _parse_key_value(parser, "--require-artifact-hash", raw)
        require_artifact_hashes[name] = digest

    require_job_conclusions: dict[str, set[str]] = {}
    for raw in args.allow_job_conclusions_raw:
        job, conclusion = _parse_key_value(parser, "--allow-job-conclusion", raw)
        require_job_conclusions.setdefault(job, set()).add(conclusion)

    expected_events = set(args.expected_events) if args.expected_events else None

    return dict(
        expected_sha=args.expected_sha,
        expected_branch=args.expected_branch,
        expected_workflow_path=args.expected_workflow_path,
        expected_events=expected_events,
        require_conclusion=require_conclusion,
        require_job_conclusions=require_job_conclusions,
        require_job_labels=require_job_labels,
        require_artifact_hashes=require_artifact_hashes,
    )


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "provisioning":
        names = [name for name, _ in args.external_dirs_raw]
        missing_names = [n for n in EXTERNAL_DIR_NAMES if n not in names]
        if missing_names:
            parser.error(
                f"provisioning requires --external-dir for {missing_names} "
                f"(M4.1 fixed profile: {EXTERNAL_DIR_NAMES!r}); got {names!r}"
            )
        if len(names) != len(set(names)):
            parser.error(f"--external-dir names must not repeat, got {names!r}")

        try:
            run_provisioning(
                args.repo,
                runner_labels=set(args.runner_labels),
                environment=args.environment,
                external_dirs=dict(args.external_dirs_raw),
            )
        except ContractError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(f"provisioning OK: repo={args.repo} environment={args.environment} external_dirs={sorted(names)}")
        return 0

    if args.command == "receipt":
        common = _parse_receipt_common_args(parser, args)
        try:
            result = check_run_receipt(
                args.repo,
                args.run_id,
                require_jobs=args.require_jobs,
                require_artifacts=args.require_artifacts,
                **common,
            )
        except ContractError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(
            f"receipt OK: run={args.run_id} sha={args.expected_sha} jobs={sorted(result['jobs'])} "
            f"artifacts={result['artifacts']}"
        )
        return 0

    if args.command == "receipt-m41":
        common = _parse_receipt_common_args(parser, args)
        try:
            result = check_m41_receipt(args.repo, args.run_id, **common)
        except ContractError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(
            f"receipt-m41 OK: run={args.run_id} sha={args.expected_sha} jobs={sorted(result['jobs'])} "
            f"artifacts={result['artifacts']}"
        )
        return 0

    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
