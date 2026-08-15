"""M4.3-REQ-007.4 / M4 Operational Acceptance Recovery — static contract test:
ordinary push/pull_request runs never schedule/wait on the self-hosted
`m3-live-regression-gate`, it resolves as a workflow_dispatch-only opt-in
informational stub with an exact-pinned no-op script and no execution
surface, and the hosted M4.3 jobs keep their expected `needs`/producer
wiring (Design.md §7.3)."""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import pytest
import yaml

_WORKFLOW_PATH = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "ci.yml"


def _load_workflow() -> dict:
    workflow = yaml.safe_load(_WORKFLOW_PATH.read_text(encoding="utf-8"))
    # PyYAML's SafeLoader parses the bare `on:` top-level key as the YAML 1.1
    # boolean `True`, not the string "on" — normalize it back so callers can
    # use the natural `workflow["on"]` accessor.
    if True in workflow and "on" not in workflow:
        workflow["on"] = workflow.pop(True)
    return workflow


def _workflow_text() -> str:
    return _WORKFLOW_PATH.read_text(encoding="utf-8")


def test_m4_assemble_needs_all_four_hosted_producers():
    workflow = _load_workflow()
    job = workflow["jobs"]["m4-assemble"]
    assert set(job["needs"]) == {"python-tests", "frontend-tests", "container", "m43-deterministic"}
    assert job["if"] == "always()"


def test_container_and_m43_deterministic_jobs_exist_hosted():
    workflow = _load_workflow()
    for job_name in ("container", "m43-deterministic"):
        job = workflow["jobs"][job_name]
        assert job["runs-on"] == "ubuntu-latest"


_M43_EVIDENCE_UPLOAD_ARTIFACT_NAMES = {
    "m43-evidence-python-tests", "m43-evidence-frontend-tests",
    "m43-evidence-container", "m43-evidence-m43-deterministic", "m4-baseline",
}


def test_m43_evidence_upload_artifact_steps_use_if_no_files_found_error():
    """Every M4.3 evidence upload step fails the job on empty artifact
    contents — the default (`warn`) would let an empty upload still read as
    job success (DR-I1-MAJ-08)."""
    workflow = _load_workflow()
    seen_names = set()
    for job in workflow["jobs"].values():
        for step in job.get("steps", []):
            if step.get("uses", "").startswith("actions/upload-artifact"):
                with_block = step.get("with", {})
                name = with_block.get("name")
                if name in _M43_EVIDENCE_UPLOAD_ARTIFACT_NAMES:
                    seen_names.add(name)
                    assert with_block.get("if-no-files-found") == "error", name
    assert seen_names == _M43_EVIDENCE_UPLOAD_ARTIFACT_NAMES


# ============================================================================
# M4 Operational Acceptance Recovery — m3-live-regression-gate NOT_ADOPTED
# stub contract (Design.md §7.3, DR-I1-MAJ-03, DR-I2-MAJ-01, DR-I3-MIN-01)
# ============================================================================

M3_GATE_JOB_KEY_LINE = "  m3-live-regression-gate:"

M3_GATE_PINNED_RUN_SCRIPT = (
    'echo "::notice::m3-live-regression-gate is NOT_ADOPTED under the current hosted/OCI release policy."\n'
    'echo "This run performed no checkout, no secrets, no environment approval, and no self-hosted runner."\n'
    'echo "See docs/milestones/m4-operational-acceptance-recovery/Requirement.md and Stop_Report.md for the reactivation path."\n'
    'exit 0\n'
)

FORBIDDEN_WORKFLOW_EXECUTABLE_PATTERNS = (
    (r"\$\{\{\s*secrets\.", "secret_interpolation"),
    (r"(?m)^\s*environment:\s", "environment_approval_key"),
    (r"runs-on:\s*\[?\s*self-hosted", "self_hosted_runner_label"),
    (r"(?i)\bollama\b", "ollama_vendor_token"),
    (r"gpt-oss", "ollama_model_token"),
    (r"\b11434\b", "ollama_default_port"),
    (r"OLLAMA_BASE_URL", "ollama_base_url_env"),
    (r"RUN_LIVE_LLM_TESTS", "live_llm_test_trigger_env"),
    (r"curl\s", "network_fetch_curl"),
    (r"wget\s", "network_fetch_wget"),
    (r"actions/checkout", "checkout_action"),
    (r"checkout@", "checkout_action_pin"),
    (r"git\s+fetch", "git_fetch_command"),
    (r"git\s+clone", "git_clone_command"),
    (r"(?m)^\s*run:.*\bscripts/", "repository_script_execution"),
)

_NEXT_TOP_LEVEL_JOB_KEY_RE = re.compile(r"(?m)^  [A-Za-z0-9_-]+:\s*$")


def _m3_gate_raw_block(workflow_text: str) -> str:
    start = workflow_text.index(M3_GATE_JOB_KEY_LINE)
    search_from = start + len(M3_GATE_JOB_KEY_LINE)
    match = _NEXT_TOP_LEVEL_JOB_KEY_RE.search(workflow_text, search_from)
    end = match.start() if match else len(workflow_text)
    return workflow_text[start:end]


_RUN_BLOCK_SCALAR_HEADER_RE = re.compile(r"(?m)^(?P<indent>[ ]*)run:[ \t]*\|[ \t]*\n")


def _m3_gate_denylist_scan_text(workflow_text: str) -> str:
    """Removes the exact-pinned `run: |` scalar body (after dedenting it to
    the parsed-string form) from the raw m3-live-regression-gate block, so
    the denylist scan below never re-scans an already exact-pinned value —
    only what remains (DR-I2-MAJ-01, DR-I3-MIN-01: dedent based on the
    `run: |` header's own indentation, not a naive `str.replace` of the
    undedented pin, which never matches and silently scans nothing removed)."""
    block = _m3_gate_raw_block(workflow_text)
    header = _RUN_BLOCK_SCALAR_HEADER_RE.search(block)
    if header is None:
        return block
    header_indent = len(header.group("indent"))
    remainder = block[header.end():]
    scalar_lines: list[str] = []
    consumed = 0
    for line in remainder.splitlines(keepends=True):
        content = line[:-1] if line.endswith("\n") else line
        if content.strip() != "":
            indent = len(content) - len(content.lstrip(" "))
            if indent <= header_indent:
                break
        scalar_lines.append(line)
        consumed += len(line)
    raw_scalar_text = "".join(scalar_lines)
    if textwrap.dedent(raw_scalar_text) != M3_GATE_PINNED_RUN_SCRIPT:
        return block
    return block[:header.end()] + remainder[consumed:]


def test_m3_live_regression_gate_is_workflow_dispatch_opt_in_only():
    workflow = _load_workflow()
    condition = workflow["jobs"]["m3-live-regression-gate"]["if"]
    assert "workflow_dispatch" in condition
    assert "enable_m3_live_regression" in condition
    assert "push" not in condition
    assert "pull_request" not in condition


def test_m3_live_regression_gate_exact_job_key_set():
    workflow = _load_workflow()
    job = workflow["jobs"]["m3-live-regression-gate"]
    assert set(job) == {"if", "runs-on", "timeout-minutes", "steps"}


def test_m3_live_regression_gate_has_no_self_hosted_or_environment():
    workflow = _load_workflow()
    job = workflow["jobs"]["m3-live-regression-gate"]
    assert job.get("runs-on") == "ubuntu-latest"
    assert "environment" not in job


def test_m3_live_regression_gate_exactly_one_step_with_exact_step_key_set():
    workflow = _load_workflow()
    job = workflow["jobs"]["m3-live-regression-gate"]
    assert len(job["steps"]) == 1
    assert set(job["steps"][0]) == {"name", "run"}


def test_m3_live_regression_gate_step_run_exact_allowlisted_script():
    workflow = _load_workflow()
    job = workflow["jobs"]["m3-live-regression-gate"]
    assert job["steps"][0]["run"] == M3_GATE_PINNED_RUN_SCRIPT


def test_m3_gate_denylist_scan_text_actually_removes_pinned_scalar():
    workflow_text = _workflow_text()
    scanned = _m3_gate_denylist_scan_text(workflow_text)
    for line in M3_GATE_PINNED_RUN_SCRIPT.splitlines():
        assert line not in scanned, line


@pytest.mark.parametrize("pattern,label", FORBIDDEN_WORKFLOW_EXECUTABLE_PATTERNS)
def test_m3_live_regression_gate_source_denylist_has_no_forbidden_executable_surfaces(pattern, label):
    workflow_text = _workflow_text()
    scanned = _m3_gate_denylist_scan_text(workflow_text)
    assert re.search(pattern, scanned) is None, label


def _render_m3_gate_job_as_raw_yaml(job: dict, run_script: str) -> str:
    lines = [M3_GATE_JOB_KEY_LINE]
    lines.append(f"    if: {job['if']}")
    lines.append(f"    runs-on: {job['runs-on']}")
    lines.append(f"    timeout-minutes: {job['timeout-minutes']}")
    lines.append("    steps:")
    lines.append(f"      - name: {job['steps'][0]['name']}")
    lines.append("        run: |")
    for line in run_script.splitlines():
        lines.append(f"          {line}")
    return "\n".join(lines) + "\n"


def test_m3_live_regression_gate_canonical_stub_literal_satisfies_full_contract_suite():
    canonical_job = {
        "if": "github.event_name == 'workflow_dispatch' && inputs.enable_m3_live_regression == true",
        "runs-on": "ubuntu-latest", "timeout-minutes": 1,
        "steps": [{"name": "NOT_ADOPTED — informational reactivation stub, no live execution",
                   "run": M3_GATE_PINNED_RUN_SCRIPT}],
    }
    rendered = _render_m3_gate_job_as_raw_yaml(canonical_job, M3_GATE_PINNED_RUN_SCRIPT)
    full_text = "on: {}\njobs:\n" + rendered + "  next-job:\n    runs-on: ubuntu-latest\n"

    assert "workflow_dispatch" in canonical_job["if"] and "push" not in canonical_job["if"] \
        and "pull_request" not in canonical_job["if"]
    assert set(canonical_job) == {"if", "runs-on", "timeout-minutes", "steps"}
    assert canonical_job.get("runs-on") == "ubuntu-latest"
    assert "environment" not in canonical_job
    assert len(canonical_job["steps"]) == 1
    assert set(canonical_job["steps"][0]) == {"name", "run"}
    assert canonical_job["steps"][0]["run"] == M3_GATE_PINNED_RUN_SCRIPT

    scanned = _m3_gate_denylist_scan_text(full_text)
    for line in M3_GATE_PINNED_RUN_SCRIPT.splitlines():
        assert line not in scanned
    for pattern, label in FORBIDDEN_WORKFLOW_EXECUTABLE_PATTERNS:
        assert re.search(pattern, scanned) is None, label


@pytest.mark.parametrize("pattern,label", FORBIDDEN_WORKFLOW_EXECUTABLE_PATTERNS)
def test_m3_live_regression_gate_source_denylist_rejects_each_forbidden_executable_surface(pattern, label):
    canonical_job = {
        "if": "github.event_name == 'workflow_dispatch' && inputs.enable_m3_live_regression == true",
        "runs-on": "ubuntu-latest", "timeout-minutes": 1,
        "steps": [{"name": "NOT_ADOPTED — informational reactivation stub, no live execution",
                   "run": M3_GATE_PINNED_RUN_SCRIPT}],
    }
    if label == "environment_approval_key":
        lines = [M3_GATE_JOB_KEY_LINE, f"    if: {canonical_job['if']}",
                 f"    runs-on: {canonical_job['runs-on']}", "    environment: prod",
                 f"    timeout-minutes: {canonical_job['timeout-minutes']}", "    steps:",
                 f"      - name: {canonical_job['steps'][0]['name']}", "        run: |"]
        for line in M3_GATE_PINNED_RUN_SCRIPT.splitlines():
            lines.append(f"          {line}")
        mutated_text = "\n".join(lines) + "\n"
    elif label == "self_hosted_runner_label":
        mutated_job = dict(canonical_job)
        mutated_job["runs-on"] = "[self-hosted, foo]"
        mutated_text = _render_m3_gate_job_as_raw_yaml(mutated_job, M3_GATE_PINNED_RUN_SCRIPT)
    elif label == "repository_script_execution":
        rendered = _render_m3_gate_job_as_raw_yaml(canonical_job, M3_GATE_PINNED_RUN_SCRIPT)
        # This pattern specifically targets a second `run:` YAML step key
        # invoking a repository script — not an arbitrary shell line that
        # merely contains "scripts/" inside the already exact-pinned block
        # scalar (that case is a plain string, no `run:` key prefix).
        mutated_text = (rendered + "      - name: extra\n"
                         "        run: python scripts/run_m4_regression_gate.py\n")
    else:
        surface_line = {
            "secret_interpolation": 'echo "${{ secrets.TOKEN }}"',
            "ollama_vendor_token": "echo ollama",
            "ollama_model_token": "echo gpt-oss",
            "ollama_default_port": "echo 11434",
            "ollama_base_url_env": "echo OLLAMA_BASE_URL",
            "live_llm_test_trigger_env": "echo RUN_LIVE_LLM_TESTS",
            "network_fetch_curl": "curl http://x",
            "network_fetch_wget": "wget http://x",
            "checkout_action": "echo actions/checkout",
            "checkout_action_pin": "echo checkout@v4",
            "git_fetch_command": "git fetch origin",
            "git_clone_command": "git clone https://x",
        }[label]
        mutated_script = M3_GATE_PINNED_RUN_SCRIPT + surface_line + "\n"
        mutated_text = _render_m3_gate_job_as_raw_yaml(canonical_job, mutated_script)

    scanned = _m3_gate_denylist_scan_text(mutated_text)
    assert re.search(pattern, scanned) is not None, label


def test_workflow_dispatch_input_enable_m3_live_regression_defaults_false():
    workflow = _load_workflow()
    input_spec = workflow["on"]["workflow_dispatch"]["inputs"]["enable_m3_live_regression"]
    assert input_spec["default"] is False
    assert input_spec["type"] == "boolean"


def test_m4_assemble_check_step_uses_v2_checker_without_legacy_flags():
    workflow = _load_workflow()
    job = workflow["jobs"]["m4-assemble"]
    run_text = None
    for step in job["steps"]:
        if step.get("name") == "Check M4 baseline state algebra":
            run_text = step["run"]
    assert run_text is not None
    assert "check_m4_baseline.py" in run_text
    for forbidden in ("--allow-legacy-v1", "--expect-operational-blocked", "--expect-hosted-",
                       "--expect-sha", "--expect-run-", "--expect-workflow-path", "--expect-event",
                       "--require-identity-binding"):
        assert forbidden not in run_text, forbidden


def test_workflow_job_set_is_exactly_five_hosted_jobs_plus_the_opt_in_stub():
    workflow = _load_workflow()
    assert set(workflow["jobs"]) == {
        "python-tests", "frontend-tests", "container", "m43-deterministic",
        "m4-assemble", "m3-live-regression-gate",
    }


def test_no_ordinary_job_needs_m3_live_regression_gate():
    workflow = _load_workflow()
    for job_name, job in workflow["jobs"].items():
        if job_name == "m3-live-regression-gate":
            continue
        needs = job.get("needs")
        if needs is None:
            continue
        needs_list = [needs] if isinstance(needs, str) else needs
        assert "m3-live-regression-gate" not in needs_list, job_name
