"""M4.3-REQ-007.4 — static contract test: the protected `m3-live-regression-gate`
block's trigger allowlist, runner labels, and environment are unchanged, and
the new hosted M4.3 jobs exist with the expected `needs`/producer wiring."""

from __future__ import annotations

from pathlib import Path

import yaml

_WORKFLOW_PATH = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "ci.yml"


def _load_workflow() -> dict:
    return yaml.safe_load(_WORKFLOW_PATH.read_text(encoding="utf-8"))


def test_protected_live_gate_trigger_runner_environment_unchanged():
    workflow = _load_workflow()
    job = workflow["jobs"]["m3-live-regression-gate"]
    assert job["runs-on"] == ["self-hosted", "ollama-m3"]
    assert job["environment"] == "m3-live-regression"
    condition = job["if"]
    assert "workflow_dispatch" in condition
    assert "github.ref == 'refs/heads/master'" in condition
    assert "pull_request" not in condition


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
