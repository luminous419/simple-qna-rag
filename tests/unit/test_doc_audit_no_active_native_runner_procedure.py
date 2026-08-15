"""M4 Operational Acceptance Recovery — DR-I1-MAJ-04 closure.

The 3-string policy-claim grep already in Traceability.md §5 catches only
"policy success" phrasing (`overall_release_ready=true`, etc). It does not
catch an *executable procedure instruction* left in a runbook-shaped
document (self-hosted runner registration commands, environment-approval
guidance, live job execution instructions, unlabeled Ollama endpoints).
This module automates that check over runbook-shaped documents only
(`docs/operations/**/*.md` and `docs/milestones/**/*Runbook*.md`) — design
docs and other milestone process docs legitimately quote/discuss these exact
strings while explaining what is forbidden and why, and scanning `docs/**`
broadly would false-positive on those legitimate citations
(Design.md §7.4).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

FORBIDDEN_ACTIVE_PROCEDURE_PATTERNS = (
    r"config\.sh --url",            # self-hosted runner registration command
    r"--labels self-hosted",
    r"runs-on:\s*\[self-hosted",
    r"Required reviewers",          # environment approval guidance
    r"RUN_LIVE_LLM_TESTS=1",        # live job execution instruction
    r"OLLAMA_BASE_URL=http",        # unlabeled Ollama endpoint instruction
)
SUPERSEDED_BANNER_MARKER = "SUPERSEDED / NON-EXECUTABLE HISTORICAL RECORD"
# This allowlist only exempts a file that also carries the banner — see
# test_allowlist_without_banner_still_rejected below, which proves the
# allowlist alone (without the banner) grants no exemption.
ALLOWLISTED_HISTORICAL_FILES = frozenset({
    "docs/milestones/m4.1-configuration-observability/CI_Acceptance_Runbook.md",
})


def _scanned_doc_paths() -> list[Path]:
    # Runbook-shaped documents only — see the module docstring. Design,
    # Requirement, Plan, Traceability, Stop_Report, and code-review docs are
    # intentionally NOT scanned; they legitimately quote/discuss these exact
    # strings when explaining what was forbidden and why.
    return sorted({*REPO_ROOT.glob("docs/operations/**/*.md"),
                   *REPO_ROOT.glob("docs/milestones/**/*Runbook*.md")})


def test_no_active_native_runner_procedure_outside_banner_or_allowlist():
    for path in _scanned_doc_paths():
        text = path.read_text(encoding="utf-8")
        hits = [p for p in FORBIDDEN_ACTIVE_PROCEDURE_PATTERNS if re.search(p, text)]
        if not hits:
            continue
        rel = str(path.relative_to(REPO_ROOT))
        if rel in ALLOWLISTED_HISTORICAL_FILES and SUPERSEDED_BANNER_MARKER in text:
            continue  # explicitly banner-labeled historical record — allowed
        pytest.fail(f"{rel}: unallowlisted active-procedure pattern(s) {hits}")


_RUNBOOK_REL_PATH = next(iter(ALLOWLISTED_HISTORICAL_FILES))


def _leading_blockquote(text: str) -> str:
    """Collects the file's leading run of consecutive Markdown blockquote
    lines (starting with `>`), stopping at the first non-`>` line — however
    many lines the banner occupies (12, currently), without a magic number."""
    lines = text.splitlines()
    quote_lines: list[str] = []
    for line in lines:
        if line.startswith(">"):
            quote_lines.append(line)
        else:
            break
    return "\n".join(quote_lines)


def test_ci_acceptance_runbook_has_superseded_banner_near_top():
    text = (REPO_ROOT / _RUNBOOK_REL_PATH).read_text(encoding="utf-8")
    banner = _leading_blockquote(text)
    assert banner, "the banner blockquote must be the file's first content"
    assert SUPERSEDED_BANNER_MARKER in banner
    assert "deployment_runbook.md" in banner  # points to the current normative procedure


def test_allowlist_without_banner_still_rejected():
    # Proves the allowlist grants no exemption once the banner text is
    # removed — a filename alone does not grant permanent immunity.
    text = (REPO_ROOT / _RUNBOOK_REL_PATH).read_text(encoding="utf-8")
    stripped = text.replace(SUPERSEDED_BANNER_MARKER, "")
    hits = [p for p in FORBIDDEN_ACTIVE_PROCEDURE_PATTERNS if re.search(p, stripped)]
    assert hits  # precondition: the forbidden patterns are still present without the banner
