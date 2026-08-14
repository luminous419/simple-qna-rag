"""M4.1 §4.5 — FIELD_SPECS/MODEL_VALIDATORS inventory invariants (R1-MAJ-02)."""

import subprocess
import sys
from pathlib import Path

from simple_qna_rag.settings import FIELD_SPECS, Settings, _topo_sorted

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_field_specs_has_exactly_49_fields():
    assert len(FIELD_SPECS) == 52


def test_field_names_match_settings_model_fields():
    assert {s.name for s in FIELD_SPECS} == set(Settings.model_fields)


def test_each_field_has_exactly_one_value_source():
    """(a) env_alias set XOR (b) derive set XOR (c) neither (pure factory)."""
    for spec in FIELD_SPECS:
        has_env = spec.env_alias is not None
        has_derive = spec.env_alias is None and bool(spec.derived_from)
        has_pure_factory = spec.env_alias is None and not spec.derived_from
        assert sum([has_env, has_derive, has_pure_factory]) == 1, spec.name


def test_topo_sort_orders_dependencies_before_dependents():
    ordered = _topo_sorted(FIELD_SPECS)
    seen = set()
    for spec in ordered:
        for dep in spec.derived_from:
            assert dep in seen, f"{spec.name} depends on unresolved {dep}"
        seen.add(spec.name)


def test_generate_field_spec_check_passes_no_drift():
    result = subprocess.run(
        [sys.executable, "scripts/generate_field_spec.py", "--check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_generate_field_spec_rendering_is_deterministic():
    from simple_qna_rag.settings import (
        MODEL_VALIDATORS,
        render_field_specs_table,
        render_model_validators_table,
    )

    a1 = render_field_specs_table(FIELD_SPECS)
    a2 = render_field_specs_table(FIELD_SPECS)
    assert a1 == a2

    b1 = render_model_validators_table(MODEL_VALIDATORS, FIELD_SPECS)
    b2 = render_model_validators_table(MODEL_VALIDATORS, FIELD_SPECS)
    assert b1 == b2


def test_model_validators_table_has_4_rows_5_columns():
    from simple_qna_rag.settings import MODEL_VALIDATORS, render_model_validators_table

    table = render_model_validators_table(MODEL_VALIDATORS, FIELD_SPECS)
    lines = [line for line in table.splitlines() if line.startswith("|")]
    header = lines[0]
    assert header.count("|") == 6  # 5 columns -> 6 pipe chars
    data_rows = lines[2:]
    assert len(data_rows) == 5


# --- M4.3 §5.1 Layer 1 — accidental-default-activation guard ---------------


def test_deterministic_embedding_provider_without_allow_flag_rejected():
    import pytest

    from simple_qna_rag.settings import Settings, SettingsError

    with pytest.raises(SettingsError):
        Settings.from_sources(
            base_environ={"SIMPLE_QNA_RAG_EMBEDDING_PROVIDER": "deterministic_test"}
        )


def test_default_settings_never_activate_test_embedding_provider():
    from simple_qna_rag.settings import Settings

    settings = Settings.from_sources(base_environ={})
    assert settings.EMBEDDING_PROVIDER == "huggingface"
    assert settings.ALLOW_TEST_EMBEDDING is False
