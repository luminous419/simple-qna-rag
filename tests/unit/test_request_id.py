"""M4.1 §6.3 — client X-Request-Id acceptance/rejection matrix."""

import uuid

import pytest

from simple_qna_rag.observability.request_context import _resolve_request_id


@pytest.mark.parametrize(
    "header_value",
    [
        "abc123",
        "a" * 64,
        "trace-id_with-dash_and_underscore",
        "1",
    ],
)
def test_valid_client_ids_are_reused(header_value):
    assert _resolve_request_id(header_value) == header_value


@pytest.mark.parametrize(
    "header_value",
    [
        None,
        "",
        "a" * 65,  # too long
        "has a space",
        "has/a/slash",
        "has:colon",
        "has\nnewline",
    ],
)
def test_invalid_or_missing_client_ids_get_a_fresh_uuid4(header_value):
    result = _resolve_request_id(header_value)
    assert result != header_value
    parsed = uuid.UUID(result)
    assert str(parsed) == result
