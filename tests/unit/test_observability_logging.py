"""M4.1 §6.4 — payload-safe logging schema tests."""

import io
import logging as stdlib_logging

import pytest

from simple_qna_rag.observability import logging as L
from simple_qna_rag.observability import metrics as M


@pytest.fixture
def capture():
    buf = io.StringIO()
    handler = L.SafeJsonHandler(stream=buf)
    original_handlers = L._logger.handlers
    L._logger.handlers = [handler]
    yield buf
    L._logger.handlers = original_handlers


def _last_record(buf: io.StringIO) -> dict:
    import json

    lines = [line for line in buf.getvalue().splitlines() if line.strip()]
    return json.loads(lines[-1])


def test_request_start_has_common_and_event_fields(capture):
    L.log_event("request_start", route="rag", method="POST")
    record = _last_record(capture)
    for key in ("timestamp", "level", "event", "service", "version", "request_id"):
        assert key in record
    assert record["event"] == "request_start"
    assert record["route"] == "rag"
    assert record["method"] == "POST"


def test_request_end_has_status_and_duration(capture):
    L.log_event("request_end", route="rag", status_code=200, duration_ms=12.5)
    record = _last_record(capture)
    assert record["status_code"] == 200
    assert record["duration_ms"] == 12.5


@pytest.mark.parametrize(
    "forbidden_field,value",
    [
        ("question", "what is the capital of France?"),
        ("answer", "Paris is the capital"),
        ("context", "some retrieved context"),
        ("sources_content", "raw doc text"),
        ("prompt", "system prompt text"),
        ("exception_text", "Traceback (most recent call last)"),
    ],
)
def test_forbidden_keys_are_dropped_not_raised(capture, forbidden_field, value):
    L.log_event("request_end", route="rag", status_code=200, duration_ms=1.0, **{forbidden_field: value})
    record = _last_record(capture)
    assert forbidden_field not in record


@pytest.mark.parametrize(
    "path_value",
    ["/Users/someone/secret.txt", "/home/someone/secret.txt", "C:\\Users\\someone\\secret.txt"],
)
def test_absolute_path_looking_values_are_dropped(capture, path_value):
    L.log_event("readiness", reason=path_value)
    record = _last_record(capture)
    assert record.get("reason") != path_value


def test_unallowed_key_for_event_is_dropped(capture):
    L.log_event("request_start", route="rag", method="POST", unexpected_field="x")
    record = _last_record(capture)
    assert "unexpected_field" not in record


def test_dropped_field_increments_injected_metrics_registry_counter(capture):
    reg = M.build_metrics_registry()
    L.log_event("request_end", route="rag", status_code=200, duration_ms=1.0, question="leak", metrics_registry=reg)
    assert reg.get_sample_value("logging_dropped_fields_total") == 1.0


def test_no_registry_injected_does_not_raise_and_skips_counting(capture):
    # metrics_registry=None (the default) must not raise even though a
    # field is dropped.
    L.log_event("request_end", route="rag", status_code=200, duration_ms=1.0, question="leak")
    record = _last_record(capture)
    assert "question" not in record


def test_log_event_strict_raises_on_forbidden_field():
    with pytest.raises(ValueError):
        L.log_event_strict("request_end", route="rag", status_code=200, duration_ms=1.0, answer="leak")


def test_log_event_strict_succeeds_for_clean_payload(capture):
    L.log_event_strict("request_end", route="rag", status_code=200, duration_ms=1.0)
    record = _last_record(capture)
    assert record["status_code"] == 200


def test_unknown_event_name_raises_value_error():
    with pytest.raises(ValueError):
        L.log_event("not_a_real_event", route="rag")


def test_startup_and_readiness_allow_request_id_none(capture):
    L.log_event("startup", reason="ok")
    record = _last_record(capture)
    assert record["request_id"] is None


def test_startup_engine_error_type_accepts_bare_exception_class_name(capture):
    L.log_event("startup", reason="engine_init_failed", engine_error_type="PermissionError")
    record = _last_record(capture)
    assert record["engine_error_type"] == "PermissionError"


@pytest.mark.parametrize(
    "bad_value",
    [
        "https://user:secret@host/path",  # not a bare identifier
        "/Users/someone/secret.txt",  # also caught by the abs-path guard
        "x" * 65,  # over the length bound
        123,  # wrong type
    ],
)
def test_startup_engine_error_type_out_of_grammar_is_clamped_not_recorded_raw(capture, bad_value):
    L.log_event("startup", reason="engine_init_failed", engine_error_type=bad_value)
    record = _last_record(capture)
    assert record.get("engine_error_type") != bad_value


def test_startup_engine_error_type_is_optional(capture):
    L.log_event("startup", reason="ok")
    record = _last_record(capture)
    assert "engine_error_type" not in record


def test_log_event_strict_raises_on_malformed_engine_error_type():
    with pytest.raises(ValueError):
        L.log_event_strict("startup", reason="engine_init_failed", engine_error_type="not a class name")


def test_confidence_is_clamped_to_3_decimals_and_unit_interval(capture):
    L.log_event("routing", decision="document_qa", confidence=1.23456)
    record = _last_record(capture)
    assert record["confidence"] == 1.0

    L.log_event("routing", decision="document_qa", confidence=-0.5)
    record = _last_record(capture)
    assert record["confidence"] == 0.0

    L.log_event("routing", decision="document_qa", confidence=0.123456)
    record = _last_record(capture)
    assert record["confidence"] == 0.123


# ---------------------------------------------------------------------------
# CR-I1-MIN-01 closure — required key/type/enum negative matrix. The
# non-strict `log_event()` path must clamp/default instead of recording a
# raw out-of-enum/wrong-typed value or silently omitting a required key;
# `log_event_strict()` must raise for the same inputs.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "event,fields,key,expected",
    [
        ("request_start", {"route": "bogus-route", "method": "GET"}, "route", "health"),
        ("request_end", {"route": "rag", "status_code": 200, "duration_ms": 1.0, "error_code": "bogus"}, "error_code", "internal"),
        ("web_search", {"stage": "bogus-stage", "duration_ms": 1.0}, "stage", "generation"),
        ("request_end", {"route": "rag", "status_code": 999, "duration_ms": 1.0}, "status_code", 500),
        ("request_end", {"route": "rag", "status_code": "200", "duration_ms": 1.0}, "status_code", 500),
        ("request_end", {"route": "rag", "status_code": 200, "duration_ms": -5.0}, "duration_ms", 0.0),
        ("request_end", {"route": "rag", "status_code": 200, "duration_ms": "not-a-number"}, "duration_ms", 0.0),
        ("request_start", {"route": "rag", "method": "GET", "level": "SUPER_CRITICAL"}, "level", "info"),
        ("routing", {"decision": "x", "confidence": "high"}, "confidence", 0.0),
        ("routing", {"decision": "x", "confidence": 5.0}, "confidence", 1.0),
    ],
)
def test_out_of_enum_or_wrong_typed_value_is_clamped_not_recorded_raw(
    capture, event, fields, key, expected
):
    L.log_event(event, **fields)
    record = _last_record(capture)
    assert record[key] == expected


@pytest.mark.parametrize(
    "event,fields,missing_key,default",
    [
        ("request_start", {"method": "GET"}, "route", "health"),
        ("request_end", {"route": "rag", "duration_ms": 1.0}, "status_code", 500),
        ("request_end", {"route": "rag", "status_code": 200}, "duration_ms", 0.0),
        ("web_search", {"duration_ms": 1.0}, "stage", "generation"),
        ("routing", {"confidence": 0.5}, "decision", "unknown"),
        ("routing", {"decision": "x"}, "confidence", 0.0),
    ],
)
def test_missing_required_key_is_filled_with_safe_default(capture, event, fields, missing_key, default):
    L.log_event(event, **fields)
    record = _last_record(capture)
    assert record[missing_key] == default


@pytest.mark.parametrize(
    "event,fields",
    [
        ("request_start", {"route": "bogus", "method": "GET"}),
        ("request_end", {"route": "rag", "status_code": 9999, "duration_ms": 1.0}),
        ("web_search", {"stage": "bogus", "duration_ms": 1.0}),
        ("request_end", {"route": "rag", "status_code": 200, "duration_ms": -1.0}),
        ("request_end", {"route": "rag", "status_code": "200", "duration_ms": 1.0}),
        ("request_start", {"route": "rag", "method": "GET", "level": "SUPER_CRITICAL"}),
        ("request_end", {"route": "rag", "status_code": 200, "duration_ms": 1.0, "error_code": "bogus"}),
    ],
)
def test_log_event_strict_raises_on_out_of_enum_or_wrong_typed_value(event, fields):
    with pytest.raises(ValueError):
        L.log_event_strict(event, **fields)


@pytest.mark.parametrize(
    "event,fields,missing",
    [
        ("request_start", {"method": "GET"}, "route"),
        ("request_end", {"route": "rag", "duration_ms": 1.0}, "status_code"),
        ("web_search", {"duration_ms": 1.0}, "stage"),
        ("routing", {"confidence": 0.5}, "decision"),
    ],
)
def test_log_event_strict_raises_on_missing_required_key(event, fields, missing):
    with pytest.raises(ValueError, match=missing):
        L.log_event_strict(event, **fields)


def test_handler_failure_is_swallowed_and_does_not_raise():
    class BrokenHandler(stdlib_logging.Handler):
        def emit(self, record):
            raise RuntimeError("backend is down")

    original_handlers = L._logger.handlers
    L._logger.handlers = [BrokenHandler()]
    try:
        L.log_event("request_start", route="rag", method="GET")  # must not raise
    finally:
        L._logger.handlers = original_handlers
